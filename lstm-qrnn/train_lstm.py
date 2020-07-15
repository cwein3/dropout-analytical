import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
import pytz
import data
import model
import sys
import json
import os
import copy
from datetime import datetime
import update_steps
from utils import batchify, get_batch, repackage_hidden
from lstm import torch_lstm_to_custom

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str,
                    help='location of the data corpus. Note: only PTB and Wiki-2 supported.')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=8000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.4,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.4,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.4,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--no_cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log_interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,
                    help='path to save the final model')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--resume', type=str,  default='',
                    help='path of model to resume')
parser.add_argument('--optimizer', type=str,  default='sgd',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to multiply the learning rate by lrdecay - accepts multiple')
parser.add_argument('--lr_decay', type=float, default=0.1, help='decay rate for learning rate')
parser.add_argument('--save_every', type=int, default=100, help='How often to save the model.')
parser.set_defaults(no_cuda=False)

update_steps.add_update_args(parser)

args = parser.parse_args()
args.model = 'LSTM'
args.tied = True

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if args.no_cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

def model_save(dirn, fn):
    if not os.path.exists(dirn):
        os.makedirs(dirn)
    fn = os.path.join(dirn, fn)
    # also dump the args here
    args_file = os.path.join(dirn, "args.txt")
    if not os.path.exists(args_file):
        args_str = ""
        for arg in vars(args):
            args_str += arg + " : " + str(getattr(args, arg)) + "\n"
        with open(args_file, "w") as f:
            f.write(args_str)
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer], f)

def save_data_dict(dirn, data_dict):
    data_dict_loc = os.path.join(dirn, "data_dict")
    torch.save(data_dict, data_dict_loc)

    # also write data dict info to a file
    data_dict_txt = os.path.join(dirn, "data_dict_txt.txt")
    last_dict = data_dict[-1]
    with open(data_dict_txt, "a") as f:
        f.write("Epoch %d, Data %s\n" % (last_dict['epoch'], json.dumps(last_dict)))

def model_load(fn):
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)

import os
import hashlib
fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
fn = os.path.join(args.data, fn)
if os.path.exists(fn):
    print('Loading cached dataset...')
    corpus = torch.load(fn)
else:
    print('Producing dataset...')
    corpus = data.Corpus(args.data)
    torch.save(corpus, fn)

eval_batch_size = 10
test_batch_size = 1
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)

###############################################################################
# Build the model
###############################################################################

from splitcross import SplitCrossEntropyLoss
criterion = None

ntokens = len(corpus.dictionary)
custom_lstm_updates = update_steps.custom_lstm_updates

model_kwargs = {
    'dropout' : args.dropout,
    'dropouth' : args.dropouth,
    'dropouti' : args.dropouti,
    'dropoute' : args.dropoute,
    'wdrop' : args.wdrop,
    'tie_weights' : args.tied,
    'no_dropout' : False, 
    'custom_lstm' : args.update_type in custom_lstm_updates 
}

model = model.RNNModel(
    'LSTM', 
    ntokens, 
    args.emsize, 
    args.nhid, 
    args.nlayers, 
    **model_kwargs)

global start_epoch
start_epoch = 1
###
if args.resume:
    print('Resuming model ...')
    resume_loc = args.resume
    model_load(resume_loc)
    optimizer.param_groups[0]['lr'] = args.lr
    model.dropouti, model.dropouth, model.dropout, model.dropoute = args.dropouti, args.dropouth, args.dropout, args.dropoute
    if args.wdrop:
        from weight_drop import WeightDrop
        for rnn in model.rnns:
            if type(rnn) == WeightDrop: rnn.dropout = args.wdrop
            #elif rnn.zoneout > 0: rnn.zoneout = args.wdrop
    if args.update_type in custom_lstm_updates:
        model = model.convert_to_custom()
###
if not criterion:
    splits = []
    criterion = SplitCrossEntropyLoss(args.emsize, splits=splits, verbose=False)
###

if not args.no_cuda:
    model = model.cuda()
    criterion = criterion.cuda()
###
params = list(model.parameters()) + list(criterion.parameters())
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Args:', args)
print('Model total parameters:', total_params)

###############################################################################
# Training code
###############################################################################

def evaluate(data_source, use_dropout=False, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    if not use_dropout:
        model.eval()
    else:
        model.train()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, hidden = model(data, hidden)
        total_loss += len(data) * criterion(model.decoder.weight, model.decoder.bias, output, targets).data
        hidden = repackage_hidden(hidden)

    # turn on eval mode at the end because we expect eval mode
    model.eval()
    return total_loss.item() / len(data_source)


def train(args):
    # Turn on training mode which enables dropout.
    total_loss = 0
    running_avg_loss = 0
    running_avg_ppl = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    batch, i = 0, 0
    eval_dict = {'avg_ppl' : 0, 'avg_loss' : 0}

    while i < train_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        # seq_len = min(seq_len, args.bptt + 10)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)

        ret_info = update_steps.update_step(
            model,
            params,
            optimizer, 
            criterion, 
            data, 
            targets, 
            hidden,
            args.update_type, 
            args)

        raw_loss, hidden = ret_info

        total_loss += raw_loss
        eval_dict['avg_loss'] += raw_loss
        eval_dict['avg_ppl'] += math.exp(raw_loss)

        optimizer.param_groups[0]['lr'] = lr2
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
            total_loss = 0
            start_time = time.time()
            sys.stdout.flush()
        ###
        batch += 1
        i += seq_len
    
    eval_dict['avg_loss'] /= batch
    eval_dict['avg_ppl'] /= batch
    
    return eval_dict

# Loop over epochs.
lr = args.lr
stored_loss = 100000000

summary_dict = []
# At any point you can hit Ctrl + C to break out of training early.
try:
    optimizer = None
    # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)
    for epoch in range(start_epoch, args.epochs):
        sys.stdout.flush()
        curr_data = {'epoch' : epoch}
        epoch_start_time = time.time()

        eval_dict = train(args)
        avg_loss = eval_dict['avg_loss']
        curr_data.update({'training_output_loss' : avg_loss, 'training_output_ppl' : math.exp(avg_loss)})
                
        val_loss = evaluate(val_data, use_dropout=False, batch_size=eval_batch_size)
        curr_data.update({'val_loss' : val_loss, 'val_ppl' : math.exp(val_loss)})
        val_loss_drop = evaluate(val_data, use_dropout=True, batch_size=eval_batch_size)
        curr_data.update({'val_loss_dropout' : val_loss_drop, 'val_ppl_dropout' : math.exp(val_loss_drop)})
        train_loss_nodrop = evaluate(train_data, use_dropout=False, batch_size=args.batch_size)
        curr_data.update({'train_loss' : train_loss_nodrop, 'train_ppl' : math.exp(train_loss_nodrop)})
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
          epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
        print('-' * 89)

        if val_loss < stored_loss:
            model_save(args.save, "best_val.pt")
            print('Saving model (new best validation)')
            stored_loss = val_loss

        if epoch in args.when:
            print('Saving model before learning rate decreased')
            model_save(args.save, 'checkpoint.e{}'.format(epoch))
            print('Decreasing learning rate by decay rate %.4f' % (args.lr_decay))
            optimizer.param_groups[0]['lr'] *= args.lr_decay

        summary_dict.append(curr_data)
        print('Best validation perplexity:', math.exp(stored_loss))
        if epoch % args.save_every == 0:
            print('Saving checkpoint at epoch %d' % (epoch))
            model_save(args.save, 'checkpoint.e{}'.format(epoch))
        save_data_dict(args.save, summary_dict)
        
        # remove the latest checkpoint to avoid collecting too many files and save a new one
        for file_name in os.listdir(args.save):
            if "latest_checkpoint" in file_name:
                os.remove(os.path.join(args.save, file_name))
        latest_checkpoint = "latest_checkpoint.e{}".format(epoch)
        model_save(args.save, latest_checkpoint)


except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
model_load(os.path.join(args.save, "best_val.pt"))

# Run on test data.
test_loss = evaluate(test_data, use_dropout=False, batch_size=test_batch_size)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
    test_loss, math.exp(test_loss), test_loss / math.log(2)))
print('=' * 89)
