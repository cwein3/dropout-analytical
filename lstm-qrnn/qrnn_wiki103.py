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
                    help='location of the data corpus.')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=2500,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=4,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.5e-3,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=8000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=15, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=140,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--no_cuda', action='store_true',
                    help='use CUDA')

parser.add_argument('--log_interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--eval_interval', type=int, default=6000, help='How often to evaluate on the validation set.')
parser.add_argument('--save', type=str,
                    help='path to save the final model')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--resume', type=str,  default='',
                    help='path of model to resume')
parser.add_argument('--optimizer', type=str,  default='sgd',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--when', nargs="+", type=int, default=[12],
                    help='When (which epochs) to multiply the learning rate by lrdecay - accepts multiple')
parser.add_argument('--lr_decay', type=float, default=0.1, help='decay rate for learning rate')
parser.add_argument('--save_every', type=int, default=24000, help='How often to save the model. (In terms of steps within the epoch, not epochs!)')

parser.set_defaults(no_cuda=False)

update_steps.add_update_args(parser)

args = parser.parse_args()
args.tied = True
args.model = 'QRNN'
args.update_type = 'drop_standard'

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
        torch.save([model, criterion, optimizer.state_dict()], f)

def save_data_dict(dirn, data_dict):
    data_dict_loc = os.path.join(dirn, "data_dict")
    torch.save(data_dict, data_dict_loc)

def model_load(fn):
    global model, criterion, opt_state_dict
    with open(fn, 'rb') as f:
        model, criterion, opt_state_dict = torch.load(f)

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
opt_state_dict = None

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
    'custom_lstm' : False 
}

model = model.RNNModel(
    args.model, 
    ntokens, 
    args.emsize, 
    args.nhid, 
    args.nlayers, 
    **model_kwargs)

global start_epoch, total_batch
start_epoch = 1
total_batch = 1
###
if args.resume:
    print('Resuming model ...')
    resume_loc = args.resume
    model_load(resume_loc)
    model.dropouti, model.dropouth, model.dropout, model.dropoute = args.dropouti, args.dropouth, args.dropout, args.dropoute
    if args.wdrop:
        from weight_drop import WeightDrop
        for rnn in model.rnns:
            if type(rnn) == WeightDrop: rnn.dropout = args.wdrop
    
###
if not criterion:
    splits = []
    if ntokens > 500000:
        # One Billion
        # This produces fairly even matrix mults for the buckets:
        # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
        splits = [4200, 35000, 180000]
    elif ntokens > 75000:
        # WikiText-103
        splits = [2800, 20000, 76000]
    print('Using', splits)
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

# Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
if args.optimizer == 'sgd':
    optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wdecay)
if args.optimizer == 'adam':
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)

if opt_state_dict:
    optimizer.load_state_dict(opt_state_dict)

stored_loss = 100000000

###############################################################################
# Training code
###############################################################################

def evaluate(data_source, use_dropout=False, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    if not use_dropout:
        model.eval()
    else:
        model.train()
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    eval_hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, eval_hidden = model(data, eval_hidden)
        total_loss += len(data) * criterion(model.decoder.weight, model.decoder.bias, output, targets).item()
        eval_hidden = repackage_hidden(eval_hidden)

    # turn on eval mode at the end because we expect eval mode
    model.eval()
    return total_loss/ len(data_source)


def train(args, eval_dict, label_noise_hparams):
    # Turn on training mode which enables dropout.
    global total_batch, stored_loss
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    running_avg_loss = 0
    running_avg_ppl = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden_size = args.batch_size*args.dropout_reps
    hidden = model.init_hidden(hidden_size)
    batch, i = 0, 0

    while i < train_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)

        # first evaluate with some probability
        
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

        optimizer.param_groups[0]['lr'] = lr2
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
            eval_dict['train_loss'].append((total_batch, cur_loss))
            eval_dict['train_ppl'].append((total_batch, math.exp(cur_loss)))

            total_loss = 0
            start_time = time.time()
            sys.stdout.flush()

        if batch % args.eval_interval == 0 and batch > 0:
            val_loss = evaluate(val_data,use_dropout=False,batch_size=eval_batch_size)
            eval_dict['val_loss'].append((total_batch, val_loss))
            eval_dict['val_ppl'].append((total_batch, math.exp(val_loss)))
            print('Evaluation | epoch {:3d} | {:5d}/{:5d} batches | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, val_loss, math.exp(val_loss)))
            if val_loss < stored_loss:
                model_save(args.save, "best_val.pt")
                print('Saving model (new best validation)')
                stored_loss = val_loss

            save_data_dict(args.save, eval_dict)
            model.reset()
            model.train()
            print('Best validation perplexity:', math.exp(stored_loss))

        if total_batch % args.save_every == 0 and total_batch > 0:
            model_save(args.save, 'checkpoint.step{}'.format(total_batch))

        ###
        batch += 1
        i += seq_len
        total_batch += 1

# Loop over epochs.
lr = args.lr
best_val_loss = []

summary_dict = {'train_loss' : [], 'val_loss' : [], 'val_ppl' : [], 'train_ppl' : []}
# At any point you can hit Ctrl + C to break out of training early.
try:
    
    for epoch in range(start_epoch, args.epochs+start_epoch):
        sys.stdout.flush()
        epoch_start_time = time.time()
        
        label_noise_hparams = {}

        train(args, summary_dict, label_noise_hparams)
                  
        val_loss = evaluate(val_data, use_dropout=False, batch_size=eval_batch_size)
        summary_dict['val_loss'].append((total_batch, val_loss))
        summary_dict['val_ppl'].append((total_batch, math.exp(val_loss)))
        
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

        best_val_loss.append(val_loss)
        save_data_dict(args.save, summary_dict)
        
        print('Best validation perplexity:', math.exp(stored_loss))
        # remove the latest checkpoint to avoid collecting too many files and save a new one
        model_save(args.save, 'checkpoint.step{}'.format(total_batch))


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
