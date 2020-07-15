import torch
import torch.nn as nn
import random
from torch.distributions.categorical import Categorical
import math
import data_reg_utils

update_types = [
	'no_reg',
	'drop_standard',
	'drop_fix_noise'
	]
exp_update_types = [
		'jreg_hess_id', 
		'jreg_sample_logit',  
		'jreg_loss']
imp_update_types = [
		'taylor_fo'
		]

data_reg_combos = ['%s+%s' % (exp_type, imp_type) for exp_type in exp_update_types for imp_type in imp_update_types]
dropout_imp_combos = ['drop_standard+%s' % (imp_type) for imp_type in imp_update_types]

update_types = update_types + exp_update_types + data_reg_combos + dropout_imp_combos

custom_lstm_updates = exp_update_types + data_reg_combos + dropout_imp_combos

# add custom args here to the update steps to make it cleaner hopefully
def add_update_args(parser):
	parser.add_argument('--update_type', 
                    choices=update_types, 
                    help='the form of the update to make.')
	parser.add_argument('--dropout_reps', type=int, default=1, help='number of averages of dropout to take')
	data_reg_utils.add_data_reg_args(parser)

# perform the update for the dropout reps
def dropout_reps_back_pass(model, criterion, data, targets, old_hidden, dropout_reps, args, use_dropout=True):
	raw_loss_track = 0
	curr_loss = 0
	for dropout_ind in range(dropout_reps):
		if dropout_ind > 0 and dropout_ind % 4 == 0:
			curr_loss = curr_loss/dropout_reps
			curr_loss.backward()
			curr_loss = 0
		output, hidden, rnn_hs, dropped_rnn_hs, emb = model(data, old_hidden, return_h=True, use_dropout=use_dropout)
		rep_raw_loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)

		curr_loss = curr_loss + rep_raw_loss
		raw_loss_track += rep_raw_loss.item()

	curr_loss = curr_loss/dropout_reps 
	curr_loss.backward()
	raw_loss_track = raw_loss_track/dropout_reps
	return raw_loss_track, hidden

def dropout_reps_dup_batch(model, criterion, data, targets, old_hidden, dropout_reps, args, use_dropout=True):
	# implementation for dropout reps where we duplicate the batch.. this is necessary for the QRNN probably
	bs = data.size(1)
	time_len = data.size(0)

	data_to_use = data.repeat(1, dropout_reps) if dropout_reps > 1 else data
	targets_reshape = targets.view(time_len, bs)
	targets_to_use = targets_reshape.repeat(1, dropout_reps).view(-1) if dropout_reps > 1 else targets
	
	output, hidden, rnn_hs, dropped_rnn_hs, emb = model(data_to_use, old_hidden, return_h=True, use_dropout=use_dropout)
	curr_loss = criterion(model.decoder.weight, model.decoder.bias, output, targets_to_use)

	curr_loss.backward()
	return curr_loss.item(), hidden

def dropout_noise(model, criterion, data, targets, old_hidden):
	dropped_losses = []
	for i in range(2):
		output, new_hidden = model(data, old_hidden, return_h=False, use_dropout=True)
		dropped_losses.append(criterion(model.decoder.weight, model.decoder.bias, output, targets))

	# normalize to have same covariance as one dropout iter
	drop_loss_track = (dropped_losses[0].item() + dropped_losses[1].item())/2
	return (dropped_losses[0] - dropped_losses[1])/math.sqrt(2), drop_loss_track

def dropout_update(model, params, optimizer, criterion, data, targets, hidden, args, use_dropout=True, fix_noise=False):
	old_hidden = hidden

	optimizer.zero_grad()
	dropout_reps = args.dropout_reps
	reps_back_pass_func = dropout_reps_dup_batch if args.model == 'QRNN' else dropout_reps_back_pass
	raw_loss, hidden = reps_back_pass_func(
		model, criterion, data, targets, old_hidden, args.dropout_reps, args, use_dropout=use_dropout)

	# if we choose to fix the noise, then we add back the noise here 
	if fix_noise:
		noise_diff, _ = dropout_noise(model, criterion, data, targets, old_hidden)
		noise_diff = noise_diff*math.sqrt(1 - 1.0/args.dropout_reps)
		noise_diff.backward()

	if args.clip: torch.nn.utils.clip_grad_norm_(params, args.clip)
	optimizer.step()

	return raw_loss, hidden

def drop_reps_imp_noise(model, params, optimizer, criterion, data, targets, hidden, args, use_dropout=True, imp_reg_method='none'):
	old_hidden = hidden

	optimizer.zero_grad()
	dropout_reps = args.dropout_reps
	reps_back_pass_func = dropout_reps_dup_batch if args.model == 'QRNN' else dropout_reps_back_pass
	raw_loss, hidden = reps_back_pass_func(
		model, criterion, data, targets, old_hidden, args.dropout_reps, args, use_dropout=use_dropout)

	output, nodrop_hidden, rnn_hs, dropped_rnn_hs, emb = model(data, old_hidden, return_h=True, use_dropout=False)
	#no_drop_loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)
	imp_loss = data_reg_utils.imp_reg(
		model, criterion, targets, emb, rnn_hs[:-1], rnn_hs[-1],
		imp_reg_method, args, loss_val=None)
	imp_loss.backward()

	if args.clip: torch.nn.utils.clip_grad_norm_(params, args.clip)
	optimizer.step()

	return raw_loss, hidden

def update_step(model, params, optimizer, criterion, data, target, hidden, update_type, args):
	if update_type == 'no_reg':
		return dropout_update(
			model,
			params,
			optimizer,
			criterion,
			data, 
			target,
			hidden,
			args,
			use_dropout=False)
	if update_type == 'drop_standard':
		return dropout_update(
			model,
			params,
			optimizer,
			criterion,
			data, 
			target,
			hidden,
			args,
			use_dropout=True)
	if update_type == 'drop_fix_noise':
		return dropout_update(
			model,
			params,
			optimizer,
			criterion,
			data,
			target,
			hidden,
			args,
			use_dropout=True,
			fix_noise=True)

	# hacky way to do this for now
	if update_type == 'drop_standard+taylor_fo':
		return drop_reps_imp_noise(
			model,
			params,
			optimizer,
			criterion,
			data,
			target,
			hidden,
			args,
			use_dropout=True,
			imp_reg_method='taylor_fo'
			)

	##### if it reaches here, then we assume that it's in the form explicit_reg+implicit_reg

	reg_types = update_type.split('+')
	if len(reg_types) == 1:
		return data_reg_utils.data_reg_update(
			model,
			params,
			optimizer,
			criterion,
			data,
			target,
			hidden,
			args,
			exp_reg_method=reg_types[0],
			imp_reg_method='none')
	else:
		return data_reg_utils.data_reg_update(
			model,
			params,
			optimizer,
			criterion,
			data,
			target,
			hidden,
			args,
			exp_reg_method=reg_types[0],
			imp_reg_method=reg_types[1])
		

	