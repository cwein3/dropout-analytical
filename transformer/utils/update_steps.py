import sys
import math
import functools

import numpy as np
import math 

import torch
import torch.nn as nn
import torch.nn.functional as F

update_types = [
	'no_drop',
	'standard_drop',
	'jac_reg'
	]

def add_update_args(parser):
	parser.add_argument('--update_type', 
		choices=update_types,
		help='The form of the update to make.')
	parser.add_argument('--dropout_reps',
		type=int,
		default=1,
		help='Number of dropout repetitions.')
	parser.add_argument('--drop_batch',
		type=int,
		default=1,
		help='Dropout reps to batch for backwards call.')
	parser.add_argument('--exp_reg_type', 
		choices=['none', 'jreg_sample_mask', 'jreg_sample_logit', 'jreg_sample_logit+loss_jac', '+loss_jac'],
		default='none',
		help='Type of explicit regularization to have.')
	parser.add_argument('--imp_reg_type',
		choices=['none', 'taylor_fo'],
		default='none',
		help='Type of implicit regularization to have.')
	parser.add_argument('--exp_coeff_type',
		choices=['spec'],
		default='spec',
		help='How to choose the appropriate regularization coefficients.')
	parser.add_argument('--imp_coeff_type',
		choices=['spec'],
		default='spec',
		help='How to choose the appropriate regularization coefficients.')
	parser.add_argument('--hess_drop',
		type=float,
		help='Explicit regularization replacing dropout (for the Hessian term).')
	parser.add_argument('--hess_dropatt',
		type=float,
		help='Explicit regularization replacing attention dropout (for the Hessian term).')
	parser.add_argument('--jac_drop',
		type=float,
		help='Explicit regularization replacing dropout (for the Jacobian term).')
	parser.add_argument('--jac_dropatt',
		type=float,
		help='Explicit regularization replacing attention dropout (for the Jacobian term).')
	parser.add_argument('--imp_drop',
		type=float,
		help='Implicit regularization replacing dropout.')
	parser.add_argument('--imp_dropatt',
		type=float,
		help='Implicit regularization replacing attention dropout.')

def chunk_wrapper(model, optimizer, data, target, mems, args, update_func, **update_kwargs):
	tot_loss = 0
	if args.batch_chunk > 1:
		data_chunks = torch.chunk(data, args.batch_chunk, 1)
		target_chunks = torch.chunk(target, args.batch_chunk, 1)
		for i in range(args.batch_chunk):
			data_i = data_chunks[i].contiguous()
			target_i = target_chunks[i].contiguous()
			ret = update_func(model, optimizer, data_chunks[i], target_chunks[i], mems[i], args, **update_kwargs)
			loss, mems[i] = ret[0], ret[1:]
			tot_loss += loss
	else:
		ret = update_func(model, optimizer, data, target, mems, args, **update_kwargs)
		loss, mems = ret[0], ret[1:]
		tot_loss += loss
	return tot_loss, mems

def standard_back_pass(model, optimizer, data, target, mems, args, dropout_reps=1, use_dropout=True):
	loss_track = 0
	curr_loss = 0
	reg_args = {'exp_reg_type' : 'none', 'imp_reg_type' : 'none'}
	if not use_dropout: dropout_reps = 1

	# mems appears to be of shape mlen x bsz, and data, target also have second dimension batch size
	for dropout_ind in range(0, dropout_reps, args.drop_batch):
		drop_batch_size = min(args.drop_batch, dropout_reps - dropout_ind)
		# duplicate the data, target, mems
		data_to_use = data.repeat(1, drop_batch_size)
		target_to_use = target.repeat(1, drop_batch_size)
		mems_to_use = []
		for mem in mems:
			repeat_idx = [1]*mem.dim()
			repeat_idx[1] = drop_batch_size
			mems_to_use.append(mem.repeat(*repeat_idx))
		ret = model(data_to_use, target_to_use, *mems_to_use, use_dropout=use_dropout, reg_args=reg_args)
		
		loss, new_mems = ret[0], ret[1:]
		loss = loss.float().mean().type_as(loss)/args.batch_chunk
		loss_track += drop_batch_size*loss.float().item()
		curr_loss = drop_batch_size*loss/dropout_reps
		if args.fp16:
			optimizer.backward(curr_loss)
		else:
			curr_loss.backward()

		bsz = data.size(1)
		new_mems = [mem[:, :bsz, :] for mem in new_mems]

	mems = new_mems
	loss_track = loss_track/dropout_reps
	return [loss_track] + mems

def standard_update(model, para_model, optimizer, optimizer_sparse, data, target, mems, args, use_dropout=True):
	model.zero_grad()
	update_kwargs = {'dropout_reps' : args.dropout_reps, 'use_dropout' : use_dropout}

	tot_loss, new_mems = chunk_wrapper(para_model, optimizer, data, target, mems, args, standard_back_pass, **update_kwargs)
	if args.fp16:
		optimizer.clip_master_grads(args.clip)
	else:
		torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

	optimizer.step()
	if args.sample_softmax > 0:
		optimizer_sparse.step()
	return tot_loss, new_mems

def jacobian_reg_update(model, para_model, optimizer, optimizer_sparse, data, target, mems, args):

	def jac_update_func(model, optimizer, data, target, mems, args):
		reg_args = {
			'exp_reg_type' : args.exp_reg_type,
			'imp_reg_type' : args.imp_reg_type,
			'exp_coeff_type' : args.exp_coeff_type,
			'imp_coeff_type' : args.imp_coeff_type,
			'hess_drop' : args.hess_drop,
			'hess_dropatt' : args.hess_dropatt,
			'jac_drop' : args.jac_drop,
			'jac_dropatt' : args.jac_dropatt,
			'imp_drop' : args.imp_drop,
			'imp_dropatt' : args.imp_dropatt
		}
		ret = model(data, target, *mems, use_dropout=False, reg_args=reg_args)
		model_loss = ret[0]
		model_loss = model_loss.float().mean().type_as(model_loss)/args.batch_chunk 
		tot_loss = model_loss
		mem_ind = 1
		if args.exp_reg_type != 'none':
			tot_loss = tot_loss + ret[mem_ind].float().mean().type_as(model_loss)/args.batch_chunk
			mem_ind += 1
		if args.imp_reg_type != 'none':
			tot_loss = tot_loss + ret[mem_ind].float().mean().type_as(model_loss)/args.batch_chunk
			mem_ind += 1
		mems = ret[mem_ind:]

		if args.fp16:
			optimizer.backward(tot_loss)
		else:
			tot_loss.backward()

		return [model_loss.float().item()] + mems

	model.zero_grad()
	update_kwargs = {}

	tot_loss, new_mems = chunk_wrapper(para_model, optimizer, data, target, mems, args, jac_update_func, **update_kwargs)
	if args.fp16:
		optimizer.clip_master_grads(args.clip)
	else:
		torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

	optimizer.step()
	if args.sample_softmax > 0:
		optimizer_sparse.step()
	return tot_loss, new_mems

def update_step(model, para_model, optimizer, optimizer_sparse, data, target, mems, args):
	if args.update_type == 'no_drop':
		return standard_update(
			model,
			para_model,
			optimizer,
			optimizer_sparse,
			data, 
			target,
			mems,
			args,
			use_dropout=False)
	if args.update_type == 'standard_drop':
		return standard_update(
			model,
			para_model,
			optimizer,
			optimizer_sparse,
			data, 
			target,
			mems,
			args,
			use_dropout=True)
	if args.update_type == 'jac_reg':
		return jacobian_reg_update(
			model,
			para_model,
			optimizer,
			optimizer_sparse,
			data,
			target,
			mems,
			args)
