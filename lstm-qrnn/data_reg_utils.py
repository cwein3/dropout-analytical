# utils for computing data-dependent regularizers such as Jacobian, Hessian, etc.
import torch
import torch.nn as nn
import random
from torch.distributions.categorical import Categorical
import math

def add_data_reg_args(parser):
	parser.add_argument('--exp_reg_samples', type=int, default=1, help='number of samples of Jacobian reg to use.')
	parser.add_argument('--exp_reg_batch', type=int, default=1, help='batch size for explicit reg samples.')

	parser.add_argument('--exp_regi', type=float, default=0, help='explicit regularization on the input.')
	parser.add_argument('--exp_regh', type=float, default=0, help='explicit regularization on the hidden layers')
	parser.add_argument('--exp_regw', type=float, default=0, help='explicit regularization on the weight matrices')
	parser.add_argument('--exp_rego', type=float, default=0, help='explicit regularization on the model output')
	parser.add_argument('--imp_regi', type=float, default=0, help='noise multiplier for implicit regularization on the input')
	parser.add_argument('--imp_regh', type=float, default=0, help='noise multiplier for implicit regularization on hidden layer')
	parser.add_argument('--imp_regw', type=float, default=0, help='noise multiplier for implicit regularization on the weights')
	parser.add_argument('--imp_rego', type=float, default=0, help='noise multipleir for implicit regularization on the output')
	parser.add_argument('--loss_regi', type=float, default=0, help='regularizer for loss jacobian on input')
	parser.add_argument('--loss_regh', type=float, default=0, help='regularizer for loss jacobian on hidden layers')
	parser.add_argument('--loss_regw', type=float, default=0, help='regularizer for loss jacobian on the weight matrices')
	parser.add_argument('--loss_rego', type=float, default=0, help='regularizer for loss jacobian on the model output')

############# helper functions
def mask_hl(hl_nomask, lock=True):
	MASK_PROB = 0.5 # keep this fixed for now
	first_dim = 1 if lock else hl_nomask.size(0)
	m = hl_nomask.data.new(first_dim, hl_nomask.size(1), hl_nomask.size(2)).bernoulli_(MASK_PROB)
	signs = (m - 0.5)*2
	signs = signs.expand_as(hl_nomask)
	return signs*hl_nomask

def mask_param(param):
	MASK_PROB = 0.5
	m = torch.ones_like(param).bernoulli_(MASK_PROB)
	signs = (m - 0.5)*2
	signs = signs.expand_as(param)
	return signs*param

def jvp(output, xs, vecs):
    if isinstance(output, torch.Tensor):
        v = torch.ones_like(output, requires_grad=True)
    else:
        raise NotImplementedError
    vjp = torch.autograd.grad(output, xs, grad_outputs=v, create_graph=True)
    jvp_out = [torch.autograd.grad(vjp_elem, v, grad_outputs=veci, create_graph=True)[0] for vjp_elem, veci in zip(vjp, vecs)]
    return jvp_out

def data_reg_update(
	model, params, optimizer, criterion, data, targets, hidden,
	args,
	exp_reg_method='none', imp_reg_method='none',
	):

	optimizer.zero_grad()

	output, emb, rnn_hs, final_exp_loss, new_hidden = exp_backwards(
		model, criterion, data, targets, hidden,
		exp_reg_method,
		args)

	if output is None:
		# we didn't have a explicit regularizer in this case
		output, new_hidden, rnn_hs, dropped_rnn_hs, emb = model(data, old_hidden, return_h=True, use_dropout=False)
		raw_loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)
		final_exp_loss = 0
	else:
	 	raw_loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)

	imp_loss = imp_reg(
		model, criterion, targets, emb, rnn_hs[:-1], rnn_hs[-1],
		imp_reg_method, args, loss_val=raw_loss)

	loss = final_exp_loss + imp_loss + raw_loss
	loss.backward()

	raw_loss = raw_loss.item()
	
	if args.clip: torch.nn.utils.clip_grad_norm_(params, args.clip)
	optimizer.step()

	return raw_loss, new_hidden

def exp_backwards(
	model, criterion, data, targets, hidden,
	exp_reg_method,
	args
	):

	exp_reg_args = {
		'exp_regi' : args.exp_regi,
		'exp_rego' : args.exp_rego,
		'exp_regh' : args.exp_regh,
		'exp_regw' : args.exp_regw,
		'loss_regi' : args.loss_regi,
		'loss_rego' : args.loss_rego,
		'loss_regh' : args.loss_regh,
		'loss_regw': args.loss_regw
	}
	old_hidden = hidden
	weight_params = model.get_weight_params()

	orig_len = data.size(0)
	orig_bs = data.size(1)
	if exp_reg_method == 'none':
		return None, None, None, None, None
	batch_lim = args.exp_reg_samples - 1 # we don't want to call backward on the last sample
	for reg_sample in range(0, batch_lim, args.exp_reg_batch):
		exp_reg_reps = min(args.exp_reg_batch, batch_lim - reg_sample)
		data_to_use = data if exp_reg_reps == 1 else data.repeat(1, exp_reg_reps)
		target_to_use = targets if exp_reg_reps == 1 else targets.view(orig_len, orig_bs).repeat(1, exp_reg_reps).view(-1)
		hidden_to_use = old_hidden if exp_reg_reps == 1 else model.dup_hidden(old_hidden, exp_reg_reps)
		output, new_hidden, rnn_hs, dropped_rnn_hs, emb = model(data_to_use, hidden_to_use, return_h=True, use_dropout=False)
		exp_loss = exp_reg(
			model, criterion, target_to_use,
			emb, rnn_hs[:-1], rnn_hs[-1], weight_params,
			exp_reg_method, exp_reg_args)
		exp_loss = exp_reg_reps*exp_loss/args.exp_reg_samples
		exp_loss.backward()

	output, new_hidden, rnn_hs, dropped_rnn_hs, emb = model(data, old_hidden, return_h=True, use_dropout=False)
	exp_loss = exp_reg(
		model, criterion, targets,
		emb, rnn_hs[:-1], rnn_hs[-1], weight_params,
		exp_reg_method, exp_reg_args)
	exp_loss = exp_loss/args.exp_reg_samples

	return output, emb, rnn_hs, exp_loss, new_hidden

def exp_reg(
	model, criterion, targets,
	emb, hl_vals, model_outs, weight_params,
	exp_reg_method, 
	exp_reg_args):
	if exp_reg_method == 'jreg_sample_logit':
		return jreg_sample_logit(
			model, criterion, targets,
			emb, hl_vals, model_outs, weight_params,
			exp_reg_args)
	if exp_reg_method == 'jreg_hess_id':
		return jreg_hess_id(
			model, criterion, targets,
			emb, hl_vals, model_outs, weight_params,
			exp_reg_args)

	if exp_reg_method == 'jreg_loss': ######### FOR THIS ONE, remember to set loss_reg parameters
		return loss_reg(
			model, criterion, targets, 
			emb, hl_vals, model_outs, weight_params,
			exp_reg_args)
	if exp_reg_method == 'none':
		return 0

def imp_reg(
	model, criterion, targets,
	emb, hl_vals, model_outs,
	imp_reg_method,
	args,
	loss_val=None):
	
	imp_reg_args = {
		'imp_regi' : args.imp_regi,
		'imp_rego' : args.imp_rego,
		'imp_regh' : args.imp_regh,
		'imp_regw' : args.imp_regw,
		'return_batch' : False
	}
	weight_params = model.get_weight_params()

	if imp_reg_method == 'taylor_fo':
		return taylor_fo(
			model, criterion, targets,
			emb, hl_vals, model_outs, weight_params,
			imp_reg_args, backprop_j=True,
			loss_val=loss_val)
	if imp_reg_method == 'none':
		return 0

def get_reg_mask(
	emb, hl_vals, model_outs, weight_params,
	regi, regh, rego, regw, lock_mask=False, 
	use_sqrt=False):
	vals_to_reg = []
	masked_vals = []
	if regi > 0:
		vals_to_reg += [emb]
		val_to_use = math.sqrt(regi) if use_sqrt else regi
		masked_vals += [val_to_use*mask_hl(emb, lock=lock_mask)]
	if regh > 0:
		vals_to_reg += hl_vals
		val_to_use = math.sqrt(regh) if use_sqrt else regh
		masked_vals += [val_to_use*mask_hl(hl, lock=lock_mask) for hl in hl_vals]
	if rego > 0:
		vals_to_reg += [model_outs]
		val_to_use = math.sqrt(rego) if use_sqrt else rego
		masked_vals += [val_to_use*mask_hl(model_outs, lock=lock_mask)]	
	if regw > 0:
		vals_to_reg += weight_params
		val_to_use = math.sqrt(regw) if use_sqrt else regw
		masked_vals += [val_to_use*mask_param(param) for param in weight_params]
	return vals_to_reg, masked_vals

def get_reg_coeffs(
	emb, hl_vals, model_outs, weight_params,
	regi, regh, rego, regw):
	vals_to_reg = []
	reg_coeffs = []
	if regi > 0:
		vals_to_reg += [emb]
		reg_coeffs += [regi]
	if regh > 0:
		vals_to_reg += hl_vals
		reg_coeffs += [regh for _ in hl_vals]
	if rego > 0:
		vals_to_reg += [model_outs]
		reg_coeffs += [rego]
	if regw > 0:
		vals_to_reg += weight_params
		reg_coeffs += [regw for _ in weight_params]
	return vals_to_reg, reg_coeffs

def jreg_hess_id(
	model, criterion, targets, 
	emb, hl_vals, model_outs, weight_params,
	exp_reg_args):
	
	orig_len = model_outs.size(0)
	orig_bs = model_outs.size(1)
	model_outs_reshape = model_outs.view(
		orig_len*orig_bs, model_outs.size(2))

	vals_to_reg, masked_vals = get_reg_mask(
		emb, hl_vals, model_outs, weight_params,
		exp_reg_args['exp_regi'],
		exp_reg_args['exp_regh'],
		exp_reg_args['exp_rego'],
		exp_reg_args['exp_regw'],
		use_sqrt=True)

	mask_prods = jvp(
		model_outs_reshape,
		vals_to_reg,
		masked_vals)

	# no bias term because we're computing derivative
	mask_prods = [torch.nn.functional.linear(mask_prod, model.decoder.weight) for mask_prod in mask_prods]
	
	# divide by the size of the vocabulary to simulate uniform distribution on vocab
	normalization = mask_prods[0].size(-1)
	
	prod_sums = [torch.sum(mask_prod**2) for mask_prod in mask_prods]
	reg_cost = 0
	for prod_sum in prod_sums:
		reg_cost += prod_sum 

	reg_cost = reg_cost/(orig_len*orig_bs)

	# we also have to divide by the vocab size - for convenience here, we assume we're working with PTB so vocab size is 
	# just size of model outs
	
	reg_cost = reg_cost/normalization
	
	return reg_cost

def sample_outputs(model, model_outs, targets, criterion, n_samples):

	# true prob is the probability of the true label - this hopefully lets us efficiently regularize both Jacobian and Hessian
	# assume criterion type is SplitCrossEntropy (but only has single split)
	# TODO: we can make this step more efficient by sampling the same way as the splits
	model_outs = model_outs.view(
		model_outs.size(0)*model_outs.size(1), model_outs.size(2))
	log_probs = criterion.logprob(model.decoder.weight, model.decoder.bias, model_outs)
	samp_dist = Categorical(logits=log_probs)
	sample_y = samp_dist.sample(torch.Size((n_samples,))).view(n_samples, log_probs.size(0), 1)
	samp_log_probs = torch.gather(log_probs.view(1, log_probs.size(0), log_probs.size(1)).repeat(n_samples, 1, 1), 2, sample_y)
	return sample_y, samp_log_probs

def loss_to_reg(
	model, model_outs, targets, criterion, loss_reg_type, exp_reg_args):
	if loss_reg_type == 'sample_logit':
		_, samp_losses = sample_outputs(model, model_outs, targets, criterion, 1)
		loss_val = -torch.mean(samp_losses)
	if loss_reg_type == 'true':
		model_outs_reshape = model_outs.view(-1, model_outs.size(2))
		loss_val = criterion(model.decoder.weight, model.decoder.bias, model_outs_reshape, targets)
	if loss_reg_type == 'max':
		model_outs_reshape = model_outs.view(-1, model_outs.size(2))
		log_probs = criterion.logprob(model.decoder.weight, model.decoder.bias, model_outs_reshape)
		losses_to_use, _ = log_probs.max(1)
		# sanity check that it's the right dimension
		assert(losses_to_use.size(0) == model_outs_reshape.size(0))
		loss_val = -torch.mean(losses_to_use)
	return loss_val

def jreg_norm_reg(
	model, criterion, targets, emb, hl_vals, model_outs, weight_params, 
	regi, regh, rego, regw, loss_to_use, exp_reg_args):
	orig_len = model_outs.size(0)
	orig_bs = model_outs.size(1)
	vals_to_reg, reg_coeffs = get_reg_coeffs(
		emb, hl_vals, model_outs, weight_params,
		regi,
		regh,
		rego,
		regw)

	j_vals = torch.autograd.grad(
		loss_to_use, 
		vals_to_reg,
		create_graph=True
		)

	reg_cost = 0
	
	for reg_coeff, jac, reg_val in zip(reg_coeffs, j_vals, vals_to_reg):
		curr_cost = torch.sum(torch.pow(jac*reg_val, 2))

		# have to multiply by orig_len*orig_bs since we double-divided by this
		reg_cost += reg_coeff*orig_len*orig_bs*curr_cost
	return reg_cost

def jreg_sample_logit(
	model, criterion, targets,
	emb, hl_vals, model_outs, weight_params,
	exp_reg_args):
	loss_to_use = loss_to_reg(
		model,
		model_outs,
		targets,
		criterion,
		'sample_logit',
		exp_reg_args)
	return jreg_norm_reg(
		model, criterion, targets,
		emb, hl_vals, model_outs, weight_params,
		exp_reg_args['exp_regi'],
		exp_reg_args['exp_regh'],
		exp_reg_args['exp_rego'],
		exp_reg_args['exp_regw'], 
		loss_to_use,
		exp_reg_args)

def loss_reg(
	model, criterion, targets,
	emb, hl_vals, model_outs, weight_params,
	exp_reg_args):
	loss_to_use = loss_to_reg(
		model,
		model_outs,
		targets,
		criterion,
		'true',
		exp_reg_args)
	return jreg_norm_reg(
		model, criterion, targets,
		emb, hl_vals, model_outs, weight_params,
		exp_reg_args['loss_regi'],
		exp_reg_args['loss_regh'],
		exp_reg_args['loss_rego'],
		exp_reg_args['loss_regw'], 
		loss_to_use,
		exp_reg_args)

def jreg_max(
	model, criterion, targets,
	emb, hl_vals, model_outs, weight_params,
	exp_reg_args):
	loss_to_use = loss_to_reg(
		model,
		model_outs,
		targets,
		criterion,
		'max',
		exp_reg_args)
	return jreg_norm_reg(
		model, criterion, targets,
		emb, hl_vals, model_outs, weight_params,
		exp_reg_args['exp_regi'],
		exp_reg_args['exp_regh'],
		exp_reg_args['exp_rego'],
		exp_reg_args['exp_regw'], 
		loss_to_use,
		exp_reg_args)

def taylor_fo(
	model, criterion, targets,
	emb, hl_vals, model_outs, weight_params,
	imp_reg_args, backprop_j=True, loss_val = None):
	orig_len = model_outs.size(0)
	orig_bs = model_outs.size(1)
	model_outs_reshape = model_outs.view(
		orig_len*orig_bs, model_outs.size(2))
	
	if loss_val is None:
		loss_val = criterion(model.decoder.weight, model.decoder.bias, model_outs_reshape, targets)

	# we take the square root here so that the parameters we pass in really measure the strength of the covariance
	vals_to_reg, masked_vals = get_reg_mask(
		emb, hl_vals, model_outs, weight_params,
		imp_reg_args['imp_regi'], 
		imp_reg_args['imp_regh'],
		imp_reg_args['imp_rego'],
		imp_reg_args['imp_regw'],
		use_sqrt=True)


	j_vals = torch.autograd.grad(
		loss_val,
		vals_to_reg,
		create_graph=True)

	reg_cost = 0

	for jac, mask_val in zip(j_vals, masked_vals):
		jac_to_use = jac if backprop_j else jac.detach()
		reg_cost += torch.sum(jac_to_use*mask_val)
	return reg_cost



