import sys
import math
import functools

import numpy as np
import math 

import torch
import torch.nn as nn
import torch.nn.functional as F

############# helper functions
def mask_hl(hl_nomask):
	MASK_PROB = 0.5 # keep this fixed for now
	
	m = torch.ones_like(hl_nomask).bernoulli_(MASK_PROB)
	signs = (m - 0.5)*2
	return signs*hl_nomask

def convert_drop_list(drop_list, reg_args, coeff_type, exp_or_imp='exp'):
	# note: exp_or_imp can either be 'hess', 'loss_jac', 'imp'
	if coeff_type == 'spec':
		dropped_vals = []
		for val_dict in drop_list:
			val_type = val_dict['type']
			if val_type == 'drop':
				reg_coeff = reg_args['%s_drop' % (exp_or_imp)]
			if val_type == 'dropatt':
				reg_coeff = reg_args['%s_dropatt' % (exp_or_imp)]

			dropped_vals.append({'reg_coeff' : reg_coeff, 'val' : val_dict['val']})

	return dropped_vals

def compute_exp_reg(loss, fake_loss, model_out, drop_list, reg_args):
	
	if reg_args['exp_reg_type'] == 'none': 
		return 0 

	hess_dropped_vals = convert_drop_list(drop_list, reg_args, reg_args['exp_coeff_type'], exp_or_imp='hess')

	hess_term = 0 # term containing the loss Hessian
	
	reg_types = reg_args['exp_reg_type'].split('+')
	
	hess_reg_type = reg_types[0]

	if hess_reg_type == 'jreg_sample_logit':
		hess_term = jreg_sample_logit(fake_loss, hess_dropped_vals)
	if hess_reg_type == '':
		hess_term = 0

	loss_term = 0

	if len(reg_types) > 1:
		jac_dropped_vals = convert_drop_list(drop_list, reg_args, reg_args['exp_coeff_type'], exp_or_imp='jac')
		jac_reg_type = reg_types[1]
		if jac_reg_type == 'loss_jac':
			loss_term = jreg_sample_logit(loss, jac_dropped_vals) # due to old naming, we're still calling it this 

	return hess_term + loss_term

def jreg_sample_logit(fake_loss, dropped_vals):
	# jacobian reg implementation with sampling logit without backprop through Hessian
	vals_to_reg = []
	reg_coeffs = []

	for val_dict in dropped_vals:
		reg_coeff = val_dict['reg_coeff']
		if reg_coeff > 0:
			vals_to_reg.append(val_dict['val'])
			reg_coeffs.append(reg_coeff)

	first_derivs = torch.autograd.grad(
		fake_loss.float().mean(),
		vals_to_reg,
		create_graph=True)

	# the term above divides by batch size and later gets squared, which means we lose an extra factor in size of fake_loss
	# therefore we must multiply it back
	reg_cost = fake_loss.new_zeros((1))
	for val_to_reg, reg_coeff, deriv in zip(vals_to_reg, reg_coeffs, first_derivs):
		reg_cost += reg_coeff*torch.sum(torch.pow(val_to_reg*deriv, 2))

	return reg_cost.view((1,1))*torch.numel(fake_loss)

def compute_imp_reg(loss, model_out, drop_list, reg_args):
	
	if reg_args['imp_reg_type'] == 'none': 
		return 0 

	dropped_vals = convert_drop_list(drop_list, reg_args, reg_args['imp_coeff_type'], exp_or_imp='imp')

	
	if reg_args['imp_reg_type'] == 'taylor_fo':
		return taylor_fo(loss, model_out, dropped_vals)

def taylor_fo(loss, model_out, dropped_vals):
	vals_to_reg = []
	masked_vals = []

	for val_dict in dropped_vals:
		reg_coeff = val_dict['reg_coeff']
		if reg_coeff > 0:
			vals_to_reg.append(val_dict['val'])
			masked_vals.append(reg_coeff*mask_hl(val_dict['val']))

	j_vals = torch.autograd.grad(
		loss.float().mean(), 
		vals_to_reg,
		create_graph=True)

	reg_cost = model_out.new_zeros((1))
	for jac, masked in zip(j_vals, masked_vals):
		reg_cost += torch.sum(jac*masked)

	return reg_cost.view((1,1))

	