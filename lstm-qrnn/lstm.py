import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class LSTMLayer(nn.Module):
	def __init__(self, input_sz, hidden_sz):
		super().__init__()
		self.input_sz = input_sz
		self.hidden_sz = hidden_sz
		self.weight_ih = Parameter(torch.Tensor(input_sz, hidden_sz*4))
		self.weight_hh = Parameter(torch.Tensor(hidden_sz, hidden_sz*4))
		self.bias = Parameter(torch.Tensor(hidden_sz*4))
		self.init_weights()

	def init_weights(self):
		for p in self.parameters():
			if p.data.ndimension() >= 2:
				nn.init.xavier_uniform_(p.data)
			else:
				nn.init.zeros_(p.data)

	def forward(self, x, init_states):
		seq_len, bs, _ = x.size()
		hidden_seq = []
		if init_states is None:
			h_t, c_t = (torch.zeros(bs, self.hidden_sz).to(x.device), 
						torch.zeros(bs, self.hidden_sz).to(x.device))
		else:
			h_t, c_t = init_states

			# the below is because the first dim is n_layers, which is always 1
			h_t = h_t[0]
			c_t = c_t[0]

		HS = self.hidden_sz
		for t in range(seq_len):
			x_t = x[t, :, :]
			gates = torch.matmul(x_t, self.weight_ih) +\
				torch.matmul(h_t, self.weight_hh) + self.bias
			i_t, f_t, g_t, o_t = (
				torch.sigmoid(gates[:, :HS]), 
				torch.sigmoid(gates[:, HS:HS*2]),
				torch.tanh(gates[:, HS*2:HS*3]),
				torch.sigmoid(gates[:, HS*3:]),
			)
			c_t = f_t*c_t + i_t*g_t
			h_t = o_t*torch.tanh(c_t)
			hidden_seq.append(h_t.unsqueeze(0))
		hidden_seq = torch.cat(hidden_seq, dim=0)

		# unsqueeze h_t and c_t to account for number of layers
		return hidden_seq, (h_t.unsqueeze(0), c_t.unsqueeze(0))

def torch_lstm_to_custom(torch_lstm):
	if type(torch_lstm) is not LSTMLayer:
		custom_lstm = LSTMLayer(torch_lstm.input_size, torch_lstm.hidden_size)
		custom_lstm.weight_ih = Parameter(torch_lstm.weight_ih_l0.transpose(0, 1).data)
		if type(torch_lstm.weight_hh_l0) == Parameter:
			custom_lstm.weight_hh = Parameter(torch_lstm.weight_hh_l0.transpose(0, 1).data)
		else:
			custom_lstm.weight_hh = Parameter(torch_lstm.weight_hh_l0.transpose(0, 1).data)
		bias_sum = torch_lstm.bias_ih_l0 + torch_lstm.bias_hh_l0
		custom_lstm.bias = Parameter(bias_sum.data)
		return custom_lstm, True
	else:
		return torch_lstm, False
