import torch
import torch.nn as nn
from torch.autograd import Variable
from lstm import LSTMLayer
from lstm import torch_lstm_to_custom
import random 

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, tie_weights=False, no_dropout=False, custom_lstm=False):
        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.use_dropout = not no_dropout

        if wdrop is None:
            wdrop = 0
        wdrop = wdrop if self.use_dropout else 0
        assert rnn_type in ['LSTM', 'QRNN', 'GRU'], 'RNN type is not supported'
        if rnn_type == 'LSTM':
            # we need to use own lstm for second order derivative
            if not custom_lstm:
                self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), 1, dropout=0) for l in range(nlayers)]
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
            else:
                self.rnns = [LSTMLayer(ninp if l==0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid)) for l in range(nlayers)]
                self.rnns = [WeightDrop(rnn, ['weight_hh'], dropout=wdrop) for rnn in self.rnns]
        if rnn_type == 'GRU':
            self.rnns = [torch.nn.GRU(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else ninp, 1, dropout=0) for l in range(nlayers)]
            self.rnns = [WeightDrop(rnn, ['weight_hh'], dropout=wdrop) for rnn in self.rnns]
        elif rnn_type == 'QRNN':
            from torchqrnn import QRNNLayer
            self.rnns = [QRNNLayer(input_size=ninp if l == 0 else nhid, hidden_size=nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True) for l in range(nlayers)]
            for rnn in self.rnns:
                rnn.linear = WeightDrop(rnn.linear, ['weight'], dropout=wdrop)
        print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            #if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.wdrop = wdrop
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights

    def reset(self):
        if self.rnn_type == 'QRNN': [r.reset() for r in self.rnns]

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, use_dropout=True, return_h=False):
        if not use_dropout:
            if self.rnn_type == 'QRNN': raise NotImplementedError
        use_dropout = use_dropout and self.use_dropout
        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training and use_dropout else 0)
        if use_dropout:
            emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            
            # if self.use_dropout isn't true, then the rnn doesn't even have a use_dropout param
            if self.use_dropout and self.rnn_type == 'LSTM':
                raw_output, new_h = rnn(raw_output, hidden[l], use_dropout=use_dropout)
            else:
                raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)

            if l != self.nlayers - 1:
                if use_dropout:
                    raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden

        if use_dropout:
            output = self.lockdrop(raw_output, self.dropout)
        else:
            output = raw_output
        outputs.append(output)

        result = output.view(output.size(0)*output.size(1), output.size(2))
        if return_h:
            return result, hidden, raw_outputs, outputs, emb
        return result, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_(),
                    weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_())
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU':
            return [weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()
                    for l in range(self.nlayers)]

    # convert to using custom lstm function
    def convert_to_custom(self):
        for ind, rnn in enumerate(self.rnns):
            if type(rnn) == WeightDrop:
                rnn.module, type_changed = torch_lstm_to_custom(rnn.module)
                if type_changed:
                    self.rnns[ind] = WeightDrop(rnn.module, ['weight_hh'], dropout=args.wdrop)
            else:
                self.rnns[ind], _ = torch_lstm_to_custom(rnn)
                self.rnns[ind] = WeightDrop(self.rnns[ind], ['weight_hh'], dropout=args.wdrop)
        return self

    def get_weight_params(self):
        weight_params = []
        for key, val in self.named_parameters():
            if key.split('_')[-1] == 'raw':
                weight_params.append(val)
        return weight_params

    def dup_hidden(self, hidden, reps):
        if isinstance(hidden, torch.Tensor):
            return hidden.repeat(1, reps, 1)
        else:
            return tuple(self.dup_hidden(h, reps) for h in hidden)