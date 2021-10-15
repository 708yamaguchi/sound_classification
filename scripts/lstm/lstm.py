#!/usr/bin/env python

#import torch
#from torch.utils.data import DataLoader
#import torch.nn as nn
#import torch.nn.functional as F
#from torch import optim

import chainer
import chainer.functions as F
import chainer.initializers as I
import chainer.links as L

class LSTM(chainer.Chain):
    def __init__(self, n_class=1000):
        super(LSTM, self).__init__()
        insize = 154587
        hidden_size = 64
        self.n_class = n_class
        with self.init_scope():
            self.l1 = L.LSTM(insize, hidden_size)
            self.l2 = L.Linear(hidden_size, n_class)

    def forward(self, x, t=None):
        h = self.l1(x)
        h = self.l2(h)
        print(h.shape)

        self.pred = F.softmax(h)
        if t is None:
            assert not chainer.config.train
            return

        self.loss = F.softmax_cross_entropy(h, t)
        self.acc = F.accuracy(self.pred, t)

        chainer.report({"loss": self.loss, "accuracy": self.acc}, self)

        return self.loss

class LSTM_2(chainer.Chain):
    def __init__(self, n_class=1):
        super(LSTM_2, self).__init__()
        insize = 154587
        hidden_size = 64
        self.n_class = n_class
        with self.init_scope():
            self.l1 = L.LSTM(insize, hidden_size)
            self.l2 = L.Linear(hidden_size, n_class)

    def forward(self, x, t=None):
        #print(x.shape)
        h = self.l1(x)
        #print(h.shape)
        h = self.l2(h)

        #print(h.shape)

        #self.pred = F.sigmoid(h)
        self.pred = h
        #print(self.pred.data.shape)
        if t is None:
            assert not chainer.config.train
            return

        #self.loss = F.sigmoid_cross_entropy(h, t)

        if t is not None:
            t = t.reshape(-1,1)
        self.loss = F.mean_squared_error(h, t)

        chainer.report({"loss": self.loss}, self)
        return self.loss

# class LSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, n_class=2):
#         super(LSTM, self).__init__()
#         self.n_class = n_class
#         self.rnn = nn.LSTM(
#             input_size = input_size,
#             hidden_size = hidden_size,
#             batch_first = True,
#             )
#         self.output = nn.Linear(hidden_size, n_class)

#     def forward(self, x, h=None):
#         output, hn = self.rnn(x, h)
#         y = self.output(output[:, -1, :])
#         return y
