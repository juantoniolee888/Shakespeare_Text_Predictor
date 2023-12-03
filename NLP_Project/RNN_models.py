import collections, time, math, random, sys, os, torch
from layers import Embedding, SoftmaxLayer, RNN, LinearLayer


class LSTMCell(RNN):
    def __init__(self, dims):
        torch.nn.Module.__init__(self)
        self.dims = dims
        self.cell = torch.nn.LSTMCell(dims, dims)
        # self.cell = torch.nn.ModuleList([torch.nn.LSTMCell(dims, dims) for _ in range(5)])

    def start(self):
        return (torch.zeros(self.dims), torch.zeros(self.dims))

    def step(self, state, inp):
        state = self.cell(inp, state)
        h, c = state
        return (state, h)


class RNNLanguageModel(torch.nn.Module):
    def __init__(self, input_vocab, output_vocab, dims):
        super().__init__()
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.dims = dims

        self.input = Embedding(len(input_vocab), dims)
        # self.input = CustomEmbeddingLayer(len(input_vocab), dims)

        self.cell = LSTMCell(dims)
        self.output = SoftmaxLayer(dims, len(input_vocab))
        # self.output = torch.nn.Sequential(
        #     torch.nn.Linear(dims, dims),  # Adjust the hidden layer size as needed
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(dims, dims),
        #     SoftmaxLayer(dims, len(input_vocab))
        # )
        
    def start(self):
        return self.cell.start()
        
    def step(self, h, num):
        y = self.input(num)
        h, y = self.cell.step(h, y)
        return h, self.output(y)
    
    def forward(self, nums):
        y = self.input(nums)
        y = self.cell(y)
        return self.output(y)