#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/21 21:41
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   model.py
# @Desc     :   

from torch import nn


class CharsRNNTorchModel(nn.Module):
    """ A placeholder class for a character-level RNN model """

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, num_layers: int):
        super().__init__()
        """ Initialise the CharsRNNModel class
        :param vocab_size: size of the vocabulary
        :param embedding_dim: dimension of the embedding layer
        :param hidden_dim: dimension of the hidden layer
        :param num_layers: number of RNN layers
        """
        self._L = vocab_size  # Lexicon/Vocabulary size
        self._N = embedding_dim  # Embedding dimension
        self._M = hidden_size  # Hidden dimension
        self._C = num_layers  # RNN layers count

        self._embed = nn.Embedding(self._L, self._N)
        self._rnn = nn.RNN(self._N, self._M, self._C, batch_first=True)
        self._fc = nn.Linear(self._M, self._L)

        self._init_params()

    def _init_params(self):
        """ Initialize model parameters """
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, X, hx=None):
        """ Forward pass of the model
        :param X: input tensor, shape (batch_size, sequence_length)
        :param hx: hidden state tensor, shape (num_layers, batch_size, hidden_dim)
        :return: output tensor and new hidden state tensor, shapes (batch_size, sequence_length, vocab_size) and (num_layers, batch_size, hidden_dim)
        """
        out = self._embed(X)
        out, hn = self._rnn(out, hx) if hx is not None else self._rnn(out, hx)
        out = self._fc(out)

        return out, hn

    def __repr__(self):
        """ Return a string representation of the model """
        return (f"CharsRNNTorchModel(vocab_size={self._L}, "
                f"embedding_dim={self._N}, "
                f"hidden_dim={self._M}, "
                f"num_layers={self._C}), "
                f"device={next(self.parameters()).device}")
