#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/22 00:14
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   03_rnn_layer.py
# @Desc     :   

from pprint import pprint
from torch import nn, Tensor, tensor, float32

from utils.helper import Timer


def main() -> None:
    """ Main Function """
    with Timer("RNN Layer Example"):
        # Define a simple RNN layer
        layer = nn.RNN(
            input_size=5,  # Number of expected features in the input
            hidden_size=3,  # Number of features in the hidden state
            num_layers=2,  # Number of recurrent layers
            batch_first=True  # Input and output tensors are provided as (batch, seq, feature)
        )

        # Create a sample input tensor with shape (batch_size, seq_length, input_size)
        vector: Tensor = tensor(
            [[[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1], [0.2, 0.3, 0.4, 0.5, 0.6]]],  # 1, 3, 5
            dtype=float32
        )
        print(f"{vector.shape} Input Tensor:")
        pprint(vector)

        # Create an initial hidden state tensor with shape (num_layers, batch_size, hidden_size)
        h0: Tensor = tensor(
            [[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]]],  # 2, 1, 3
            dtype=float32
        )
        print(f"{h0.shape} Hx Tensor:")
        pprint(h0)

        # Forward pass through the RNN layer
        output, hidden = layer(vector, h0)
        print(f"{output.shape} RNN Output:")
        pprint(output)
        pprint(f"{hidden.shape} RNN Hidden State:")
        print(hidden)


if __name__ == "__main__":
    main()
