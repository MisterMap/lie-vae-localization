import torch.nn as nn


def activation_function(activation_type):
    if activation_type == "relu":
        return nn.ReLU()
    elif activation_type == "tanh":
        return nn.Tanh()
    elif activation_type == "leaky_relu":
        return nn.LeakyReLU()
    else:
        raise ValueError(f"Unknown activation type {activation_type}")
