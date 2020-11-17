import torch.nn as nn


class Swish(nn.Module):
    """https://arxiv.org/abs/1710.05941"""
    def forward(self, x):
        return x * nn.functional.sigmoid(x)


def activation_function(activation_type):
    if activation_type == "relu":
        return nn.ReLU()
    elif activation_type == "tanh":
        return nn.Tanh()
    elif activation_type == "leaky_relu":
        return nn.LeakyReLU()
    elif activation_type == "swish":
        return Swish()
    else:
        raise ValueError(f"Unknown activation type {activation_type}")
