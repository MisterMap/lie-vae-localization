import torch.nn as nn
import torch
from .utils import activation_function


class PoseVaeEncoder(nn.Module):
    def __init__(self, hidden_dimensions, previous_dim=4, activation_type="leaky_relu", **_):
        super().__init__()
        modules = []
        for dim in hidden_dimensions:
            modules.append(nn.Linear(previous_dim, dim))
            modules.append(nn.BatchNorm1d(dim))
            modules.append(activation_function(activation_type))
            previous_dim = dim
        self._encoder = nn.Sequential(*modules)
        self._last_dim = hidden_dimensions[-1]

    def final_dimension(self):
        return self._last_dim

    def forward(self, x):
        x = torch.cat(x, dim=1)
        return self._encoder(x)
