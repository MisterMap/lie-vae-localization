import torch.nn as nn
import torch
from .utils import activation_function
from .attention import AttentionBlock
from .pose_distributions import pose_distribution


class PoseVaeDecoder(nn.Module):
    def __init__(self, latent_space_size, pose_distribution_type, hidden_dimensions, activation_type="leaky_relu",
                 attention=False, constant_logvar=False, **_):
        super().__init__()
        self._pose_distribution = pose_distribution(pose_distribution_type)
        self._mean_linear = nn.Linear(hidden_dimensions[0], self._pose_distribution.mean_dimension)
        self._logvar_linear = nn.Linear(hidden_dimensions[0], self._pose_distribution.logvar_dimension)
        self._decoder = nn.Sequential(*self.make_modules(latent_space_size, hidden_dimensions, activation_type,
                                                         attention))
        self._logvar = None
        if constant_logvar:
            self._logvar = nn.Parameter(torch.zeros(self._pose_distribution.logvar_dimension))
        self._attention = None
        if attention:
            self._attention = AttentionBlock(latent_space_size)

    @staticmethod
    def make_modules(latent_space_size, hidden_dimensions, activation_type="relu", attention=False):
        modules = []
        previous_dim = latent_space_size
        for dim in reversed(hidden_dimensions):
            modules.append(nn.Linear(previous_dim, dim))
            modules.append(nn.BatchNorm1d(dim))
            modules.append(activation_function(activation_type))
            previous_dim = dim
        return modules

    def forward(self, x):
        if self._attention is not None:
            x = self._attention(x)
        x = self._decoder(x)
        mean = self._mean_linear(x)
        if self._logvar is None:
            logvar = self._logvar_linear(x)
        else:
            logvar = self._logvar
        return mean, logvar

