import torch.nn as nn


class PoseVaeDecoder(nn.Module):
    def __init__(self, latent_space_size, hidden_dimensions):
        super().__init__()
        self._translation_linear = nn.Linear(hidden_dimensions[0], 2)
        self._translation_logvar_linear = nn.Linear(hidden_dimensions[0], 2)
        self._rotation_linear = nn.Linear(hidden_dimensions[0], 2)
        self._rotation_logvar_linear = nn.Linear(hidden_dimensions[0], 2)

        previous_dim = latent_space_size
        modules = []
        for dim in reversed(hidden_dimensions):
            modules.append(nn.Linear(previous_dim, dim))
            modules.append(nn.BatchNorm1d(dim))
            modules.append(nn.LeakyReLU())
            previous_dim = dim
        self._decoder = nn.Sequential(*modules)

    def forward(self, x):
        x = self._decoder(x)
        translation = self._translation_linear(x)
        rotation = self._rotation_linear(x)
        translation_logvar = self._translation_logvar_linear(x)
        rotation_logvar = self._rotation_logvar_linear(x)
        return translation, rotation, translation_logvar, rotation_logvar

