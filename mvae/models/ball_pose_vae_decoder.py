import torch.nn as nn


class BallPoseVaeDecoder(nn.Module):
    def __init__(self, latent_space_size, hidden_dimensions):
        super().__init__()
        self._center_linear = nn.Linear(hidden_dimensions[0], 2)
        self._center_logvar_linear = nn.Linear(hidden_dimensions[0], 2)
        self._decoder = nn.Sequential(*self.make_modules(latent_space_size, hidden_dimensions))

    @staticmethod
    def make_modules(latent_space_size, hidden_dimensions):
        previous_dim = latent_space_size
        modules = []
        for dim in reversed(hidden_dimensions):
            modules.append(nn.Linear(previous_dim, dim))
            modules.append(nn.BatchNorm1d(dim))
            modules.append(nn.LeakyReLU())
            previous_dim = dim
        return modules

    def forward(self, x):
        x = self._decoder(x)
        center = self._center_linear(x)
        center_logvar = self._center_logvar_linear(x)
        return center, center_logvar

