import torch.nn as nn


class ImageVaeEncoder(nn.Module):
    def __init__(self, latent_space_size, hidden_dimensions, input_channels=3, kernel_size=3, max_pull=2,
                 image_size=32):
        super().__init__()
        padding = kernel_size // 2
        modules = []
        previous_dim = input_channels
        for dim in hidden_dimensions:
            modules.append(nn.Conv2d(previous_dim, dim, kernel_size, padding=padding, padding_mode="reflect"))
            modules.append(nn.MaxPool2d(max_pull, max_pull))
            modules.append(nn.BatchNorm2d(dim))
            modules.append(nn.LeakyReLU())
            previous_dim = dim
        self._conv_part = nn.Sequential(*modules)
        final_image_size = image_size // max_pull ** len(hidden_dimensions)
        self._fc_mu = nn.Sequential(
            nn.Linear(final_image_size * final_image_size * hidden_dimensions[-1], latent_space_size),
        )
        self._fc_logvar = nn.Sequential(
            nn.Linear(final_image_size * final_image_size * hidden_dimensions[-1], latent_space_size),
        )
        self._final_image_size = final_image_size
        self._last_dim = hidden_dimensions[-1]

    def forward(self, x):
        x = self._conv_part(x)
        x = x.reshape(-1, self._final_image_size * self._final_image_size * self._last_dim)
        return x
