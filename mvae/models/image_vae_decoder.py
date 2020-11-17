import torch
import torch.nn as nn
from torchnlp.nn import Attention


class ImageVaeDecoder(nn.Module):
    def __init__(self, latent_space_size, hidden_dimensions, input_channels=3, kernel_size=3, max_pull=2,
                 image_size=32, attention=False):
        super().__init__()
        final_image_size = image_size // max_pull ** len(hidden_dimensions)
        self._fc_latent = nn.Linear(latent_space_size, final_image_size * final_image_size * hidden_dimensions[-1])
        if attention:
            self._fc_latent = nn.Sequential(Attention(latent_space_size), self._fc_latent)
        padding = kernel_size // 2

        previous_dim = input_channels
        modules = []
        for dim in hidden_dimensions:
            modules.append(nn.Conv2d(dim, previous_dim, kernel_size, padding=padding, padding_mode="reflect"))
            modules.append(nn.Upsample(scale_factor=max_pull, mode="bilinear", align_corners=False))
            modules.append(nn.LeakyReLU())
            modules.append(nn.BatchNorm2d(dim))
            previous_dim = dim
        modules.reverse()
        self._mu_deconv_part = nn.Sequential(*modules)
        self._final_image_size = final_image_size
        self._last_dim = hidden_dimensions[-1]

    def forward(self, x):
        x = self._fc_latent(x).reshape(x.size()[0], self._last_dim, self._final_image_size, self._final_image_size)
        return torch.sigmoid(self._mu_deconv_part(x))
