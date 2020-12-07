import torch.nn as nn

from .vae_base import VAEBase


class ImageVAE(VAEBase):
    def __init__(self, encoder, decoder, latent_dimension, regularized):
        super().__init__(encoder, decoder, latent_dimension, regularized)
        self._image_loss = nn.MSELoss(reduction="sum")

    def nll_part_loss(self, x, target):
        return self._image_loss(x, target)

    def sample_x(self, z):
        return self.decoder(z)
