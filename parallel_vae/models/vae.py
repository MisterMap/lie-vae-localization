import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

from .image_vae_decoder import ImageVaeDecoder
from .image_vae_encoder import ImageVaeEncoder


# noinspection PyArgumentList
class VAE(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters(params)
        self.encoder = ImageVaeEncoder(self.hparams.latent_dimension, **self.hparams.encoder)
        self.decoder = ImageVaeDecoder(self.hparams.latent_dimension, **self.hparams.encoder)
        self._loss = nn.MSELoss(reduction="sum")
        self._sample_loss = nn.MSELoss(reduction="none")
        self._current_beta = self.hparams.beta
        self._gamma = self.hparams.gamma

        self._mu_linear = nn.Linear(self.hparams.encoder.hidden_dimensions[-1], self.hparams.latent_dimension)
        self._logvar_linear = nn.Linear(self.hparams.encoder.hidden_dimensions[-1], self.hparams.latent_dimension)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)

    def calculate_beta(self):
        return self._current_beta

    def update_beta(self):
        self._current_beta += self._gamma
        self._current_beta = np.clip(self._current_beta, 0, 1)

    def on_train_epoch_end(self, outputs) -> None:
        super().on_train_epoch_end(outputs)
        self.update_beta()

    def training_step(self, batch, batch_index):
        x, loss, nll_part, kl_part = self.forward(batch["image"])
        self.log("train_loss", loss)
        self.log("train_kl", kl_part)
        self.log("train_nll", nll_part)
        self.log("train_elbo", kl_part + nll_part)
        return loss

    def generate_z(self, x):
        hidden_x = self.encoder(x)
        z_mu = self._mu_linear(hidden_x)
        z_logvar = self._logvar_linear(hidden_x)
        epsilon = torch.randn_like(z_mu)
        z = z_mu + torch.exp(0.5 * z_logvar) * epsilon
        return z, z_mu, z_logvar

    def forward(self, x):
        z, z_mu, z_logvar = self.generate_z(x)
        reconstructed_x = self.decoder(z)
        scale = x.size()[0]
        kl_part = self.kl(z_mu, z_logvar) / scale
        x: torch.Tensor
        nll_part = self.nll_part_loss(reconstructed_x, x) / scale
        beta = self.calculate_beta()
        loss = kl_part * beta + nll_part
        return reconstructed_x, loss, nll_part, kl_part

    def log_importance_weight(self, x):
        z, z_mu, z_logvar = self.generate_z(x)
        reconstructed_x = self.decoder(z)
        log_p_x = -torch.sum(self._sample_loss(reconstructed_x, x), dim=[1, 2, 3])
        log_p_z = torch.sum(torch.distributions.normal.Normal(0, 1).log_prob(z), dim=1)
        log_q_z = torch.sum(torch.distributions.normal.Normal(z_mu, torch.exp(0.5 * z_logvar)).log_prob(z),
                            dim=1)
        return log_p_x + log_p_z - log_q_z

    @staticmethod
    def log_p_z(z):
        return torch.sum(torch.distributions.normal.Normal(0, 1.).log_prob(z), dim=1)

    def calculate_nll(self, x, sample_count=100):
        x = torch.repeat_interleave(x, sample_count, dim=0)
        log_prediction = self.log_importance_weight(x).reshape(-1, sample_count)
        return -torch.mean(torch.logsumexp(log_prediction, dim=1) - np.log(sample_count))

    def generate_x(self, n=25, device="cpu"):
        eps = torch.randn(size=(n, self._latent_dimension), device=device)
        x = self.decoder(eps)
        return x

    def reconstruct_x(self, x):
        x_mean, _, _, _ = self.forward(x)
        return x_mean

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.hparams.optimizer)
        return optimizer

    def nll_part_loss(self, x, target):
        return self._loss(x, target)

    @staticmethod
    def kl(z_mean, z_logvar):
        kl_divergence_element = -0.5 * (-z_mean ** 2 - torch.exp(z_logvar) + 1 + z_logvar)
        return kl_divergence_element.sum()
