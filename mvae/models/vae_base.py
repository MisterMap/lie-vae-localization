import torch.nn as nn

from ..utils.math_torch import *


# noinspection PyArgumentList
class VAEBase(nn.Module):
    def __init__(self, encoder, decoder, latent_dimension, regularized=False):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.regularized = regularized

        final_encoder_dimension = self.encoder.final_dimension()
        self._mu_linear = nn.Linear(final_encoder_dimension, latent_dimension)
        self._logvar_linear = nn.Linear(final_encoder_dimension, latent_dimension)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)

    def generate_z(self, x):
        hidden_x = self.encoder(x)
        z_mu = self._mu_linear(hidden_x)
        z_logvar = self._logvar_linear(hidden_x)
        return z_mu, z_logvar

    def loss(self, x, beta=1):
        _, _, nll_part, kl_part = self.forward(x)
        loss = kl_part * beta + nll_part
        losses = {
            "loss": loss,
            "kl_part": kl_part,
            "nll_part": nll_part
        }
        return losses

    def forward(self, x):
        z_mu, z_logvar = self.generate_z(x)
        if self.regularized:
            z_mu, z_logvar = deregularize_normal_distribution(z_mu, z_logvar)
        z = reparametrize(z_mu, z_logvar)
        reconstructed_x = self.decoder(z)
        scale = z.size()[0]
        kl_part = kl(z_mu, z_logvar) / scale
        nll_part = self.nll_part_loss(reconstructed_x, x) / scale
        return reconstructed_x, z, nll_part, kl_part

    def calculate_nll(self, x, sample_count=100):
        x = torch.repeat_interleave(x, sample_count, dim=0)
        log_prediction = self.log_importance_weight(x).reshape(-1, sample_count)
        return -torch.mean(torch.logsumexp(log_prediction, dim=1) - np.log(sample_count))

    def generate_x(self, n=25, device="cpu"):
        eps = torch.randn(size=(n, self._latent_dimension), device=device)
        x = self.decoder(eps)
        return x

    def sample_x(self, z):
        raise NotImplementedError()

    def reconstruct_x(self, x):
        x_mean, _, _, _ = self.forward(x)
        return x_mean

    def nll_part_loss(self, x, target):
        raise NotImplementedError()
