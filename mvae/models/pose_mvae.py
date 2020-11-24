from .pose_distributions import pose_distribution

import pytorch_lightning as pl
import torch.nn as nn

from ..utils import *


# noinspection PyArgumentList
class PoseMVAE(pl.LightningModule):
    def __init__(self, params, image_encoder, image_decoder, pose_encoder, pose_decoder):
        super().__init__()
        self.save_hyperparameters(params)
        self.image_encoder = image_encoder
        self.image_decoder = image_decoder
        self.pose_encoder = pose_encoder
        self.pose_decoder = pose_decoder
        self.pose_distribution = pose_distribution(params.pose_distribution)

        self._image_loss = nn.MSELoss(reduction="sum")

        self._current_beta = self.hparams.beta
        self._gamma = self.hparams.gamma

        image_final_encoder_dimension = self.image_encoder.final_dimension()
        self._image_mu_linear = nn.Linear(image_final_encoder_dimension, self.hparams.latent_dimension)
        self._image_logvar_linear = nn.Linear(image_final_encoder_dimension, self.hparams.latent_dimension)

        pose_final_encoder_dimension = self.pose_encoder.final_dimension()
        self._pose_mu_linear = nn.Linear(pose_final_encoder_dimension, self.hparams.latent_dimension)
        self._pose_logvar_linear = nn.Linear(pose_final_encoder_dimension, self.hparams.latent_dimension)

        self._centers = None
        self._colors = None
        self._lim_range = ((0, 1), (0, 1))
        self.init_weights()

    def set_points_information(self, centers, colors, lim_range):
        self._centers = centers
        self._colors = colors
        self._lim_range = lim_range

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
        generated, losses = self.forward(batch)
        train_losses = {}
        for key, value in losses.items():
            train_losses[f"train_{key}"] = value
        self.log_dict(train_losses)
        return losses["loss"]

    def validation_step(self, batch, batch_index):
        if batch_index == 0:
            self.show_images(batch)
        generated, losses = self.forward(batch)
        val_losses = {}
        for key, value in losses.items():
            val_losses[f"val_{key}"] = value
        self.log_dict(val_losses)
        return losses["loss"]

    def show_images(self, batch):
        image_reconstruction_figure = show_pose_mvae_reconstruction(self, batch, 10, dpi=200, figsize=(3, 6),
                                                                    facecolor="gray")
        self.logger.log_figure("image_reconstruction_figure", image_reconstruction_figure, self.global_step)
        pose_reconstruction_figure = show_pose_mvae_reconstruction_pose(self, batch, 10, dpi=200, figsize=(3, 6),
                                                                        facecolor="gray")
        self.logger.log_figure("pose_reconstruction_figure", pose_reconstruction_figure, self.global_step)
        for i in range(min(3, batch["image"].shape[0])):
            image_figure = show_image(batch, i)
            self.logger.log_figure(f"image_{i}", image_figure, self.global_step)
            image_figure = show_pose_sampling(self, batch, i, self._lim_range, self._centers, self._colors, dpi=200)
            self.logger.log_figure(f"pose_sampling_{i}", image_figure, self.global_step)

    def generate_z(self, position, image):
        image_hidden = self.image_encoder(image)
        pose_hidden = self.pose_encoder(position)

        pose_z_mu = self._pose_mu_linear(pose_hidden)
        pose_z_logvar = self._pose_logvar_linear(pose_hidden)

        image_z_mu = self._image_mu_linear(image_hidden)
        image_z_logvar = self._image_logvar_linear(image_hidden)

        mu = torch.cat([pose_z_mu[None], image_z_mu[None], torch.zeros_like(image_z_mu)[None]], dim=0)
        logvar = torch.cat([pose_z_logvar[None], image_z_logvar[None], torch.zeros_like(image_z_logvar)[None]], dim=0)

        z_mu, z_logvar = self.calculate_distribution_product(mu, logvar)
        z = self.reparametrize(z_mu, z_logvar)
        return z, z_mu, z_logvar

    @staticmethod
    def calculate_distribution_product(mu, logvar):
        log_denominator = torch.logsumexp(-logvar, dim=0)
        result_logvar = -log_denominator
        weights = torch.exp(-logvar - log_denominator[None])
        result_mu = torch.sum(mu * weights, dim=0)
        return result_mu, result_logvar

    def forward(self, batch):
        position = batch["position"]
        image = batch["image"]
        batch_size = image.size()[0]
        z, z_mu, z_logvar = self.generate_z(position, image)
        reconstructed_position = self.pose_decoder(z)
        reconstructed_image = self.image_decoder(z)
        kl_part = self.kl(z_mu, z_logvar) / batch_size
        x: torch.Tensor
        image_nll_part = self.image_nll_part_loss(reconstructed_image, image) / batch_size
        pose_nll_part = self.pose_nll_part_loss(reconstructed_position, position) / batch_size
        beta = self.calculate_beta()
        loss = kl_part * beta + image_nll_part + pose_nll_part
        losses = {
            "loss": loss,
            "kl_part": kl_part,
            "elbo": kl_part + image_nll_part + pose_nll_part,
            "nll_part": image_nll_part + pose_nll_part,
            "image_nll_part": image_nll_part,
            "pose_nll_part": pose_nll_part,
        }
        reconstructed_output = {
            "image": reconstructed_image,
            "position": reconstructed_position
        }
        return reconstructed_output, losses

    def reconstruct_image_from_position(self, position, reparametrize=True):
        pose_hidden = self.pose_encoder(position)

        pose_z_mu = self._pose_mu_linear(pose_hidden)
        pose_z_logvar = self._pose_logvar_linear(pose_hidden)

        return self._reconstruct_image(pose_z_mu, pose_z_logvar, reparametrize)

    def reconstruct_image_from_image(self, image, reparametrize=True):
        image_hidden = self.image_encoder(image)

        image_z_mu = self._image_mu_linear(image_hidden)
        image_z_logvar = self._image_logvar_linear(image_hidden)
        return self._reconstruct_image(image_z_mu, image_z_logvar, reparametrize)

    def _reconstruct_image(self, z_mu, z_logvar, reparametrize=True):
        z = self._generate_z(z_mu, z_logvar, reparametrize)
        reconstructed_image = self.image_decoder(z)
        return reconstructed_image

    def _reconstruct_pose(self, z_mu, z_logvar, reparametrize=True):
        z = self._generate_z(z_mu, z_logvar, reparametrize)
        reconstructed_image = self.pose_decoder(z)
        return reconstructed_image

    def _generate_z(self, z_mu, z_logvar, reparametrize=True):
        mu = torch.cat([z_mu[None], torch.zeros_like(z_mu)[None]], dim=0)
        logvar = torch.cat([z_logvar[None], torch.zeros_like(z_logvar)[None]], dim=0)

        z_mu, z_logvar = self.calculate_distribution_product(mu, logvar)
        if reparametrize:
            z = self.reparametrize(z_mu, z_logvar)
        else:
            z = z_mu
        return z

    def reconstruct_pose_from_image(self, image, reparametrize):
        image_hidden = self.image_encoder(image)

        image_z_mu = self._image_mu_linear(image_hidden)
        image_z_logvar = self._image_logvar_linear(image_hidden)
        return self._reconstruct_pose(image_z_mu, image_z_logvar, reparametrize)

    @staticmethod
    def reparametrize(z_mu, z_logvar):
        epsilon = torch.randn_like(z_mu)
        z = z_mu + torch.exp(0.5 * z_logvar) * epsilon
        return z

    def configure_optimizers(self):
        if "betas" in self.hparams.optimizer.keys():
            beta1 = float(self.hparams.optimizer.betas.split(" ")[0])
            beta2 = float(self.hparams.optimizer.betas.split(" ")[1])
            self.hparams.optimizer.betas = (beta1, beta2)
        optimizer = torch.optim.Adam(self.parameters(), **self.hparams.optimizer)
        if "scheduler" in self.hparams.keys():
            torch.optim.lr_scheduler.StepLR(optimizer, **self.hparams.scheduler)
        return optimizer

    def image_nll_part_loss(self, x, target):
        return self._image_loss(x, target)

    def pose_nll_part_loss(self, reconstructed_pose, target_pose):
        log_prob = self.pose_distribution.log_prob(target_pose, reconstructed_pose[0], reconstructed_pose[1])
        return torch.sum(log_prob)

    @staticmethod
    def kl(z_mean, z_logvar):
        kl_divergence_element = -0.5 * (-z_mean ** 2 - torch.exp(z_logvar) + 1 + z_logvar)
        return kl_divergence_element.sum()
