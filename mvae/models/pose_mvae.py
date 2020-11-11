import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import math


# noinspection PyArgumentList
class PoseMVAE(pl.LightningModule):
    def __init__(self, params, image_encoder, image_decoder, pose_encoder, pose_decoder):
        super().__init__()
        self.save_hyperparameters(params)
        self.image_encoder = image_encoder
        self.image_decoder = image_decoder
        self.pose_encoder = pose_encoder
        self.pose_decoder = pose_decoder

        self._image_loss = nn.MSELoss(reduction="sum")

        self._current_beta = self.hparams.beta
        self._gamma = self.hparams.gamma

        image_final_encoder_dimension = self.image_encoder.final_dimension()
        self._image_mu_linear = nn.Linear(image_final_encoder_dimension, self.hparams.latent_dimension)
        self._image_logvar_linear = nn.Linear(image_final_encoder_dimension, self.hparams.latent_dimension)

        pose_final_encoder_dimension = self.pose_encoder.final_dimension()
        self._pose_mu_linear = nn.Linear(pose_final_encoder_dimension, self.hparams.latent_dimension)
        self._pose_logvar_linear = nn.Linear(pose_final_encoder_dimension, self.hparams.latent_dimension)

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
        generated, losses = self.forward(batch)
        train_losses = {}
        for key, value in train_losses.items():
            train_losses[f"train_{key}"] = value
        self.log_dict(train_losses)
        return losses["loss"]

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

    def reconstruct_image_from_position(self, position):
        pose_hidden = self.pose_encoder(position)

        pose_z_mu = self._pose_mu_linear(pose_hidden)
        pose_z_logvar = self._pose_logvar_linear(pose_hidden)

        return self._reconstruct_image(pose_z_mu, pose_z_logvar)

    def reconstruct_image_from_image(self, image):
        image_hidden = self.image_encoder(image)

        image_z_mu = self._image_mu_linear(image_hidden)
        image_z_logvar = self._image_logvar_linear(image_hidden)
        return self._reconstruct_image(image_z_mu, image_z_logvar)

    def _reconstruct_image(self, z_mu, z_logvar):
        mu = torch.cat([z_mu[None], torch.zeros_like(z_mu)[None]], dim=0)
        logvar = torch.cat([z_logvar[None], torch.zeros_like(z_logvar)[None]], dim=0)

        z_mu, z_logvar = self.calculate_distribution_product(mu, logvar)
        z = self.reparametrize(z_mu, z_logvar)
        reconstructed_image = self.image_decoder(z)
        return reconstructed_image

    @staticmethod
    def reparametrize(z_mu, z_logvar):
        epsilon = torch.randn_like(z_mu)
        z = z_mu + torch.exp(0.5 * z_logvar) * epsilon
        return z

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.hparams.optimizer)
        return optimizer

    def image_nll_part_loss(self, x, target):
        return self._image_loss(x, target)

    def pose_nll_part_loss(self, reconstructed_pose, target_pose):
        translation = reconstructed_pose[0]
        rotation = torch.nn.functional.normalize(reconstructed_pose[1])

        truth_translation = target_pose[0]
        truth_rotation = torch.nn.functional.normalize(target_pose[1])

        translation_logvar = reconstructed_pose[2]
        rotation_logvar = reconstructed_pose[3]

        translation_loss = -self.normal_log_prob(translation, truth_translation, translation_logvar)
        rotation_loss = -self.normal_log_prob(rotation, truth_rotation, rotation_logvar)
        return torch.sum(translation_loss) + torch.sum(rotation_loss)

    @staticmethod
    def normal_log_prob(value, mu, logvar):
        log_prob = -((value - mu) ** 2) / (2 * torch.exp(logvar)) - 0.5 * logvar - math.log(math.sqrt(2 * math.pi))
        return log_prob

    @staticmethod
    def kl(z_mean, z_logvar):
        kl_divergence_element = -0.5 * (-z_mean ** 2 - torch.exp(z_logvar) + 1 + z_logvar)
        return kl_divergence_element.sum()
