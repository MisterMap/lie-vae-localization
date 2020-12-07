import pytorch_lightning as pl
import torch.nn as nn

from ..utils import *
from ..utils.math_torch import *


# noinspection PyArgumentList
class JointPoseVAE(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters(params)

        self._current_beta = self.hparams.beta
        self._gamma = self.hparams.gamma

        self._centers = None
        self._colors = None
        self._lim_range = ((0, 1), (0, 1))
        self._radius = None
        self._image_size = None
        self._resolution = None
        self.init_weights()

    def set_points_information(self, centers, colors, lim_range, image_size, resolution, radius):
        self._centers = centers
        self._colors = colors
        self._lim_range = lim_range
        self._image_size = image_size
        self._resolution = resolution
        self._radius = radius

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
        image_figure = show_image_from_pose_sampling(self, batch, [0, 1, 2, 3, 4], self._image_size,
                                                     self._radius, self._resolution, self._centers, self._colors,
                                                     dpi=200, figsize=(6, 6), facecolor="gray")
        self.logger.log_figure("image_from_pose_sampling", image_figure, self.global_step)
        for i in range(min(3, batch["image"].shape[0])):
            image_figure = show_image(batch, i)
            self.logger.log_figure(f"image_{i}", image_figure, self.global_step)
            image_figure = show_pose_sampling_from_image(self, batch, i, self._lim_range, self._centers, self._colors,
                                                         dpi=200)
            self.logger.log_figure(f"pose_sampling_from_image_{i}", image_figure, self.global_step)
            image_figure = show_pose_sampling_from_pose(self, batch, i, self._lim_range, self._centers, self._colors,
                                                        dpi=200)
            self.logger.log_figure(f"pose_sampling_from_pose_{i}", image_figure, self.global_step)
            image_figure = show_pose_sampling_from_joint(self, batch, i, self._lim_range, self._centers,
                                                         self._colors, dpi=200)
            self.logger.log_figure(f"pose_sampling_from_joint_{i}", image_figure, self.global_step)

    def random_position(self, position):
        angle = torch.atan2(position[1][:, 0], position[1][:, 1]) + torch.randn_like(
            position[1][:, 0]) * self.hparams.delta_angle
        rotation = torch.zeros_like(position[1])
        rotation[:, 0] = torch.sin(angle)
        rotation[:, 1] = torch.cos(angle)
        position = [position[0] + torch.randn_like(position[0]) * self.hparams.delta_position, rotation]
        return position

    def forward(self, batch):
        return NotImplementedError()

    def generate_z(self, position, image):
        return NotImplementedError()

    def generate_z_from_pose(self, position):
        return NotImplementedError()

    def generate_z_from_image(self, image):
        return NotImplementedError()

    def sample_pose_from_z(self, z):
        return NotImplementedError()

    def reconstruct_image_from_z(self, z):
        return NotImplementedError()

    def reconstruct_image_from_pose(self, position):
        z_mu, z_logvar = self.generate_z_from_pose(position)
        z = reparametrize(z_mu, z_logvar)
        return self.reconstruct_image_from_z(z)

    def reconstruct_image_from_image(self, image):
        z_mu, z_logvar = self.generate_z_from_image(image)
        z = reparametrize(z_mu, z_logvar)
        return self.reconstruct_image_from_z(z)

    def sample_pose_from_image(self, image):
        z_mu, z_logvar = self.generate_z_from_image(image)
        z = reparametrize(z_mu, z_logvar)
        return self.sample_pose_from_z(z)

    def sample_pose_from_pose(self, position):
        z_mu, z_logvar = self.generate_z_from_pose(position)
        z = reparametrize(*deregularize_normal_distribution(z_mu, z_logvar))
        return self.sample_pose_from_z(z)

    def sample_pose_from_joint(self, position, image):
        z_mu, z_logvar = self.generate_z(position, image)
        z = reparametrize(z_mu, z_logvar)
        return self.sample_pose_from_z(z)

    def configure_optimizers(self):
        if "betas" in self.hparams.optimizer.keys():
            beta1 = float(self.hparams.optimizer.betas.split(" ")[0])
            beta2 = float(self.hparams.optimizer.betas.split(" ")[1])
            self.hparams.optimizer.betas = (beta1, beta2)
        optimizer = torch.optim.Adam(self.parameters(), **self.hparams.optimizer)
        if "scheduler" in self.hparams.keys():
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **self.hparams.scheduler)
            return [optimizer], [scheduler]
        return optimizer
