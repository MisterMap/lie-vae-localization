import pytorch_lightning as pl
import torch.nn as nn

from .pose_distributions import pose_distribution
from ..utils.pose_net_result_evaluator import *
from ..utils.math_torch import *


# noinspection PyArgumentList
class PoseNet(pl.LightningModule):
    def __init__(self, params, image_encoder, pose_decoder, latent_dimension, pose_distribution_type):
        super().__init__()
        self.save_hyperparameters(params)

        self.encoder = image_encoder
        self.decoder = pose_decoder

        final_encoder_size = self.encoder.final_dimension()
        self._hidden_layer = nn.Linear(final_encoder_size, latent_dimension)

        self.pose_distribution = pose_distribution(pose_distribution_type)

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

    def on_train_epoch_end(self, outputs) -> None:
        super().on_train_epoch_end(outputs)

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

    def forward(self, batch):
        image = batch["image"]
        hidden = self.encoder(image)
        hidden = self._hidden_layer(hidden)
        position = self.decoder(hidden)
        losses = {
            "loss": self.nll_part_loss(position, batch["position"])
        }
        return position, losses

    def nll_part_loss(self, reconstructed_pose, target_pose):
        log_prob = self.pose_distribution.log_prob(target_pose, reconstructed_pose[0], reconstructed_pose[1])
        return torch.sum(log_prob)

    def sample_pose_from_image(self, image):
        hidden = self.encoder(image)
        hidden = self._hidden_layer(hidden)
        position = self.decoder(hidden)
        return self.pose_distribution.sample(position[0], position[1])

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
