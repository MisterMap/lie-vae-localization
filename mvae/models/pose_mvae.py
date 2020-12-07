from .joint_pose_vae import JointPoseVAE
from ..utils.math_torch import *


# noinspection PyArgumentList
class PoseMVAE(JointPoseVAE):
    def __init__(self, params, image_vae, pose_vae):
        super().__init__(params)
        self.image_vae = image_vae
        self.pose_vae = pose_vae

    def generate_z(self, position, image):
        image_z_mu, image_z_logvar = self.image_vae.generate_z(image)
        pose_z_mu, pose_z_logvar = self.pose_vae.generate_z(position)

        mu = torch.cat([pose_z_mu[None], image_z_mu[None], torch.zeros_like(image_z_mu)[None]], dim=0)
        logvar = torch.cat([pose_z_logvar[None], image_z_logvar[None], torch.zeros_like(image_z_logvar)[None]], dim=0)

        z_mu, z_logvar = calculate_distribution_product(mu, logvar)
        return z_mu, z_logvar

    def pose_augmentation_loss(self, batch, beta):
        position = self.random_position(batch["position"])
        losses = self.pose_vae.loss(position, beta)
        return losses

    def forward(self, batch):
        position = batch["position"]
        image = batch["image"]
        batch_size = image.size()[0]
        z_mu, z_logvar = self.generate_z(position, image)
        z = reparametrize(z_mu, z_logvar)
        reconstructed_position = self.pose_vae.decoder(z)
        reconstructed_image = self.image_vae.decoder(z)
        kl_part = kl(z_mu, z_logvar) / batch_size
        image_nll_part = self.image_vae.nll_part_loss(reconstructed_image, image) / batch_size
        pose_nll_part = self.pose_vae.nll_part_loss(reconstructed_position, position) / batch_size
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
        if self.hparams.pose_augmentation:
            augmentation_loss = self.pose_augmentation_loss(batch)
            losses.update(add_prefix("augmentation", augmentation_loss))
            losses["loss"] += losses["augmentation_loss"]
        if self.hparams.separate_elbo:
            image_loss = self.image_vae.loss(image, beta)
            losses.update(add_prefix("image_elbo", image_loss))
            position_loss = self.pose_vae.loss(position, beta)
            losses.update(add_prefix("pose_elbo", position_loss))
            losses["loss"] += losses["image_elbo_loss"] + losses["pose_elbo_loss"]
        reconstructed_output = {
            "image": reconstructed_image,
            "position": reconstructed_position
        }
        return reconstructed_output, losses

    def generate_z_from_pose(self, position):
        z_mu, z_logvar = self.pose_vae.generate_z(position)
        return deregularize_normal_distribution(z_mu, z_logvar)

    def generate_z_from_image(self, image):
        z_mu, z_logvar = self.image_vae.generate_z(image)
        return deregularize_normal_distribution(z_mu, z_logvar)

    def sample_pose_from_z(self, z):
        return self.pose_vae.sample_x(z)

    def reconstruct_image_from_z(self, z):
        return self.image_vae.decoder(z)
