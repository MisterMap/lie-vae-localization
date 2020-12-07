import torch

from .pose_distributions import pose_distribution
from .vae_base import VAEBase


class PoseVAE(VAEBase):
    def __init__(self, encoder, decoder, latent_dimension, regularized, pose_distribution_type):
        super().__init__(encoder, decoder, latent_dimension, regularized)
        self.pose_distribution = pose_distribution(pose_distribution_type)

    def nll_part_loss(self, reconstructed_pose, target_pose):
        log_prob = self.pose_distribution.log_prob(target_pose, reconstructed_pose[0], reconstructed_pose[1])
        return torch.sum(log_prob)

    def sample_x(self, z):
        return self.pose_distribution.sample(self.decoder(z))
