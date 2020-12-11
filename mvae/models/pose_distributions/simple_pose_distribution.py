from .pose_distribution import PoseDistribution
import torch
from ...utils.math import normal_log_prob


class SimplePoseDistribution(PoseDistribution):
    @property
    def mean_dimension(self):
        return 4

    @property
    def logvar_dimension(self):
        return 4

    def log_prob(self, value, mean, logvar):
        mean_translation = mean[:, 0:2]
        mean_rotation = torch.nn.functional.normalize(mean[:, 2:4])

        value_translation = value[0]
        value_rotation = torch.nn.functional.normalize(value[1])

        translation_logvar = logvar[:, 0:2]
        rotation_logvar = logvar[:, 2:4]

        translation_loss = -normal_log_prob(value_translation, mean_translation, translation_logvar)
        rotation_loss = -normal_log_prob(value_rotation, mean_rotation, rotation_logvar)
        return translation_loss + rotation_loss

    def sample(self, mean, logvar):
        mu = mean[:, 0:2]
        logvar = logvar[:, 0:2]

        epsilon = torch.randn_like(mu)

        positions = mu + epsilon * torch.exp(0.5 * logvar)
        positions = positions.cpu().detach().numpy()
        return positions

    def sample_position(self, mean, logvar):
        if logvar.dim() < 2:
            logvar = logvar[None]
        mu = mean[:, 0:2]
        epsilon = torch.randn_like(mu)
        translations = mu + epsilon * torch.exp(0.5 * logvar[:, 0:2])

        mu = torch.nn.functional.normalize(mean[:, 2:4])
        epsilon = torch.randn_like(mu)
        rotations = mu + epsilon * torch.exp(0.5 * logvar[:, 2:4])
        positions = torch.zeros(mean.shape[0], 3)
        positions[:, 0] = translations[:, 0]
        positions[:, 1] = translations[:, 1]
        positions[:, 2] = torch.atan2(rotations[:, 1], rotations[:, 0])
        return positions.detach().cpu().numpy()

