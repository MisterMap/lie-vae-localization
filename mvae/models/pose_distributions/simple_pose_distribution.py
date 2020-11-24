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
