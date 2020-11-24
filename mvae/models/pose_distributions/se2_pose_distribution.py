from .pose_distribution import PoseDistribution
import torch
from liegroups.torch import SE2
import math


class Se2PoseDistribution(PoseDistribution):
    @property
    def mean_dimension(self):
        return 4

    @property
    def logvar_dimension(self):
        return 6

    def log_prob(self, value, mean, logvar):
        mean_matrix = self.make_matrix(mean[:, 0:2], torch.nn.functional.normalize(mean[:, 2:4]))
        value_matrix = self.make_matrix(value[0], torch.nn.functional.normalize(value[1]))
        delta_matrix = torch.bmm(mean_matrix, self.pose_matrix_inverse(value_matrix))
        delta_log = SE2.log(SE2.from_matrix(delta_matrix.clone()))

        inverse_sigma_matrix = torch.inverse(self.get_sigma_matrix(logvar))
        delta_log = torch.bmm(inverse_sigma_matrix, delta_log[:, :, None])[:, :, 0]
        log_determinant = self.get_logvar_determinant(logvar)

        log_prob = torch.sum(delta_log ** 2 / 2., dim=1) - 0.5 * log_determinant - 3 * math.log(math.sqrt(2 * math.pi))
        return log_prob

    def sample(self, mean, logvar):
        mean_matrix = self.make_matrix(mean[:, 0:2], torch.nn.functional.normalize(mean[:, 2:4]))
        sigma_matrix = self.get_sigma_matrix(logvar)
        epsilon = torch.randn(mean.shape[0], 3, device=mean.device)
        delta = torch.bmm(sigma_matrix, epsilon[:, :, None])[:, :, 0]
        positions = torch.bmm(SE2.exp(delta).as_matrix(), mean_matrix)
        translations = torch.zeros(mean.shape[0], 2)
        translations[:, 0] = positions[:, 0, 2]
        translations[:, 1] = positions[:, 1, 2]
        translations = translations.cpu().detach().numpy()
        return translations

    @staticmethod
    def make_matrix(translation, rotation):
        matrix = torch.zeros(rotation.shape[0], 3, 3, device=translation.device)
        matrix[:, 2, 2] = 1
        matrix[:, 0, 0] = rotation[:, 0]
        matrix[:, 0, 1] = -rotation[:, 1]
        matrix[:, 1, 1] = rotation[:, 0]
        matrix[:, 1, 0] = rotation[:, 1]
        matrix[:, 0, 2] = translation[:, 0]
        matrix[:, 1, 2] = translation[:, 1]
        return matrix

    @staticmethod
    def pose_matrix_inverse(matrix):
        result = torch.zeros_like(matrix)
        rotation_part = matrix[:, :2, :2]
        translation_part = matrix[:, :2, 2]
        rotation_part_transposed = torch.transpose(rotation_part, 1, 2)
        result[:, :2, :2] = rotation_part_transposed
        result[:, :2, 2] = -torch.bmm(rotation_part_transposed, translation_part[:, :, None])[:, :, 0]
        result[:, 2, 2] = 1
        return result

    @staticmethod
    def get_sigma_matrix(logvar):
        matrix = torch.zeros(logvar.shape[0], 3, 3, device=logvar.device)
        matrix[:, 0, 0] = torch.exp(0.5 * logvar[:, 0])
        matrix[:, 1, 1] = torch.exp(0.5 * logvar[:, 1])
        matrix[:, 2, 2] = torch.exp(0.5 * logvar[:, 2])
        matrix[:, 1, 0] = torch.exp(0.25 * logvar[:, 1]) * torch.exp(0.25 * logvar[:, 0]) * logvar[:, 3]
        matrix[:, 1, 2] = torch.exp(0.25 * logvar[:, 1]) * torch.exp(0.25 * logvar[:, 2]) * logvar[:, 4]
        matrix[:, 2, 0] = torch.exp(0.25 * logvar[:, 2]) * torch.exp(0.25 * logvar[:, 0]) * logvar[:, 5]
        return matrix

    @staticmethod
    def get_logvar_determinant(logvar):
        return logvar[:, 0] + logvar[:, 1] + logvar[:, 2]
