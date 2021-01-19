from .se2_pose_distribution import Se2PoseDistribution
import torch
from liegroups.torch import SE2
import math


class MultiHeadSe2PoseDistribution(Se2PoseDistribution):
    def __init__(self, head_count=10):
        self._head_count = head_count

    @property
    def mean_dimension(self):
        return 4 * self._head_count

    @property
    def logvar_dimension(self):
        return 6 * self._head_count

    def log_prob(self, value, mean, logvar):
        batch_size = mean.shape[0]
        mean = mean.reshape(mean.shape[0] * self._head_count, mean.shape[1] // self._head_count)
        logvar = logvar.reshape(logvar.shape[0] * self._head_count, logvar.shape[1] // self._head_count)
        new_value = (torch.repeat_interleave(value[0], self._head_count, 0),
                     torch.repeat_interleave(value[1], self._head_count, 0))
        log_prob = super().log_prob(new_value, mean, logvar).view(batch_size, self._head_count)
        return log_prob.sum(dim=1)

    def sample(self, mean, logvar):
        batch_size = mean.shape[0]
        mean = mean.view(batch_size * self._head_count, mean.shape[1] // self._head_count)
        mean_matrix = self.make_matrix(mean[:, :2], mean[:, 2:])
        log_mean = SE2.log(SE2.from_matrix(mean_matrix, normalize=False))
        if log_mean.dim() < 2:
            log_mean = log_mean[None]
        logvar = logvar.view(batch_size * self._head_count, logvar.shape[1] // self._head_count)
        inverse_sigma_matrix = self.get_inverse_sigma_matrix(logvar)
        inverse_covariance_matrix = torch.bmm(inverse_sigma_matrix.transpose(1, 2), inverse_sigma_matrix)
        result_inverse_covariance_matrix = torch.sum(inverse_covariance_matrix.reshape(-1, self._head_count, 3, 3), dim=1)
        result_covariance_matrix = torch.inverse(result_inverse_covariance_matrix)
        factors = torch.bmm(result_covariance_matrix.repeat_interleave(self._head_count, 0), inverse_covariance_matrix)
        scaled_log_mean = torch.bmm(factors, log_mean[:, :, None])[:, :, 0]
        result_log_mean = torch.sum(scaled_log_mean.reshape(-1, self._head_count, 3), dim=1)
        mean_matrix = SE2.exp(result_log_mean).as_matrix()
        if mean_matrix.dim() < 3:
            mean_matrix = mean_matrix[None]

        try:
            # inverse_sigma_matrix = torch.cholesky(result_inverse_covariance_matrix)
            # sigma_matrix = torch.inverse(inverse_sigma_matrix)
            sigma_matrix = torch.cholesky(result_covariance_matrix + torch.eye(3, device=mean.device) * 1e-4)
        except RuntimeError as msg:
            print(inverse_covariance_matrix)
            print(result_inverse_covariance_matrix)
            print(result_covariance_matrix)
            print("Cholesky error", msg)
            sigma_matrix = (torch.eye(3, device=mean.device) * 1e4).expand(batch_size, 3, 3)

        epsilon = torch.randn(batch_size, 3, device=mean.device)
        delta = torch.bmm(sigma_matrix, epsilon[:, :, None])[:, :, 0]
        delta_matrix = SE2.exp(delta).as_matrix()
        if delta_matrix.dim() < 3:
            delta_matrix = delta_matrix[None]
        position_matrix = torch.bmm(mean_matrix, delta_matrix)
        positions = torch.zeros(batch_size, 3)
        positions[:, 0] = position_matrix[:, 0, 2]
        positions[:, 1] = position_matrix[:, 1, 2]
        positions[:, 2] = torch.atan2(position_matrix[:, 1, 0], position_matrix[:, 0, 0])
        positions = positions.cpu().detach().numpy()
        return positions

    def mean_position(self, mean, logvar):
        batch_size = mean.shape[0]
        mean = mean.view(batch_size * self._head_count, mean.shape[1] // self._head_count)
        mean_matrix = self.make_matrix(mean[:, :2], mean[:, 2:])
        log_mean = SE2.log(SE2.from_matrix(mean_matrix, normalize=False))
        if log_mean.dim() < 2:
            log_mean = log_mean[None]
        logvar = logvar.view(batch_size * self._head_count, logvar.shape[1] // self._head_count)
        inverse_sigma_matrix = self.get_inverse_sigma_matrix(logvar)
        inverse_covariance_matrix = torch.bmm(inverse_sigma_matrix.transpose(1, 2), inverse_sigma_matrix)
        result_inverse_covariance_matrix = torch.sum(inverse_covariance_matrix.reshape(-1, self._head_count, 3, 3), dim=1)
        result_covariance_matrix = torch.inverse(result_inverse_covariance_matrix)
        factors = torch.bmm(result_covariance_matrix.repeat_interleave(self._head_count, 0), inverse_covariance_matrix)
        scaled_log_mean = torch.bmm(factors, log_mean[:, :, None])[:, :, 0]
        result_log_mean = torch.sum(scaled_log_mean.reshape(-1, self._head_count, 3), dim=1)
        mean_matrix = SE2.exp(result_log_mean).as_matrix()
        if mean_matrix.dim() < 3:
            mean_matrix = mean_matrix[None]
        positions = torch.zeros(batch_size, 2, device=mean.device)
        positions[:, 0] = mean_matrix[:, 0, 2]
        positions[:, 1] = mean_matrix[:, 1, 2]
        return positions

