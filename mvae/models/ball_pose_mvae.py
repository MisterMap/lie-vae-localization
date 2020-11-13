from .pose_mvae import PoseMVAE
import torch


class BallPoseMVAE(PoseMVAE):
    def pose_nll_part_loss(self, reconstructed_pose, target_pose):
        truth_center = target_pose

        center = reconstructed_pose[0]
        center_logvar = reconstructed_pose[1]

        translation_loss = -self.normal_log_prob(center, truth_center, center_logvar)
        return torch.sum(translation_loss)
