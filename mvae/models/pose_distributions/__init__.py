from .simple_pose_distribution import SimplePoseDistribution
from .se2_pose_distribution import Se2PoseDistribution
from .multi_head_se2_pose_distribution import MultiHeadSe2PoseDistribution


def pose_distribution(pose_distribution_type):
    if pose_distribution_type == "simple":
        return SimplePoseDistribution()
    elif pose_distribution_type == "se2":
        return Se2PoseDistribution()
    elif pose_distribution_type == "se2_multi_head":
        return MultiHeadSe2PoseDistribution()
    else:
        ValueError("Unknown pose distribution type {}".format(pose_distribution_type))
