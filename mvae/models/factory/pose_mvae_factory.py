from ..pose_mvae import PoseMVAE
from ..image_vae_decoder import ImageVaeDecoder
from ..image_vae_encoder import ImageVaeEncoder
from ..pose_vae_decoder import PoseVaeDecoder
from ..pose_vae_encoder import PoseVaeEncoder
from ..ball_pose_vae_decoder import BallPoseVaeDecoder
from ..ball_pose_vae_encoder import BallPoseVaeEncoder
from ..ball_pose_mvae import BallPoseMVAE
from ..pose_vae import PoseVAE
from ..image_vae import ImageVAE


class PoseMVAEFactory(object):
    @staticmethod
    def make_pose_vae(params, regularized=False):
        pose_encoder = PoseVaeEncoder(**params.pose_encoder)
        pose_decoder = PoseVaeDecoder(params.latent_dimension, params.pose_distribution, **params.pose_encoder)
        return PoseVAE(pose_encoder, pose_decoder, params.latent_dimension, regularized, params.pose_distribution)

    @staticmethod
    def make_image_vae(params, regularized=False):
        image_encoder = ImageVaeEncoder(**params.image_encoder)
        image_decoder = ImageVaeDecoder(params.latent_dimension, **params.image_encoder)
        return ImageVAE(image_encoder, image_decoder, params.latent_dimension, regularized)

    @staticmethod
    def make_model(params):
        image_vae = PoseMVAEFactory.make_image_vae(params, True)
        pose_vae = PoseMVAEFactory.make_pose_vae(params, True)
        return PoseMVAE(params, image_vae, pose_vae)

    # @staticmethod
    # def make_ball_pose_mvae_model(params):
    #     image_encoder = ImageVaeEncoder(**params.image_encoder)
    #     image_decoder = ImageVaeDecoder(params.latent_dimension, **params.image_encoder)
    #     pose_encoder = BallPoseVaeEncoder(**params.pose_encoder)
    #     pose_decoder = BallPoseVaeDecoder(params.latent_dimension, **params.pose_encoder)
    #     return BallPoseMVAE(params, image_encoder, image_decoder, pose_encoder, pose_decoder)
