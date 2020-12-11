from ..image_vae import ImageVAE
from ..image_vae_decoder import ImageVaeDecoder
from ..image_vae_encoder import ImageVaeEncoder
from ..pose_mvae import PoseMVAE
from ..pose_net import PoseNet
from ..pose_vae import PoseVAE
from ..pose_vae_decoder import PoseVaeDecoder
from ..pose_vae_encoder import PoseVaeEncoder


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
    def make_mvae(params):
        image_vae = PoseMVAEFactory.make_image_vae(params, True)
        pose_vae = PoseMVAEFactory.make_pose_vae(params, True)
        return PoseMVAE(params, image_vae, pose_vae)

    def make_model(self, params):
        model_name = params.model_name
        if model_name == "pose_mvae":
            return self.make_mvae(params)
        elif model_name == "pose_net":
            return self.make_pose_net(params)

    @staticmethod
    def make_pose_net(params):
        pose_decoder = PoseVaeDecoder(params.latent_dimension, params.pose_distribution, **params.pose_encoder)
        image_encoder = ImageVaeEncoder(**params.image_encoder)
        return PoseNet(params, image_encoder, pose_decoder, params.latent_dimension, params.pose_distribution)
