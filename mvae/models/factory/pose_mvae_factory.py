from ..pose_mvae import PoseMVAE
from ..image_vae_decoder import ImageVaeDecoder
from ..image_vae_encoder import ImageVaeEncoder
from ..pose_vae_decoder import PoseVaeDecoder
from ..pose_vae_encoder import PoseVaeEncoder


class PoseMVAEFactory(object):
    @staticmethod
    def make_model(params):
        image_encoder = ImageVaeEncoder(**params.image_encoder)
        image_decoder = ImageVaeDecoder(params.latent_dimension, **params.image_encoder)
        pose_encoder = PoseVaeEncoder(**params.pose_encoder)
        pose_decoder = PoseVaeDecoder(params.latent_dimension, **params.pose_encoder)
        return PoseMVAE(params, image_encoder, image_decoder, pose_encoder, pose_decoder)
