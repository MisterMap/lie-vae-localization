import unittest
import os
from mvae.data import ToyDataModule
from mvae.models import PoseMVAEFactory
import pytorch_lightning as pl
from pytorch_lightning.utilities.parsing import AttributeDict
from mvae.utils import TensorBoardLogger
import torch
import numpy as np


class TestPoseNet(unittest.TestCase):
    def setUp(self) -> None:
        torch.autograd.set_detect_anomaly(True)
        current_folder = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(current_folder, "datasets", "toy_dataset", "three_point_dataset.npz")
        self._data_module = ToyDataModule(dataset_path, rotation_augmentation=False)
        params = AttributeDict(
            optimizer=AttributeDict(),
            image_encoder=AttributeDict(
                hidden_dimensions=[16, 32, 64],
                attention=True,
            ),
            pose_encoder=AttributeDict(
                hidden_dimensions=[256, 256],
                attention=True,
                constant_logvar=False,
                activation_type="swish",
            ),
            latent_dimension=256,
            beta=1,
            gamma=0,
            pose_distribution="se2",
            pose_augmentation=False,
            separate_elbo=False,
            delta_position=1,
            delta_angle=1,
        )
        self._model = PoseMVAEFactory().make_pose_net(params)
        data = np.load(dataset_path, allow_pickle=True)["arr_0"]
        centers = data.item()["point_centers"]
        colors = data.item()["point_colors"]
        self._model.set_points_information(centers, colors, ((0, 4), (0, 4)), 3.2, 0.1, 0.6)

    # noinspection PyTypeChecker
    def test_training(self):
        trainer = pl.Trainer(logger=TensorBoardLogger("lightning_logs"), max_epochs=1, gpus=1)
        trainer.fit(self._model, self._data_module)
