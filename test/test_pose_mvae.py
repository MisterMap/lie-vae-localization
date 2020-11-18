import unittest
import os
from mvae.data import ToyDataModule
from mvae.models import PoseMVAEFactory
import pytorch_lightning as pl
from pytorch_lightning.utilities.parsing import AttributeDict
from mvae.utils import TensorBoardLogger


class TestPoseMVAE(unittest.TestCase):
    def setUp(self) -> None:
        current_folder = os.path.dirname(os.path.abspath(__file__))
        dataset_folder = os.path.join(current_folder, "datasets", "toy_dataset", "dataset.npz")
        self._data_module = ToyDataModule(dataset_folder)
        params = AttributeDict(
            optimizer=AttributeDict(),
            image_encoder=AttributeDict(
                hidden_dimensions=[64, 128],
                attention=True,
            ),
            pose_encoder=AttributeDict(
                hidden_dimensions=[32, 32],
                attention=True,
            ),
            latent_dimension=20,
            beta=1,
            gamma=0,
        )
        self._model = PoseMVAEFactory().make_model(params)

    # noinspection PyTypeChecker
    def test_training(self):
        trainer = pl.Trainer(logger=TensorBoardLogger("lightning_logs"), max_epochs=1, gpus=1)
        trainer.fit(self._model, self._data_module)
