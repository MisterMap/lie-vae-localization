import unittest
import os
from mvae.data import ToyDataModule
from mvae.models import VAE
import pytorch_lightning as pl
from pytorch_lightning.utilities.parsing import AttributeDict


class TestVAE(unittest.TestCase):
    def setUp(self) -> None:
        current_folder = os.path.dirname(os.path.abspath(__file__))
        dataset_folder = os.path.join(current_folder, "datasets", "toy_dataset", "dataset.npz")
        self._data_module = ToyDataModule(dataset_folder)
        params = AttributeDict(
            optimizer=AttributeDict(),
            encoder=AttributeDict(
                hidden_dimensions=[64, 128]
            ),
            latent_dimension=20,
            beta=1,
            gamma=0,
        )
        self._model = VAE(params)

    # noinspection PyTypeChecker
    def test_training(self):
        trainer = pl.Trainer(max_epochs=1, gpus=1)
        trainer.fit(self._model, self._data_module)
