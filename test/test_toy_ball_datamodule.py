import unittest
import os
from mvae.data import ToyBallDataModule
import torch


class TestToyDataModule(unittest.TestCase):
    def setUp(self) -> None:
        current_folder = os.path.dirname(os.path.abspath(__file__))
        dataset_folder = os.path.join(current_folder, "datasets", "toy_dataset", "ball_dataset.npz")
        self._data_module = ToyBallDataModule(dataset_folder)

    def test_load(self):
        self.assertEqual(len(self._data_module._train_dataset), 4096)
        batches = self._data_module.train_dataloader()
        for batch in batches:
            self.assertEqual(batch["image"].shape, torch.Size([128, 3, 32, 32]))
            self.assertEqual(batch["position"].shape, torch.Size([128, 2]))
            break
