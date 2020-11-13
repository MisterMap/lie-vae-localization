import pytorch_lightning as pl
from .toy_ball_dataset import ToyBallDataset
import torch.utils.data


class ToyBallDataModule(pl.LightningDataModule):
    def __init__(self, path, batch_size=128, num_workers=4):
        super().__init__()
        self._train_dataset = ToyBallDataset(path)
        self._batch_size = batch_size
        self._num_workers = num_workers
        print(f"[ToyDataModule] - train dataset size {len(self._train_dataset)}")

    def train_dataloader(self, *args, **kwargs):
        return torch.utils.data.DataLoader(self._train_dataset, self._batch_size, True, pin_memory=True,
                                           num_workers=self._num_workers)
