import pytorch_lightning as pl
from .toy_dataset import ToyDataset
import torch.utils.data


class ToyDataModule(pl.LightningDataModule):
    def __init__(self, path, batch_size=128, num_workers=4, rotation_augmentation=True, split=(0.9, 0.1), seed=0):
        super().__init__()
        torch.random.manual_seed(seed)
        self._raw_train_dataset = ToyDataset(path, rotation_augmentation)
        self._rotation_augmentation = rotation_augmentation
        self._batch_size = batch_size
        self._num_workers = num_workers
        train_length = len(self._raw_train_dataset)
        lengths = int(train_length * split[0]), train_length - int(train_length * split[0])

        self._train_dataset, self._validation_dataset = torch.utils.data.random_split(self._raw_train_dataset, lengths)
        print(f"{self._validation_dataset.indices[0]}")
        print(f"[ToyDataModule] - train dataset size {len(self._train_dataset)}")
        print(f"[ToyDataModule] - validation dataset size {len(self._validation_dataset)}")

    def train_dataloader(self, *args, **kwargs):
        self._raw_train_dataset.set_rotation_augmentation(self._rotation_augmentation)
        return torch.utils.data.DataLoader(self._train_dataset, self._batch_size, True, pin_memory=True,
                                           num_workers=self._num_workers)

    def val_dataloader(self, *args, **kwargs):
        self._raw_train_dataset.set_rotation_augmentation(False)
        return torch.utils.data.DataLoader(self._train_dataset, self._batch_size, False, pin_memory=True,
                                           num_workers=self._num_workers)
