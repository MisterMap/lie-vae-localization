import numpy as np
from torch.utils.data import Dataset


class ToyDataset(Dataset):
    def __init__(self, path):
        data = np.load(path, allow_pickle=True)["arr_0"]
        self._images = data.item()["images"]
        self._trajectory = data.item()["trajectory"]

    def __len__(self):
        return len(self._trajectory)

    def __getitem__(self, index):
        translation = self._trajectory[index].astype(np.float32)[:2]
        angle = self._trajectory[index].astype(np.float32)[2]
        rotation = np.array([np.cos(angle), np.sin(angle)])
        data = {
            "image": (self._images[index].transpose(2, 0, 1) / 255.).astype(np.float32),
            "position": (translation, rotation)
        }
        return data
