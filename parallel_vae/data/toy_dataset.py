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
        data = {
            "image": (self._images[index].transpose(2, 0, 1) / 255.).astype(np.float32),
            "position": self._trajectory[index].astype(np.float32)
        }
        return data
