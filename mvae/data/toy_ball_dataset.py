import numpy as np
from torch.utils.data import Dataset
import cv2


class ToyBallDataset(Dataset):
    def __init__(self, path):
        data = np.load(path, allow_pickle=True)["arr_0"]
        self._images = data.item()["images"]
        self._centers = data.item()["centers"]

    def __len__(self):
        return len(self._images)

    def __getitem__(self, index):
        image = self._images[index]
        center = np.array(self._centers[index], dtype=np.float32)
        data = {
            "image": (image.transpose(2, 0, 1) / 255.).astype(np.float32),
            "position": center
        }
        return data
