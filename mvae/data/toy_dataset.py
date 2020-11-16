import numpy as np
from torch.utils.data import Dataset
import cv2


class ToyDataset(Dataset):
    def __init__(self, path, rotation_augmentation=True):
        data = np.load(path, allow_pickle=True)["arr_0"]
        self._images = data.item()["images"]
        self._trajectory = data.item()["trajectory"]
        self._rotation_augmentation = rotation_augmentation

    def __len__(self):
        return len(self._trajectory)

    @staticmethod
    def rotate_data_point(image, angle):
        angle_delta = np.random.random() * 2 * np.pi
        angle = angle + angle_delta
        angle_delta = angle_delta / np.pi * 180
        matrix = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle_delta, 1)
        image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), borderValue=(255, 255, 255))
        return image, angle

    def __getitem__(self, index):
        image = self._images[index]
        translation = self._trajectory[index].astype(np.float32)[:2]
        angle = self._trajectory[index].astype(np.float32)[2]
        if self._rotation_augmentation:
            image, angle = self.rotate_data_point(image, angle)
        rotation = np.array([np.cos(angle), np.sin(angle)]).astype(np.float32)
        data = {
            "image": (image.transpose(2, 0, 1) / 255.).astype(np.float32),
            "position": (translation, rotation)
        }
        return data
