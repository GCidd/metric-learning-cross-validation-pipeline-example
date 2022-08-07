from typing import List, Tuple
import numpy as np
import os
import cv2
from sklearn.preprocessing import LabelEncoder
from torch import Tensor
from torch.cuda import empty_cache as cuda_empty_cache
from torch.utils.data import Dataset


class PathListDataset(Dataset):
    def __init__(self, x_paths: List[str], y: List, transform=None, device: str = 'cuda'):
        self.X, self.y = x_paths, y
        self.classes = np.unique(self.y)
        self.n_classes = len(self.classes)
        self._lbl_encoder = LabelEncoder()
        self.y = self._lbl_encoder.fit_transform(self.y).astype(np.float64)

        self._cache = {
            x: None
            for x in self.X
        }

        self.data = self.X
        self.targets = self.y

        self.transform = transform
        self.device = device

    @staticmethod
    def walk_dataset(directory: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Walks the image directory provided and returns the file paths of all the files in the directory 
        and the class labels as numpy arrays.

        Args:
            directory (str): Directory to walk.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Image file paths and class labels.
        """
        X_dirs = []
        y = []
        for root, _, filenames in os.walk(directory):
            if len(filenames) == 0: continue
            _class = os.path.basename(root)
            y += [_class] * len(filenames)
            X_dirs += [
                os.path.join(root, filename)
                for filename in filenames
            ]

        return np.array(X_dirs), np.array(y)

    @staticmethod
    def calculate_mean_std(directory: str, transform=None) -> Tuple[List, List]:
        """
        Calculates the running mean and std of an image dataset of the image directory provided.

        Args:
            directory (str): Directory of image dataset.
            transform (optional): torch.Transform to apply on the image before returning it. Defaults to None.

        Returns:
            Tuple[List, List]: The running mean and std of the image dataset.
        """
        channels_sum = np.zeros(3)
        channels_sqr_sum = np.zeros(3)
        imgs_count = 0
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                filepath = os.path.join(root, filename)
                img = cv2.imread(filepath).astype(float)

                if transform is not None:
                    img = transform(img).numpy()
                    img = np.moveaxis(img, 0, -1)

                h, w, c = img.shape
                img = img.reshape(h * w, c) / 255.

                imgs_count += 1
                channels_sum += img.sum(0)
                channels_sqr_sum += (img ** 2).sum(0)

        mean = channels_sum / (imgs_count * h * w)
        variance = (channels_sqr_sum / (imgs_count * h * w)) - (mean ** 2)
        std = np.sqrt(variance)

        return mean, std

    def empty_cache(self):
        for img in self._cache.values():
            del img
        cuda_empty_cache()
        self._cache = {}
    
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img_path = self.X[idx]
        if self._cache.get(img_path, None) is not None:
            img = self._cache[img_path]
        else:
            img = cv2.imread(img_path)
            if self.transform:
                img = self.transform(img)
            self._cache[img_path] = img

        y = Tensor(
            [self.y[idx]]
        )

        img = img.to(self.device)
        y = y.to(self.device).long()

        return img, y
