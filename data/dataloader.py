import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import zoom
from typing import List, Tuple, Optional


class EMNISTQuantumDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        target_digits: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        target_size: Tuple[int, int] = (4, 4),
        normalize_range: Tuple[float, float] = (0, 1),
        max_samples_per_digit: Optional[int] = None,
    ):
        self.target_size = target_size
        self.normalize_range = normalize_range

        df = pd.read_csv(csv_path)

        df = df[df["label"].isin(target_digits)]

        if max_samples_per_digit:
            df = df.groupby("label").head(max_samples_per_digit)

        self.labels = df["label"].values.astype(int)
        self.images = df.drop("label", axis=1).values

        self.images = self.images.reshape(-1, 16, 16)

        if target_size != (16, 16):
            self.images = self._downsample_images(self.images, target_size)

        self.images = self._normalize(self.images, normalize_range)

    def _downsample_images(
        self, images: np.ndarray, target_size: Tuple[int, int]
    ) -> np.ndarray:
        scale_y = target_size[0] / 16
        scale_x = target_size[1] / 16

        downsampled = np.array(
            [zoom(img, (scale_y, scale_x), order=1) for img in images]
        )
        return downsampled

    def _normalize(
        self, images: np.ndarray, range_tuple: Tuple[float, float]
    ) -> np.ndarray:
        min_val, max_val = range_tuple
        images = (images + 1) / 2
        images = images * (max_val - min_val) + min_val
        return images.astype(np.float32)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = torch.tensor(self.images[idx]).flatten()
        label = torch.tensor(self.labels[idx])
        return image, label


def prepare_amplitude_encoding_data(images: np.ndarray) -> np.ndarray:
    flat = images.reshape(images.shape[0], -1)

    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = flat / norms

    return normalized


def get_dataloader(
    csv_path: str,
    target_digits: List[int] = [0, 1],
    target_size: Tuple[int, int] = (4, 4),
    batch_size: int = 32,
    shuffle: bool = True,
    max_samples_per_digit: Optional[int] = None,
    normalize_range: Tuple[float, float] = (0, 1),
) -> DataLoader:
    dataset = EMNISTQuantumDataset(
        csv_path=csv_path,
        target_digits=target_digits,
        target_size=target_size,
        normalize_range=normalize_range,
        max_samples_per_digit=max_samples_per_digit,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,  # Parallel data loading for faster GPU training
        drop_last=True,
        pin_memory=True,  # Faster data transfer to GPU
    )
