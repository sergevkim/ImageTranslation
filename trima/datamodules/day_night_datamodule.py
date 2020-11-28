from pathlib import Path
from PIL import Image
from typing import List, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize, ToTensor


class DayNightDataset(Dataset):
    def __init__(
            self,
            day_image_paths: List[Path],
            night_image_paths: List[Path],
            new_size: Tuple[int, int]=(224, 244),
        ):
        self.day_image_paths = day_image_paths
        self.night_image_paths = night_image_paths
        self.new_size = new_size

    def __len__(self) -> int:
        return len(self.day_image_paths)

    def __getitem__(
            self,
            idx: int,
        ) -> Tuple[Tensor, Tensor]:
        day_image_path = self.day_image_paths[idx]
        day_image = Image.open(day_image_path)
        resized_day_image = Resize()(day_image)
        day_tensor = ToTensor()(resized_day_image)

        night_image_path = self.night_image_paths[idx]
        night_image = Image.open(night_image_path)
        resized_night_image = Resize(size=self.new_size)(night_image)
        night_tensor = ToTensor()(resized_night_image)

        return day_tensor, night_tensor


class ProtostarDataModule:
    def __init__(
            self,
            data_path: Path,
            batch_size: int,
            num_workers: int,
        ):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    @staticmethod
    def prepare_data(
            data_path: Path,
        ):
        pass

    def setup(
            self,
            val_ratio: float,
        ) -> None:
        data = self.prepare_data(
            data_path=self.data_path,
        )
        full_dataset = ProtostarDataset(
            day_image_paths=data['day_image_paths'],
            night_image_paths=data['night_image_paths'],
        )

        full_size = len(full_dataset)
        val_size = int(val_ratio * full_size)
        train_size = full_size - val_size

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            dataset=full_dataset,
            lengths=[train_size, val_size],
        )

    def train_dataloader(self) -> DataLoader:
        train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        return train_dataloader

    def val_dataloader(self) -> DataLoader:
        val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        return val_dataloader

    def test_dataloader(self):
        pass

