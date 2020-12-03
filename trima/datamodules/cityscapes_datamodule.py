from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import (
    Compose,
    Normalize,
    Resize,
    ToTensor,
)


class CityscapesDataset(Dataset):
    def __init__(
            self,
            dual_image_paths: List[Path],
            new_size: Tuple[int, int]=(256, 256),
        ):
        self.dual_image_paths = dual_image_paths
        self.new_size = new_size

        transform_list = [
            ToTensor(),
            #Normalize(
            #    (0.485, 0.456, 0.406),
            #    (0.229, 0.224, 0.225),
            #),
        ]
        self.transform = Compose(transform_list)

    def __len__(self) -> int:
        return len(self.dual_image_paths)

    def __getitem__(
            self,
            idx: int,
        ) -> Tuple[Tensor, Tensor]:
        dual_image_path = self.dual_image_paths[idx]
        dual_image = Image.open(dual_image_path)
        dual_tensor = self.transform(dual_image)

        c, h, w = dual_tensor.shape
        left_tensor, right_tensor = torch.split(
            tensor=dual_tensor,
            split_size_or_sections=[w // 2, w // 2],
            dim=2,
        )

        return left_tensor, right_tensor


class CityscapesDataModule:
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
            one_batch_overfit: bool=False,
        ) -> Dict[str, List[Path]]:
        trainval_data_path = data_path / 'train'

        trainval_image_paths = sorted(list(
            p
            for p in trainval_data_path.glob('*')
        ))

        if one_batch_overfit:
            print("WARNING! One batch overfitting!")
            trainval_image_paths = trainval_image_paths[:3]
            print(trainval_image_paths)

        data = dict(
            trainval_image_paths=trainval_image_paths,
        )

        return data

    def setup(
            self,
            val_ratio: float,
            one_batch_overfit: bool=False,
        ) -> None:
        data = self.prepare_data(
            data_path=self.data_path,
            one_batch_overfit=one_batch_overfit,
        )
        trainval_dataset = CityscapesDataset(
            dual_image_paths=data['trainval_image_paths'],
        )

        trainval_size = len(trainval_dataset)
        val_size = max(int(val_ratio * trainval_size), 1)
        train_size = trainval_size - val_size

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            dataset=trainval_dataset,
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

