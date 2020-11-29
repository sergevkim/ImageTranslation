from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class CommonArguments:
    device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    verbose: bool = True
    version: str = '0.0.1'


@dataclass
class DataArguments:
    batch_size: int = 64
    data_path: Path = Path('./data/cityscapes')
    learning_rate: float = 3e-4
    num_workers: int = 4
    val_ratio: float = 0.1


@dataclass
class TrainArguments:
    max_epoch: int = 10
    scheduler_gamma: float = 0.5
    scheduler_step_size: int = 10


@dataclass
class SpecificArguments:
    encoder_blocks_num: int = 8
    decoder_blocks_num: int = 8

