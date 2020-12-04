from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class CommonArguments:
    device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    verbose: bool = True
    version: str = '1.0.0'


@dataclass
class DataArguments:
    batch_size: int = 2
    data_path: Path = Path('./data/facades')
    learning_rate: float = 3e-4
    num_workers: int = 4
    val_ratio: float = 0.1


@dataclass
class TrainArguments:
    max_epoch: int = 10
    one_batch_overfit: bool = True
    scheduler_gamma: float = 0.5
    scheduler_step_size: int = 10


@dataclass
class SpecificArguments:
    generator_blocks_num: int = 8
    discriminator_blocks_num: int = 4

