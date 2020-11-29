from collections import OrderedDict
from typing import Tuple

import torch
from torch import Tensor
from torch.nn import (
    BatchNorm2d,
    Conv2d,
    ConvTranspose2d,
    Dropout,
    LeakyReLU,
    MaxPool2d,
    Module,
    ReLU,
    Sequential,
)


class EncoderBlock(Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
        ):
        super().__init__()
        block_ordered_dict = OrderedDict(
            conv=Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            batch_norm=BatchNorm2d(num_features=out_channels),
            leaky_relu=LeakyReLU(negative_slope=0.2),
            max_pool=MaxPool2d(),
        )
        self.block_sequential = Sequential(block_ordered_dict)

    def forward(
            self,
            x: Tensor,
        ) -> Tensor:
        return self.block_sequential(x)


class DecoderBlock(Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            dropout_p: float=0.5,
        ):
        super().__init__()
        block_ordered_dict = OrderedDict(
            conv=ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            batch_norm=BatchNorm2d(),
            dropout=Dropout(p=dropout_p),
            relu=ReLU(),
        )
        self.block_sequential = Sequential(block_ordered_dict)

    def forward(
            self,
            x: Tensor,
        ) -> Tensor:
        return self.block_sequential(x)


class Pix2PixEncoder(Module):
    def __init__(
            self,
            blocks_num: int,
        ):
        super().__init__()
        blocks_ordered_dict = OrderedDict()
        blocks_ordered_dict[f'block_0'] = EncoderBlock(
            in_channels=3,
            out_channels=512,
        )

        for i in range(1, blocks_num):
            blocks_ordered_dict[f'block_{i}'] = EncoderBlock(
                in_channels=512,
                out_channels=512,
            )

        self.blocks_sequential = Sequential(blocks_ordered_dict)

    def forward(
            self,
            x: Tensor,
        ) -> Tensor:
        return self.blocks_sequential(x)


class Pix2PixDecoder(Module):
    def __init__(
            self,
        ):
        super().__init__()
        blocks_ordered_dict = OrderedDict()

        for i in range(blocks_num - 1):
            blocks_ordered_dict[f'block_{i}'] = DecoderBlock(
                in_channels=512,
                out_channels=512,
            )

        blocks_ordered_dict[f'block_{blocks_num - 1}'] = DecoderBlock(
            in_channels=512,
            out_channels=3,
        )

        self.blocks_sequential = Sequential(blocks_ordered_dict)

    def forward(
            self,
            x: Tensor,
        ) -> Tensor:
        return self.blocks_sequential(x)

