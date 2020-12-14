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
    Module,
    ModuleList,
    ReLU,
    Sequential,
    Sigmoid,
    Tanh,
)


class ResidualBlock(Module):
    def __init__(
            self,
            dim: int,
        ):
        super().__init__()

        block = OrderedDict()
        block['conv_0'] = Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            padding=1,
        )
        block['batch_norm_0'] = BatchNorm2d(dim)
        block['relu_0'] = ReLU()
        block['conv_1'] = Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            padding=1,
        )
        block['batch_norm_1'] = BatchNorm2d(dim)
        block['relu_1'] = ReLU()

        self.block = Sequential(block)

    def forward(self, x):
        x_1 = self.block(x)

        return x + x_1


class CycleGANX2YGenerator(Module):
    def __init__(
            self,
            num_blocks: int = 4,
            in_channels: int = 3,
            out_channels: int = 3,
            dim: int = 64,
        ):
        super().__init__()

        blocks = OrderedDict()

        blocks['first_conv'] = Conv2d(
            in_channels=in_channels,
            out_channels=dim,
            kernel_size=1,
        )
        blocks['first_relu'] = ReLU()

        for i in range(1, num_blocks + 1):
            blocks[f'res_block_{i}'] = ResidualBlock(dim=dim)

        blocks['last_conv'] = Conv2d(
            in_channels=dim,
            out_channels=out_channels,
            kernel_size=1,
        )
        blocks['tanh'] = Tanh()

        self.blocks = Sequential(blocks)

    def forward(self, x):
        output = self.blocks(x)

        return output


class CycleGANYDiscriminator(Module):
    def __init__(
            self,
            num_blocks: int = 4,
            in_channels: int = 3,
            out_channels: int = 3,
            dim: int = 64,
        ):
        super().__init__()
        blocks = OrderedDict()

        blocks['first_conv'] = Conv2d(
            in_channels=in_channels,
            out_channels=dim,
            kernel_size=1,
        )
        blocks['first_relu'] = LeakyReLU(0.2)

        for i in range(num_blocks):
            blocks[f'block_{i}'] = Sequential(
                Conv2d(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=3,
                    padding=1,
                ),
                LeakyReLU(0.2),
                BatchNorm2d(dim),
            )

        blocks['last_conv'] = Conv2d(
            in_channels=dim,
            out_channels=out_channels,
            kernel_size=1,
        )
        blocks['tanh'] = Tanh()

        self.blocks = Sequential(blocks)

    def forward(self, x):
        output = self.blocks(x)

        return output


CycleGANY2XGenerator = CycleGANX2YGenerator
CycleGANXDiscriminator = CycleGANYDiscriminator

