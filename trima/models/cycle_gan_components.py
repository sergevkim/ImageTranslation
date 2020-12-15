from collections import OrderedDict
from typing import Tuple

import torch
from torch import Tensor
from torch.nn import (
    BatchNorm2d,
    Conv2d,
    ConvTranspose2d,
    Dropout,
    InstanceNorm2d,
    LeakyReLU,
    Module,
    ModuleList,
    ReLU,
    Sequential,
    Sigmoid,
    Tanh,
)


class ConvolutionalLayer(Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            padding: int,
            stride: int = 1,
            transpose: bool = False,
            norm: bool = True,
            leaky: bool = False,
        ):
        super().__init__()
        conv_block = OrderedDict()

        if not transpose:
            conv_block['conv'] = Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
            )
        else:
            conv_block['conv'] = ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
            )
            conv_block['conv_1'] = Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            )

        if norm:
            conv_block['norm'] = InstanceNorm2d(num_features=out_channels)

        if not leaky:
            conv_block['relu'] = ReLU()
        else:
            conv_block['relu'] = LeakyReLU(0.2)

        self.conv_block = Sequential(conv_block)

    def forward(self, x):
        x = self.conv_block(x)

        return x


class ResidualBlock(Module):
    def __init__(
            self,
            pieces_num: int = 2,
            dim: int = 256,
        ):
        super().__init__()
        res_block = OrderedDict()

        for i in range(pieces_num):
            piece = OrderedDict()
            piece['conv'] = Conv2d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=3,
                padding=1,
            )
            piece['batch_norm'] = BatchNorm2d(dim)

            if i != pieces_num - 1:
                piece['relu'] = ReLU()

            res_block[f'piece_{i}'] = Sequential(piece)

        self.res_block = Sequential(res_block)

    def forward(self, x):
        x_1 = self.res_block(x)

        return x + x_1


class CycleGANX2YGenerator(Module):
    def __init__(
            self,
            num_blocks: int = 4,
        ):
        super().__init__()

        self.encoder = Sequential(
            ConvolutionalLayer(
                in_channels=3,
                out_channels=64,
                kernel_size=7,
                padding=3,
            ),
            ConvolutionalLayer(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
                stride=2,
            ),
            ConvolutionalLayer(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1,
                stride=2,
            ),
        )

        transformer = OrderedDict()

        for i in range(num_blocks):
            transformer['res_block_{i}'] = ResidualBlock()

        self.transformer = Sequential(transformer)

        self.decoder = Sequential(
            ConvolutionalLayer(
                in_channels=256,
                out_channels=128,
                kernel_size=2,
                padding=0,
                stride=2,
                transpose=True,
            ),
            ConvolutionalLayer(
                in_channels=128,
                out_channels=64,
                kernel_size=2,
                padding=0,
                stride=2,
                transpose=True,
            ),
            ConvolutionalLayer(
                in_channels=64,
                out_channels=3,
                kernel_size=7,
                padding=3,
                transpose=True,
            ),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.transformer(x)
        x = self.decoder(x)

        return x


class CycleGANYDiscriminator(Module):
    def __init__(
            self,
            num_blocks: int = 4,
        ):
        super().__init__()
        self.body = Sequential(
            ConvolutionalLayer(
                in_channels=3,
                out_channels=64,
                kernel_size=4,
                padding=0,
                stride=2,
                norm=False,
                leaky=True,
            ),
            ConvolutionalLayer(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                padding=0,
                stride=2,
                leaky=True,
            ),
            ConvolutionalLayer(
                in_channels=128,
                out_channels=256,
                kernel_size=4,
                padding=0,
                stride=2,
                leaky=True,
            ),
            ConvolutionalLayer(
                in_channels=256,
                out_channels=512,
                kernel_size=4,
                padding=0,
                stride=2,
                leaky=True,
            ),
            Conv2d(
                in_channels=512,
                out_channels=1,
                kernel_size=1,
            ),
            Sigmoid(),
        )

    def forward(self, x):
        output = self.body(x)

        return output


CycleGANY2XGenerator = CycleGANX2YGenerator
CycleGANXDiscriminator = CycleGANYDiscriminator

