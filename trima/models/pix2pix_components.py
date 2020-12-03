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
    Tanh,
)


class EncoderBlock(Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 4,
            stride: int = 2,
            padding: int = 1,
        ):
        super().__init__()
        block_ordered_dict = OrderedDict(
            conv = Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            #batch_norm=BatchNorm2d(num_features=out_channels),
            leaky_relu=LeakyReLU(negative_slope=0.2),
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
            kernel_size: int = 4,
            stride: int = 2,
            padding: int = 1,
            dropout_p: float = 0.5,
        ):
        super().__init__()
        block_ordered_dict = OrderedDict(
            conv=ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            #batch_norm=BatchNorm2d(num_features=out_channels),
            dropout=Dropout(p=dropout_p),
            relu=ReLU(),
        )
        self.block_sequential = Sequential(block_ordered_dict)

    def forward(
            self,
            x: Tensor,
        ) -> Tensor:
        return self.block_sequential(x)


class Pix2PixUNet(Module):
    def __init__(
            self,
            blocks_num: int,
            hidden_dim: int = 512,
            out_channels: int = 3,
        ):
        super().__init__()
        self.blocks_num = blocks_num
        self.hidden_dim = hidden_dim

        self.encoder_blocks = ModuleList()
        self.encoder_blocks.append(EncoderBlock(
            in_channels=3,
            out_channels=hidden_dim,
        ))
        for i in range(1, blocks_num):
            self.encoder_blocks.append(EncoderBlock(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
            ))

        self.decoder_blocks = ModuleList()
        self.decoder_blocks.append(DecoderBlock(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            dropout_p=0.5,
        ))

        for i in range(1, blocks_num):
            self.decoder_blocks.append(DecoderBlock(
                in_channels=hidden_dim * 2,
                out_channels=hidden_dim,
                dropout_p=0.5 if i < 3 else 0,
            ))

        self.last_conv = Sequential(
            Conv2d(
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            Tanh(),
        )

    def forward(
            self,
            x: Tensor,
        ) -> Tensor:
        encoded_x = list(None for i in range(self.blocks_num))

        for i, encoder_block in enumerate(self.encoder_blocks):
            x = encoder_block(x)
            encoded_x[i] = x.clone().detach()

        x = self.decoder_blocks[0](x)

        for i, decoder_block in enumerate(self.decoder_blocks):
            if i == 0:
                continue

            xx = torch.cat(
                tensors=[x, encoded_x[self.blocks_num - 1 - i]],
                dim=1,
            )
            x = decoder_block(xx)

        x = self.last_conv(x)

        return x


if __name__ == '__main__':
    generator = Pix2PixUNet(blocks_num=8)
    discriminator = Pix2PixConvNet()

    print(generator)
    print(discriminator)

