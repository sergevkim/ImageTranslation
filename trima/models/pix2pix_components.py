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


class EncoderBlock(Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 4,
            stride: int = 2,
            padding: int = 1,
            batch_norm: bool = True,
            negative_slope: float = 0.2,
        ):
        super().__init__()
        block_ordered_dict = OrderedDict()
        block_ordered_dict['conv_0'] = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        block_ordered_dict['conv_1'] = Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )

        if batch_norm:
            block_ordered_dict['batch_norm'] = BatchNorm2d(
                num_features=out_channels,
            )

        block_ordered_dict['leaky_relu'] = LeakyReLU(
            negative_slope=negative_slope,
        )

        self.encoder_block = Sequential(block_ordered_dict)

    def forward(
            self,
            x: Tensor,
        ) -> Tensor:
        return self.encoder_block(x)


class DecoderBlock(Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 4,
            stride: int = 2,
            padding: int = 1,
            dropout_p: float = 0.5,
            batch_norm: bool = True,
        ):
        super().__init__()
        block_ordered_dict = OrderedDict()
        block_ordered_dict['conv_transpose'] = ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        if batch_norm:
            block_ordered_dict['batch_norm'] = BatchNorm2d(
                num_features=out_channels,
            )

        if dropout_p > 0:
            block_ordered_dict['dropout'] = Dropout(p=dropout_p)

        block_ordered_dict['relu'] = ReLU()

        self.decoder_block = Sequential(block_ordered_dict)

    def forward(
            self,
            x: Tensor,
        ) -> Tensor:
        return self.decoder_block(x)


class Pix2PixUNet(Module):
    def __init__(
            self,
            blocks_num: int,
            in_channels: int = 3,
            out_channels: int = 3,
            hidden_dim: int = 64,
        ):
        super().__init__()
        self.blocks_num = blocks_num
        self.hidden_dim = hidden_dim

        self.encoder_blocks = ModuleList()
        self.encoder_blocks.append(EncoderBlock(
            in_channels=in_channels,
            out_channels=hidden_dim,
            batch_norm=False,
        ))
        for i in range(1, blocks_num):
            self.encoder_blocks.append(EncoderBlock(
                in_channels=hidden_dim,
                out_channels=hidden_dim * 2,
            ))
            hidden_dim *= 2

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
                kernel_size=1,
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

        for i, decoder_block in enumerate(self.decoder_blocks):
            if i == 0:
                continue

            x = decoder_block(x)
            x = torch.cat(
                tensors=[x, encoded_x[self.blocks_num - 1 - i]],
                dim=1,
            )

        x = self.last_conv(x)

        return x


class Pix2PixConvNet(Module):
    def __init__(
            self,
            blocks_num: int = 4,
            in_channels: int = 6,
            hidden_dim: int = 128,
        ):
        super().__init__()
        blocks_ordered_dict = OrderedDict()

        blocks_ordered_dict['block_0'] = EncoderBlock(
            in_channels=in_channels,
            out_channels=hidden_dim,
            batch_norm=False,
        )

        for i in range(1, blocks_num):
            blocks_ordered_dict[f'block_{i}'] = EncoderBlock(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
            )

        self.blocks = Sequential(blocks_ordered_dict)
        self.last_conv = Sequential(
            Conv2d(
                in_channels=hidden_dim,
                out_channels=1,
                kernel_size=1,
                padding=1,
            ),
            Sigmoid(),
        )

    def forward(
            self,
            images: Tensor,
            conditions: Tensor,
        ) -> Tensor:
        xx = torch.cat(
            tensors=[images, conditions],
            dim=1,
        )
        x = self.blocks(xx)
        x = self.last_conv(x)

        return x


if __name__ == '__main__':
    generator = Pix2PixUNet(blocks_num=8)
    discriminator = Pix2PixConvNet(
        blocks_num=4,
        in_channels=6,
    )

    print(generator)
    print(discriminator)

