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
        block_ordered_dict['conv'] = Conv2d(
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


class CycleGANUNet(Module):
    def __init__(
            self,
            blocks_num: int,
            in_channels: int = 3,
            out_channels: int = 3,
            hidden_dim: int = 512,
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


class CycleGANConvNet(Module):
    def __init__(
            self,
            blocks_num: int = 4,
            in_channels: int = 3,
        ):
        super().__init__()
        blocks_ordered_dict = OrderedDict()

        blocks_ordered_dict['block_0'] = EncoderBlock(
            in_channels=in_channels,
            out_channels=in_channels * 2,
            batch_norm=False,
        )

        cur_channels = in_channels * 2

        for i in range(1, blocks_num):
            blocks_ordered_dict[f'block_{i}'] = EncoderBlock(
                in_channels=cur_channels,
                out_channels=cur_channels * 2,
            )
            cur_channels *= 2

        self.blocks = Sequential(blocks_ordered_dict)
        self.last_conv = Sequential(
            Conv2d(
                in_channels=cur_channels,
                out_channels=1,
                kernel_size=1,
                padding=1,
            ),
            Sigmoid(),
        )

    def forward(
            self,
            x: Tensor
        ) -> Tensor:

        x = self.blocks(x)
        x = self.last_conv(x)

        return x


if __name__ == '__main__':
    generator = CycleGANUNet(blocks_num=8)
    discriminator = CycleGANConvNet(
        blocks_num=4,
        in_channels=3,
    )

    print(generator)
    print(discriminator)
