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
            batch_norm=BatchNorm2d(num_features=out_channels),
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
            blocks_num: int,
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


class Pix2PixUNet(Module):
    def __init__(
            self,
            blocks_num: int,
        ):
        super().__init__()
        self.blocks_num = blocks_num

        self.encoder_blocks = ModuleList()
        self.encoder_blocks.append(EncoderBlock(
            in_channels=3,
            out_channels=512,
        ))
        for i in range(1, blocks_num):
            self.encoder_blocks.append(EncoderBlock(
                in_channels=512,
                out_channels=512,
            ))

        self.decoder_blocks = ModuleList()
        self.decoder_blocks.append(DecoderBlock(
            in_channels=512,
            out_channels=512,
        ))
        for i in range(1, blocks_num - 1):
            self.decoder_blocks.append(DecoderBlock(
                in_channels=1024,
                out_channels=512,
            ))
        self.decoder_blocks.append(DecoderBlock(
            in_channels=1024,
            out_channels=3,
        ))

    def forward(
            self,
            x: Tensor,
        ) -> Tensor:
        encoded_x = list(None for i in range(self.blocks_num))

        for i, encoder_block in enumerate(self.encoder_blocks):
            x = encoder_block(x)
            encoded_x[i] = x

        x = self.decoder_blocks[0](x)

        for i, decoder_block in enumerate(self.decoder_blocks):
            if i == 0:
                continue

            xx = torch.cat(
                tensors=[x, encoded_x[self.blocks_num - 1 - i]],
                dim=1,
            )
            x = decoder_block(xx)

        return x


if __name__ == '__main__':
    model = Pix2PixUNet(blocks_num=8)
    print(model)

