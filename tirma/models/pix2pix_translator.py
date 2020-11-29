from typing import Tuple

import torch
from torch import Tensor
from torch.nn import (
    Module,
    MSELoss,
)
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler, StepLR
from torch.optim.optimizer import Optimizer

from tirma.models.pix2pix_blocks import Pix2PixDecoder, Pix2PixEncoder


class Pix2PixTranslator(Module):
    def __init__(
            self,
            device: torch.device,
            learning_rate: float,
            scheduler_step_size: int,
            scheduler_gamma: float,
            encoder_blocks_num: int,
            decoder_blocks_num: int,
            verbose: bool,
        ):
        super().__init__()

        self.device = device
        self.learning_rate = learning_rate
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma
        self.criterion = MSELoss()
        self.verbose = verbose

        self.encoder = Pix2PixEncoder(
            blocks_num=encoder_blocks_num,
        )
        self.decoder = Pix2PixDecoder(
            blocks_num=decoder_blocks_num,
        )

    def forward(
            self,
            x: Tensor,
        ) -> Tensor:
        encoded_x = self.encoder(x)
        decoded_x = self.decoder(encoded_x)

        return decoded_x

    def training_step(
            self,
            batch: Tensor,
            batch_idx: int,
        ) -> Tensor:
        left, right = batch
        left = left.to(self.device)
        right = right.to(self.device)

        left_hat = self(right)

        loss = self.criterion(left_hat, left)

        return loss

    def training_step_end(self):
        pass

    def training_epoch_end(self):
        pass

    def validation_step(
            self,
            batch: Tensor,
            batch_idx: int,
        ) -> Tensor:
        pass

    def validation_step_end(self):
        pass

    def validation_epoch_end(self):
        pass

    def configure_optimizers(self) -> Tuple[Optimizer, _LRScheduler]:
        optimizer = Adam(
            params=self.parameters(),
            lr=self.learning_rate,
        )
        scheduler = StepLR(
            optimizer=optimizer,
            step_size=self.scheduler_step_size,
            gamma=self.scheduler_gamma,
        )

        return optimizer, scheduler

