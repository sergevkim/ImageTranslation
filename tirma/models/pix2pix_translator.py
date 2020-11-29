from typing import Tuple

import torch
from torch import Tensor
from torch.nn import (
    Module,
    MSELoss,
)
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR
from torch.optim.optimizer import Optimizer

from tirma.models.pix2pix_blocks import Pix2PixDecoder, Pix2PixEncoder


class Pix2PixTranslator(Module):
    def __init__(
            self,
            device: torch.device,
            learning_rate: float,
            scheduler_step_size: int,
            scheduler_gamma: float,
            verbose: bool,
        ):
        super().__init__()

        self.device = device
        self.criterion = MSELoss()

        self.encoder = Pix2PixEncoder()
        self.decoder = Pix2PixDecoder()

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
        scheduler = LambdaLR(
            optimizer=optimizer,
            step_size=self.step_size,
            gamma=self.gamma,
            verbose=self.verbose,
        )

        return optimizer, scheduler

