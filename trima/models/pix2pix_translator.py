from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn import (
    BCELoss,
    L1Loss,
    Module,
    MSELoss,
)
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler, StepLR
from torch.optim.optimizer import Optimizer

from trima.models.pix2pix_components import (
    Pix2PixUNet,
    Pix2PixConvNet,
)


class Pix2PixTranslator(Module):
    def __init__(
            self,
            learning_rate: float = 3e-4,
            scheduler_step_size: int = 10,
            scheduler_gamma: float = 0.5,
            generator_blocks_num: int = 8,
            discriminator_blocks_num: int = 4,
            verbose: bool = True,
            device: torch.device = torch.device('cpu'),
        ):
        super().__init__()

        self.device = device
        self.learning_rate = learning_rate
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma
        self.l1_criterion = L1Loss()
        self.adv_criterion = BCELoss()
        self.verbose = verbose
        self.generator = Pix2PixUNet(
            blocks_num=generator_blocks_num,
            in_channels=3,
            out_channels=3,
        )
        self.discriminator = Pix2PixConvNet(
            blocks_num=discriminator_blocks_num,
            in_channels=3,
        )

    def forward(
            self,
            x: Tensor,
        ) -> Tensor:
        generate_x = self.generator(x)

        return generated_x

    def generator_training_step(
            self,
            ground_truths: Tensor,
            gen_outputs: Tensor,
            fake_predicts: Tensor,
            lambda_coef: float = 100,
        ) -> Tensor:
        g_fake_loss = self.adv_criterion(fake_predicts, 1)
        g_l1_loss = self.l1_criterion(gen_outputs, ground_truths)
        g_loss = g_fake_loss + lambda_coef * g_l1_loss

        return generator_loss

    def discriminator_training_step(
            self,
            fake_predicts: Tensor,
            real_predicts: Tensor,
        ) -> Tensor:
        d_fake_loss = self.adv_criterion(fake_predicts, 0)
        d_real_loss = self.adv_criterion(real_predicts, 1)
        d_loss = d_fake_loss + d_real_loss

        return discriminator_loss

    def training_step(
            self,
            batch: Tensor,
            batch_idx: int,
        ) -> Tensor:
        ground_truths, inputs = batch
        ground_truths = ground_truths.to(self.device)
        inputs = inputs.to(self.device)

        gen_outputs = self.generator(inputs) #TODO add noise
        fake_predicts = self.discriminator(gen_outputs) #TODO add condition inputs
        real_predicts = self.discriminator(ground_truths) #TODO add condition inputs

        generator_loss = self.generator_training_step(
            ground_truths=ground_truths,
            gen_outputs=gen_outputs,
            fake_predicts=fake_predicts,
        )

        discriminator_loss = self.discriminator_training_step(
            fake_predicts=fake_predicts,
            real_predicts=real_predicts,
        )

        loss = generator_loss + discriminator_loss

        return loss

    def training_step_end(self):
        pass

    def training_epoch_end(
            self,
            epoch_idx: int,
        ) -> None:
        if self.verbose:
            print(f"Training epoch {epoch_idx} is over.")

    def validation_step(
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

    def validation_step_end(self):
        pass

    def validation_epoch_end(
            self,
            epoch_idx: int,
        ) -> None:
        if self.verbose:
            print(f"Validation epoch {epoch_idx} is over.")

    def configure_optimizers(
            self,
        ) -> Tuple[List[Optimizer], List[_LRScheduler]]:
        generator_optimizer = Adam(
            params=self.generator.parameters(),
            lr=self.learning_rate,
        )
        discriminator_optimizer = Adam(
            params=self.discriminator.parameters(),
            lr=self.learning_rate,
        )
        optimizers = [generator_optimizer, discriminator_optimizer]

        scheduler = StepLR(
            optimizer=optimizer,
            step_size=self.scheduler_step_size,
            gamma=self.scheduler_gamma,
        )
        schedulers = [scheduler]

        return optimizers, scheduler


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    translator = Pix2PixTranslator(
        device=device,
    ).to(device)

    print(translator)

