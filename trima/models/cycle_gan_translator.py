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

from cycle_gan_components import (
    CycleGANX2YGenerator,
    CycleGANXDiscriminator,
    CycleGANY2XGenerator,
    CycleGANYDiscriminator,
)


class CycleGANTranslator(Module):
    def __init__(
            self,
            device: torch.device = torch.device('cpu'),
            learning_rate: float = 3e-4,
            scheduler_step_size: int = 10,
            scheduler_gamma: float = 0.5,
            verbose: bool = True,
            generator_blocks_num: int = 8,
            discriminator_blocks_num: int = 4,
        ):
        super().__init__()

        self.device = device
        self.learning_rate = learning_rate
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma
        self.verbose = verbose

        self.y_l1_criterion = L1Loss()
        self.x_l1_criterion = L1Loss()
        self.y_adv_criterion = BCELoss()
        self.x_adv_criterion = BCELoss()

        self.x2y_generator = CycleGANX2YGenerator()
        self.y2x_generator = CycleGANY2XGenerator()
        self.y_discriminator = CycleGANYDiscriminator()
        self.x_discriminator = CycleGANXDiscriminator()

    def forward(
            self,
            x: Tensor,
        ) -> Tensor:
        generated_x = self.generator(x)

        return generated_x

    def y_adv_training_step(
            y_predicts,
            y_1_predicts,
        ):
        fake_loss = self.y_adv_criterion(
            y_1_predicts,
            torch.ones_like(y_1_predicts),
        )
        real_loss = self.y_adv_criterion(
            y_predicts,
            torch.zeros_like(y_predicts),
        )

        loss = fake_loss + real_loss

        return loss

    def x_adv_training_step(
            x_predicts,
            x_1_predicts,
        ):
        fake_loss = self.x_adv_criterion(
            x_1_predicts,
            torch.ones_like(x_1_predicts),
        )
        real_loss = self.x_adv_criterion(
            x_predicts,
            torch.zeros_like(x_predicts),
        )

        loss = fake_loss + real_loss

        return loss

    def cyc_training_step(
            self,
            x_2,
            y_2,
            x,
            y,
        ) -> Tensor:
        x_l1_loss = self.x_l1_criterion(
            x_2,
            x,
        )
        y_l1_loss = self.y_l1_criterion(
            y_2,
            y,
        )

        loss = x_l1_loss + y_l1_loss

        return loss

    def training_step(
            self,
            batch: Tensor,
            batch_idx: int,
        ) -> Tensor:
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        y_1 = self.x2y_generator(x)
        x_1 = self.y2x_generator(y)
        x_2 = self.y2x_generator(y_1)
        y_2 = self.x2y_generator(x_1)

        y_predicts = self.y_discriminator(y)
        x_predicts = self.x_discriminator(x)
        y_1_predicts = self.y_discriminator(y_1)
        x_1_predicts = self.x_discriminator(x_1)
        #x_2_predicts = self.x_discriminator(x_2)
        #y_2_predicts = self.y_discriminator(y_2)

        y_adv_loss = self.y_adv_training_step(
            y_predicts,
            y_1_predicts,
        )
        x_adv_loss = self.x_adv_training_step(
            x_predicts,
            x_1_predicts,
        )
        cyc_loss = self.cyc_training_step(
            x_2,
            y_2,
            x,
            y,
        )

        loss = x_adv_loss + y_adv_loss + cyc_loss

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
        loss = self.training_step(batch, batch_idx)

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

        generator_scheduler = StepLR(
            optimizer=generator_optimizer,
            step_size=self.scheduler_step_size,
            gamma=self.scheduler_gamma,
        )
        discriminator_scheduler = StepLR(
            optimizer=discriminator_optimizer,
            step_size=self.scheduler_step_size,
            gamma=self.scheduler_gamma,
        )
        schedulers = [generator_scheduler, discriminator_scheduler]

        return optimizers, schedulers


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    translator = CycleGANTranslator(
        device=device,
    ).to(device)

    print(translator)

