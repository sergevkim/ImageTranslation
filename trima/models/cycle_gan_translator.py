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

from trima.models.cycle_gan_components import (
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
            lambda_coef: float = 20,
        ):
        super().__init__()

        self.device = device
        self.learning_rate = learning_rate
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma
        self.verbose = verbose

        self.lambda_coef = lambda_coef

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
        generated_y = self.x2y_generator(x)

        return generated_y

    def y_adv_training_step(
            self,
            y_predicts,
            y_1_predicts,
        ):
        fake_loss = self.y_adv_criterion(
            y_1_predicts,
            torch.zeros_like(y_1_predicts),
        )
        real_loss = self.y_adv_criterion(
            y_predicts,
            torch.ones_like(y_predicts),
        )

        loss = fake_loss + real_loss

        return loss

    def x_adv_training_step(
            self,
            x_predicts,
            x_1_predicts,
        ):
        fake_loss = self.x_adv_criterion(
            x_1_predicts,
            torch.zeros_like(x_1_predicts),
        )
        real_loss = self.x_adv_criterion(
            x_predicts,
            torch.ones_like(x_predicts),
        )

        loss = fake_loss + real_loss

        return loss

    def training_step(
            self,
            batch: Tensor,
            batch_idx: int,
            optimizer_idx: int,
        ) -> Tensor:
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        if optimizer_idx % 4 == 0:
            y_1 = self.x2y_generator(x)
            y_1_predicts = self.y_discriminator(y_1)
            x_2 = self.y2x_generator(y_1)

            g_adv_loss = self.y_adv_criterion(
                input=y_1_predicts,
                target=torch.ones_like(y_1_predicts),
            )
            cyc_loss = self.x_l1_criterion(
                x_2,
                x,
            )
            loss = g_adv_loss + cyc_loss

            return loss

        elif optimizer_idx % 4 == 1:
            x_1 = self.y2x_generator(y)
            x_1_predicts = self.x_discriminator(x_1)
            y_2 = self.x2y_generator(x_1)

            g_adv_loss = self.x_adv_criterion(
                input=x_1_predicts,
                target=torch.ones_like(x_1_predicts),
            )
            cyc_loss = self.y_l1_criterion(
                y_2,
                y,
            )
            loss = g_adv_loss + cyc_loss

            return loss

        elif optimizer_idx % 4 == 2:
            y_1 = self.x2y_generator(x)
            y_predicts = self.y_discriminator(y)
            y_1_predicts = self.y_discriminator(y_1)

            fake_loss = self.y_adv_criterion(
                input=y_1_predicts,
                target=torch.zeros_like(y_1_predicts),
            )
            real_loss = self.y_adv_criterion(
                input=y_predicts,
                target=torch.ones_like(y_predicts),
            )
            loss = fake_loss + real_loss

            return loss

        elif optimizer_idx % 4 == 3:
            x_1 = self.y2x_generator(y)
            x_predicts = self.x_discriminator(x)
            x_1_predicts = self.x_discriminator(x_1)

            fake_loss = self.x_adv_criterion(
                input=x_1_predicts,
                target=torch.zeros_like(x_1_predicts),
            )
            real_loss = self.y_adv_criterion(
                input=x_predicts,
                target=torch.ones_like(x_predicts),
            )
            loss = fake_loss + real_loss

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
        loss = self.training_step(
            batch,
            batch_idx,
            optimizer_idx=1,
        )

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
        x2y_generator_optimizer = Adam(
            params=self.x2y_generator.parameters(),
            lr=self.learning_rate,
        )
        y2x_generator_optimizer = Adam(
            params=self.y2x_generator.parameters(),
            lr=self.learning_rate,
        )
        y_discriminator_optimizer = Adam(
            params=self.y_discriminator.parameters(),
            lr=self.learning_rate,
        )
        x_discriminator_optimizer = Adam(
            params=self.x_discriminator.parameters(),
            lr=self.learning_rate,
        )
        optimizers = [
            x2y_generator_optimizer,
            y2x_generator_optimizer,
            y_discriminator_optimizer,
            x_discriminator_optimizer,
        ]

        return optimizers, []


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    translator = CycleGANTranslator(
        device=device,
    ).to(device)

    print(translator)

