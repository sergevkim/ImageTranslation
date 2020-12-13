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
            generator_blocks_num: int = 8,
            discriminator_blocks_num: int = 4,
        ):
        super().__init__()

        self.device = device
        self.learning_rate = learning_rate
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma
        self.verbose = verbose

        self.l1_criterion = MSELoss()
        self.adv_criterion = BCELoss()

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

    def y_generator_training_step(
            self,
            ground_truths: Tensor,
            gen_outputs: Tensor,
            fake_predicts: Tensor,
            lambda_coef: float = 100,
        ) -> Tensor: #TODO ALL
        g_fake_loss = self.adv_criterion(
            input=fake_predicts,
            target=torch.ones_like(fake_predicts),
        )
        g_l1_loss = self.l1_criterion(gen_outputs, ground_truths)
        g_loss = g_fake_loss + lambda_coef * g_l1_loss

        return g_loss

    def y_discriminator_training_step(
            self,
            fake_predicts: Tensor,
            real_predicts: Tensor,
        ) -> Tensor: #TODO ALL
        d_fake_loss = self.adv_criterion(
            input=fake_predicts,
            target=torch.zeros_like(fake_predicts),
        )
        d_real_loss = self.adv_criterion(
            input=real_predicts,
            target=torch.ones_like(real_predicts),
        )
        d_loss = d_fake_loss + d_real_loss

        return d_loss

    def training_step(
            self,
            batch: Tensor,
            batch_idx: int,
        ) -> Tensor:
        x_original, y_original = batch
        x_original = x_original.to(self.device)

        y_generated = self.x2y_generator(x_original)
        fake_y_predicts = self.y_discriminator(y_generated)
        real_y_predicts = self.y_discriminator(y_original)

        x_generated = self.y2x_generator(y_generated)
        fake_x_predicts = self.x_discriminator(x_generated)
        real_x_predicts = self.x_discriminator(x_original)

        y_generator_loss = self.y_generator_training_step()
        y_discriminator_loss = self.y_discriminator_training_step()
        x_generator_loss = self.x_generator_training_step()
        x_discriminator_loss = self.x_discriminator_training_step()

        loss = (
            y_generator_loss
            + y_discriminator_loss
            + x_generator_loss
            + x_discriminator_loss
        )

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
    translator = Pix2PixTranslator(
        device=device,
    ).to(device)

    print(translator)

