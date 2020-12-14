from argparse import ArgumentParser

from trima.datamodules import CityscapesDataModule
from trima.loggers import NeptuneLogger
from trima.models import (
    CycleGANTranslator,
    Pix2PixTranslator,
)
from trima.trainer import Trainer
from trima.utils import Randomer

from config import (
    CommonArguments,
    DataArguments,
    TrainArguments,
    SpecificArguments,
)


def main(args):
    Randomer.set_seed(seed=args.seed)

    model = CycleGANTranslator(
        learning_rate=args.learning_rate,
        scheduler_gamma=args.scheduler_gamma,
        scheduler_step_size=args.scheduler_step_size,
        generator_blocks_num=args.generator_blocks_num,
        discriminator_blocks_num=args.discriminator_blocks_num,
        verbose=args.verbose,
        device=args.device,
    ).to(args.device)
    datamodule = CityscapesDataModule(
        data_path=args.data_path,
        batch_size=args.batch_size if not args.one_batch_overfit else 2,
        num_workers=args.num_workers,
    )
    datamodule.setup(
        val_ratio=args.val_ratio,
        one_batch_overfit=args.one_batch_overfit,
    )

    #logger = NeptuneLogger(
    #    api_key=None,
    #    project_name=None,
    #)
    logger = None
    trainer = Trainer(
        logger=logger,
        max_epoch=args.max_epoch,
        verbose=args.verbose,
        version=args.version,
    )

    trainer.fit(
        model=model,
        datamodule=datamodule,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    default_args_dict = {
        **vars(CommonArguments()),
        **vars(DataArguments()),
        **vars(TrainArguments()),
        **vars(SpecificArguments()),
    }

    for arg, value in default_args_dict.items():
        parser.add_argument(
            f'--{arg}',
            type=type(value),
            default=value,
            help=f'<{arg}>, default: {value}',
        )

    args = parser.parse_args()

    main(args)

