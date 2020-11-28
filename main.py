from argparse import ArgumentParser
from pathlib import Path

from trima.datamodules import CityscapesDataModule
from trima.loggers import NeptuneLogger
from trima.models import CycleGANTranslator, Pix2PixTranslator
from trima.trainer import Trainer

from config import Arguments


def main(args):
    model = Pix2PixTranslator(
        learning_rate=args.learning_rate
        scheduler_gamma=args.scheduler_gamma,
        scheduler_step_size=args.scheduler_step_size,
        verbose=args.verbose,
    )
    datamodule = CityscapesDataModule(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    datamodule.setup(val_ratio=args.val_ratio)

    logger = NeptuneLogger(
        api_key=None,
        project_name=None,
    )
    trainer = Trainer(
        logger=logger,
        verbose=args.verbose,
        version=args.version,
    )

    trainer.fit(
        model=model,
        datamodule=datamodule,
    )


if __name__ == "__main__":
    #parser = ArgumentParser()
    #args = parser.parse_args()
    args = Arguments()

    main(args)

