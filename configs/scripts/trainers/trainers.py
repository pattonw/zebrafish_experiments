from dacapo.store.create_store import create_config_store
from dacapo.store.config_store import DuplicateNameError
from dacapo.experiments.trainers import GunpowderTrainerConfig
from dacapo.experiments.trainers.gp_augments import (
    SimpleAugmentConfig,
    ElasticAugmentConfig,
    IntensityAugmentConfig,
    IntensityScaleShiftAugmentConfig,
    GammaAugmentConfig,
)

import click

import logging
from pathlib import Path
import yaml
import math


@click.group()
@click.option(
    "--log-level",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    default="INFO",
)
def cli(log_level):
    logging.basicConfig(level=getattr(logging, log_level.upper()))


@cli.command()
@click.option("--force/--no-force", default=False)
def update(force):

    config_store = create_config_store()

    gunpowder_config = GunpowderTrainerConfig(
        name="default",
        batch_size=2,
        learning_rate=0.0001,
        augments=[
            SimpleAugmentConfig(),
            ElasticAugmentConfig(
                control_point_spacing=(100, 100, 100),
                control_point_displacement_sigma=(10.0, 10.0, 10.0),
                rotation_interval=(0, math.pi / 2.0),
                subsample=8,
                uniform_3d_rotation=True,
            ),
            IntensityAugmentConfig(
                scale=(0.85, 1.15),
                shift=(-0.1, 0.1),
                clip=True,
            ),
        ],
        num_data_fetchers=20,
        snapshot_interval=10000,
        min_masked=0.01,
    )
    gunpowder_test_config = GunpowderTrainerConfig(
        name="test-trainer",
        batch_size=1,
        learning_rate=0.0001,
        augments=[
            SimpleAugmentConfig(),
            ElasticAugmentConfig(
                control_point_spacing=(100, 100, 100),
                control_point_displacement_sigma=(10.0, 10.0, 10.0),
                rotation_interval=(0, math.pi / 2.0),
                subsample=8,
                uniform_3d_rotation=True,
            ),
            IntensityAugmentConfig(
                scale=(0.25, 1.75),
                shift=(-0.5, 0.35),
                clip=True,
            ),
        ],
        num_data_fetchers=1,
        snapshot_interval=1,
        min_masked=0.15,
    )

    gunpowder_config_v3 = GunpowderTrainerConfig(
        name="default_v4",
        batch_size=2,
        learning_rate=0.0001,
        augments=[
            ElasticAugmentConfig(
                control_point_spacing=(20, 20, 20),
                control_point_displacement_sigma=(5.0, 5.0, 5.0),
                rotation_interval=(0, math.pi / 2.0),
                subsample=8,
                uniform_3d_rotation=True,
            ),
            IntensityAugmentConfig(
                scale=(0.7, 1.3),
                shift=(-0.2, 0.2),
                clip=True,
            ),
        ],
        num_data_fetchers=20,
        snapshot_interval=500,
    )

    trainer_config_names = []
    for trainer_config in [gunpowder_config, gunpowder_config_v3, gunpowder_test_config]:
        try:
            config_store.store_trainer_config(trainer_config)
        except DuplicateNameError as e:
            if force:
                config_store.trainers.delete_one({"name": trainer_config.name})
                config_store.store_trainer_config(trainer_config)
            else:
                raise e

        trainer_config_names.append(trainer_config.name)

    output_path = Path("configs/zebrafish/trainers")
    if not output_path.exists():
        output_path.mkdir(parents=True)
    (output_path / "trainers.yaml").open("w").write(
        yaml.safe_dump(trainer_config_names)
    )


if __name__ == "__main__":
    cli()
