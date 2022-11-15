from dacapo.store.create_store import create_config_store
from dacapo.experiments.architectures import CNNectomeUNetConfig

from funlib.geometry import Coordinate

import click

import logging
from pathlib import Path
import yaml


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
def update():

    config_store = create_config_store()

    upsample_config = CNNectomeUNetConfig(
        name="upsample-unet",
        input_shape=Coordinate(216, 216, 216),
        eval_shape_increase=Coordinate(72, 72, 72),
        fmaps_in=1,
        num_fmaps=12,
        fmaps_out=72,
        fmap_inc_factor=6,
        downsample_factors=[(2, 2, 2), (3, 3, 3), (3, 3, 3)],
        constant_upsample=True,
    )

    architecture_configs = []
    for architecture_config in [
        upsample_config,
    ]:
        config_store.store_architecture_config(architecture_config)
        architecture_configs.append(architecture_config.name)

    output_path = Path("configs/zebrafish/architectures")
    if not output_path.exists():
        output_path.mkdir(parents=True)
    (output_path / "architectures.yaml").open("w").write(
        yaml.safe_dump(architecture_configs)
    )


if __name__ == "__main__":
    cli()
