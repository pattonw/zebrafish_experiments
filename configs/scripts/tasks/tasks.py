from dacapo.store.create_store import create_config_store
from dacapo.experiments.tasks import AffinitiesTaskConfig

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

    affinities = AffinitiesTaskConfig(
        name=f"3d_lsdaffs",
        neighborhood=[
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (3, 0, 0),
            (0, 3, 0),
            (0, 0, 3),
            (9, 0, 0),
            (0, 9, 0),
            (0, 0, 9),
        ],
        lsds=True,
    )

    task_configs = []
    for task_config in [
        affinities,
    ]:
        config_store.store_task_config(task_config)
        task_configs.append(task_config.name)

    output_path = Path("configs/zebrafish/tasks")
    if not output_path.exists():
        output_path.mkdir(parents=True)
    (output_path / "tasks.yaml").open("w").write(
        yaml.safe_dump(task_configs)
    )


if __name__ == "__main__":
    cli()
