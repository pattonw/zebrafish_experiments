from dacapo.store.create_store import create_config_store
from dacapo.store.config_store import DuplicateNameError
from dacapo.experiments.datasplits import TrainValidateDataSplitConfig
from dacapo.experiments.datasplits.datasets import RawGTDatasetConfig
from dacapo.experiments.datasplits.datasets.arrays import (
    ZarrArrayConfig,
    IntensitiesArrayConfig,
    OnesArrayConfig,
)

from funlib.geometry import Coordinate

import numpy as np
import click

import logging
from pathlib import Path
import yaml
import random


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
    # Load yamls
    yaml_root_dir = Path("configs/yamls/zebrafish")
    assert yaml_root_dir.exists(), f"{yaml_root_dir} does not exist!"

    # constants
    constants_yaml = yaml_root_dir / "constants.yaml"
    constants = yaml.safe_load(constants_yaml.open("r").read())

    # targets. What classes we need to preprocess the data into
    target_yaml = yaml_root_dir / "targets.yaml"
    targets = yaml.safe_load(target_yaml.open("r").read())

    # dataset yamls. What crops do we have per dataset, and where is the raw data
    dataset_yaml = yaml_root_dir / "datasets.yaml"
    datasets = yaml.safe_load(dataset_yaml.open("r").read())

    def create_dataset(sample, organelle):
        raw_config = IntensitiesArrayConfig(
            f"zebrafish_raw_{sample}",
            ZarrArrayConfig(
                f"zebrafish_raw_{sample}_uint8",
                constants["dataset_container"],
                constants["raw_dataset"].format(sample=sample),
            ),
            0,
            255,
        )

        mask_config = ZarrArrayConfig(
            f"zebrafish_mask_{sample}_{organelle}",
            constants["dataset_container"],
            constants["mask_dataset"].format(sample=sample, organelle=organelle),
        )
        if Path(
            constants["dataset_container"],
            constants["gt_dataset"].format(sample=sample, organelle=organelle),
        ).exists():
            gt_config = ZarrArrayConfig(
                f"zebrafish_gt_{sample}_{organelle}",
                constants["dataset_container"],
                constants["gt_dataset"].format(sample=sample, organelle=organelle),
            )
        else:
            gt_config = OnesArrayConfig(
                f"zebrafish_gt_full_{sample}_{organelle}", mask_config
            )
            gt_config = IntensitiesArrayConfig(
                f"zebrafish_gt_empty_{sample}_{organelle}", gt_config, 1, 2
            )
        points_path = Path(
            constants["dataset_container"],
            constants["points_dataset"].format(sample=sample, organelle=organelle)
            + ".npz",
        )
        if points_path.exists():
            points = np.load(points_path)["arr_0"]
            points = set(Coordinate(*coords) for coords in zip(*points))
            offsets = [
                Coordinate(1, 0, 0),
                Coordinate(-1, 0, 0),
                Coordinate(0, 1, 0),
                Coordinate(0, -1, 0),
                Coordinate(0, 0, 1),
                Coordinate(0, 0, -1),
            ]
            decimated_points = []
            for point in points:
                if (
                    sum([point + offset in points for offset in offsets])
                    < len(offsets) - 1
                ):
                    decimated_points.append(Coordinate(*tuple(point)[::-1]))
            print(sample, organelle, len(points), len(decimated_points))
            points = decimated_points
            points = random.choices(points, k=2000)
        else:
            points = None
        return RawGTDatasetConfig(
            f"zebrafish_{sample}_{organelle}",
            raw_config=raw_config,
            gt_config=gt_config,
            mask_config=mask_config,
            sample_points=points,
            weight=5 if "-" in sample else 1,
        )

    config_store = create_config_store()
    datasplit_configs = []
    for organelle in targets:
        datasplit_config = TrainValidateDataSplitConfig(
            f"zebrafish_{organelle}",
            train_configs=[
                create_dataset(sample, organelle) for sample in datasets["train"]
            ],
            validate_configs=[
                create_dataset(sample, organelle) for sample in datasets["validate"]
            ],
        )
        datasplit_configs.append(datasplit_config)

    datasplit_config_names = []
    for datasplit_config in datasplit_configs:
        try:
            config_store.store_datasplit_config(datasplit_config)
        except DuplicateNameError as e:
            if force:
                config_store.datasplits.delete_one({"name": datasplit_config.name})
                config_store.store_datasplit_config(datasplit_config)
            else:
                raise e

        datasplit_config_names.append(datasplit_config.name)

    output_path = Path("configs/zebrafish/datasplits")
    if not output_path.exists():
        output_path.mkdir(parents=True)
    (output_path / "datasplits.yaml").open("w").write(
        yaml.safe_dump(datasplit_config_names)
    )


if __name__ == "__main__":
    cli()
