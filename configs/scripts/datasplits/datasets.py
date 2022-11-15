from dacapo.store.create_store import create_config_store
from dacapo.store.config_store import DuplicateNameError
from dacapo.experiments.datasplits import TrainValidateDataSplitConfig
from dacapo.experiments.datasplits.datasets import RawGTDatasetConfig
from dacapo.experiments.datasplits.datasets.arrays import (
    ZarrArrayConfig,
    IntensitiesArrayConfig,
)

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

    def create_dataset(crop, index, organelle):
        raw_config = IntensitiesArrayConfig(
            f"zebrafish_raw_{crop}_{index}",
            ZarrArrayConfig(
                f"zebrafish_raw_{crop}_{index}_uint8",
                constants["dataset_container"],
                constants["raw_dataset"].format(crop=crop, index=index),
            ),
            0,
            255,
        )
        gt_config = ZarrArrayConfig(
            f"zebrafish_gt_{crop}_{index}_{organelle}",
            constants["dataset_container"],
            constants["gt_dataset"].format(crop=crop, index=index, organelle=organelle),
        )
        mask_config = ZarrArrayConfig(
            f"zebrafish_mask_{crop}_{index}_{organelle}",
            constants["dataset_container"],
            constants["mask_dataset"].format(
                crop=crop, index=index, organelle=organelle
            ),
        )
        return RawGTDatasetConfig(
            f"zebrafish_{crop}_{index}_{organelle}", raw_config, gt_config, mask_config
        )

    config_store = create_config_store()
    datasplit_configs = []
    for organelle in targets:
        datasplit_config = TrainValidateDataSplitConfig(
            f"zebrafish_{organelle}",
            train_configs=[
                create_dataset(crop, index, organelle)
                for crop, index in datasets["train"]
            ],
            validate_configs=[
                create_dataset(crop, index, organelle)
                for crop, index in datasets["validate"]
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
