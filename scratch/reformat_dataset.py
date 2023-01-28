import yaml
import zarr
import numpy as np

from pathlib import Path
import itertools

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

in_container = zarr.open(constants["input_container"])
out_container = zarr.open(constants["dataset_container"])


def copy_data(crop, index, organelle):

    raw = in_container[constants["raw_dataset"].format(crop=crop, index=index)][:]
    out_container[constants["raw_dataset"].format(crop=crop, index=index)] = raw.astype(np.uint8)

    try:
        in_data = in_container[
            constants["gt_dataset"].format(crop=crop, index=index, organelle=organelle)
        ][:]
    except KeyError as e:
        try:
            in_data = in_container[
                constants["gt_dataset"].format(crop=crop, index=index, organelle="all")
            ][:]
        except KeyError as e:
            # assume organelle not present in this crop
            in_data = np.zeros(raw.shape)

    for bad_id, new_id in (
        constants["data_problems"]
        .get(crop, {})
        .get(index, {})
        .get(organelle, {})
        .get("id_mapping", [])
    ):
        in_data[in_data == bad_id] = new_id

    mask_id = (
        constants["data_problems"]
        .get(crop, {})
        .get(index, {})
        .get(organelle, {})
        .get("mask_id", None)
    )
    if mask_id is not None:
        in_mask = in_data != (
            constants["data_problems"]
            .get(crop, {})
            .get(index, {})
            .get(organelle, {})
            .get("mask_id", -1)
        )
    else:
        in_mask = np.ones_like(in_data)
    in_data[in_mask == 0] = 0

    out_container[
        constants["gt_dataset"].format(crop=crop, index=index, organelle=organelle)
    ] = in_data.astype(np.uint64)
    out_container[
        constants["mask_dataset"].format(crop=crop, index=index, organelle=organelle)
    ] = in_mask.astype(np.uint64)


for organelle in targets:
    for crop, index in itertools.chain(datasets["train"], datasets["validate"]):
        copy_data(crop, index, organelle)
