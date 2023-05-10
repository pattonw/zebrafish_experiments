import yaml
import zarr
import numpy as np

from pathlib import Path
import itertools
import csv

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

# id annotations
id_annotations_yaml = yaml_root_dir / "id_annotations.yaml"
id_annotations = yaml.safe_load(id_annotations_yaml.open("r").read())

in_container = zarr.open(constants["input_container"])
out_container = zarr.open(constants["dataset_container"])


def copy_data(crop, index, organelle):

    raw = in_container[constants["raw_dataset"].format(sample=f"{crop}-{index}")][:]
    out_container[
        constants["raw_dataset"].format(sample=f"{crop}-{index}")
    ] = raw.astype(np.uint8)

    try:
        in_data = in_container[
            constants["gt_dataset"].format(
                sample=f"{crop}-{index}", organelle=organelle
            )
        ][:]
    except KeyError as e:
        try:
            in_data = in_container[
                constants["gt_dataset"].format(
                    sample=f"{crop}-{index}", organelle="all"
                )
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
        constants["gt_dataset"].format(sample=f"{crop}-{index}", organelle=organelle)
    ] = in_data.astype(np.uint64)
    out_container[
        constants["mask_dataset"].format(sample=f"{crop}-{index}", organelle=organelle)
    ] = in_mask.astype(np.uint64)


def update_masks(crop, index, targets):
    # any annotated organelle can be masked in for the other organelle training volumes
    annotated_organelles = (
        sum(
            [
                out_container[
                    constants["gt_dataset"].format(
                        sample=f"{crop}-{index}", organelle=organelle
                    )
                ][:]
                for organelle in targets
            ]
        )
        > 0
    )
    for organelle in targets:
        mask_dataset = out_container[
            constants["mask_dataset"].format(
                sample=f"{crop}-{index}", organelle=organelle
            )
        ]
        mask_dataset[:] = (mask_dataset + annotated_organelles) > 0


def relabel(organelle: str, sample: str, annotation_type: str):
    print(organelle, sample, annotation_type)
    if annotation_type == "good":
        pass
    elif annotation_type == "negative":
        return
    elif annotation_type == "segments":
        return
    else:
        return

    with open(
        constants["id_annotations_path"].format(
            organelle=organelle, sample=sample, annotation_type=annotation_type
        ),
        newline="",
    ) as f:
        reader = csv.reader(f)
        rows = list(reader)

    if annotation_type == "good":
        row = rows[0]
        good_ids = [int(x) for x in row]
    elif annotation_type == "negative":
        row = rows[0]
        bad_ids = [int(x) for x in row]
    elif annotation_type == "segments":
        segments = [[int(y) for y in x if len(y) > 0] for x in zip(*rows)]

    fragments = in_container[
        constants["gt_dataset"].format(sample=sample, organelle=organelle)
    ][:]
    try:
        in_gt = out_container[
            constants["gt_dataset"].format(sample=sample, organelle=organelle)
        ][:]
    except KeyError:
        in_gt = np.zeros_like(fragments)
    try:
        in_mask = out_container[
            constants["mask_dataset"].format(sample=sample, organelle=organelle)
        ][:]
    except KeyError:
        in_mask = np.zeros_like(fragments)

    if annotation_type == "good":
        row = rows[0]
        good_ids = [int(x) for x in row]
        mask = np.isin(fragments, good_ids)
        in_gt[mask] = fragments[mask]
        in_mask = np.stack([in_mask, mask]).max(axis=0)
    elif annotation_type == "negative":
        bad_ids = [int(x) for x in row]
        mask = np.isin(fragments, bad_ids)
        in_mask = np.stack([in_mask, mask]).max(axis=0)
    elif annotation_type == "segments":
        for segment in segments:
            segment_id = min(segment)
            mask = np.isin(fragments, segment)
            in_gt[mask] = segment_id
            in_mask = np.stack([in_mask, mask]).max(axis=0)
    else:
        return

    raw = in_container[constants["raw_dataset"].format(sample=sample)][:]
    out_container[constants["raw_dataset"].format(sample=sample)] = raw.astype(np.uint8)

    print(f"Mask sum: {in_mask.sum()}")
    out_container[
        constants["gt_dataset"].format(sample=sample, organelle=organelle)
    ] = in_gt.astype(np.uint64)
    out_container[
        constants["mask_dataset"].format(sample=sample, organelle=organelle)
    ] = in_mask.astype(np.uint64)


def merge_masks(organelle_a: str, organelle_b: str, sample: str):

    try:
        in_gt_a = (
            out_container[
                constants["gt_dataset"].format(sample=sample, organelle=organelle_a)
            ][:]
            > 0
        )
        in_mask_a = out_container[
            constants["mask_dataset"].format(sample=sample, organelle=organelle_a)
        ][:]
    except KeyError as e:
        in_gt_a = None
        in_mask_a = None
    try:
        in_gt_b = (
            out_container[
                constants["gt_dataset"].format(sample=sample, organelle=organelle_b)
            ][:]
            > 0
        )
        in_mask_b = out_container[
            constants["mask_dataset"].format(sample=sample, organelle=organelle_b)
        ][:]
    except KeyError as e:
        in_gt_b = None
        in_mask_b = None

    a_masks = ([in_mask_a] if in_mask_a is not None else []) + (
        [in_gt_b] if in_gt_b is not None else []
    )
    b_masks = ([in_mask_b] if in_mask_b is not None else []) + (
        [in_gt_a] if in_gt_a is not None else []
    )
    in_mask_a = np.stack(a_masks).max(axis=0)
    in_mask_b = np.stack(b_masks).max(axis=0)

    out_container[
        constants["mask_dataset"].format(sample=sample, organelle=organelle_a)
    ] = in_mask_a.astype(np.uint64)
    out_container[
        constants["mask_dataset"].format(sample=sample, organelle=organelle_b)
    ] = in_mask_b.astype(np.uint64)


def generate_points(sample: str, organelle: str):

    in_mask = out_container[
        constants["mask_dataset"].format(sample=sample, organelle=organelle)
    ][:]

    (z, y, x) = np.where(in_mask)

    np.savez(
        constants["dataset_container"]
        + "/"
        + constants["points_dataset"].format(sample=sample, organelle=organelle),
        [z, y, x],
    )


RELABEL = True
MERGE_MASKS = False
COPY_DATA = False
UPDATE_MASKS = False
GENERATE_POINTS = True

condition = lambda sample, organelle, annotation_type: (
    organelle == "cells" and sample == "23_mid1"
)

if RELABEL:
    for organelle, samples in id_annotations.items():
        for sample, annotation_types in samples.items():
            for annotation_type in annotation_types:

                if not condition(sample, organelle, annotation_type):
                    continue
                relabel(organelle, sample, annotation_type)

if MERGE_MASKS:
    for organelle_a, organelle_b in itertools.combinations(id_annotations.keys(), 2):
        organelle_a_samples = set(id_annotations[organelle_a].keys())
        organelle_b_samples = set(id_annotations[organelle_b].keys())
        for sample in organelle_a_samples.union(organelle_b_samples):

            if not (condition(sample, organelle_a, None) or condition(sample, organelle_b, None)):
                continue
            merge_masks(organelle_a, organelle_b, sample)

if COPY_DATA:
    for organelle in targets:
        for crop, index in itertools.chain(datasets["train"], datasets["validate"]):
            if not condition(None, organelle, None):
                continue
            copy_data(crop, index, organelle)

if UPDATE_MASKS:
    for crop, index in itertools.chain(datasets["train"], datasets["validate"]):
        if not condition(None, None, None):
            continue
        update_masks(crop, index, targets)

if GENERATE_POINTS:
    for sample, organelle in itertools.product(
        [
            "16_bot",
            "23_bot",
            "23_mid1",
        ],
        [
            "axons",
            "cells",
            "vessel",
        ],
    ):
        if not condition(sample, organelle, None):
            continue
        generate_points(sample, organelle)
