import daisy
import logging
import json
import sys
import pymongo
import time

from funlib.segment.arrays import relabel, replace_values
from funlib.geometry import Coordinate
from funlib.persistence import open_ds, Array, graphs
from scipy.ndimage import measurements
import daisy
import logging
import numpy as np
import waterz

import mahotas
import numpy as np
import logging
import waterz
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.filters import gaussian_filter, maximum_filter

import mwatershed as mws

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def watershed_in_block(
    affs,
    block,
    context,
    rag_provider,
    fragments_out,
    num_voxels_in_block,
    filter_fragments=0,
    mask=None,
    epsilon_agglomerate=0,
):
    """
    Args:
        filter_fragments (float):
            Filter fragments that have an average affinity lower than this
            value.
        min_seed_distance (int):
            Controls distance between seeds in the initial watershed. Reducing
            this value improves downsampled segmentation.
    """

    total_roi = affs.roi

    if fragments_out.to_ndarray(block.write_roi).max() > 0:
        return

    logger.info("reading affs from %s", block.read_roi)

    offsets = affs.data.attrs["offsets"]
    affs = affs.intersect(block.read_roi)
    affs.materialize()
    affs.data = affs.data[: len(offsets)]

    if affs.dtype == np.uint8:
        logger.info("Assuming affinities are in [0,255]")
        max_affinity_value = 255.0
        affs.data = affs.data.astype(np.float64)
    else:
        max_affinity_value = 1.0

    if affs.data.max() < 1e-3:
        return

    affs.data /= max_affinity_value

    if mask is not None:
        raise NotImplementedError("mask not yet supported")

        logger.info("reading mask from %s", block.read_roi)
        mask_data = get_mask_data_in_roi(mask, affs.roi, affs.voxel_size)
        logger.info("masking affinities")
        affs.data *= mask_data

    # extract fragments
    shift = np.array(
        [-0.4 if max(offset) <= 1 else -0.7 for offset in offsets]
    ).reshape((-1, *((1,) * (len(affs.data.shape) - 1))))
    fragments_data = mws.agglom(
        affs.data
        + shift
        + (gaussian_filter(affs.data, sigma=(0, *(Coordinate(context) / 3))) - 0.5)
        * 0.01
        + np.random.randn(*affs.data.shape) * 0.001,
        offsets=offsets,
    )

    logger.info(
        str(
            (
                fragments_data.min(),
                fragments_data.max(),
                affs.data.min(),
                affs.data.max(),
            )
        )
    )

    if mask is not None:
        raise NotImplementedError("mask not yet supported")
        fragments_data *= mask_data.astype(np.uint64)

    if filter_fragments > 0:
        average_affs = np.mean(affs.data, axis=0)

        filtered_fragments = []

        fragment_ids = np.unique(fragments_data)

        for fragment, mean in zip(
            fragment_ids, measurements.mean(average_affs, fragments_data, fragment_ids)
        ):
            if mean < filter_fragments:
                filtered_fragments.append(fragment)

        filtered_fragments = np.array(filtered_fragments, dtype=fragments_data.dtype)
        replace = np.zeros_like(filtered_fragments)
        replace_values(fragments_data, filtered_fragments, replace, inplace=True)

    if epsilon_agglomerate > 0:
        logger.info(
            "Performing initial fragment agglomeration until %f", epsilon_agglomerate
        )

        generator = waterz.agglomerate(
            affs=affs.data,
            thresholds=[epsilon_agglomerate],
            fragments=fragments_data,
            scoring_function="OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, false>>",
            discretize_queue=256,
            return_merge_history=False,
            return_region_graph=False,
        )
        fragments_data[:] = next(generator)

        # cleanup generator
        for _ in generator:
            pass

    # # todo add key value replacement option

    fragments = Array(fragments_data, affs.roi, affs.voxel_size)

    # crop fragments to write_roi
    fragments = fragments[block.write_roi]
    fragments.materialize()
    max_id = fragments.data.max()

    # ensure we don't have IDs larger than the number of voxels (that would
    # break uniqueness of IDs below)
    if max_id > num_voxels_in_block:
        logger.warning(
            "fragments in %s have max ID %d, relabelling...", block.write_roi, max_id
        )
        fragments.data, max_id = relabel(fragments.data)

        assert max_id < num_voxels_in_block

    # ensure unique IDs
    id_bump = block.block_id[1] * num_voxels_in_block
    logger.info("bumping fragment IDs by %i", id_bump)
    fragments.data[fragments.data > 0] += id_bump
    fragment_ids = np.array([x for x in np.unique(fragments.data) if x != 0])

    # store fragments
    logger.info(f"writing {len(fragment_ids)} fragments to {block.write_roi}")
    fragments_out[block.write_roi] = fragments

    # following only makes a difference if fragments were found
    if max_id == 0:
        return

    # get fragment centers
    fragment_centers = {
        fragment: block.write_roi.get_offset() + affs.voxel_size * Coordinate(center)
        for fragment, center in zip(
            fragment_ids,
            measurements.center_of_mass(fragments.data, fragments.data, fragment_ids),
        )
        if not np.isnan(center[0])
    }

    # store nodes
    rag = rag_provider[block.write_roi]
    rag.add_nodes_from(
        [
            (node, {"center_z": c[0], "center_y": c[1], "center_x": c[2]})
            for node, c in fragment_centers.items()
        ]
    )
    rag.write_nodes(block.write_roi)


def extract_fragments_worker(input_config):
    logger.info(sys.argv)

    with open(input_config, "r") as f:
        config = json.load(f)

    logger.info(config)

    sample_name = config["sample_name"]
    affs_file = config["affs_file"]
    affs_dataset = config["affs_dataset"]
    fragments_file = config["fragments_file"]
    fragments_dataset = config["fragments_dataset"]
    db_name = config["db_name"]
    db_host = config["db_host"]
    context = config["context"]
    num_voxels_in_block = config["num_voxels_in_block"]
    fragments_in_xy = config["fragments_in_xy"]
    epsilon_agglomerate = config["epsilon_agglomerate"]
    filter_fragments = config["filter_fragments"]
    replace_sections = config["replace_sections"]
    shrink_objects = config["shrink_objects"]

    logger.info("Reading affs from %s", affs_file)
    affs = open_ds(affs_file, affs_dataset, mode="r")

    logger.info("writing fragments to %s", fragments_file)
    fragments = open_ds(fragments_file, fragments_dataset, mode="r+")

    if config["mask_file"] != None:
        logger.info("Reading mask from %s", config["mask_file"])
        mask = open_ds(config["mask_file"], config["mask_dataset"], mode="r")

    else:
        mask = None

    # open RAG DB
    logger.info("Opening RAG DB...")
    rag_provider = graphs.MongoDbGraphProvider(
        db_name,
        host=db_host,
        mode="r+",
        directed=False,
        position_attribute=["center_z", "center_y", "center_x"],
        edges_collection=f"{sample_name}_edges",
        nodes_collection=f"{sample_name}_nodes",
        meta_collection=f"{sample_name}_meta",
    )
    logger.info("RAG DB opened")

    # open block done DB
    mongo_client = pymongo.MongoClient(db_host)
    db = mongo_client[db_name]
    blocks_extracted = db[f"{sample_name}_fragment_blocks_extracted"]

    client = daisy.Client()

    while True:
        logger.info("getting block")
        with client.acquire_block() as block:
            logger.info(f"got block {block}")

            if block is None:
                break

            start = time.time()

            logger.info("block read roi begin: %s", block.read_roi.get_begin())
            logger.info("block read roi shape: %s", block.read_roi.get_shape())
            logger.info("block write roi begin: %s", block.write_roi.get_begin())
            logger.info("block write roi shape: %s", block.write_roi.get_shape())

            watershed_in_block(
                affs,
                block,
                context,
                rag_provider,
                fragments,
                num_voxels_in_block=num_voxels_in_block,
                mask=mask,
                epsilon_agglomerate=epsilon_agglomerate,
                filter_fragments=filter_fragments,
            )

            document = {
                "num_cpus": 5,
                "block_id": block.block_id,
                "read_roi": (block.read_roi.get_begin(), block.read_roi.get_shape()),
                "write_roi": (block.write_roi.get_begin(), block.write_roi.get_shape()),
                "start": start,
                "duration": time.time() - start,
            }
            blocks_extracted.insert_one(document)
            logger.info(f"releasing block: {block}")


if __name__ == "__main__":
    extract_fragments_worker(sys.argv[1])
