import json
import hashlib
import logging
import numpy as np
import os
import daisy
import pymongo
import time
import subprocess

from funlib.persistence import open_ds, prepare_ds
from funlib.geometry import Coordinate

logging.basicConfig(level=logging.INFO)


def extract_fragments(
    sample_name: str,
    affs_file,
    affs_dataset,
    fragments_file,
    fragments_dataset,
    block_size,
    context,
    db_host,
    db_name,
    num_workers,
    fragments_in_xy,
    epsilon_agglomerate=0,
    mask_file=None,
    mask_dataset=None,
    filter_fragments=0,
    replace_sections=None,
    drop=False,
    shrink_objects=0,
    billing=None,
):
    """Run agglomeration in parallel blocks. Requires that affinities have been
    predicted before.

    Args:

        affs_file,
        affs_dataset,

        block_size (``tuple`` of ``int``):

            The size of one block in world units.

        context (``tuple`` of ``int``):

            The context to consider for fragment extraction and agglomeration,
            in world units.

        db_host (``string``):

            Where to find the MongoDB server.

        db_name (``string``):

            The name of the MongoDB database to use.

        num_workers (``int``):

            How many blocks to run in parallel.
    """

    logging.info("Reading affs from %s", affs_file)
    affs = open_ds(affs_file, affs_dataset, mode="r")

    client = pymongo.MongoClient(db_host)
    db = client[db_name]

    completed_collection_name = f"{sample_name}_fragment_blocks_extracted"
    completed_collection = None

    if completed_collection_name in db.list_collection_names():
        completed_collection = db[completed_collection_name]
        if drop:
            print(f"dropping {completed_collection}")
            db.drop_collection(completed_collection)
    if f"{sample_name}_nodes" in db.list_collection_names():
        nodes_collection = db[f"{sample_name}_nodes"]
        if drop:
            print(f"dropping {nodes_collection}")
            db.drop_collection(nodes_collection)
    if f"{sample_name}_meta" in db.list_collection_names():
        meta_collection = db[f"{sample_name}_meta"]
        if drop:
            print(f"dropping {meta_collection}")
            db.drop_collection(meta_collection)
    for collection_name in db.list_collection_names():
        if collection_name.startswith(f"{sample_name}_edges"):
            edges_collection = db[collection_name]
            if drop:
                print(f"dropping {edges_collection}")
                db.drop_collection(edges_collection)

    if completed_collection_name not in db.list_collection_names():
        completed_collection = db[completed_collection_name]
        completed_collection.create_index(
            [("block_id", pymongo.ASCENDING)], name="block_id"
        )

    complete_cache = set(
        [tuple(doc["block_id"]) for doc in completed_collection.find()]
    )

    # prepare fragments dataset
    fragments = prepare_ds(
        fragments_file,
        fragments_dataset,
        affs.roi,
        # total_roi,
        affs.voxel_size,
        np.uint64,
        daisy.Roi((0, 0, 0), block_size),
        delete=drop,
    )

    context = daisy.Coordinate(context)
    total_roi = affs.roi.grow(context, context)
    # total_roi = total_roi.grow(context, context)
    read_roi = daisy.Roi((0,) * affs.roi.dims, block_size).grow(context, context)
    write_roi = daisy.Roi((0,) * affs.roi.dims, block_size)

    num_voxels_in_block = (write_roi / affs.voxel_size).size

    extract_fragments_task = daisy.Task(
        f"{sample_name}_extract_fragments",
        total_roi=total_roi,
        read_roi=read_roi,
        write_roi=write_roi,
        process_function=lambda: start_worker(
            sample_name,
            affs_file,
            affs_dataset,
            fragments_file,
            fragments_dataset,
            db_host,
            db_name,
            context,
            fragments_in_xy,
            epsilon_agglomerate,
            mask_file,
            mask_dataset,
            filter_fragments,
            num_voxels_in_block,
            replace_sections,
            shrink_objects,
            billing,
        ),
        check_function=lambda b: check_block(completed_collection, complete_cache, b),
        num_workers=num_workers,
        read_write_conflict=False,
        fit="shrink",
    )

    daisy.run_blockwise([extract_fragments_task])


def start_worker(
    sample_name: str,
    affs_file,
    affs_dataset,
    fragments_file,
    fragments_dataset,
    db_host,
    db_name,
    context,
    fragments_in_xy,
    epsilon_agglomerate,
    mask_file,
    mask_dataset,
    filter_fragments,
    num_voxels_in_block,
    replace_sections,
    shrink_objects,
    billing,
):
    worker_id = daisy.Context.from_env()["worker_id"]
    task_id = daisy.Context.from_env()["task_id"]

    logging.info("worker %s started...", worker_id)

    logging.info("epsilon_agglomerate: %s", epsilon_agglomerate)
    logging.info("mask_file: %s", mask_file)
    logging.info("mask_dataset: %s", mask_dataset)
    logging.info("filter_fragments: %s", filter_fragments)

    output_basename = daisy.get_worker_log_basename(worker_id, task_id)

    log_out = output_basename.parent / f"worker_{worker_id}.out-bsub"
    log_err = output_basename.parent / f"worker_{worker_id}.err-bsub"

    config = {
        "sample_name": sample_name,
        "affs_file": affs_file,
        "affs_dataset": affs_dataset,
        "fragments_file": fragments_file,
        "fragments_dataset": fragments_dataset,
        "db_host": db_host,
        "db_name": db_name,
        "context": context,
        "fragments_in_xy": fragments_in_xy,
        "epsilon_agglomerate": epsilon_agglomerate,
        "mask_file": mask_file,
        "mask_dataset": mask_dataset,
        "filter_fragments": filter_fragments,
        "num_voxels_in_block": num_voxels_in_block,
        "replace_sections": replace_sections,
        "shrink_objects": shrink_objects,
    }

    config_str = "".join(["%s" % (v,) for v in config.values()])
    config_hash = abs(int(hashlib.md5(config_str.encode()).hexdigest(), 16))

    config_file = os.path.join(output_basename.parent, "%d.config" % config_hash)

    with open(config_file, "w") as f:
        json.dump(config, f)

    logging.info("Running block with config %s..." % config_file)

    worker = "./02-1_extract_fragments_worker.py"

    command = f"python {worker} {config_file}"

    subprocess.run(
        ["bsub", "-I", "-P", billing, "-n", "2", "-o", log_out, "-e", log_err, command]
    )


def check_block(completed_collection, complete_cache, block):
    done = (
        block.block_id in complete_cache
        or len(list(completed_collection.find({"block_id": block.block_id}))) >= 1
    )

    return done


if __name__ == "__main__":
    voxel_size = Coordinate(1, 1, 1)
    block_size = Coordinate(256, 256, 256) * voxel_size
    context = Coordinate(16, 16, 16) * voxel_size
    start = time.time()

    extract_fragments(
        sample_name=f"s17-stitched",
        affs_file="/nrs/funke/pattonw/predictions/zebrafish/zebrafish.n5",
        affs_dataset=f"predictions/2023-05-09/s17/cells_finetuned_3d_lsdaffs_zebrafish_cells_upsample-unet_default_v3__1__60000",
        fragments_file="/nrs/funke/pattonw/predictions/zebrafish/zebrafish.n5",
        fragments_dataset="predictions/2023-05-09/s17/cells_finetuned_3d_lsdaffs_zebrafish_cells_upsample-unet_default_v3__1__60000_fragments",
        block_size=tuple(block_size),
        context=tuple(context),
        db_host="mongodb://microdosingAdmin:Cu2CO3OH2@funke-mongodb2.int.janelia.org:27017",
        db_name="dacapo_zebrafish",
        num_workers=64,
        fragments_in_xy=False,
        epsilon_agglomerate=0.1,
        mask_file=None,
        mask_dataset=None,
        filter_fragments=0.5,
        replace_sections=None,
        drop=True,
        shrink_objects=1,
        billing="funke",
    )

    end = time.time()

    seconds = end - start
    minutes = seconds / 60
    hours = minutes / 60
    days = hours / 24

    print(
        "Total time to extract fragments: %f seconds / %f minutes / %f hours / %f days"
        % (seconds, minutes, hours, days)
    )
