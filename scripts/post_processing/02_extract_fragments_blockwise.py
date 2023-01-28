import json
import hashlib
import logging
import numpy as np
import os
import daisy
import pymongo
import time
import subprocess

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
    affs = daisy.open_ds(affs_file, affs_dataset, mode="r")

    client = pymongo.MongoClient(db_host)
    db = client[db_name]

    completed_collection_name = f"{sample_name}_fragment_blocks_extracted"
    completed_collection = None

    if completed_collection_name in db.list_collection_names():
        completed_collection = db[completed_collection_name]
        if drop:
            print(f"dropping {completed_collection}")
            db.drop_collection(completed_collection)

    if completed_collection_name not in db.list_collection_names():
        completed_collection = db[completed_collection_name]
        completed_collection.create_index(
            [("block_id", pymongo.ASCENDING)], name="block_id"
        )

    # prepare fragments dataset
    fragments = daisy.prepare_ds(
        fragments_file,
        fragments_dataset,
        affs.roi,
        # total_roi,
        affs.voxel_size,
        np.uint64,
        daisy.Roi((0, 0, 0), block_size),
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
            billing,
        ),
        check_function=lambda b: check_block(completed_collection, b),
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
        ["bsub", "-P", "funke", "-n", "1", "-o", log_out, "-e", log_err, command]
    )
    # subprocess.run(command.split() + [">", log_out, "&>", log_err])


def check_block(completed_collection, block):

    done = len(list(completed_collection.find({"block_id": block.block_id}))) >= 1

    return done


if __name__ == "__main__":
    # samples = [16, 17, 23]
    samples = ["2022-12-15/16"]
    samples = ["2023-01-17/23_bot", "2023-01-17/23_mid1", "2023-01-17/23_top"]
    for sample in samples:
        start = time.time()

        extract_fragments(
            sample_name=f"{sample}",
            affs_file="/nrs/funke/pattonw/predictions/zebrafish/zebrafish.n5",
            affs_dataset=f"predictions/{sample}/0",
            fragments_file="/nrs/funke/pattonw/predictions/zebrafish/zebrafish.n5",
            fragments_dataset=f"processed/{sample}/cells/fragments",
            block_size=(256, 256, 256),
            context=(16, 16, 16),
            db_host="mongodb://microdosingAdmin:Cu2CO3OH2@funke-mongodb2.int.janelia.org:27017",
            db_name="zebrafish_postprocessing",
            num_workers=50,
            fragments_in_xy=False,
            epsilon_agglomerate=0.1,
            mask_file=None,
            mask_dataset=None,
            filter_fragments=0.01,
            replace_sections=None,
            drop=False,
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
