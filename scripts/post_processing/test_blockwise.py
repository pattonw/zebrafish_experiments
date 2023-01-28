import json
import hashlib
import logging
import numpy as np
import os
import daisy
import pymongo
import time
import subprocess
import sys

logging.basicConfig(level=logging.INFO)


def test_task(
    affs_file,
    affs_dataset,
    block_size,
    context,
    db_host,
    db_name,
    num_workers,
    drop,
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

    completed_collection_name = f"16_test_blocks_completed"
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

    assert (
        completed_collection_name in db.list_collection_names()
        and completed_collection is not None
    )

    context = daisy.Coordinate(context)
    total_roi = affs.roi.grow(context, context)
    read_roi = daisy.Roi((0,) * affs.roi.dims, block_size).grow(context, context)
    write_roi = daisy.Roi((0,) * affs.roi.dims, block_size)

    extract_fragments_task = daisy.Task(
        f"16_test",
        total_roi=total_roi,
        read_roi=read_roi,
        write_roi=write_roi,
        process_function=lambda: start_worker(),
        check_function=lambda b: check_block(completed_collection, b),
        num_workers=num_workers,
        read_write_conflict=False,
        fit="shrink",
    )

    daisy.run_blockwise([extract_fragments_task])


def start_worker():

    worker_id = daisy.Context.from_env()["worker_id"]
    task_id = daisy.Context.from_env()["task_id"]

    logging.info("worker %s started...", worker_id)
    output_basename = daisy.get_worker_log_basename(worker_id, task_id)

    log_out = output_basename.parent / f"worker_{worker_id}.out-bsub"
    log_err = output_basename.parent / f"worker_{worker_id}.err-bsub"

    worker = "./test_worker.py"

    command = f"python {worker}"

    subprocess.run(
        ["bsub", "-P", "funke", "-n", "1", "-o", log_out, "-e", log_err, command]
    )


def check_block(completed_collection, block):

    done = len(list(completed_collection.find({"block_id": block.block_id}))) >= 1

    return done


if __name__ == "__main__":

    drop = bool(int(sys.argv[1]))

    start = time.time()

    test_task(
        affs_file="/nrs/funke/pattonw/predictions/zebrafish/zebrafish.n5",
        affs_dataset=f"predictions/2022-12-15/16/0",
        block_size=(256, 256, 256),
        context=(16, 16, 16),
        db_host="mongodb://microdosingAdmin:Cu2CO3OH2@funke-mongodb2.int.janelia.org:27017",
        db_name="zebrafish_postprocessing",
        num_workers=10,
        drop=drop,
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
