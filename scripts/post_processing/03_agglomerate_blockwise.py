import json
import hashlib
import logging
import os
import daisy
import time
import subprocess
import pymongo

from funlib.persistence import open_ds
from funlib.geometry import Coordinate

logging.basicConfig(level=logging.INFO)


def agglomerate(
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
    merge_function,
    drop=False,
    billing=None,
):
    """Run agglomeration in parallel blocks. Requires that affinities have been
        predicted before.

        Args:

            in_file (``string``):

                The input file containing affs and fragments.

            affs_dataset, fragments_dataset (``string``):

                Where to find the affinities and fragments.

            block_size (``tuple`` of ``int``):

                The size of one block in world units.

            context (``tuple`` of ``int``):

                The context to consider for fragment extraction and agglomeration,
                in world units.

            db_host (``string``):

                Where to find the MongoDB server.

            db_name (``string``):

                The name of the MongoDB database to use.
    open_ds(
            num_workers (``int``):

                How many blocks to run in parallel.

            merge_function (``string``):

                Symbolic name of a merge function. See dictionary below.
    """

    logging.info("Reading affs from %s", affs_file)
    affs = open_ds(affs_file, affs_dataset, mode="r")

    logging.info("Reading fragments from %s", fragments_file)
    fragments = open_ds(fragments_file, fragments_dataset, mode="r")

    client = pymongo.MongoClient(db_host)
    db = client[db_name]

    completed_collection_name = (
        f"{sample_name}_{merge_function}_agglom_blocks_completed"
    )
    completed_collection = None

    if completed_collection_name in db.list_collection_names():
        completed_collection = db[completed_collection_name]
        if drop:
            print(f"dropping {completed_collection}")
            db.drop_collection(completed_collection)
    if f"{sample_name}_edges" in db.list_collection_names():
        edges_collection = db[f"{sample_name}_edges"]
        if drop:
            print(f"dropping {edges_collection}")
            db.drop_collection(edges_collection)

    if completed_collection_name not in db.list_collection_names():
        completed_collection = db[completed_collection_name]
        completed_collection.create_index(
            [("block_id", pymongo.ASCENDING)], name="block_id"
        )

    context = daisy.Coordinate(context)

    # change back
    total_roi = fragments.roi.grow(context, context)
    #

    read_roi = daisy.Roi((0,) * affs.roi.dims, block_size).grow(context, context)
    write_roi = daisy.Roi((0,) * affs.roi.dims, block_size)

    skipped_blocks = []

    agglom_task = daisy.Task(
        f"{sample_name}_{merge_function}_agglomerate",
        total_roi,
        read_roi,
        write_roi,
        process_function=lambda: start_worker(
            sample_name,
            affs_file,
            affs_dataset,
            fragments_file,
            fragments_dataset,
            db_host,
            db_name,
            merge_function,
            billing,
        ),
        check_function=lambda b: check_block(completed_collection, b, skipped_blocks),
        num_workers=num_workers,
        read_write_conflict=False,
        fit="shrink",
    )

    daisy.run_blockwise([agglom_task])


def start_worker(
    sample_name: str,
    affs_file,
    affs_dataset,
    fragments_file,
    fragments_dataset,
    db_host,
    db_name,
    merge_function,
    billing,
):
    worker_id = daisy.Context.from_env()["worker_id"]
    task_id = daisy.Context.from_env()["task_id"]

    logging.info("worker %s started...", worker_id)

    output_basename = daisy.get_worker_log_basename(worker_id, task_id)

    log_out = output_basename.parent / f"worker_{worker_id}.out-bsub"
    log_err = output_basename.parent / f"worker_{worker_id}.err-bsub"

    try:
        os.makedirs(output_basename.parent)
    except:
        pass

    config = {
        "sample_name": sample_name,
        "affs_file": affs_file,
        "affs_dataset": affs_dataset,
        "fragments_file": fragments_file,
        "fragments_dataset": fragments_dataset,
        "db_host": db_host,
        "db_name": db_name,
        "merge_function": merge_function,
    }

    config_str = "".join(["%s" % (v,) for v in config.values()])
    config_hash = abs(int(hashlib.md5(config_str.encode()).hexdigest(), 16))

    config_file = os.path.join(output_basename.parent, "%d.config" % config_hash)

    with open(config_file, "w") as f:
        json.dump(config, f)

    logging.info("Running block with config %s..." % config_file)

    if merge_function == "mwatershed":
        worker = "./03-2_agglomerate_worker.py"
    else:
        worker = "./03-1_agglomerate_worker.py"

    command = f"python {worker} {config_file}"

    subprocess.run(
        ["bsub", "-I", "-P", billing, "-n", "1", "-o", log_out, "-e", log_err, command]
    )


def check_block(completed_collection, block, skipped_blocks):
    done = len(list(completed_collection.find({"block_id": block.block_id}))) >= 1

    return done


if __name__ == "__main__":
    voxel_size = Coordinate(1, 1, 1)
    block_size = Coordinate(256, 256, 256) * voxel_size
    context = Coordinate(16, 16, 16) * voxel_size
    start = time.time()

    agglomerate(
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
        merge_function="mwatershed",
        drop=False,
        billing="funke",
    )

    end = time.time()

    seconds = end - start
    minutes = seconds / 60
    hours = minutes / 60
    days = hours / 24

    print(
        "Total time to agglomerate: %f seconds / %f minutes / %f hours / %f days"
        % (seconds, minutes, hours, days)
    )
