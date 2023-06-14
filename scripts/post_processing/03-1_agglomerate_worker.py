from funlib.persistence import graphs

import daisy
import logging
import lsd.post
import json
import sys
import pymongo
import time

from funlib.persistence import open_ds

logging.basicConfig(level=logging.INFO)


def agglomerate_worker(input_config):

    logging.info(sys.argv)

    with open(input_config, "r") as f:
        config = json.load(f)

    sample_name = config["sample_name"]
    affs_file = config["affs_file"]
    affs_dataset = config["affs_dataset"]
    fragments_file = config["fragments_file"]
    fragments_dataset = config["fragments_dataset"]
    db_host = config["db_host"]
    db_name = config["db_name"]
    merge_function = config["merge_function"]

    waterz_merge_function = {
        "hist_quant_10": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 10, ScoreValue, 256, false>>",
        "hist_quant_10_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 10, ScoreValue, 256, true>>",
        "hist_quant_25": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, false>>",
        "hist_quant_25_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, true>>",
        "hist_quant_50": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, false>>",
        "hist_quant_50_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, true>>",
        "hist_quant_75": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, false>>",
        "hist_quant_75_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, true>>",
        "hist_quant_90": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 90, ScoreValue, 256, false>>",
        "hist_quant_90_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 90, ScoreValue, 256, true>>",
        "mean": "OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>",
    }[merge_function]

    logging.info("Reading affs from %s" % affs_file)
    affs = open_ds(affs_file, affs_dataset, mode="r")
    fragments = open_ds(fragments_file, fragments_dataset, mode="r+")

    # open RAG DB
    logging.info("Opening RAG DB...")
    rag_provider = graphs.MongoDbGraphProvider(
        db_name,
        host=db_host,
        mode="r+",
        directed=False,
        nodes_collection=sample_name + "_nodes",
        edges_collection=sample_name + "_edges_" + merge_function,
        position_attribute=["center_z", "center_y", "center_x"],
    )
    logging.info("RAG DB opened")

    # open block done DB
    client = pymongo.MongoClient(db_host)
    db = client[db_name]
    completed_collection_name = f"{sample_name}_agglom_blocks_completed"
    completed_collection = db[completed_collection_name]

    client = daisy.Client()

    while True:

        with client.acquire_block() as block:

            if block is None:
                break

            start = time.time()

            lsd.post.agglomerate_in_block(
                affs,
                fragments,
                rag_provider,
                block,
                merge_function=waterz_merge_function,
                threshold=1.0,
                aff_min=-1,
            )

            document = {
                "num_cpus": 5,
                "block_id": block.block_id,
                "read_roi": (block.read_roi.get_begin(), block.read_roi.get_shape()),
                "write_roi": (block.write_roi.get_begin(), block.write_roi.get_shape()),
                "start": start,
                "duration": time.time() - start,
            }
            completed_collection.insert_one(document)


if __name__ == "__main__":

    agglomerate_worker(sys.argv[1])
