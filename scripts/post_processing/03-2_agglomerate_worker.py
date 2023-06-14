from funlib.segment.arrays import relabel
from funlib.persistence import graphs

import daisy

import numpy as np
from scipy.ndimage import measurements

import logging
import json
import sys
import pymongo
import time
import itertools

from funlib.persistence import open_ds

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def agglomerate_in_block(affs, fragments, rag_provider, block: daisy.Block):
    logger.info(
        "Agglomerating in block %s with context of %s", block.write_roi, block.read_roi
    )

    # get the sub-{affs, fragments, graph} to work on
    affs = affs.intersect(block.read_roi)
    fragments = fragments.to_ndarray(affs.roi, fill_value=0)
    fragment_ids = np.array([x for x in np.unique(fragments) if x != 0])
    num_frags = len(fragment_ids)
    frag_mapping = {old: seq for seq, old in zip(range(1, num_frags + 1), fragment_ids)}
    rev_frag_mapping = {
        seq: old for seq, old in zip(range(1, num_frags + 1), fragment_ids)
    }
    for old, seq in frag_mapping.items():
        fragments[fragments == old] = seq
    rag = rag_provider[affs.roi]
    if len(fragment_ids) == 0:
        return

    logger.debug("affs shape: %s", affs.shape)
    logger.debug("fragments shape: %s", fragments.shape)
    # logger.debug("fragments num: %d", n)

    # convert affs to float32 ndarray with values between 0 and 1
    offsets = affs.data.attrs["offsets"]
    affs = affs.to_ndarray()
    if affs.dtype == np.uint8:
        affs = affs.astype(np.float32) / 255.0

    # separate affinities and offsets by range
    adjacents = [offset for offset in offsets if max(offset) <= 1]
    lr_offsets = offsets[len(adjacents) :]
    affs, lr_affs = (
        affs[: len(adjacents)],
        affs[len(adjacents) : len(adjacents) + len(lr_offsets)],
    )

    max_offset = [max(axis) for axis in zip(*adjacents)]
    base_fragments = np.expand_dims(
        fragments[tuple(slice(0, -m) for m in max_offset)], 0
    )
    base_affs = affs[(slice(None, None),) + tuple(slice(0, -m) for m in max_offset)]
    offset_frags = []
    for offset in adjacents:
        offset_frags.append(
            fragments[
                tuple(
                    slice(o, (-m + o) if m != o else None)
                    for o, m in zip(offset, max_offset)
                )
            ]
        )

    offset_frags = np.stack(offset_frags, axis=0)
    mask = offset_frags != base_fragments

    # cantor pairing function
    mismatched_labels = (
        (offset_frags + base_fragments) * (offset_frags + base_fragments + 1) // 2
        + base_fragments
    ) * mask
    mismatched_ids = np.array([x for x in np.unique(mismatched_labels) if x != 0])
    adjacent_score = measurements.median(
        base_affs,
        mismatched_labels,
        mismatched_ids,
    )
    adjacent_count = measurements.sum(mask, mismatched_labels, mismatched_ids)
    total_count = measurements.sum(mask, offset_frags, range(1, num_frags + 1))
    total_count = {
        frag: count for frag, count in zip(range(1, num_frags + 1), total_count)
    }
    adjacent_map = {
        seq_id: (float(med_score) - 0.5) * float(adj_count)
        for seq_id, med_score, adj_count in zip(
            mismatched_ids, adjacent_score, adjacent_count
        )
    }

    max_lr_offset = [max(axis) for axis in zip(*lr_offsets)]
    base_lr_fragments = fragments[tuple(slice(0, -m) for m in max_lr_offset)]
    base_lr_affs = lr_affs[
        (slice(None, None),) + tuple(slice(0, -m) for m in max_lr_offset)
    ]
    lr_offset_frags = []
    for offset in lr_offsets:
        lr_offset_frags.append(
            fragments[
                tuple(
                    slice(o, (-m + o) if m != o else None)
                    for o, m in zip(offset, max_lr_offset)
                )
            ]
        )
    lr_offset_frags = np.stack(lr_offset_frags, axis=0)
    lr_mask = lr_offset_frags != base_lr_fragments
    # cantor pairing function
    lr_mismatched_labels = (
        (lr_offset_frags + base_lr_fragments)
        * (lr_offset_frags + base_lr_fragments + 1)
        // 2
        + base_lr_fragments
    ) * lr_mask
    lr_mismatched_ids = np.array([x for x in np.unique(lr_mismatched_labels) if x != 0])
    lr_adjacent_score = measurements.median(
        base_lr_affs,
        lr_mismatched_labels,
        lr_mismatched_ids,
    )
    # lr_adjacent_count = measurements.sum(
    #     lr_mask, lr_mismatched_labels, lr_mismatched_ids
    # )
    # lr_total_count = measurements.sum(lr_mask, lr_offset_frags, range(1, num_frags+1))
    # lr_total_count = {frag: count for frag, count in zip(range(1, num_frags+1), lr_total_count)}
    lr_adjacent_map = {
        seq_id: float(med_score)
        for seq_id, med_score in zip(lr_mismatched_ids, lr_adjacent_score)
    }

    for seq_id_u, seq_id_v in itertools.combinations(range(1, num_frags + 1), 2):
        cantor_id_u = (
            (seq_id_u + seq_id_v) * (seq_id_u + seq_id_v + 1)
        ) // 2 + seq_id_u
        cantor_id_v = (
            (seq_id_u + seq_id_v) * (seq_id_u + seq_id_v + 1)
        ) // 2 + seq_id_v
        if (
            cantor_id_u in adjacent_map
            or cantor_id_v in adjacent_map
            or cantor_id_u in lr_adjacent_map
            or cantor_id_v in lr_adjacent_map
        ):
            adj_weight_u = adjacent_map.get(cantor_id_u, None)
            adj_weight_v = adjacent_map.get(cantor_id_v, None)
            if adj_weight_u is not None and adj_weight_v is not None:
                adj_weight = (
                    (adj_weight_v + adj_weight_u)
                    / float(total_count[seq_id_u] + total_count[seq_id_v])
                    * 4
                )
                adj_weight += 0.5
            elif adj_weight_u is not None:
                adj_weight = (
                    adj_weight_u
                    / float(total_count[seq_id_u] + total_count[seq_id_v])
                    * 4
                )
                adj_weight += 0.5
            elif adj_weight_v is not None:
                adj_weight = (
                    adj_weight_v
                    / float(total_count[seq_id_u] + total_count[seq_id_v])
                    * 4
                )
                adj_weight += 0.5
            else:
                adj_weight = None
            lr_weight_u = lr_adjacent_map.get(cantor_id_u, None)
            lr_weight_v = lr_adjacent_map.get(cantor_id_v, None)
            if lr_weight_u is None and lr_weight_v is None:
                lr_weight = None
            elif lr_weight_u is None:
                lr_weight = lr_weight_v
            elif lr_weight_v is None:
                lr_weight = lr_weight_u
            else:
                lr_weight = (lr_weight_u + lr_weight_v) / 2
            rag.add_edge(
                rev_frag_mapping[seq_id_u],
                rev_frag_mapping[seq_id_v],
                adj_weight=adj_weight,
                lr_weight=lr_weight,
            )

    # write back results (only within write_roi)
    logger.info(f"writing {len(rag.edges)} edges to DB...")
    logger.info(
        f"num frags: {len(fragment_ids)}, num_adj: {len(adjacent_map)}, num_lr_adj: {len(lr_adjacent_map)}"
    )
    rag.write_edges(block.write_roi)


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

    # waterz_merge_function = {
    #     "hist_quant_10": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 10, ScoreValue, 256, false>>",
    #     "hist_quant_10_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 10, ScoreValue, 256, true>>",
    #     "hist_quant_25": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, false>>",
    #     "hist_quant_25_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 25, ScoreValue, 256, true>>",
    #     "hist_quant_50": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, false>>",
    #     "hist_quant_50_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, true>>",
    #     "hist_quant_75": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, false>>",
    #     "hist_quant_75_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 75, ScoreValue, 256, true>>",
    #     "hist_quant_90": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 90, ScoreValue, 256, false>>",
    #     "hist_quant_90_initmax": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 90, ScoreValue, 256, true>>",
    #     "mean": "OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>",
    # }[merge_function]

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
    completed_collection_name = (
        f"{sample_name}_{merge_function}_agglom_blocks_completed"
    )
    completed_collection = db[completed_collection_name]

    client = daisy.Client()

    while True:
        with client.acquire_block() as block:
            if block is None:
                break

            start = time.time()

            agglomerate_in_block(affs, fragments, rag_provider, block)

            document = {
                "num_cpus": 1,
                "block_id": block.block_id,
                "read_roi": (block.read_roi.get_begin(), block.read_roi.get_shape()),
                "write_roi": (block.write_roi.get_begin(), block.write_roi.get_shape()),
                "start": start,
                "duration": time.time() - start,
            }
            completed_collection.insert_one(document)


if __name__ == "__main__":
    agglomerate_worker(sys.argv[1])
