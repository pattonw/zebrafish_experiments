import daisy
import logging
import time
import os
import numpy as np
from funlib.segment.graphs.impl import connected_components
from funlib.persistence import open_ds, graphs

import mwatershed as mws

logging.basicConfig(level=logging.INFO)
logging.getLogger("funlib.persistence.graphs.shared_graph_provider").setLevel(
    logging.DEBUG
)


def find_segments(
    db_host,
    db_name,
    fragments_file,
    fragments_dataset,
    edges_collection,
    thresholds_minmax,
    thresholds_step,
    sample_name,
):
    """

    Args:

        db_host (``string``):

            Where to find the MongoDB server.

        db_name (``string``):

            The name of the MongoDB database to use.

        fragments_file (``string``):

            Path to the file containing the fragments.

        edges_collection (``string``):

            The name of the MongoDB database collection to use.

        thresholds_minmax (``list`` of ``int``):

            The lower and upper bound to use (i.e [0,1]) when generating
            thresholds.

        thresholds_step (``float``):

            The step size to use when generating thresholds between min/max.

        roi_offset (array-like of ``int``):

            The starting point (inclusive) of the ROI. Entries can be ``None``
            to indicate unboundedness.

        roi_shape (array-like of ``int``):

            The shape of the ROI. Entries can be ``None`` to indicate
            unboundedness.

    """

    scale = False

    print("Reading graph from DB ", db_name, edges_collection)
    start = time.time()

    graph_provider = graphs.MongoDbGraphProvider(
        db_name,
        db_host,
        nodes_collection=f"{sample_name}_nodes",
        meta_collection=f"{sample_name}_meta",
        edges_collection=edges_collection,
        position_attribute=["center_z", "center_y", "center_x"],
    )

    print("Got Graph provider")

    fragments = open_ds(fragments_file, fragments_dataset)

    print("Opened fragments")

    roi = fragments.roi
    # roi = daisy.Roi(roi.offset, (20000, 10000, 10000))

    print("Getting graph for roi %s" % roi)

    graph = graph_provider.get_graph(roi)

    print("Read graph in %.3fs" % (time.time() - start))

    if graph.number_of_nodes == 0:
        print("No nodes found in roi %s" % roi)
        return

    nodes = np.array(graph.nodes)
    edges = np.stack(list(graph.edges), axis=0)
    adj_scores = np.array([graph.edges[tuple(e)]["adj_weight"] for e in edges]).astype(
        np.float32
    )
    lr_scores = np.array([graph.edges[tuple(e)]["lr_weight"] for e in edges]).astype(
        np.float32
    )

    print("Complete RAG contains %d nodes, %d edges" % (len(nodes), len(edges)))

    out_dir = os.path.join(fragments_file, "luts_full")

    os.makedirs(out_dir, exist_ok=True)

    start = time.time()

    segment(
        nodes,
        edges,
        adj_scores,
        lr_scores,
        edges_collection,
        out_dir,
    )

    print("Created and stored lookup tables in %.3fs" % (time.time() - start))


def segment(nodes, edges, adj_scores, lr_scores, edges_collection, out_dir):
    # drop out mask will be added to scores. anything with a 1 added will always be above threshold
    # dropout_mask = np.random.randn(len(scores)) / 500
    # dropout_mask = dropout_mask.astype(np.float32)
    # scores += dropout_mask

    adj_bias = 0.0
    lr_bias = -1.0
    edges = [
        (adj + adj_bias, u, v)
        for adj, (u, v) in zip(adj_scores, edges)
        if not np.isnan(adj) and adj is not None
    ] + [
        (lr_adj + lr_bias, u, v)
        for lr_adj, (u, v) in zip(lr_scores, edges)
        if not np.isnan(lr_adj) and lr_adj is not None
    ]
    edges = sorted(
        edges,
        key=lambda edge: abs(edge[0]),
        reverse=True,
    )
    edges = [(bool(aff > 0), u, v) for aff, u, v in edges]
    lut = mws.cluster(edges)
    inputs, outputs = zip(*lut)

    start = time.time()
    print("%.3fs" % (time.time() - start))

    start = time.time()
    lut = np.array([inputs, outputs])

    print("%.3fs" % (time.time() - start))

    lookup = "seg_%s" % (edges_collection)
    lookup = lookup.replace("/", "-")

    out_file = os.path.join(out_dir, lookup)

    np.savez_compressed(out_file, fragment_segment_lut=lut, edges=edges)

    print("%.3fs" % (time.time() - start))


if __name__ == "__main__":
    start = time.time()
    find_segments(
        db_host="mongodb://microdosingAdmin:Cu2CO3OH2@funke-mongodb2.int.janelia.org:27017",
        db_name="dacapo_zebrafish",
        fragments_file="/nrs/funke/pattonw/predictions/zebrafish/zebrafish.n5",
        fragments_dataset="predictions/2023-05-09/s17/cells_finetuned_3d_lsdaffs_zebrafish_cells_upsample-unet_default_v3__1__60000_fragments",
        edges_collection=f"s17-stitched_edges_mwatershed",
        thresholds_minmax=[0.0, 1.0],
        thresholds_step=0.05,
        sample_name=f"s17-stitched",
    )
    print("Took %.3f seconds to find segments and store LUTs" % (time.time() - start))
