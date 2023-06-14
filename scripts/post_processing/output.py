import click

import logging


@click.group()
@click.option(
    "--log-level",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    default="INFO",
)
def cli(log_level):
    logging.basicConfig(level=getattr(logging, log_level.upper()))


@cli.command()
@click.option("-c", "--container", type=click.Path(exists=True, file_okay=False))
@click.option("-f", "--frags", type=str)
@click.option("-l", "--lut", type=str)
@click.option("-o", "--output", type=str)
@click.option("-n", "--num-threads", type=int, default=1)
def relabel(container, frags, lut, output, num_threads):
    from funlib.segment.arrays import replace_values
    from funlib.persistence import open_ds, prepare_ds
    from funlib.geometry import Roi
    import daisy

    import zarr
    import numpy as np

    from pathlib import Path

    frag_ds = open_ds(container, frags, mode="r")
    segs_ds = prepare_ds(
        container, output, frag_ds.roi, frag_ds.voxel_size, dtype=np.uint64
    )
    mapping = np.load(Path(container, lut))["fragment_segment_lut"].astype(np.uint64)
    print(mapping)

    def process_block(block):
        fragments = frag_ds.to_ndarray(block.read_roi)
        if fragments.max() == 0:
            return
        segments = replace_values(fragments.astype(np.uint64), mapping[0], mapping[1])
        segs_ds[block.write_roi] = segments

    task = daisy.Task(
        "relabel_blockwise",
        frag_ds.roi,
        Roi((0,) * frag_ds.roi.dims, segs_ds.data.chunks) * frag_ds.voxel_size,
        Roi((0,) * frag_ds.roi.dims, segs_ds.data.chunks) * frag_ds.voxel_size,
        process_block,
        check_function=None,
        init_callback_fn=None,
        read_write_conflict=False,
        num_workers=num_threads,
        max_retries=2,
        fit="valid",
        timeout=None,
        upstream_tasks=None,
    )

    daisy.run_blockwise([task])


@cli.command()
@click.option("-c", "--container", type=click.Path(exists=True, file_okay=False))
@click.option("-d", "--dataset", type=str)
@click.option("-oc", "--output-container", type=str)
@click.option("-od", "--output-dataset", type=str)
@click.option("--cmin", type=float, default=0)
@click.option("--cmax", type=float, default=1)
@click.option("-n", "--num-threads", type=int, default=1)
@click.option("-ctx", "--context", type=int, default=48)
def size_filter(
    container,
    dataset,
    output_container,
    output_dataset,
    cmin,
    cmax,
    num_threads,
    context,
):
    from funlib.segment.arrays import replace_values
    from funlib.persistence import open_ds, prepare_ds, Array
    from funlib.geometry import Roi, Coordinate
    import daisy

    import zarr
    import numpy as np

    frag_ds = open_ds(container, dataset, mode="r")
    filtered_ds = prepare_ds(
        output_container,
        output_dataset,
        frag_ds.roi,
        frag_ds.voxel_size,
        dtype=np.uint64,
    )

    def process_block(block):
        instance_array = Array(
            frag_ds.to_ndarray(block.read_roi, fill_value=0),
            block.read_roi,
            frag_ds.voxel_size,
        )
        objects, counts = np.unique(instance_array.data, return_counts=True)
        if 0 in objects:
            zero_ind = list(objects).index(0)
            objects = np.array([o for i, o in enumerate(objects) if i != zero_ind])
            counts = np.array([c for i, c in enumerate(counts) if i != zero_ind])

        mask = np.invert((cmin < counts) * (cmax > counts))
        outlier_ids, outlier_counts = objects[mask], counts[mask]

        instance_array.data = replace_values(
            instance_array.data.astype(np.uint64),
            outlier_ids.astype(np.uint64),
            [0] * len(outlier_ids),
        )

        filtered_ds[block.write_roi] = instance_array.to_ndarray(block.write_roi)

    task = daisy.Task(
        "filter_blockwise",
        frag_ds.roi.grow(context, context),
        Roi((0,) * frag_ds.roi.dims, Coordinate(filtered_ds.data.chunks) + context * 2)
        * frag_ds.voxel_size,
        Roi((context,) * frag_ds.roi.dims, filtered_ds.data.chunks)
        * frag_ds.voxel_size,
        process_block,
        check_function=None,
        init_callback_fn=None,
        read_write_conflict=False,
        num_workers=num_threads,
        max_retries=2,
        fit="valid",
        timeout=None,
        upstream_tasks=None,
    )

    daisy.run_blockwise([task])


if __name__ == "__main__":
    cli()
