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
def relabel(container, frags, lut, output):
    from funlib.segment.arrays import replace_values

    import zarr
    import numpy as np

    from pathlib import Path

    zarr_container = zarr.open(container)
    fragments = zarr_container[frags][:]

    mapping = np.load(Path(container, lut))["fragment_segment_lut"]
    segments = replace_values(fragments, mapping[0], mapping[1])

    zarr_container.create_dataset(output, data=segments, overwrite=True)


@cli.command()
@click.option("-c", "--container", type=click.Path(exists=True, file_okay=False))
@click.option("-d", "--dataset", type=str)
@click.option("--cmin", type=float, default=0)
@click.option("--cmax", type=float, default=1)
def size_filter(container, dataset, cmin, cmax):
    from funlib.segment.arrays import replace_values

    import zarr
    import numpy as np

    zarr_container = zarr.open(container)
    instance_array = zarr_container[dataset][:]
    objects, counts = np.unique(instance_array, return_counts=True)
    if 0 in objects:
        zero_ind = list(objects).index(0)
        objects = np.array([o for i, o in enumerate(objects) if i != zero_ind])
        counts = np.array([c for i, c in enumerate(counts) if i != zero_ind])

    mask =  np.invert((cmin < counts) * (cmax > counts))
    outlier_ids, outlier_counts = objects[mask], counts[mask]

    filtered_instance_array = replace_values(instance_array, outlier_ids, [0]*len(outlier_ids))    
        
    zarr_container.create_dataset(dataset + "_filtered", data=filtered_instance_array, overwrite=True)

    print(outlier_ids, outlier_counts, len(mask), sum(mask))


if __name__ == "__main__":
    cli()
