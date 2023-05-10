import dacapo
from dacapo.store.create_store import create_config_store, create_weights_store
from dacapo.experiments import Run

import daisy

import numpy as np
import torch
import click

import sys
import logging

logger = logging.getLogger(__file__)


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
@click.option("-n", "--name", type=str)
@click.option("-c", "--criterion", type=str)
@click.option("-oc", "--out_container", type=click.Path(exists=True, file_okay=False))
@click.option("-od", "--out_dataset", type=str)
@click.option("-ic", "--in_container", type=click.Path(exists=True, file_okay=False))
@click.option("-id", "--in_dataset", type=str)
@click.option("--min-raw", type=float, default=0)
@click.option("--max-raw", type=float, default=255)
def start_worker(
    name,
    criterion,
    out_container,
    out_dataset,
    in_container,
    in_dataset,
    min_raw,
    max_raw,
):
    shift = min_raw
    scale = max_raw - min_raw

    device = torch.device("cuda")

    client = daisy.Client()

    config_store = create_config_store()
    weights_store = create_weights_store()

    run_config = config_store.retrieve_run_config(name)
    run = Run(run_config)

    model = run.model
    try:
        weights_store._load_best(run, criterion)
    except FileNotFoundError:
        iteration = int(criterion)
        weights = weights_store.retrieve_weights(run, iteration)
        model.load_state_dict(weights.model)

    model = run.model.to(device)

    raw_dataset = daisy.open_ds(in_container, in_dataset)

    voxel_size = raw_dataset.voxel_size
    output_voxel_size = model.scale(voxel_size)

    out_dataset = daisy.open_ds(
        out_container,
        f"{out_dataset}",
        mode="r+",
    )

    with torch.no_grad():
        while True:
            with client.acquire_block() as block:
                if block is None:
                    break
                raw_input = (
                    raw_dataset.to_ndarray(
                        roi=block.read_roi, fill_value=shift + scale
                    ).astype(np.float32)
                    - shift
                ) / scale
                raw_input = np.expand_dims(raw_input, (0, 1))
                write_roi = block.write_roi.intersect(out_dataset.roi)

                predictions = daisy.Array(
                    model.forward(torch.from_numpy(raw_input).float().to(device))
                    .detach()
                    .cpu()
                    .numpy()[0],
                    block.write_roi,
                    output_voxel_size,
                )

                write_data = predictions.to_ndarray(write_roi)
                out_dataset[write_roi] = (255 / (1 + np.exp(-write_data))).astype(np.uint8)

                block.status = daisy.BlockStatus.SUCCESS


if __name__ == "__main__":
    cli()
