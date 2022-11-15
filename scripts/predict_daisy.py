from dacapo.store.create_store import create_config_store
from dacapo.experiments import Run

import daisy

import click
import numpy as np

import subprocess
import sys
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


def spawn_worker(
    name,
    criterion,
    channels,
    out_container,
    out_dataset,
    in_container,
    in_dataset,
    billing,
    local=True,
    min_raw=0,
    max_raw=255,
):
    def run_worker():
        if local:
            subprocess.run(
                [
                    "python",
                    "scripts/predict_worker.py",
                    "start-worker",
                    "-n",
                    f"{name}",
                    "-c",
                    f"{criterion}",
                    "-cs",
                    f"{channels}",
                    "-oc",
                    f"{out_container}",
                    "-od",
                    f"{out_dataset}",
                    "-ic",
                    f"{in_container}",
                    "-id",
                    f"{in_dataset}",
                    "--min-raw",
                    f"{min_raw}",
                    "--max-raw",
                    f"{max_raw}",
                ]
            )
        else:
            subprocess.run(
                [
                    "bsub",
                    "-P",
                    billing,
                    "-J",
                    "pred",
                    "-q",
                    "gpu_rtx",
                    "-n",
                    "2",
                    "-gpu",
                    "num=1",
                    "-o",
                    f"prediction_logs/{name}.out",
                    "-e",
                    f"prediction_logs/{name}.err",
                    "python",
                    "scripts/predict_worker.py",
                    "start-worker",
                    "-n",
                    f"{name}",
                    "-c",
                    f"{criterion}",
                    "-cs",
                    f"{channels}",
                    "-oc",
                    f"{out_container}",
                    "-od",
                    f"{out_dataset}",
                    "-ic",
                    f"{in_container}",
                    "-id",
                    f"{in_dataset}",
                    "--min-raw",
                    f"{min_raw}",
                    "--max-raw",
                    f"{max_raw}",
                ]
            )

    return run_worker


@cli.command()
@click.option("-n", "--name", type=str)
@click.option("-c", "--criterion", type=str)
@click.option("-cs", "--channels", type=str)
@click.option("-oc", "--out_container", type=click.Path(file_okay=False))
@click.option("-od", "--out_dataset", type=str)
@click.option("-ic", "--in_container", type=click.Path(exists=True, file_okay=False))
@click.option("-id", "--in_dataset", type=str)
@click.option("-w", "--workers", type=int, default=1)
@click.option(
    "-roi",
    "--roi",
    type=str,
    required=False,
    help="The roi to predict on. Passed in as [lower:upper, lower:upper, ... ]",
)
@click.option("--local/--bsub", default=True)
@click.option("--billing", default=None)
@click.option("--min-raw", type=float, default=0)
@click.option("--max-raw", type=float, default=255)
def predict(
    name,
    criterion,
    channels,
    out_container,
    out_dataset,
    in_container,
    in_dataset,
    workers,
    roi,
    local,
    billing,
    min_raw,
    max_raw,
):
    if not local:
        assert billing is not None
    parsed_channels = [channel.split(":") for channel in channels.split(",")]

    raw = daisy.open_ds(in_container, in_dataset)

    config_store = create_config_store()
    run_config = config_store.retrieve_run_config(name)
    run = Run(run_config)
    model = run.model

    if roi is not None:
        parsed_start, parsed_end = zip(
            *[
                tuple(int(coord) for coord in axis.split(":"))
                for axis in roi.strip("[]").split(",")
            ]
        )
        parsed_roi = daisy.Roi(
            daisy.Coordinate(parsed_start),
            daisy.Coordinate(parsed_end) - daisy.Coordinate(parsed_start),
        )
    else:
        parsed_roi = raw.roi

    total_write_roi = raw.roi
    output_voxel_size = model.scale(raw.voxel_size)

    read_shape = model.eval_input_shape * raw.voxel_size
    write_shape = model.compute_output_shape(model.eval_input_shape)[1] * output_voxel_size
    context = (read_shape - write_shape) / 2
    read_roi = daisy.Roi((0,) * read_shape.dims, read_shape)
    write_roi = read_roi.grow(-context, -context)

    total_write_roi = parsed_roi
    total_read_roi = total_write_roi.grow(context, context)

    for indexes, channel in parsed_channels:
        num_channels = None if "-" not in indexes else len(indexes.split("-"))
        daisy.prepare_ds(
            out_container,
            f"{out_dataset}/{channel}",
            total_roi=total_write_roi,
            voxel_size=output_voxel_size,
            write_size=write_roi.shape,
            dtype=np.uint8,
            num_channels=num_channels,
        )

    task = daisy.Task(
        "test_server_task",
        total_roi=total_read_roi,
        read_roi=read_roi,
        write_roi=write_roi,
        process_function=spawn_worker(
            name,
            criterion,
            channels,
            out_container,
            out_dataset,
            in_container,
            in_dataset,
            billing,
            local,
            min_raw,
            max_raw,
        ),
        check_function=None,
        read_write_conflict=False,
        fit="overhang",
        num_workers=workers,
        max_retries=2,
        timeout=None,
    )

    daisy.run_blockwise([task])


if __name__ == "__main__":
    cli()
