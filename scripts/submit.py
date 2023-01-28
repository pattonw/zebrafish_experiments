from dacapo.store.create_store import create_config_store
from dacapo.experiments.run_config import RunConfig

from funlib.geometry import Coordinate

import click

import logging
from pathlib import Path
import yaml
import subprocess
from pprint import pprint
import itertools


constants = yaml.safe_load(
    Path("configs/yamls/zebrafish/constants.yaml").open("r").read()
)


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
@click.option("-r", "--run", type=click.Path(exists=True, dir_okay=False))
@click.option("-b", "--billing", type=str)
@click.option("-q", "--queue", type=str)
@click.option("-n", "--num-cpus", type=int)
def run(run, billing, queue, num_cpus):
    run_file = Path(run)
    group_name = run_file.parent.name
    run_names = yaml.safe_load(Path(run).open("r").read())
    print(run_names)
    for run_name in run_names:
        log_dir = Path(f"runs/{run_name}/logs")
        log_dir.mkdir(exist_ok=True, parents=True)

        command = [
            "bsub",
            "-P",
            billing,
            "-We",
            "72:00",
            "-J",
            group_name,
            "-q",
            queue,
            "-n",
            f"{num_cpus}",
            "-gpu",
            "num=1",
            "-o",
            f"runs/{run_name}/logs/1.out",
            "-e",
            f"runs/{run_name}/logs/1.err",
            "dacapo",
            "train",
            "-r",
            run_name,
        ]

        subprocess.run(command)


@cli.command()
@click.option("-p", "--prediction", type=click.Path(exists=True, dir_okay=False))
@click.option("-w", "--workers", type=int, default=30)
def predict(prediction, workers):
    prediction_data = yaml.safe_load(Path(prediction).open("r").read())

    name = prediction_data["name"]
    jobs = []

    # defaults
    default_datasets = prediction_data.get("datasets")
    billing = prediction_data["billing"]

    for matrix in prediction_data["predictions"]:
        offset, shape = matrix.get("roi", (None, None))
        start = Coordinate(offset) if offset is not None else None
        end = start + Coordinate(shape) if start is not None else None

        outputs, datasets, setups, criteria = (
            matrix["outputs"],
            matrix.get("datasets"),
            matrix["setups"],
            matrix["criteria"],
        )
        assert (datasets is None) != (
            default_datasets is None
        ), f"Default vs matrix datasets. Provide 1: {default_datasets}, {datasets}!"

        for dataset, setup, criterion in itertools.product(
            datasets if datasets is not None else default_datasets, setups, criteria
        ):

            if start is not None:
                roi = ",".join(list(f"{s}:{e}" for s, e in zip(start, end)))

            dataset_name = dataset["name"]
            dataset_array = dataset["array"]
            dataset_container = dataset["container"]


            command = (
                [
                    "python",
                    "scripts/predict_daisy.py",
                    "predict",
                    "-n",
                    setup["name"],
                    "-c",
                    criterion,
                    "-oc",
                    constants["prediction_container"],
                    "-od",
                    f"predictions/{name}/{dataset_name}",
                    "-ic",
                    dataset_container,
                    "-id",
                    dataset_array,
                ]
                + (
                    [
                        "--bsub",
                        "--billing",
                        billing,
                        "-w",
                        f"{workers}",
                    ]
                    if workers > 1
                    else []
                )
                + (
                    [
                        "-roi",
                        f"[{roi}]",
                    ]
                    if start is not None
                    else []
                )
            )

            print("starting:", name, dataset, setup, criterion)
            print(command)
            subprocess.run(command)
            print("finished!")


@cli.command()
@click.option("-p", "--prediction", type=click.Path(exists=True, dir_okay=False))
@click.option("-w", "--workers", type=int)
def predict_depricated(prediction, workers):
    targets = yaml.safe_load(
        Path("configs/yamls/zebrafish/targets/prediction_targets.yaml").open("r").read()
    )
    prediction_data = yaml.safe_load(Path(prediction).open("r").read())

    best_predictors = prediction_data["setups"]

    jobs = []

    for crop in prediction_data["predictions"]:
        offset, shape = crop.get("roi", (None, None))
        crop_jobs = {
            "name": crop["name"],
            "checkpoints": dict(),
        }
        if offset is not None:

            crop_jobs["start"] = Coordinate(offset)
            crop_jobs["end"] = (Coordinate(offset) + Coordinate(shape),)

        outputs = crop["outputs"]
        for output in outputs:
            best_predictor = best_predictors.get(output)
            if best_predictor is not None:
                output_index = targets[best_predictor["target"]].index(output)
                checkpoint_preds = crop_jobs["checkpoints"].setdefault(
                    (
                        f"/nrs/zebrafish/pattonw/training/finetuning/experiments/"
                        f"{best_predictor['setup']}/runs/{best_predictor['repetition']}"
                        f"/weights/{best_predictor['iteration']}.checkpoint",
                        best_predictor["architecture"],
                        best_predictor["task"],
                    ),
                    list(),
                )
                checkpoint_preds.append(f"{output_index}:{output}")
        jobs.append(crop_jobs)

    for job in jobs:
        for (checkpoint, architecture, task), channels in job["checkpoints"].items():
            if "start" in job:
                roi = ",".join(
                    list(f"{s}:{e}" for s, e in zip(job["start"], job["end"]))
                )
            else:
                roi = None

            command = [
                "python",
                "scripts/depricated/predict_daisy.py",
                "predict",
                "-c",
                checkpoint,
                "-a",
                architecture,
                "-t",
                task,
                "-cs",
                ",".join(channels),
                "-oc",
                "/nrs/zebrafish/pattonw/predictions/jrc_mus-liver/jrc_mus-liver.n5",
                "-od",
                f"evaluations-best/{job['name']}",
                "-ic",
                "/groups/zebrafish/zebrafish/data/jrc_mus-liver/jrc_mus-liver.n5",
                "-id",
                "volumes/raw/s0",
                "--bsub",
                "-w",
                f"{workers}",
            ]
            if roi is not None:
                command += [
                    "-roi",
                    f"[{roi}]",
                ]

            print("starting:", job["name"], checkpoint, channels)
            subprocess.run(command)
            print("finished!")


if __name__ == "__main__":
    cli()
