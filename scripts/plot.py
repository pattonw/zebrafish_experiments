from dataclasses import replace
from dacapo.store.create_store import create_stats_store, create_config_store
from dacapo.experiments import Run

import matplotlib.pyplot as plt
import numpy as np
import click
import xarray as xr

import math
from pathlib import Path
import logging
import itertools
import pickle

logger = logging.getLogger(__file__)


def replace_outliers(data, perc=0.99):

    # calculate percentile
    threshold = data.quantile(perc)

    # find outliers and replace them with max among remaining values
    mask = data.where(abs(data) <= threshold)
    max_value = mask.max().values
    # .where replace outliers with nan
    # mask = mask.fillna(max_value)
    data = mask

    return data


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
@click.option(
    "-pt",
    "--plot-type",
    type=click.Choice(["BAR", "SCATTER", "LINE", "VIOLIN"], case_sensitive=False),
    default=None,
)
@click.option("-dir", "--directory", type=click.Path(file_okay=False))
@click.option("-r", "--run", type=str, multiple=True)
@click.option("-cr", "--criterion", type=str, multiple=True)
@click.option("-ds", "--dataset", type=str, multiple=True)
@click.option("-s", "--smooth", type=int, default=100)
def plot(plot_type, directory, run, criterion, dataset, smooth):
    directory = Path(directory)
    if not directory.exists():
        directory.mkdir(parents=True)

    runs = list(run)
    runs_info = list(zip(*[run.split(":") for run in runs]))
    if len(runs_info) == 3:
        runs, coords, categories = runs_info
        coords = [float(coord) for coord in coords]
    elif len(runs_info) == 2:
        runs, categories = runs_info
        coords = None
    elif len(runs_info) == 1:
        (runs,) = runs_info
        categories = runs
    runs = list(runs)
    categories = list(categories)
    colormap = {cat: col for cat, col in zip(set(categories), plt.cm.Dark2.colors)}
    criteria = list(criterion)
    datasets = {
        dataset.split(":")[0]: dataset.split(":")[-1].split(",") for dataset in dataset
    }

    stats_store = create_stats_store()
    config_store = create_config_store()

    xarrays = []
    loss_arrays = []
    for run in runs:
        run_config = config_store.retrieve_run_config(run_name=run)
        try:
            validation_scores = Run.get_validation_scores(run_config)
        except KeyError as e:
            for crop in run_config.datasplit_config.train_configs:
                if "174" in crop.name:
                    print(crop)
            for crop in run_config.datasplit_config.validate_configs:
                if "174" in crop.name:
                    print(crop)
            raise e
        validation_scores.scores = (
            stats_store.retrieve_validation_iteration_scores(run)
        )
        logger.info(
            f"Run: {run} validated until: {validation_scores.validated_until()}"
        )
        xarrays.append(validation_scores.to_xarray())
        training_stats = stats_store.retrieve_training_stats(run)
        logger.info(f"Run: {run} trained until: {training_stats.trained_until()}")
        loss_arrays.append(training_stats.to_xarray())

    # combine all of the xarrays with a new axis for runs
    runs = xr.DataArray(runs, coords={"runs": runs})
    try:
        xarray = xr.concat(xarrays, dim=runs).sel(criteria=criteria)
    except KeyError as e:
        print(list(xarrays[0].criteria.values), criteria)
        raise e
    losses = xr.concat(loss_arrays, dim=runs)

    dataset_name_mapping = {
        dataset.name: dataset for dataset in xarray.coords["datasets"].values
    }
    classes = ["_mito", "_nucleus", "_er", "_many_v2", "_many", "_all", "_ecs"]

    def strip_organelles(string):
        for organelle_class in classes:
            string = string.replace(organelle_class, "")
        return string

    stripped_names = set()
    for name in dataset_name_mapping:
        for organelle_class in classes:
            name = name.replace(organelle_class, "")
        stripped_names.add(name)

    datasets = (
        {
            stripped_name: [
                dataset
                for dataset in xarray.coords["datasets"].values
                if strip_organelles(dataset.name) == stripped_name
            ]
            for stripped_name in stripped_names
        }
        if len(datasets) == 0
        else {
            group_name: [dataset_name_mapping[dataset] for dataset in dataset_group]
            for group_name, dataset_group in datasets.items()
        }
    )

    # get the max accross "parameters" dimension
    best_parameters, xarray = validation_scores.get_best(xarray, dim="parameters")
    sorted_runs = sorted(list(losses.coords["runs"].values))
    enumerated_datasets = list(
        enumerate(sorted(list(datasets.items()), key=lambda x: x[0]))
    )

    print(xarray.iterations)

    if plot_type == "LINE":
        for criterion in criteria:
            fig, ax = plt.subplots(
                1 + len(datasets), figsize=[16, (1 + len(datasets)) * 6]
            )
            for run in sorted_runs:
                run_loss = losses.sel(runs=run, iterations=losses.coords["iterations"])
                # run_loss = replace_outliers(run_loss)
                if smooth > 0:
                    run_loss = run_loss.rolling(iterations=smooth).mean()
                run_loss.plot.line(x="iterations", ax=ax[0], label=run)
            for i, (group_name, dataset_group) in enumerated_datasets:
                for dataset in dataset_group:
                    dataset_data = xarray.sel(criteria=criterion, datasets=dataset)
                    for run in dataset_data.coords["runs"].values:
                        run_validation = dataset_data.sel(runs=run)
                        if all(np.isnan(run_validation)):
                            continue
                        run_validation.plot.line(
                            x="iterations", ax=ax[1 + i], label=f"{run}"
                        )
                ax[1 + i].set_title(f"{group_name}_{criterion}")
                ax[1 + i].legend()
            ax[0].set_title("Loss")
            ax[0].legend()
            # ax[0].set_ylim([0, 0.2])
            plt.savefig(
                f"{directory}/{criterion}_line.png",
                bbox_inches="tight",
            )

    elif plot_type == "BAR":
        _, xarray = validation_scores.get_best(xarray, dim="iterations")
        for criterion in criteria:
            for group_name, dataset_group in datasets.items():
                best_scores = xarray.sel(criteria=criterion, datasets=dataset)

                plt.bar(best_scores.coords["runs"], best_scores.values)
                plt.savefig(
                    f"{directory}/{dataset.name}_{criterion}_bar.png",
                    bbox_inches="tight",
                )

    elif plot_type == "VIOLIN":
        indexes, xarray = validation_scores.get_best(xarray, dim="iterations")
        for criterion in criteria:
            for group_name, dataset_group in datasets.items():
                fig, ax = plt.subplots()
                category_data = []
                for dataset in dataset_group:
                    best_scores = xarray.sel(criteria=criterion, datasets=dataset)
                    for category in set(categories):
                        indx = [
                            i for i, cat in enumerate(categories) if cat == category
                        ]
                        if all(
                            [math.isnan(score) for score in best_scores.values[indx]]
                        ):
                            continue
                        else:
                            category_data.append(best_scores.values[indx])
                print(category_data)
                ax.violinplot(
                    dataset=category_data,
                    showextrema=False,
                    showmeans=True,
                )
                ax.set_title(f"{group_name}, {criterion}")
                ax.legend()
                plt.savefig(
                    f"{directory}/{group_name}_{criterion}_violin.png",
                    bbox_inches="tight",
                )

    elif plot_type == "SCATTER":
        indexes, xarray = validation_scores.get_best(xarray, dim="iterations")
        for criterion, dataset, run in itertools.product(
            xarray.coords["criteria"], xarray.coords["datasets"], xarray.coords["runs"]
        ):
            if not np.isnan(
                xarray.sel(criteria=criterion, datasets=dataset, runs=run).item()
            ):
                print(
                    criterion.item(),
                    dataset.item(),
                    run.item(),
                    indexes.sel(criteria=criterion, datasets=dataset, runs=run).item(),
                    xarray.sel(criteria=criterion, datasets=dataset, runs=run).item(),
                )
        for criterion in criteria:
            print(f"{criterion}: {xarray.sel(criteria=criterion)}")
            for group_name, dataset_group in datasets.items():
                fig, ax = plt.subplots()
                for dataset in dataset_group:
                    best_scores = xarray.sel(criteria=criterion, datasets=dataset)
                    for category in set(categories):
                        indx = [
                            i for i, cat in enumerate(categories) if cat == category
                        ]
                        if all(
                            [math.isnan(score) for score in best_scores.values[indx]]
                        ):
                            continue
                        ax.scatter(
                            np.array(coords)[indx],
                            best_scores.values[indx],
                            c=[colormap[category]],
                            label=category,
                        )
                ax.set_title(f"{group_name}, {criterion}")
                ax.legend()
                plt.savefig(
                    f"{directory}/{group_name}_{criterion}_scatter.png",
                    bbox_inches="tight",
                )
    else:
        pickle.dump(xarray, open("xarray.xr", "wb"))
        indexes, xarray = validation_scores.get_best(xarray, dim="iterations")
        print(indexes, xarray)
        for run, dataset, criterion in itertools.product(
            indexes.coords["runs"],
            indexes.coords["datasets"],
            indexes.coords["criteria"],
        ):
            print(
                run.item(),
                dataset.item(),
                criterion.item(),
                indexes.sel(criteria=criterion, datasets=dataset, runs=run).item(),
                xarray.sel(criteria=criterion, datasets=dataset, runs=run).item(),
            )


if __name__ == "__main__":
    cli()
