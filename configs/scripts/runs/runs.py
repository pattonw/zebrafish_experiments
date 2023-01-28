from dacapo.store.create_store import create_config_store
from dacapo.experiments.run_config import RunConfig
from dacapo.store.config_store import DuplicateNameError
from dacapo.experiments.starts import StartConfig

import click

import logging
from pathlib import Path
import yaml
import itertools


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
@click.option("-d", "--datasplit", type=str, multiple=True)
@click.option("-a", "--architecture", type=str, multiple=True)
@click.option("-ta", "--task", type=str, multiple=True)
@click.option("-tr", "--trainer", type=str, multiple=True)
@click.option("-s", "--start", type=str, multiple=True)
@click.option("-r", "--repetitions", type=int)
@click.option("-i", "--iterations", type=int)
@click.option("-v", "--validation-interval", type=int)
@click.option("--force/--no-force", default=False)
def update(
    name,
    datasplit,
    architecture,
    task,
    trainer,
    start,
    repetitions,
    iterations,
    validation_interval,
    force,
):
    datasplits = datasplit
    architectures = architecture
    tasks = task
    trainers = trainer
    starts = start

    config_store = create_config_store()
    datasplits = [
        config_store.retrieve_datasplit_config(datasplit) for datasplit in datasplits
    ]
    architectures = [
        config_store.retrieve_architecture_config(architecture)
        for architecture in architectures
    ]
    tasks = [config_store.retrieve_task_config(task) for task in tasks]
    trainers = [config_store.retrieve_trainer_config(trainer) for trainer in trainers]

    run_config_names = []
    for (
        datasplit_config,
        architecture_config,
        task_config,
        trainer_config,
        start,
    ) in itertools.product(datasplits, architectures, tasks, trainers, starts):
        if len(start) > 0:
            run, criterion = start.split(":")
            start_config = StartConfig(run=run, criterion=criterion)
        else:
            start_config = None
        for i in range(repetitions):
            run_config = RunConfig(
                name=("_").join(
                    [
                        "scratch" if start_config is None else "finetuned",
                        task_config.name,
                        datasplit_config.name,
                        architecture_config.name,
                        trainer_config.name,
                    ]
                )
                + f"__{i}",
                task_config=task_config,
                datasplit_config=datasplit_config,
                architecture_config=architecture_config,
                trainer_config=trainer_config,
                num_iterations=iterations,
                validation_interval=validation_interval,
                repetition=i,
                start_config=start_config,
            )
            try:
                config_store.store_run_config(run_config)
            except DuplicateNameError as e:
                if force:
                    config_store.runs.delete_one({"name": run_config.name})
                    config_store.store_run_config(run_config)
                else:
                    run_names = config_store.retrieve_run_config_names()
                    versions = set(
                        [
                            name[
                                len(run_config.name.split("__")[0])
                                + 1 : -(len(run_config.name.split("__")[1]) + 2)
                            ]
                            for name in run_names
                            if name.startswith(run_config.name.split("__")[0])
                            and name.endswith(run_config.name.split("__")[1])
                        ]
                    )
                    versions.remove("")
                    versions = [int(ver[1:]) for ver in versions if ver.startswith("v")]
                    next_version = len(versions) + 1
                    run_config.name = f"{run_config.name.split('__')[0]}_v{next_version}__{run_config.name.split('__')[1]}"
                    config_store.store_run_config(run_config)
            run_config_names.append(run_config.name)

    output_path = Path(f"configs/zebrafish/runs/{name}")
    if not output_path.exists():
        output_path.mkdir(parents=True)
    (output_path / "runs.yaml").open("w").write(yaml.safe_dump(run_config_names))
    (output_path / "tasks.yaml").open("w").write(
        yaml.safe_dump([task.name for task in tasks])
    )
    (output_path / "datasplits.yaml").open("w").write(
        yaml.safe_dump([datasplit.name for datasplit in datasplits])
    )
    (output_path / "architectures.yaml").open("w").write(
        yaml.safe_dump([architecture.name for architecture in architectures])
    )
    (output_path / "trainers.yaml").open("w").write(
        yaml.safe_dump([trainer.name for trainer in trainers])
    )
    (output_path / "starts.yaml").open("w").write(
        yaml.safe_dump([start if len(start) > 0 else None for start in starts])
    )


if __name__ == "__main__":
    cli()
