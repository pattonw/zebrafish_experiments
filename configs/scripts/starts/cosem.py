from dacapo.store.create_store import create_config_store
from dacapo.experiments.run_config import RunConfig
from dacapo.store.config_store import DuplicateNameError
from dacapo.validate import validate

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
@click.option("-dbh", "--db-host", type=str)
@click.option("-dbn", "--db-name", type=str)
@click.option("-r", "--run", type=str)
@click.option("-c", "--criterion", type=str)
def update(db_host, db_name, run, criterion):
    pass


if __name__ == "__main__":
    cli()
