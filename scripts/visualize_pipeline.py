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
@click.option("-r", "--run-config", type=str)
def visualize_pipeline(run_config):
    from dacapo.experiments import Run
    from dacapo.store import create_config_store, create_array_store

    from funlib.geometry import Coordinate
    import gunpowder as gp

    import neuroglancer

    import numpy as np

    config_store = create_config_store()
    config = config_store.retrieve_run_config(run_config)
    run = Run(config)

    array_store = create_array_store()
    run.trainer.iteration = 0
    run.trainer.num_data_fetchers = 1
    run.trainer.batch_size = 1
    run.trainer.build_batch_provider(
        run.datasplit.train,
        run.model,
        run.task,
        array_store.snapshot_container(run.name),
    )

    pipeline = run.trainer._pipeline
    request = run.trainer._request

    def load_batch(event):
        print("fetching batch")
        with gp.build(pipeline):
            batch = pipeline.request_batch(request)

        with viewer.txn() as s:
            while len(s.layers) > 0:
                del s.layers[0]

            # reverse order for raw so we can set opacity to 1, this
            # way higher res raw replaces low res when available
            for name, array in batch.items():
                print(name)
                data = array.data[0]

                channel_dims = len(data.shape) - len(array.spec.voxel_size)
                assert channel_dims <= 1

                dims = neuroglancer.CoordinateSpace(
                    names=["c^", "z", "y", "x"][-len(data.shape) :],
                    units="nm",
                    scales=tuple([1] * channel_dims) + tuple(array.spec.voxel_size),
                )

                local_vol = neuroglancer.LocalVolume(
                    data=data,
                    voxel_offset=tuple([0] * channel_dims)
                    + tuple((-array.spec.roi.shape / 2) / array.spec.voxel_size),
                    dimensions=dims,
                )

                s.layers[str(name)] = neuroglancer.ImageLayer(source=local_vol)

            s.layout = neuroglancer.row_layout(
                [
                    neuroglancer.column_layout(
                        [
                            neuroglancer.LayerGroupViewer(layers=[str(k) for k, v in batch.items()]),
                        ]
                    )
                ]
            )

    neuroglancer.set_server_bind_address("0.0.0.0")

    viewer = neuroglancer.Viewer()

    viewer.actions.add("load_batch", load_batch)

    with viewer.config_state.txn() as s:
        s.input_event_bindings.data_view["keyt"] = "load_batch"

    print(viewer)
    load_batch(None)

    input("Enter to quit!")


if __name__ == "__main__":
    cli()
