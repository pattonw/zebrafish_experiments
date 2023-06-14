# zebrafish_experiments
scripts and configs for running zebrafish experiments

## To Avoid conflicts

Update the `dacapo.yaml` file to point to your own mongodb and file storage path.

## Organization:

```
├── configs  # everything related to data and dacapo configurations
│   ├── zebrafish  # ready to use dacapo config names including configs (architectures, datasplits, tasks, trainers, runs, predictions)
│   ├── scripts  # creation of dacapo configs
│   ├── yamls  # machine readable format for experiment configurations
├── scripts  # reusable cli scripts for common operations
├── scratch  # one-time scripts that aren't particularly reusable
├── runs  # logs from running training
```

## installation

create conda environment with `python >= 3.10`

`pip install git+https://github.com/funkelab/dacapo`

`pip install git+https://github.com/pattonw/funlib.show.neuroglancer@funlib-update`

`pip install -r requirements.txt`


## usage
use `--help` to get more info about script parameters

### parsing spreadsheet

Data prep: data needs to be converted into n5s or zarrs for training. Once this is done you can use `scratch/reformat_dataset.py` to compute masks, sample points, etc. for the data.

### creating dacapo configs
`--force` flag replaces the config in the mongodb with the new version.
run configs contain copies of the configs that were used to create them so overwriting `Task`, `Trainer`, `Architecture`, and `DataSplit` configs won't affect old run configs.
1) `python configs/scripts/datasplits/datasets.py update --force`  # getting very slow due to large number of resolution/target/dataset combinations
2) `python configs/scripts/tasks/tasks.py update --force`
3) `python configs/scripts/trainers/trainers.py update --force`
4) `python configs/scripts/architectures/architectures.py update --force`

### creating a new run config
`python configs/scripts/runs/runs.py update`

### running experiment
`python scripts/submit.py run`

### logging
find logs in `runs` directory.

### plotting
`python scripts/plot.py plot`

### prediction
Single prediction from command line can be done using `scripts/predict_daisy`.

If you want to run multiple predictions and or use a config file to have a record of what was run you can use
`python scripts/submit.py predict`
Example config files can be found in `configs/zebrafish/predictions`.

### postprocessing
Postprocessing scripts can be found in `scripts/post_processing`. There are a couple versions for post processing workers. Post processing is broken down into 3 steps.
2) generate fragments (2: mwatershed or 1: waterz)
    - using 2 tends to do better but is slower and can generate many small fragments that slow down later processing
3) agglomerate fragments (2: mwatershed or 1: waterz)
    - using 2 utilizes long range affinities as well as short range affinities to generate more edges. Tends to do significantly better on datasets where false merges occur.
4) lut generation (2: mwatershed or 1: waterz)
    - 1 uses only positive edges to agglomerate and can be prone to agglomerating entire volumes into a single id. 2 uses mwatershed to run mutex watershed with negative edges and performs significantly better in cases where merge errors are an issue.