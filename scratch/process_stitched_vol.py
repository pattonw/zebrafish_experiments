import rawpy
import imageio
import numpy as np
from tqdm import tqdm
import zarr

from pathlib import Path

# Script for turning a raw file into a zarr dataset

path = Path(
    "/nrs/funke/pattonw/data/zebrafish/stitched/top_left_right_bottom_resliced_8555x5155x4419.raw"
)
container = zarr.open(
    "/nrs/funke/pattonw/predictions/zebrafish/zebrafish.n5", mode="r+"
)
output_data = container.create_dataset(
    "volumes/s17/raw", dtype=np.uint8, overwrite=True, shape=(8555, 5155, 4419)
)

n_z = 8555
size_y = 5155
size_x = 4419
size_z = 8555
count = size_x * size_y
start_z = 0
end_z = size_z
n_bytes = 1  # Number of bytes in a uint8 (for offset)
fd = open(path, "rb")
fd.seek(start_z)

for i in tqdm(range(start_z, end_z, n_z), desc=path.stem):
    offset = 0
    data = (
        np.fromfile(fd, dtype="b", offset=offset, count=n_z * count)
        .reshape(size_x, size_y, -1)
        .transpose((2, 1, 0))
    )
    output_data[i : i + data.shape[0]] = data
fd.close()
