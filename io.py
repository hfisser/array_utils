import rasterio as rio
import numpy as np


def rio_read_all_bands(file_path):
    with rio.open(file_path, "r") as src:
        meta = src.meta
        n_bands = src.count
        arr = np.zeros((src.count, src.height, src.width))
        for i in range(n_bands):
            arr[i] = src.read(i+1)
    return arr, meta
