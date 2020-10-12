import numpy as np
import geopandas as gpd
import pandas as pd
import xarray as xr
import rasterio as rio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from numba import jit
from rasterio import Affine, features


# extracts coordinates at value in np array and returns points as GeoDataFrame
# data 2d np array
# match_value Float value in data where point coordinates are extracted
# lon_lat dict of:
# "lon": np array longitude values"
# "lat": np array latitude values"
# crs String EPSG:XXXX
def points_from_np(data, match_value, lon_lat, crs):
    indices = np.argwhere(data == match_value)
    if len(indices) > 0:
        lat_indices = indices[:, [0]]
        lon_indices = indices[:, [1]]
        lat_coords = lon_lat["lat"][lat_indices]
        lon_coords = lon_lat["lon"][lon_indices]
        points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(lon_coords, lat_coords))
        points.crs = crs
        return points


def raster_to_points(raster, lon_lat, field_name, crs):
    points_list = []
    match_values = np.unique(raster[(raster != 0) * ~np.isnan(raster)])  # by pixel value
    for x in match_values:
        points = points_from_np(raster, x, lon_lat, crs=crs)
        points[field_name] = [x] * len(points)
        points_list.append(points)
    return gpd.GeoDataFrame(pd.concat(points_list, ignore_index=True))


# lat, lon
def transform_lat_lon(lat, lon):
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    trans = Affine.translation(lon[0], lat[0])
    scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
    return trans * scale


def lat_from_meta(meta):
    try:
        t, h = meta["transform"], meta["height"]
    except KeyError as e:
        raise e
    return np.arange(t[5], t[5] + (t[4] * h), h)


def lon_from_meta(meta):
    try:
        t, w = meta["transform"], meta["width"]
    except KeyError as e:
        raise e
    return np.arange(t[2], t[2] + (t[0] * w), w)


def rasterize(polygons, lat, lon, fill=np.nan):
    transform = transform_lat_lon(lat, lon)
    out_shape = (len(lat), len(lon))
    raster = features.rasterize(polygons.geometry, out_shape=out_shape,
                                fill=fill, transform=transform,
                                dtype=float)
    return xr.DataArray(raster, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))


def rescale(arr, min_val, max_val):
    arr += -(np.min(arr))
    arr /= np.max(arr) / (max_val - min_val)
    arr += min_val
    return arr


def rio_read_all_bands(file_path):
    with rio.open(file_path, "r") as src:
        meta = src.meta
        n_bands = src.count
        arr = np.zeros((src.count, src.height, src.width))
        for i in range(n_bands):
            arr[i] = src.read(i+1)
    return arr, meta


@jit(nopython=True, parallel=True)
def normalized_ratio(a, b):
    return (a - b) / (a + b)


def plot_img(img):
    if isinstance(img, str):
        img = mpimg.imread(img)
    else:
        img = rescale(img, 0, 255).astype(np.uint8)
        if img.shape[0] == 3:
            img = img.swapaxes(0, 2)
    imgplot = plt.imshow(img)
    plt.show()

