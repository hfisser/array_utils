import numpy as np
from rasterio.transform import Affine


def lat_from_meta(meta):
    try:
        t, h = meta["transform"], meta["height"]
    except KeyError as e:
        raise e
    return np.arange(t[5], t[5] + (t[4] * h), t[4])


def lon_from_meta(meta):
    try:
        t, w = meta["transform"], meta["width"]
    except KeyError as e:
        raise e
    return np.arange(t[2], t[2] + (t[0] * w), t[0])


# lat, lon
def transform_lat_lon(lat, lon):
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    trans = Affine.translation(lon[0], lat[0])
    scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
    return trans * scale
