"""
APPMAR 1.0 Library

Marianella Bolívar
Diego Casas
Germán Rivillas Ospina, PhD
"""

import logging
import os
from ftplib import FTP
import pickle

import xarray as xr
import numpy as np
from osgeo import gdal

logging.basicConfig(level=logging.INFO)

HOST = "polar.ncep.noaa.gov"
PATH_MULTI = "/history/waves/multi_1"
PATH_SINGLE = "/history/waves/nww3"

DIR = [
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1)
]

ONE_OVER_SQRT2 = 2**-0.5

def merge_data(par):
    xs = []
    for x in par.values():
        xs += x
    return xs


def save_obj(obj, fname):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

def _year_month(dirname):
    y = int(dirname[:4])
    m = int(dirname[4:])
    return y, m


def _grid_parameter(filename):
    gid, pid = filename.split(".")[1:3]
    return gid, pid


def _grid_parameter_year_month(filename):
    gid, pid, ym = filename.split(".")[:3]
    y = int(ym[:4])
    m = int(ym[4:])
    return gid, pid, y, m


def _grid_parameter_year_month_grb2(filename):
    gid, pid, ym = filename.split(".")[1:4]
    y = int(ym[:4])
    m = int(ym[4:])
    return gid, pid, y, m


def download_data(grid_ids, par_ids, years, months):
    dld = []
    os.makedirs("data", exist_ok=True)
    with FTP(HOST) as ftp:
        msg = ftp.login()
        logging.info("Log in to %s: %s", HOST, msg)
        msg = ftp.cwd(PATH_MULTI)
        logging.debug("Chande directory to %s: %s", PATH_MULTI, msg)
        for dn in ftp.nlst():
            if not dn.isnumeric():
                continue
            y, m = _year_month(dn)
            if y not in years or m not in months:
                logging.debug("%s is not in year/month range", dn)
                continue
            path = f"{PATH_MULTI}/{dn}/gribs"
            msg = ftp.cwd(path)
            logging.debug("Chande directory to %s: %s", path, msg)
            for fn in ftp.nlst():
                if not fn.endswith(".grb2"):
                    continue
                gid, pid = _grid_parameter(fn)
                if gid not in grid_ids or pid not in par_ids:
                    logging.debug("%s is not a requested grid/parameter", fn)
                    continue
                if os.path.getsize(f"data/{fn}") != ftp.size(fn):
                    with open(f"data/{fn}", "wb") as f:
                        msg = ftp.retrbinary(f"RETR {fn}", f.write)
                    logging.info("Download %s: %s", fn, msg)
                else:
                    logging.info("%s already exists: Skip download", fn)
                dld.append((y, m))
            msg = ftp.cwd(PATH_MULTI)
            logging.debug("Chande directory to %s: %s", PATH_MULTI, msg)
        msg = ftp.cwd(PATH_SINGLE)
        logging.debug("Chande directory to %s: %s", PATH_SINGLE, msg)
        for fn in ftp.nlst():
            if not fn.endswith(".grb"):
                continue
            gid, pid, y, m = _grid_parameter_year_month(fn)
            if y not in years or m not in months:
                logging.debug("%s is not in year/month range", fn)
                continue
            if gid not in grid_ids or pid not in par_ids:
                logging.debug("%s is not a requested grid/parameter", fn)
                continue
            if (y, m) in dld:
                logging.debug(
                    "%d-%02d already downloaded from multi_1: Skip download", y, m)
                continue
            if os.path.getsize(f"data/{fn}") != ftp.size(fn):
                with open(f"data/{fn}", "wb") as f:
                    msg = ftp.retrbinary(f"RETR {fn}", f.write)
                logging.info("Download %s: %s", fn, msg)
            else:
                logging.info("%s already exists: Skip download", fn)


def frequency_curve(par_id, months, lon, lat):
    """Returns an array parameter season mean and their cumulative probability estimate."""
    xall = []
    for fn in os.listdir("data"):
        if fn.endswith(".grb"):
            _, pid, y, m = _grid_parameter_year_month(fn)
        elif fn.endswith(".grb2"):
            _, pid, y, m = _grid_parameter_year_month_grb2(fn)
        else:
            continue
        if pid != par_id:
            continue
        if m not in months:
            continue
        logging.info("Reading %s", fn)
        x = xr.load_dataarray(f"data/{fn}", engine="cfgrib").sel(longitude=lon, latitude=lat, method="nearest").values
        xall.extend(x)
    n = len(xall)
    p = np.arange(1, n + 1) / (n + 1)
    return np.flip(np.sort(xall)), p


def joint_distribution(par_ids, months, lon, lat):
    """Returns an array parameter season mean and their cumulative probability estimate."""
    done1 = {}
    done2 = {}
    for fn in os.listdir("data"):
        if fn.endswith(".grb"):
            _, pid, y, m = _grid_parameter_year_month(fn)
        elif fn.endswith(".grb2"):
            _, pid, y, m = _grid_parameter_year_month_grb2(fn)
        else:
            continue
        if pid not in par_ids:
            continue
        if m not in months:
            continue
        logging.info("Reading %s", fn)
        x = xr.load_dataarray(f"data/{fn}", engine="cfgrib").sel(longitude=lon, latitude=lat, method="nearest").values
        if pid == par_ids[0]:
            done1[(y, m)] = x
        else:
            done2[(y, m)] = x
    xall = [[], []]
    for k in done1:
        xall[0].extend(done1[k])
        xall[1].extend(done2[k])
    if len(xall[0]) != len(xall[1]):
        raise Exception("Missing or corrupt files: Check the data directory for corrupt files and download again.")
    xall = np.array(xall)
    return xall[:, ~np.isnan(xall.sum(axis=0))]


def load_data(par_ids, months, lon, lat):
    """TO DO"""
    done = {
        "dp": {},
        "hs": {},
        "tp": {}
    }
    for fn in os.listdir("data"):
        if fn.endswith(".grb"):
            _, pid, y, m = _grid_parameter_year_month(fn)
        elif fn.endswith(".grb2"):
            _, pid, y, m = _grid_parameter_year_month_grb2(fn)
        else:
            continue
        if pid not in par_ids:
            continue
        if m not in months:
            continue
        logging.info("Reading %s", fn)
        x = xr.load_dataarray(f"data/{fn}", engine="cfgrib").sel(longitude=lon, latitude=lat, method="nearest").values
        done[pid][(y, m)] = [*x]
    return done


def weibull_data(par_id, months, lon, lat):
    """Returns an array parameter season mean and their cumulative probability estimate."""
    done = {}
    for fn in os.listdir("data"):
        if fn.endswith(".grb"):
            _, pid, y, m = _grid_parameter_year_month(fn)
        elif fn.endswith(".grb2"):
            _, pid, y, m = _grid_parameter_year_month_grb2(fn)
        else:
            continue
        if pid != par_id:
            continue
        if m not in months:
            continue
        logging.info("Reading %s", fn)
        x = xr.load_dataarray(f"data/{fn}", engine="cfgrib").sel(longitude=lon, latitude=lat, method="nearest").values
        if y in done:
            done[y].append(x.max())
        else:
            done[y] = [x.max()]
    xall = []
    for i, y in enumerate(done):
        xall.append(max(done[y]))
    return np.array(xall)


def gen_dataset_from_raster(fname, lon, lat, time):
    ds = gdal.Open(fname)
    wind = []
    for i in range(1, ds.RasterCount + 1, 2):
        band_u = ds.GetRasterBand(i)
        band_v = ds.GetRasterBand(i + 1)
        nodata_u = band_u.GetNoDataValue()
        nodata_v = band_v.GetNoDataValue()
        arr_u = np.ma.masked_equal(band_u.ReadAsArray(), nodata_u)
        arr_v = np.ma.masked_equal(band_v.ReadAsArray(), nodata_v)
        wind.append((arr_u**2 + arr_v**2)**0.5)
    ds = xr.DataArray(np.ma.stack(wind), coords=[time, lat, lon], dims=["time", "latitude", "longitude"])
    return ds


def load_max(par_id):
    """TO DO"""
    done = {}
    interp = False
    for fn in os.listdir("data"):
        if not fn.endswith(".grb2"):
            continue
        _, pid, y, m = _grid_parameter_year_month_grb2(fn)
        if pid != par_id:
            continue
        if pid == "wind":
            wind = xr.load_dataset(f"data/{fn}", engine="cfgrib")
            vel = (wind.u**2 + wind.v**2)**0.5
            xarr = vel.max("step")
        else:
            xarr = xr.load_dataarray(f"data/{fn}", engine="cfgrib").max("step")
        if y in done:
            done[y].append(xarr)
        else:
            done[y] = [xarr]
    if len(done) != 0:
        interp = True
        ref = xarr
    for fn in os.listdir("data"):
        if not fn.endswith(".grb"):
            continue
        _, pid, y, m = _grid_parameter_year_month(fn)
        if pid != par_id:
            continue
        if pid == "wind":
            da = xr.load_dataarray(f"data/{fn}", engine="cfgrib")
            xarr = gen_dataset_from_raster(f"data/{fn}", da.longitude.values, da.latitude.values, da.time.values)
        else:
            xarr = xr.load_dataarray(f"data/{fn}", engine="cfgrib")
        # Compute element-wise maximum
        if interp:
            xarr = xarr.interp_like(ref).max("time")
        else:
            xarr = xarr.max("time")
        if y in done:
            done[y].append(xarr)
        else:
            done[y] = [xarr]
    maxima = []
    for y in done:
        maximum = xr.full_like(done[y][0], np.nan)
        for arr in done[y]:
            maximum = np.fmax(maximum.values, arr)
        maxima.append(maximum)
    return maxima


def interp_idw(arr):
    """Inverse Distance Weighted (IDW) Interpolation with Python"""
    bak = arr.copy()
    nrow, ncol = arr.shape
    for i in range(nrow):
        for j in range(ncol):
            if np.isnan(bak[i, j]):
                s1 = 0
                s2 = 0
                for di, dj in DIR:
                    try:
                        bak_dir = bak[i + di, j + dj]
                    except IndexError:
                        continue
                    if np.isnan(bak_dir):
                        continue
                    if abs(di) == abs(dj):
                        w = ONE_OVER_SQRT2
                    else:
                        w = 1
                    s1 += w*bak_dir
                    s2 += w
                if s2 == 0:
                    continue
                arr[i, j] = s1 / s2
    return arr
