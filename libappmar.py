"""
APPMAR 1.0 Library

Marianella Bolívar
Diego Casas
Germán Rivillas Ospina, PhD
"""

import logging
import os
import urllib
import shutil
import pickle
import configparser

import xarray as xr
import numpy as np
from osgeo import gdal
from scipy.stats import linregress
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator

logging.basicConfig(level=logging.INFO)

URL_BASE = 'https://polar.ncep.noaa.gov/waves/hindcasts/'
PATH_SINGLE = "nww3/{grid}.{param}.{year}{month:02}.grb"
PATH_MULTI = "multi_1/{year}{month:02}/gribs/multi_1.{grid}.{param}.{year}{month:02}.grb2"

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

SEASONS = ["Winter", "Summer", "Spring", "Fall"]

CONF_FILE = "config.ini"

ONE_OVER_SQRT2 = 2**-0.5
PNEX100 = 0.99

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


def download_grib(gridtype, **metadata):
    if gridtype == "single":
        path = PATH_SINGLE.format(**metadata)
    else:
        path = PATH_MULTI.format(**metadata)
    fn = "data/" + os.path.split(path)[-1]
    with urllib.request.urlopen(URL_BASE + path) as res:
        if not os.path.exists(fn) or os.path.getsize(fn) != res.length:
            with open(fn, 'wb') as file:
                shutil.copyfileobj(res, file)


def download_data(grid_ids, par_ids, years, months):
    if "_" in grid_ids[0]:
        multi, single = grid_ids
    else:
        single, multi = grid_ids
    os.makedirs("data", exist_ok=True)
    for y in years:
        for m in months:
            for par in par_ids:
                try:
                    download_grib("multi", grid=multi, param=par, year=y, month=m)
                    logging.info("Downloaded %d/%d %s, %s (Multigrid)", y, m, par, multi)
                    continue
                except urllib.error.HTTPError as e:
                    if e.code != 404:
                        raise e
                    logging.debug("Can't find %d/%d %s, %s (Multigrid)", y, m, par, multi)
                try:
                    download_grib("single", grid=single, param=par, year=y, month=m)
                    logging.info("Downloaded %d/%d %s, %s (NWW3 grid)", y, m, par, single)
                except urllib.error.HTTPError as e:
                    if e.code != 404:
                        raise e
                    logging.debug("Can't find %d/%d %s, %s (NWW3 grid)", y, m, par, single)


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


def parse_config():
    config = configparser.ConfigParser()
    config.read(CONF_FILE)
    months = {"All": [*range(1, 13)]}
    for season in SEASONS:
        months[season] = [int(x) for x in config["seasons"][season].split(",")]
    point = config["location"]["point"]
    box = config["location"]["box"]
    return months, point, box


def get_defaults():
    if os.path.exists(CONF_FILE):
        months, point, box = parse_config()
    else:
        # Caribbean seasons by default
        months = {
            "Winter": [12, 1, 2],
            "Summer": [6, 7, 8],
            "Spring": [3, 4, 5],
            "Fall": [9, 10, 11],
            "All": [*range(1, 13)]
        }
        # Magdalena River Mouth by default
        point = "-74.85,11.13"
        box = "-75.3,-74.1,10,11.5"
    return months, point, box


def plotpos(n):
    i = np.arange(1, n + 1)
    P = i / (n + 1)
    return P


def fit_weibull3(x, tol=1e-4, delta=1e-5, verbose=False):
    y = plotpos(len(x))
    eta = -np.log(-np.log(y))
    xmax = max(x)
    e = 1
    n = 1
    r1 = 0
    while e > tol:
        loc = xmax * n + delta
        xi = -np.log(loc - x)
        r0 = r1
        slope, intercept, r1, *_ = linregress(xi, eta)
        e = r1**2 - r0**2
        if verbose:
            print(n, loc)
        n += 1
    shape = slope
    scale = np.exp(intercept/shape)
    return shape, loc, scale, r1

def plot_weibull(arr, ax):
    xobs = np.sort(arr)
    shape, loc, scale, r = fit_weibull3(xobs)
    x100 = loc - scale * (-np.log(PNEX100))**(1/shape)
    xfit = np.array([xobs[0], x100])
    pobs = plotpos(len(xobs))
    pfit = np.exp(-((loc - xfit) / scale)**shape)
    ax.set_xscale('function', functions=(lambda x: -np.log(loc - x), lambda xi: loc - np.exp(-xi)))
    ax.set_yscale('function', functions=(lambda y: -np.log(-np.log(y)), lambda eta: np.exp(-np.exp(-eta))))
    ax.set_ylim([0.009, 0.991])
    ax.set_yticks([0.50, 0.80, 0.90, 0.96, 0.98, 0.99])
    ax.set_yticklabels([2, 5, 10, 25, 50, 100])
    ax.grid()
    ax.set_xlabel("Significant wave height (m)")
    ax.set_ylabel("Return period (years)")
    ax.scatter(xobs, pobs, color='r', marker='+')
    ax.plot(xfit, pfit, 'k', label=f"Shape = {shape:.4f}\nLocation = {loc:.4f}\nScale = {scale:.4f}\nr² = {r**2:.4}")
    ax.legend(handletextpad=0, handlelength=0)

def compute_clusters(pairs):
    scaler = StandardScaler()
    scaled_pairs = scaler.fit_transform(pairs)
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(scaled_pairs)
        sse.append(kmeans.inertia_)
    kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
    k = kl.elbow
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(scaled_pairs)
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    labels = kmeans.labels_
    return centers, labels


def plot_clusters(pairs, centers, labels, ax):
    ax.scatter(pairs[:, 0], pairs[:, 1], c=labels.astype(float), s=0.5, cmap="Set1")
    ax.scatter(centers[:, 0], centers[:, 1], c="white", edgecolor="k")
    ax.set_xlabel("Significant wave height (m)")
    ax.set_ylabel("Peak period (s)")
