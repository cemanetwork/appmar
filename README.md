# APPMAR 1.0

A toolbox for management of meteorological and marine data on limited information regions.

* German Rivillas-Ospina
* Marianella Bolivar
* Mauro Maza Chamorro
* Gabriel Ruiz
* Diego Casas
* Roberto Guerrero

This application is composed of two main modules: the first module allows the downloading of
information from the database (NOAA - WW3); the second module uses the principles of statistical
mathematics for the treatment of waves and wind. The importance of this simple application is
based on the free and agile access to meteorological and marine information for a coastal project.
The determination of representative conditions of sea states ultimately will govern the process of
design of coastal and oceanic infrastructure. The analysis of historical time series of local waves
and winds allows the evaluation of average regimes or operational design, the ultimate limit states
or extreme design, and the storms or design by persistence. In spite that the former analysis is a
common task for coastal engineers, the codes generated are seldom shared for public use. In
summary, for operational purposes is useful to have a freeware that can assist in the data processing
for decision making and forcing of the mathematical models that are part of the common practice
of coastal, oceanic and offshore engineering. This application has been tested in the Caribbean area
of Colombia where meteorological and marine information are scarce.

## Dependencies

APPMAR is written in Python 3, and it requires a variety of dependencies to run. We strongly recommend using the Conda package manager to install Python 3 and APPMAR dependencies. You can obtain Conda by installing [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/).

**Note:** APPMAR has been tested only in Python 3.8.2 on Windows 10 with dependencies installed from the conda-forge repository. If you find problems trying to run APPMAR on a different platform o Python version, please open an issue.

In order to install APPMAR dependencies, open Anaconda Prompt (or any terminal with the conda command in its PATH environment variable) and execute the following command to create a new Conda environment called `appmarenv` and install only Python and APPMAR dependencies on it:

```
conda create -y -c conda-forge -n appmarenv python=3 wxpython matplotlib scipy numpy windrose cartopy xarray gdal cfgrib
```

Then you can activate the recently created environment by executing:

```
conda activate appmarenv
```

The last step is to install the `weibull` package. This package is not available on the Anaconda repository or conda-forge, but it can be downloaded from PyPI using pip, the Python package installer:

```
pip install weibull
```

Keep in mind that the previous installation steps are only required for the first time you run APPMAR.
## Run

After installing dependencies, you can run APPMAR 1.0 by navigating to its directory and executing `appmar.py` from a command line with an active Conda environment (the same environment in which you installed the dependencies):

```
cd path/to/appmar/directory
conda activate appmarenv
python appmar.py
```

## Use

First, download data of the grid IDs and year range you want. If you want to download data from both single-grid (Jul 1999 - Nov 2007) and multi-grid (Feb 2005 - present) datasets, you must provide IDs of equivalent grids separated by comma without space (e.g. `wna,at_10m`). GRIB files are downloaded into the `data` subdirectory. The download process usually takes a long time because NOAA's FTP server provides slow download speeds. Once the download is completed, you can navigate to the analysis module and generate short-term and long-term statistical plots. The first time you generate a plot, APPMAR 1.0 caches processed data into the `tmp` subdirectory in order to speed up future plot generation from the same grid/year range. If you want to generate fresh plots, delete the `tmp` subdirectory. Also, if you have problems with APPMAR 1.0 not identifying a new grid ID, delete the `data` subdirectory and download the GRIB files again.
