# APPMAR 1.0

*If you prefer to extract raw data series for your own analyses, try [APPMAR 2](https://github.com/cemanetwork/appmar2) (WIP).*

A toolbox for management of meteorological and marine data on limited information regions.

* German Rivillas-Ospina
* Diego Casas
* Mauro Maza Chamorro
* Marianella Bolivar
* Gabriel Ruiz
* Roberto Guerrero
* José Horrillo
* Karina Díaz

APPMAR is an application written in the Python programming language. The goal of this program is to perform the downloading, processing, and analysis of wind and wave data on regions of limited information availability. This application is composed of graphical user interface (GUI) that contain two main modules: the first module allows the downloading of information from WAVEWATCH III® (WW3) production hindcasts by the National Oceanic and Atmospheric Administration (NOAA); the second module uses the principles of statistical mathematics for processing and analyzing wave and wind data. This application provides useful graphical results that describe mean and extreme wave and wind climate. APPMAR generates plots of exceedance probability, joint probability distribution, wave direction, Weibull distribution, and storm frequency analysis. Currently, APPMAR only downloads and analyzes wave and wind data from WW3 hindcasts, but it is under active development and can be extended to other datasets and marine climate parameters.

## Installation

APPMAR is written in Python 3.7, and it requires a variety of dependencies to run. We strongly recommend using the Conda package manager to install Python 3.7 and APPMAR dependencies. You can obtain Conda by installing [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/).

**Note:** APPMAR has been tested only in Python >=3.7 on Windows 10 with dependencies installed via conda. If you find problems trying to run APPMAR on a different platform o Python version, please open an issue.

In order to install and run APPMAR, follow these steps:

1. Download and extract APPMAR source code.

2. Open Anaconda Prompt (or any terminal with the conda command in its PATH environment variable).

3. From the command line, navigate to the APPMAR directory (e.g. `C:\Users\user\Desktop\appmar`). Use `cd` or any command available in your OS for the purpose of changing the current directory:

```
cd C:\Users\user\Desktop\appmar
```

4. Use the following command to create a new Conda environment called `appmarenv` with Python 3.7 and APPMAR dependencies:

```
conda env create -f environment.yml
```

5. Activate the recently created environment:

```
conda activate appmarenv
```

6. Now you can start APPMAR by executing:

```
python appmar.py
```

## Run

The next time you want to use APPMAR, only follow steps 2, 3, 5 and 6.

## Use

First, download data of the grid IDs and year range you want. If you want to download data from both single-grid (Jul 1999 - Nov 2007) and multi-grid (Feb 2005 - present) datasets, you must provide IDs of equivalent grids separated by comma without space (e.g. `wna,at_10m`). GRIB files are downloaded into the `data` subdirectory. The download process usually takes a long time because NOAA's FTP server provides slow download speeds.

Once the download is completed, you can navigate to the analysis module and generate short-term and long-term statistical plots. The first time you generate a plot, APPMAR 1.0 caches processed data into the `tmp` subdirectory in order to speed up future plot generation from the same grid/year range.

If you want to generate fresh plots, delete the `tmp` subdirectory. Also, if you have problems with APPMAR 1.0 not identifying a new grid ID, delete the `data` subdirectory and download the GRIB files again.

## Update

When an update is available, download the the new source code and replace the `appmar.py` and `libappmar.py` files. Delete the `tmp` and `__pycache__` directories for changes to take effect the next time you start APPMAR.

The last version of APPMAR implements the k-means clustering method to find representative sea state scenarios. In you have a previous version, download the new `appmar.py` and `libappmar.py` files, and install the `scikit-learn` and `kneed` packages on the APPMAR environment. In order to install the new dependencies, run the following commands from the command lines:

```
conda activate appmarenv
conda install -c conda-forge scikit-learn kneed
```
