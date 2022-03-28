[![DOI:10.1016/j.cageo.2022.105098](https://zenodo.org/badge/DOI/10.1016/j.cageo.2022.105098.svg)](https://doi.org/10.1016/j.cageo.2022.105098)


# APPMAR 1.0

*If you prefer to extract raw data series for your own analyses, try [APPMAR 2](https://github.com/cemanetwork/appmar2) (WIP).*

A Python application for downloading and analyzing of WAVEWATCH III® wave and wind data.

German Rivillas-Ospina, Diego Casas, Mauro Antonio Maza-Chamorro, Marianella Bolívar, Gabriel Ruiz, Roberto Guerrero,
José M. Horrillo-Caraballo, Milton Guerrero, Karina Díaz, Roberto del Rio, Erick Campos

Highlights:

* Free and open-source Python application for wave and wind climate analysis.
* Downloads data from WAVEWATCH III hindcast.
* Performs mean and extreme climate analysis.
* Provides a GUI to improve interaction with the user.
* Especially useful on regions of limited data availability.

APPMAR 1.0 is an application written in the Python programming language that downloads, processes, and analyzes wind and wave data. This application is composed of a graphical user interface (GUI) that contains two main modules: the first module downloads data from WAVEWATCH III® (WW3) production hindcasts by the National Oceanic and Atmospheric Administration (NOAA); the second module applies statistical mathematics for processing and analyzing wave and wind data. This application provides useful graphical results that describe mean and extreme wave and wind climate. APPMAR generates plots of exceedance probability, joint probability distribution, wave direction, Weibull distribution, and storm frequency analysis. Currently, APPMAR only downloads and analyzes wave and wind data from WW3 hindcasts, but it is under development to other datasets and marine climate parameters (see [APPMAR 2](https://github.com/cemanetwork/appmar2)). This application has been tested in the Magdalena River mouth, Colombia, and Cancún, México, where observational wave and wind data are scarce.

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
