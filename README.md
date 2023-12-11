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

**Dependencies:**

- cfgrib
- gdal
- wxpython
- numpy<1.20
- matplotlib==3.2
- scipy
- xarray
- pandas
- cartopy
- scikit-learn
- kneed

In order to install and run APPMAR, follow these steps:

1. Open Anaconda Prompt (or any terminal with the conda command in its PATH environment variable).

2. Create a new conda environment for APPMAR and its dependencies:

```
conda create -n my-new-env -c conda-forge "python>=3.7" cfgrib gdal wxpython numpy<1.20 matplotlib=3.2 scipy xarray pandas cartopy scikit-learn kneed
```

3. Activate the new environment:

```
conda activate my-new-env
```

4. Install APPMAR via pip:

```
pip install appmar
```

5. Now you can start APPMAR by executing:

```
appmar
```

## Run

The next time you want to use APPMAR, only follow steps 1, 3 and 5. More information about APPMAR use and implementation can be found on the article ([DOI: 10.1016/j.cageo.2022.105098](https://doi.org/10.1016/j.cageo.2022.105098)).

## Update

When an update is available, open the Anaconda Prompt, activate your environment (step 3) and upgrade with:

```
pip install --upgrade appmar
```
