from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='appmar',
    version='1.0.0',
    packages=find_packages(),
    install_requires=["cfgrib", "gdal", "wxpython", "numpy", "matplotlib==3.2",
                      "scipy", "xarray", "pandas", "cartopy", "scikit-learn", "kneed"],
    python_requires='>=3.7',
    include_package_data=True,
    entry_points={'console_scripts': ['appmar = appmar.appmar:main']},
    author='CEMAN',
    description='Python program for marine climate analysis.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords='marine climate',
    url='https://github.com/cemanetwork/appmar',
    classifiers=['License :: OSI Approved :: MIT License']
)
