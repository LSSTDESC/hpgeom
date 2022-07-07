# HPGeom
A lightweight implementation of HEALPix geometry functions, wrapped in a numpy interface.

The implementation is based on the geometric functions in the original [HEALPix](https://healpix.jpl.nasa.gov/) C++ library code.
This package has an all-new API, and also includes compatibility functions with the same interface as [healpy](https://healpy.readthedocs.io/en/latest/) for an easy transition from healpy to `HPGeom`.

## Requirements:

At runtime, `HPGeom` requires [numpy](https://github.com/numpy/numpy).

At build time, a working C compiler is also required.

The full suite of tests require [healpy](https://healpy.readthedocs.io/en/latest/).

## Install:

The easiest way to install `HPGeom` is from pypi (`pip install hpgeom`) or from conda-forge (`conda install hpgeom -c conda-forge`).

To install the package from source go to the parent directory of the package and run `pip install .`.
To include all test requirements, install optional dependencies with `pip install .[test,test_with_healpy]`.

## Documentation:

Documentation is available at https://hpgeom.readthedocs.io/en/latest .

## Acknowledgements:

The `HPGeom` code was written by Eli Rykoff based on [HEALPix](https://healpix.jpl.nasa.gov/) C++ code as well as work by Matt Becker and Erin Sheldon.

This software was developed under the Rubin Observatory Legacy Survey of Space and Time (LSST) Dark Energy Science Collaboration (DESC) using LSST DESC resources.
The DESC acknowledges ongoing support from the Institut National de Physique Nucléaire et de Physique des Particules in France; the Science & Technology Facilities Council in the United Kingdom; and the Department of Energy, the National Science Foundation, and the LSST Corporation in the United States.
DESC uses resources of the IN2P3 Computing Center (CC-IN2P3--Lyon/Villeurbanne - France) funded by the Centre National de la Recherche Scientifique; the National Energy Research Scientific Computing Center, a DOE Office of Science User Facility supported by the Office of Science of the U.S. Department of Energy under Contract No. DE-AC02-05CH11231; STFC DiRAC HPC Facilities, funded by UK BEIS National E-infrastructure capital grants; and the UK particle physics grid, supported by the GridPP Collaboration.
This work was performed in part under DOE Contract DE-AC02-76SF00515.
