`HPGeom`: HEALPix Geometry Routines with Python and Numpy
=========================================================

`HPGeom` is a lightweight implementation of HEALPix geometry functions, wrapped in a numpy interface.
The implementation is based on the geometric functions in the original HEALPix_ C++ library code.
This has an all-new API, but there are compatibility functions with the same interfaces as healpy_ for easy transition from healpy_ to `HPGeom`.

The `HPGeom` package addresses multiple issues in the default healpy_ routines.

* `HPGeom` is lightweight, with no plotting or i/o routines.  Therefore it is fast to load with no dependencies on matplotlib_ or cfitsio_.
* `HPGeom` only depends on python, numpy_ and a working C compiler.

The code is hosted in GitHub_.
Please use the `issue tracker <https://github.com/lsstdesc/hpgeom/issues>`_ to let us know about any problems or questions with the code.
The list of released versions of this package can be found `here <https://github.com/lsstdesc/hpgeom/releases>`_, with the main branch including the most recent (non-released) development.

The `HPGeom` code was written by Eli Rykoff based on HEALPix_ c++ code as well as work by Matt Becker and Erin Sheldon.
This software was developed under the Rubin Observatory Legacy Survey of Space and Time (LSST) Dark Energy Science Collaboration (DESC) using LSST DESC resources.
The DESC acknowledges ongoing support from the Institut National de Physique Nucléaire et de Physique des Particules in France; the Science & Technology Facilities Council in the United Kingdom; and the Department of Energy, the National Science Foundation, and the LSST Corporation in the United States.
DESC uses resources of the IN2P3 Computing Center (CC-IN2P3--Lyon/Villeurbanne - France) funded by the Centre National de la Recherche Scientifique; the National Energy Research Scientific Computing Center, a DOE Office of Science User Facility supported by the Office of Science of the U.S. Department of Energy under Contract No. DE-AC02-05CH11231; STFC DiRAC HPC Facilities, funded by UK BEIS National E-infrastructure capital grants; and the UK particle physics grid, supported by the GridPP Collaboration.
This work was performed in part under DOE Contract DE-AC02-76SF00515.


.. _HEALPix: https://healpix.jpl.nasa.gov/
.. _DESC: https://lsst-desc.org/
.. _healpy: https://healpy.readthedocs.io/en/latest/
.. _GitHub: https://github.com/lsstdesc/hpgeom
.. _matplotlib: https://matplotlib.org/
.. _numpy: https://numpy.org/
.. _cfitsio: https://heasarc.gsfc.nasa.gov/fitsio/

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   basic_interface


Modules API Reference
=====================

.. toctree::
   :maxdepth: 3

   modules

Search
======

* :ref:`search`


.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
