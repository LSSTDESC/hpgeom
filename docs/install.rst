Install
=======

`HPGeom` requires the following python package and its dependencies:

* `numpy <https://github.com/numpy/numpy>`_

`HPGeom` is available at `pypi <https://pypi.org/project/hpgeom>`_ and `conda-forge <https://anaconda.org/conda-forge/hpgeom>`_.
The most convenient way of installing the latest released version is simply:

.. code-block:: python

  conda install -c conda-forge hpgeom

or

.. code-block:: python

  pip install hpgeom

To install from source, you can run from the root directory (provided you have a working C compiler):

.. code-block:: python

  pip install .

In order to additionally install the requirements for running tests, use:

.. code-block:: python

  pip install .[test,test_with_healpy]
