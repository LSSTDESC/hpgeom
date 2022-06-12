from setuptools import setup, Extension, find_packages
import numpy


ext = Extension(
    "hpgeom._hpgeom",
    [
        "hpgeom/healpix_geom.c",
        "hpgeom/hpgeom_utils.c",
        "hpgeom/hpgeom.c",
    ],
)

setup(
    name="hpgeom",
    packages=find_packages(),
    version="0.0.1",
    ext_modules=[ext],
    include_dirs=numpy.get_include(),
)
