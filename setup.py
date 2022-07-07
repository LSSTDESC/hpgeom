from setuptools import setup, Extension
import numpy


ext = Extension(
    "hpgeom._hpgeom",
    [
        "hpgeom/hpgeom_stack.c",
        "hpgeom/hpgeom_utils.c",
        "hpgeom/healpix_geom.c",
        "hpgeom/hpgeom.c",
    ],
)

setup(
    ext_modules=[ext],
    include_dirs=numpy.get_include(),
)
