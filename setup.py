from setuptools import setup, Extension
import numpy
import sys


libraries = []
if sys.platform == 'win32':
    # Windows: no pthread library needed, use native threads
    pass
else:
    # Linux/macOS: link with pthread
    libraries.append('pthread')


ext = Extension(
    "hpgeom._hpgeom",
    [
        "hpgeom/hpgeom_stack.c",
        "hpgeom/hpgeom_utils.c",
        "hpgeom/healpix_geom.c",
        "hpgeom/hpgeom.c",
    ],
    include_dirs=[numpy.get_include()],
    libraries=libraries,
)


setup(
    ext_modules=[ext],
)
