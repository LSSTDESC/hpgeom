from setuptools import setup, Extension
import numpy
import sys


ext = Extension(
    "hpgeom._hpgeom",
    [
        "hpgeom/hpgeom_stack.c",
        "hpgeom/hpgeom_utils.c",
        "hpgeom/healpix_geom.c",
        "hpgeom/hpgeom.c",
    ],
)

extra_link_args = []
if sys.platform == 'win32':
    # Windows: no pthread library needed, use native threads
    pass
else:
    # Linux/macOS: link with pthread
    extra_link_args.append('-pthread')

setup(
    ext_modules=[ext],
    include_dirs=numpy.get_include(),
    extra_link_args=extra_link_args,
)
