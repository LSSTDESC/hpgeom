try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("hpgeom")
except PackageNotFoundError:
    # package is not installed
    pass

from . import hpgeom

from .hpgeom import *
