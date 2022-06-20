import numpy as np

from ._hpgeom import (
    angle_to_pixel,
    pixel_to_angle,
    query_circle,
    nest_to_ring,
    ring_to_nest,
    boundaries,
)

__all__ = [
    'angle_to_pixel',
    'pixel_to_angle',
    'query_circle',
    'query_circle_vec',
    'lonlat_to_thetaphi',
    'nest_to_ring',
    'ring_to_nest',
    'nside_to_npixel',
    'npixel_to_nside',
    'nside_to_pixel_area',
    'nside_to_resolution',
    'nside_to_order',
    'order_to_nside',
    'boundaries',
    'UNSEEN',
]

# To add:
#  query_polygon (with lonlat option as default!)
#  vector_to_pixel
#  pixel_to_vector
#  vector_to_angle (python or c?)
#  angle_to_vector (python or c?)
#  boundaries (with lonlat option as default!)
#  neighbors (with an o)

UNSEEN = -1.6375e+30
max_nside = 1 << 29


def lonlat_to_thetaphi(lon, lat, degrees=True):
    """Convert longitude/latitude to theta/phi.

    Parameters
    ----------
    lon : `np.ndarray` (N,) or `float`
        Longitude array or scalar.
    lat : `np.ndarray` (N,) or `float`
        Latitude array or scalar.
    degrees : `bool`, optional
        If True, longitude and latitude will be in degrees.

    Returns
    -------
    theta : `np.ndarray` (N,) or `float`
        Theta (co-latitude), in radians, array or scalar.
    phi : `np.ndarray` (N,) or `float`
        Phi (longitude), in radians, array or scalar.
    """
    if degrees:
        lon_ = np.deg2rad(lon % 360.)
        lat_ = np.deg2rad(lat)
    else:
        lon_ = lon % (2.*np.pi)
        lat_ = lat

    if np.any((lat_ < -np.pi/2.) | (lat_ > np.pi/2.)):
        raise ValueError("Latitude out of range.")

    theta = np.pi/2. - lat_
    phi = lon_

    return theta, phi


def query_circle_vec(nside, vec, radius, inclusive=False, fact=4, nest=True):
    """Returns pixels whose centers lie within the circle defined by vec
    and radius (in radians) if inclusive is False, or which overlap with
    this circle (if inclusive is True).

    Parameters
    ----------
    nside : `int`
        HEALPix nside. Must be power of 2 for nest ordering.
    vec : iterable with 3 elements, `float`
        The coordinates of the unit vector defining the circle center.
    radius : `float`
        The radius (in radians) of the circle.
    inclusive : `bool`, optional
        If False, return the exact set of pixels whose pixel centers lie
        within the circle. If True, return all pixels that overlap with
        the disk. This is an approximation and may return a few extra
        pixels.
    fact : `int`, optional
        Only used when inclusive=True. The overlap test is performed at
        a resolution fact*nside. For nest ordering, fact must be a power
        of 2, and nside*fact must always be <= 2**29.  For ring ordering
        fact may be any positive integer.
    nest : `bool`, optional
        If True, use nest ordering.

    Returns
    -------
    pixels : `np.ndarray` (N,)
        Array of pixels (`np.int64`) which cover the circle.
    """
    if len(vec) != 3:
        raise ValueError("vec must be 3 elements.")

    norm = np.sqrt(np.sum(np.square(vec)))
    theta = np.arccos(vec[2] / norm)
    phi = np.arctan2(vec[1], vec[0])
    phi %= (2*np.pi)

    return query_circle(nside, theta, phi, radius, inclusive=inclusive, fact=fact, nest=nest, lonlat=False)


def nside_to_npixel(nside):
    """Return the number of pixels given an nside.

    Parameters
    ----------
    nside : `int` or `np.ndarray` (N,)
        HEALPix nside

    Returns
    -------
    npixel : `int` or `np.ndarray` (N,)
        Number of pixels associated with that nside.
    """
    _nside = np.atleast_1d(nside)
    if np.any((_nside < 0) | (_nside > max_nside)):
        raise ValueError("Illegal nside value (must be 0 <= nside <= 2**29)")
    return 12*nside*nside


def npixel_to_nside(npixel):
    """Return the nside given a number of pixels.

    Parameters
    ----------
    npixel : `int` or `np.ndarray` (N,)
        Number of pixels.

    Returns
    -------
    nside : `int` or `np.ndarray` (N,)
        HEALPix nside associated with that number of pixels.
    """
    nside = np.sqrt(np.atleast_1d(npixel)/12.0)
    if np.any(nside != np.floor(nside)):
        raise ValueError("Illegal npixel (it must be 12*nside*nside)")

    if len(nside) == 1:
        return int(nside)
    else:
        return nside.astype(np.int64)


def nside_to_pixel_area(nside, degrees=True):
    """Return the pixel area given an nside in square degrees or square radians.

    Parameters
    ----------
    nside : `int`
        HEALPix nside parameter.
    degrees : `bool`, optional
        Return area in square degrees?

    Returns
    -------
    pixel_area : `float`
        Pixel area in square degrees or square radians.
    """
    pixel_area = 4*np.pi/nside_to_npixel(nside)

    if degrees:
        value = np.rad2deg(np.rad2deg(pixel_area))
    else:
        value = pixel_area

    return value


def nside_to_resolution(nside, units='degrees'):
    """Return the approximate resolution (pixel size in radians, arcseconds, arcminutes,
    or degrees) given an nside.

    Resolution is just the square root of the pixel area, which is an approximation
    given the varying pixel shapes.

    Parameters
    ----------
    nside : `int`
        HEALPix nside parameter.
    units : `str`, optional
        Units to return.  Valid options are ``radians``, ``degrees``, ``arcminutes``,
        ``arcseconds``.

    Returns
    -------
    resolution : `float`
        Approximate pixel size in specified units.
    """
    resolution = np.sqrt(nside_to_pixel_area(nside, degrees=False))

    if units == 'radians':
        value = resolution
    elif units == 'degrees':
        value = np.rad2deg(resolution)
    elif units == 'arcminutes':
        value = np.rad2deg(resolution)*60.
    elif units == 'arcseconds':
        value = np.rad2deg(resolution)*60.*60.
    else:
        raise ValueError("Invalid units.  Must be radians, degrees, arcminutes, or arcseconds.")

    return value

def nside_to_order(nside):
    """Return the resolution order for a given nside.

    Parameters
    ----------
    nside : `int`
        HEALPix nside parameter.  Will raise ValueError if nside is not valid
        (must be a power of 2 and less than 2**30).

    Returns
    -------
    order : `int`
        Order corresponding to given nside, such that nside = 2**order.
    """
    _nside = np.atleast_1d(nside)
    if np.any((_nside <= 0) | (_nside > max_nside) | ((_nside & (_nside - 1)) != 0)):
        raise ValueError("nside must be postive power of 2, and less than 2**30")

    if (hasattr(nside, '__len__')):
        return np.round(np.log2(nside)).astype(np.int64)
    else:
        return int(np.round(np.log2(nside)))


def order_to_nside(order):
    """Return the nside for a given order.

    Parameters
    ----------
    order : `int` or `np.ndarray`
        Resolution order.  Will raise ValueError if order is not valid
        (must be 0 to 29 inclusive).

    Returns
    -------
    nside : `int`
        HEALPix nside corresponding to given order, such that nside = 2**order.
    """
    if np.any((order != np.int64(order)) | (order < 0) | (order > 29)):
        raise ValueError("Order must be integer, 0<=order<=29.")

    return 2**order
