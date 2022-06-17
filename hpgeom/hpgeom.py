import numpy as np

from ._hpgeom import (
    angle_to_pixel,
    pixel_to_angle,
    query_circle,
    nest_to_ring,
    ring_to_nest,
)

__all__ = [
    'angle_to_pixel',
    'pixel_to_angle',
    'query_circle',
    'query_circle_vec',
    'lonlat_to_thetaphi',
    'nest_to_ring',
    'ring_to_nest',
    'UNSEEN',
]

# To add:
#  nside_to_npixel
#  npixel_to_nside
#  query_polygon (with lonlat option)
#  nside_to_pixarea
#  nside_to_order
#  nside_to_resolution
#  vector_to_pixel
#  vector_to_angle
#  angle_to_pixel
#  angle_to_vector
#  boundaries (with lonlat option)

UNSEEN = -1.6375e+30


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
