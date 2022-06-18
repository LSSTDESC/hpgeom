import warnings

from ._hpgeom import angle_to_pixel, pixel_to_angle, nest_to_ring, ring_to_nest
from .hpgeom import (
    query_circle_vec,
    nside_to_npixel,
    npixel_to_nside,
    nside_to_pixel_area,
    nside_to_resolution,
    nside_to_order,
    order_to_nside,
)

__all__ = [
    'ang2pix',
    'pix2ang',
    'query_disc',
    'ring2nest',
    'nest2ring',
    'nside2npix',
    'npix2nside',
    'nside2pixarea',
    'nside2resol',
    'nside2order',
    'order2nside',
]


def ang2pix(nside, theta, phi, nest=False, lonlat=False):
    """Convert angles to pixels, with healpy interface.

    Parameters
    ----------
    nside : `int`
        The HEALPix nside parameter.
    theta : `np.ndarray` (N,) or scalar
        Angular coordinate theta (radians) or longitude (degrees).
    phi : `np.ndarray` (N,) or scalar
        Angular coordinate phi (radians) or latitude (degrees).
    nest : `bool`, optional
        If True, NEST pixel ordering scheme, otherwise RING.
    lonlat : `bool`, optional
        If True, input angles are lon/lat instead of theta/phi.

    Returns
    -------
    pix : `np.ndarray` (N,) or int
        HEALPix pixel numbers.
    """
    return angle_to_pixel(nside, theta, phi, nest=nest, lonlat=lonlat)


def pix2ang(nside, pix, nest=False, lonlat=False):
    """Convert pixels to angles, with healpy interface.

    Parameters
    ----------
    nside : `int`
        The HEALPix nside parameter.
    pix : `np.ndarrat` (N,) or scalar
        Pixel numbers.
    nest : `bool`, optional
        If True, NEST pixel ordering scheme, otherwise RING.
    lonlat : `bool`, optional
        If True, input angles are lon/lat instead of theta/phi.

    Returns
    -------
    theta : `np.ndarray` (N,) or scalar
        Angular coordinate theta (radians) or longitude (degrees).
    phi : `np.ndarray` (N,) or scalar
        Angular coordinate phi (radians) or latitude (degrees).
    """
    return pixel_to_angle(nside, pix, nest=nest, lonlat=lonlat)


def query_disc(nside, vec, radius, inclusive=False, fact=4, nest=False, buff=None):
    """Returns pixels whose centers lie within the disk defined by
    *vec* and *radius* (in radians) (if *inclusive* is False), or which
    overlap with this disk (if *inclusive* is True).

    Parameters
    ----------
    nside : `int`
        The HEALPix nside parameter.
    vec : float, sequence of 3 elements
        The coordinates of unit vector defining the disk center.
    radius : `float`
        The radius (in radians) of the disk
    inclusive : `bool`, optional
        If False, return the exact set of pixels whose pixel centers lie
        within the disk; if True, return all pixels that overlap with the disk,
        and maybe a few more. Default: False
    fact : `int`, optional
        Only used when inclusive=True. The overlapping test will be done at
        the resolution fact*nside. For NESTED ordering, fact must be a power of 2, less than 2**30,
        else it can be any positive integer.
    nest: `bool`, optional
        If True, assume NESTED pixel ordering, otherwise, RING pixel ordering
    buff: `int` array, optional
        If provided, this numpy array is used to contain the return values and must be
        at least long enough to do so.  Note that this is provided for compatibility
        only and is not optimized.

    Returns
    -------
    pixels : `np.ndarray` (N,)
        Array of pixels (`np.int64`) which cover the disk.
    """
    pixels = query_circle_vec(nside, vec, radius, inclusive=inclusive, fact=fact, nest=nest)

    if buff is not None:
        warnings.warn("In hpgeom, setting buff is less performant than simply returning the pixels.")
        buff[0: len(pixels)] = pixels

    return pixels


def nest2ring(nside, pix):
    """Convert pixel number from nest to ring ordering.

    Parameters
    ----------
    nside : `int`, scalar
        The healpix nside parameter.  Must be power of 2.
    pix : `int` or `np.ndarray` (N,)
        The pixel numbers in nest scheme.

    Returns
    -------
    pix : `int` or `np.ndarray` (N,)
        The pixel numbers in ring scheme.
    """
    return nest_to_ring(nside, pix)


def ring2nest(nside, pix):
    """Convert pixel number from ring to nest ordering.

    Parameters
    ----------
    nside : `int`, scalar
        The healpix nside parameter.  Must be power of 2.
    pix : `int` or `np.ndarray` (N,)
        The pixel numbers in ring scheme.

    Returns
    -------
    pix : `int` or `np.ndarray` (N,)
        The pixel numbers in nest scheme.
    """
    return ring_to_nest(nside, pix)


def nside2npix(nside):
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
    return nside_to_npixel(nside)


def npix2nside(nside):
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
    return npixel_to_nside(nside)


def nside2pixarea(nside, degrees=False):
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
    return nside_to_pixel_area(nside, degrees=degrees)


def nside2resol(nside, arcmin=False):
    """Return the approximate resolution (pixel size in radians or arcminutes),
    given an nside.

    Resolution is just the square root of the pixel area, which is an approximation
    given the varying pixel shapes.

    Parameters
    ----------
    nside : `int`
        HEALPix nside parameter.
    arcmin : `bool`, optional
       If True, return resolution in arcminutes, otherwise radians.

    Returns
    -------
    resolution : `float`
        Approximate pixel size in specified units.
    """
    if arcmin:
        units = 'arcminutes'
    else:
        units = 'radians'
    return nside_to_resolution(nside, units=units)


def nside2order(nside):
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
    return nside_to_order(nside)


def order2nside(order):
    """Return the nside for a given order.

    Parameters
    ----------
    order : `int`
        Resolution order.  Will raise ValueError if order is not valid
        (must be 0 to 29 inclusive).

    Returns
    -------
    nside : `int`
        HEALPix nside corresponding to given order, such that nside = 2**order.
    """
    return order_to_nside(order)
