import numpy as np

from ._hpgeom import (
    angle_to_pixel,
    pixel_to_angle,
    query_circle,
    query_polygon,
    query_ellipse,
    query_box,
    nest_to_ring,
    ring_to_nest,
    vector_to_pixel,
    pixel_to_vector,
    boundaries,
    neighbors,
    max_pixel_radius,
    get_interpolation_weights,
)

__all__ = [
    'thetaphi_to_lonlat',
    'lonlat_to_thetaphi',
    'angle_to_pixel',
    'pixel_to_angle',
    'query_circle',
    'query_circle_vec',
    'query_polygon',
    'query_polygon_vec',
    'query_ellipse',
    'query_box',
    'lonlat_to_thetaphi',
    'nest_to_ring',
    'ring_to_nest',
    'nside_to_npixel',
    'npixel_to_nside',
    'nside_to_pixel_area',
    'nside_to_resolution',
    'nside_to_order',
    'order_to_nside',
    'angle_to_vector',
    'vector_to_angle',
    'vector_to_pixel',
    'pixel_to_vector',
    'boundaries',
    'neighbors',
    'max_pixel_radius',
    'get_interpolation_weights',
    'reorder',
    'UNSEEN',
]

UNSEEN = -1.6375e+30
max_nside = 1 << 29


def lonlat_to_thetaphi(lon, lat, degrees=True):
    """Convert longitude/latitude to theta/phi.

    Parameters
    ----------
    lon, lat : `float` or `np.ndarray` (N,)
        Longitude/latitude.
    degrees : `bool`, optional
        If True, longitude and latitude will be in degrees.

    Returns
    -------
    theta, phi : `float` or `np.ndarray` (N,)
        Co-latitude (theta) and longitude (phi) in radians.
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


def thetaphi_to_lonlat(theta, phi, degrees=True):
    """Convert theta/phi to longitude/latitude.

    Parameters
    ----------
    theta, phi : `float` or `np.ndarray` (N,)
        Co-latitude (theta) and longitude (phi) in radians.
    degrees : `bool`, optional
        If True, longitude and latitude will be in degrees.

    Returns
    -------
    lon, lat : `float` or `np.ndarray` (N,)
        Longitude/latitude.
    """
    lon = phi
    lat = -(theta - np.pi/2.)
    if (degrees):
        lon = np.rad2deg(lon)
        lat = np.rad2deg(lat)

    return lon, lat


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

    theta, phi = vector_to_angle(vec, lonlat=False)

    return query_circle(
        nside,
        theta[0],
        phi[0],
        radius,
        inclusive=inclusive,
        fact=fact,
        nest=nest,
        lonlat=False,
    )


def query_polygon_vec(nside, vertices, inclusive=False, fact=4, nest=True):
    """Returns pixels whose centers lie within the convex polygon defined
    by vertices (if inclusive is False), or which overlap with the polygon
    (if inclusive is True).

    Parameters
    ----------
    nside : `int`
        HEALPix nside. Must be power of 2 for nest ordering.
    vertices : `np.ndarray` (N, 3)
        Vertex array containing the vertices of the polygon.
    inclusive : `bool`, optional
        If False, return the exact set of pixels whose pixel centers lie
        within the polygon; if True, return all pixels that overlap with the polygon,
        and maybe a few more.
    fact : `int`, optional
        Only used when inclusive=True. The overlapping test will be done at
        the resolution fact*nside. For NESTED ordering, fact must be a power of 2, less than 2**30,
        else it can be any positive integer.
    nest: `bool`, optional
        Use nest ordering scheme?
    """
    theta, phi = vector_to_angle(vertices, lonlat=False)

    return query_polygon(nside, theta, phi, inclusive=inclusive, fact=fact, nest=nest, lonlat=False)


def nside_to_npixel(nside):
    """Return the number of pixels given an nside.

    Parameters
    ----------
    nside : `int` or `np.ndarray` (N,)
        HEALPix nside.

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
    if (np.any(npixel <= 0)):
        raise ValueError("Illegal npixel (must be positive)")
    nside = np.sqrt(np.atleast_1d(npixel)/12.0)
    if np.any(nside != np.floor(nside)):
        raise ValueError("Illegal npixel (it must be 12*nside*nside)")

    if len(nside) == 1:
        return int(nside[0])
    else:
        return nside.astype(np.int64)


def nside_to_pixel_area(nside, degrees=True):
    """Return the pixel area given an nside in square degrees or square radians.

    Parameters
    ----------
    nside : `int`
        HEALPix nside parameter.
    degrees : `bool`, optional
        Return area in square degrees? Otherwise square radians.

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
    nside : `int` or `np.ndarray` (N,)
        HEALPix nside parameter.

    Returns
    -------
    order : `int` or `np.ndarray` (N,)
        Order corresponding to given nside, such that nside = 2**order.

    Raises
    ------
    ValueError if nside is not valid (must be power of 2 and less than 2**30).
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
    order : `int` or `np.ndarray` (N,)
        Resolution order.  Will raise ValueError if order is not valid
        (must be 0 to 29 inclusive).

    Returns
    -------
    nside : `int` or `np.ndarray` (N,)
        HEALPix nside corresponding to given order, such that nside = 2**order.
    """
    if np.any((order != np.int64(order)) | (order < 0) | (order > 29)):
        raise ValueError("Order must be integer, 0<=order<=29.")

    return 2**order


def _check_theta_phi(theta, phi):
    """Check that theta/phi are valid.

    Parameters
    ----------
    theta : `float` or `np.ndarray` (N,)
        Co-latitude in radians.
    phi : `float` or `np.ndarray` (N,)
        Longitude in radians.

    Raises
    ------
    ValueError if theta or phi is invalid.
    """
    _theta = np.atleast_1d(theta)
    _phi = np.atleast_1d(phi)
    if np.any((_theta < 0.0) | (_theta > np.pi)):
        raise ValueError("Co-latitude (theta) out of range.")
    if np.any((_phi < 0.0) | (_phi > 2*np.pi)):
        raise ValueError("Longitude (phi) out of range.")


def angle_to_vector(a, b, lonlat=True, degrees=True):
    """Convert angles to cartesion (x, y, z) unit vectors.

    Parameters
    ----------
    a, b : `float` or `np.ndarray` (N,)
        Longitude/latitude (if lonlat=True) or Co-latitude(theta)/longitude(phi)
        (if lonlat=False). Longitude/latitude will be in degrees if degrees=True
        and in radians if degrees=False. Theta/phi are always in radians.
    lonlat : `bool`, optional
        Use longitude/latitude instead of co-latitude/longitude (radians).
    degrees : `bool`, optional
        If lonlat is True then this sets if the units are degrees or
        radians.

    Returns
    -------
    vec : `np.ndarray` (N, 3) or (3,)
        If a, b are vectors, returns a 2D array with dimensions (N, 3) with
        one vector per row.  If not vectors, a 1D array with dimensions (3,).
    """
    if lonlat:
        theta, phi = lonlat_to_thetaphi(a, b, degrees=degrees)
    else:
        _check_theta_phi(a, b)
        theta = a
        phi = b

    sin_theta = np.sin(theta)
    return np.array([sin_theta * np.cos(phi), sin_theta * np.sin(phi), np.cos(theta)]).transpose()


def vector_to_angle(vec, lonlat=True, degrees=True):
    """Convert cartesian (x, y, z) vectors to angles.

    Parameters
    ----------
    vec : `np.ndarray` (3,) or (N, 3)
        The vectors to convert to angles.
    lonlat : `bool`, optional
        Use longitude/latitude for a, b instead of co-latitude/longitude.
    degrees : `bool`, optional
        If lonlat=True then this sets if the units are degrees or radians.

    Returns
    -------
    a, b : `float` or `np.ndarray` (N,)
        Longitude/latitude (if lonlat=True) or Co-latitude(theta)/longitude(phi)
        (if lonlat=False). Longitude/latitude will be in degrees if degrees=True
        and in radians if degrees=False. Theta/phi are always in radians.
    """
    vec = np.atleast_1d(vec).reshape(-1, 3)
    norm = np.sqrt(np.sum(np.square(vec), axis=1))
    theta = np.arccos(vec[:, 2]/norm)
    phi = np.arctan2(vec[:, 1], vec[:, 0]) % (2.*np.pi)
    if lonlat:
        return thetaphi_to_lonlat(theta, phi, degrees=degrees)
    else:
        return theta, phi


def reorder(map_in, ring_to_nest=True):
    """Reorder the pixels in a map from ring to nest or nest to ring ordering.

    Parameters
    ----------
    map_in : `np.ndarray`
        Input map to reorder.
    ring_to_nest : `bool`, optional
        If True, convert ring ordering to nest ordering. If False, convert nest
        ordering to ring ordering.

    Returns
    -------
    map_out : `np.ndarray`
        Reordered map.
    """
    from ._hpgeom import ring_to_nest as convert_ring_to_nest
    from ._hpgeom import nest_to_ring as convert_nest_to_ring

    # Find the nside
    npix = map_in.size
    nside = npixel_to_nside(npix)
    # Confirm that the nside is legal for nest ordering
    _ = nside_to_order(nside)
    if (nside > 128):
        groupsize = npix // 24
    else:
        groupsize = npix

    map_out = np.zeros_like(map_in)
    for group in range(npix // groupsize):
        pixels = np.arange(group*groupsize, (group + 1)*groupsize)
        if ring_to_nest:
            map_out[convert_ring_to_nest(nside, pixels)] = map_in[pixels]
        else:
            map_out[convert_nest_to_ring(nside, pixels)] = map_in[pixels]

    return map_out
