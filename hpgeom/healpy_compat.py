import warnings

from ._hpgeom import (
    angle_to_pixel,
    pixel_to_angle,
    nest_to_ring,
    ring_to_nest,
    neighbors,
    vector_to_pixel,
    pixel_to_vector,
    max_pixel_radius,
    get_interpolation_weights,
)
from .hpgeom import (
    query_circle_vec,
    query_polygon_vec,
    nside_to_npixel,
    npixel_to_nside,
    nside_to_pixel_area,
    nside_to_resolution,
    nside_to_order,
    order_to_nside,
    angle_to_vector,
    vector_to_angle,
    UNSEEN,
)

from .hpgeom import boundaries as hpgeom_boundaries

__all__ = [
    'ang2pix',
    'pix2ang',
    'query_disc',
    'query_polygon',
    'ring2nest',
    'nest2ring',
    'nside2npix',
    'npix2nside',
    'nside2pixarea',
    'nside2resol',
    'nside2order',
    'order2nside',
    'ang2vec',
    'vec2ang',
    'pix2vec',
    'vec2pix',
    'boundaries',
    'get_all_neighbours',
    'max_pixrad',
    'get_interp_weights',
    'UNSEEN',
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
        and maybe a few more.
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


def query_polygon(nside, vertices, inclusive=False, fact=4, nest=False, buff=None):
    """Returns pixels whose centers lie within the polygon defined by
    *vertices* (if *inclusive* is False), or which overlap with the polygon
    (if *inclusive* is True).

    Parameters
    ----------
    nside : `int`
        The HEALPix nside parameter.
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
    pixels = query_polygon_vec(nside, vertices, inclusive=inclusive, fact=fact, nest=nest)

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


def ang2vec(theta, phi, lonlat=False):
    """Convert angles to cartesian unit vectors.

    Parameters
    ----------
    theta : `float` or `np.ndarray` (N,)
        Co-latitude (theta, radians) or longitude (degrees).
    phi : `float` or `np.ndarray` (N,)
        Longitude (phi, radians) or latitude (degrees).
    lonlat : `bool`, optional
        Use longitude/latitude (degrees) instead of longitude/co-latitude (radians).

    Returns
    -------
    vec : `np.ndarray` (N, 3) or (3,)
        If theta, phi are vectors, returns a 2D array with dimensions (N, 3) with
        one vector per row.  If not vectors, a 1D array with dimensions (3,).
    """
    return angle_to_vector(theta, phi, lonlat=lonlat, degrees=True)


def vec2ang(vectors, lonlat=False):
    """Convert cartesian (x, y, z) vectors to angles.

    Parameters
    ----------
    vec : `np.ndarray` (3,) or (N, 3)
        The vectors to convert to angles.
    lonlat : `bool`, optional
        Use longitude/latitude instead of co-latitude/longitude (radians).
    degrees : `bool`, optional
        If lonlat is True then this sets if the units are degrees or
        radians.

    Returns
    -------
    theta : `float` or `np.ndarray` (N,)
        Longitude or co-latitude theta (radians if lonlat=False, degrees if
        lonlat=True and degrees=True).
    phi : `float` or `np.ndarray` (N,)
        Latitude or longitude phi (radians if lonlat=False, degrees if lonlat=True
        and degrees=True).
    """
    return vector_to_angle(vectors, lonlat=lonlat, degrees=True)


def boundaries(nside, pix, step=1, nest=False):
    """Returns an array containing vectors to the boundary of the
    given pixel.

    The returned array has shape (3, 4*step) (for a single pixel) or
    (N, 3, 4*step) for multiple pixels.  The elements are the x,y,z
    positions on the unit sphere of the pixel boundary. To retrieve
    corners, specify step=1.

    Parameters
    ----------
    nside : `int` or `np.ndarray` (N,)
        HEALPix nside.  Must be power of 2 for nest ordering.
    pix : `int` or `np.ndarray` (N,)
        Pixel number(s).
    step : `int`, optional
        Number of steps for each side of the pixel.
    nest : `bool`, optional
        Use nest ordering scheme?

    Returns
    -------
    boundary : `np.ndarray` (N, 3, 4*step)
        x,y,z for positions on the boundary of the pixel.
    """
    theta, phi = hpgeom_boundaries(nside, pix, step=step, nest=nest)

    if theta.ndim == 1:
        # Single pixel
        return angle_to_vector(theta, phi).transpose()
    else:
        # Multiple pixels
        return angle_to_vector(theta, phi).transpose([1, 2, 0])


def pix2vec(nside, pix, nest=False):
    """Convert pixels to cartesian vectors (x, y, z).

    Parameters
    ----------
    nside : `int` or `np.ndarray` (N,)
        HEALPix nside.  Must be power of 2 for nest ordering.
    pix : `int` or `np.ndarray` (N,)
        Pixel number(s).
    nest : `bool`, optional
        Use nest ordering scheme?

    Returns
    -------
    x : `np.ndarray` (N,)
        x coordinates of vectors.
    y : `np.ndarray` (N,)
        y coordinates of vectors.
    z : `np.ndarray` (N,)
        z coordinates of vectors.
    """
    return pixel_to_vector(nside, pix, nest=nest)


def vec2pix(nside, x, y, z, nest=False):
    """Convert cartesian vectors (x, y, z) to pixels.

    Parameters
    ----------
    nside : `int` or `np.ndarray` (N,)
        HEALPix nside.  Must be power of 2 for nest ordering.
    x : `np.ndarray` (N,)
        x coordinates of vectors.
    y : `np.ndarray` (N,)
        y coordinates of vectors.
    z : `np.ndarray` (N,)
        z coordinates of vectors.
    nest : `bool`, optional
        Use nest ordering scheme?

    Returns
    -------
    pix : `int` or `np.ndarray` (N,)
        Pixel number(s).
    """
    return vector_to_pixel(nside, x, y, z, nest=nest)


def get_all_neighbours(nside, theta, phi=None, nest=False, lonlat=False):
    """Return the 8 nearest neighbor pixels.

    Parameters
    ----------
    nside : `int`
        HEALPix nside. Must be power of 2 for nest ordering.
    theta : `np.ndarray` (N,) or `float` or `int`
        If phi is None, this is interpreted as integer pixel numbers.
        If phi is not None, then this is interpreted as angular coordinate
        theta (if lonlat=False) or longitude (if lonlat=True).
    phi : `np.ndarray` (N,) or `float`, optional
        Angular coordinate phi (if lonlat=False) or latitude (if lonlat=True).
    nest : `bool`, optional
        Use nest ordering scheme?
    lonlat : `bool`, optional
        Use longitude/latitude (degrees) instead of co-latitude/longitude (radians)

    Returns
    -------
    pixels : `np.ndarray` (8,) or (8, N)
        Neighboring pixel numbers of the SW, W, NW, N, NE, E, SE, and S neighbors.
        If a neighbor does not exist (as can be the case for W, N, E, and S), the
        corresponding pixel number will be -1.
    """
    if phi is not None:
        _pix = angle_to_pixel(nside, theta, phi, nest=nest, lonlat=lonlat, degrees=True)
    else:
        _pix = theta

    neigh = neighbors(nside, _pix, nest=nest)

    return neigh.transpose()


def max_pixrad(nside, degrees=False):
    """Compute maximum angular distance between any pixel and its corners.

    Parameters
    ----------
    nside : `int` or `np.ndarray`
        HEALPix nside.
    degrees : `bool`, optional
        If True, returns pixel radius in degrees, otherwise radians.

    Returns
    -------
    radii : `np.ndarray` (N, ) or `float`
        Angular distance(s) in degrees or radians.
    """
    return max_pixel_radius(nside, degrees=degrees)


def get_interp_weights(nside, theta, phi=None, nest=False, lonlat=False):
    """Return the 4 closest pixels and weights for bilinear interpolation.

    Parameters
    ----------
    nside : `int`
        HEALPix nside.
    theta, phi : `float` or `np.ndarray`
        If phi is not given, theta is interpreted as pixel number,
        otherwise theta/phi are angular coordinates.
    nest : `bool`, optional
        Use nest ordering scheme?
    lonlat : `bool`, optional
        Use longitude/latitude (degrees) instead of co-latitude/longitude (radians).

    Returns
    -------
    pixels : `np.ndarray` (4, N)
        Array of pixel neighbors (4 for each input position).
    weights : `np.ndarray` (4, N)
        Array of pixel weights corresponding with pixels.
    """
    if phi is None:
        _theta, _phi = pixel_to_angle(nside, theta, nest=nest, lonlat=False)
        pixels, weights = get_interpolation_weights(nside, _theta, _phi, nest=nest, lonlat=False)
    else:
        pixels, weights = get_interpolation_weights(nside, theta, phi, nest=nest, lonlat=lonlat)

    return pixels.T, weights.T
