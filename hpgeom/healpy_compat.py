from ._hpgeom import angle_to_pixel, pixel_to_angle

__all__ = ['ang2pix', 'pix2ang']


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
