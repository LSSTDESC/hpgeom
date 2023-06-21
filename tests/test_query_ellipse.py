import numpy as np
import pytest
import warnings

import hpgeom

from utils import match_arrays


@pytest.mark.parametrize("nside_radius", [(2**5, 2.0),
                                          (2**10, 1.0),
                                          (2**20, 0.01)])
@pytest.mark.parametrize("lon", [0.0, 90.0, 180.0, 270.0])
@pytest.mark.parametrize("lat", [-45.0, 0.0, 45.0, 90.0, -90.0])
def test_query_ellipse_circle_nest(nside_radius, lon, lat):
    """Test query_ellipse with circles, nest ordering."""
    nside = nside_radius[0]
    radius = nside_radius[1]

    # First, non-inclusive
    pixels_ellipse = hpgeom.query_ellipse(nside, lon, lat, radius, radius, 0.0)
    pixels_circle = hpgeom.query_circle(nside, lon, lat, radius)

    np.testing.assert_array_equal(pixels_ellipse, pixels_circle)

    # Second, inclusive.
    pixels_ellipse = hpgeom.query_ellipse(nside, lon, lat, radius, radius, 0.0, inclusive=True)
    pixels_circle = hpgeom.query_circle(nside, lon, lat, radius, inclusive=True)

    np.testing.assert_array_equal(pixels_ellipse, pixels_circle)


@pytest.mark.parametrize("nside_radius", [(2**5, 2.0),
                                          (2**10, 1.0),
                                          (2**20, 0.01)])
@pytest.mark.parametrize("lon", [0.0, 90.0, 180.0, 270.0])
@pytest.mark.parametrize("lat", [-45.0, 0.0, 45.0, 90.0, -90.0])
def test_query_ellipse_circle_ring(nside_radius, lon, lat):
    """Test query_ellipse with circles, ring ordering."""
    nside = nside_radius[0]
    radius = nside_radius[1]

    # First, non-inclusive
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pixels_ellipse = hpgeom.query_ellipse(nside, lon, lat, radius, radius, 0.0, nest=False)

    pixels_circle = hpgeom.query_circle(nside, lon, lat, radius, nest=False)

    np.testing.assert_array_equal(pixels_ellipse, pixels_circle)

    # Second, inclusive.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pixels_ellipse = hpgeom.query_ellipse(
            nside,
            lon,
            lat,
            radius,
            radius,
            0.0,
            inclusive=True,
            nest=False
        )

    pixels_circle = hpgeom.query_circle(nside, lon, lat, radius, inclusive=True, nest=False)

    np.testing.assert_array_equal(pixels_ellipse, pixels_circle)


def _pos_in_ellipse(lon, lat, lon_0, lat_0, major, minor, alpha_deg):
    """Get which positions are in an ellipse.

    Following
    https://math.stackexchange.com/questions/3747965/points-within-an-ellipse-on-the-globe

    The sign of alpha has been reversed from the equations posted there so that it is
    defined as the angle East (clockwise) of North.

    Parameters
    ----------
    lon, lat : `np.ndarray`
    lon_0, lat_0 : `float`
    major, minor : `float`
    alpha_deg : `float`

    Returns
    -------
    indices : `np.ndarray`
    """
    theta, phi = hpgeom.lonlat_to_thetaphi(lon, lat)
    theta_0, phi_0 = hpgeom.lonlat_to_thetaphi(lon_0, lat_0)
    vec = hpgeom.angle_to_vector(lon, lat)

    a = np.deg2rad(major)
    b = np.deg2rad(minor)

    gamma = np.sqrt(a**2. - b**2.)
    alpha = np.deg2rad(alpha_deg)

    F1_x = (np.cos(alpha)*np.sin(gamma)*np.cos(phi_0)*np.cos(theta_0)
            + np.sin(alpha)*np.sin(gamma)*np.sin(phi_0)
            + np.cos(gamma)*np.cos(phi_0)*np.sin(theta_0))
    F2_x = (-np.cos(alpha)*np.sin(gamma)*np.cos(phi_0)*np.cos(theta_0)
            - np.sin(alpha)*np.sin(gamma)*np.sin(phi_0)
            + np.cos(gamma)*np.cos(phi_0)*np.sin(theta_0))
    F1_y = (np.cos(alpha)*np.sin(gamma)*np.sin(phi_0)*np.cos(theta_0)
            - np.sin(alpha)*np.sin(gamma)*np.cos(phi_0)
            + np.cos(gamma)*np.sin(phi_0)*np.sin(theta_0))
    F2_y = (-np.cos(alpha)*np.sin(gamma)*np.sin(phi_0)*np.cos(theta_0)
            + np.sin(alpha)*np.sin(gamma)*np.cos(phi_0)
            + np.cos(gamma)*np.sin(phi_0)*np.sin(theta_0))
    F1_z = np.cos(gamma)*np.cos(theta_0) - np.cos(alpha)*np.sin(gamma)*np.sin(theta_0)
    F2_z = np.cos(gamma)*np.cos(theta_0) + np.cos(alpha)*np.sin(gamma)*np.sin(theta_0)

    cos_d1 = vec[:, 0]*F1_x + vec[:, 1]*F1_y + vec[:, 2]*F1_z
    cos_d2 = vec[:, 0]*F2_x + vec[:, 1]*F2_y + vec[:, 2]*F2_z

    inside = ((np.arccos(cos_d1) + np.arccos(cos_d2)) < 2*a)

    return inside


@pytest.mark.parametrize("nside_major_minor", [(2**5, 2.0, 1.0),
                                               (2**10, 1.0, 0.7),
                                               (2**20, 0.01, 0.009)])
@pytest.mark.parametrize("alpha", [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])
@pytest.mark.parametrize("lon", [0.0, 90.0, 180.0, 270.0])
@pytest.mark.parametrize("lat", [-45.0, 0.0, 45.0, 90.0, -90.0])
def test_query_ellipse_nest_exclusive(nside_major_minor, alpha, lon, lat):
    """Test query_ellipse with ellipses, nest ordering."""
    nside = nside_major_minor[0]
    major = nside_major_minor[1]
    minor = nside_major_minor[2]

    # First, non-inclusive
    pixels_ellipse = hpgeom.query_ellipse(nside, lon, lat, major, minor, alpha)
    pixels_circle = hpgeom.query_circle(nside, lon, lat, major*1.05)
    lon_pix, lat_pix = hpgeom.pixel_to_angle(nside, pixels_circle)
    cut = _pos_in_ellipse(lon_pix, lat_pix, lon, lat, major, minor, alpha)
    pixels_circle_ellipse = pixels_circle[cut]

    np.testing.assert_array_equal(pixels_ellipse, pixels_circle_ellipse)


# The low-res nside=32 tests inclusive are ... inconclusive ... so we skip them.
@pytest.mark.parametrize("nside_major_minor", [(2**10, 1.0, 0.7),
                                               (2**20, 0.01, 0.009)])
@pytest.mark.parametrize("alpha", [0.0, 45.0, 135.0, 225.0, 315.0])
@pytest.mark.parametrize("lon", [0.0, 90.0, 180.0, 270.0])
@pytest.mark.parametrize("lat", [-45.0, 0.0, 45.0, 90.0, -90.0])
def test_query_ellipse_nest_inclusive(nside_major_minor, alpha, lon, lat):
    """Test query_ellipse with ellipses, nest ordering."""
    nside = nside_major_minor[0]
    major = nside_major_minor[1]
    minor = nside_major_minor[2]

    pixels = hpgeom.query_ellipse(nside, lon, lat, major, minor, alpha)
    pixels_ellipse = hpgeom.query_ellipse(nside, lon, lat, major, minor, alpha, inclusive=True)

    # Ensure all the inner pixels are in the inclusive pixels
    sub1, sub2 = match_arrays(pixels_ellipse, pixels)
    assert sub2.size == pixels.size

    # Look at the boundaries of the pixels, check if any are included.
    pixels_circle = hpgeom.query_circle(nside, lon, lat, major*1.1)
    boundaries_lon, boundaries_lat = hpgeom.boundaries(nside, pixels_circle, step=4)
    cut = _pos_in_ellipse(boundaries_lon.ravel(), boundaries_lat.ravel(), lon, lat, major, minor, alpha)
    test = cut.reshape(boundaries_lon.shape).sum(axis=1)
    pixels_circle_ellipse = pixels_circle[test > 0]

    # Ensure all these pixels are in the inclusive list
    sub1, sub2 = match_arrays(pixels_circle_ellipse, pixels_ellipse)
    assert sub1.size == pixels_circle_ellipse.size


def test_query_ellipse_radians():
    """Test query_ellipse, use lonlat and radians."""
    nside = 1024
    major = 2.0
    minor = 1.0
    alpha = 55.0
    lon = 0.0
    lat = 0.0

    pixels_deg = hpgeom.query_ellipse(nside, lon, lat, major, minor, alpha)
    pixels_rad = hpgeom.query_ellipse(
        nside,
        np.deg2rad(lon),
        np.deg2rad(lat),
        np.deg2rad(major),
        np.deg2rad(minor),
        np.deg2rad(alpha),
        degrees=False
    )

    np.testing.assert_array_equal(pixels_rad, pixels_deg)


def test_query_ellipse_thetaphi():
    """Test query_ellipse, use theta_phi."""
    nside = 1024
    major = 2.0
    minor = 1.0
    alpha = -55.0
    lon = 45.0
    lat = 45.0

    pixels_deg = hpgeom.query_ellipse(nside, lon, lat, major, minor, alpha)
    theta, phi = hpgeom.lonlat_to_thetaphi(lon, lat)
    pixels_thetaphi = hpgeom.query_ellipse(
        nside,
        theta,
        phi,
        np.deg2rad(major),
        np.deg2rad(minor),
        np.deg2rad(alpha),
        lonlat=False
    )

    np.testing.assert_array_equal(pixels_thetaphi, pixels_deg)


@pytest.mark.parametrize("fact", [1, 2, 4, 8])
def test_query_ellipse_fact(fact):
    """Test query_ellipse, use other fact values."""
    nside = 1024
    lon = 90.0
    lat = 20.0
    major = 2.0
    minor = 1.0
    alpha = 75.0

    pixels_ellipse = hpgeom.query_ellipse(nside, lon, lat, major, minor, alpha, inclusive=True, fact=fact)

    # Look at the boundaries of the pixels, check if any are included.
    pixels_circle = hpgeom.query_circle(nside, lon, lat, major*1.1)
    boundaries_lon, boundaries_lat = hpgeom.boundaries(nside, pixels_circle, step=4)
    cut = _pos_in_ellipse(boundaries_lon.ravel(), boundaries_lat.ravel(), lon, lat, major, minor, alpha)
    test = cut.reshape(boundaries_lon.shape).sum(axis=1)
    pixels_circle_ellipse = pixels_circle[test > 0]

    # We don't get 100% overlap, unfortunately, so we have to look above a threshold
    sub1 = np.searchsorted(pixels_ellipse, pixels_circle_ellipse)
    bad, = (sub1 == pixels_ellipse.size).nonzero()
    sub1[bad] = pixels_ellipse.size - 1
    sub2, = (pixels_ellipse[sub1] == pixels_circle_ellipse).nonzero()
    sub1 = sub1[sub2]

    assert sub1.size >= int(0.9*pixels_ellipse.size)
    assert sub1.size >= int(0.9*pixels_circle_ellipse.size)


def test_query_ellipse_badinputs():
    """Test query ellipse with bad inputs."""
    with pytest.raises(ValueError, match=r"lat .* out of range"):
        # Latitude out of range
        hpgeom.query_ellipse(2048, 0.0, 100.0, 1.0, 0.5, 0.0)

    with pytest.raises(ValueError, match=r"lat .* out of range"):
        # Latitude out of range
        hpgeom.query_ellipse(2048, 0.0, -100.0, 1.0, 0.5, 0.0)

    with pytest.raises(ValueError, match=r"colatitude \(theta\) .* out of range"):
        # theta out of range
        hpgeom.query_ellipse(2048, -0.1, 0.0, 0.1, 0.05, 0.0, lonlat=False)

    with pytest.raises(ValueError, match=r"colatitude \(theta\) .* out of range"):
        # theta out of range
        hpgeom.query_ellipse(2048, np.pi + 0.1, 0.0, 0.1, 0.05, 0.0, lonlat=False)

    with pytest.raises(ValueError, match=r"longitude \(phi\) .* out of range"):
        # phi out of range
        hpgeom.query_ellipse(2048, 0.0, -0.1, 0.1, 0.05, 0.0, lonlat=False)

    with pytest.raises(ValueError, match=r"longitude \(phi\) .* out of range"):
        # phi out of range
        hpgeom.query_ellipse(2048, 0.0, 2*np.pi + 0.1, 0.1, 0.05, 0.0, lonlat=False)

    with pytest.raises(ValueError, match=r"Semi-major axis must be positive"):
        # Semi-major out of range
        hpgeom.query_ellipse(2048, 0.0, 0.0, 0.0, 0.1, 0.0)

    with pytest.raises(ValueError, match=r"Semi-minor axis must be positive"):
        # Semi-major out of range
        hpgeom.query_ellipse(2048, 0.0, 0.0, 0.2, 0.0, 0.0)

    with pytest.raises(ValueError, match=r"Semi-major axis must be >= semi-minor axis"):
        # Semi-major out of range
        hpgeom.query_ellipse(2048, 0.0, 0.0, 0.2, 0.3, 0.0)

    with pytest.raises(ValueError, match=r"Inclusive factor .* must be positive"):
        # Illegal fact (must be positive)
        hpgeom.query_ellipse(2048, 0.0, 0.0, 1.0, 0.5, 0.0, inclusive=True, fact=0)

    with pytest.raises(ValueError, match=r"Inclusive factor \* nside must be \<\="):
        # Illegal fact (too large)
        hpgeom.query_ellipse(2**28, 0.0, 0.0, 0.0001, 0.00005, 0.0, inclusive=True, fact=4)

    with pytest.raises(ValueError, match=r"Inclusive factor .* must be power of 2 for nest"):
        # Illegal fact (must be power of 2 for nest)
        hpgeom.query_ellipse(2048, 0.0, 0.0, 1.0, 0.5, 0.0, inclusive=True, fact=3)

    # Different platforms have different strings here, but they all say ``integer``.
    with pytest.raises(TypeError, match=r"integer"):
        # Illegal fact (must be integer)
        hpgeom.query_ellipse(2048, 0.0, 0.0, 1.0, 0.5, 0.0, inclusive=True, nest=False, fact=3.5)

    # Check resource warning
    with pytest.warns(ResourceWarning, match=r"natively supports nest ordering"):
        hpgeom.query_ellipse(1024, 0.0, 1.0, 0.5, 0.5, 0.0, nest=False)
