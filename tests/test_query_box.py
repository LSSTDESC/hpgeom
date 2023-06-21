import numpy as np
import pytest
import warnings

import hpgeom

from utils import match_arrays


EPSILON = 2.220446049250313e-16


def _pos_in_box(lon, lat, lon0, lon1, lat0, lat1):
    """Check which positions are in a box.

    Parameters
    ----------
    lon, lat : `np.ndarray`
    lon0, lon1, lat0, lat1 : `float`

    Returns
    -------
    indices : `np.ndarray`
    """
    _lon0 = (np.deg2rad(lon0) - EPSILON) % (2.*np.pi)
    _lon1 = np.deg2rad(lon1) % (2.*np.pi)
    _lat0 = np.deg2rad(lat0) - EPSILON
    _lat1 = np.deg2rad(lat1)

    _lon = np.deg2rad(lon)
    _lat = np.deg2rad(lat)

    if _lon0 < _lon1:
        inside = ((_lon >= _lon0) & (_lon < _lon1) & (_lat >= _lat0) & (_lat < _lat1))
    else:
        inside = (((_lon < _lon1) | (_lon >= _lon0)) & (_lat >= _lat0) & (_lat < _lat1))

    return inside


@pytest.mark.parametrize("nside_radius", [(2**7, 2.0),
                                          (2**10, 1.0),
                                          (2**20, 0.01)])
@pytest.mark.parametrize("lon", [0.0, 90.0, 180.0, 270.0])
@pytest.mark.parametrize("lat", [-45.0, 0.0, 45.0])
def test_query_box_square(nside_radius, lon, lat):
    """Test query_box with squares."""
    nside = nside_radius[0]
    radius = nside_radius[1]

    box = [lon - radius, lon + radius, lat - radius, lat + radius]

    pixels = hpgeom.query_box(nside, *box)

    pixels_circle = hpgeom.query_circle(nside, lon, lat, radius*2.5)
    lon_circle, lat_circle = hpgeom.pixel_to_angle(nside, pixels_circle)
    inside = _pos_in_box(lon_circle, lat_circle, *box)

    np.testing.assert_array_equal(pixels, pixels_circle[inside])

    # Do inclusive
    pixels_box = hpgeom.query_box(nside, *box, inclusive=True)

    # Ensure all the inner pixels are in the inclusive pixels
    sub1, sub2 = match_arrays(pixels_box, pixels)
    assert sub2.size == pixels.size

    # Look at boundaries of the pixels, check if any are included.
    pixels_circle = hpgeom.query_circle(nside, lon, lat, radius*2.5)
    boundaries_lon, boundaries_lat = hpgeom.boundaries(nside, pixels_circle, step=4)
    cut = _pos_in_box(boundaries_lon.ravel(), boundaries_lat.ravel(), *box)
    test = cut.reshape(boundaries_lon.shape).sum(axis=1)
    pixels_circle_box = pixels_circle[test > 0]

    # Ensure all these pixels are in the inclusive list
    sub1, sub2 = match_arrays(pixels_circle_box, pixels_box)
    assert sub1.size == pixels_circle_box.size


@pytest.mark.parametrize("nside_radius", [(2**7, 2.0),
                                          (2**10, 1.0),
                                          (2**20, 0.01)])
@pytest.mark.parametrize("lon", [0.0, 360.0])
@pytest.mark.parametrize("lat", [-45.0, 0.0, 45.0])
def test_query_box_edge(nside_radius, lon, lat):
    """Test query_box at the longitude edges."""
    nside = nside_radius[0]
    radius = nside_radius[1]

    if (lon == 0.0):
        box = [lon, lon + 2*radius, lat - radius, lat + radius]
    else:
        box = [lon - 2*radius, lon, lat - radius, lat + radius]

    pixels = hpgeom.query_box(nside, *box)

    pixels_circle = hpgeom.query_circle(nside, (box[0] + box[1])/2., lat, radius*2.5)
    lon_circle, lat_circle = hpgeom.pixel_to_angle(nside, pixels_circle)
    inside = _pos_in_box(lon_circle, lat_circle, *box)

    np.testing.assert_array_equal(pixels, pixels_circle[inside])

    # Do inclusive
    pixels_box = hpgeom.query_box(nside, *box, inclusive=True)

    # Ensure all the inner pixels are in the inclusive pixels
    sub1, sub2 = match_arrays(pixels_box, pixels)
    assert sub2.size == pixels.size

    # Look at boundaries of the pixels.
    pixels_circle = hpgeom.query_circle(nside, (box[0] + box[1])/2., lat, radius*2.5)
    boundaries_lon, boundaries_lat = hpgeom.boundaries(nside, pixels_circle, step=4)
    cut = _pos_in_box(boundaries_lon.ravel(), boundaries_lat.ravel(), *box)
    test = cut.reshape(boundaries_lon.shape).sum(axis=1)
    pixels_circle_box = pixels_circle[test > 0]

    # Ensure all these pixels are in the inclusive list
    sub1, sub2 = match_arrays(pixels_circle_box, pixels_box)
    assert sub1.size == pixels_circle_box.size

    # And ensure that we did wrap over the edge
    lon_box, lat_box = hpgeom.pixel_to_angle(nside, pixels_box)
    if (lon == 0.0):
        assert lon_box.max() > 355.0
    else:
        assert lon_box.min() < 5.0


@pytest.mark.parametrize("nside_radius", [(2**5, 2.0),
                                          (2**10, 1.0),
                                          (2**20, 0.01)])
@pytest.mark.parametrize("lon", [0.0, 90.0, 180.0, 270.0])
@pytest.mark.parametrize("lat", [90.0, -90.0])
def test_query_box_pole(nside_radius, lon, lat):
    """Test query_box at the latitude poles."""
    nside = nside_radius[0]
    radius = nside_radius[1]

    if (lat == 90.0):
        box = [lon - radius, lon + radius, lat - 2*radius, lat]
    else:
        box = [lon - radius, lon + radius, lat, lat + 2*radius]

    pixels = hpgeom.query_box(nside, *box)

    pixels_circle = hpgeom.query_circle(nside, lon, (box[2] + box[3])/2., radius*2.5)
    lon_circle, lat_circle = hpgeom.pixel_to_angle(nside, pixels_circle)
    inside = _pos_in_box(lon_circle, lat_circle, *box)

    np.testing.assert_array_equal(pixels, pixels_circle[inside])

    # Do inclusive
    pixels_box = hpgeom.query_box(nside, *box, inclusive=True)

    # Ensure all the inner pixels are in the inclusive pixels
    sub1, sub2 = match_arrays(pixels_box, pixels)
    assert sub2.size == pixels.size

    # Look at boundaries of the pixels, check if any are included.
    pixels_circle = hpgeom.query_circle(nside, lon, lat, radius*2.5)
    boundaries_lon, boundaries_lat = hpgeom.boundaries(nside, pixels_circle, step=4)
    cut = _pos_in_box(boundaries_lon.ravel(), boundaries_lat.ravel(), *box)
    test = cut.reshape(boundaries_lon.shape).sum(axis=1)
    pixels_circle_box = pixels_circle[test > 0]

    # Ensure all these pixels are in the inclusive list
    sub1, sub2 = match_arrays(pixels_circle_box, pixels_box)
    if (lon == 0.0 and lat == -90.0):
        # There is some oddity with the polar pixel boundaries that needs to
        # be investigated.
        assert sub1.size == (pixels_circle_box.size - 2)
    else:
        assert sub1.size == pixels_circle_box.size


@pytest.mark.parametrize("nside", [2**5, 2**10])
@pytest.mark.parametrize("lat", [90.0, -90.0])
def test_query_box_pole_full_longitude(nside, lat):
    """Test query_box at the latitude poles, full longitude."""
    radius = 2.0

    if (lat == 90.0):
        box = [0.0, 360.0, lat - radius, lat]
    else:
        box = [0.0, 360.0, lat, lat + radius]

    pixels = hpgeom.query_box(nside, *box)

    # This should be the same as a circle at the pole.
    pixels_circle = hpgeom.query_circle(nside, 0.0, lat, radius)

    np.testing.assert_array_equal(pixels, pixels_circle)

    # Do inclusive
    pixels_box = hpgeom.query_box(nside, *box, inclusive=True)

    # Ensure all the inner pixels are in the inclusive pixels
    sub1, sub2 = match_arrays(pixels_box, pixels)
    assert sub2.size == pixels.size

    pixels_circle_inc = hpgeom.query_circle(nside, 0.0, lat, radius, inclusive=True)

    np.testing.assert_array_equal(pixels_box, pixels_circle_inc)


@pytest.mark.parametrize("nside", [2**5, 2**10])
@pytest.mark.parametrize("lat", [-45.0, 0.0, 45.0])
def test_query_box_full_longitude(nside, lat):
    """Test query_box at the latitude poles."""
    radius = 1.0

    box = [0.0, 360.0, lat - radius, lat + radius]

    pixels = hpgeom.query_box(nside, *box)

    pixels_all = np.arange(hpgeom.nside_to_npixel(nside))
    lon_all, lat_all = hpgeom.pixel_to_angle(nside, pixels_all)
    inside = ((lat_all >= box[2]) & (lat_all < box[3]))

    np.testing.assert_array_equal(pixels, pixels_all[inside])

    # Do inclusive
    pixels_box = hpgeom.query_box(nside, *box, inclusive=True)

    # Ensure all the inner pixels are in the inclusive pixels
    sub1, sub2 = match_arrays(pixels_box, pixels)
    assert sub2.size == pixels.size

    # Look at boundaries of the pixels, check if any are included.
    inside_plus = ((lat_all >= (lat - 2*radius)) & (lat_all <= (lat + 2*radius)))
    boundaries_lon, boundaries_lat = hpgeom.boundaries(nside, pixels_all[inside_plus], step=4)
    cut = ((boundaries_lat.ravel() >= box[2]) & (boundaries_lat.ravel() < box[3]))
    test = cut.reshape(boundaries_lat.shape).sum(axis=1)
    pixels_all_box = pixels_all[inside_plus][test > 0]

    # Ensure all these pixels are in the inclusive list
    sub1, sub2 = match_arrays(pixels_all_box, pixels_box)
    assert sub1.size == pixels_all_box.size


@pytest.mark.parametrize("fact", [1, 2, 4, 8])
def test_query_box_fact(fact):
    """Test query_box, use other fact values."""
    nside = 1024
    radius = 1.0
    lon = 90.0
    lat = 0.0

    box = [lon - radius, lon + radius, lat - radius, lat + radius]

    pixels = hpgeom.query_box(nside, *box)

    pixels_box = hpgeom.query_box(nside, *box, inclusive=True, fact=fact)

    # Ensure all the inner pixels are in the inclusive pixels
    sub1, sub2 = match_arrays(pixels_box, pixels)
    assert sub2.size == pixels.size

    # Look at boundaries of the pixels, check if any are included.
    pixels_circle = hpgeom.query_circle(nside, lon, lat, radius*2.5)
    boundaries_lon, boundaries_lat = hpgeom.boundaries(nside, pixels_circle, step=4)
    cut = _pos_in_box(boundaries_lon.ravel(), boundaries_lat.ravel(), *box)
    test = cut.reshape(boundaries_lon.shape).sum(axis=1)
    pixels_circle_box = pixels_circle[test > 0]

    # Ensure all these pixels are in the inclusive list
    sub1, sub2 = match_arrays(pixels_circle_box, pixels_box)
    assert sub1.size == pixels_circle_box.size


def test_query_box_empty():
    """Test query box when it is empty."""

    pixels = hpgeom.query_box(1024, 25.0, 25.0, 0.0, 2.0)
    assert pixels.size == 0

    pixels = hpgeom.query_box(1024, 90.0, 92.0, 45.0, 45.0)
    assert pixels.size == 0


def test_query_box_radians():
    """Test query_box, use lonlat and radians."""
    nside = 1024
    lon = 90.0
    lat = 45.0
    radius = 1.0

    box = [lon - radius, lon + radius, lat - radius, lat + radius]

    pixels_deg = hpgeom.query_box(nside, *box)
    pixels_rad = hpgeom.query_box(nside, *np.deg2rad(box), degrees=False)

    np.testing.assert_array_equal(pixels_rad, pixels_deg)


def test_query_box_thetaphi():
    """Test query_box, use theta_phi."""
    nside = 1024
    lon = 90.0
    lat = 45.0
    radius = 1.0

    box = [lon - radius, lon + radius, lat - radius, lat + radius]

    pixels_deg = hpgeom.query_box(nside, *box)
    theta, phi = hpgeom.lonlat_to_thetaphi(np.array(box[0: 2]), np.array(box[2:]))
    box_thetaphi = [theta[1], theta[0], phi[0], phi[1]]
    pixels_thetaphi = hpgeom.query_box(nside, *box_thetaphi, lonlat=False)

    np.testing.assert_array_equal(pixels_thetaphi, pixels_deg)


def test_query_box_ring():
    """Test query_box, ring ordering."""
    nside = 1024
    lon = 90.0
    lat = 45.0
    radius = 1.0

    box = [lon - radius, lon + radius, lat - radius, lat + radius]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pixels_ring = hpgeom.query_box(nside, *box, nest=False)

    pixels_nest = hpgeom.query_box(nside, *box)
    pixels_nest_to_ring = hpgeom.nest_to_ring(nside, pixels_nest)
    pixels_nest_to_ring.sort()

    np.testing.assert_array_equal(pixels_ring, pixels_nest_to_ring)


def test_query_box_badinputs():
    """Test query_box with bad inputs."""
    with pytest.raises(ValueError, match=r"lat .* out of range"):
        hpgeom.query_box(2048, 0.0, 10.0, 80.0, 100.0)

    with pytest.raises(ValueError, match=r"lat .* out of range"):
        hpgeom.query_box(2048, 0.0, 10.0, -100.0, -80.0)

    with pytest.raises(ValueError, match=r"colatitude \(theta\) .* out of range"):
        hpgeom.query_box(2048, -0.1, 0.0, 0.0, 0.2, lonlat=False)

    with pytest.raises(ValueError, match=r"colatitude \(theta\) .* out of range"):
        hpgeom.query_box(2048, 0.0, np.pi + 1, 0.0, 0.2, lonlat=False)

    with pytest.raises(ValueError, match=r"longitude \(phi\) .* out of range"):
        hpgeom.query_box(2048, 0.0, 0.1, -0.1, 0.2, lonlat=False)

    with pytest.raises(ValueError, match=r"longitude \(phi\) .* out of range"):
        hpgeom.query_box(2048, 0.0, 0.1, 0.1, 3*np.pi, lonlat=False)

    with pytest.raises(ValueError, match=r"b1/lat1 must be >= b0/lat0"):
        hpgeom.query_box(2048, 0.0, 0.1, 1.0, 0.0)

    with pytest.raises(ValueError, match=r"a1/colatitude1 must be <= a0/colatitude0"):
        hpgeom.query_box(2048, 1.0, 0.9, 0.0, 0.1, lonlat=False)

    with pytest.raises(ValueError, match=r"Inclusive factor .* must be positive"):
        hpgeom.query_box(2048, 0.0, 1.0, 0.0, 1.0, inclusive=True, fact=0)

    with pytest.raises(ValueError, match=r"Inclusive factor \* nside must be \<\="):
        hpgeom.query_box(2**28, 0.0, 1.0, 0.0, 1.0, inclusive=True, fact=4)

    with pytest.raises(ValueError, match=r"Inclusive factor .* must be power of 2 for nest"):
        hpgeom.query_box(2048, 0.0, 1.0, 0.0, 1.0, inclusive=True, fact=3)

    # Different platforms have different strings here, but they all say ``integer``.
    with pytest.raises(TypeError, match=r"integer"):
        hpgeom.query_box(2048, 0.0, 1.0, 0.0, 1.0, inclusive=True, nest=False, fact=3.5)

    # Check resource warning
    with pytest.warns(ResourceWarning, match=r"natively supports nest ordering"):
        hpgeom.query_box(1024, 0.0, 1.0, 0.0, 1.0, nest=False)
