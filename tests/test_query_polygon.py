import numpy as np
import pytest

try:
    import healpy as hp
    has_healpy = True
except ImportError:
    has_healpy = False

import hpgeom


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
@pytest.mark.parametrize("nside_delta", [(2**5, 2.0),
                                         (2**10, 1.0),
                                         (2**18, 0.01)])
@pytest.mark.parametrize("shape", ["triangle", "square", "hexagon"])
@pytest.mark.parametrize("lon_ref", [0.0, 90.0, 180.0, 270.0])
@pytest.mark.parametrize("lat_ref", [-43.0, 0.0, 43.0, 85.0, -85.0])
def test_query_polygon_nest(nside_delta, shape, lon_ref, lat_ref):
    """Test query_polygon, nest ordering."""
    nside = nside_delta[0]
    delta = nside_delta[1]

    if shape == "triangle":
        lon = np.array([lon_ref, lon_ref + delta, lon_ref + delta])
        lat = np.array([lat_ref, lat_ref + delta, lat_ref - delta])
    elif shape == "square":
        lon = np.array([lon_ref, lon_ref + delta, lon_ref + delta, lon_ref])
        lat = np.array([lat_ref, lat_ref, lat_ref + delta, lat_ref + delta])
    else:
        lon = np.array([
            lon_ref,
            lon_ref + delta/2.,
            lon_ref + delta,
            lon_ref + 3*delta/2.,
            lon_ref + delta,
            lon_ref + delta/2.,
        ])
        lat = np.array([
            lat_ref,
            lat_ref + delta,
            lat_ref + delta,
            lat_ref,
            lat_ref - delta,
            lat_ref - delta
        ])

    # Test forward, non-inclusive
    pixels_hpgeom = hpgeom.query_polygon(nside, lon, lat)
    vec = hpgeom.angle_to_vector(lon, lat, lonlat=True)
    pixels_healpy = hp.query_polygon(nside, vec, nest=True)

    np.testing.assert_array_equal(pixels_hpgeom, pixels_healpy)

    # Test forward, non-inclusive, closed polygon
    lon2 = np.append(lon, lon[0])
    lat2 = np.append(lat, lat[0])

    pixels_hpgeom2 = hpgeom.query_polygon(nside, lon2, lat2)
    # Note healpy does not support closed polygons, so we use the first
    # run as a comparison.

    np.testing.assert_array_equal(pixels_hpgeom2, pixels_hpgeom)

    # Test reversed, non-inclusive
    pixels_hpgeom = hpgeom.query_polygon(nside, lon[::-1], lat[::-1])
    vec = hpgeom.angle_to_vector(lon[::-1], lat[::-1], lonlat=True)
    pixels_healpy = hp.query_polygon(nside, vec, nest=True)

    np.testing.assert_array_equal(pixels_hpgeom, pixels_healpy)

    # Test forward, inclusive
    pixels_hpgeom = hpgeom.query_polygon(nside, lon, lat, inclusive=True)
    vec = hpgeom.angle_to_vector(lon, lat, lonlat=True)
    pixels_healpy = hp.query_polygon(nside, vec, nest=True, inclusive=True)

    np.testing.assert_array_equal(pixels_hpgeom, pixels_healpy)


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
@pytest.mark.parametrize("nside_delta", [
    (2**5, 2.0),
    (2**10, 1.0),
    (2**18, 0.01),
    (2**10 - 2, 1.0)
])
@pytest.mark.parametrize("shape", ["triangle", "square", "hexagon"])
@pytest.mark.parametrize("lon_ref", [0.0, 90.0, 180.0])
@pytest.mark.parametrize("lat_ref", [-44.0, 0.0, 44.0, 85.0, -85.0])
def test_query_polygon_ring(nside_delta, shape, lon_ref, lat_ref):
    """Test query_polygon, ring ordering."""
    nside = nside_delta[0]
    delta = nside_delta[1]

    if shape == "triangle":
        lon = np.array([lon_ref, lon_ref + delta, lon_ref + delta])
        lat = np.array([lat_ref, lat_ref + delta, lat_ref - delta])
    elif shape == "square":
        lon = np.array([lon_ref, lon_ref + delta, lon_ref + delta, lon_ref])
        lat = np.array([lat_ref, lat_ref, lat_ref + delta, lat_ref + delta])
    else:
        lon = np.array([
            lon_ref,
            lon_ref + delta/2.,
            lon_ref + delta,
            lon_ref + 3*delta/2.,
            lon_ref + delta,
            lon_ref + delta/2.,
        ])
        lat = np.array([
            lat_ref,
            lat_ref + delta,
            lat_ref + delta,
            lat_ref,
            lat_ref - delta,
            lat_ref - delta
        ])

    # Test forward, non-inclusive
    pixels_hpgeom = hpgeom.query_polygon(nside, lon, lat, nest=False)
    vec = hpgeom.angle_to_vector(lon, lat, lonlat=True)
    pixels_healpy = hp.query_polygon(nside, vec, nest=False)

    np.testing.assert_array_equal(pixels_hpgeom, pixels_healpy)

    # Test reversed, non-inclusive
    pixels_hpgeom_rev = hpgeom.query_polygon(nside, lon[::-1], lat[::-1], nest=False)
    vec = hpgeom.angle_to_vector(lon[::-1], lat[::-1], lonlat=True)
    pixels_healpy = hp.query_polygon(nside, vec, nest=False)

    np.testing.assert_array_equal(pixels_hpgeom_rev, pixels_healpy)

    # Test forward, inclusive
    pixels_hpgeom = hpgeom.query_polygon(nside, lon, lat, inclusive=True, nest=False)
    vec = hpgeom.angle_to_vector(lon, lat, lonlat=True)
    pixels_healpy = hp.query_polygon(nside, vec, nest=False, inclusive=True)

    np.testing.assert_array_equal(pixels_hpgeom, pixels_healpy)


def test_query_polygon_radians():
    """Test query_polygon, use lonlat and radians."""
    nside = 1024
    delta = 1.0
    lon_ref = 180.0
    lat_ref = 0.0
    lon = np.array([lon_ref, lon_ref + delta, lon_ref + delta, lon_ref])
    lat = np.array([lat_ref, lat_ref, lat_ref + delta, lat_ref + delta])

    pixels_deg = hpgeom.query_polygon(nside, lon, lat)
    pixels_rad = hpgeom.query_polygon(nside, np.deg2rad(lon), np.deg2rad(lat), degrees=False)

    np.testing.assert_array_equal(pixels_deg, pixels_rad)


def test_query_polygon_thetaphi():
    """Test query_polygon, use theta_phi."""
    nside = 1024
    delta = 1.0
    lon_ref = 180.0
    lat_ref = 0.0
    lon = np.array([lon_ref, lon_ref + delta, lon_ref + delta, lon_ref])
    lat = np.array([lat_ref, lat_ref, lat_ref + delta, lat_ref + delta])

    pixels_lonlat = hpgeom.query_polygon(nside, lon, lat)
    theta, phi = hpgeom.lonlat_to_thetaphi(lon, lat)
    pixels_thetaphi = hpgeom.query_polygon(nside, theta, phi, lonlat=False)

    np.testing.assert_array_equal(pixels_lonlat, pixels_thetaphi)


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
@pytest.mark.parametrize("fact", [1, 2, 4, 8])
def test_query_polygon_fact(fact):
    """Test query_polygon, use other fact values."""
    nside = 1024
    delta = 1.0
    lon_ref = 180.0
    lat_ref = 0.0
    lon = np.array([lon_ref, lon_ref + delta, lon_ref + delta, lon_ref])
    lat = np.array([lat_ref, lat_ref, lat_ref + delta, lat_ref + delta])

    pixels_hpgeom = hpgeom.query_polygon(nside, lon, lat, fact=fact)
    vec = hpgeom.angle_to_vector(lon, lat)
    pixels_healpy = hp.query_polygon(nside, vec, nest=True, fact=fact)

    np.testing.assert_array_equal(pixels_hpgeom, pixels_healpy)


def test_query_polygon_vec():
    """Test query_polygon_vec."""
    nside = 1024
    delta = 1.0
    lon_ref = 180.0
    lat_ref = 0.0
    lon = np.array([lon_ref, lon_ref + delta, lon_ref + delta, lon_ref])
    lat = np.array([lat_ref, lat_ref, lat_ref + delta, lat_ref + delta])

    pixels_lonlat = hpgeom.query_polygon(nside, lon, lat)
    vec = hpgeom.angle_to_vector(lon, lat)
    pixels_vec = hpgeom.query_polygon_vec(nside, vec)

    np.testing.assert_array_equal(pixels_lonlat, pixels_vec)


def test_query_polygon_return_pixel_ranges():
    """Test query_polygon with return_pixel_ranges."""
    nside = 1024
    delta = 1.0
    lon_ref = 180.0
    lat_ref = 0.0
    lon = np.array([lon_ref, lon_ref + delta, lon_ref + delta, lon_ref])
    lat = np.array([lat_ref, lat_ref, lat_ref + delta, lat_ref + delta])

    pixels = hpgeom.query_polygon(nside, lon, lat)

    pixel_ranges = hpgeom.query_polygon(nside, lon, lat, return_pixel_ranges=True)
    pixels_from_ranges = hpgeom.pixel_ranges_to_pixels(pixel_ranges)

    assert pixel_ranges.size < pixels.size

    np.testing.assert_array_equal(pixels, pixels_from_ranges)

    # And try a tiny polygon that has no pixels.
    delta = 0.001
    lon = np.array([lon_ref, lon_ref + delta, lon_ref + delta, lon_ref])
    lat = np.array([lat_ref, lat_ref, lat_ref + delta, lat_ref + delta])

    pixels = hpgeom.query_polygon(nside, lon, lat)
    pixel_ranges = hpgeom.query_polygon(nside, lon, lat, return_pixel_ranges=True)
    pixels_from_ranges = hpgeom.pixel_ranges_to_pixels(pixel_ranges)

    np.testing.assert_array_equal(pixels, pixels_from_ranges)
    assert len(pixel_ranges) == 0


def test_query_polygon_badinputs():
    """Test query_polygon with bad inputs."""
    nside = 1024
    delta = 1.0
    lon_ref = 180.0
    lat_ref = 0.0
    lon = np.array([lon_ref, lon_ref + delta, lon_ref + delta, lon_ref])
    lat = np.array([lat_ref, lat_ref, lat_ref + delta, lat_ref + delta])

    _lat = lat.copy()
    _lat[0] = 100.0
    with pytest.raises(ValueError, match=r"lat .* out of range"):
        hpgeom.query_polygon(nside, lon, _lat)

    _lat = lat.copy()
    _lat[0] = -100.0
    with pytest.raises(ValueError, match=r"lat .* out of range"):
        hpgeom.query_polygon(nside, lon, _lat)

    _theta, _phi = hpgeom.lonlat_to_thetaphi(lon, lat)
    _theta[0] = -0.1
    with pytest.raises(ValueError, match=r"colatitude \(theta\) .* out of range"):
        hpgeom.query_polygon(nside, _theta, _phi, lonlat=False)

    _lon = np.append(lon, lon_ref + delta/2.)
    _lat = np.append(lat, lat_ref + delta/2.)
    with pytest.raises(RuntimeError, match="Polygon is not convex"):
        hpgeom.query_polygon(nside, _lon, _lat)

    with pytest.raises(ValueError, match=r"Inclusive factor .* must be positive"):
        hpgeom.query_polygon(nside, lon, lat, inclusive=True, fact=0)

    with pytest.raises(ValueError, match=r"Inclusive factor \* nside must be \<\="):
        hpgeom.query_polygon(2**28, lon, lat, inclusive=True, fact=4)

    with pytest.raises(ValueError, match=r"Inclusive factor .* must be power of 2 for nest"):
        hpgeom.query_polygon(nside, lon, lat, inclusive=True, fact=3)

    # Different platforms have different strings here, but they all say ``integer``.
    with pytest.raises(TypeError, match=r"integer"):
        # Illegal fact (must be integer)
        hpgeom.query_polygon(nside, lon, lat, inclusive=True, nest=False, fact=3.5)

    with pytest.raises(RuntimeError, match=r"Can only use return_pixel_ranges with nest"):
        hpgeom.query_polygon(nside, lon, lat, nest=False, return_pixel_ranges=True)


def test_query_polygon_lsstcam_region():
    """Test query_polygon with an LSSTCam region."""

    # This region works with lsst-sphgeom.
    vertices = np.array([[0.5022837, 0.69468255, -0.51490507],
                         [0.51289086, 0.68943526, -0.51148996],
                         [0.51640143, 0.68766516, -0.51033929],
                         [0.52654024, 0.67506305, -0.51676421],
                         [0.52640267, 0.67247327, -0.52026909],
                         [0.52626019, 0.6700308 , -0.52355414],
                         [0.52520076, 0.65511388, -0.54312979],
                         [0.5249898, 0.65259656, -0.54635469],
                         [0.51381171, 0.65032841, -0.55952702],
                         [0.51001432, 0.65223101, -0.56078526],
                         [0.5064929 , 0.65398453, -0.56193343],
                         [0.50296519, 0.65572598, -0.56307144],
                         [0.48157927, 0.66597104, -0.56970517],
                         [0.47144083, 0.67858435, -0.56326443],
                         [0.47164957, 0.68127065, -0.55983657],
                         [0.47220747, 0.6887757, -0.5500983],
                         [0.47237857, 0.69125198, -0.54683561],
                         [0.47284995, 0.69861958, -0.53697636],
                         [0.47299081, 0.70104594, -0.53367994],
                         [0.48416058, 0.7033129, -0.52051848],
                         [0.48803028, 0.70150837, -0.51933847],
                         [0.49159976, 0.69982473, -0.51824224]],
                        )
    pixels = hpgeom.query_polygon_vec(1024, vertices)
