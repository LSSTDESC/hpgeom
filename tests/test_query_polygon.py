import numpy as np
import pytest

try:
    import healpy as hp
    has_healpy = True
except ImportError:
    has_healpy = False

try:
    from astropy.coordinates import SkyCoord
    has_astropy = True
except ImportError:
    has_astropy = False

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


def _inside_convex_polygon(lon_v, lat_v, lon, lat):
    """Test for points inside a convex polygon.

    Parameters
    ----------
    lon_v : `np.ndarray`
        Polygon longitudes, degrees.
    lat_v : `np.ndarray`
        Polygon latitudes, degrees.
    lon : `np.ndarray`
        Test point longitudes.
    lat : `np.ndarray`
        Test point latitudes.

    Returns
    -------
    in_polygon : `np.ndarray`
        Boolean array of which points are in the polygon.

    """
    V = hpgeom.angle_to_vector(lon_v, lat_v)
    N = np.cross(V, np.roll(V, -1, axis=0))
    if np.all(N @ V.sum(axis=0) < 0):
        N = -N
    X = hpgeom.angle_to_vector(lon, lat)
    return np.all(X @ N.T >= 0, axis=1)


frames = ["equatorial", "icrs", "c", "galactic", "g", "ecliptic", "barycentricmeanecliptic", "e"]


@pytest.mark.skipif(not has_astropy, reason="Skipping test without astropy")
@pytest.mark.parametrize("query_frame", frames)
@pytest.mark.parametrize("pixel_frame", frames)
@pytest.mark.parametrize("closed", [False, True])
def test_query_polygon_frames(query_frame, pixel_frame, closed):
    """Test query_polygon with frame conversions."""
    nside = 4096
    lon_qf = np.array([30.0, 30.0, 33.0, 36.0, 36.0])
    lat_qf = np.array([50.0, 53.0, 56.0, 53.0, 50.0])

    if closed:
        lon_qf_query = np.append(lon_qf, lon_qf[0])
        lat_qf_query = np.append(lat_qf, lat_qf[0])
    else:
        lon_qf_query = lon_qf
        lat_qf_query = lat_qf

    if query_frame in ["equatorial", "icrs", "c"]:
        astropy_qf = "icrs"
    elif query_frame in ["galactic", "g"]:
        astropy_qf = "galactic"
    else:
        astropy_qf = "barycentricmeanecliptic"

    if pixel_frame in ["equatorial", "icrs", "c"]:
        astropy_pf = "icrs"
    elif pixel_frame in ["galactic", "g"]:
        astropy_pf = "galactic"
    else:
        astropy_pf = "barycentricmeanecliptic"

    # Brute-force reference
    # We find the center of the polygon; rotate it; find max radius.
    cvec = hpgeom.angle_to_vector(lon_qf, lat_qf).sum(axis=0)
    c_lon_qf = np.degrees(np.arctan2(cvec[1], cvec[0])) % 360
    c_lat_qf = np.degrees(np.arcsin(cvec[2] / np.linalg.norm(cvec)))
    center = SkyCoord(c_lon_qf, c_lat_qf, frame=astropy_qf, unit="degree")
    rmax = center.separation(SkyCoord(lon_qf, lat_qf, frame=astropy_qf, unit="degree")).degree.max()

    # Convert the center to the pixel frame.
    if astropy_pf == "icrs":
        c_lon_pf, c_lat_pf = center.icrs.ra.degree, center.icrs.dec.degree
    elif astropy_pf == "galactic":
        c_lon_pf, c_lat_pf = center.galactic.l.degree, center.galactic.b.degree
    else:
        eclip = center.barycentricmeanecliptic
        c_lon_pf, c_lat_pf = eclip.lon.degree, eclip.lat.degree

    # Get the candidates within a circle and select down.
    candidates = hpgeom.query_circle(nside, c_lon_pf, c_lat_pf, rmax + 0.1)
    cand_coord = SkyCoord(*hpgeom.pixel_to_angle(nside, candidates), frame=astropy_pf, unit="degree")

    # Back to original frame.
    if astropy_qf == "icrs":
        cand_lon_qf, cand_lat_qf = cand_coord.icrs.ra.degree, cand_coord.icrs.dec.degree
    elif astropy_qf == "galactic":
        cand_lon_qf, cand_lat_qf = cand_coord.galactic.l.degree, cand_coord.galactic.b.degree
    else:
        eclip = cand_coord.barycentricmeanecliptic
        cand_lon_qf, cand_lat_qf = eclip.lon.degree, eclip.lat.degree

    pixels1 = np.sort(candidates[_inside_convex_polygon(lon_qf, lat_qf, cand_lon_qf, cand_lat_qf)])

    # Direct query
    pixels2 = hpgeom.query_polygon(
        nside,
        lon_qf_query,
        lat_qf_query,
        query_frame=query_frame,
        pixel_frame=pixel_frame,
    )

    np.testing.assert_array_equal(pixels2, pixels1)


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
