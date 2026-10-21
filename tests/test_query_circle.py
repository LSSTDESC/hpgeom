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


@pytest.mark.parametrize("nside", [2**0, 2**5, 2**10])
def test_query_circle_fullsky_nest(nside):
    """Test query_circle over the full sky, nest ordering."""
    pixels = hpgeom.query_circle(nside, 0.0, 0.0, 190.0)

    np.testing.assert_array_equal(pixels, np.arange(12*nside*nside))


@pytest.mark.parametrize("nside", [2**0, 2**5, 2**10])
def test_query_circle_fullsky_ring(nside):
    """Test query_circle over the full sky, ring ordering."""
    pixels = hpgeom.query_circle(nside, 0.0, 0.0, 190.0)

    np.testing.assert_array_equal(pixels, np.arange(12*nside*nside))


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
@pytest.mark.parametrize("nside_radius", [(2**5, 2.0),
                                          (2**10, 1.0),
                                          (2**20, 0.01)])
@pytest.mark.parametrize("lon", [0.0, 90.0, 180.0, 270.0])
@pytest.mark.parametrize("lat", [-45.0, 0.0, 45.0, 90.0, -90.0])
def test_query_circle_nest(nside_radius, lon, lat):
    """Test query_circle, nest ordering."""
    nside = nside_radius[0]
    radius = nside_radius[1]

    # First, non-inclusive
    pixels_hpgeom = hpgeom.query_circle(nside, lon, lat, radius)

    vec = hp.ang2vec(lon, lat, lonlat=True)
    pixels_healpy = hp.query_disc(nside, vec, np.deg2rad(radius), nest=True)

    np.testing.assert_array_equal(pixels_hpgeom, pixels_healpy)

    # Second, inclusive.
    pixels_hpgeom = hpgeom.query_circle(nside, lon, lat, radius, inclusive=True)

    vec = hp.ang2vec(lon, lat, lonlat=True)
    pixels_healpy = hp.query_disc(nside, vec, np.deg2rad(radius), inclusive=True, nest=True)

    np.testing.assert_array_equal(pixels_hpgeom, pixels_healpy)


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
@pytest.mark.parametrize("nside_radius", [(2**5, 2.0),
                                          (2**10, 1.0),
                                          (2**20, 0.01),
                                          (2**10 - 2, 0.5)])
@pytest.mark.parametrize("lon", [0.0, 90.0, 180.0, 270.0])
@pytest.mark.parametrize("lat", [-45.0, 0.0, 45.0, 90.0, -90.0])
def test_query_circle_ring(nside_radius, lon, lat):
    """Test query_circle, ring ordering."""
    nside = nside_radius[0]
    radius = nside_radius[1]

    # First, non-inclusive
    pixels_hpgeom = hpgeom.query_circle(nside, lon, lat, radius, nest=False)

    vec = hp.ang2vec(lon, lat, lonlat=True)
    pixels_healpy = hp.query_disc(nside, vec, np.deg2rad(radius))

    np.testing.assert_array_equal(pixels_hpgeom, pixels_healpy)

    # Second, inclusive
    pixels_hpgeom = hpgeom.query_circle(nside, lon, lat, radius, inclusive=True, nest=False)

    vec = hp.ang2vec(lon, lat, lonlat=True)
    pixels_healpy = hp.query_disc(nside, vec, np.deg2rad(radius), inclusive=True)

    np.testing.assert_array_equal(pixels_hpgeom, pixels_healpy)


def test_query_circle_radians():
    """Test query_circle, use lonlat and radians."""
    lon = 10.0
    lat = 20.0
    radius = 1.0
    nside = 2048

    pixels_deg = hpgeom.query_circle(nside, lon, lat, radius)
    pixels_rad = hpgeom.query_circle(
        nside,
        np.deg2rad(lon),
        np.deg2rad(lat),
        np.deg2rad(radius),
        degrees=False
    )

    np.testing.assert_array_equal(pixels_rad, pixels_deg)


def test_query_circle_thetaphi():
    """Test query_circle, use theta_phi."""
    lon = 10.0
    lat = 20.0
    radius = 1.0
    nside = 2048

    pixels_deg = hpgeom.query_circle(nside, lon, lat, radius)

    theta, phi = hpgeom.lonlat_to_thetaphi(lon, lat)
    pixels_thetaphi = hpgeom.query_circle(nside, theta, phi, np.deg2rad(radius), lonlat=False)

    np.testing.assert_array_equal(pixels_thetaphi, pixels_deg)


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
@pytest.mark.parametrize("fact", [1, 2, 4, 8])
def test_query_circle_fact(fact):
    """Test query_circle, use other fact values."""
    lon = 10.0
    lat = 20.0
    radius = 0.5
    nside = 4096

    pixels_hpgeom = hpgeom.query_circle(nside, lon, lat, radius, inclusive=True, fact=fact)
    vec = hp.ang2vec(lon, lat, lonlat=True)
    pixels_healpy = hp.query_disc(nside, vec, np.deg2rad(radius), inclusive=True, fact=fact, nest=True)

    np.testing.assert_array_equal(pixels_hpgeom, pixels_healpy)


def test_query_circle_vec():
    """Test query_circle_vec."""
    lon = 10.0
    lat = 20.0
    radius = 0.5
    nside = 4096

    pixels_qc = hpgeom.query_circle(nside, lon, lat, radius)

    # Convert to vec
    # FIXME: use internal when available
    theta, phi = hpgeom.lonlat_to_thetaphi(lon, lat)
    sintheta = np.sin(theta)
    vec = [sintheta*np.cos(phi), sintheta*np.sin(phi), np.cos(theta)]

    pixels_qcv = hpgeom.query_circle_vec(nside, vec, np.deg2rad(radius))

    np.testing.assert_array_equal(pixels_qcv, pixels_qc)


def test_query_circle_return_pixel_ranges():
    """Test query_circle with return_pixel_ranges."""
    lon = 10.0
    lat = 20.0
    radius = 0.5
    nside = 4096

    pixels = hpgeom.query_circle(nside, lon, lat, radius)

    pixel_ranges = hpgeom.query_circle(nside, lon, lat, radius, return_pixel_ranges=True)
    pixels_from_ranges = hpgeom.pixel_ranges_to_pixels(pixel_ranges)

    assert pixel_ranges.size < pixels.size

    np.testing.assert_array_equal(pixels, pixels_from_ranges)

    # And try a tiny circle that has no pixels.
    radius = 0.001
    pixels = hpgeom.query_circle(nside, lon, lat, radius)
    pixel_ranges = hpgeom.query_circle(nside, lon, lat, radius, return_pixel_ranges=True)
    pixels_from_ranges = hpgeom.pixel_ranges_to_pixels(pixel_ranges)

    np.testing.assert_array_equal(pixels, pixels_from_ranges)
    assert len(pixel_ranges) == 0


frames = ["equatorial", "icrs", "c", "galactic", "g", "ecliptic", "barycentricmeanecliptic", "e"]


@pytest.mark.skipif(not has_astropy, reason="Skipping test without astropy")
@pytest.mark.parametrize("query_frame", frames)
@pytest.mark.parametrize("pixel_frame", frames)
def test_query_circle_frames(query_frame, pixel_frame):
    """Test query_circle with frame conversions."""
    lon_qf = 10.0
    lat_qf = 20.0
    radius = 0.5
    nside = 4096

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
    center = SkyCoord(lon_qf, lat_qf, frame=astropy_qf, unit="degree")
    if astropy_pf == "icrs":
        lon_pf, lat_pf = center.icrs.ra.degree, center.icrs.dec.degree
    elif astropy_pf == "galactic":
        lon_pf, lat_pf = center.galactic.l.degree, center.galactic.b.degree
    else:
        lon_pf, lat_pf = center.barycentricmeanecliptic.lon.degree, center.barycentricmeanecliptic.lat.degree

    candidates = hpgeom.query_circle(nside, lon_pf, lat_pf, radius + 0.1)
    cand_lon_pf, cand_lat_pf = hpgeom.pixel_to_angle(nside, candidates)
    sep = SkyCoord(cand_lon_pf, cand_lat_pf, frame=astropy_pf, unit="degree").separation(center).degree
    pixels1 = np.sort(candidates[sep <= radius])

    # Direct query
    pixels2 = hpgeom.query_circle(
        nside,
        lon_qf,
        lat_qf,
        radius,
        query_frame=query_frame,
        pixel_frame=pixel_frame,
    )

    np.testing.assert_array_equal(pixels2, pixels1)


def test_query_circle_badinputs():
    """Test query circle with bad inputs."""
    with pytest.raises(ValueError, match=r"lat .* out of range"):
        # Latitude out of range
        hpgeom.query_circle(2048, 0.0, 100.0, 1.0)

    with pytest.raises(ValueError, match=r"lat .* out of range"):
        # Latitude out of range
        hpgeom.query_circle(2048, 0.0, -100.0, 1.0)

    with pytest.raises(ValueError, match=r"colatitude \(theta\) .* out of range"):
        # theta out of range
        hpgeom.query_circle(2048, -0.1, 0.0, 0.1, lonlat=False)

    with pytest.raises(ValueError, match=r"colatitude \(theta\) .* out of range"):
        # theta out of range
        hpgeom.query_circle(2048, np.pi + 0.1, 0.0, 0.1, lonlat=False)

    with pytest.raises(ValueError, match=r"longitude \(phi\) .* out of range"):
        # phi out of range
        hpgeom.query_circle(2048, 0.0, -0.1, 0.1, lonlat=False)

    with pytest.raises(ValueError, match=r"longitude \(phi\) .* out of range"):
        # phi out of range
        hpgeom.query_circle(2048, 0.0, 2*np.pi + 0.1, 0.1, lonlat=False)

    with pytest.raises(ValueError, match=r"Radius must be positive"):
        # Radius out of range
        hpgeom.query_circle(2048, 0.0, 0.0, 0.0)

    with pytest.raises(ValueError, match=r"Inclusive factor .* must be positive"):
        # Illegal fact (must be positive)
        hpgeom.query_circle(2048, 0.0, 0.0, 1.0, inclusive=True, fact=0)

    with pytest.raises(ValueError, match=r"Inclusive factor \* nside must be \<\="):
        # Illegal fact (too large)
        hpgeom.query_circle(2**28, 0.0, 0.0, 0.0001, inclusive=True, fact=4)

    with pytest.raises(ValueError, match=r"Inclusive factor .* must be power of 2 for nest"):
        # Illegal fact (must be power of 2 for nest)
        hpgeom.query_circle(2048, 0.0, 0.0, 1.0, inclusive=True, fact=3)

    with pytest.raises(RuntimeError, match=r"Can only use return_pixel_ranges with nest"):
        hpgeom.query_circle(2048, 0.0, 0.0, 1.0, nest=False, return_pixel_ranges=True)

    # Different platforms have different strings here, but they all say ``integer``.
    with pytest.raises(TypeError, match=r"integer"):
        # Illegal fact (must be integer)
        hpgeom.query_circle(2048, 0.0, 0.0, 1.0, inclusive=True, nest=False, fact=3.5)
