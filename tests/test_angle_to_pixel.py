import numpy as np
import pytest

try:
    import healpy as hp
    has_healpy = True
except ImportError:
    has_healpy = False

import hpgeom


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
@pytest.mark.parametrize("nside", [2**0, 2**5, 2**10, 2**15, 2**20, 2**25, 2**29, 2**10 - 100])
def test_angle_to_pixel_ring(nside):
    """Test angle_to_pixel for ring scheme."""
    np.random.seed(12345)

    lon = np.random.uniform(low=0.0, high=360.0, size=1_000_000)
    lat = np.random.uniform(low=-90.0, high=90.0, size=1_000_000)

    pix_hpgeom = hpgeom.angle_to_pixel(nside, lon, lat, nest=False, lonlat=True, degrees=True)
    pix_healpy = hp.ang2pix(nside, lon, lat, nest=False, lonlat=True)

    np.testing.assert_array_equal(pix_hpgeom, pix_healpy)

    pix_hpgeom = hpgeom.angle_to_pixel(
        nside,
        np.deg2rad(lon),
        np.deg2rad(lat),
        nest=False,
        lonlat=True,
        degrees=False
    )

    np.testing.assert_array_equal(pix_hpgeom, pix_healpy)

    phi = np.deg2rad(lon)
    theta = -np.deg2rad(lat) + np.pi/2.

    pix_hpgeom = hpgeom.angle_to_pixel(nside, theta, phi, nest=False, lonlat=False)
    pix_healpy = hp.ang2pix(nside, theta, phi, nest=False, lonlat=False)

    np.testing.assert_array_equal(pix_hpgeom, pix_healpy)


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
@pytest.mark.parametrize("nside", [2**0, 2**5, 2**10, 2**15, 2**20, 2**25, 2**29])
def test_angle_to_pixel_nest(nside):
    """Test angle_to_pixel for nest scheme."""
    np.random.seed(12345)

    lon = np.random.uniform(low=0.0, high=360.0, size=1_000_000)
    lat = np.random.uniform(low=-90.0, high=90.0, size=1_000_000)

    pix_hpgeom = hpgeom.angle_to_pixel(nside, lon, lat, nest=True, lonlat=True, degrees=True)
    pix_healpy = hp.ang2pix(nside, lon, lat, nest=True, lonlat=True)

    np.testing.assert_array_equal(pix_hpgeom, pix_healpy)

    pix_hpgeom = hpgeom.angle_to_pixel(
        nside,
        np.deg2rad(lon),
        np.deg2rad(lat),
        nest=True,
        lonlat=True,
        degrees=False
    )

    np.testing.assert_array_equal(pix_hpgeom, pix_healpy)

    theta, phi = hpgeom.lonlat_to_thetaphi(lon, lat)

    pix_hpgeom = hpgeom.angle_to_pixel(nside, theta, phi, nest=True, lonlat=False)
    pix_healpy = hp.ang2pix(nside, theta, phi, nest=True, lonlat=False)

    np.testing.assert_array_equal(pix_hpgeom, pix_healpy)


@pytest.mark.parametrize("nside", [2**0, 2**5, 2**10, 2**15, 2**20, 2**25, 2**29])
def test_angle_to_pixel_scalar(nside):
    """Test angle_to_pixel for scalars."""
    np.random.seed(12345)

    lon = np.random.uniform(low=0.0, high=360.0, size=100)
    lat = np.random.uniform(low=-90.0, high=90.0, size=100)

    pix_arr = hpgeom.angle_to_pixel(nside, lon, lat, nest=True, lonlat=True, degrees=True)

    pix_scalar1 = hpgeom.angle_to_pixel(nside, lon[0], lat[0], nest=True, lonlat=True, degrees=True)

    assert(pix_scalar1 == pix_arr[0])

    pix_scalar2 = hpgeom.angle_to_pixel(
        nside,
        float(lon[0]),
        float(lat[0]),
        nest=True,
        lonlat=True,
        degrees=True
    )

    assert(pix_scalar2 == pix_arr[0])


def test_angle_to_pixel_mismatched_dims():
    """Test angle_to_pixel errors when dimensions are mismatched."""
    np.random.seed(12345)

    lon = np.random.uniform(low=0.0, high=360.0, size=100)
    lat = np.random.uniform(low=-90.0, high=90.0, size=100)

    with pytest.raises(ValueError, match=r"arrays could not be broadcast together"):
        hpgeom.angle_to_pixel([2048, 4096], lon[0], lat)

    with pytest.raises(ValueError, match=r"arrays could not be broadcast together"):
        hpgeom.angle_to_pixel([2048, 4096], lon, lat[0])

    with pytest.raises(ValueError, match=r"arrays could not be broadcast together"):
        hpgeom.angle_to_pixel(2048, lon[0: 5], lat)

    with pytest.raises(ValueError, match=r"arrays could not be broadcast together"):
        hpgeom.angle_to_pixel(2048, lon, lat[0: 5])

    with pytest.raises(ValueError, match=r"arrays could not be broadcast together"):
        hpgeom.angle_to_pixel(2048, lon.reshape((10, 10)), lat)

    with pytest.raises(ValueError, match=r"arrays could not be broadcast together"):
        hpgeom.angle_to_pixel(2048, lon, lat.reshape((10, 10)))


def test_angle_to_pixel_bad_nside():
    """Test angle_to_pixel errors when given a bad nside."""
    np.random.seed(12345)

    lon = np.random.uniform(low=0.0, high=360.0, size=100)
    lat = np.random.uniform(low=-90.0, high=90.0, size=100)

    with pytest.raises(ValueError, match=r"nside .* must be positive"):
        hpgeom.angle_to_pixel(-10, lon, lat, nest=False)

    with pytest.raises(ValueError, match=r"nside .* must be positive"):
        hpgeom.angle_to_pixel(-10, lon, lat, nest=True)

    with pytest.raises(ValueError, match=r"nside .* must be power of 2"):
        hpgeom.angle_to_pixel(2040, lon, lat, nest=True)

    with pytest.raises(ValueError, match=r"nside .* must not be greater"):
        hpgeom.angle_to_pixel(2**30, lon, lat, nest=True)


def test_angle_to_pixel_bad_coords():
    """Test angle_to_pixel errors when given bad coords."""
    with pytest.raises(ValueError, match=r"lat .* out of range"):
        # Dec out of range
        hpgeom.angle_to_pixel(2048, 0.0, 100.0)

    with pytest.raises(ValueError, match=r"lat .* out of range"):
        # Dec out of range
        hpgeom.angle_to_pixel(2048, 0.0, -100.0)

    with pytest.raises(ValueError, match=r"colatitude \(theta\) .* out of range"):
        # theta out of range
        hpgeom.angle_to_pixel(2048, -0.1, 0.0, lonlat=False)

    with pytest.raises(ValueError, match=r"colatitude \(theta\) .* out of range"):
        # theta out of range
        hpgeom.angle_to_pixel(2048, np.pi + 0.1, 0.0, lonlat=False)

    with pytest.raises(ValueError, match=r"longitude \(phi\) .* out of range"):
        # phi out of range
        hpgeom.angle_to_pixel(2048, 0.0, -0.1, lonlat=False)

    with pytest.raises(ValueError, match=r"longitude \(phi\) .* out of range"):
        # phi out of range
        hpgeom.angle_to_pixel(2048, 0.0, 2*np.pi + 0.1, lonlat=False)
