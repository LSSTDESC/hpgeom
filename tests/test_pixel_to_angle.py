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
def test_pixel_to_angle_ring(nside):
    """Test pixel_to_angle for ring scheme."""
    np.random.seed(12345)

    pix = np.random.randint(low=0, high=12*nside*nside-1, size=1_000_000, dtype=np.int64)

    lon_hpgeom, lat_hpgeom = hpgeom.pixel_to_angle(nside, pix, nest=False, lonlat=True, degrees=True)
    lon_healpy, lat_healpy = hp.pix2ang(nside, pix, nest=False, lonlat=True)

    np.testing.assert_array_almost_equal(lon_hpgeom, lon_healpy)
    np.testing.assert_array_almost_equal(lat_hpgeom, lat_healpy)

    lon_hpgeom, lat_hpgeom = hpgeom.pixel_to_angle(
        nside,
        pix,
        nest=False,
        lonlat=True,
        degrees=False
    )

    np.testing.assert_array_almost_equal(np.rad2deg(lon_hpgeom), lon_healpy)
    np.testing.assert_array_almost_equal(np.rad2deg(lat_hpgeom), lat_healpy)

    theta_hpgeom, phi_hpgeom = hpgeom.pixel_to_angle(nside, pix, nest=False, lonlat=False)
    theta_healpy, phi_healpy = hp.pix2ang(nside, pix, nest=False, lonlat=False)

    np.testing.assert_array_almost_equal(theta_hpgeom, theta_healpy)
    np.testing.assert_array_almost_equal(phi_hpgeom, phi_healpy)


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
@pytest.mark.parametrize("nside", [2**0, 2**5, 2**10, 2**15, 2**20, 2**25, 2**29])
def test_pixel_to_angle_nest(nside):
    """Test pixel_to_angle for nest scheme."""
    np.random.seed(12345)

    pix = np.random.randint(low=0, high=12*nside*nside-1, size=1_000_000, dtype=np.int64)

    lon_hpgeom, lat_hpgeom = hpgeom.pixel_to_angle(nside, pix, nest=True, lonlat=True, degrees=True)
    lon_healpy, lat_healpy = hp.pix2ang(nside, pix, nest=True, lonlat=True)

    np.testing.assert_array_almost_equal(lon_hpgeom, lon_healpy)
    np.testing.assert_array_almost_equal(lat_hpgeom, lat_healpy)

    lon_hpgeom, lat_hpgeom = hpgeom.pixel_to_angle(
        nside,
        pix,
        nest=True,
        lonlat=True,
        degrees=False
    )

    np.testing.assert_array_almost_equal(np.rad2deg(lon_hpgeom), lon_healpy)
    np.testing.assert_array_almost_equal(np.rad2deg(lat_hpgeom), lat_healpy)

    theta_hpgeom, phi_hpgeom = hpgeom.pixel_to_angle(nside, pix, nest=True, lonlat=False)
    theta_healpy, phi_healpy = hp.pix2ang(nside, pix, nest=True, lonlat=False)

    np.testing.assert_array_almost_equal(theta_hpgeom, theta_healpy)
    np.testing.assert_array_almost_equal(phi_hpgeom, phi_healpy)


@pytest.mark.parametrize("nside", [2**0, 2**5, 2**10, 2**15, 2**20, 2**25, 2**29])
def test_pixel_to_angle_scalar(nside):
    """Test pixel_to_angle for scalars."""
    np.random.seed(12345)

    pix = np.random.randint(low=0, high=12*nside*nside-1, size=100, dtype=np.int64)

    lon_arr, lat_arr = hpgeom.pixel_to_angle(nside, pix, nest=True, lonlat=True, degrees=True)
    lon_scalar1, lat_scalar1 = hpgeom.pixel_to_angle(nside, pix[0], nest=True, lonlat=True, degrees=True)

    assert lon_scalar1 == lon_arr[0]
    assert lat_scalar1 == lat_arr[0]

    lon_scalar2, lat_scalar2 = hpgeom.pixel_to_angle(
        nside,
        int(pix[0]),
        nest=True,
        lonlat=True,
        degrees=True
    )

    assert lon_scalar2 == lon_arr[0]
    assert lat_scalar2 == lat_arr[0]


@pytest.mark.parametrize("nside", [2**0, 2**5, 2**10, 2**15, 2**20, 2**25, 2**29])
def test_pixel_to_angle_bad_pix(nside):
    """Test pixel_to_angle errors when given bad pixel"""

    with pytest.raises(ValueError, match=r"Pixel value .* out of range"):
        hpgeom.pixel_to_angle(nside, -1)

    with pytest.raises(ValueError, match=r"Pixel value .* out of range"):
        hpgeom.pixel_to_angle(nside, 12*nside*nside)


def test_pixel_to_angle_bad_nside():
    """Test pixel_to_angle errors when given a bad nside."""
    np.random.seed(12345)

    pix = np.random.randint(low=0, high=12*2048*2048-1, size=100, dtype=np.int64)

    with pytest.raises(ValueError, match=r"nside .* must be positive"):
        hpgeom.pixel_to_angle(-10, pix, nest=False)

    with pytest.raises(ValueError, match=r"nside .* must be positive"):
        hpgeom.pixel_to_angle(-10, pix, nest=True)

    with pytest.raises(ValueError, match=r"nside .* must be power of 2"):
        hpgeom.pixel_to_angle(2040, pix, nest=True)

    with pytest.raises(ValueError, match=r"nside .* must not be greater"):
        hpgeom.pixel_to_angle(2**30, pix, nest=True)
