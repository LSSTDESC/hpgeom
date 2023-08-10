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
def test_interpolation_ring(nside):
    """Test get_interpolation_weights for ring scheme."""
    np.random.seed(12345)

    n_point = 1_000_000

    lon = np.random.uniform(low=0.0, high=360.0, size=n_point)
    lat = np.random.uniform(low=-90.0, high=90.0, size=n_point)

    interp_pix_hpgeom, interp_wgt_hpgeom = hpgeom.get_interpolation_weights(
        nside,
        lon,
        lat,
        nest=False,
        lonlat=True,
        degrees=True,
    )
    assert interp_pix_hpgeom.shape == (n_point, 4)
    assert interp_wgt_hpgeom.shape == (n_point, 4)

    interp_pix_healpy, interp_wgt_healpy = hp.get_interp_weights(
        nside,
        lon,
        lat,
        nest=False,
        lonlat=True,
    )

    np.testing.assert_array_equal(interp_pix_hpgeom, interp_pix_healpy.T)
    np.testing.assert_array_almost_equal(interp_wgt_hpgeom, interp_wgt_healpy.T)


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
@pytest.mark.parametrize("nside", [2**0, 2**5, 2**10, 2**15, 2**20, 2**25, 2**29])
def test_interpolation_nest(nside):
    """Test get_interpolation_weights for nest scheme."""
    np.random.seed(12345)

    n_point = 1_000_000

    lon = np.random.uniform(low=0.0, high=360.0, size=n_point)
    lat = np.random.uniform(low=-90.0, high=90.0, size=n_point)

    interp_pix_hpgeom, interp_wgt_hpgeom = hpgeom.get_interpolation_weights(
        nside,
        lon,
        lat,
        nest=True,
        lonlat=True,
        degrees=True,
    )
    assert interp_pix_hpgeom.shape == (n_point, 4)
    assert interp_wgt_hpgeom.shape == (n_point, 4)

    interp_pix_healpy, interp_wgt_healpy = hp.get_interp_weights(
        nside,
        lon,
        lat,
        nest=True,
        lonlat=True,
    )

    np.testing.assert_array_equal(interp_pix_hpgeom, interp_pix_healpy.T)
    np.testing.assert_array_almost_equal(interp_wgt_hpgeom, interp_wgt_healpy.T)


@pytest.mark.parametrize("nside", [2**0, 2**5, 2**10, 2**15, 2**20, 2**25, 2**29])
def test_interpolation_scalar(nside):
    """Test get_interpolation_weights for scalars."""
    np.random.seed(12345)

    n_point = 100

    lon = np.random.uniform(low=0.0, high=360.0, size=n_point)
    lat = np.random.uniform(low=-90.0, high=90.0, size=n_point)

    interp_pix, interp_wgt = hpgeom.get_interpolation_weights(
        nside,
        lon,
        lat,
        nest=True,
        lonlat=True,
        degrees=True,
    )

    interp_pix_scalar, interp_wgt_scalar = hpgeom.get_interpolation_weights(
        nside,
        float(lon[0]),
        float(lat[0]),
        nest=True,
        lonlat=True,
        degrees=True,
    )

    assert interp_pix_scalar.shape == (4,)
    assert interp_wgt_scalar.shape == (4,)

    np.testing.assert_array_equal(interp_pix_scalar, interp_pix[0, :])
    np.testing.assert_array_almost_equal(interp_wgt_scalar, interp_wgt[0, :])


def test_interpolation_mismatched_dims():
    """Test get_interpolation_weights when dimensions are mismatched."""
    np.random.seed(12345)

    lon = np.random.uniform(low=0.0, high=360.0, size=100)
    lat = np.random.uniform(low=-90.0, high=90.0, size=100)

    with pytest.raises(ValueError, match=r"must have same number of dimensions"):
        hpgeom.get_interpolation_weights([2048, 4096], lon[0], lat)

    with pytest.raises(ValueError, match=r"must have same number of dimensions"):
        hpgeom.get_interpolation_weights([2048, 4096], lon, lat[0])

    with pytest.raises(ValueError, match=r"arrays could not be broadcast together"):
        hpgeom.get_interpolation_weights(2048, lon[0: 5], lat)

    with pytest.raises(ValueError, match=r"arrays could not be broadcast together"):
        hpgeom.get_interpolation_weights(2048, lon, lat[0: 5])

    with pytest.raises(ValueError, match=r"array must be at most 1D"):
        hpgeom.get_interpolation_weights(2048, lon.reshape((10, 10)), lat)

    with pytest.raises(ValueError, match=r"must have same number of dimensions"):
        hpgeom.get_interpolation_weights(2048, lon, lat.reshape((10, 10)))


def test_interpolation_bad_nside():
    """Test get_interpolation_weights when given a bad nside."""
    np.random.seed(12345)

    lon = np.random.uniform(low=0.0, high=360.0, size=100)
    lat = np.random.uniform(low=-90.0, high=90.0, size=100)

    with pytest.raises(ValueError, match=r"nside .* must be positive"):
        hpgeom.get_interpolation_weights(-10, lon, lat, nest=False)

    with pytest.raises(ValueError, match=r"nside .* must be positive"):
        hpgeom.get_interpolation_weights(-10, lon, lat, nest=True)

    with pytest.raises(ValueError, match=r"nside .* must be power of 2"):
        hpgeom.get_interpolation_weights(2040, lon, lat, nest=True)

    with pytest.raises(ValueError, match=r"nside .* must not be greater"):
        hpgeom.get_interpolation_weights(2**30, lon, lat, nest=True)


def test_interpolation_bad_coords():
    """Test get_interpolation_weights when given bad coords."""
    with pytest.raises(ValueError, match=r"lat .* out of range"):
        # Dec out of range
        hpgeom.get_interpolation_weights(2048, 0.0, 100.0)

    with pytest.raises(ValueError, match=r"lat .* out of range"):
        # Dec out of range
        hpgeom.get_interpolation_weights(2048, 0.0, -100.0)

    with pytest.raises(ValueError, match=r"colatitude \(theta\) .* out of range"):
        # theta out of range
        hpgeom.get_interpolation_weights(2048, -0.1, 0.0, lonlat=False)

    with pytest.raises(ValueError, match=r"colatitude \(theta\) .* out of range"):
        # theta out of range
        hpgeom.get_interpolation_weights(2048, np.pi + 0.1, 0.0, lonlat=False)

    with pytest.raises(ValueError, match=r"longitude \(phi\) .* out of range"):
        # phi out of range
        hpgeom.get_interpolation_weights(2048, 0.0, -0.1, lonlat=False)

    with pytest.raises(ValueError, match=r"longitude \(phi\) .* out of range"):
        # phi out of range
        hpgeom.get_interpolation_weights(2048, 0.0, 2*np.pi + 0.1, lonlat=False)
