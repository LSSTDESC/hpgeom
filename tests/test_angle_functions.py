import numpy as np
import pytest

try:
    import healpy as hp
    has_healpy = True
except ImportError:
    has_healpy = False

import hpgeom


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
def test_thetaphi_to_lonlat():
    """Test thetaphi_to_lonlat."""
    np.random.seed(12345)

    theta = np.random.uniform(0, np.pi, 1_000_000)
    phi = np.random.uniform(0, 2*np.pi, 1_000_000)

    lon_hpgeom, lat_hpgeom = hpgeom.thetaphi_to_lonlat(float(theta[0]), float(phi[0]))
    lon_healpy, lat_healpy = hp.pixelfunc.thetaphi2lonlat(float(theta[0]), float(phi[0]))

    np.testing.assert_almost_equal(lon_hpgeom, lon_healpy)
    np.testing.assert_almost_equal(lat_hpgeom, lat_healpy)

    lon_hpgeom, lat_hpgeom = hpgeom.thetaphi_to_lonlat(theta, phi)
    lon_healpy, lat_healpy = hp.pixelfunc.thetaphi2lonlat(theta, phi)

    np.testing.assert_array_almost_equal(lon_hpgeom, lon_healpy)
    np.testing.assert_array_almost_equal(lat_hpgeom, lat_healpy)


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
def test_lonlat_to_thetaphi():
    """Test thetaphi_to_lonlat."""
    np.random.seed(12345)

    lon = np.random.uniform(0, 360.0, 1_000_000)
    lat = np.random.uniform(-90.0, 90.0, 1_000_000)

    theta_hpgeom, phi_hpgeom = hpgeom.thetaphi_to_lonlat(float(lon[0]), float(lat[0]))
    theta_healpy, phi_healpy = hp.pixelfunc.thetaphi2lonlat(float(lon[0]), float(lat[0]))

    np.testing.assert_almost_equal(theta_hpgeom, theta_healpy)
    np.testing.assert_almost_equal(phi_hpgeom, phi_healpy)

    theta_hpgeom, phi_hpgeom = hpgeom.thetaphi_to_lonlat(lon, lat)
    theta_healpy, phi_healpy = hp.pixelfunc.thetaphi2lonlat(lon, lat)

    np.testing.assert_array_almost_equal(theta_hpgeom, theta_healpy)
    np.testing.assert_array_almost_equal(phi_hpgeom, phi_healpy)
