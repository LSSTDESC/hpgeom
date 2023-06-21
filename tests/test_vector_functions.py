import numpy as np
import pytest

try:
    import healpy as hp
    has_healpy = True
except ImportError:
    has_healpy = False

import hpgeom


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
def test_angle_to_vector():
    """Test angle_to_vector."""
    np.random.seed(12345)

    lon = np.random.uniform(0, 360.0, 1_000_000)
    lat = np.random.uniform(-90.0, 90.0, 1_000_000)

    vec_hpgeom = hpgeom.angle_to_vector(lon, lat)
    vec_healpy = hp.ang2vec(lon, lat, lonlat=True)

    np.testing.assert_array_almost_equal(vec_hpgeom, vec_healpy)

    vec_hpgeom = hpgeom.angle_to_vector(float(lon[0]), float(lat[0]))
    vec_healpy = hp.ang2vec(float(lon[0]), float(lat[0]), lonlat=True)

    assert vec_hpgeom.shape == (3, )
    np.testing.assert_array_almost_equal(vec_hpgeom, vec_healpy)

    theta, phi = hpgeom.lonlat_to_thetaphi(lon, lat)

    vec_hpgeom = hpgeom.angle_to_vector(theta, phi, lonlat=False)
    vec_healpy = hp.ang2vec(theta, phi)

    np.testing.assert_array_almost_equal(vec_hpgeom, vec_healpy)


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
def test_vector_to_angle():
    """Test angle_to_vector."""
    np.random.seed(12345)

    # Does not need to be normalized.
    vec = np.zeros((1_000_000, 3))
    vec[:, 0] = np.random.uniform(-1, 1, 1_000_000)
    vec[:, 1] = np.random.uniform(-1, 1, 1_000_000)
    vec[:, 2] = np.random.uniform(-1, 1, 1_000_000)

    lon_hpgeom, lat_hpgeom = hpgeom.vector_to_angle(vec)
    lon_healpy, lat_healpy = hp.vec2ang(vec, lonlat=True)

    np.testing.assert_array_almost_equal(lon_hpgeom, lon_healpy)
    np.testing.assert_array_almost_equal(lat_hpgeom, lat_healpy)

    theta_hpgeom, phi_hpgeom = hpgeom.vector_to_angle(vec, lonlat=False)
    theta_healpy, phi_healpy = hp.vec2ang(vec)

    np.testing.assert_array_almost_equal(theta_hpgeom, theta_healpy)
    np.testing.assert_array_almost_equal(phi_hpgeom, phi_healpy)


def test_angle_to_vector_badinput():
    """Test angle_to_vector."""

    with pytest.raises(ValueError, match=r"Latitude out of range"):
        hpgeom.angle_to_vector(0.0, 100.0, lonlat=True)

    with pytest.raises(ValueError, match=r"Co-latitude \(theta\) out of range"):
        hpgeom.angle_to_vector(2*np.pi, 0.0, lonlat=False)
