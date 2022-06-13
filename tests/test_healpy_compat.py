import numpy as np

import hpgeom
import hpgeom.healpy_compat as hpc


def test_ang2pix():
    """Test hpgeom.healpy_compat.ang2pix."""
    np.random.seed(12345)

    nside = 2048

    lon = np.random.uniform(low=0.0, high=360.0, size=1_000_000)
    lat = np.random.uniform(low=-90.0, high=90.0, size=1_000_000)

    phi = np.deg2rad(lon)
    theta = -np.deg2rad(lat) + np.pi/2.

    pix_hpgeom = hpgeom.angle_to_pixel(nside, theta, phi, nest=False, lonlat=False)
    pix_hpcompat = hpc.ang2pix(nside, theta, phi)
    np.testing.assert_array_equal(pix_hpcompat, pix_hpgeom)

    pix_hpgeom = hpgeom.angle_to_pixel(nside, theta, phi, nest=True, lonlat=False)
    pix_hpcompat = hpc.ang2pix(nside, theta, phi, nest=True)
    np.testing.assert_array_equal(pix_hpcompat, pix_hpgeom)

    pix_hpgeom = hpgeom.angle_to_pixel(nside, lon, lat, nest=False, lonlat=True)
    pix_hpcompat = hpc.ang2pix(nside, lon, lat, lonlat=True)
    np.testing.assert_array_equal(pix_hpcompat, pix_hpgeom)


def test_pix2ang():
    """Test hpgeom.healpy_compat.pix2ang."""
    np.random.seed(12345)

    nside = 2048

    pix = np.random.randint(low=0, high=12*nside*nside-1, size=1_000_000)

    theta_hpgeom, phi_hpgeom = hpgeom.pixel_to_angle(nside, pix, nest=False, lonlat=False)
    theta_hpcompat, phi_hpcompat = hpc.pix2ang(nside, pix)
    np.testing.assert_array_almost_equal(theta_hpcompat, theta_hpgeom)
    np.testing.assert_array_almost_equal(phi_hpcompat, phi_hpgeom)

    theta_hpgeom, phi_hpgeom = hpgeom.pixel_to_angle(nside, pix, nest=True, lonlat=False)
    theta_hpcompat, phi_hpcompat = hpc.pix2ang(nside, pix, nest=True)
    np.testing.assert_array_almost_equal(theta_hpcompat, theta_hpgeom)
    np.testing.assert_array_almost_equal(phi_hpcompat, phi_hpgeom)

    lon_hpgeom, lat_hpgeom = hpgeom.pixel_to_angle(nside, pix, nest=False, lonlat=True)
    lon_hpcompat, lat_hpcompat = hpc.pix2ang(nside, pix, lonlat=True)
    np.testing.assert_array_almost_equal(lon_hpcompat, lon_hpgeom)
    np.testing.assert_array_almost_equal(lat_hpcompat, lat_hpgeom)
