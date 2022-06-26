import numpy as np
import pytest

try:
    import healpy as hp
    has_healpy = True
except ImportError:
    has_healpy = False

import hpgeom
import hpgeom.healpy_compat as hpc


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
def test_ang2pix():
    """Test hpgeom.healpy_compat.ang2pix."""
    np.random.seed(12345)

    nside = 2048

    lon = np.random.uniform(low=0.0, high=360.0, size=1_000_000)
    lat = np.random.uniform(low=-90.0, high=90.0, size=1_000_000)
    theta, phi = hpgeom.lonlat_to_thetaphi(lon, lat)

    pix_hpcompat = hpc.ang2pix(nside, theta, phi)
    pix_healpy = hp.ang2pix(nside, theta, phi)
    np.testing.assert_array_equal(pix_hpcompat, pix_healpy)

    pix_hpcompat = hpc.ang2pix(nside, theta, phi, nest=True)
    pix_healpy = hp.ang2pix(nside, theta, phi, nest=True)
    np.testing.assert_array_equal(pix_hpcompat, pix_healpy)

    pix_hpcompat = hpc.ang2pix(nside, lon, lat, lonlat=True)
    pix_healpy = hpc.ang2pix(nside, lon, lat, lonlat=True)
    np.testing.assert_array_equal(pix_hpcompat, pix_healpy)


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
def test_pix2ang():
    """Test hpgeom.healpy_compat.pix2ang."""
    np.random.seed(12345)

    nside = 2048

    pix = np.random.randint(low=0, high=12*nside*nside-1, size=1_000_000, dtype=np.int64)

    theta_hpcompat, phi_hpcompat = hpc.pix2ang(nside, pix)
    theta_healpy, phi_healpy = hp.pix2ang(nside, pix)
    np.testing.assert_array_almost_equal(theta_hpcompat, theta_healpy)
    np.testing.assert_array_almost_equal(phi_hpcompat, phi_healpy)

    theta_hpcompat, phi_hpcompat = hpc.pix2ang(nside, pix, nest=True)
    theta_healpy, phi_healpy = hp.pix2ang(nside, pix, nest=True)
    np.testing.assert_array_almost_equal(theta_hpcompat, theta_healpy)
    np.testing.assert_array_almost_equal(phi_hpcompat, phi_healpy)

    lon_hpcompat, lat_hpcompat = hpc.pix2ang(nside, pix, lonlat=True)
    lon_healpy, lat_healpy = hp.pix2ang(nside, pix, lonlat=True)
    np.testing.assert_array_almost_equal(lon_hpcompat, lon_healpy)
    np.testing.assert_array_almost_equal(lat_hpcompat, lat_healpy)


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
def test_query_disc():
    """Test hpgeom.healpy_compat.query_disc."""
    np.random.seed(12345)

    nside = 2048
    lon = 10.0
    lat = 20.0
    radius = 0.5

    theta, phi = hpgeom.lonlat_to_thetaphi(lon, lat)
    sintheta = np.sin(theta)
    vec = [sintheta*np.cos(phi), sintheta*np.sin(phi), np.cos(theta)]

    pix_hpcompat = hpc.query_disc(nside, vec, radius)
    pix_healpy = hp.query_disc(nside, vec, radius)
    np.testing.assert_array_equal(pix_hpcompat, pix_healpy)

    pix_hpcompat = hpc.query_disc(nside, vec, radius, nest=True)
    pix_healpy = hp.query_disc(nside, vec, radius, nest=True)
    np.testing.assert_array_equal(pix_hpcompat, pix_healpy)

    pix_hpcompat = hpc.query_disc(nside, vec, radius, inclusive=True)
    pix_healpy = hp.query_disc(nside, vec, radius, inclusive=True)
    np.testing.assert_array_equal(pix_hpcompat, pix_healpy)


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
def test_ring2nest():
    """Test hpgeom.healpy_compat.ring2nest."""
    pix_hpcompat = hpc.ring2nest(2048, 1000)
    pix_healpy = hp.ring2nest(2048, 1000)

    assert(pix_hpcompat == pix_healpy)

    pix_hpcompat = hpc.ring2nest(2048, np.arange(100))
    pix_healpy = hp.ring2nest(2048, np.arange(100))

    np.testing.assert_array_equal(pix_hpcompat, pix_healpy)


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
def test_nest2ring():
    """Test hpgeom.healpy_compat.nest2ring."""
    pix_hpcompat = hpc.nest2ring(2048, 1000)
    pix_healpy = hp.nest2ring(2048, 1000)

    assert(pix_hpcompat == pix_healpy)

    pix_hpcompat = hpc.nest2ring(2048, np.arange(100))
    pix_healpy = hp.nest2ring(2048, np.arange(100))

    np.testing.assert_array_equal(pix_hpcompat, pix_healpy)


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
def test_nside2npix():
    """Test hpgeom.healpy_compat.nside2npix."""
    npix_hpcompat = hpc.nside2npix(2048)
    npix_healpy = hp.nside2npix(2048)

    assert(npix_hpcompat == npix_healpy)


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
def test_npix2nside():
    """Test hpgeom.healpy_compat.npix2nside."""
    nside_hpcompat = hpc.nside2npix(12*2048*2048)
    nside_healpy = hp.nside2npix(12*2048*2048)

    assert(nside_hpcompat == nside_healpy)

    with pytest.raises(ValueError, match=r"Illegal npixel"):
        hpc.npix2nside(100)


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
def test_nside2pixarea():
    """Test hpgeom.healpy_compat.nside2pixarea."""
    pixarea_hpcompat = hpc.nside2pixarea(1024)
    pixarea_healpy = hp.nside2pixarea(1024)

    assert (pixarea_hpcompat == pixarea_healpy)

    pixarea_hpcompat = hpc.nside2pixarea(1024, degrees=True)
    pixarea_healpy = hp.nside2pixarea(1024, degrees=True)

    assert (pixarea_hpcompat == pixarea_healpy)


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
def test_nside2resol():
    """Test hpgeom.healpy_compat.nside2resol."""
    resol_hpcompat = hpc.nside2resol(1024)
    resol_healpy = hp.nside2resol(1024)

    assert (resol_hpcompat == resol_healpy)

    resol_hpcompat = hpc.nside2resol(1024, arcmin=True)
    resol_healpy = hp.nside2resol(1024, arcmin=True)

    assert (resol_hpcompat == resol_healpy)


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
def test_nside2order():
    """Test hpgeom.healpy_compat.nside2order."""
    order_hpcompat = hpc.nside2order(1024)
    order_healpy = hp.nside2order(1024)

    assert (order_hpcompat == order_healpy)


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
def test_order2nside():
    """Test hpgeom.healpy_compat.order2nside."""
    nside_hpcompat = hpc.order2nside(10)
    nside_healpy = hp.order2nside(10)

    assert (nside_hpcompat == nside_healpy)


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
def test_ang2vec():
    """Test hpgeom.healpy_compat.ang2vec."""
    np.random.seed(12345)

    lon = np.random.uniform(0, 360.0, 1000)
    lat = np.random.uniform(-90.0, 90.0, 1000)

    vec_hpgeom = hpc.ang2vec(lon, lat, lonlat=True)
    vec_healpy = hp.ang2vec(lon, lat, lonlat=True)

    np.testing.assert_array_almost_equal(vec_hpgeom, vec_healpy)

    theta, phi = hpgeom.lonlat_to_thetaphi(lon, lat)

    vec_hpgeom = hpc.ang2vec(theta, phi)
    vec_healpy = hp.ang2vec(theta, phi)

    np.testing.assert_array_almost_equal(vec_hpgeom, vec_healpy)


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
def test_vec2ang():
    """Test hpgeom.healpy_compat.vec2ang."""
    np.random.seed(12345)

    vec = np.zeros((1000, 3))
    vec[:, 0] = np.random.uniform(-1, 1, 1000)
    vec[:, 1] = np.random.uniform(-1, 1, 1000)
    vec[:, 2] = np.random.uniform(-1, 1, 1000)

    theta_hpgeom, phi_hpgeom = hpc.vec2ang(vec)
    theta_healpy, phi_healpy = hp.vec2ang(vec)

    np.testing.assert_array_almost_equal(theta_hpgeom, theta_healpy)
    np.testing.assert_array_almost_equal(phi_hpgeom, phi_healpy)

    lon_hpgeom, lat_hpgeom = hpc.vec2ang(vec, lonlat=True)
    lon_healpy, lat_healpy = hp.vec2ang(vec, lonlat=True)

    np.testing.assert_array_almost_equal(lon_hpgeom, lon_healpy)
    np.testing.assert_array_almost_equal(lat_hpgeom, lat_healpy)


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
def test_vec2pix():
    """Test hpgeom.healpy_compat.vec2pix."""
    np.random.seed(12345)

    nside = 2048

    vec = np.zeros((1000, 3))
    vec[:, 0] = np.random.uniform(-1, 1, 1000)
    vec[:, 1] = np.random.uniform(-1, 1, 1000)
    vec[:, 2] = np.random.uniform(-1, 1, 1000)

    pix_hpgeom = hpc.vec2pix(nside, vec[:, 0], vec[:, 1], vec[:, 2])
    pix_healpy = hp.vec2pix(nside, vec[:, 0], vec[:, 1], vec[:, 2])

    np.testing.assert_array_equal(pix_hpgeom, pix_healpy)

    pix_hpgeom = hpc.vec2pix(nside, vec[:, 0], vec[:, 1], vec[:, 2], nest=True)
    pix_healpy = hp.vec2pix(nside, vec[:, 0], vec[:, 1], vec[:, 2], nest=True)

    np.testing.assert_array_equal(pix_hpgeom, pix_healpy)


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
def test_pix2vec():
    """Test hpgeom.healpy_compat.pix2vec."""
    np.random.seed(12345)

    nside = 2048

    pix = np.random.randint(low=0, high=12*nside*nside-1, size=1000, dtype=np.int64)

    x_hpgeom, y_hpgeom, z_hpgeom = hpc.pix2vec(nside, pix)
    x_healpy, y_healpy, z_healpy = hp.pix2vec(nside, pix)

    np.testing.assert_array_almost_equal(x_hpgeom, x_healpy)
    np.testing.assert_array_almost_equal(y_hpgeom, y_healpy)
    np.testing.assert_array_almost_equal(z_hpgeom, z_healpy)

    x_hpgeom, y_hpgeom, z_hpgeom = hpc.pix2vec(nside, pix, nest=True)
    x_healpy, y_healpy, z_healpy = hp.pix2vec(nside, pix, nest=True)

    np.testing.assert_array_almost_equal(x_hpgeom, x_healpy)
    np.testing.assert_array_almost_equal(y_hpgeom, y_healpy)
    np.testing.assert_array_almost_equal(z_hpgeom, z_healpy)


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
def test_boundaries():
    """Test hpgeom.healpy_compat.boundaries."""
    # Test single pixel.

    vec_hpgeom = hpc.boundaries(1024, 1000)
    vec_healpy = hp.boundaries(1024, 1000)

    np.testing.assert_array_almost_equal(vec_hpgeom, vec_healpy)

    vec_hpgeom = hpc.boundaries(1024, 1000, step=4)
    vec_healpy = hp.boundaries(1024, 1000, step=4)

    np.testing.assert_array_almost_equal(vec_hpgeom, vec_healpy)

    vec_hpgeom = hpc.boundaries(1024, 1000, nest=False)
    vec_healpy = hp.boundaries(1024, 1000, nest=False)

    np.testing.assert_array_almost_equal(vec_hpgeom, vec_healpy)

    # Test multiple pixels.
    vec_hpgeom = hpc.boundaries(1024, [1000, 1200])
    vec_healpy = hp.boundaries(1024, [1000, 1200])

    np.testing.assert_array_almost_equal(vec_hpgeom, vec_healpy)

    vec_hpgeom = hpc.boundaries(1024, [1000, 1200], step=4)
    vec_healpy = hp.boundaries(1024, [1000, 1200], step=4)

    np.testing.assert_array_almost_equal(vec_hpgeom, vec_healpy)

    vec_hpgeom = hpc.boundaries(1024, [1000, 1200], nest=False)
    vec_healpy = hp.boundaries(1024, [1000, 1200], nest=False)

    np.testing.assert_array_almost_equal(vec_hpgeom, vec_healpy)


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
def test_get_all_neighbours():
    """Test hpgeom.healpy_compat.get_all_neighbours."""
    # Test single pixel
    neighbors_hpgeom = hpc.get_all_neighbours(1024, 100)
    neighbors_healpy = hp.get_all_neighbours(1024, 100)

    np.testing.assert_array_equal(neighbors_hpgeom, neighbors_healpy)

    # Test multiple pixels
    neighbors_hpgeom = hpc.get_all_neighbours(1024, [100, 200])
    neighbors_healpy = hp.get_all_neighbours(1024, [100, 200])

    np.testing.assert_array_equal(neighbors_hpgeom, neighbors_healpy)

    # Test single theta/phi
    neighbors_hpgeom = hpc.get_all_neighbours(1024, 0.5, phi=0.5)
    neighbors_healpy = hp.get_all_neighbours(1024, 0.5, phi=0.5)

    np.testing.assert_array_equal(neighbors_hpgeom, neighbors_healpy)

    # Test multiple theta/phi
    neighbors_hpgeom = hpc.get_all_neighbours(1024, [0.5, 0.6], phi=[0.5, 0.6])
    neighbors_healpy = hp.get_all_neighbours(1024, [0.5, 0.6], phi=[0.5, 0.6])

    np.testing.assert_array_equal(neighbors_hpgeom, neighbors_healpy)

    # Test single lon/lat
    neighbors_hpgeom = hpc.get_all_neighbours(1024, 0.5, phi=0.5, lonlat=True)
    neighbors_healpy = hp.get_all_neighbours(1024, 0.5, phi=0.5, lonlat=True)

    np.testing.assert_array_equal(neighbors_hpgeom, neighbors_healpy)

    # Test multiple lon/lat
    neighbors_hpgeom = hpc.get_all_neighbours(1024, [0.5, 0.6], phi=[0.5, 0.6], lonlat=True)
    neighbors_healpy = hp.get_all_neighbours(1024, [0.5, 0.6], phi=[0.5, 0.6], lonlat=True)

    np.testing.assert_array_equal(neighbors_hpgeom, neighbors_healpy)

    # Test multiple lon/lat, nest=True
    neighbors_hpgeom = hpc.get_all_neighbours(1024, [0.5, 0.6], phi=[0.5, 0.6], lonlat=True, nest=True)
    neighbors_healpy = hp.get_all_neighbours(1024, [0.5, 0.6], phi=[0.5, 0.6], lonlat=True, nest=True)

    np.testing.assert_array_equal(neighbors_hpgeom, neighbors_healpy)


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
def test_query_polygon():
    """Test hpgeom.healpy_compat.query_polygon"""
    nside = 1024
    delta = 1.0
    lon_ref = 180.0
    lat_ref = 0.0
    lon = np.array([lon_ref, lon_ref + delta, lon_ref + delta, lon_ref])
    lat = np.array([lat_ref, lat_ref, lat_ref + delta, lat_ref + delta])
    vec = hpgeom.angle_to_vector(lon, lat)

    pix_hpcompat = hpc.query_polygon(nside, vec)
    pix_healpy = hp.query_polygon(nside, vec)
    np.testing.assert_array_equal(pix_hpcompat, pix_healpy)

    pix_hpcompat = hpc.query_polygon(nside, vec, nest=True)
    pix_healpy = hp.query_polygon(nside, vec, nest=True)
    np.testing.assert_array_equal(pix_hpcompat, pix_healpy)

    pix_hpcompat = hpc.query_polygon(nside, vec, inclusive=True)
    pix_healpy = hp.query_polygon(nside, vec, inclusive=True)
    np.testing.assert_array_equal(pix_hpcompat, pix_healpy)
