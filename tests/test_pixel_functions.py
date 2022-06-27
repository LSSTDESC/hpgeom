import numpy as np
import pytest

try:
    import healpy as hp
    has_healpy = True
except ImportError:
    has_healpy = False

import hpgeom


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
@pytest.mark.parametrize("nside", [2**0, 2**5, 2**10, 2**20, 2**29])
def test_nside_to_npixel(nside):
    """Test nside_to_npixel."""
    npixel_hpgeom = hpgeom.nside_to_npixel(nside)
    npixel_healpy = hp.nside2npix(nside)

    assert (npixel_hpgeom == npixel_healpy)

    npixel_hpgeom = hpgeom.nside_to_npixel(np.array([nside, nside]))
    npixel_healpy = hp.nside2npix(np.array([nside, nside]))

    np.testing.assert_array_equal(npixel_hpgeom, npixel_healpy)


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
@pytest.mark.parametrize("nside", [2**0, 2**5, 2**10, 2**20, 2**29])
def test_npixel_to_nside(nside):
    """Test npixel_to_nside."""
    npixel = 12*nside*nside

    nside_hpgeom = hpgeom.npixel_to_nside(npixel)
    nside_healpy = hp.npix2nside(npixel)

    assert (nside_hpgeom == nside_healpy)

    nside_hpgeom = hpgeom.npixel_to_nside(np.array([npixel, npixel]))

    np.testing.assert_array_equal(nside_hpgeom, nside)


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
@pytest.mark.parametrize("nside", [2**0, 2**5, 2**10, 2**20, 2**29])
def test_nside_to_pixel_area(nside):
    """Test nside_to_pixel_area."""
    pixel_area_hpgeom = hpgeom.nside_to_pixel_area(nside)
    pixel_area_healpy = hp.nside2pixarea(nside, degrees=True)

    assert (pixel_area_hpgeom == pixel_area_healpy)

    pixel_area_hpgeom = hpgeom.nside_to_pixel_area(nside, degrees=False)
    pixel_area_healpy = hp.nside2pixarea(nside, degrees=False)

    assert (pixel_area_hpgeom == pixel_area_healpy)

    pixel_area_hpgeom_arr = hpgeom.nside_to_pixel_area(np.array([nside, nside]))

    np.testing.assert_array_equal(pixel_area_hpgeom_arr, hpgeom.nside_to_pixel_area(nside))


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
@pytest.mark.parametrize("nside", [2**0, 2**5, 2**10, 2**20, 2**29])
def test_nside_to_resolution(nside):
    """Test nside_to_resolution."""
    resolution_hpgeom = hpgeom.nside_to_resolution(nside)
    resolution_healpy = np.rad2deg(hp.nside2resol(nside))

    assert (resolution_hpgeom == resolution_healpy)

    resolution_hpgeom = hpgeom.nside_to_resolution(nside, units='radians')
    resolution_healpy = hp.nside2resol(nside)

    assert (resolution_hpgeom == resolution_healpy)

    resolution_hpgeom = hpgeom.nside_to_resolution(nside, units='arcminutes')
    resolution_healpy = np.rad2deg(hp.nside2resol(nside))*60.

    assert (resolution_hpgeom == resolution_healpy)

    resolution_hpgeom = hpgeom.nside_to_resolution(nside, units='arcseconds')
    resolution_healpy = np.rad2deg(hp.nside2resol(nside))*60.*60.

    assert (resolution_hpgeom == resolution_healpy)

    resolution_hpgeom_arr = hpgeom.nside_to_resolution(np.array([nside, nside]))

    np.testing.assert_array_almost_equal(resolution_hpgeom_arr, hpgeom.nside_to_resolution(nside))


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
@pytest.mark.parametrize("nside", [2**0, 2**5, 2**10, 2**20, 2**29])
def test_nside_to_order(nside):
    """Test nside_to_order."""
    order_hpgeom = hpgeom.nside_to_order(nside)
    order_healpy = hp.nside2order(nside)

    assert (order_hpgeom == order_healpy)

    order_hpgeom_arr = hpgeom.nside_to_order(np.array([nside, nside]))

    np.testing.assert_array_equal(order_hpgeom_arr, order_hpgeom)


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
@pytest.mark.parametrize("order", [0, 5, 10, 20, 29])
def test_order_to_nside(order):
    """Test order_to_nside."""
    nside_hpgeom = hpgeom.order_to_nside(order)
    nside_healpy = hp.order2nside(order)

    assert (nside_hpgeom == nside_healpy)

    nside_hpgeom_arr = hpgeom.order_to_nside(np.array([order, order]))

    np.testing.assert_array_equal(nside_hpgeom_arr, nside_hpgeom)


def test_bad_nsides():
    """Test raising when bad nsides given."""

    with pytest.raises(ValueError, match=r"Illegal nside value"):
        hpgeom.nside_to_npixel(-1)

    with pytest.raises(ValueError, match=r"Illegal nside value"):
        hpgeom.nside_to_npixel(2**30)

    with pytest.raises(ValueError, match=r"Illegal nside value"):
        hpgeom.nside_to_npixel(np.array([1024, -1]))

    with pytest.raises(ValueError, match=r"Illegal npixel"):
        hpgeom.npixel_to_nside(-1)

    with pytest.raises(ValueError, match=r"Illegal npixel"):
        hpgeom.npixel_to_nside(10000)

    with pytest.raises(ValueError, match=r"Illegal npixel"):
        hpgeom.npixel_to_nside(np.array([-1, 12*1024*1024]))

    with pytest.raises(ValueError, match=r"Illegal nside value"):
        hpgeom.nside_to_pixel_area(2**30)

    with pytest.raises(ValueError, match=r"Illegal nside value"):
        hpgeom.nside_to_pixel_area(np.array([-1, 1024]))

    with pytest.raises(ValueError, match=r"Illegal nside value"):
        hpgeom.nside_to_resolution(2**30)

    with pytest.raises(ValueError, match=r"Illegal nside value"):
        hpgeom.nside_to_resolution(np.array([-1, 1024]))
