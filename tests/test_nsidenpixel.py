import numpy as np
import pytest

try:
    import healpy as hp
    has_healpy = True
except ImportError:
    has_healpy = False

import hpgeom


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
@pytest.mark.parametrize("nside", [2**0, 2**5, 2**10, 2**15, 2**20, 2**25, 1000])
def test_nside_to_npixel(nside):
    """Test nside_to_npixel."""
    npixel_hpgeom = hpgeom.nside_to_npixel(nside)
    npixel_healpy = hp.nside2npix(nside)

    assert(npixel_hpgeom == npixel_healpy)

    npixel_hpgeom2 = hpgeom.nside_to_npixel(np.zeros(10, dtype=np.int64) + nside)

    np.testing.assert_array_equal(npixel_hpgeom2, npixel_healpy)


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
@pytest.mark.parametrize("nside", [2**0, 2**5, 2**10, 2**15, 2**20, 2**25])
def test_npixel_to_nside(nside):
    """Test npixel_to_nside."""
    npixel = 12*nside*nside

    nside_hpgeom = hpgeom.npixel_to_nside(npixel)
    nside_healpy = hp.npix2nside(npixel)

    assert(nside_hpgeom == nside_healpy)

    nside_hpgeom2 = hpgeom.npixel_to_nside(np.zeros(10, dtype=np.int64) + npixel)

    np.testing.assert_array_equal(nside_hpgeom2, nside_healpy)


def test_npixel_to_nside_bad_npixel():
    """Test npixel_to_nside with bad npixel values."""

    with pytest.raises(ValueError):
        hpgeom.npixel_to_nside(100)

    npixels = np.zeros(100, dtype=np.int64)
    npixels[:] = 12*2048*2048
    npixels[50] = 100

    with pytest.raises(ValueError):
        hpgeom.npixel_to_nside(npixels)
