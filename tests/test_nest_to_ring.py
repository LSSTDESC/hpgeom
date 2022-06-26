import numpy as np
import pytest

try:
    import healpy as hp
    has_healpy = True
except ImportError:
    has_healpy = False

import hpgeom


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
@pytest.mark.parametrize("nside", [2**0, 2**5, 2**10])
def test_nest_to_ring_allpix(nside):
    """Test nest_to_ring for all pixels."""
    ring_pix = np.arange(12*nside*nside)

    nest_pix_hpgeom = hpgeom.nest_to_ring(nside, ring_pix)
    nest_pix_healpy = hp.nest2ring(nside, ring_pix)

    np.testing.assert_array_equal(nest_pix_hpgeom, nest_pix_healpy)


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
@pytest.mark.parametrize("nside", [2**15, 2**20, 2**25])
def test_nest_to_ring_samplepix(nside):
    """Test nest_to_ring for sampled pixels."""
    np.random.seed(12345)

    ring_pix = np.random.randint(low=0, high=12*nside*nside - 1, size=1_000_000, dtype=np.int64)

    nest_pix_hpgeom = hpgeom.nest_to_ring(nside, ring_pix)
    nest_pix_healpy = hp.nest2ring(nside, ring_pix)

    np.testing.assert_array_equal(nest_pix_hpgeom, nest_pix_healpy)


@pytest.mark.parametrize("nside", [2**0, 2**5, 2**10, 2**15, 2**20, 2**25, 2**29])
def test_nest_to_ring_bad_pix(nside):
    """Test nest_to_ring errors when given bad pixel"""

    with pytest.raises(ValueError, match=r"Pixel value .* out of range"):
        hpgeom.nest_to_ring(nside, -1)

    with pytest.raises(ValueError, match=r"Pixel value .* out of range"):
        hpgeom.nest_to_ring(nside, 12*nside*nside)


def test_nest_to_ring_bad_nside():
    """test nest_to_ring errors when given a bad nside."""

    with pytest.raises(ValueError, match=r"nside .* must be positive"):
        hpgeom.nest_to_ring(-10, 100)

    with pytest.raises(ValueError, match=r"nside .* must be power of 2"):
        hpgeom.nest_to_ring(1020, 100)
