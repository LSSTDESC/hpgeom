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
def test_ring_to_nest_allpix(nside):
    """Test ring_to_nest for all pixels."""
    ring_pix = np.arange(12*nside*nside)

    nest_pix_hpgeom = hpgeom.ring_to_nest(nside, ring_pix)
    nest_pix_healpy = hp.ring2nest(nside, ring_pix)

    np.testing.assert_array_equal(nest_pix_hpgeom, nest_pix_healpy)


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
@pytest.mark.parametrize("nside", [2**15, 2**20, 2**25])
def test_ring_to_nest_samplepix(nside):
    """Test ring_to_nest for sampled pixels."""
    np.random.seed(12345)

    ring_pix = np.random.randint(low=0, high=12*nside*nside - 1, size=1_000_000)

    nest_pix_hpgeom = hpgeom.ring_to_nest(nside, ring_pix)
    nest_pix_healpy = hp.ring2nest(nside, ring_pix)

    np.testing.assert_array_equal(nest_pix_hpgeom, nest_pix_healpy)


@pytest.mark.parametrize("nside", [2**0, 2**5, 2**10, 2**15, 2**20, 2**25, 2**29])
def test_ring_to_nest_bad_pix(nside):
    """Test ring_to_nest errors when given bad pixel"""

    with pytest.raises(ValueError, match=r"Pixel value .* out of range"):
        hpgeom.ring_to_nest(nside, -1)

    with pytest.raises(ValueError, match=r"Pixel value .* out of range"):
        hpgeom.ring_to_nest(nside, 12*nside*nside)


def test_ring_to_nest_bad_nside():
    """test ring_to_nest errors when given a bad nside."""

    with pytest.raises(ValueError, match=r"nside .* must be positive"):
        hpgeom.ring_to_nest(-10, 100)

    with pytest.raises(ValueError, match=r"nside .* must be power of 2"):
        hpgeom.ring_to_nest(1020, 100)
