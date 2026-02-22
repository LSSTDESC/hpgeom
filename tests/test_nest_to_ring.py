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
    nest_pix = np.arange(12*nside*nside)

    ring_pix_hpgeom = hpgeom.nest_to_ring(nside, nest_pix)
    ring_pix_healpy = hp.nest2ring(nside, nest_pix)

    np.testing.assert_array_equal(ring_pix_hpgeom, ring_pix_healpy)


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
@pytest.mark.parametrize("nside", [2**15, 2**20, 2**25])
def test_nest_to_ring_samplepix(nside):
    """Test nest_to_ring for sampled pixels."""
    np.random.seed(12345)

    nest_pix = np.random.randint(low=0, high=12*nside*nside - 1, size=1_000_000, dtype=np.int64)

    ring_pix_hpgeom = hpgeom.nest_to_ring(nside, nest_pix)
    ring_pix_healpy = hp.nest2ring(nside, nest_pix)

    np.testing.assert_array_equal(ring_pix_hpgeom, ring_pix_healpy)


@pytest.mark.parametrize("size", [1_000, 10_000_001])
@pytest.mark.parametrize("n_threads", [2])
def test_nest_to_ring_threads(size, n_threads):
    """Test nest_to_ring multi-threaded."""
    nside = 2**15

    nest_pix = np.random.randint(low=0, high=12*nside*nside - 1, size=size, dtype=np.int64)

    ring_pix_single = hpgeom.nest_to_ring(nside, nest_pix, n_threads=1)
    ring_pix = hpgeom.nest_to_ring(nside, nest_pix, n_threads=n_threads)

    np.testing.assert_array_equal(ring_pix, ring_pix_single)


@pytest.mark.parametrize("nside", [2**0, 2**5, 2**10, 2**15, 2**20, 2**25, 2**29])
@pytest.mark.parametrize("n_threads", [1, 2])
def test_nest_to_ring_scalar(nside, n_threads):
    """Test nest_to_ring for scalars."""
    np.random.seed(12345)

    nest_pix = np.random.randint(low=0, high=12*nside*nside-1, size=100, dtype=np.int64)
    ring_arr = hpgeom.nest_to_ring(nside, nest_pix, n_threads=n_threads)
    ring_scalar = hpgeom.nest_to_ring(nside, nest_pix[0], n_threads=n_threads)

    assert ring_scalar == ring_arr[0]
    assert not isinstance(ring_scalar, np.ndarray)

    ring_scalar2 = hpgeom.nest_to_ring(nside, int(nest_pix[0]), n_threads=n_threads)

    assert ring_scalar2 == ring_arr[0]
    assert not isinstance(ring_scalar2, np.ndarray)


@pytest.mark.parametrize("n_threads", [1, 2])
def test_nest_to_ring_zerolength(n_threads):
    """Test nest_to_ring zero-length."""
    pix = hpgeom.nest_to_ring(1024, [], n_threads=n_threads)

    assert len(pix) == 0


@pytest.mark.parametrize("nside", [2**0, 2**5, 2**10, 2**15, 2**20, 2**25, 2**29])
@pytest.mark.parametrize("n_threads", [1, 2])
def test_nest_to_ring_bad_pix(nside, n_threads):
    """Test nest_to_ring errors when given bad pixel"""

    with pytest.raises(ValueError, match=r"Pixel value .* out of range"):
        hpgeom.nest_to_ring(nside, np.full(100_000, -1), n_threads=n_threads)

    with pytest.raises(ValueError, match=r"Pixel value .* out of range"):
        hpgeom.nest_to_ring(nside, np.full(100_000, 12*nside*nside), n_threads=n_threads)


@pytest.mark.parametrize("n_threads", [1, 2])
def test_nest_to_ring_bad_nside(n_threads):
    """test nest_to_ring errors when given a bad nside."""

    with pytest.raises(ValueError, match=r"nside .* must be positive"):
        hpgeom.nest_to_ring(-10, np.full(100_000, 100), n_threads=n_threads)

    with pytest.raises(ValueError, match=r"nside .* must be power of 2"):
        hpgeom.nest_to_ring(1020, np.full(100_000, 100), n_threads=n_threads)
