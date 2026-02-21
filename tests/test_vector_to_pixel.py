import numpy as np
import pytest

try:
    import healpy as hp
    has_healpy = True
except ImportError:
    has_healpy = False

import hpgeom


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
@pytest.mark.parametrize("nside", [2**0, 2**5, 2**10, 2**15, 2**20, 2**25, 2**29])
@pytest.mark.parametrize("scheme", ["nest", "ring"])
def test_vector_to_pixel(nside, scheme):
    """Test vector_to_pixel."""
    np.random.seed(12345)

    if (scheme == "nest"):
        nest = True
    else:
        nest = False

    x = np.random.uniform(low=-1.0, high=1.0, size=1_000_000)
    y = np.random.uniform(low=-1.0, high=1.0, size=1_000_000)
    z = np.random.uniform(low=-1.0, high=1.0, size=1_000_000)

    pix_hpgeom = hpgeom.vector_to_pixel(nside, x, y, z, nest=nest)
    pix_healpy = hp.vec2pix(nside, x, y, z, nest=nest)

    np.testing.assert_array_equal(pix_hpgeom, pix_healpy)


@pytest.mark.parametrize("size", [1_000, 10_000_000])
@pytest.mark.parametrize("n_threads", [2])
def test_vector_to_pixel_threads(size, n_threads):
    """Test vector_to_pixel multi-threaded."""
    np.random.seed(12345)

    nside = 2**15

    x = np.random.uniform(low=-1.0, high=1.0, size=size)
    y = np.random.uniform(low=-1.0, high=1.0, size=size)
    z = np.random.uniform(low=-1.0, high=1.0, size=size)

    pix_single = hpgeom.vector_to_pixel(nside, x, y, z, n_threads=1)
    pix = hpgeom.vector_to_pixel(nside, x, y, z, n_threads=n_threads)

    np.testing.assert_array_equal(pix, pix_single)


@pytest.mark.parametrize("n_threads", [1, 2])
def test_vector_to_pixel_scalar(n_threads):
    """Test vector_to_pixel, scalar pixel."""
    x = 0.5
    y = 0.5
    z = 0.5

    pix_hpgeom = hpgeom.vector_to_pixel(1024, x, y, z, n_threads=n_threads)

    assert isinstance(pix_hpgeom, np.int64)
    assert not isinstance(pix_hpgeom, np.ndarray)


@pytest.mark.parametrize("n_threads", [1, 2])
def test_vector_to_pixel_zerolength(n_threads):
    """Test vector_to_pixel, zero length."""
    pix = hpgeom.vector_to_pixel(1024, [], [], [], n_threads=n_threads)

    assert len(pix) == 0
