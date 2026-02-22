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
def test_pixel_to_vector(nside, scheme):
    """Test pixel_to_vector."""
    np.random.seed(12345)

    if (scheme == "nest"):
        nest = True
    else:
        nest = False

    pix = np.random.randint(low=0, high=hpgeom.nside_to_npixel(nside) - 1, size=1_000_000, dtype=np.int64)

    x_hpgeom, y_hpgeom, z_hpgeom = hpgeom.pixel_to_vector(nside, pix, nest=nest)
    x_healpy, y_healpy, z_healpy = hp.pix2vec(nside, pix, nest=nest)

    np.testing.assert_array_equal(x_hpgeom, x_healpy)
    np.testing.assert_array_equal(y_hpgeom, y_healpy)
    np.testing.assert_array_equal(z_hpgeom, z_healpy)


@pytest.mark.parametrize("size", [1_001, 10_000_001])
@pytest.mark.parametrize("n_threads", [2])
def test_pixel_to_vector_threads(size, n_threads):
    """Test pixel_to_vector multi-threaded."""
    np.random.seed(12345)

    nside = 2**15

    pix = np.random.randint(low=0, high=hpgeom.nside_to_npixel(nside) - 1, size=size, dtype=np.int64)

    x_single, y_single, z_single = hpgeom.pixel_to_vector(nside, pix, n_threads=1)
    x, y, z = hpgeom.pixel_to_vector(nside, pix, n_threads=n_threads)

    np.testing.assert_array_equal(x, x_single)
    np.testing.assert_array_equal(y, y_single)
    np.testing.assert_array_equal(z, z_single)


@pytest.mark.parametrize("nside", [2**0, 2**5, 2**10, 2**15, 2**20, 2**25, 2**29])
@pytest.mark.parametrize("n_threads", [1, 2])
def test_pixel_to_vector_scalar(nside, n_threads):
    np.random.seed(12345)

    pix = np.random.randint(low=0, high=hpgeom.nside_to_npixel(nside) - 1, size=100, dtype=np.int64)

    x_arr, y_arr, z_arr = hpgeom.pixel_to_vector(nside, pix, n_threads=n_threads)
    x_scalar, y_scalar, z_scalar = hpgeom.pixel_to_vector(nside, pix[0], n_threads=n_threads)

    assert x_scalar == x_arr[0]
    assert y_scalar == y_arr[0]
    assert z_scalar == z_arr[0]
    assert not isinstance(x_scalar, np.ndarray)
    assert not isinstance(y_scalar, np.ndarray)
    assert not isinstance(z_scalar, np.ndarray)


@pytest.mark.parametrize("n_threads", [1, 2])
def test_pixel_to_vector_zerolength(n_threads):
    x, y, z = hpgeom.pixel_to_vector(1024, [], n_threads=n_threads)

    assert len(x) == 0
    assert len(y) == 0
    assert len(z) == 0


@pytest.mark.parametrize("nside", [2**0, 2**5, 2**10, 2**15, 2**20, 2**25, 2**29])
@pytest.mark.parametrize("n_threads", [1, 2])
def test_pixel_to_vector_bad_pix(nside, n_threads):
    """Test pixel_to_angle errors when given bad pixel"""

    with pytest.raises(ValueError, match=r"Pixel value .* out of range"):
        hpgeom.pixel_to_vector(nside, np.full(20_000, -1), n_threads=n_threads)

    with pytest.raises(ValueError, match=r"Pixel value .* out of range"):
        hpgeom.pixel_to_vector(nside, np.full(20_000, 12*nside*nside), n_threads=n_threads)


@pytest.mark.parametrize("n_threads", [1, 2])
def test_pixel_to_vector_bad_nside(n_threads):
    """Test pixel_to_angle errors when given a bad nside."""
    np.random.seed(12345)

    pix = np.random.randint(low=0, high=12*2048*2048-1, size=20_000, dtype=np.int64)

    with pytest.raises(ValueError, match=r"nside .* must be positive"):
        hpgeom.pixel_to_vector(-10, pix, nest=False, n_threads=n_threads)

    with pytest.raises(ValueError, match=r"nside .* must be power of 2"):
        hpgeom.pixel_to_vector(2040, pix, nest=True, n_threads=n_threads)

    with pytest.raises(ValueError, match=r"nside .* must not be greater than"):
        hpgeom.pixel_to_vector(2**30, pix, nest=True, n_threads=n_threads)
