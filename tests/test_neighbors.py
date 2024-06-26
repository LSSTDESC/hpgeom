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
def test_neighbors(nside, scheme):
    """Test neighbors."""
    np.random.seed(12345)

    if scheme == "nest":
        nest = True
    else:
        nest = False

    pix = np.random.randint(low=0, high=12*nside*nside-1, size=1_000_000, dtype=np.int64)

    neighbors_hpgeom = hpgeom.neighbors(nside, pix, nest=nest)
    neighbors_healpy = hp.get_all_neighbours(nside, pix, nest=nest).T

    np.testing.assert_array_equal(neighbors_hpgeom, neighbors_healpy)


def test_neighbors_single_pixel():
    """Test neighbors, single pixel."""
    neighbors = hpgeom.neighbors(1024, 100)
    assert neighbors.shape == (8, )

    neighbors2 = hpgeom.neighbors(1024, [100, 200])
    np.testing.assert_array_equal(neighbors, neighbors2[0, :])


def test_neighbors_multiple_nside():
    """Test neighbors, multiple nside."""
    neighbors = hpgeom.neighbors([1024, 2048], [100, 200])
    assert neighbors.shape == (2, 8)

    neighbors2_1 = hpgeom.neighbors(1024, 100)
    np.testing.assert_array_equal(neighbors[0, :], neighbors2_1)
    neighbors2_2 = hpgeom.neighbors(2048, 200)
    np.testing.assert_array_equal(neighbors[1, :], neighbors2_2)

    neighbors3 = hpgeom.neighbors([1024, 2048], 100)
    assert neighbors3.shape == (2, 8)


def test_neighbors_zerolength():
    """Test neighbors, zero length."""
    neighbors = hpgeom.neighbors(1024, [])

    assert len(neighbors) == 0


def test_neighbors_bad_input():
    """Test neighbors, bad input."""
    with pytest.raises(ValueError, match=r"Pixel value .* out of range"):
        hpgeom.neighbors(1024, -1)

    with pytest.raises(ValueError, match=r"nside .* must be positive"):
        hpgeom.neighbors(-1, 100)
