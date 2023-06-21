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


def test_vector_to_pixel_single():
    """Test vector_to_pixel, single pixel."""
    x = 0.5
    y = 0.5
    z = 0.5

    pix_hpgeom = hpgeom.vector_to_pixel(1024, x, y, z)

    assert isinstance(pix_hpgeom, np.int64)
