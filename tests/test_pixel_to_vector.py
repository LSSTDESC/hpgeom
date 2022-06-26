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


@pytest.mark.parametrize("nside", [2**0, 2**5, 2**10, 2**15, 2**20, 2**25, 2**29])
def test_pixel_to_vector_bad_pix(nside):
    """Test pixel_to_angle errors when given bad pixel"""

    with pytest.raises(ValueError, match=r"Pixel value .* out of range"):
        hpgeom.pixel_to_vector(nside, -1)

    with pytest.raises(ValueError, match=r"Pixel value .* out of range"):
        hpgeom.pixel_to_vector(nside, 12*nside*nside)


def test_pixel_to_vector_bad_nside():
    """Test pixel_to_angle errors when given a bad nside."""
    np.random.seed(12345)

    pix = np.random.randint(low=0, high=12*2048*2048-1, size=100, dtype=np.int64)

    with pytest.raises(ValueError, match=r"nside .* must be positive"):
        hpgeom.pixel_to_vector(-10, pix, nest=False)

    with pytest.raises(ValueError, match=r"nside .* must be power of 2"):
        hpgeom.pixel_to_vector(2040, pix, nest=True)

    with pytest.raises(ValueError, match=r"nside .* must not be greater than"):
        hpgeom.pixel_to_vector(2**30, pix, nest=True)
