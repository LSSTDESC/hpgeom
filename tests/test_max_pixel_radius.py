import numpy as np
import pytest

try:
    import healpy as hp
    has_healpy = True
except ImportError:
    has_healpy = False

import hpgeom


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
@pytest.mark.parametrize("degrees", [False, True])
def test_max_pixel_radius(degrees):
    """Test max_pixel_radius."""
    nsides = 2**np.arange(29)
    nsides_offset = nsides + 2

    # Compute array-like
    radii_hpgeom = hpgeom.max_pixel_radius(nsides, degrees=degrees)
    radii_healpy = hp.max_pixrad(nsides, degrees=degrees)

    np.testing.assert_array_almost_equal(radii_hpgeom, radii_healpy)

    # Compute with offset, ring-like.
    # Note that we can't compare this to healpy because healpy crashes.
    radii_hpgeom = hpgeom.max_pixel_radius(nsides_offset, degrees=degrees)

    # Compute individual
    radius_hpgeom = hpgeom.max_pixel_radius(1024, degrees=degrees)
    radius_healpy = hp.max_pixrad(1024, degrees=degrees)

    np.testing.assert_almost_equal(radius_hpgeom, radius_healpy)


@pytest.mark.parametrize("size", [1_000, 10_000_001])
@pytest.mark.parametrize("n_threads", [2])
def test_max_pixel_radius_threads(size, n_threads):
    """Test max_pixel_radius with threads."""
    nsides = 2**np.random.choice(20, size=size)

    radii_single = hpgeom.max_pixel_radius(nsides, n_threads=1)
    radii = hpgeom.max_pixel_radius(nsides, n_threads=n_threads)

    np.testing.assert_array_equal(radii, radii_single)


@pytest.mark.parametrize("n_threads", [1, 2])
def test_max_pixel_radius_scalar(n_threads):
    """Test max_pixel_radius with a scalar."""
    nsides = 2**np.arange(29)

    radii_arr = hpgeom.max_pixel_radius(nsides, n_threads=n_threads)
    radius_scalar = hpgeom.max_pixel_radius(nsides[0], n_threads=n_threads)

    assert radius_scalar == radii_arr[0]
    assert not isinstance(radius_scalar, np.ndarray)


@pytest.mark.parametrize("n_threads", [1, 2])
def test_max_pixel_radius_zerolength(n_threads):
    """Test max_pixel_radius for zero-length nsides."""
    radii = hpgeom.max_pixel_radius([], n_threads=n_threads)

    assert len(radii) == 0


@pytest.mark.parametrize("n_threads", [1, 2])
def test_max_pixel_radius_badinputs(n_threads):
    """Test max_pixel_radius with bad inputs."""
    with pytest.raises(ValueError, match=r"nside .* must be positive"):
        hpgeom.max_pixel_radius(np.full(20_000, -10), n_threads=n_threads)
