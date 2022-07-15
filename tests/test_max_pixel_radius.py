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


def test_max_pixel_radius_badinputs():
    """Test max_pixel_radius with bad inputs."""
    with pytest.raises(ValueError, match=r"nside .* must be positive"):
        hpgeom.max_pixel_radius(-10)
