import numpy as np
import pytest

try:
    import healpy as hp
    has_healpy = True
except ImportError:
    has_healpy = False

import hpgeom


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
@pytest.mark.parametrize("nside", [2**0, 2**5, 2**10, 2**20, 2**29])
@pytest.mark.parametrize("pixfrac", [0.0, 0.25, 0.5, 0.75])
@pytest.mark.parametrize("step", [1, 4, 8])
@pytest.mark.parametrize("scheme", ["nest", "ring"])
def test_boundaries_single(nside, pixfrac, step, scheme):
    """Test boundaries, single pixel."""
    pix = int(pixfrac*hpgeom.nside_to_npixel(nside))
    if (scheme == "nest"):
        nest = True
    else:
        nest = False

    theta_hpgeom, phi_hpgeom = hpgeom.boundaries(nside, pix, step=step, nest=nest, lonlat=False)
    theta_healpy, phi_healpy = hp.vec2ang(hp.boundaries(nside, pix, step=step, nest=nest).T)

    assert theta_hpgeom.shape == (step*4,)
    assert phi_hpgeom.shape == (step*4,)

    np.testing.assert_array_almost_equal(theta_hpgeom, theta_healpy)
    np.testing.assert_array_almost_equal(phi_hpgeom, phi_healpy)

    lon_hpgeom, lat_hpgeom = hpgeom.boundaries(nside, pix, step=step, nest=nest, lonlat=True, degrees=True)
    lon_healpy, lat_healpy = hp.vec2ang(hp.boundaries(nside, pix, step=step, nest=nest).T, lonlat=True)

    np.testing.assert_array_almost_equal(lon_hpgeom, lon_healpy)
    np.testing.assert_array_almost_equal(lat_hpgeom, lat_healpy)

    lonrad_hpgeom, latrad_hpgeom = hpgeom.boundaries(
        nside,
        pix,
        step=step,
        nest=nest,
        lonlat=True,
        degrees=False
    )
    np.testing.assert_array_almost_equal(lonrad_hpgeom, np.deg2rad(lon_healpy))
    np.testing.assert_array_almost_equal(latrad_hpgeom, np.deg2rad(lat_healpy))


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
@pytest.mark.parametrize("nside", [2**10, 2**20])
@pytest.mark.parametrize("pixfrac", [0.25, 0.5])
@pytest.mark.parametrize("step", [1, 4, 8])
@pytest.mark.parametrize("scheme", ["nest", "ring"])
@pytest.mark.parametrize("npix", [1, 5])
def test_boundaries_multiple(nside, pixfrac, step, scheme, npix):
    """Test boundaries, single pixel."""
    pixels = np.array([int(pixfrac*hpgeom.nside_to_npixel(nside))]*npix)
    if (scheme == "nest"):
        nest = True
    else:
        nest = False

    theta_hpgeom, phi_hpgeom = hpgeom.boundaries(nside, pixels, step=step, nest=nest, lonlat=False)
    theta_healpy, phi_healpy = hp.vec2ang(hp.boundaries(
        nside,
        pixels,
        step=step,
        nest=nest
    ).transpose([0, 2, 1]))
    theta_healpy = theta_healpy.reshape((npix, step*4))
    phi_healpy = phi_healpy.reshape((npix, step*4))

    assert theta_hpgeom.shape == (npix, step*4,)
    assert phi_hpgeom.shape == (npix, step*4,)

    np.testing.assert_array_almost_equal(theta_hpgeom, theta_healpy)
    np.testing.assert_array_almost_equal(phi_hpgeom, phi_healpy)

    lon_hpgeom, lat_hpgeom = hpgeom.boundaries(nside, pixels, step=step, nest=nest, lonlat=True, degrees=True)
    lon_healpy, lat_healpy = hp.vec2ang(hp.boundaries(
        nside,
        pixels,
        step=step,
        nest=nest,
    ).transpose([0, 2, 1]), lonlat=True)
    lon_healpy = lon_healpy.reshape((npix, step*4))
    lat_healpy = lat_healpy.reshape((npix, step*4))

    np.testing.assert_array_almost_equal(lon_hpgeom, lon_healpy)
    np.testing.assert_array_almost_equal(lat_hpgeom, lat_healpy)

    lonrad_hpgeom, latrad_hpgeom = hpgeom.boundaries(
        nside,
        pixels,
        step=step,
        nest=nest,
        lonlat=True,
        degrees=False
    )
    np.testing.assert_array_almost_equal(lonrad_hpgeom, np.deg2rad(lon_healpy))
    np.testing.assert_array_almost_equal(latrad_hpgeom, np.deg2rad(lat_healpy))


def test_boundaries_bad_inputs():
    """Test boundaries with bad inputs."""

    with pytest.raises(ValueError, match=r"nside .* must be positive"):
        hpgeom.boundaries(-1, 0)

    with pytest.raises(ValueError, match=r"Pixel value .* out of range"):
        hpgeom.boundaries(2**10, -1)

    with pytest.raises(ValueError, match=r"nside .* must be power of 2"):
        hpgeom.boundaries(2**10 - 2, -1, nest=True)
