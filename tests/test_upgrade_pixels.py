import numpy as np
import pytest

try:
    import healpy as hp
    has_healpy = True
except ImportError:
    has_healpy = False

import hpgeom


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
@pytest.mark.parametrize("nsides", [(64, 64),
                                    (64, 256),
                                    (1024, 4096)])
def test_upgrade_pixel_ranges(nsides):
    nside, nside_upgrade = nsides

    pixel_ranges = np.zeros((4, 2), dtype=np.int64)
    pixel_ranges[0, :] = [100, 1000]
    pixel_ranges[1, :] = [10000, 11000]
    pixel_ranges[2, :] = [20000, 21000]
    pixel_ranges[3, :] = [35000, 36000]

    pixel_ranges_upgrade = hpgeom.upgrade_pixel_ranges(nside, pixel_ranges, nside_upgrade)

    assert pixel_ranges_upgrade.shape == pixel_ranges.shape

    pixels_upgrade = hpgeom.pixel_ranges_to_pixels(pixel_ranges_upgrade)

    m = np.zeros(hpgeom.nside_to_npixel(nside)) + hpgeom.UNSEEN
    m[hpgeom.pixel_ranges_to_pixels(pixel_ranges)] = 1.0

    m_upgrade = hp.ud_grade(m, nside_upgrade, order_in="NESTED", order_out="NESTED")
    m_upgrade_pixels, = np.where(m_upgrade > hpgeom.UNSEEN)

    np.testing.assert_array_equal(pixels_upgrade, m_upgrade_pixels)


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
@pytest.mark.parametrize("nsides", [(64, 64),
                                    (64, 256),
                                    (1024, 4096)])
@pytest.mark.parametrize("nest", [True, False])
def test_upgrade_pixels(nsides, nest):
    nside, nside_upgrade = nsides

    pixels1 = np.arange(1000, 5000)
    pixels2 = np.arange(15000, 2000)
    pixels = np.concatenate((pixels1, pixels2))

    pixels_upgrade = hpgeom.upgrade_pixels(nside, pixels, nside_upgrade, nest=nest)

    m = np.zeros(hpgeom.nside_to_npixel(nside)) + hpgeom.UNSEEN
    m[pixels] = 1.0

    if nest:
        order = "NESTED"
    else:
        order = "RING"

    m_upgrade = hp.ud_grade(m, nside_upgrade, order_in=order, order_out=order)
    m_upgrade_pixels, = np.where(m_upgrade > hpgeom.UNSEEN)

    if not nest:
        pixels_upgrade = np.sort(pixels_upgrade)
    np.testing.assert_array_equal(pixels_upgrade, m_upgrade_pixels)


def test_upgrade_pixel_ranges_bad_input():
    # Test wrong shape for ranges.
    with pytest.raises(ValueError, match="Input pixel_ranges must be of shape"):
        hpgeom.upgrade_pixel_ranges(32, np.array([1, 2, 3]), 64)

    # Test for non-integer nsides.
    with pytest.raises(ValueError, match="Input nside and nside_upgrade must be integers"):
        hpgeom.upgrade_pixel_ranges(32.0, np.zeros((5, 2), dtype=np.int64), 64)

    with pytest.raises(ValueError, match="Input nside and nside_upgrade must be integers"):
        hpgeom.upgrade_pixel_ranges(32, np.zeros((5, 2), dtype=np.int64), 64.0)

    # Test for nsides not power of two.
    with pytest.raises(ValueError, match="Input nside must be a power of two"):
        hpgeom.upgrade_pixel_ranges(31, np.zeros((5, 2), dtype=np.int64), 64)

    with pytest.raises(ValueError, match="Output nside_upgrade must be a power of two"):
        hpgeom.upgrade_pixel_ranges(32, np.zeros((5, 2), dtype=np.int64), 63)

    # Test for nside_upgrade less than nside.
    with pytest.raises(ValueError, match="The value for nside_upgrade must be >= nside"):
        hpgeom.upgrade_pixel_ranges(32, np.zeros((5, 2), dtype=np.int64), 16)

    # Test for correct pixel ranges.
    with pytest.raises(ValueError, match="Input pixels/ranges out of range"):
        hpgeom.upgrade_pixel_ranges(32, np.zeros((5, 2), dtype=np.int64) - 1, 64)

    with pytest.raises(ValueError, match="Input pixels/ranges out of range"):
        hpgeom.upgrade_pixel_ranges(32, np.zeros((5, 2), dtype=np.int64) + 1_000_000, 64)


def test_upgrade_pixels_bad_input():
    # Test for non-integer nsides.
    with pytest.raises(ValueError, match="Input nside and nside_upgrade must be integers"):
        hpgeom.upgrade_pixels(32.0, [0, 1], 64)

    with pytest.raises(ValueError, match="Input nside and nside_upgrade must be integers"):
        hpgeom.upgrade_pixels(32, [0, 1], 64.0)

    # Test for nsides not power of two.
    with pytest.raises(ValueError, match="Input nside must be a power of two"):
        hpgeom.upgrade_pixels(31, [0, 1], 64)

    with pytest.raises(ValueError, match="Output nside_upgrade must be a power of two"):
        hpgeom.upgrade_pixels(32, [0, 1], 63)

    # Test for nside_upgrade less than nside.
    with pytest.raises(ValueError, match="The value for nside_upgrade must be >= nside"):
        hpgeom.upgrade_pixels(32, [0, 1], 16)

    # Test for correct pixel ranges.
    with pytest.raises(ValueError, match="Input pixels/ranges out of range"):
        hpgeom.upgrade_pixels(32, [-1, 0], 64)

    with pytest.raises(ValueError, match="Input pixels/ranges out of range"):
        hpgeom.upgrade_pixels(32, [0, 1_000_000], 64)
