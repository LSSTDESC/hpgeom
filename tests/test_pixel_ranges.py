import numpy as np
import pytest

import hpgeom as hpg


def _pixel_ranges_to_pixels_numpy(pixel_ranges, inclusive=0):
    pixels = np.zeros(np.sum(pixel_ranges[:, 1] - pixel_ranges[:, 0] + inclusive), dtype=np.int64)

    counter = 0
    for i in range(pixel_ranges.shape[0]):
        start = counter
        stop = counter + pixel_ranges[i, 1] - pixel_ranges[i, 0] + inclusive
        pixels[start: stop] = np.arange(pixel_ranges[i, 0], pixel_ranges[i, 1] + inclusive)
        counter += stop - start

    return pixels


def test_pixel_ranges_exclusive():
    """Test pixel_ranges_to_pixels, exclusive mode."""
    range1 = np.array(
        [
            [0, 10],
            [10, 20],
            [30, 40],
        ]
    )

    pixels = hpg.pixel_ranges_to_pixels(range1)
    pixels_test = _pixel_ranges_to_pixels_numpy(range1)

    np.testing.assert_array_equal(pixels, pixels_test)


def test_pixel_ranges_inclusive():
    """Test pixel_ranges_to_pixels, inclusive mode."""
    range1 = np.array(
        [
            [0, 10],
            [10, 20],
            [30, 40],
        ]
    )

    pixels = hpg.pixel_ranges_to_pixels(range1, inclusive=True)
    pixels_test = _pixel_ranges_to_pixels_numpy(range1, inclusive=1)

    np.testing.assert_array_equal(pixels, pixels_test)


def test_pixel_ranges_empty():
    """Test pixel ranges, empty."""
    range1 = np.zeros((0, 2), dtype=np.int64)

    pixels = hpg.pixel_ranges_to_pixels(range1)

    assert pixels.dtype == np.int64
    assert len(pixels) == 0


def test_pixel_ranges_scalar():
    """Test pixel ranges, scalar."""
    range1 = np.zeros((1, 2), dtype=np.int64)
    range1[0, 0] = 100
    range1[0, 1] = 101

    pixels = hpg.pixel_ranges_to_pixels(range1)

    assert pixels == 100
    assert len(pixels) == 1


def test_pixel_ranges_bad():
    """Test pixel_ranges_to_pixels, bad inputs."""
    with pytest.raises(ValueError, match=r"pixel_ranges must be 2D"):
        hpg.pixel_ranges_to_pixels([])

    with pytest.raises(ValueError, match=r"pixel_ranges must be 2D"):
        hpg.pixel_ranges_to_pixels(np.zeros(10, dtype=np.int64))

    with pytest.raises(ValueError, match=r"pixel_ranges must be 2D"):
        hpg.pixel_ranges_to_pixels(np.zeros((10, 3), dtype=np.int64))

    with pytest.raises(TypeError, match=r"Cannot cast array"):
        hpg.pixel_ranges_to_pixels(np.zeros((10, 2), dtype=np.float64))

    with pytest.raises(ValueError, match=r"must all be"):
        test = np.zeros((10, 2), dtype=np.int64)
        test[5, 1] = -1
        hpg.pixel_ranges_to_pixels(test)
