import numpy as np

import hpgeom


def test_pixel_ranges_union_circles():
    """Test pixel_ranges_union with circles."""

    n_circle = 100

    np.random.seed(123456)

    lons = np.random.uniform(low=5.0, high=6.0, size=n_circle)
    lats = np.random.uniform(low=5.0, high=6.0, size=n_circle)
    radii = np.random.uniform(low=0.001, high=0.1, size=n_circle)

    # Do the slow way.
    pixels_list = []
    for i in range(n_circle):
        pixels = hpgeom.query_circle(2**17, lons[i], lats[i], radii[i])
        pixels_list.append(pixels)
    all_pixels = np.unique(np.concatenate(pixels_list))

    # Do the fast normalized pixel ranges way.
    pixel_ranges_list = []
    for i in range(n_circle):
        pixel_ranges = hpgeom.query_circle(2**17, lons[i], lats[i], radii[i], return_pixel_ranges=True)
        pixel_ranges_list.append(pixel_ranges)
    all_pixels2 = hpgeom.pixel_ranges_to_pixels(hpgeom.pixel_ranges_union(pixel_ranges_list))

    np.testing.assert_array_equal(all_pixels2, all_pixels)


def test_pixel_ranges_union_one_empty():
    """Test pixel_ranges_union with one empty range."""

    n_circle = 100

    np.random.seed(123456)

    lons = np.random.uniform(low=5.0, high=6.0, size=n_circle)
    lats = np.random.uniform(low=5.0, high=6.0, size=n_circle)
    radii = np.random.uniform(low=0.001, high=0.1, size=n_circle)
    radii[10] = 1e-128

    # Do the old (slow) way.
    pixels_list = []
    for i in range(n_circle):
        pixels = hpgeom.query_circle(2**17, lons[i], lats[i], radii[i])
        pixels_list.append(pixels)
    all_pixels = np.unique(np.concatenate(pixels_list))

    # And the new (fast) way.
    pixel_ranges_list = []
    for i in range(n_circle):
        pixel_ranges = hpgeom.query_circle(2**17, lons[i], lats[i], radii[i], return_pixel_ranges=True)
        pixel_ranges_list.append(pixel_ranges)

    # Ensure we have an empty pixel range in there.
    assert pixel_ranges_list[10].shape == (0, 2)

    all_pixels2 = hpgeom.pixel_ranges_to_pixels(hpgeom.pixel_ranges_union(pixel_ranges_list))

    np.testing.assert_array_equal(all_pixels2, all_pixels)


def test_pixel_ranges_union_empty():
    """Test normalize_pixel_ranges with all empty ranges."""
    ranges = hpgeom.pixel_ranges_union([np.zeros((0, 2))])

    assert ranges.shape == (0, 2)

    ranges = hpgeom.pixel_ranges_union([])

    assert ranges.shape == (0, 2)
