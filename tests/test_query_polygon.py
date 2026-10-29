import numpy as np
import pytest

try:
    import healpy as hp
    has_healpy = True
except ImportError:
    has_healpy = False

try:
    from astropy.coordinates import SkyCoord
    has_astropy = True
except ImportError:
    has_astropy = False

import hpgeom


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
@pytest.mark.parametrize("nside_delta", [(2**5, 2.0),
                                         (2**10, 1.0),
                                         (2**18, 0.01)])
@pytest.mark.parametrize("shape", ["triangle", "square", "hexagon"])
@pytest.mark.parametrize("lon_ref", [0.0, 90.0, 180.0, 270.0])
@pytest.mark.parametrize("lat_ref", [-43.0, 0.0, 43.0, 85.0, -85.0])
def test_query_polygon_nest(nside_delta, shape, lon_ref, lat_ref):
    """Test query_polygon, nest ordering."""
    nside = nside_delta[0]
    delta = nside_delta[1]

    if shape == "triangle":
        lon = np.array([lon_ref, lon_ref + delta, lon_ref + delta])
        lat = np.array([lat_ref, lat_ref + delta, lat_ref - delta])
    elif shape == "square":
        lon = np.array([lon_ref, lon_ref + delta, lon_ref + delta, lon_ref])
        lat = np.array([lat_ref, lat_ref, lat_ref + delta, lat_ref + delta])
    else:
        lon = np.array([
            lon_ref,
            lon_ref + delta/2.,
            lon_ref + delta,
            lon_ref + 3*delta/2.,
            lon_ref + delta,
            lon_ref + delta/2.,
        ])
        lat = np.array([
            lat_ref,
            lat_ref + delta,
            lat_ref + delta,
            lat_ref,
            lat_ref - delta,
            lat_ref - delta
        ])

    # Test forward, non-inclusive
    pixels_hpgeom = hpgeom.query_polygon(nside, lon, lat)
    vec = hpgeom.angle_to_vector(lon, lat, lonlat=True)
    pixels_healpy = hp.query_polygon(nside, vec, nest=True)

    np.testing.assert_array_equal(pixels_hpgeom, pixels_healpy)

    # Test forward, non-inclusive, closed polygon
    lon2 = np.append(lon, lon[0])
    lat2 = np.append(lat, lat[0])

    pixels_hpgeom2 = hpgeom.query_polygon(nside, lon2, lat2)
    # Note healpy does not support closed polygons, so we use the first
    # run as a comparison.

    np.testing.assert_array_equal(pixels_hpgeom2, pixels_hpgeom)

    # Test reversed, non-inclusive
    pixels_hpgeom = hpgeom.query_polygon(nside, lon[::-1], lat[::-1])
    vec = hpgeom.angle_to_vector(lon[::-1], lat[::-1], lonlat=True)
    pixels_healpy = hp.query_polygon(nside, vec, nest=True)

    np.testing.assert_array_equal(pixels_hpgeom, pixels_healpy)

    # Test forward, inclusive
    pixels_hpgeom = hpgeom.query_polygon(nside, lon, lat, inclusive=True)
    vec = hpgeom.angle_to_vector(lon, lat, lonlat=True)
    pixels_healpy = hp.query_polygon(nside, vec, nest=True, inclusive=True)

    np.testing.assert_array_equal(pixels_hpgeom, pixels_healpy)


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
@pytest.mark.parametrize("nside_delta", [
    (2**5, 2.0),
    (2**10, 1.0),
    (2**18, 0.01),
    (2**10 - 2, 1.0)
])
@pytest.mark.parametrize("shape", ["triangle", "square", "hexagon"])
@pytest.mark.parametrize("lon_ref", [0.0, 90.0, 180.0])
@pytest.mark.parametrize("lat_ref", [-44.0, 0.0, 44.0, 85.0, -85.0])
def test_query_polygon_ring(nside_delta, shape, lon_ref, lat_ref):
    """Test query_polygon, ring ordering."""
    nside = nside_delta[0]
    delta = nside_delta[1]

    if shape == "triangle":
        lon = np.array([lon_ref, lon_ref + delta, lon_ref + delta])
        lat = np.array([lat_ref, lat_ref + delta, lat_ref - delta])
    elif shape == "square":
        lon = np.array([lon_ref, lon_ref + delta, lon_ref + delta, lon_ref])
        lat = np.array([lat_ref, lat_ref, lat_ref + delta, lat_ref + delta])
    else:
        lon = np.array([
            lon_ref,
            lon_ref + delta/2.,
            lon_ref + delta,
            lon_ref + 3*delta/2.,
            lon_ref + delta,
            lon_ref + delta/2.,
        ])
        lat = np.array([
            lat_ref,
            lat_ref + delta,
            lat_ref + delta,
            lat_ref,
            lat_ref - delta,
            lat_ref - delta
        ])

    # Test forward, non-inclusive
    pixels_hpgeom = hpgeom.query_polygon(nside, lon, lat, nest=False)
    vec = hpgeom.angle_to_vector(lon, lat, lonlat=True)
    pixels_healpy = hp.query_polygon(nside, vec, nest=False)

    np.testing.assert_array_equal(pixels_hpgeom, pixels_healpy)

    # Test reversed, non-inclusive
    pixels_hpgeom_rev = hpgeom.query_polygon(nside, lon[::-1], lat[::-1], nest=False)
    vec = hpgeom.angle_to_vector(lon[::-1], lat[::-1], lonlat=True)
    pixels_healpy = hp.query_polygon(nside, vec, nest=False)

    np.testing.assert_array_equal(pixels_hpgeom_rev, pixels_healpy)

    # Test forward, inclusive
    pixels_hpgeom = hpgeom.query_polygon(nside, lon, lat, inclusive=True, nest=False)
    vec = hpgeom.angle_to_vector(lon, lat, lonlat=True)
    pixels_healpy = hp.query_polygon(nside, vec, nest=False, inclusive=True)

    np.testing.assert_array_equal(pixels_hpgeom, pixels_healpy)


def test_query_polygon_radians():
    """Test query_polygon, use lonlat and radians."""
    nside = 1024
    delta = 1.0
    lon_ref = 180.0
    lat_ref = 0.0
    lon = np.array([lon_ref, lon_ref + delta, lon_ref + delta, lon_ref])
    lat = np.array([lat_ref, lat_ref, lat_ref + delta, lat_ref + delta])

    pixels_deg = hpgeom.query_polygon(nside, lon, lat)
    pixels_rad = hpgeom.query_polygon(nside, np.deg2rad(lon), np.deg2rad(lat), degrees=False)

    np.testing.assert_array_equal(pixels_deg, pixels_rad)


def test_query_polygon_thetaphi():
    """Test query_polygon, use theta_phi."""
    nside = 1024
    delta = 1.0
    lon_ref = 180.0
    lat_ref = 0.0
    lon = np.array([lon_ref, lon_ref + delta, lon_ref + delta, lon_ref])
    lat = np.array([lat_ref, lat_ref, lat_ref + delta, lat_ref + delta])

    pixels_lonlat = hpgeom.query_polygon(nside, lon, lat)
    theta, phi = hpgeom.lonlat_to_thetaphi(lon, lat)
    pixels_thetaphi = hpgeom.query_polygon(nside, theta, phi, lonlat=False)

    np.testing.assert_array_equal(pixels_lonlat, pixels_thetaphi)


@pytest.mark.skipif(not has_healpy, reason="Skipping test without healpy")
@pytest.mark.parametrize("fact", [1, 2, 4, 8])
def test_query_polygon_fact(fact):
    """Test query_polygon, use other fact values."""
    nside = 1024
    delta = 1.0
    lon_ref = 180.0
    lat_ref = 0.0
    lon = np.array([lon_ref, lon_ref + delta, lon_ref + delta, lon_ref])
    lat = np.array([lat_ref, lat_ref, lat_ref + delta, lat_ref + delta])

    pixels_hpgeom = hpgeom.query_polygon(nside, lon, lat, fact=fact)
    vec = hpgeom.angle_to_vector(lon, lat)
    pixels_healpy = hp.query_polygon(nside, vec, nest=True, fact=fact)

    np.testing.assert_array_equal(pixels_hpgeom, pixels_healpy)


def test_query_polygon_vec():
    """Test query_polygon_vec."""
    nside = 1024
    delta = 1.0
    lon_ref = 180.0
    lat_ref = 0.0
    lon = np.array([lon_ref, lon_ref + delta, lon_ref + delta, lon_ref])
    lat = np.array([lat_ref, lat_ref, lat_ref + delta, lat_ref + delta])

    pixels_lonlat = hpgeom.query_polygon(nside, lon, lat)
    vec = hpgeom.angle_to_vector(lon, lat)
    pixels_vec = hpgeom.query_polygon_vec(nside, vec)

    np.testing.assert_array_equal(pixels_lonlat, pixels_vec)


def test_query_polygon_return_pixel_ranges():
    """Test query_polygon with return_pixel_ranges."""
    nside = 1024
    delta = 1.0
    lon_ref = 180.0
    lat_ref = 0.0
    lon = np.array([lon_ref, lon_ref + delta, lon_ref + delta, lon_ref])
    lat = np.array([lat_ref, lat_ref, lat_ref + delta, lat_ref + delta])

    pixels = hpgeom.query_polygon(nside, lon, lat)

    pixel_ranges = hpgeom.query_polygon(nside, lon, lat, return_pixel_ranges=True)
    pixels_from_ranges = hpgeom.pixel_ranges_to_pixels(pixel_ranges)

    assert pixel_ranges.size < pixels.size

    np.testing.assert_array_equal(pixels, pixels_from_ranges)

    # And try a tiny polygon that has no pixels.
    delta = 0.001
    lon = np.array([lon_ref, lon_ref + delta, lon_ref + delta, lon_ref])
    lat = np.array([lat_ref, lat_ref, lat_ref + delta, lat_ref + delta])

    pixels = hpgeom.query_polygon(nside, lon, lat)
    pixel_ranges = hpgeom.query_polygon(nside, lon, lat, return_pixel_ranges=True)
    pixels_from_ranges = hpgeom.pixel_ranges_to_pixels(pixel_ranges)

    np.testing.assert_array_equal(pixels, pixels_from_ranges)
    assert len(pixel_ranges) == 0


def _inside_convex_polygon(lon_v, lat_v, lon, lat):
    """Test for points inside a convex polygon.

    Parameters
    ----------
    lon_v : `np.ndarray`
        Polygon longitudes, degrees.
    lat_v : `np.ndarray`
        Polygon latitudes, degrees.
    lon : `np.ndarray`
        Test point longitudes.
    lat : `np.ndarray`
        Test point latitudes.

    Returns
    -------
    in_polygon : `np.ndarray`
        Boolean array of which points are in the polygon.

    """
    V = hpgeom.angle_to_vector(lon_v, lat_v)
    N = np.cross(V, np.roll(V, -1, axis=0))
    if np.all(N @ V.sum(axis=0) < 0):
        N = -N
    X = hpgeom.angle_to_vector(lon, lat)
    return np.all(X @ N.T >= 0, axis=1)


frames = ["equatorial", "icrs", "c", "galactic", "g", "ecliptic", "barycentricmeanecliptic", "e"]


@pytest.mark.skipif(not has_astropy, reason="Skipping test without astropy")
@pytest.mark.parametrize("query_frame", frames)
@pytest.mark.parametrize("pixel_frame", frames)
@pytest.mark.parametrize("closed", [False, True])
def test_query_polygon_frames(query_frame, pixel_frame, closed):
    """Test query_polygon with frame conversions."""
    nside = 4096
    lon_qf = np.array([30.0, 30.0, 33.0, 36.0, 36.0])
    lat_qf = np.array([50.0, 53.0, 56.0, 53.0, 50.0])

    if closed:
        lon_qf_query = np.append(lon_qf, lon_qf[0])
        lat_qf_query = np.append(lat_qf, lat_qf[0])
    else:
        lon_qf_query = lon_qf
        lat_qf_query = lat_qf

    if query_frame in ["equatorial", "icrs", "c"]:
        astropy_qf = "icrs"
    elif query_frame in ["galactic", "g"]:
        astropy_qf = "galactic"
    else:
        astropy_qf = "barycentricmeanecliptic"

    if pixel_frame in ["equatorial", "icrs", "c"]:
        astropy_pf = "icrs"
    elif pixel_frame in ["galactic", "g"]:
        astropy_pf = "galactic"
    else:
        astropy_pf = "barycentricmeanecliptic"

    # Brute-force reference
    # We find the center of the polygon; rotate it; find max radius.
    cvec = hpgeom.angle_to_vector(lon_qf, lat_qf).sum(axis=0)
    c_lon_qf = np.degrees(np.arctan2(cvec[1], cvec[0])) % 360
    c_lat_qf = np.degrees(np.arcsin(cvec[2] / np.linalg.norm(cvec)))
    center = SkyCoord(c_lon_qf, c_lat_qf, frame=astropy_qf, unit="degree")
    rmax = center.separation(SkyCoord(lon_qf, lat_qf, frame=astropy_qf, unit="degree")).degree.max()

    # Convert the center to the pixel frame.
    if astropy_pf == "icrs":
        c_lon_pf, c_lat_pf = center.icrs.ra.degree, center.icrs.dec.degree
    elif astropy_pf == "galactic":
        c_lon_pf, c_lat_pf = center.galactic.l.degree, center.galactic.b.degree
    else:
        eclip = center.barycentricmeanecliptic
        c_lon_pf, c_lat_pf = eclip.lon.degree, eclip.lat.degree

    # Get the candidates within a circle and select down.
    candidates = hpgeom.query_circle(nside, c_lon_pf, c_lat_pf, rmax + 0.1)
    cand_coord = SkyCoord(*hpgeom.pixel_to_angle(nside, candidates), frame=astropy_pf, unit="degree")

    # Back to original frame.
    if astropy_qf == "icrs":
        cand_lon_qf, cand_lat_qf = cand_coord.icrs.ra.degree, cand_coord.icrs.dec.degree
    elif astropy_qf == "galactic":
        cand_lon_qf, cand_lat_qf = cand_coord.galactic.l.degree, cand_coord.galactic.b.degree
    else:
        eclip = cand_coord.barycentricmeanecliptic
        cand_lon_qf, cand_lat_qf = eclip.lon.degree, eclip.lat.degree

    pixels1 = np.sort(candidates[_inside_convex_polygon(lon_qf, lat_qf, cand_lon_qf, cand_lat_qf)])

    # Direct query
    pixels2 = hpgeom.query_polygon(
        nside,
        lon_qf_query,
        lat_qf_query,
        query_frame=query_frame,
        pixel_frame=pixel_frame,
    )

    np.testing.assert_array_equal(pixels2, pixels1)


def test_query_polygon_badinputs():
    """Test query_polygon with bad inputs."""
    nside = 1024
    delta = 1.0
    lon_ref = 180.0
    lat_ref = 0.0
    lon = np.array([lon_ref, lon_ref + delta, lon_ref + delta, lon_ref])
    lat = np.array([lat_ref, lat_ref, lat_ref + delta, lat_ref + delta])

    _lat = lat.copy()
    _lat[0] = 100.0
    with pytest.raises(ValueError, match=r"lat .* out of range"):
        hpgeom.query_polygon(nside, lon, _lat)

    _lat = lat.copy()
    _lat[0] = -100.0
    with pytest.raises(ValueError, match=r"lat .* out of range"):
        hpgeom.query_polygon(nside, lon, _lat)

    _theta, _phi = hpgeom.lonlat_to_thetaphi(lon, lat)
    _theta[0] = -0.1
    with pytest.raises(ValueError, match=r"colatitude \(theta\) .* out of range"):
        hpgeom.query_polygon(nside, _theta, _phi, lonlat=False)

    # Non-convex polygons are supported with nest ordering only.
    _lon = np.append(lon, lon_ref + delta/2.)
    _lat = np.append(lat, lat_ref + delta/2.)
    with pytest.warns(ResourceWarning, match="Non-convex polygon"):
        hpgeom.query_polygon(nside, _lon, _lat, nest=False)

    # Self-intersecting polygons are not supported.
    _lon = np.array([lon_ref, lon_ref + delta, lon_ref + delta, lon_ref])
    _lat = np.array([lat_ref, lat_ref + delta, lat_ref, lat_ref + delta])
    with pytest.raises(RuntimeError, match="Polygon is not simple"):
        hpgeom.query_polygon(nside, _lon, _lat)

    with pytest.raises(ValueError, match=r"Inclusive factor .* must be positive"):
        hpgeom.query_polygon(nside, lon, lat, inclusive=True, fact=0)

    with pytest.raises(ValueError, match=r"Inclusive factor \* nside must be \<\="):
        hpgeom.query_polygon(2**28, lon, lat, inclusive=True, fact=4)

    with pytest.raises(ValueError, match=r"Inclusive factor .* must be power of 2 for nest"):
        hpgeom.query_polygon(nside, lon, lat, inclusive=True, fact=3)

    # Different platforms have different strings here, but they all say ``integer``.
    with pytest.raises(TypeError, match=r"integer"):
        # Illegal fact (must be integer)
        hpgeom.query_polygon(nside, lon, lat, inclusive=True, nest=False, fact=3.5)

    with pytest.raises(RuntimeError, match=r"Can only use return_pixel_ranges with nest"):
        hpgeom.query_polygon(nside, lon, lat, nest=False, return_pixel_ranges=True)


def _tangent_basis(lon, lat):
    c = hpgeom.angle_to_vector(lon, lat)
    t = np.array([0.0, 0.0, 1.0]) if abs(c[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(t, c)
    e1 /= np.linalg.norm(e1)
    return c, e1, np.cross(c, e1)


def _star_polygon(lon_0, lat_0, nvert, rmin, rmax, rng):
    """Random star-shaped (simple, generally non-convex) polygon about (lon_0, lat_0)."""
    c, e1, e2 = _tangent_basis(lon_0, lat_0)
    psi = np.sort(rng.uniform(0, 2*np.pi, nvert))
    r = np.deg2rad(rng.uniform(rmin, rmax, nvert))
    v = (np.cos(r)[:, None]*c
         + np.sin(r)[:, None]*(np.cos(psi)[:, None]*e1 + np.sin(psi)[:, None]*e2))
    return np.rad2deg(np.arctan2(v[:, 1], v[:, 0])) % 360.0, np.rad2deg(np.arcsin(v[:, 2]))


def _inside_polygon_gnomonic(lon_v, lat_v, lon, lat, lon_0, lat_0):
    """Even-odd test in the gnomonic projection about (lon_0, lat_0).

    Great circles project to straight lines, so this is exact for polygons
    that lie within the hemisphere centered on (lon_0, lat_0).
    """
    c, e1, e2 = _tangent_basis(lon_0, lat_0)
    v = hpgeom.angle_to_vector(lon_v, lat_v)
    assert np.all(v @ c > 0)
    xv, yv = (v @ e1)/(v @ c), (v @ e2)/(v @ c)
    p = hpgeom.angle_to_vector(lon, lat)
    w = p @ c
    front = w > 0
    w = np.where(front, w, 1.0)
    x, y = (p @ e1)/w, (p @ e2)/w
    inside = np.zeros(len(x), dtype=bool)
    for xi, yi, xj, yj in zip(xv, yv, np.roll(xv, 1), np.roll(yv, 1)):
        crosses = (yi > y) != (yj > y)
        dy = np.where(yj == yi, 1.0, yj - yi)
        inside ^= crosses & (x < (xj - xi)*(y - yi)/dy + xi)
    return inside & front


def _dist_to_boundary(lon_v, lat_v, lon, lat):
    """Minimum angular distance (radians) from points to the polygon edges."""
    v = hpgeom.angle_to_vector(lon_v, lat_v)
    p = hpgeom.angle_to_vector(lon, lat)
    d = np.full(len(p), np.pi)
    for a, b in zip(v, np.roll(v, -1, axis=0)):
        n = np.cross(a, b)
        n /= np.linalg.norm(n)
        inspan = (p @ np.cross(n, a) >= 0) & (p @ np.cross(b, n) >= 0)
        dperp = np.arcsin(np.clip(np.abs(p @ n), 0, 1))
        dend = np.arccos(np.clip(np.maximum(p @ a, p @ b), -1, 1))
        d = np.minimum(d, np.where(inspan, dperp, dend))
    return d


def _check_nonconvex(nside, lon_v, lat_v, lon_0, lat_0, rmax):
    """Compare against brute force; differences may only be exact boundary ties."""
    pixels = hpgeom.query_polygon(nside, lon_v, lat_v)

    radius = rmax + 2*hpgeom.max_pixel_radius(nside)
    candidates = hpgeom.query_circle(nside, lon_0, lat_0, radius)
    cand_lon, cand_lat = hpgeom.pixel_to_angle(nside, candidates)
    inside = _inside_polygon_gnomonic(lon_v, lat_v, cand_lon, cand_lat, lon_0, lat_0)
    expected = np.sort(candidates[inside])

    diff = np.setxor1d(pixels, expected)
    if len(diff) > 0:
        diff_lon, diff_lat = hpgeom.pixel_to_angle(nside, diff)
        np.testing.assert_array_less(_dist_to_boundary(lon_v, lat_v, diff_lon, diff_lat), 1e-12)

    return pixels, expected


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "lon_0, lat_0, nvert, rmin, rmax, nside",
    [
        (10.0, 20.0, 8, 1.0, 3.0, 256),
        (0.5, -30.0, 40, 2.0, 6.0, 512),
        (359.0, 5.0, 200, 1.0, 4.0, 1024),
        (100.0, 88.0, 30, 3.0, 6.0, 256),
        (200.0, -89.0, 25, 2.0, 5.0, 256),
        (45.0, 0.0, 600, 0.5, 2.0, 2048),
        (300.0, 60.0, 12, 10.0, 40.0, 64),
        (123.0, -45.0, 7, 0.001, 0.003, 2**17),
    ],
)
def test_query_polygon_nonconvex(lon_0, lat_0, nvert, rmin, rmax, nside, reverse):
    """Test non-convex polygons against a brute-force reference."""
    rng = np.random.default_rng(12345)
    lon_v, lat_v = _star_polygon(lon_0, lat_0, nvert, rmin, rmax, rng)
    if reverse:
        lon_v, lat_v = lon_v[::-1], lat_v[::-1]

    pixels, expected = _check_nonconvex(nside, lon_v, lat_v, lon_0, lat_0, rmax)
    assert len(pixels) > 0

    # Pixel ranges agree with pixels.
    ranges = hpgeom.query_polygon(nside, lon_v, lat_v, return_pixel_ranges=True)
    np.testing.assert_array_equal(hpgeom.pixel_ranges_to_pixels(ranges), pixels)

    # Inclusive: superset, contains the vertices, and nothing far from the polygon.
    pixels_inc = hpgeom.query_polygon(nside, lon_v, lat_v, inclusive=True, fact=8)
    assert np.all(np.isin(pixels, pixels_inc))
    assert np.all(np.isin(hpgeom.angle_to_pixel(nside, lon_v, lat_v), pixels_inc))
    inc_lon, inc_lat = hpgeom.pixel_to_angle(nside, pixels_inc)
    near = (_inside_polygon_gnomonic(lon_v, lat_v, inc_lon, inc_lat, lon_0, lat_0)
            | (_dist_to_boundary(lon_v, lat_v, inc_lon, inc_lat)
               <= hpgeom.max_pixel_radius(nside, degrees=False)*(1 + 1e-7)))
    assert np.all(near)


def _u_shaped_footprint():
    """A large U-shaped footprint with edges on the lon=0 meridian and collinear vertices."""
    def seg(lon0, lat0, lon1, lat1, step):
        n = max(2, int(np.ceil(max(abs(lon1 - lon0), abs(lat1 - lat0))/step)) + 1)
        return np.linspace(lon0, lon1, n)[:-1], np.linspace(lat0, lat1, n)[:-1]

    pieces = [seg(-60, -65, 100, -65, 0.5), seg(100, -65, 100, 5, 10), seg(100, 5, 40, 5, 0.5),
              seg(40, 5, 40, -40, 9), seg(40, -40, 0, -40, 0.5), seg(0, -40, 0, 5, 9),
              seg(0, 5, -60, 5, 0.5), seg(-60, 5, -60, -65, 10)]
    lon = np.concatenate([p[0] for p in pieces]) % 360.0
    lat = np.concatenate([p[1] for p in pieces])
    return lon, lat


def test_query_polygon_nonconvex_footprint():
    """Test a large survey-like footprint, including meridian-aligned edges."""
    lon_v, lat_v = _u_shaped_footprint()

    _check_nonconvex(256, lon_v, lat_v, 20.0, -30.0, 85.0)

    # Area from Gauss-Bonnet versus the pixel count at high resolution.
    v = hpgeom.angle_to_vector(lon_v, lat_v)
    turn = 0.0
    for a, b, c in zip(np.roll(v, 1, axis=0), v, np.roll(v, -1, axis=0)):
        t_in = np.cross(np.cross(a, b), b)
        t_out = np.cross(np.cross(b, c), b)
        turn += np.arctan2(np.cross(t_in, t_out) @ b, t_in @ t_out)
    area = 2*np.pi - abs(turn)

    nside = 2**14
    ranges = hpgeom.query_polygon(nside, lon_v, lat_v, return_pixel_ranges=True)
    npix = np.sum(ranges[:, 1] - ranges[:, 0])
    np.testing.assert_allclose(npix*hpgeom.nside_to_pixel_area(nside, degrees=False), area, rtol=1e-4)


def test_query_polygon_nonconvex_collinear():
    """A polygon with collinear vertices (formerly a degenerate corner) works with nest."""
    lon = np.array([0.0, 5.0, 10.0, 10.0, 0.0])
    lat = np.array([0.0, 0.0, 0.0, 10.0, 10.0])
    pixels = hpgeom.query_polygon(256, lon, lat)
    pixels_ref = hpgeom.query_polygon(256, lon[[0, 2, 3, 4]], lat[[0, 2, 3, 4]])
    np.testing.assert_array_equal(pixels, pixels_ref)

    with pytest.warns(ResourceWarning, match="Non-convex polygon"):
        pixels_ring = hpgeom.query_polygon(256, lon, lat, nest=False)

    np.testing.assert_array_equal(np.sort(hpgeom.ring_to_nest(256, pixels_ring)), pixels)


def test_query_polygon_nonconvex_frames():
    """Rotated non-convex polygons match polygons with pre-rotated vertices."""
    lon_v, lat_v = _star_polygon(50.0, 30.0, 50, 2.0, 6.0, np.random.default_rng(1))
    pixels = hpgeom.query_polygon(512, lon_v, lat_v, query_frame="galactic")
    assert len(pixels) > 0

    pixels_same = hpgeom.query_polygon(512, lon_v, lat_v, query_frame="g", pixel_frame="g")
    np.testing.assert_array_equal(pixels_same, hpgeom.query_polygon(512, lon_v, lat_v))
