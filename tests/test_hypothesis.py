import numpy as np
import xarray as xr
import xproj  # noqa ignore
from hypothesis import given
from hypothesis import strategies as st

from xcorduroy.DEM import compute_terrain
from xcorduroy.types import Slope


def make_geo_da(
    data, x_name="x", y_name="y", x_coords=None, y_coords=None, epsg="epsg:4326"
):
    height, width = data.shape

    if x_coords is None:
        x_coords = np.arange(width, dtype=float)
    if y_coords is None:
        y_coords = np.arange(height, dtype=float)

    da = xr.DataArray(
        data,
        coords={y_name: y_coords, x_name: x_coords},
        dims=(y_name, x_name),
        name="elevation",
    )
    return da.proj.assign_crs(spatial_ref=epsg, allow_override=True)


maybe_nan_floats = st.one_of(
    st.floats(min_value=-100, max_value=100), st.just(float("nan"))
)


@st.composite
def sparse_nan_dem(draw, size=5, max_nans=3):
    """A 5x5 DEM with a few NaN cells.

    Sparse on purpose: the interesting case is an isolated NaN, whose
    neighbours are all valid. Drawing each cell independently as
    "float or NaN" buries that case under grids that are half NaN.
    """
    values = draw(
        st.lists(
            st.floats(min_value=-100, max_value=100),
            min_size=size * size,
            max_size=size * size,
        )
    )
    data = np.array(values).reshape((size, size))

    n_nans = draw(st.integers(min_value=0, max_value=max_nans))
    if n_nans:
        flat_idx = draw(
            st.lists(
                st.integers(min_value=0, max_value=size * size - 1),
                min_size=n_nans,
                max_size=n_nans,
                unique=True,
            )
        )
        data.flat[flat_idx] = np.nan
    return data


@given(
    res=st.floats(min_value=0.1, max_value=100.0),
    data=sparse_nan_dem(),
)
def test_terrain_nan_propagation(res, data):
    """NaN in, NaN out: a nodata cell and all of its neighbours are masked."""
    da = make_geo_da(data)

    result = compute_terrain(da, mode=Slope(), resolution=res, crs=da.proj.crs).values

    nan_in = np.isnan(data)

    # The cell itself. The Horn kernel weights the centre 0, so this only holds
    # because the kernel masks it explicitly.
    assert np.isnan(result[nan_in]).all()

    # Every cell with a NaN in its 3x3 stencil, edges included (the boundary is
    # padded by edge replication, which carries NaN inward).
    padded = np.pad(nan_in, 1, mode="edge")
    stencil_nan = np.zeros_like(nan_in)
    for dr in (0, 1, 2):
        for dc in (0, 1, 2):
            stencil_nan |= padded[dr : dr + 5, dc : dc + 5]
    assert np.isnan(result[stencil_nan]).all()

    # And the converse: a clean stencil must produce a finite value.
    assert np.isfinite(result[~stencil_nan]).all()


@given(
    elevations=st.lists(
        st.floats(min_value=-500, max_value=500), min_size=25, max_size=25
    )
)
def test_slope_rotation_invariance(elevations):
    data = np.array(elevations).reshape((5, 5))
    da = make_geo_da(data)

    slope_orig = da.dem.slope(resolution=10.0)

    da_rotated = make_geo_da(np.rot90(data, k=2))
    slope_rotated = da_rotated.dem.slope(resolution=10.0)

    np.testing.assert_allclose(
        slope_orig.values,
        np.rot90(slope_rotated.values, k=2),
        atol=1e-5,
        equal_nan=True,
    )


@given(
    elevations=st.lists(
        st.floats(min_value=-500, max_value=500), min_size=25, max_size=25
    )
)
def test_aspect_bounds(elevations):
    data = np.array(elevations).reshape((5, 5))
    da = make_geo_da(data)

    aspect = da.dem.aspect(resolution=10.0)

    if not np.all(np.isnan(aspect)):
        assert np.nanmin(aspect) >= 0
        assert np.nanmax(aspect) <= 360.0


@given(
    elevations=st.lists(
        st.floats(min_value=-100, max_value=100), min_size=25, max_size=25
    ),
    offset=st.floats(min_value=1000, max_value=5000),
)
def test_translation_invariance(elevations, offset):
    data = np.array(elevations).reshape((5, 5))
    da_low = make_geo_da(data)
    da_high = make_geo_da(data + offset)

    hs_low = da_low.dem.hillshade(resolution=10.0)
    hs_high = da_high.dem.hillshade(resolution=10.0)

    np.testing.assert_allclose(hs_low.values, hs_high.values, atol=1e-5, equal_nan=True)


@given(
    d=st.data(),
    width=st.integers(min_value=5, max_value=20),
    height=st.integers(min_value=5, max_value=20),
)
def test_variable_dimensions_and_discovery(d, width, height):
    elev_values = d.draw(
        st.lists(
            st.floats(min_value=0, max_value=1000),
            min_size=width * height,
            max_size=width * height,
        )
    )
    data = np.array(elev_values).reshape((height, width))

    da = make_geo_da(data, x_name="lon", y_name="lat")

    slope = da.dem.slope()

    assert slope.shape == (height, width)
    assert slope.dims == ("lat", "lon")


@given(
    elevations=st.lists(
        st.floats(min_value=-500, max_value=500), min_size=25, max_size=25
    )
)
def test_flat_surface_zero_slope(elevations):
    data = np.full((5, 5), elevations[0])
    da = make_geo_da(data)

    slope = da.dem.slope(resolution=10.0)
    assert np.allclose(slope.values, 0, atol=1e-8)


@given(
    base=st.floats(min_value=0, max_value=100),
    gradient=st.floats(min_value=0.1, max_value=10),
)
def test_linear_ramp_constant_slope(base, gradient):
    x = np.arange(5)
    y = np.arange(5)
    xx, yy = np.meshgrid(x, y)
    data = base + gradient * xx

    da = make_geo_da(data)
    slope = da.dem.slope(resolution=1.0)

    interior_slopes = slope.values[1:-1, 1:-1]
    if interior_slopes.size > 0:
        assert np.std(interior_slopes) < 0.5


@given(
    elevations=st.lists(
        st.floats(min_value=-500, max_value=500), min_size=25, max_size=25
    )
)
def test_slope_bounds(elevations):
    data = np.array(elevations).reshape((5, 5))
    da = make_geo_da(data)

    slope = da.dem.slope(resolution=10.0)

    valid_slopes = slope.values[np.isfinite(slope.values)]
    if len(valid_slopes) > 0:
        assert np.all(valid_slopes >= 0)
        assert np.all(valid_slopes <= 90)


@given(
    elevations=st.lists(
        st.floats(min_value=-500, max_value=500), min_size=25, max_size=25
    ),
    res1=st.floats(min_value=1, max_value=10),
    scale=st.floats(min_value=2, max_value=5),
)
def test_resolution_scaling(elevations, res1, scale):
    data = np.array(elevations).reshape((5, 5))
    da = make_geo_da(data)

    slope1 = da.dem.slope(resolution=res1)
    slope2 = da.dem.slope(resolution=res1 * scale)

    if np.isfinite(slope1).any() and np.isfinite(slope2).any():
        mean1 = np.nanmean(slope1.values)
        mean2 = np.nanmean(slope2.values)
        assert mean2 <= mean1 or np.isclose(mean1, 0, atol=0.1)


@given(
    elevations=st.lists(
        st.floats(min_value=-500, max_value=500), min_size=25, max_size=25
    )
)
def test_hillshade_bounds(elevations):
    data = np.array(elevations).reshape((5, 5))
    da = make_geo_da(data)

    hillshade = da.dem.hillshade(resolution=10.0)

    valid_values = hillshade.values[np.isfinite(hillshade.values)]
    if len(valid_values) > 0:
        assert np.all(valid_values >= 0)
        assert np.all(valid_values <= 1.0)


@given(
    elevations=st.lists(
        st.floats(min_value=-500, max_value=500), min_size=25, max_size=25
    ),
    azimuth=st.floats(min_value=0, max_value=360),
    altitude=st.floats(min_value=0, max_value=90),
)
def test_hillshade_with_parameters(elevations, azimuth, altitude):
    data = np.array(elevations).reshape((5, 5))
    da = make_geo_da(data)

    hillshade = da.dem.hillshade(resolution=10.0, azimuth=azimuth, altitude=altitude)

    assert hillshade.shape == (5, 5)
    assert np.all(np.isfinite(hillshade) | np.isnan(hillshade))


@st.composite
def geographic_coords(draw, min_lat=-85, max_lat=85, min_lon=-179, max_lon=179):
    """Ascending lon/lat arrays, kept clear of the poles and the dateline."""
    lat_start = draw(st.floats(min_value=min_lat, max_value=max_lat - 1.0))
    lat_end = draw(st.floats(min_value=lat_start + 0.5, max_value=max_lat))
    lon_start = draw(st.floats(min_value=min_lon, max_value=max_lon - 1.0))
    lon_end = draw(st.floats(min_value=lon_start + 0.5, max_value=max_lon))

    n_lats = draw(st.integers(min_value=5, max_value=12))
    n_lons = draw(st.integers(min_value=5, max_value=12))

    return (
        np.linspace(lon_start, lon_end, n_lons),
        np.linspace(lat_start, lat_end, n_lats),
    )


crs_strategy = st.sampled_from(
    [
        "epsg:4326",  # WGS84 geographic
        "epsg:3857",  # web mercator
        "epsg:32633",  # a UTM zone
    ]
)


@given(coords=geographic_coords(), epsg=crs_strategy, d=st.data())
def test_explicit_resolution_ignores_crs(coords, epsg, d):
    """resolution= bypasses degree-to-metre scaling, so the CRS cannot matter.

    Only the derived path consults ``crs.is_geographic``. Given an explicit
    resolution, a geographic and a projected DEM over the same grid must agree
    exactly.
    """
    lons, lats = coords
    elevations = d.draw(
        st.lists(
            st.floats(min_value=-500, max_value=8000),
            min_size=lons.size * lats.size,
            max_size=lons.size * lats.size,
        )
    )
    data = np.array(elevations).reshape((lats.size, lons.size))

    geo = make_geo_da(data, x_coords=lons, y_coords=lats, epsg="epsg:4326")
    other = make_geo_da(data, x_coords=lons, y_coords=lats, epsg=epsg)

    res = (30.0, 30.0)
    expected = compute_terrain(
        geo, mode=Slope(), resolution=res, crs=geo.proj.crs
    ).values
    got = compute_terrain(
        other, mode=Slope(), resolution=res, crs=other.proj.crs
    ).values

    np.testing.assert_array_equal(got, expected)


@given(coords=geographic_coords(), d=st.data())
def test_geographic_spacing_scales_by_cos_latitude(coords, d):
    """The derived geographic path is the metre path at the same cell size.

    Longitude degrees shrink by cos(mean latitude); latitude degrees do not.
    Deriving the resolution from geographic coordinates must match passing the
    equivalent metre resolution explicitly.
    """
    lons, lats = coords
    elevations = d.draw(
        st.lists(
            st.floats(min_value=-500, max_value=8000),
            min_size=lons.size * lats.size,
            max_size=lons.size * lats.size,
        )
    )
    data = np.array(elevations).reshape((lats.size, lons.size))
    geo = make_geo_da(data, x_coords=lons, y_coords=lats, epsg="epsg:4326")

    m_per_deg = 111320.0
    cos_lat = np.cos(np.deg2rad(float(np.mean(lats))))
    res_y = float(np.median(np.diff(lats))) * m_per_deg
    res_x = float(np.median(np.diff(lons))) * m_per_deg * cos_lat

    derived = geo.dem.slope().values
    explicit = compute_terrain(
        geo, mode=Slope(), resolution=(res_y, res_x), crs=geo.proj.crs
    ).values

    np.testing.assert_allclose(derived, explicit, rtol=1e-5, equal_nan=True)
