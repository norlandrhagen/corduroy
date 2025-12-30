from hypothesis import given, strategies as st
import numpy as np
import xarray as xr
from corduroy.DEM import compute_terrain
import xproj  # noqa ignore


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


@given(
    res=st.floats(min_value=0.1, max_value=100.0),
    elevations=st.lists(maybe_nan_floats, min_size=25, max_size=25),
)
def test_terrain_nan_propagation(res, elevations):
    """Test that NaN work"""
    data = np.array(elevations).reshape((5, 5))
    da = make_geo_da(data)

    result = compute_terrain(da, mode="slope", resolution=res, crs=da.proj.crs)

    assert np.all(np.isfinite(result) | np.isnan(result))

    if np.isnan(data).all():
        assert np.isnan(result).all()

    if not np.isnan(data).any():
        assert np.isfinite(result).any() or np.allclose(data, data.flat[0])


@given(
    elevations=st.lists(
        st.floats(min_value=-500, max_value=500), min_size=25, max_size=25
    )
)
def test_slope_rotation_invariance(elevations):
    """Slope magnitude should be invariant to 180° rotation"""
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
    """A perfect linear ramp should have constant slope in interior"""
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
    """slope should be between 0 and 90 deg"""
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
    """Slopes should decreaes if we increase resolution"""
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
    """Hillshade values should be bounded from 0-255 or 0-1"""
    data = np.array(elevations).reshape((5, 5))
    da = make_geo_da(data)

    hillshade = da.dem.hillshade(resolution=10.0)

    valid_values = hillshade.values[np.isfinite(hillshade.values)]
    if len(valid_values) > 0:
        assert np.all(valid_values >= 0)
        assert np.all(valid_values <= 255)


@given(
    elevations=st.lists(
        st.floats(min_value=-500, max_value=500), min_size=25, max_size=25
    ),
    azimuth=st.floats(min_value=0, max_value=360),
    altitude=st.floats(min_value=0, max_value=90),
)
def test_hillshade_with_parameters(elevations, azimuth, altitude):
    """Can we adjust hillshade params"""
    data = np.array(elevations).reshape((5, 5))
    da = make_geo_da(data)

    hillshade = da.dem.hillshade(resolution=10.0, azimuth=azimuth, altitude=altitude)

    assert hillshade.shape == (5, 5)
    assert np.all(np.isfinite(hillshade) | np.isnan(hillshade))
