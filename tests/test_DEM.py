import numpy as np
import pytest
import xarray as xr

import xcorduroy  # noqa ignore


def test_dimension_auto_discovery(dem_factory):
    """Test that xcorduroy finds 'longitude' and 'latitude' automagically"""
    da = dem_factory(x_name="longitude", y_name="latitude")

    slp = da.dem.slope()

    assert "longitude" in slp.coords
    assert "latitude" in slp.coords
    assert slp.max() > 0


def test_hillshade_execution(dem_factory):
    da = dem_factory(shape=(10, 10))
    result = da.dem.hillshade(azimuth=315, altitude=45)

    assert result.name == "hillshade"
    assert 0 <= result.min() <= result.max() <= 1
    assert result.dtype == np.float32


def test_slope_auto_resolution(dem_factory):
    da = dem_factory(shape=(10, 10))
    slp = da.dem.slope()
    assert slp.shape == da.shape
    assert slp.max() > 0


def test_dataset_accessor_all_methods(ds_factory):
    ds = ds_factory(var_names=["elevation"])
    assert ds.dem.slope().name == "slope"
    assert ds.dem.aspect().name == "aspect"
    assert ds.dem.hillshade().name == "hillshade"


def test_pyramid_math(pyramid_dem):
    slope = pyramid_dem.dem.slope(resolution=1.0)
    aspect = pyramid_dem.dem.aspect(resolution=1.0)

    assert slope.sel(x=3, y=2) > 0
    # y ascends in the fixture, so y=1 is south of the peak: downslope is south.
    np.testing.assert_allclose(aspect.sel(x=2, y=1), 180.0, atol=1e-3)
    np.testing.assert_allclose(aspect.sel(x=2, y=3), 0.0, atol=1e-3)
    np.testing.assert_allclose(aspect.sel(x=3, y=2), 90.0, atol=1e-3)
    np.testing.assert_allclose(aspect.sel(x=1, y=2), 270.0, atol=1e-3)
    assert np.isnan(aspect.sel(x=2, y=2))


def test_dask_chunk_seams(dem_factory):
    data = np.linspace(0, 10, 400).reshape(20, 20).astype(np.float32)

    da_solid = dem_factory(shape=(20, 20))
    da_solid.values = data

    da_chunked = dem_factory(shape=(20, 20), chunks={"y": 4, "x": 4})
    da_chunked.values = data

    slope_solid = da_solid.dem.slope(resolution=1.0)
    slope_chunked = da_chunked.dem.slope(resolution=1.0).compute()

    np.testing.assert_allclose(slope_solid.values, slope_chunked.values, atol=1e-6)


def test_dateline_jump(dem_factory):
    x_coords = np.array([178, 179, 180, -179, -178])
    da = dem_factory(shape=(5, 5)).assign_coords(x=x_coords)
    da.values[:, 2] = 10.0

    slope = da.dem.slope()
    assert not np.isnan(slope.sel(x=180, y=2))


def test_explicit_dimension_mapping(dem_factory):
    """Test that manually providing x and y works"""
    da = dem_factory(x_name="east", y_name="north")

    slp = da.dem.slope(x="east", y="north")

    assert "east" in slp.coords
    assert "north" in slp.coords
    assert slp.max() > 0


def test_negative_elevations(dem_factory):
    """Test that negative elevations will work"""
    da = dem_factory(shape=(5, 5))
    da.values = da.values - 100

    slope = da.dem.slope(resolution=1.0)

    assert slope.shape == (5, 5)
    assert np.all(np.isfinite(slope))


def _flanks(da, offset=8):
    """{name: (y, x)} points on the flanks of a hill DEM. North is larger y."""
    c = float(da.x.values[len(da.x) // 2])
    d = round(offset * 0.7)
    return {
        "N": (c + offset, c),
        "S": (c - offset, c),
        "E": (c, c + offset),
        "W": (c, c - offset),
        "NW": (c + d, c - d),
        "SE": (c - d, c + d),
    }


def _at(da, pt):
    return float(da.sel(y=pt[0], x=pt[1]))


def test_aspect_compass_convention(hill_dem):
    """Aspect is downslope direction, clockwise from north, for either y order."""
    aspect = hill_dem.dem.aspect()
    f = _flanks(hill_dem)
    expected = {"N": 0.0, "E": 90.0, "S": 180.0, "W": 270.0, "NW": 315.0, "SE": 135.0}
    for name, want in expected.items():
        got = _at(aspect, f[name])
        assert abs((got - want + 180) % 360 - 180) < 1.0, (name, got, want)


def test_hillshade_lit_from_azimuth(hill_dem):
    f = _flanks(hill_dem)
    hs_nw = hill_dem.dem.hillshade(azimuth=315, altitude=45)
    hs_se = hill_dem.dem.hillshade(azimuth=135, altitude=45)

    assert _at(hs_nw, f["NW"]) > 0.7
    assert _at(hs_nw, f["SE"]) < 0.1
    assert _at(hs_se, f["SE"]) > 0.7
    assert _at(hs_se, f["NW"]) < 0.1


def test_flat_surface(dem_factory):
    da = dem_factory(shape=(6, 6), epsg="epsg:32612")
    da.values[:] = 1200.0

    assert np.isnan(da.dem.aspect().values).all()
    np.testing.assert_allclose(da.dem.slope().values, 0.0)
    np.testing.assert_allclose(
        da.dem.hillshade(altitude=45).values, np.sin(np.deg2rad(45)), rtol=1e-6
    )


def test_geographic_scaling_is_anisotropic(make_hill):
    """In EPSG:4326 a degree of longitude is shorter than a degree of latitude."""
    n = 41
    z = make_hill(n, sigma=8.0, peak=2000.0)
    lat0 = 48.6
    step = 0.001
    lat = lat0 + step * np.arange(n)[::-1]
    lon = -113.7 + step * np.arange(n)
    geo = xr.DataArray(
        z, coords={"lat": lat, "lon": lon}, dims=("lat", "lon")
    ).proj.assign_crs(spatial_ref="epsg:4326", allow_override=True)

    m_per_deg = 111320.0
    proj = xr.DataArray(
        z,
        coords={
            "y": lat * m_per_deg,
            "x": lon * m_per_deg * np.cos(np.deg2rad(lat0)),
        },
        dims=("y", "x"),
    ).proj.assign_crs(spatial_ref="epsg:32612", allow_override=True)

    np.testing.assert_allclose(
        geo.dem.slope().values, proj.dem.slope().values, rtol=1e-3, atol=1e-3
    )
    np.testing.assert_allclose(
        geo.dem.aspect().values, proj.dem.aspect().values, rtol=1e-3, atol=1e-2
    )
    # E flank steeper than N flank by 1/cos(lat) because lon cells are shorter
    c = n // 2
    grad = np.tan(np.deg2rad(geo.dem.slope().values))
    ratio = grad[c, c + 8] / grad[c - 8, c]
    np.testing.assert_allclose(ratio, 1 / np.cos(np.deg2rad(lat0)), rtol=0.02)


def test_coarse_projected_resolution_not_wrapped(dem_factory):
    """Dateline wrap must not fire on projected coords with spacing > 180."""
    da = dem_factory(shape=(8, 8), epsg="epsg:32612")
    da = da.assign_coords(x=da.x.values * 250.0, y=da.y.values * 250.0)
    np.testing.assert_allclose(
        da.dem.slope().values, da.dem.slope(resolution=250.0).values
    )


def test_transposed_dims(hill_dem):
    expected = hill_dem.dem.aspect()
    got = hill_dem.transpose("x", "y").dem.aspect()
    np.testing.assert_allclose(got.values, expected.values, equal_nan=True)


def test_rejects_non_2d(hill_dem):
    with pytest.raises(ValueError, match="2D"):
        hill_dem.expand_dims("band").dem.slope()


@pytest.mark.parametrize("mode", ["slope", "aspect", "hillshade"])
@pytest.mark.parametrize("chunks", [None, {"y": 5, "x": 5}])
def test_nan_cell_masks_full_3x3(dem_factory, mode, chunks):
    """A nodata cell must be NaN itself, not just its neighbours."""
    da = dem_factory(shape=(9, 9), epsg="epsg:32612", chunks=chunks)
    da = da.copy(data=da.values)
    da.values[4, 4] = np.nan

    out = getattr(da.dem, mode)().compute().values
    assert np.isnan(out[3:6, 3:6]).all()
    assert not np.isnan(out[0:3, 0:3]).any()


def test_single_row_raises(dem_factory):
    da = dem_factory(shape=(1, 8), epsg="epsg:32612")
    with pytest.raises(ValueError, match="Cannot derive spacing along 'y'"):
        da.dem.slope()


def test_single_row_works_with_explicit_resolution(dem_factory):
    da = dem_factory(shape=(1, 8), epsg="epsg:32612")
    assert np.isfinite(da.dem.slope(resolution=1.0).values).all()


@pytest.mark.parametrize("resolution", [(1.0,), (1.0, 2.0, 3.0)])
def test_bad_resolution_tuple_raises(dem_factory, resolution):
    da = dem_factory(shape=(5, 5), epsg="epsg:32612")
    with pytest.raises(ValueError, match="scalar or a \\(y_res, x_res\\) pair"):
        da.dem.slope(resolution=resolution)
