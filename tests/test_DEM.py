import numpy as np
import corduroy  # noqa ignore


def test_dimension_auto_discovery(dem_factory):
    """Test that corduroy finds 'longitude' and 'latitude' automagically"""
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
    assert aspect.sel(x=2, y=1) == 180.0
    assert aspect.sel(x=2, y=3) == 0.0


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
