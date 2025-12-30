import pytest
import xarray as xr
import corduroy  # noqa ignore


def test_dataset_auto_discovery(synthetic_dem):
    """Test that ds.dem.hillshade() finds the elev var automatically"""
    ds = synthetic_dem.to_dataset(name="elevation")
    result = ds.dem.hillshade(resolution=10.0)

    assert isinstance(result, xr.DataArray)
    assert result.name == "hillshade"


def test_dataset_explicit_call(synthetic_dem):
    """Test that ds.dem('name').hillshade() works"""
    ds = synthetic_dem.to_dataset(name="custom_name")
    result = ds.dem("custom_name").hillshade(resolution=10.0)

    assert result.shape == synthetic_dem.shape


def test_dataset_discovery_ambiguity(synthetic_dem):
    """Test that an error is raised when multiple variables exist and none are named dem"""
    ds = xr.Dataset({"var1": synthetic_dem, "var2": synthetic_dem})
    with pytest.raises(ValueError, match="multiple variables found."):
        ds.dem.hillshade()


def test_dataset_accessor_proxy(ds_factory):
    """Test that ds.dem() correctly finds variables names"""

    ds_std = ds_factory(var_names=["elevation"])
    assert ds_std.dem()._obj.name == "elevation"

    ds_alt = ds_factory(var_names=["z"])
    assert ds_alt.dem()._obj.name == "z"

    ds_multi = ds_factory(var_names=["low_res", "high_res"])
    assert ds_multi.dem("high_res")._obj.name == "high_res"

    with pytest.raises(ValueError, match="multiple variables found."):
        ds_multi.dem().slope()


def test_dataset_with_no_spatial_vars(dem_factory):
    """Test error handling when dataset has no spatial vars"""
    ds = xr.Dataset({"time": (["time"], [1, 2, 3]), "temp": (["time"], [20, 21, 22])})

    with pytest.raises(AttributeError, match="Could not id an elevation var"):
        ds.dem.slope()


def test_dataset_all_vars_same_priority(dem_factory):
    """Test when multiple vars exist but none match common names"""
    da1 = dem_factory(shape=(5, 5))
    da2 = dem_factory(shape=(5, 5))

    ds = xr.Dataset({"var_a": da1, "var_b": da2})

    with pytest.raises(ValueError, match="multiple variables found."):
        ds.dem.slope()
