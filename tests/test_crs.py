import pytest
import xarray as xr
import numpy as np
import xproj  # noqa ignore
import xcorduroy  # noqa ignore


def test_missing_crs_raises_error():
    da = xr.DataArray(np.zeros((5, 5)), dims=("y", "x"))

    with pytest.raises(ValueError, match="No CRS found on DataArray"):
        _ = da.dem.slope()


def test_geographic_crs_z_factor(dem_factory):
    """Test that geographic CRS applies appropriate z-factor"""
    # Create DEMs at different latitudes
    da_equator = dem_factory(shape=(5, 5), epsg="epsg:4326")
    da_equator = da_equator.assign_coords(y=np.linspace(0, 1, 5))  # Near equator

    da_polar = dem_factory(shape=(5, 5), epsg="epsg:4326")
    da_polar = da_polar.assign_coords(y=np.linspace(80, 81, 5))  # Near pole

    slope_eq = da_equator.dem.slope()
    slope_polar = da_polar.dem.slope()

    assert np.nanmean(slope_polar) > np.nanmean(slope_eq)


def test_projected_crs_no_z_factor(dem_factory):
    """Test that projected CRS uses z_factor=1.0"""
    da = dem_factory(shape=(5, 5), epsg="epsg:32633")  # UTM zone 33N

    slope = da.dem.slope(resolution=10.0)

    assert np.all(np.isfinite(slope))


def test_explicit_z_factor_override(dem_factory):
    """Test that explicit z_factor overrides CRS-based calculation"""
    da = dem_factory(shape=(5, 5), epsg="epsg:4326")

    slope_default = da.dem.slope(resolution=10.0)
    slope_override = da.dem.slope(resolution=10.0, z_factor=1.0)

    # Should be different (unless we're at equator)
    assert not np.allclose(slope_default.values, slope_override.values)
