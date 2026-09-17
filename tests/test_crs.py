import numpy as np
import pytest
import xarray as xr
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


def test_explicit_z_factor_is_vertical_exaggeration(dem_factory):
    """z_factor scales elevation; explicit resolution skips degree-to-metre scaling"""
    da = dem_factory(shape=(5, 5), epsg="epsg:4326")

    slope_default = da.dem.slope(resolution=10.0)
    slope_one = da.dem.slope(resolution=10.0, z_factor=1.0)
    slope_two = da.dem.slope(resolution=10.0, z_factor=2.0)

    np.testing.assert_allclose(slope_default.values, slope_one.values)
    assert np.all(slope_two.values >= slope_one.values)
    assert not np.allclose(slope_two.values, slope_one.values)


def test_polar_longitude_scaling_is_clamped(dem_factory):
    """Near a pole, cos(lat) collapses res_x; the clamp keeps slope sane and warns."""
    da = dem_factory(shape=(5, 5), epsg="epsg:4326")
    da = da.assign_coords(y=np.linspace(89.99, 90.0, 5), x=np.linspace(0.0, 0.01, 5))

    with pytest.warns(RuntimeWarning, match="within 0.1 degrees of the pole"):
        slope_polar = da.dem.slope()

    # Same 0.01 degree span, mean latitude exactly at the 89.9 degree clamp.
    da_clamp = da.assign_coords(y=np.linspace(89.895, 89.905, 5))
    slope_clamp = da_clamp.dem.slope()

    # The clamped polar array matches the 89.9 degree case rather than saturating.
    np.testing.assert_allclose(
        slope_polar.values, slope_clamp.values, rtol=1e-3, atol=1e-3
    )
    assert float(np.nanmax(slope_polar)) < 89.0
