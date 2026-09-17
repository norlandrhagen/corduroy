import numpy as np
import pytest
import xarray as xr
import xproj  # noqa ignore


def _apply_spatial_metadata(obj, epsg, x_name, y_name):
    if x_name != "x" or y_name != "y":
        obj = obj.rename({"x": x_name, "y": y_name})
    return obj.proj.assign_crs(spatial_ref=epsg, allow_override=True)


@pytest.fixture
def dem_factory():
    def _make_dem(
        shape=(10, 10), chunks=None, epsg="epsg:4326", x_name="x", y_name="y"
    ):
        y_grad, x_grad = np.meshgrid(
            np.arange(shape[0]), np.arange(shape[1]), indexing="ij"
        )
        data = (x_grad + y_grad).astype(np.float32)

        da = xr.DataArray(
            data,
            coords={"y": np.arange(shape[0]), "x": np.arange(shape[1])},
            dims=("y", "x"),
            name="elevation",
        )

        da = _apply_spatial_metadata(da, epsg, x_name, y_name)

        return da.chunk(chunks) if chunks else da

    return _make_dem


@pytest.fixture
def ds_factory(dem_factory):
    def _make_ds(
        var_names=["elevation"],
        shape=(10, 10),
        chunks=None,
        epsg="epsg:4326",
        x_name="x",
        y_name="y",
    ):
        ds = xr.Dataset(
            {
                name: dem_factory(
                    shape=shape, chunks=chunks, epsg=epsg, x_name=x_name, y_name=y_name
                )
                for name in var_names
            }
        )
        return _apply_spatial_metadata(ds, epsg, x_name, y_name)

    return _make_ds


@pytest.fixture
def synthetic_dem(dem_factory):
    """A high-point center synthetic dataset"""
    da = dem_factory(shape=(5, 5))
    da.values[2, 2] = 30.0
    return da


@pytest.fixture
def pyramid_dem(dem_factory):
    """a 5x5 pyramid for slope testing."""
    data = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 2, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    da = dem_factory(shape=(5, 5))
    da.values = data
    return da


def _make_hill(n=41, sigma=8.0, peak=100.0):
    """Gaussian hill, row 0 = first y coordinate."""
    c = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    return peak * np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * sigma**2))


@pytest.fixture
def make_hill():
    return _make_hill


@pytest.fixture(params=["descending", "ascending"])
def hill_dem(request):
    """North-up (descending y) or south-up (ascending y) projected hill DEM."""
    n = 41
    z = _make_hill(n)
    y = np.arange(n, dtype=float)
    if request.param == "descending":
        y = y[::-1]
    da = xr.DataArray(
        z, coords={"y": y, "x": np.arange(n, dtype=float)}, dims=("y", "x")
    )
    return da.proj.assign_crs(spatial_ref="epsg:32612", allow_override=True)
