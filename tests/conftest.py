import pytest
import numpy as np
import xarray as xr
import xproj  # noqa ignore
import hypothesis.strategies as st


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

        da = da.rename({"x": x_name, "y": y_name})
        da = da.proj.assign_crs(spatial_ref=epsg, allow_override=True)

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


@st.composite
def geographic_coords(draw, min_lat=-89, max_lat=89, min_lon=-179, max_lon=179):
    """Generate geographic coord arrays"""
    lat_start = draw(st.floats(min_value=min_lat, max_value=max_lat - 0.01))
    lat_end = draw(st.floats(min_value=lat_start + 0.01, max_value=max_lat))

    lon_start = draw(st.floats(min_value=min_lon, max_value=max_lon - 0.01))
    lon_end = draw(st.floats(min_value=lon_start + 0.01, max_value=max_lon))

    n_lats = draw(st.integers(min_value=5, max_value=20))
    n_lons = draw(st.integers(min_value=5, max_value=20))

    lats = np.linspace(lat_start, lat_end, n_lats)
    lons = np.linspace(lon_start, lon_end, n_lons)

    return lons, lats


@st.composite
def dem_with_nans(draw, min_size=5, max_size=20):
    """Generate DEM with nan patterns"""
    width = draw(st.integers(min_size, max_size))
    height = draw(st.integers(min_size, max_size))

    elevations = draw(
        st.lists(
            st.floats(-500, 8000), min_size=width * height, max_size=width * height
        )
    )

    nan_fraction = draw(st.floats(0, 0.3))
    n_nans = int(width * height * nan_fraction)

    if n_nans > 0:
        nan_indices = draw(
            st.lists(
                st.integers(0, width * height - 1),
                min_size=n_nans,
                max_size=n_nans,
                unique=True,
            )
        )

        for idx in nan_indices:
            elevations[idx] = float("nan")

    return np.array(elevations).reshape(height, width), height, width


@st.composite
def dem_data(draw, min_size=5, max_size=20, allow_nans=True):
    """gen dem data"""
    width = draw(st.integers(min_size, max_size))
    height = draw(st.integers(min_size, max_size))

    if allow_nans:
        value_strategy = st.one_of(st.floats(-500, 8000), st.just(float("nan")))
    else:
        value_strategy = st.floats(-500, 8000)

    elevations = draw(
        st.lists(value_strategy, min_size=width * height, max_size=width * height)
    )

    return np.array(elevations).reshape(height, width), height, width


@st.composite
def valid_bbox(draw):
    """Generate valid bounding boxes that don't cross dateline/poles... but should we check that?"""
    min_lon = draw(st.floats(min_value=-180, max_value=179))
    max_lon = draw(st.floats(min_value=min_lon + 0.1, max_value=180))
    min_lat = draw(st.floats(min_value=-90, max_value=89))
    max_lat = draw(st.floats(min_value=min_lat + 0.1, max_value=90))
    return (min_lon, min_lat, max_lon, max_lat)


@st.composite
def crs_strategy(draw):
    """crs opts"""
    return draw(
        st.sampled_from(
            [
                "epsg:4326",  # WEG84
                "epsg:3857",  # web mercator
                "epsg:32633",  # random UTM zone
                "epsg:32610",
            ]
        )
    )
