from typing import Optional, Tuple, Hashable, Iterable, Any
import xarray as xr
from .DEM import compute_terrain
from .types import Slope, Aspect, Hillshade


def _is_match(name: Hashable, options: Iterable[str]) -> bool:
    """Helper to safely check if a dimension/variable matches a string list."""
    return isinstance(name, str) and name.lower() in options


@xr.register_dataarray_accessor("dem")
class DEMDataArrayAccessor:
    def __init__(self, xarray_obj: xr.DataArray):
        self._obj = xarray_obj

    def _discover_dims(self, x: Optional[str], y: Optional[str]) -> Tuple[str, str]:
        x_options = ["x", "lon", "longitude", "long"]
        y_options = ["y", "lat", "latitude"]

        dims = self._obj.dims
        final_x = x or next((str(d) for d in dims if _is_match(d, x_options)), "x")
        final_y = y or next((str(d) for d in dims if _is_match(d, y_options)), "y")

        return final_x, final_y

    @property
    def crs(self) -> Any:
        try:
            crs = self._obj.proj.crs
        except (AttributeError, ValueError):
            crs = None

        if crs is None:
            raise ValueError(
                "Corduroy Error: No CRS found on DataArray. "
                "Terrain operations require a CRS for accurate scaling."
            )
        return crs

    def slope(self, x=None, y=None, resolution=None, **kwargs):
        x_dim, y_dim = self._discover_dims(x, y)
        return compute_terrain(
            self._obj,
            Slope(),
            resolution,
            crs=self.crs,
            x_dim=x_dim,
            y_dim=y_dim,
            **kwargs,
        )

    def aspect(self, x=None, y=None, resolution=None, **kwargs):
        x_dim, y_dim = self._discover_dims(x, y)
        return compute_terrain(
            self._obj,
            Aspect(),
            resolution,
            crs=self.crs,
            x_dim=x_dim,
            y_dim=y_dim,
            **kwargs,
        )

    def hillshade(
        self, x=None, y=None, resolution=None, azimuth=315.0, altitude=45.0, **kwargs
    ):
        x_dim, y_dim = self._discover_dims(x, y)
        mode = Hillshade(azimuth=azimuth, altitude=altitude)
        return compute_terrain(
            self._obj,
            mode,
            resolution,
            crs=self.crs,
            x_dim=x_dim,
            y_dim=y_dim,
            **kwargs,
        )


@xr.register_dataset_accessor("dem")
class DEMDatasetAccessor:
    def __init__(self, xarray_obj: xr.Dataset):
        self._obj = xarray_obj

    def __call__(self, name: Optional[str] = None) -> DEMDataArrayAccessor:
        if name:
            return self._obj[name].dem

        # 1. Search by common names
        common_names = ["elevation", "dem", "height", "z"]
        found_common = [v for v in self._obj.data_vars if _is_match(v, common_names)]

        if len(found_common) == 1:
            return self._obj[found_common[0]].dem
        elif len(found_common) > 1:
            raise ValueError(
                "multiple variables found. Specify variable: ds['elevation'].dem.slope()"
            )

        # 2. Fallback: Search by dimensionality
        spatial_vars = [v for v in self._obj.data_vars if self._obj[v].ndim >= 2]
        if len(spatial_vars) == 1:
            return self._obj[spatial_vars[0]].dem
        elif len(spatial_vars) > 1:
            raise ValueError(
                "multiple variables found. Specify variable: ds['elevation'].dem.slope()"
            )

        # 3. No match found
        raise AttributeError("Could not id an elevation var. Try name='variable_name'")

    def slope(self, **kwargs):
        return self.__call__().slope(**kwargs)

    def aspect(self, **kwargs):
        return self.__call__().aspect(**kwargs)

    def hillshade(self, **kwargs):
        return self.__call__().hillshade(**kwargs)
