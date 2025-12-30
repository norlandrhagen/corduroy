from typing import Optional
import xarray as xr
from .DEM import compute_terrain
import xproj  # noqa ignore


@xr.register_dataarray_accessor("dem")
class DEMDataArrayAccessor:
    def __init__(self, xarray_obj: xr.DataArray):
        self._obj = xarray_obj

    def _discover_dims(self, x: Optional[str], y: Optional[str]):
        x_options = ["x", "lon", "longitude", "long"]
        y_options = ["y", "lat", "latitude"]

        dims = self._obj.dims
        final_x = x or next((d for d in dims if d.lower() in x_options), "x")
        final_y = y or next((d for d in dims if d.lower() in y_options), "y")

        return final_x, final_y

    @property
    def crs(self):
        try:
            crs = self._obj.proj.crs
        except (AttributeError, ValueError):
            crs = None

        if crs is None:
            raise ValueError(
                """Corduroy Error: No CRS found on DataArray.
        Terrain operations require a CRS for accurate scaling.

        You can use xproj to assign a CRS to your dataset:
        import xproj
        da = da.proj.assign_crs("EPSG:4326")"""
            )
        return crs

    def slope(self, x=None, y=None, resolution=None, **kwargs):
        x_dim, y_dim = self._discover_dims(x, y)
        return compute_terrain(
            self._obj,
            "slope",
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
            "aspect",
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
        return compute_terrain(
            self._obj,
            "hillshade",
            resolution,
            crs=self.crs,
            x_dim=x_dim,
            y_dim=y_dim,
            azimuth=azimuth,
            altitude=altitude,
            **kwargs,
        )


@xr.register_dataset_accessor("dem")
class DEMDatasetAccessor:
    def __init__(self, xarray_obj: xr.Dataset):
        self._obj = xarray_obj

    def __call__(self, name: Optional[str] = None) -> "DEMDataArrayAccessor":
        if name:
            return self._obj[name].dem

        common_names = ["elevation", "dem", "height", "z"]
        for var in self._obj.data_vars:
            if var.lower() in common_names:
                return self._obj[var].dem

        spatial_vars = [v for v in self._obj.data_vars if self._obj[v].ndim >= 2]
        if len(spatial_vars) == 1:
            return self._obj[spatial_vars[0]].dem

        raise ValueError(
            "Multiple variables found. Specify target: ds.dem('elevation').slope()"
        )

    def hillshade(self, x=None, y=None, **kwargs) -> xr.DataArray:
        return self.__call__().hillshade(x=x, y=y, **kwargs)

    def slope(self, x=None, y=None, **kwargs) -> xr.DataArray:
        return self.__call__().slope(x=x, y=y, **kwargs)

    def aspect(self, x=None, y=None, **kwargs) -> xr.DataArray:
        return self.__call__().aspect(x=x, y=y, **kwargs)
