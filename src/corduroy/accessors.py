from typing import Optional, Tuple, Hashable, Iterable, Any
import xarray as xr
from .DEM import compute_terrain
from .types import Slope, Aspect, Hillshade


def _is_match(name: Hashable, options: Iterable[str]) -> bool:
    """Check if variable name matches"""
    return isinstance(name, str) and name.lower() in options


@xr.register_dataarray_accessor("dem")
class DEMDataArrayAccessor:
    """
    Xarray accessor for terrain analysis on DataArrays.

    Provides methods to compute slope, aspect, and hillshade from elevation data.
    Access via da.dem.slope(), da.dem.aspect(), or da.dem.hillshade().
    """

    def __init__(self, xarray_obj: xr.DataArray):
        self._obj = xarray_obj

    def _discover_dims(self, x: Optional[str], y: Optional[str]) -> Tuple[str, str]:
        """
        Auto-detect x and y dimension names from common conventions.

        Args:
            x: Explicit x dimension name or None to auto-detect
            y: Explicit y dimension name or None to auto-detect

        Returns:
            Tuple of (x_dim, y_dim) names
        """
        x_options = ["x", "lon", "longitude", "long"]
        y_options = ["y", "lat", "latitude"]

        dims = self._obj.dims
        final_x = x or next((str(d) for d in dims if _is_match(d, x_options)), "x")
        final_y = y or next((str(d) for d in dims if _is_match(d, y_options)), "y")

        return final_x, final_y

    @property
    def crs(self) -> Any:
        """
        Get coordinate reference system from DataArray.

        Returns:
            CRS object from xproj accessor

        Raises:
            ValueError: If no CRS is found
        """
        try:
            crs = self._obj.proj.crs
        except (AttributeError, ValueError):
            crs = None

        if crs is None:
            raise ValueError(
                "No CRS found on DataArray. You can assign a crs with: import xproj; ds.proj.assign_crs(spatial_ref='EPSG:4326', allow_override=True)"
            )
        return crs

    def slope(self, x=None, y=None, resolution=None, **kwargs):
        """
        Compute slope in degrees.

        Args:
            x: X dimension name (auto-detected if None)
            y: Y dimension name (auto-detected if None)
            resolution: Cell size or (y_res, x_res) tuple
            **kwargs: Additional arguments (e.g., z_factor)

        Returns:
            DataArray with slope values in degrees
        """
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
        """
        Compute aspect in degrees (0-360).

        Args:
            x: X dimension name (auto-detected if None)
            y: Y dimension name (auto-detected if None)
            resolution: Cell size or (y_res, x_res) tuple
            **kwargs: Additional arguments (e.g., z_factor)

        Returns:
            DataArray with aspect values in degrees
        """
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
        """
        Compute hillshade (0-1 shaded relief).

        Args:
            x: X dimension name (auto-detected if None)
            y: Y dimension name (auto-detected if None)
            resolution: Cell size or (y_res, x_res) tuple
            azimuth: Light source direction in degrees (0-360)
            altitude: Light source angle above horizon in degrees (0-90)
            **kwargs: Additional arguments (e.g., z_factor)

        Returns:
            DataArray with hillshade values (0-1)
        """
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
    """
    Xarray accessor for terrain analysis on Datasets.

    Auto-detects elevation variable or access specific variables via ds.dem('var_name').
    Provides methods to compute slope, aspect, and hillshade.
    """

    def __init__(self, xarray_obj: xr.Dataset):
        self._obj = xarray_obj

    def __call__(self, name: Optional[str] = None) -> DEMDataArrayAccessor:
        """
        Get DEM accessor for a specific variable or auto-detect elevation variable.

        Args:
            name: Variable name or None to auto-detect

        Returns:
            DEMDataArrayAccessor for the selected variable

        Raises:
            ValueError: If multiple candidate variables found
            AttributeError: If no suitable elevation variable found
        """
        if name:
            return self._obj[name].dem

        common_names = ["elevation", "dem", "height", "z", "band_data"]
        found_common = [v for v in self._obj.data_vars if _is_match(v, common_names)]

        if len(found_common) == 1:
            return self._obj[found_common[0]].dem
        elif len(found_common) > 1:
            raise ValueError(
                "multiple variables found. Specify variable: ds['elevation'].dem.slope()"
            )

        spatial_vars = [v for v in self._obj.data_vars if self._obj[v].ndim >= 2]
        if len(spatial_vars) == 1:
            return self._obj[spatial_vars[0]].dem
        elif len(spatial_vars) > 1:
            raise ValueError(
                "multiple variables found. Specify variable: ds['elevation'].dem.slope()"
            )

        raise AttributeError("Could not id an elevation var. Try name='variable_name'")

    def slope(self, **kwargs):
        """Compute slope. See DEMDataArrayAccessor.slope for details."""
        return self.__call__().slope(**kwargs)

    def aspect(self, **kwargs):
        """Compute aspect. See DEMDataArrayAccessor.aspect for details."""
        return self.__call__().aspect(**kwargs)

    def hillshade(self, **kwargs):
        """Compute hillshade. See DEMDataArrayAccessor.hillshade for details."""
        return self.__call__().hillshade(**kwargs)
