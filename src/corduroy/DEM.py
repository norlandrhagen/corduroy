from typing import Optional, Any
import numpy as np
import xarray as xr
from .types import ModeType, Hillshade


def _terrain_kernel(
    data: np.ndarray,
    res_x: float,
    res_y: float,
    mode: str,
    z_factor: float = 1.0,
    azimuth: float = 315.0,
    altitude: float = 45.0,
) -> np.ndarray:
    """
    Compute terrain analysis using a 3x3 kernel.

    Calculates slope, aspect, or hillshade from elevation data using
    finite differences to estimate gradients.

    Args:
        data: 2D elevation array with 1-pixel padding
        res_x: Cell size in x direction
        res_y: Cell size in y direction
        mode: Terrain mode ('slope', 'aspect', or 'hillshade')
        z_factor: Vertical exaggeration factor
        azimuth: Light source azimuth in degrees (0-360)
        altitude: Light source altitude in degrees (0-90)

    Returns:
        Computed terrain array (slope in degrees, aspect in degrees, or hillshade 0-1)
    """
    z = data * z_factor

    res_x = res_x if res_x != 0 else 1e-9
    res_y = res_y if res_y != 0 else 1e-9

    dz_dx = (
        (z[0:-2, 2:] + 2 * z[1:-1, 2:] + z[2:, 2:])
        - (z[0:-2, 0:-2] + 2 * z[1:-1, 0:-2] + z[2:, 0:-2])
    ) / (8.0 * res_x)

    dz_dy = (
        (z[0:-2, 0:-2] + 2 * z[0:-2, 1:-1] + z[0:-2, 2:])
        - (z[2:, 0:-2] + 2 * z[2:, 1:-1] + z[2:, 2:])
    ) / (8.0 * res_y)

    if mode == "slope":
        magnitude = np.hypot(dz_dx, dz_dy)
        return np.rad2deg(np.arctan(magnitude)).astype(np.float32)

    if mode == "aspect":
        aspect = np.rad2deg(np.arctan2(dz_dy, -dz_dx))
        return np.mod(450 - aspect, 360).astype(np.float32)

    az_rad = np.deg2rad(360.0 - azimuth)
    alt_rad = np.deg2rad(altitude)
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    aspect_rad = np.arctan2(-dz_dx, dz_dy)

    shaded = np.sin(alt_rad) * np.cos(slope_rad) + np.cos(alt_rad) * np.sin(
        slope_rad
    ) * np.cos(az_rad - aspect_rad)

    return np.clip(shaded, 0, 1).astype(np.float32)


def compute_terrain(
    da: xr.DataArray,
    mode: ModeType,
    resolution: Optional[float | int | tuple] = None,
    crs: Any = None,
    x_dim: str = "x",
    y_dim: str = "y",
    **kwargs,
) -> xr.DataArray:
    """
    Compute terrain analysis from elevation DataArray.

    Args:
        da: Input elevation DataArray
        mode: Terrain mode (Slope, Aspect, or Hillshade instance)
        resolution: Cell size as scalar or (y_res, x_res). Auto-detected if None
        crs: Coordinate reference system for geographic z-factor adjustment
        x_dim: Name of x dimension
        y_dim: Name of y dimension
        **kwargs: Additional arguments passed to terrain kernel (e.g., z_factor)

    Returns:
        DataArray with computed terrain values and updated metadata
    """
    x_coords = da[x_dim]
    y_coords = da[y_dim]

    if resolution is None:
        if hasattr(da, "rio") and da.rio.resolution() is not None:
            res_y, res_x = da.rio.resolution()
        else:
            dx = np.diff(x_coords)
            dx = np.where(np.abs(dx) > 180, 360 - np.abs(dx), dx)
            res_x = float(np.abs(np.median(dx)))
            res_y = float(np.abs(np.median(np.diff(y_coords))))
    else:
        if isinstance(resolution, (int, float)):
            res_y = res_x = float(resolution)
        else:
            res_y, res_x = float(resolution[0]), float(resolution[1])

    z_factor = kwargs.pop("z_factor", None)
    if z_factor is None:
        if crs and crs.is_geographic:
            mean_lat = float(y_coords.mean())
            z_factor = 1.0 / (111320.0 * np.cos(np.deg2rad(mean_lat)))
        else:
            z_factor = 1.0

    kernel_kwargs = {
        **kwargs,
        "res_x": abs(res_x),
        "res_y": abs(res_y),
        "mode": mode.name,
        "z_factor": z_factor,
    }

    if isinstance(mode, Hillshade):
        kernel_kwargs.update({"azimuth": mode.azimuth, "altitude": mode.altitude})

    if da.chunks is not None:
        out_data = da.data.map_overlap(
            _terrain_kernel,
            depth=1,
            boundary="nearest",
            trim=False,
            meta=np.array((), dtype=np.float32),
            **kernel_kwargs,
        )
    else:
        padded = np.pad(da.data, pad_width=1, mode="edge")
        out_data = _terrain_kernel(padded, **kernel_kwargs)  # type: ignore[invalid-argument-type]

    return xr.DataArray(
        out_data,
        coords=da.coords,
        dims=da.dims,
        name=mode.name,
        attrs={**da.attrs, "units": mode.units, "long_name": mode.long_name},
    )
