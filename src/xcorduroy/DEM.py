import warnings
from typing import Any

import numpy as np
import xarray as xr

from .types import Hillshade, ModeType

# Metres per degree of latitude (WGS84 mean).
_M_PER_DEG = 111320.0

# Floor on cos(latitude) when converting degrees of longitude to metres, at
# 89.9 degrees. Without it a polar array collapses res_x toward zero and every
# cell reads as a cliff.
_MIN_COS_LAT = float(np.cos(np.deg2rad(89.9)))

# Placeholder step for a length-1 coordinate: sign only, magnitude unusable.
_UNKNOWN_STEP = 1.0


def _terrain_kernel(
    data: np.ndarray,
    res_x: float,
    res_y: float,
    mode: str,
    z_factor: float = 1.0,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    x_sign: float = 1.0,
    y_sign: float = 1.0,
) -> np.ndarray:
    """
    Compute slope, aspect or hillshade from a 1-pixel-padded 2D elevation array.

    Gradients use the Horn (1981) 3x3 kernel. Aspect and hillshade follow the
    ESRI/GDAL conventions: aspect is the downslope compass direction in degrees
    clockwise from north (flat cells are NaN); hillshade is
    ``cos(zenith)cos(slope) + sin(zenith)sin(slope)cos(az - aspect)`` clipped to 0-1.

    Args:
        data: 2D elevation array with 1-pixel padding, rows along y, columns along x
        res_x: Cell size along x, in the same units as elevation
        res_y: Cell size along y, in the same units as elevation
        mode: 'slope', 'aspect' or 'hillshade'
        z_factor: Vertical exaggeration factor
        azimuth: Light source azimuth in degrees clockwise from north
        altitude: Light source altitude above the horizon in degrees
        x_sign: +1 if x increases with column index (eastward), else -1
        y_sign: +1 if y increases with row index (northward), else -1

    Returns:
        float32 array of slope (degrees), aspect (degrees) or hillshade (0-1)
    """
    z = data * z_factor

    res_x = res_x if res_x != 0 else 1e-9
    res_y = res_y if res_y != 0 else 1e-9

    # Gradient toward increasing column / row index.
    dz_dcol = (
        (z[0:-2, 2:] + 2 * z[1:-1, 2:] + z[2:, 2:])
        - (z[0:-2, 0:-2] + 2 * z[1:-1, 0:-2] + z[2:, 0:-2])
    ) / (8.0 * res_x)
    dz_drow = (
        (z[2:, 0:-2] + 2 * z[2:, 1:-1] + z[2:, 2:])
        - (z[0:-2, 0:-2] + 2 * z[0:-2, 1:-1] + z[0:-2, 2:])
    ) / (8.0 * res_y)

    # Gradient positive eastward / northward regardless of array orientation.
    dz_dx = x_sign * dz_dcol
    dz_dy = y_sign * dz_drow

    magnitude = np.hypot(dz_dx, dz_dy)

    # The Horn kernel gives the centre cell weight 0, so a nodata cell never
    # enters its own stencil. Propagate it explicitly, otherwise a NaN hole
    # comes back as a ring of NaN around a finite, meaningless centre.
    center_nan = np.isnan(z[1:-1, 1:-1])

    if mode == "slope":
        slope = np.rad2deg(np.arctan(magnitude))
        return np.where(center_nan, np.nan, slope).astype(np.float32)

    if mode == "aspect":
        # Compass bearing of the downslope vector (-dz_dx, -dz_dy).
        aspect = np.mod(np.rad2deg(np.arctan2(-dz_dx, -dz_dy)), 360.0)
        aspect = np.where(center_nan | (magnitude == 0), np.nan, aspect)
        return aspect.astype(np.float32)

    zenith_rad = np.deg2rad(90.0 - altitude)
    az_math_rad = np.deg2rad(np.mod(360.0 - azimuth + 90.0, 360.0))
    slope_rad = np.arctan(magnitude)
    # Math angle (counter-clockwise from east) of the downslope vector.
    aspect_math_rad = np.arctan2(-dz_dy, -dz_dx)

    shaded = np.cos(zenith_rad) * np.cos(slope_rad) + np.sin(zenith_rad) * np.sin(
        slope_rad
    ) * np.cos(az_math_rad - aspect_math_rad)

    return np.where(center_nan, np.nan, np.clip(shaded, 0, 1)).astype(np.float32)


def _coord_step(coords: np.ndarray, wrap_360: bool, dim: str) -> float:
    """Signed median spacing of a 1D coordinate array.

    A single coordinate carries no spacing; assume ascending so the caller can
    still use the sign when ``resolution=`` was given explicitly.
    """
    if coords.size < 2:
        return _UNKNOWN_STEP
    d = np.diff(coords.astype(float))
    if wrap_360:
        d = np.where(np.abs(d) > 180, d - 360 * np.sign(d), d)
    return float(np.median(d))


def compute_terrain(
    da: xr.DataArray,
    mode: ModeType,
    resolution: float | int | tuple[float, float] | None = None,
    crs: Any = None,
    x_dim: str = "x",
    y_dim: str = "y",
    **kwargs: Any,
) -> xr.DataArray:
    """
    Compute terrain analysis from a 2D elevation DataArray.

    Args:
        da: Input elevation DataArray with dims ``y_dim`` and ``x_dim``
        mode: Terrain mode (Slope, Aspect, or Hillshade instance)
        resolution: Cell size as scalar or (y_res, x_res) in elevation units.
            Derived from coordinates if None. For geographic CRSs the degree
            spacing is converted to metres using the mean latitude; within 0.1
            degrees of a pole that scaling is clamped and a ``RuntimeWarning``
            is raised.
        crs: pyproj-like CRS with ``is_geographic``. Used for degree-to-metre scaling
        x_dim: Name of x dimension
        y_dim: Name of y dimension
        **kwargs: Passed to the kernel (e.g. ``z_factor`` vertical exaggeration)

    Returns:
        DataArray with computed terrain values
    """
    if da.ndim != 2:
        raise ValueError(f"Expected a 2D DataArray, got dims {da.dims}")
    da = da.transpose(y_dim, x_dim)

    x_coords = da[x_dim].values
    y_coords = da[y_dim].values
    is_geographic = bool(crs is not None and getattr(crs, "is_geographic", False))

    dx = _coord_step(x_coords, wrap_360=is_geographic, dim=x_dim)
    dy = _coord_step(y_coords, wrap_360=False, dim=y_dim)
    x_sign = -1.0 if dx < 0 else 1.0
    y_sign = -1.0 if dy < 0 else 1.0

    if resolution is None:
        for dim, coords in ((y_dim, y_coords), (x_dim, x_coords)):
            if coords.size < 2:
                raise ValueError(
                    f"Cannot derive spacing along {dim!r} from {coords.size} "
                    "coordinate(s). Pass resolution= explicitly, or use a DEM "
                    "with at least 2 cells per dimension."
                )
        res_x, res_y = abs(dx), abs(dy)
        if is_geographic:
            mean_lat = float(np.mean(y_coords))
            cos_lat = float(np.cos(np.deg2rad(mean_lat)))
            if cos_lat < _MIN_COS_LAT:
                warnings.warn(
                    f"Mean latitude {mean_lat:.4f} is within 0.1 degrees of the "
                    "pole; longitude spacing is clamped to the 89.9 degree "
                    "equivalent. Reproject to a polar CRS, or pass resolution=, "
                    "for meaningful results.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                cos_lat = _MIN_COS_LAT
            res_x *= _M_PER_DEG * cos_lat
            res_y *= _M_PER_DEG
    elif isinstance(resolution, (int, float)):
        res_y = res_x = abs(float(resolution))
    else:
        if len(resolution) != 2:
            raise ValueError(
                f"resolution must be a scalar or a (y_res, x_res) pair, "
                f"got {len(resolution)} values: {resolution!r}"
            )
        res_y, res_x = abs(float(resolution[0])), abs(float(resolution[1]))

    kernel_kwargs = {
        "z_factor": 1.0,
        **kwargs,
        "res_x": res_x,
        "res_y": res_y,
        "mode": mode.name,
        "x_sign": x_sign,
        "y_sign": y_sign,
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
        attrs={"units": mode.units, "long_name": mode.long_name},
    )
