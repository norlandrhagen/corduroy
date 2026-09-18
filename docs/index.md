---
hide:
  - toc
---

# xcorduroy

Slope, aspect and hillshade from digital elevation models, as an Xarray accessor.

`xcorduroy` is deliberately small: `numpy`, `xarray`, `dask` and `xproj`, nothing
else. Gradients use the Horn (1981) 3x3 kernel; aspect and hillshade follow the
ESRI/GDAL conventions (not numerically cross-checked against `gdaldem`). Chunked arrays are
handled with a dask halo, so the result is identical whether the DEM is lazy or
in memory.

!!! warning "Experimental"
    APIs may change without notice.

## Installation

```bash
uv add xcorduroy
# or
pip install xcorduroy
```

## Quick look

```python
import xarray as xr
import xproj    # registers the .proj accessor
import xcorduroy  # registers the .dem accessor

ds = xr.open_dataset("DEM.zarr", engine="zarr", chunks="auto")
ds = ds.proj.assign_crs(spatial_ref="EPSG:4326")

hillshade = ds["dem"].dem.hillshade()
slope = ds["dem"].dem.slope()
aspect = ds["dem"].dem.aspect()
```

See [Usage](usage.md) for the full worked example, and the
[API Reference](api.md) for conventions and signatures.

## Scope

The methods are inspired by `xdem` and `xarray-spatial`. If you need
well-validated functions for scientific analysis, use one of those. This is a
limited-scope, lightweight take on a few of the methods, not a replacement.

## What's in the name

Corduroy is the textured snow surface left by groomers, regular peaks and
valleys.
