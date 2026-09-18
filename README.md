# xcorduroy

**Dask aware lightweight DEM utilities for Xarray**

[Documentation](https://norlandrhagen.github.io/corduroy/) ·
[Usage](https://norlandrhagen.github.io/corduroy/usage/) ·
[API](https://norlandrhagen.github.io/corduroy/api/)

`xcorduroy` is a small Xarray accessor for computing hillshade, slope and aspect
from DEMs, with `dask`, `numpy`, `xarray` and `xproj` as its only dependencies.

> **Warning:** experimental. APIs may change without notice.

## Installation

```bash
uv add xcorduroy
# or
pip install xcorduroy
```

## Example

```python
import xarray as xr
import xproj    # registers the .proj accessor
import xcorduroy  # registers the .dem accessor

ds = xr.open_dataset("DEM.zarr", engine="zarr", chunks="auto")
ds = ds.proj.assign_crs(spatial_ref="EPSG:4326")

slope = ds["dem"].dem.slope()          # degrees, 0-90
aspect = ds["dem"].dem.aspect()        # degrees clockwise from north; flat cells NaN
hillshade = ds["dem"].dem.hillshade()  # 0-1, light from azimuth 315 / altitude 45
```

A plotted, runnable version is in `notebooks/DEM_example.ipynb`. See
[Usage](https://norlandrhagen.github.io/corduroy/usage/) for `resolution=`,
`z_factor=`, dimension naming and chunking, and
[API Reference](https://norlandrhagen.github.io/corduroy/api/) for conventions
and the degree-to-metre approximation.

## Scope

The methods are inspired by `xdem` and `xarray-spatial`. If you need
well-validated functions for scientific analysis, use one of those. This is a
limited-scope, lightweight take on a few of the methods, not a replacement.

## Development

```bash
uv sync --all-groups
uv run pytest tests -n auto     # tests
uv run ty check src/            # type check
uv run prek run --all-files     # lint and format
uv run mkdocs serve             # docs preview
```

See [Contributing](https://norlandrhagen.github.io/corduroy/contributing/).

## What's in the name

Corduroy is the textured snow surface left by groomers, regular peaks and
valleys.
