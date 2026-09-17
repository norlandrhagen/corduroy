# Usage

## Basic example

Every method hangs off the `.dem` accessor, which `import xcorduroy` registers on
both `DataArray` and `Dataset`. A CRS is required — `xcorduroy` uses it to decide
whether coordinate spacing is in degrees or metres.

```python
import matplotlib.pyplot as plt
import xarray as xr
import xproj    # registers the .proj accessor
import xcorduroy  # registers the .dem accessor

ds = xr.open_dataset("DEM.zarr", engine="zarr", chunks="auto")
ds = ds.proj.assign_crs(spatial_ref="EPSG:4326")

slope = ds["dem"].dem.slope()
aspect = ds["dem"].dem.aspect()
hillshade = ds["dem"].dem.hillshade()

fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
ds["dem"].plot(ax=axes[0, 0], cmap="terrain")
slope.plot(ax=axes[0, 1], cmap="magma")
aspect.plot(ax=axes[1, 0], cmap="twilight")
hillshade.plot(ax=axes[1, 1], cmap="gray")
plt.tight_layout()
```

A runnable version of this lives in `notebooks/DEM_example.ipynb`.

## Assigning a CRS

Without a CRS the accessor raises rather than guessing:

```python
ValueError: No CRS found on DataArray. You can assign a crs with: ...
```

Assign one with [xproj](https://github.com/benbovy/xproj):

```python
ds = ds.proj.assign_crs(spatial_ref="EPSG:4326", allow_override=True)
```

## Datasets vs DataArrays

On a `Dataset` the accessor finds the elevation variable for you. It looks for a
variable named `elevation`, `dem`, `height`, `z` or `band_data`; failing that, it
takes the single variable with two or more dimensions. Anything ambiguous raises:

```python
ds.dem.slope()              # auto-detect the elevation variable
ds.dem("elevation").slope() # name it explicitly
ds["elevation"].dem.slope() # or go straight to the DataArray
```

## Dimension names

`x` / `y`, `lon` / `lat`, `longitude` / `latitude` and `long` are recognised
automatically. Anything else needs naming:

```python
da.dem.slope(x="easting", y="northing")
```

Descending `y` (north-up rasters, the common case) is detected from the
coordinate values, so aspect and hillshade come out in the right orientation
either way.

## Chunked input

If the DataArray is dask-backed, the computation is applied with
`map_overlap` and a 1-cell halo, so values at chunk seams match the in-memory
result exactly:

```python
da = da.chunk({"y": 2048, "x": 2048})
slope = da.dem.slope()   # lazy; nothing is computed yet
slope.compute()
```

Array edges are padded by repeating the edge row/column, matching `gdaldem`.

## Resolution and units

By default cell size is derived from the coordinates:

- **Projected CRS** — spacing is used as-is, so it must already be in the same
  units as elevation (metres for most projected CRSs).
- **Geographic CRS** — degree spacing is converted to metres using the mean
  latitude of the array, `111320 m` per degree of latitude and
  `111320 * cos(lat)` per degree of longitude.

The mean-latitude approximation is fine for a tile and poor for a continental
extent. Within 0.1 degrees of a pole the longitude scaling is clamped and a
`RuntimeWarning` is raised; reproject to a polar CRS in that case.

Override it with `resolution=`, which is always taken in **elevation units** and
never degree-converted:

```python
da.dem.slope(resolution=30.0)          # square 30 m cells
da.dem.slope(resolution=(30.0, 20.0))  # (y_res, x_res)
```

## Vertical exaggeration

`z_factor` multiplies elevation before the gradient is taken. It is accepted by
all three methods:

```python
da.dem.hillshade(z_factor=2.0)  # exaggerate relief 2x
```

## Light source

`hillshade` takes the light position in degrees, defaulting to the GDAL
convention of the sun in the northwest, halfway up the sky:

```python
da.dem.hillshade(azimuth=315.0, altitude=45.0)
```

`azimuth` is clockwise from north; `altitude` is above the horizon.
