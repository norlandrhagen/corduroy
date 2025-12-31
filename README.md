# xCorduroy - Dask aware lightweight DEM utilities for Xarray

`xcorduroy` is a lightweight (`dask`, `numpy`, `xarray` and `xproj`) Xarray accessor for library for calculating hillshade, slope angle and aspects from DEMs. 

**Warning: experimental**


## Usage

#### Installation

```python
uv add xcorduroy
# or 
pip install xcorduroy
```

#### Notebooks
An example notebook can be found in `notebooks/DEM_example.ipynb`


#### Example
```python
import xarray as xr
import xcorduroy # This is needed for the .dem accessor
import xproj # This is needed for the .proj accessor

# Load a 2D Raster DEM
ds = xr.open_dataset("DEM.zarr", engine="zarr", chunks="auto")

# Make sure you have a crs registered
ds = ds.proj.assign_crs("EPSG:4326")

# Calculate hillshade. Note you can use the `dem` accessor.
hillshade = ds['dem'].dem.hillshade()
slope = ds['dem'].dem.slope()
aspect = ds['dem'].dem.aspect()


# Plotting
fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)

ds['dem'].plot(ax=axes[0, 0], cmap='terrain', add_colorbar=True)
axes[0, 0].set_title("Input DEM")

slope.plot(ax=axes[0, 1], cmap='magma')
axes[0, 1].set_title("Slope")

aspect.plot(ax=axes[1, 0], cmap='twilight')
axes[1, 0].set_title("Aspect")

hillshade.plot(ax=axes[1, 1], cmap='gray')
axes[1, 1].set_title("Hillshade")

plt.tight_layout()

```


### Methods
The three current methods implemented are `.hillshade()`, `.slope()` and `.aspect()`.  The hillshade method is based off of the `Horn, 1981` method. Details can be found in `src/xcorduroy/DEM.py`. They are inspired by similar methods in `xdem` and `xarray-spatial`. If you are looking for well-validated functions for scientific analysis, check out either of them.  This library is a limited scope lightweight take on some of the methods, not a replacement. 

## Development

This project uses `uv` for dependency management, `pytest` and `hypothesis` for testing,  `ty` for type-checking and `ruff` for linting. 


### Sync development environment 

```python
uv sync --all-extras
```


### Run type checking
```python
uv run ty check 
```

### Run linter

```python
uv run pre-commit run all-files
```

### Run tests
```
uv run pytest tests/
```


#### What's in the name

Corduroy is a textured snow surface left by groomers that forms regular peaks and valleys. 
