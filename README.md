# Corduroy - Dask aware lightweight DEM utilities for Xarray

** Warning: experimental**


## Usage

#### Installation

```python
uv add "corduroy @ git+https://github.com/norlandrhagen/corduroy"
# or 
pip install "git+https://github.com/norlandrhagen/corduroy"
```

```python
import xarray as xr
import corduroy # This is needed for the .dem accessor
import xproj # This is needed for the .proj accessor

# Load a 2D Raster DEM
ds = xr.open_dataset("DEM.zarr", engine="zarr", chunks="auto")

# Make sure you have a crs registered
ds = ds.proj.assign_crs("EPSG:4326")

# Calculate hillshade. Note you can use the `dem` accessor.
hs = ds['dem'].dem.hillshade()

hs['dem'].plot()
```


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