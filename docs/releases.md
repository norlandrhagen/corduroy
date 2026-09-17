# Release notes

## 0.0.3 (unreleased)

### Fixed

- `__version__` is now read from the installed distribution metadata instead of
  being hand-maintained in `__init__.py`, where it had drifted to `0.0.1` while
  `pyproject.toml` said `0.0.3`. The publish workflow smoke-tests by printing
  `__version__`, so a release advertised the wrong version.

- A `NaN` cell no longer comes back with a finite value of its own. The Horn
  kernel gives the centre cell weight 0, so a nodata cell never entered its own
  stencil: the output was a ring of `NaN` around a valid-looking, meaningless
  centre. Slope, aspect and hillshade now mask the centre explicitly, so one
  `NaN` input cell yields a full 3x3 `NaN` block.

- A DEM with a single row or column now raises a `ValueError` instead of
  returning all `NaN`. Deriving spacing from one coordinate produced
  `np.median([])` -> `NaN`, and the only signal was a numpy `RuntimeWarning`.
  Pass `resolution=` to work with such a DEM.

- `resolution=` as a sequence is validated: anything other than a
  `(y_res, x_res)` pair raises a `ValueError` rather than an `IndexError` or a
  silent truncation.

- Slope no longer saturates near the poles. For a geographic CRS, longitude
  spacing is scaled to metres by `cos(mean_latitude)`, which collapses toward
  zero above ~89.9 degrees and made every cell read as a cliff (88.6 degrees of
  slope on data that reads 0.2 degrees at the equator). The scaling is now
  floored at its 89.9-degree value and raises a `RuntimeWarning` pointing at
  `resolution=` or a polar CRS.

- A DataArray whose dimensions are not recognised (`row`/`col` rather than
  `x`/`y`, `lat`/`lon`, …) now raises a `ValueError` naming the dimensions it
  found, the names it looked for, and the `x=` / `y=` override, instead of
  failing inside `transpose`.

### Added

- `compute_terrain`, `Slope`, `Aspect`, `Hillshade` and `TerrainMode` are
  exported from the package root. Previously only the two accessor classes were.

- Documentation site at
  [norlandrhagen.github.io/corduroy](https://norlandrhagen.github.io/corduroy/),
  with usage, design notes and an API reference.

### Changed

- Python 3.12 is now supported (was 3.13+), and dependency floors were lowered to
  what the code actually needs: `dask>=2024.1.0`, `xarray>=2025.9.0`,
  `numpy>=2.1.0`. CI resolves those exact minimums in a
  `--resolution lowest-direct` job and runs the suite against them, alongside
  3.12, 3.13 and 3.14.

## 0.0.2

- Aspect returns `NaN` on flat cells (hydro-flattened lakes, plateaus) rather
  than reporting them as facing due north.
- `z_factor` vertical exaggeration on all three methods.

## 0.0.1

- Initial release: `.dem` accessor on `DataArray` and `Dataset` with `slope()`,
  `aspect()` and `hillshade()`, Horn (1981) gradients, dask `map_overlap` support
  and CRS-aware degree-to-metre scaling.
