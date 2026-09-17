# Design

## The kernel

All three methods come from one function, `_terrain_kernel` in
`src/xcorduroy/DEM.py`. It takes a 1-pixel-padded 2D elevation array and returns
an array of the same shape as the unpadded input.

Gradients use the Horn (1981) 3x3 kernel — a weighted finite difference over the
eight neighbours, with the centre-row and centre-column neighbours weighted 2x:

```
dz/dcol = ((z[-1,+1] + 2*z[0,+1] + z[+1,+1])
         - (z[-1,-1] + 2*z[0,-1] + z[+1,-1])) / (8 * res_x)
```

and the transpose for `dz/drow`. This is what `gdaldem` uses, which is why the
outputs match it.

## Array orientation

The kernel works in array index space, but slope and aspect are defined in world
space. `compute_terrain` takes the signed median spacing of each coordinate axis
and passes `x_sign` / `y_sign` into the kernel, which flips the gradients so they
are positive eastward and northward regardless of how the raster is stored. A
north-up raster (descending `y`) and a south-up one give the same answer.

Longitude is differenced with a dateline guard: a step larger than 180 degrees is
wrapped, so an array crossing the antimeridian does not read as one enormous cell.

## Conventions

| Output | Range | Notes |
|---|---|---|
| `slope` | 0–90 degrees | `atan(hypot(dz_dx, dz_dy))` |
| `aspect` | 0–360 degrees | Downslope direction, clockwise from north. Flat cells are `NaN`. |
| `hillshade` | 0–1 | Clipped, `float32` |

Hillshade is the standard Lambertian term:

```
cos(zenith) * cos(slope) + sin(zenith) * sin(slope) * cos(azimuth - aspect)
```

with `zenith = 90 - altitude`, and azimuth converted from compass degrees to the
counter-clockwise-from-east convention that `arctan2` returns.

### Why flat cells are NaN

A perfectly flat cell has no downslope direction. `arctan2(0, 0)` returns `0`,
which would report every flat cell as facing due north — visible as a false wall
of colour across hydro-flattened lakes and plateaus. `xcorduroy` returns `NaN`
there instead, matching `gdaldem`. Slope and hillshade are unaffected: a flat cell
has zero slope and takes the illumination of a horizontal surface.

## Degrees to metres

Elevation is in metres, but a geographic CRS gives coordinate spacing in degrees.
Dividing one by the other is meaningless, so for a geographic CRS the spacing is
scaled to metres before the gradient is taken:

```
res_y = dy * 111320
res_x = dx * 111320 * cos(mean_latitude)
```

`111320` is the WGS84 mean metres per degree of latitude. Using the **mean**
latitude of the array is the approximation: the true longitude scale varies across
the array, and the error grows with its north–south extent. For a tile this is
negligible; for a continental extent it is not, and the right answer is to
reproject to an equal-area or local projected CRS first.

Near a pole `cos(lat)` collapses toward zero, which would make every cell read as
a cliff. The scaling is floored at its 89.9-degree value and a `RuntimeWarning` is
raised.

An explicit `resolution=` bypasses all of this and is taken in elevation units.

## Chunked arrays

A 3x3 kernel needs one cell of context beyond each chunk boundary. For dask-backed
input the kernel is applied through `map_overlap` with `depth=1`,
`boundary="nearest"` and `trim=False`: dask grows each chunk by one cell on every
side, the kernel consumes that halo, and the output chunk comes back the same
shape as the input chunk. Seam values are therefore identical to the in-memory
result — `tests/test_DEM.py::test_dask_chunk_seams` asserts exactly that.

The in-memory path does the same thing with `np.pad(..., mode="edge")`, so the two
code paths agree at array edges as well as at chunk seams.

## NaN handling

`NaN` is not treated as nodata; it propagates. A single `NaN` cell in the input
produces a 3x3 block of `NaN` in the output: every neighbour of that cell had it
in its stencil, and the centre cell is masked explicitly, since the Horn kernel
gives it weight 0 and would otherwise return a finite, meaningless value. Mask or fill nodata before calling if that is not what you
want.

## Testing

Beyond unit tests against hand-derived analytic expectations, `tests/test_hypothesis.py`
asserts properties that must hold for any input: a flat surface has zero slope, a
linear ramp has constant slope, slope is invariant under translation and rotation
of the elevation field, aspect stays in 0–360, hillshade stays in 0–1, and halving
the resolution doubles the gradient.
