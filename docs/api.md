# API Reference

The public API is the two accessors plus the function behind them.
`import xcorduroy` registers `.dem` on `DataArray` and `Dataset`; in most cases
that is all you need, and `compute_terrain` is the escape hatch for calling the
machinery directly.

Terrain modes (`Slope`, `Aspect`, `Hillshade`) are frozen dataclasses carrying the
output's name, units and `long_name`, and in the hillshade case the light
position. They are what `compute_terrain` dispatches on.

::: xcorduroy.DEMDataArrayAccessor

::: xcorduroy.DEMDatasetAccessor

::: xcorduroy.compute_terrain

::: xcorduroy.types.TerrainMode

::: xcorduroy.types.Slope

::: xcorduroy.types.Aspect

::: xcorduroy.types.Hillshade
