from importlib.metadata import PackageNotFoundError, version

from .accessors import DEMDataArrayAccessor, DEMDatasetAccessor
from .DEM import compute_terrain
from .types import Aspect, Hillshade, Slope, TerrainMode

try:
    __version__ = version("xcorduroy")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"

__all__ = [
    "Aspect",
    "DEMDataArrayAccessor",
    "DEMDatasetAccessor",
    "Hillshade",
    "Slope",
    "TerrainMode",
    "compute_terrain",
]
