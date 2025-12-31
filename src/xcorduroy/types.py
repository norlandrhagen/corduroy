from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class TerrainMode(ABC):
    """Base class for all terrain analysis methods."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def units(self) -> str: ...

    @property
    @abstractmethod
    def long_name(self) -> str: ...


@dataclass(frozen=True)
class Slope(TerrainMode):
    name: str = "slope"
    units: str = "degrees"
    long_name: str = "terrain_slope"


@dataclass(frozen=True)
class Aspect(TerrainMode):
    name: str = "aspect"
    units: str = "degrees"
    long_name: str = "terrain_aspect"


@dataclass(frozen=True)
class Hillshade(TerrainMode):
    azimuth: float = 315.0
    altitude: float = 45.0
    name: str = "hillshade"
    units: str = "0-1"
    long_name: str = "hillshade"


ModeType = Slope | Aspect | Hillshade
