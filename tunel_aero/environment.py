"""Środowisko: atmosfera + wiatr + pogoda + teren."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .atmosphere import Atmosphere, AtmosphereState
from .math3d import GRAVITY
from .weather import Degradation, Weather
from .wind import WindField, WindSample


@dataclass
class EnvSample:
    t: float
    atm: AtmosphereState
    wind: WindSample
    deg: Degradation
    h_agl: float
    alt_msl: float
    ground_d: float = 0.0        # współrzędna D terenu pod pojazdem (NED)
    gravity: float = GRAVITY

    @property
    def wind_ned(self) -> np.ndarray:
        return self.wind.total


class Environment:
    """Płaski teren na wysokości ground_elevation [m n.p.m.]; początek układu NED w punkcie startu."""

    def __init__(self, atmosphere: Atmosphere | None = None, wind: WindField | None = None,
                 weather: Weather | None = None, ground_elevation: float = 0.0):
        self.atmosphere = atmosphere or Atmosphere()
        self.wind = wind or WindField()
        self.weather = weather or Weather()
        self.ground_elevation = ground_elevation

    def terrain_height(self, pn: float, pe: float) -> float:
        """Wysokość terenu względem punktu startu [m] (miejsce na mapę terenu)."""
        return 0.0

    def sample(self, t: float, dt: float, pos_ned: np.ndarray, airspeed: float,
               advance: bool = True) -> EnvSample:
        terrain = self.terrain_height(pos_ned[0], pos_ned[1])
        h_agl = -pos_ned[2] - terrain
        alt_msl = self.ground_elevation - pos_ned[2]
        atm = self.atmosphere.at(alt_msl, t)
        wind = self.wind.sample(t, dt, pos_ned, max(h_agl, 0.0), airspeed, advance)
        return EnvSample(t, atm, wind, self.weather.degradation(), h_agl, alt_msl, -terrain)
