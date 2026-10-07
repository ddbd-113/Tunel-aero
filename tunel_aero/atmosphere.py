"""Model atmosfery: ISA (ICAO/ISO 2533) z odchyłką temperatury, ciśnieniem QNH,
tendencją ciśnienia (zmiana pogody w trakcie lotu) i wilgotnością powietrza.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

R_DRY = 287.05287     # J/(kg K)
R_VAPOR = 461.5       # J/(kg K)
GAMMA = 1.4
G0 = 9.80665
T0_ISA = 288.15       # K
P0_ISA = 101325.0     # Pa
RHO0_ISA = 1.225      # kg/m^3

# Warstwy ISA: (wysokość bazowa [m], gradient temperatury [K/m])
_LAYERS = ((0.0, -0.0065), (11000.0, 0.0), (20000.0, 0.001), (32000.0, 0.0028))


@dataclass
class AtmosphereState:
    temperature: float      # K
    pressure: float         # Pa
    density: float          # kg/m^3
    speed_of_sound: float   # m/s
    viscosity: float        # Pa s
    humidity: float         # wilgotność względna 0..1

    @property
    def temperature_c(self) -> float:
        return self.temperature - 273.15

    @property
    def density_ratio(self) -> float:
        return self.density / RHO0_ISA

    @property
    def density_altitude(self) -> float:
        """Wysokość gęstościowa [m] - kluczowa dla osiągów (gorąco + wysoko = mało ciągu)."""
        sigma = max(self.density_ratio, 1e-6)
        return (T0_ISA / 0.0065) * (1.0 - sigma ** (1.0 / 4.2559))


def saturation_vapor_pressure(t_c: float) -> float:
    """Ciśnienie pary nasyconej [Pa] (równanie Bucka)."""
    if t_c >= 0:
        return 611.21 * math.exp((18.678 - t_c / 234.5) * (t_c / (257.14 + t_c)))
    return 611.15 * math.exp((23.036 - t_c / 333.7) * (t_c / (279.82 + t_c)))


def isa_temperature(h: float) -> float:
    """Standardowa temperatura ISA [K] na wysokości geopotencjalnej h [m]."""
    T = T0_ISA
    for i, (hb, lapse) in enumerate(_LAYERS):
        h_top = _LAYERS[i + 1][0] if i + 1 < len(_LAYERS) else 1e9
        if h <= h_top:
            return T + lapse * (h - hb)
        T += lapse * (h_top - hb)
    return T


def pressure_altitude(p: float, p_ref: float = P0_ISA) -> float:
    """Wysokość ciśnieniowa [m] wg ISA (tak liczy ją barometr autopilota)."""
    return (T0_ISA / 0.0065) * (1.0 - (p / p_ref) ** (R_DRY * 0.0065 / G0))


class Atmosphere:
    """Atmosfera z odchyłką od ISA.

    Parametry:
        temperature_offset: odchyłka temperatury od ISA [K] (stała w całym profilu),
        sea_level_pressure: ciśnienie zredukowane do poziomu morza QNH [Pa],
        relative_humidity: wilgotność względna 0..1,
        pressure_tendency: zmiana QNH w czasie [Pa/s] (np. nadciągający front).
    """

    def __init__(self, temperature_offset: float = 0.0, sea_level_pressure: float = P0_ISA,
                 relative_humidity: float = 0.0, pressure_tendency: float = 0.0):
        self.dT = temperature_offset
        self.p0 = sea_level_pressure
        self.rh = max(0.0, min(1.0, relative_humidity))
        self.dp_dt = pressure_tendency

    @classmethod
    def from_ground_conditions(cls, ground_elevation: float, temperature_c: float | None = None,
                               qnh_hpa: float | None = None, relative_humidity: float = 0.0,
                               pressure_tendency_hpa_h: float = 0.0, isa_offset: float | None = None):
        """Tworzy atmosferę z warunków zmierzonych przy ziemi (jak z komunikatu METAR)."""
        if temperature_c is not None:
            dT = (temperature_c + 273.15) - isa_temperature(ground_elevation)
        else:
            dT = isa_offset or 0.0
        p0 = (qnh_hpa if qnh_hpa is not None else P0_ISA / 100.0) * 100.0
        return cls(dT, p0, relative_humidity, pressure_tendency_hpa_h * 100.0 / 3600.0)

    def at(self, altitude_msl: float, t: float = 0.0) -> AtmosphereState:
        h = max(-1000.0, min(altitude_msl, 47000.0))
        p = self.p0 + self.dp_dt * t
        T_base = T0_ISA + self.dT
        for i, (hb, lapse) in enumerate(_LAYERS):
            h_top = _LAYERS[i + 1][0] if i + 1 < len(_LAYERS) else 1e9
            dh = min(h, h_top) - hb
            if abs(lapse) > 1e-12:
                T_new = T_base + lapse * dh
                p *= (T_new / T_base) ** (-G0 / (R_DRY * lapse))
            else:
                T_new = T_base
                p *= math.exp(-G0 * dh / (R_DRY * T_base))
            T_base = T_new
            if h <= h_top:
                break
        T = T_base
        t_c = T - 273.15
        e = self.rh * saturation_vapor_pressure(t_c)
        e = min(e, 0.5 * p)
        rho = (p - e) / (R_DRY * T) + e / (R_VAPOR * T)
        a = math.sqrt(GAMMA * R_DRY * T)
        mu = 1.458e-6 * T ** 1.5 / (T + 110.4)   # Sutherland
        return AtmosphereState(T, p, rho, a, mu, self.rh)
