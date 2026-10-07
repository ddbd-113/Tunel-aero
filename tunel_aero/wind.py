"""Pole wiatru: profil wiatru średniego w warstwie przyziemnej, turbulencja
Drydena (MIL-F-8785C / MIL-HDBK-1797), podmuchy dyskretne (1-cos),
mikroburst (model Oseguery-Bowlesa, NASA TM-100632) i kominy termiczne.

Wszystkie prędkości wiatru są w układzie NED (prędkość ruchu powietrza).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .math3d import wind_from_to_ned

FT = 0.3048
KAPPA = 0.4  # stała von Karmana

# Kategorie terenu (EN 1991-1-4 / Eurokod 1): długość szorstkości z0 [m] i wysokość minimalna z_min [m]
TERRAIN_ROUGHNESS = {
    "water": 0.003,      # morze, jeziora (kat. 0)
    "snow": 0.01,        # płaski teren, śnieg, lotnisko (kat. I)
    "open": 0.05,        # otwarty teren, trawa, pojedyncze przeszkody (kat. II) - teren odniesienia
    "farmland": 0.1,     # pola uprawne z żywopłotami
    "suburbs": 0.3,      # przedmieścia, wsie, zabudowa jednorodzinna (kat. III)
    "forest": 0.5,       # las
    "city": 1.0,         # zwarta zabudowa miejska, budynki > 15 m (kat. IV)
}
Z0_REF = 0.05


def terrain_z_min(z0: float) -> float:
    return 1.0 if z0 <= 0.01 else 2.0 if z0 <= 0.05 else 3.0 if z0 <= 0.1 else 5.0 if z0 <= 0.5 else 10.0


# Intensywności turbulencji: (W20 [m/s] dla niskich wysokości, sigma [m/s] dla dużych wysokości)
TURBULENCE_LEVELS = {
    "none": (0.0, 0.0),
    "light": (15 * 0.5144, 1.0),
    "moderate": (30 * 0.5144, 2.3),
    "severe": (45 * 0.5144, 4.6),
}


class WindProfile:
    """Wiatr średni zależny od wysokości.

    Tryby:
      * log (domyślnie) - profil logarytmiczny wg EN 1991-1-4: `speed` to prędkość z prognozy
        pogody (10 m nad terenem otwartym), a lokalny teren (z0) zmienia profil:
        V(h) = speed * k_r * ln(max(h, z_min) / z0),  k_r = 0.19 (z0 / 0.05)^0.07.
        Nad miastem wiatr przy ziemi jest słabszy, ale dużo bardziej porywisty.
        Z reference: local - `speed` zmierzona lokalnie na wysokości reference_height.
      * power - prawo potęgowe V(h) = V_ref (h/h_ref)^alpha,
      * constant - wiatr niezależny od wysokości,
      * layers - tabela [wysokość AGL, prędkość, kierunek] np. z sondażu aerologicznego.
    """

    def __init__(self, speed: float = 0.0, direction: float = 0.0, reference_height: float = 10.0,
                 roughness: float = Z0_REF, profile: str = "log", power_exponent: float = 0.143,
                 layers: list | None = None, max_height: float = 600.0, reference: str = "forecast"):
        self.speed = speed
        self.direction = direction
        self.h_ref = reference_height
        self.z0 = roughness
        self.z_min = terrain_z_min(roughness)
        self.profile = profile if not layers else "layers"
        self.alpha = power_exponent
        self.max_height = max_height
        self.reference = reference
        self.k_r = 0.19 * (roughness / Z0_REF) ** 0.07
        self.layers = None
        if layers:
            arr = np.array(sorted(layers, key=lambda r: r[0]), dtype=float)
            n, e = zip(*(wind_from_to_ned(s, d) for s, d in arr[:, 1:3]))
            self.layers = (arr[:, 0], np.array(n), np.array(e))

    def _scale(self, h: float) -> float:
        h = min(h, self.max_height)
        if self.profile == "constant":
            return 1.0
        if self.profile == "power":
            return (max(h, 0.0) / self.h_ref) ** self.alpha
        if h <= 0.0:
            return 0.0
        hz = max(h, self.z_min)
        if self.reference == "local":
            return math.log(hz / self.z0) / math.log(max(self.h_ref, self.z_min) / self.z0)
        return self.k_r * math.log(hz / self.z0)

    def mean_wind(self, h_agl: float) -> np.ndarray:
        if self.layers is not None:
            hs, ns, es = self.layers
            if h_agl < hs[0]:
                hz = max(h_agl, self.z_min)
                s = 0.0 if h_agl <= 0 else min(1.0, math.log(hz / self.z0) / math.log(max(hs[0], self.z_min) / self.z0))
                return np.array([ns[0] * s, es[0] * s, 0.0])
            return np.array([np.interp(h_agl, hs, ns), np.interp(h_agl, hs, es), 0.0])
        vn, ve = wind_from_to_ned(self.speed, self.direction)
        s = self._scale(h_agl)
        return np.array([vn * s, ve * s, 0.0])

    def friction_velocity(self) -> float:
        """Prędkość dynamiczna u* [m/s] - miara turbulencji mechanicznej (u* = kappa k_r V_b)."""
        if self.layers is not None:
            hs, ns, es = self.layers
            v = math.hypot(ns[0], es[0])
            return KAPPA * v / math.log(max(hs[0], 2 * self.z0) / self.z0)
        if self.profile == "log" and self.reference != "local":
            return KAPPA * self.k_r * self.speed
        h = max(self.h_ref, self.z_min)
        return KAPPA * self.speed * self._scale(h) / math.log(h / self.z0)


class DrydenTurbulence:
    """Turbulencja Drydena - filtry kształtujące napędzane białym szumem.

    intensity: 'none' | 'light' | 'moderate' | 'severe' | 'auto' | liczba (sigma [m/s])
      'auto' - intensywność z teorii warstwy przyziemnej (sigma_u = 2.4 u*, sigma_v = 1.9 u*,
               sigma_w = 1.25 u*), zależna od prędkości wiatru i szorstkości terenu
               (w mieście turbulencja jest kilka razy silniejsza niż nad wodą).
    """

    def __init__(self, intensity="auto", profile: WindProfile | None = None,
                 rng: np.random.Generator | None = None):
        self.intensity = intensity
        self.profile = profile or WindProfile()
        self.rng = rng or np.random.default_rng()
        self.xu = 0.0
        self.xv = np.zeros(2)
        self.xw = np.zeros(2)
        self.value_body_frame = np.zeros(3)

    def params(self, h_agl: float) -> tuple[np.ndarray, np.ndarray]:
        """Zwraca (sigma[u,v,w], L[u,v,w]) w metrach."""
        h_ft = max(h_agl / FT, 10.0)
        # skale długości (MIL-F-8785C)
        if h_ft <= 1000:
            Lw = h_ft
            Lu = Lv = h_ft / (0.177 + 0.000823 * h_ft) ** 1.2
        elif h_ft >= 2000:
            Lu = Lv = Lw = 1750.0
        else:
            f = (h_ft - 1000) / 1000
            Lu = Lv = 1000 / (0.177 + 0.823) ** 1.2 * (1 - f) + 1750 * f
            Lw = 1000 * (1 - f) + 1750 * f
        L = np.array([Lu, Lv, Lw]) * FT

        if isinstance(self.intensity, (int, float)):
            sig = np.full(3, float(self.intensity))
        elif self.intensity == "auto":
            us = self.profile.friction_velocity()
            decay = 1.0 if h_agl < 300 else max(0.3, 1.0 - 0.7 * (h_agl - 300) / 700)
            sig = np.array([2.4, 1.9, 1.25]) * us * decay
        else:
            w20, sig_hi = TURBULENCE_LEVELS[self.intensity]
            sw = 0.1 * w20
            hf = min(h_ft, 1000.0)
            su = sw / (0.177 + 0.000823 * hf) ** 0.4
            sig_lo = np.array([su, su, sw])
            if h_ft <= 1000:
                sig = sig_lo
            elif h_ft >= 2000:
                sig = np.full(3, sig_hi)
            else:
                f = (h_ft - 1000) / 1000
                sig = sig_lo * (1 - f) + sig_hi * f
        return sig, L

    def update(self, dt: float, h_agl: float, airspeed: float, wind_heading: float) -> np.ndarray:
        """Krok filtra; zwraca turbulencję w NED. wind_heading - kierunek, W KTÓRYM wieje wiatr [rad]."""
        if self.intensity in ("none", 0, 0.0):
            return np.zeros(3)
        sig, L = self.params(h_agl)
        V = max(airspeed, 2.0)  # hipoteza zamrożonej turbulencji Taylora
        n = self.rng.standard_normal(3)
        # u: filtr 1. rzędu (dokładna dyskretyzacja, wariancja sigma_u^2)
        a = math.exp(-V * dt / L[0])
        self.xu = a * self.xu + sig[0] * math.sqrt(1 - a * a) * n[0]
        out = [self.xu]
        # v, w: filtry 2. rzędu Drydena (1 + sqrt3 T s)/(1 + T s)^2
        for i, x in ((1, self.xv), (2, self.xw)):
            T = L[i] / V
            a = math.exp(-dt / T)
            x[0] = a * x[0] + math.sqrt((1 - a * a) / (2 * T)) * n[i]
            x[1] += dt / T * (x[0] - x[1])
            out.append(sig[i] * math.sqrt(T) * (math.sqrt(3) * x[0] + (1 - math.sqrt(3)) * x[1]))
        u, v, w = out
        self.value_body_frame = np.array(out)
        c, s = math.cos(wind_heading), math.sin(wind_heading)
        return np.array([u * c - v * s, u * s + v * c, w])


@dataclass
class DiscreteGust:
    """Podmuch dyskretny o profilu 1-cos. vertical > 0 oznacza podmuch w górę."""
    t_start: float
    duration: float = 2.0
    speed: float = 5.0
    direction: float = 0.0     # skąd wieje [deg]
    vertical: float = 0.0      # [m/s], + w górę
    shape: str = "bump"        # bump: narasta i opada; step: narasta i trzyma się

    def value(self, t: float) -> np.ndarray:
        if t < self.t_start:
            return np.zeros(3)
        tau = t - self.t_start
        if self.shape == "step":
            k = 0.5 * (1 - math.cos(math.pi * min(tau / self.duration, 1.0)))
        else:
            if tau > self.duration:
                return np.zeros(3)
            k = 0.5 * (1 - math.cos(2 * math.pi * tau / self.duration))
        vn, ve = wind_from_to_ned(self.speed, self.direction)
        return k * np.array([vn, ve, -self.vertical])


@dataclass
class Microburst:
    """Mikroburst - silny prąd zstępujący rozpływający się przy ziemi (uskok wiatru).

    Model Oseguery-Bowlesa (1988, NASA TM-100632) - analityczny, spełnia równanie ciągłości.
    Intensywność podaje się jako maksymalny wypływ poziomy przy ziemi `max_outflow` [m/s]
    (typowo 10-25 m/s) albo jako prąd zstępujący wysoko w osi `max_downdraft` [m/s].
    """
    north: float = 0.0
    east: float = 0.0
    radius: float = 600.0          # promień kolumny prądu zstępującego [m]
    max_outflow: float | None = 12.0
    max_downdraft: float | None = None
    z_star: float = 200.0          # charakterystyczna głębokość wypływu [m]
    eps: float = 30.0              # grubość warstwy przyściennej [m]
    t_start: float = 0.0
    ramp_time: float = 20.0        # czas narastania intensywności [s]
    drift_north: float = 0.0       # przesuwanie się komórki burzowej [m/s]
    drift_east: float = 0.0

    def __post_init__(self):
        if self.max_downdraft is not None:
            self.lam = self.max_downdraft / (self.z_star - self.eps)
        else:
            zp = math.log(self.z_star / self.eps) / (1 / self.eps - 1 / self.z_star)
            fmax = math.exp(-zp / self.z_star) - math.exp(-zp / self.eps)
            self.lam = self.max_outflow / (self.radius / 2 * 0.6381 * fmax)

    def value(self, t: float, pn: float, pe: float, h_agl: float) -> np.ndarray:
        if t < self.t_start:
            return np.zeros(3)
        k = min(1.0, (t - self.t_start) / max(self.ramp_time, 1e-6))
        lam = self.lam
        cn = self.north + self.drift_north * (t - self.t_start)
        ce = self.east + self.drift_east * (t - self.t_start)
        dn, de = pn - cn, pe - ce
        r = math.hypot(dn, de)
        z = max(h_agl, 0.0)
        R = self.radius
        ex = math.exp(-(r / R) ** 2)
        fz = math.exp(-z / self.z_star) - math.exp(-z / self.eps)
        w_up = -lam * ex * (self.eps * (math.exp(-z / self.eps) - 1) - self.z_star * (math.exp(-z / self.z_star) - 1))
        if r < 1e-3:
            un = ue = 0.0
        else:
            ur = lam * R * R / (2 * r) * (1 - ex) * fz
            un, ue = ur * dn / r, ur * de / r
        return k * np.array([un, ue, -w_up])


@dataclass
class Thermal:
    """Komin termiczny (noszenie) z pierścieniem opadania wokół (model Gedeona)."""
    north: float = 0.0
    east: float = 0.0
    radius: float = 80.0
    core_updraft: float = 3.0      # [m/s]
    top: float = 1500.0            # podstawa chmur / wierzchołek termiki AGL [m]
    drift_with_wind: bool = True

    def value(self, pn: float, pe: float, h_agl: float, drift: tuple[float, float]) -> np.ndarray:
        cn = self.north + (drift[0] if self.drift_with_wind else 0.0)
        ce = self.east + (drift[1] if self.drift_with_wind else 0.0)
        r2 = ((pn - cn) ** 2 + (pe - ce) ** 2) / self.radius ** 2
        if r2 > 9:
            return np.zeros(3)
        fz = min(1.0, max(0.0, h_agl / 30.0)) * min(1.0, max(0.0, (self.top - h_agl) / 100.0))
        w = self.core_updraft * math.exp(-r2) * (1 - r2) * fz
        return np.array([0.0, 0.0, -w])


@dataclass
class WindSample:
    total: np.ndarray
    mean: np.ndarray
    turbulence: np.ndarray
    gust: np.ndarray
    convective: np.ndarray       # mikroburst + termika


class WindField:
    def __init__(self, profile: WindProfile | None = None, turbulence: DrydenTurbulence | None = None,
                 gusts: list[DiscreteGust] | None = None, microbursts: list[Microburst] | None = None,
                 thermals: list[Thermal] | None = None):
        self.profile = profile or WindProfile()
        self.turbulence = turbulence
        self.gusts = gusts or []
        self.microbursts = microbursts or []
        self.thermals = thermals or []
        self._drift = np.zeros(2)

    @staticmethod
    def random_gusts(rng: np.random.Generator, duration: float, rate_per_min: float = 2.0,
                     speed=(2.0, 6.0), gust_duration=(1.0, 4.0), direction: float = 0.0,
                     direction_spread: float = 45.0, vertical=(-1.0, 1.0)) -> list[DiscreteGust]:
        gusts, t = [], 0.0
        if rate_per_min <= 0:
            return gusts
        while True:
            t += rng.exponential(60.0 / rate_per_min)
            if t > duration:
                return gusts
            gusts.append(DiscreteGust(
                t_start=t, duration=rng.uniform(*gust_duration), speed=rng.uniform(*speed),
                direction=direction + rng.uniform(-direction_spread, direction_spread),
                vertical=rng.uniform(*vertical)))

    def sample(self, t: float, dt: float, pos_ned: np.ndarray, h_agl: float, airspeed: float,
               advance: bool = True) -> WindSample:
        mean = self.profile.mean_wind(h_agl)
        heading = math.atan2(mean[1], mean[0]) if (mean[0] or mean[1]) else 0.0
        if self.turbulence is not None and advance:
            turb = self.turbulence.update(dt, h_agl, airspeed, heading)
            self._last_turb = turb
        else:
            turb = getattr(self, "_last_turb", np.zeros(3))
        gust = np.zeros(3)
        for g in self.gusts:
            if g.t_start <= t <= g.t_start + g.duration or g.shape == "step":
                gust = gust + g.value(t)
        conv = np.zeros(3)
        for m in self.microbursts:
            conv = conv + m.value(t, pos_ned[0], pos_ned[1], h_agl)
        if self.thermals:
            if advance:
                self._drift += self.profile.mean_wind(max(h_agl, 50.0))[:2] * dt
            for th in self.thermals:
                conv = conv + th.value(pos_ned[0], pos_ned[1], h_agl, tuple(self._drift))
        total = mean + turb + gust + conv
        if h_agl < 0.5:  # przy samej ziemi brak składowej pionowej
            total = total * np.array([1.0, 1.0, max(h_agl, 0.0) / 0.5])
        return WindSample(total, mean, turb, gust, conv)
