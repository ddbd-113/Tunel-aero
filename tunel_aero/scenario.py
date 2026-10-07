"""Scenariusze testowe w YAML: budowa pojazdu, środowiska, misji, awarii i ocena kryteriów."""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from . import tunnel
from .atmosphere import Atmosphere
from .control import FixedWingAutopilot, MultirotorAutopilot
from .environment import Environment
from .failures import FailureManager
from .metrics import compute_metrics, evaluate_criteria
from .sensors import NavigationSystem
from .sim import FlightLog, Simulation
from .vehicles import build_vehicle, resolve_vehicle_config
from .weather import Weather
from .wind import (TERRAIN_ROUGHNESS, DiscreteGust, DrydenTurbulence, Microburst, Thermal, WindField,
                   WindProfile)

PACKAGE_ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ narzędzia
def load_yaml(path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_dotted(d: dict, key: str, value) -> None:
    """set_dotted(cfg, 'environment.wind.speed', 12) - ustawia zagnieżdżony klucz."""
    parts = key.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def _resolve_path(p: str, base_dir: Path) -> Path:
    cand = (base_dir / p)
    if cand.exists():
        return cand
    for root in (PACKAGE_ROOT, Path.cwd()):
        if (root / p).exists():
            return root / p
    raise FileNotFoundError(p)


def load_scenario(path_or_dict, overrides: dict | None = None) -> tuple[dict, Path]:
    if isinstance(path_or_dict, dict):
        sc, base = copy.deepcopy(path_or_dict), Path.cwd()
    else:
        path = Path(path_or_dict)
        sc, base = load_yaml(path), path.resolve().parent
    if "extends" in sc:   # dziedziczenie scenariuszy
        parent, pbase = load_scenario(_resolve_path(sc.pop("extends"), base))
        sc = deep_merge(parent, sc)
    for k, v in (overrides or {}).items():
        if v is None:   # None usuwa klucz (np. mission.airspeed -> prędkość przelotowa pojazdu)
            parent = sc
            parts_ = k.split(".")
            for p_ in parts_[:-1]:
                parent = parent.get(p_, {}) if isinstance(parent, dict) else {}
            if isinstance(parent, dict):
                parent.pop(parts_[-1], None)
            continue
        set_dotted(sc, k, v)
    veh = sc.get("vehicle", {})
    if isinstance(veh, str):
        vpath = _resolve_path(veh, base)
        veh = load_yaml(vpath)
        veh.setdefault("_base_dir", str(vpath.resolve().parent))
    elif isinstance(veh, dict):
        veh = dict(veh)
        veh.setdefault("_base_dir", str(base))
    sc["vehicle"] = deep_merge(veh, sc.get("vehicle_overrides", {}))
    return sc, base


# ------------------------------------------------------------------ budowa
def build_environment(cfg: dict, rng: np.random.Generator, duration: float) -> Environment:
    elev = cfg.get("ground_elevation", 0.0)
    atm = Atmosphere.from_ground_conditions(
        elev, temperature_c=cfg.get("temperature_c"), qnh_hpa=cfg.get("qnh_hpa"),
        relative_humidity=cfg.get("humidity", 0.5),
        pressure_tendency_hpa_h=cfg.get("pressure_tendency_hpa_h", 0.0), isa_offset=cfg.get("isa_offset"))
    w = cfg.get("wind", {}) or {}
    terrain = w.get("terrain", "open")
    if "roughness" not in w and terrain not in TERRAIN_ROUGHNESS:
        raise ValueError(f"Nieznany teren '{terrain}'. Dostępne: {', '.join(TERRAIN_ROUGHNESS)}")
    rough = w.get("roughness", TERRAIN_ROUGHNESS.get(terrain, 0.05))
    profile = WindProfile(speed=w.get("speed", 0.0), direction=w.get("direction", 0.0),
                          reference_height=w.get("reference_height", 10.0), roughness=rough,
                          profile=w.get("profile", "log"), layers=w.get("layers"),
                          reference=w.get("reference", "forecast"))
    turb_cfg = w.get("turbulence", "auto" if w.get("speed", 0) or w.get("layers") else "none")
    turbulence = None if turb_cfg in ("none", None, 0) else DrydenTurbulence(turb_cfg, profile, rng)
    gusts = [DiscreteGust(t_start=g.get("t", 0.0), duration=g.get("duration", 2.0), speed=g.get("speed", 5.0),
                          direction=g.get("direction", w.get("direction", 0.0)), vertical=g.get("vertical", 0.0),
                          shape=g.get("shape", "bump")) for g in w.get("gusts", [])]
    rg = w.get("random_gusts")
    if rg:
        gusts += WindField.random_gusts(
            rng, duration, rate_per_min=rg.get("rate_per_min", 2.0), speed=tuple(rg.get("speed", (2, 6))),
            gust_duration=tuple(rg.get("duration", (1, 4))), direction=w.get("direction", 0.0),
            direction_spread=rg.get("direction_spread", 45.0), vertical=tuple(rg.get("vertical", (-1, 1))))
    mbs = [Microburst(**m) for m in w.get("microbursts", [])]
    ths = [Thermal(**t) for t in w.get("thermals", [])]
    wind = WindField(profile, turbulence, gusts, mbs, ths)
    ic = cfg.get("icing", {}) or {}
    weather = Weather(rain_rate=cfg.get("rain_mm_h", 0.0), lwc=ic.get("lwc", 0.0),
                      cloud_base=ic.get("cloud_base", 0.0), cloud_top=ic.get("cloud_top", 0.0),
                      accretion_rate=ic.get("accretion_rate", 2.0e-4), deicing=ic.get("deicing", False))
    return Environment(atm, wind, weather, elev)


@dataclass
class ScenarioResult:
    name: str
    scenario: dict
    log: FlightLog
    metrics: dict
    criteria: list
    seed: int
    extra: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(c["passed"] for c in self.criteria)


def build_simulation(sc: dict, seed: int | None = None) -> tuple[Simulation, dict]:
    seed = sc.get("seed", 0) if seed is None else seed
    rng = np.random.default_rng(seed)
    duration = float(sc.get("duration", 120.0))
    env = build_environment(sc.get("environment", {}) or {}, rng, duration)
    vcfg = resolve_vehicle_config(sc["vehicle"])
    sc["vehicle"] = vcfg
    veh = build_vehicle(vcfg, rng)
    batt_t = (vcfg.get("battery", {}) or {}).get("temperature_c")
    ground = env.atmosphere.at(env.ground_elevation)
    if batt_t is None:   # akumulator w temperaturze otoczenia, chyba że podgrzany
        veh.battery.temp_c = ground.temperature_c
    mission = sc.get("mission", {}) or {}
    start = mission.get("start", {}) or {}
    info: dict = {"seed": seed}
    if veh.kind == "multirotor":
        ap = MultirotorAutopilot(veh.controller_model(), mission, sc.get("gains"))
        x0 = veh.initial_state(north=start.get("north", 0.0), east=start.get("east", 0.0),
                               altitude=start.get("altitude", 0.0), yaw_deg=start.get("yaw_deg", mission.get("yaw_deg", 0.0)))
        if start.get("altitude", 0.0) > 0:
            ap.phase = "mission" if ap.wps else "hold"
            ap.home = np.array([x0[0], x0[1], 0.0])
            ap.pos_sp = x0[0:3].copy()
        dt = sc.get("dt", 0.005)
    else:
        alt0 = start.get("altitude", 100.0)
        atm0 = env.atmosphere.at(env.ground_elevation + alt0)
        V = mission.get("airspeed", vcfg.get("cruise_speed", 17.0))
        tas = V / math.sqrt(atm0.density / 1.225)   # prędkość zadana to IAS
        model = tunnel.fixed_wing_controller_model(veh, tas, atm0.density)
        tr = model["trim"]
        info["trim"] = tr
        if not tr.feasible:
            info["warning"] = f"brak trymu dla {V} m/s (przepustnica {tr.throttle:.2f})"
        ap = FixedWingAutopilot(model, mission, sc.get("gains"))
        heading = start.get("heading_deg")
        if heading is None and ap.wps:
            wp = ap.wps[0]
            heading = math.degrees(math.atan2(wp[1] - start.get("east", 0.0), wp[0] - start.get("north", 0.0)))
        wind0 = env.wind.profile.mean_wind(alt0)
        x0 = veh.initial_state(north=start.get("north", 0.0), east=start.get("east", 0.0), altitude=alt0,
                               yaw_deg=heading or 0.0, airspeed=tas,
                               trim=(tr.alpha, tr.elevator, min(tr.throttle, 1.0)), wind_ned=wind0)
        dt = sc.get("dt", 0.01)
    nav = NavigationSystem(sc.get("sensors"), rng, sc.get("navigation", "realistic"))
    failures = FailureManager(sc.get("failures", []))
    sim = Simulation(veh, env, ap, nav, failures, x0, dt=dt, duration=duration,
                     log_rate=sc.get("log_rate", 25.0), integrator=sc.get("integrator", "rk4"),
                     control_rate=sc.get("control_rate"), name=sc.get("name", ""))
    return sim, info


def run_scenario(path_or_dict, seed: int | None = None, overrides: dict | None = None,
                 progress: bool = False) -> ScenarioResult:
    sc, _ = load_scenario(path_or_dict, overrides)
    sim, info = build_simulation(sc, seed)
    log = sim.run(progress=progress)
    if "warning" in info:
        log.events.insert(0, (0.0, "UWAGA: " + info["warning"]))
    metrics = compute_metrics(log, sim, sc)
    crit = evaluate_criteria(metrics, sc.get("criteria", {"no_crash": True}))
    return ScenarioResult(sc.get("name", "scenariusz"), sc, log, metrics, crit, info["seed"], info)
