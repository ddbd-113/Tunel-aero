"""API dla aplikacji webowej (Pyodide w przeglądarce). Wszystkie funkcje zwracają JSON (str)."""
from __future__ import annotations

import io
import json
import math
from pathlib import Path

import numpy as np
import yaml

from .scenario import PACKAGE_ROOT, load_scenario, load_yaml

_LAST = {}


def _default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "__dict__"):
        return {k: v for k, v in o.__dict__.items() if not k.startswith("_")}
    return str(o)


def _json(obj) -> str:
    return json.dumps(obj, default=_default, ensure_ascii=False, allow_nan=False)


def _files(files) -> dict | None:
    if files is None:
        return None
    if hasattr(files, "to_py"):
        files = files.to_py()
    out = {}
    for k, v in dict(files).items():
        out[str(k)] = v.to_bytes() if hasattr(v, "to_bytes") else bytes(v)
    return out


# ------------------------------------------------------------------ katalog
def catalog() -> str:
    scen = []
    for f in sorted((PACKAGE_ROOT / "scenarios").rglob("*.yaml")):
        if "montecarlo" in f.parts:
            continue
        sc = load_yaml(f)
        v = sc.get("vehicle")
        kind = "multirotor"
        if isinstance(v, str):
            kind = load_yaml((f.parent / v).resolve()).get("type", "multirotor")
        scen.append({"path": str(f.relative_to(PACKAGE_ROOT)), "name": sc.get("name", f.stem),
                     "description": " ".join(str(sc.get("description", "")).split()),
                     "kind": "fixed_wing" if kind in ("fixed_wing", "cad_aircraft") else kind,
                     "expect_pass": sc.get("expect_pass", True)})
    veh = []
    for f in sorted((PACKAGE_ROOT / "vehicles").glob("*.yaml")):
        c = load_yaml(f)
        veh.append({"path": str(f.relative_to(PACKAGE_ROOT)), "name": c.get("name", f.stem), "kind": c.get("type")})
    ex = PACKAGE_ROOT / "examples" / "moj_samolot"
    example = {"yaml": (ex / "aircraft.yaml").read_text(encoding="utf-8") if (ex / "aircraft.yaml").exists() else "",
               "files": sorted(p.name for p in ex.glob("*.stl"))}
    return _json({"scenarios": scen, "vehicles": veh, "example": example})


def scenario_info(path: str) -> str:
    sc, _ = load_scenario(PACKAGE_ROOT / path)
    env = sc.get("environment", {}) or {}
    wind = env.get("wind", {}) or {}
    ic = env.get("icing", {}) or {}
    return _json({
        "name": sc.get("name"), "description": " ".join(str(sc.get("description", "")).split()),
        "duration": sc.get("duration", 120), "seed": sc.get("seed", 0),
        "ground_elevation": env.get("ground_elevation", 0), "temperature_c": env.get("temperature_c", 15),
        "humidity": env.get("humidity", 0.5), "rain_mm_h": env.get("rain_mm_h", 0),
        "wind_speed": wind.get("speed", 0), "wind_direction": wind.get("direction", 0),
        "terrain": wind.get("terrain", "open"), "turbulence": wind.get("turbulence", "auto" if wind.get("speed") else "none"),
        "icing_lwc": ic.get("lwc", 0), "icing_base": ic.get("cloud_base", 0), "icing_top": ic.get("cloud_top", 0),
        "failures": sc.get("failures", []) or [], "criteria": sc.get("criteria", {}),
        "navigation": sc.get("navigation", "realistic"),
        "vehicle_name": (sc.get("vehicle") or {}).get("name", ""),
    })


# ------------------------------------------------------------------ YAML
def yaml_load(text: str) -> str:
    return _json(yaml.safe_load(text) or {})


def yaml_dump(obj_json: str) -> str:
    return yaml.safe_dump(json.loads(obj_json), allow_unicode=True, sort_keys=False)


# ------------------------------------------------------------------ samolot z CAD
def analyze_aircraft(cfg_json: str, files=None) -> str:
    from .cad.aircraft import build_aircraft, vehicle_to_yaml_dict
    cfg = json.loads(cfg_json)
    cfg["type"] = "cad_aircraft"
    vehicle, d = build_aircraft(cfg, base_dir=None, files=_files(files))
    vis = vehicle.get("_visual")
    tr = d.get("trim")
    res = {
        "name": d["name"], "checks": d["checks"],
        "summary": {
            "mass": d["mass"], "span_mm": d["b"] * 1000, "area_dm2": d["S"] * 100, "mac_mm": d["mac"] * 1000,
            "AR": d["AR"], "wing_loading": d["wing_loading_g_dm2"], "static_margin": d["static_margin"] * 100,
            "cg_pct_mac": d["cg_pct_mac"] * 100, "np_pct_mac": d["np_pct_mac"] * 100, "v_stall": d["v_stall"],
            "cruise": d["cruise_speed"], "CLmax": d["CLmax"], "CD0": d["CD0"], "Vh": d["Vh"], "Vv": d["Vv"],
            "tw": d.get("static_thrust_to_weight", 0.0),
            "trim_alpha": math.degrees(tr.alpha) if tr else None, "trim_elevator": math.degrees(tr.elevator) if tr else None,
            "trim_throttle": tr.throttle * 100 if tr else None, "trim_power": tr.power if tr else None,
        },
        "cg_cad": d["cg_cad"], "np_cad": d["np_cad"], "cg_range_cad": d["cg_range_cad"], "units": d["units"],
        "inertia": d["inertia"], "derivatives": {k: float(v) for k, v in d["derivatives"].items()},
        "cd_parts": d["cd_parts"],
        "surfaces": {r: {k: v for k, v in s.items() if k != "sections"} for r, s in d["surfaces"].items()},
        "mesh": {"v": np.round(vis["v"].reshape(-1), 4), "f": vis["f"].reshape(-1)} if vis is not None else None,
        "np_body": float(d["derivatives"]["Cma"] / d["derivatives"]["CLa"] * d["mac"]),
        "mac": d["mac"],
        "vehicle_yaml": yaml.safe_dump(vehicle_to_yaml_dict(vehicle), allow_unicode=True, sort_keys=False),
    }
    return _json(res)


# ------------------------------------------------------------------ symulacja
def run_test(scenario_path: str, overrides_json: str = "{}", vehicle_json: str = "", files=None,
             progress=None) -> str:
    from .metrics import compute_metrics, evaluate_criteria
    from .scenario import build_simulation
    from .viewer import replay_html
    overrides = json.loads(overrides_json or "{}")
    if vehicle_json:
        v = json.loads(vehicle_json)
        if "path" in v:                      # gotowy pojazd z katalogu
            v = str(PACKAGE_ROOT / v["path"])
        else:                                # samolot z CAD (pliki z przeglądarki)
            v["type"] = "cad_aircraft"
            v["_files"] = _files(files)
        overrides["vehicle"] = v
        overrides["vehicle_overrides"] = {}
        overrides["mission.airspeed"] = None
    sc, _ = load_scenario(PACKAGE_ROOT / scenario_path, overrides)
    sim, info = build_simulation(sc)
    log = sim.run(callback=progress)
    if "warning" in info:
        log.events.insert(0, (0.0, "UWAGA: " + info["warning"]))
    metrics = compute_metrics(log, sim, sc)
    crit = evaluate_criteria(metrics, sc.get("criteria", {"no_crash": True}))

    from .scenario import ScenarioResult
    result = ScenarioResult(sc.get("name", "scenariusz"), sc, log, metrics, crit, info["seed"], info)
    d = log.data
    step = max(1, len(d["t"]) // 1200)
    keys = ["t", "alt", "est_alt", "tas", "est_ias", "gs", "wind_speed", "roll", "pitch", "batt_soc", "batt_v",
            "throttle", "power", "temp_c", "ice", "turb", "updraft", "h_sp", "v_sp", "track_error"]
    series = {k: np.round(d[k][::step].astype(float), 3) for k in keys if k in d}
    buf = io.StringIO()
    cols = [k for k in d if k != "phase"]
    buf.write(",".join(cols + ["phase"]) + "\n")
    for i in range(len(d["t"])):
        buf.write(",".join(f"{d[k][i]:.5g}" for k in cols) + f",{d['phase'][i]}\n")
    _LAST["csv"] = buf.getvalue()
    _LAST["scenario_yaml"] = yaml.safe_dump(_strip(sc), allow_unicode=True, sort_keys=False)
    if progress is not None:
        progress(1.0)
    return _json({
        "name": result.name, "passed": result.passed, "expect_pass": sc.get("expect_pass", True),
        "end_reason": log.meta.get("end_reason", ""), "vehicle": log.meta.get("vehicle", ""),
        "kind": log.meta.get("vehicle_kind"), "criteria": crit,
        "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float, bool, str))},
        "events": log.events, "series": series, "replay_html": replay_html(result),
    })


def last_csv() -> str:
    return _LAST.get("csv", "")


def last_scenario_yaml() -> str:
    return _LAST.get("scenario_yaml", "")


def _strip(o):
    if isinstance(o, dict):
        return {k: _strip(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_strip(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o
