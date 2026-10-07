"""Metryki lotu i kryteria zaliczenia testu (PASS/FAIL)."""
from __future__ import annotations

import math

import numpy as np

AIRBORNE_PHASES = ("mission", "rtl", "loiter", "hold", "land")


def compute_metrics(log, sim=None, scenario: dict | None = None) -> dict:
    d = log.data
    meta = log.meta
    t = d["t"]
    phase = d["phase"]
    kind = meta.get("vehicle_kind", "multirotor")
    in_mission = np.array([p == "mission" for p in phase])
    airborne = np.array([p in AIRBORNE_PHASES for p in phase]) & (d["alt"] > 0.5)
    tilt = np.degrees(np.arccos(np.clip(np.cos(np.radians(d["roll"])) * np.cos(np.radians(d["pitch"])), -1, 1)))
    dist = float(np.sum(np.hypot(np.diff(d["pn"]), np.diff(d["pe"])))) if len(t) > 1 else 0.0
    nav_err = np.hypot(d["est_pn"] - d["pn"], d["est_pe"] - d["pe"])
    alt_err_baro = d["est_alt"] - (-d["pd"])
    flying = d["power"] > 1.0
    m = {
        "flight_time": float(t[-1]) if len(t) else 0.0,
        "crashed": bool(meta.get("crashed")),
        "crash_reason": meta.get("crash_reason", ""),
        "mission_complete": bool(meta.get("mission_complete")),
        "end_reason": meta.get("end_reason", ""),
        "distance_km": dist / 1000.0,
        "max_track_error": float(np.max(d["track_error"][in_mission])) if in_mission.any() else 0.0,
        "mean_track_error": float(np.mean(d["track_error"][in_mission])) if in_mission.any() else 0.0,
        "p95_track_error": float(np.percentile(d["track_error"][in_mission], 95)) if in_mission.any() else 0.0,
        "max_tilt_deg": float(np.max(tilt[airborne])) if airborne.any() else float(np.max(tilt)),
        "min_altitude": float(np.min(d["alt"][in_mission])) if in_mission.any() else float(np.min(d["alt"])),
        "max_altitude": float(np.max(d["alt"])),
        "energy_wh": float(d["energy_wh"][-1]),
        "final_soc": float(d["batt_soc"][-1]),
        "min_voltage": float(np.min(d["batt_v"][flying])) if flying.any() else float(d["batt_v"][-1]),
        "max_current": float(np.max(d["batt_i"])),
        "max_battery_temp": float(np.max(d["batt_temp"])),
        "min_battery_temp": float(np.min(d["batt_temp"])),
        "max_wind": float(np.max(d["wind_speed"])),
        "max_turbulence": float(np.max(d["turb"])),
        "max_updraft": float(np.max(d["updraft"])),
        "max_downdraft": float(-np.min(d["updraft"])),
        "max_ice": float(np.max(d["ice"])),
        "max_nav_error": float(np.max(nav_err)),
        "max_baro_error": float(np.max(np.abs(alt_err_baro))),
        "min_temp_c": float(np.min(d["temp_c"])),
        "max_density_alt": float(np.max(d["density_alt"])),
        "events": len(log.events),
    }
    dtl = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    if kind == "multirotor":
        m["max_throttle"] = float(np.max(d["throttle_max"]))
        m["saturation_time"] = float(np.sum((d["throttle_max"] > 0.98) | (d["saturated"] > 0.5)) * dtl)
        m["max_vrs"] = float(np.max(d["vrs"]))
        m["mean_hover_power"] = float(np.median(d["power"][airborne])) if airborne.any() else 0.0
        if scenario:
            mis = scenario.get("mission", {}) or {}
            wps = mis.get("waypoints") or []
            if mis.get("return_home"):
                target = (0.0, 0.0)
            elif wps:
                w = wps[-1]
                target = (w["north"], w["east"]) if isinstance(w, dict) else (w[0], w[1])
            else:
                target = (0.0, 0.0)
            m["landing_error"] = float(math.hypot(d["pn"][-1] - target[0], d["pe"][-1] - target[1]))
    else:
        m["min_airspeed"] = float(np.min(d["ias"][airborne])) if airborne.any() else float(np.min(d["ias"]))
        m["max_airspeed"] = float(np.max(d["ias"]))
        m["stall_time"] = float(np.sum(d["stall"] > 0.5) * dtl)
        m["max_load_factor"] = float(np.max(np.abs(d["load_factor"])))
        m["max_alpha"] = float(np.max(d["alpha"]))
        h_err = np.abs(d["alt"] - d["h_sp"])
        m["max_altitude_error"] = float(np.max(h_err[in_mission])) if in_mission.any() else float(np.max(h_err))
    return m


# nazwa kryterium -> (metryka, porównanie)
CRITERIA = {
    "no_crash": ("crashed", "bool_not"),
    "mission_complete": ("mission_complete", "bool"),
    "max_track_error": ("max_track_error", "le"),
    "p95_track_error": ("p95_track_error", "le"),
    "mean_track_error": ("mean_track_error", "le"),
    "max_tilt_deg": ("max_tilt_deg", "le"),
    "min_altitude": ("min_altitude", "ge"),
    "min_battery_soc": ("final_soc", "ge"),
    "max_energy_wh": ("energy_wh", "le"),
    "min_voltage": ("min_voltage", "ge"),
    "max_battery_temp": ("max_battery_temp", "le"),
    "max_landing_error": ("landing_error", "le"),
    "max_nav_error": ("max_nav_error", "le"),
    "max_saturation_time": ("saturation_time", "le"),
    "min_airspeed": ("min_airspeed", "ge"),
    "max_airspeed": ("max_airspeed", "le"),
    "max_stall_time": ("stall_time", "le"),
    "max_load_factor": ("max_load_factor", "le"),
    "max_altitude_error": ("max_altitude_error", "le"),
    "min_flight_time": ("flight_time", "ge"),
    "max_ice": ("max_ice", "le"),
}

LABELS = {
    "no_crash": "brak katastrofy", "mission_complete": "misja ukończona",
    "max_track_error": "maks. odchyłka od trasy [m]", "p95_track_error": "odchyłka od trasy (95 percentyl) [m]",
    "mean_track_error": "średnia odchyłka od trasy [m]", "max_tilt_deg": "maks. pochylenie [deg]",
    "min_altitude": "min. wysokość w misji [m]", "min_battery_soc": "bateria na końcu [0-1]",
    "max_energy_wh": "zużyta energia [Wh]", "min_voltage": "min. napięcie pod obciążeniem [V]",
    "max_battery_temp": "maks. temperatura baterii [C]", "max_landing_error": "błąd miejsca lądowania [m]",
    "max_nav_error": "maks. błąd nawigacji [m]", "max_saturation_time": "czas nasycenia silników [s]",
    "min_airspeed": "min. prędkość IAS [m/s]", "max_airspeed": "maks. prędkość IAS [m/s]",
    "max_stall_time": "czas w przeciągnięciu [s]", "max_load_factor": "maks. przeciążenie [g]",
    "max_altitude_error": "maks. błąd wysokości [m]", "min_flight_time": "czas lotu [s]",
    "max_ice": "maks. oblodzenie [0-1]",
}


def evaluate_criteria(metrics: dict, criteria: dict) -> list[dict]:
    out = []
    for name, thr in (criteria or {}).items():
        if name not in CRITERIA:
            raise ValueError(f"Nieznane kryterium '{name}'. Dostępne: {', '.join(CRITERIA)}")
        key, op = CRITERIA[name]
        val = metrics.get(key)
        if val is None:
            out.append({"name": name, "label": LABELS.get(name, name), "value": None, "threshold": thr,
                        "passed": False})
            continue
        if op == "bool_not":
            val = not val
            ok = val if thr else True
        elif op == "bool":
            ok = bool(val) if thr else True
        elif op == "le":
            ok = val <= thr
        else:
            ok = val >= thr
        out.append({"name": name, "label": LABELS.get(name, name), "value": val, "threshold": thr,
                    "passed": bool(ok)})
    return out
