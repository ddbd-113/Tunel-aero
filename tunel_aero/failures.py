"""Wstrzykiwanie awarii i zakłóceń w trakcie lotu.

Typy (pole `type` w scenariuszu):
  motor               - awaria silnika: {motor: 1, efficiency: 0.0, detection_delay: 0.5}
  propeller_damage    - uszkodzone śmigło: {motor: 2, thrust_factor: 0.7}
  control_surface     - zablokowany ster: {surface: aileron|elevator|rudder, position_deg: 5,
                        detection_delay: 1.0 (opcjonalnie - autopilot przełącza się na ster kierunku)}
  gps_loss            - utrata sygnału GNSS (np. zagłuszanie, kanion miejski)
  gps_degraded        - pogorszona dokładność: {sigma_factor: 5}
  gps_spoofing        - fałszywy sygnał przesuwający pozycję: {drift_north: 0.5, drift_east: 0.0} [m/s]
  mag_interference    - zakłócenie magnetometru (linie energetyczne, stal): {heading_error_deg: 25}
  pitot_blocked       - zatkana rurka Pitota (owad, lód, woda)
  battery_cell        - uszkodzone ogniwo: {capacity_factor: 0.7, resistance_factor: 2.0}
  link_loss           - utrata łącza RC/telemetrii (wyzwala failsafe)
  imu_vibration       - zwiększone drgania/szum IMU: {factor: 5}

Każda awaria ma `t` (czas startu [s]) i opcjonalnie `duration` [s].
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

SURFACE_INDEX = {"aileron": 0, "elevator": 1, "rudder": 2}


@dataclass
class FailureEvent:
    type: str
    t: float
    duration: float | None = None
    params: dict = field(default_factory=dict)
    active: bool = False
    done: bool = False

    def describe(self) -> str:
        p = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.type}({p})" if p else self.type


class FailureManager:
    def __init__(self, events: list[dict] | None = None):
        self.events: list[FailureEvent] = []
        for e in events or []:
            e = dict(e)
            typ = e.pop("type")
            t = float(e.pop("t", 0.0))
            dur = e.pop("duration", None)
            self.events.append(FailureEvent(typ, t, dur, e))
        self.log: list[tuple[float, str]] = []
        self._pending_detect: list[tuple[float, int]] = []

    def update(self, t: float, vehicle, nav, autopilot) -> None:
        for ev in self.events:
            if not ev.active and not ev.done and t >= ev.t:
                ev.active = True
                self._apply(ev, t, vehicle, nav, autopilot, on=True)
                self.log.append((t, f"AWARIA: {ev.describe()}"))
            if ev.active and ev.duration is not None and t >= ev.t + ev.duration:
                ev.active = False
                ev.done = True
                self._apply(ev, t, vehicle, nav, autopilot, on=False)
                self.log.append((t, f"koniec zakłócenia: {ev.type}"))
        for item in list(self._pending_detect):
            if t >= item[0]:
                self._pending_detect.remove(item)
                what = item[1]
                if isinstance(what, tuple):
                    if hasattr(autopilot, "notify_surface_failure"):
                        autopilot.notify_surface_failure(what[1], t)
                elif hasattr(autopilot, "notify_motor_failure"):
                    autopilot.notify_motor_failure(what, t)

    def _apply(self, ev: FailureEvent, t, vehicle, nav, autopilot, on: bool) -> None:
        p = ev.params
        f = nav.faults
        typ = ev.type
        if typ in ("motor", "propeller_damage"):
            i = int(p.get("motor", 1)) - 1
            if on:
                vehicle.props.health[i] = p.get("efficiency", 0.0) if typ == "motor" else p.get("thrust_factor", 0.7)
                if typ == "motor" and p.get("detection_delay", 0.5) is not None and vehicle.props.health[i] < 0.3:
                    self._pending_detect.append((t + p.get("detection_delay", 0.5), i))
            else:
                vehicle.props.health[i] = 1.0
        elif typ == "control_surface":
            idx = SURFACE_INDEX[p.get("surface", "aileron")]
            if on:
                vehicle.stuck[idx] = math.radians(p.get("position_deg", math.degrees(vehicle.delta[idx])))
                if p.get("detection_delay") is not None:
                    self._pending_detect.append((t + p["detection_delay"], ("surface", idx)))
            else:
                vehicle.stuck.pop(idx, None)
        elif typ == "gps_loss":
            f.gps_lost = on
        elif typ == "gps_degraded":
            f.gps_sigma_factor = p.get("sigma_factor", 5.0) if on else 1.0
        elif typ == "gps_spoofing":
            f.gps_spoof_rate = np.array([p.get("drift_north", 0.5), p.get("drift_east", 0.0)]) if on else np.zeros(2)
        elif typ == "mag_interference":
            f.mag_offset_deg = p.get("heading_error_deg", 20.0) if on else 0.0
        elif typ == "pitot_blocked":
            f.pitot_blocked = on
        elif typ == "battery_cell":
            b = vehicle.battery
            b.capacity_factor_fault = p.get("capacity_factor", 0.7) if on else 1.0
            b.resistance_factor_fault = p.get("resistance_factor", 2.0) if on else 1.0
        elif typ == "link_loss":
            f.link_lost = on
        elif typ == "imu_vibration":
            f.imu_noise_factor = p.get("factor", 5.0) if on else 1.0
        else:
            raise ValueError(f"Nieznany typ awarii: {typ}")
