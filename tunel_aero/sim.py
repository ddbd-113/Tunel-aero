"""Pętla symulacji: środowisko -> awarie -> nawigacja -> autopilot -> aktuatory -> dynamika -> log."""
from __future__ import annotations

import csv
import math
import time
from collections import defaultdict

import numpy as np

from .math3d import euler_from_quat, quat_normalize, quat_to_dcm


class FlightLog:
    def __init__(self):
        self._d: dict[str, list] = defaultdict(list)
        self.data: dict[str, np.ndarray] = {}
        self.events: list[tuple[float, str]] = []
        self.meta: dict = {}

    def append(self, row: dict) -> None:
        for k, v in row.items():
            self._d[k].append(v)

    def finalize(self) -> "FlightLog":
        for k, v in self._d.items():
            self.data[k] = np.array(v) if not isinstance(v[0], str) else np.array(v, dtype=object)
        self.events.sort(key=lambda e: e[0])
        return self

    def __getitem__(self, k):
        return self.data[k]

    def __contains__(self, k):
        return k in self.data

    def to_csv(self, path) -> None:
        keys = list(self.data.keys())
        n = len(self.data["t"])
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(keys)
            for i in range(n):
                w.writerow([(f"{self.data[k][i]:.5g}" if not isinstance(self.data[k][i], str) else self.data[k][i])
                            for k in keys])


class Simulation:
    def __init__(self, vehicle, environment, autopilot, navigation, failures, x0: np.ndarray,
                 dt: float = 0.005, duration: float = 120.0, log_rate: float = 50.0,
                 integrator: str = "rk4", control_rate: float | None = None, name: str = ""):
        self.vehicle = vehicle
        self.env = environment
        self.ap = autopilot
        self.nav = navigation
        self.failures = failures
        self.x = np.array(x0, dtype=float)
        self.dt = dt
        self.duration = duration
        self.log_every = max(1, int(round(1.0 / (log_rate * dt))))
        self.integrator = integrator
        self.ctrl_every = 1 if not control_rate else max(1, int(round(1.0 / (control_rate * dt))))
        self.name = name
        self.t = 0.0
        self.step_i = 0
        self.cmd = None
        self.log = FlightLog()
        self.end_reason = ""
        self._flags: dict[str, bool] = {}
        self._sat_time = 0.0
        self._last_wind = np.zeros(3)
        self.nav_state = None
        self.env_sample = None
        self.external_cmd = None   # sterowanie z zewnątrz (API gym)

    # ------------------------------------------------------------------
    def _event(self, msg: str) -> None:
        self.log.events.append((round(self.t, 3), msg))

    def _flag(self, key: str, cond: bool, msg_on: str, msg_off: str | None = None) -> None:
        prev = self._flags.get(key, False)
        if cond and not prev:
            self._event(msg_on)
        elif not cond and prev and msg_off:
            self._event(msg_off)
        self._flags[key] = cond

    def step(self) -> bool:
        """Jeden krok symulacji. Zwraca False, gdy symulacja się zakończyła."""
        veh, dt, x = self.vehicle, self.dt, self.x
        airspeed = float(np.linalg.norm(x[3:6] - self._last_wind))
        es = self.env.sample(self.t, dt, x[0:3], airspeed)
        self.env_sample = es
        self._last_wind = es.wind.total
        self.failures.update(self.t, veh, self.nav, self.ap)
        nav = self.nav.update(self.t, dt, x, veh, es)
        self.nav_state = nav
        status = {"battery_soc": veh.battery.soc, "link_lost": self.nav.faults.link_lost,
                  "on_ground": veh.on_ground}
        if self.external_cmd is not None:
            self.cmd = self.external_cmd
        elif self.step_i % self.ctrl_every == 0 or self.cmd is None:
            self.cmd = self.ap.update(self.t, dt * self.ctrl_every, nav, status)
        veh.step_actuators(dt, self.cmd, es)

        k1 = veh.dynamics(x, es, record=True)
        if self.integrator == "rk4":
            k2 = veh.dynamics(x + 0.5 * dt * k1, es)
            k3 = veh.dynamics(x + 0.5 * dt * k2, es)
            k4 = veh.dynamics(x + dt * k3, es)
            xn = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        else:  # półjawny Euler
            xn = x.copy()
            xn[3:6] = x[3:6] + dt * k1[3:6]
            xn[0:3] = x[0:3] + dt * xn[3:6]
            xn[10:13] = x[10:13] + dt * k1[10:13]
            xn[6:10] = x[6:10] + dt * k1[6:10]
        xn[6:10] = quat_normalize(xn[6:10])
        self.x = xn

        veh.battery.update(dt, veh.aux.get("power", 0.0), es.atm.temperature_c)
        self.env.weather.update(dt, es.h_agl, es.atm.temperature_c, veh.icing_exposure_speed(airspeed))
        self._monitor(es)
        if self.step_i % self.log_every == 0:
            self._log(es, nav)
        self.t += dt
        self.step_i += 1

        if veh.crashed:
            self.end_reason = f"KATASTROFA: {veh.crash_reason}"
            self._event(self.end_reason)
            self._log(es, nav)
            return False
        if not np.all(np.isfinite(self.x)):
            self.end_reason = "błąd numeryczny (NaN)"
            veh.crashed = True
            veh.crash_reason = self.end_reason
            return False
        if getattr(self.ap, "phase", "") == "landed":
            self._landed_t = getattr(self, "_landed_t", self.t)
            if self.t - self._landed_t > 2.0:
                self.end_reason = "wylądowano"
                return False
        if self.t >= self.duration - 1e-9:
            self.end_reason = "koniec czasu symulacji"
            return False
        return True

    def run(self, progress: bool = False, callback=None) -> FlightLog:
        """progress - pasek w konsoli; callback(ułamek 0..1) - np. pasek postępu w przeglądarce."""
        wall = time.time()
        n_total = int(self.duration / self.dt)
        every = max(1, n_total // 100)
        last_pct = -1
        while self.step():
            if callback is not None and self.step_i % every == 0:
                callback(min(0.99, self.step_i / n_total))
            if progress:
                pct = int(100 * self.step_i / n_total)
                if pct // 10 != last_pct // 10:
                    print(f"\r  symulacja {self.name}: {pct:3d}%", end="", flush=True)
                    last_pct = pct
        if progress:
            print(f"\r  symulacja {self.name}: 100% ({time.time() - wall:.1f} s)")
        self.log.events.extend(self.ap.events)
        self.log.events.extend(self.failures.log)
        self.log.meta.update(end_reason=self.end_reason, crashed=self.vehicle.crashed,
                             crash_reason=self.vehicle.crash_reason, sim_time=self.t,
                             wall_time=time.time() - wall, vehicle=self.vehicle.name,
                             vehicle_kind=self.vehicle.kind,
                             mission_complete=bool(getattr(self.ap, "mission_complete", False)))
        return self.log.finalize()

    # ------------------------------------------------------------------
    def _monitor(self, es) -> None:
        veh, aux = self.vehicle, self.vehicle.aux
        if veh.kind == "fixed_wing":
            st = aux.get("stall", 0)
            self._flag("stall", (st > 0.5 or (self._flags.get("stall", False) and st > 0.3)) and es.h_agl > 0.5,
                       f"PRZECIĄGNIĘCIE (alfa={math.degrees(aux.get('alpha', 0)):.1f} deg)", "wyjście z przeciągnięcia")
            self._flag("vne", aux.get("airspeed", 0) > veh.vne, f"przekroczona prędkość VNE {veh.vne:.0f} m/s")
            nz = aux.get("load_factor", 1.0)
            self._flag("overload", abs(nz) > veh.n_max, f"przekroczone przeciążenie konstrukcji n={nz:.1f}")
            self._flag("ice", es.deg.ice > 0.3, "znaczne oblodzenie płatowca (>30%)")
            self._flag("pitot", self.nav.pitot_iced or self.nav.faults.pitot_blocked,
                       "zatkana/zamarznięta rurka Pitota - błędne wskazanie prędkości")
        else:
            vrs = aux.get("vrs", 0)
            self._flag("vrs", vrs > 0.4 or (self._flags.get("vrs", False) and vrs > 0.15),
                       "stan pierścienia wirowego (VRS) - utrata ciągu", "wyjście z VRS")
            sat = (float(np.max(veh.throttle)) > 0.98 or getattr(self.ap, "saturated", False)) and not veh.on_ground
            self._sat_time = self._sat_time + self.dt if sat else 0.0
            self._flag("sat", self._sat_time > 1.0, "nasycenie silników - brak zapasu ciągu", "zapas ciągu odzyskany")
            self._flag("ice", es.deg.ice > 0.3, "znaczne oblodzenie śmigieł (>30%)")
        b = veh.battery
        self._flag("batt_low", b.soc < 0.15, f"bateria poniżej 15% (U={b.voltage:.1f} V)")
        self._flag("batt_dead", b.depleted, "BATERIA WYCZERPANA - utrata zasilania")
        self._flag("gps", not self.nav.raw.get("gps_valid", True), "brak sygnału GPS", "GPS odzyskany")

    def _log(self, es, nav) -> None:
        x, veh = self.x, self.vehicle
        q = x[6:10]
        roll, pitch, yaw = euler_from_quat(q)
        R = quat_to_dcm(q)
        v_air_b = R.T @ (x[3:6] - es.wind.total)
        tas = float(np.linalg.norm(v_air_b))
        w = es.wind
        b = veh.battery
        row = {
            "t": self.t, "pn": x[0], "pe": x[1], "pd": x[2], "alt": es.h_agl, "alt_msl": es.alt_msl,
            "vn": x[3], "ve": x[4], "vd": x[5],
            "roll": math.degrees(roll), "pitch": math.degrees(pitch), "yaw": math.degrees(yaw),
            "p": math.degrees(x[10]), "q": math.degrees(x[11]), "r": math.degrees(x[12]),
            "qw": q[0], "qx": q[1], "qy": q[2], "qz": q[3],
            "tas": tas, "ias": tas * math.sqrt(es.atm.density / 1.225), "gs": math.hypot(x[3], x[4]),
            "alpha": math.degrees(math.atan2(v_air_b[2], v_air_b[0])) if tas > 0.5 else 0.0,
            "load_factor": veh.aux.get("load_factor", 1.0),
            "wind_n": w.total[0], "wind_e": w.total[1], "wind_d": w.total[2],
            "wind_speed": math.hypot(w.total[0], w.total[1]),
            "wind_mean": math.hypot(w.mean[0], w.mean[1]),
            "turb": float(np.linalg.norm(w.turbulence)), "gust": float(np.linalg.norm(w.gust)),
            "updraft": -w.convective[2], "downburst_h": math.hypot(w.convective[0], w.convective[1]),
            "temp_c": es.atm.temperature_c, "pressure_hpa": es.atm.pressure / 100, "rho": es.atm.density,
            "density_alt": es.atm.density_altitude, "rain": self.env.weather.rain_rate, "ice": es.deg.ice,
            "batt_v": b.voltage, "batt_i": b.current, "batt_soc": b.soc, "batt_temp": b.temp_c,
            "power": veh.aux.get("power", 0.0), "energy_wh": b.energy_used_wh,
            "thrust": veh.aux.get("thrust_total", 0.0),
            "est_pn": nav.pos[0], "est_pe": nav.pos[1], "est_alt": nav.alt, "est_ias": nav.airspeed,
            "gps_ok": 1.0 if self.nav.raw.get("gps_valid", True) else 0.0,
            "sp_n": float(getattr(self.ap, "pos_sp", np.zeros(3))[0]) if veh.kind == "multirotor" else 0.0,
            "sp_e": float(getattr(self.ap, "pos_sp", np.zeros(3))[1]) if veh.kind == "multirotor" else 0.0,
            "track_error": self.ap.true_track_error(x[0:3]) if hasattr(self.ap, "true_track_error") else 0.0,
            "track_error_est": getattr(self.ap, "track_error", 0.0),
            "phase": getattr(self.ap, "phase", ""),
        }
        if veh.kind == "multirotor":
            row["throttle"] = float(np.mean(veh.throttle))
            row["throttle_max"] = float(np.max(veh.throttle))
            row["vrs"] = veh.aux.get("vrs", 0.0)
            row["saturated"] = 1.0 if getattr(self.ap, "saturated", False) else 0.0
            for i, th in enumerate(veh.throttle):
                row[f"m{i + 1}"] = float(th)
        else:
            row["throttle"] = veh.throttle
            row["aileron"] = math.degrees(veh.delta[0])
            row["elevator"] = math.degrees(veh.delta[1])
            row["rudder"] = math.degrees(veh.delta[2])
            row["stall"] = veh.aux.get("stall", 0.0)
            row["h_sp"] = getattr(self.ap, "h_sp", 0.0)
            row["v_sp"] = getattr(self.ap, "v_sp", 0.0)
            row["roll_sp"] = math.degrees(getattr(self.ap, "roll_sp", 0.0))
            row["pitch_sp"] = math.degrees(getattr(self.ap, "pitch_sp", 0.0))
        self.log.append(row)
