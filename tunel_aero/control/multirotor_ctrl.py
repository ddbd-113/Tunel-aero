"""Autopilot wielowirnikowca w stylu PX4/ArduPilot:
pozycja -> prędkość -> przyspieszenie/wektor ciągu -> orientacja -> prędkości kątowe -> mikser.

Fazy misji: idle -> takeoff -> mission -> (rtl) -> land -> landed
Failsafe: niski poziom baterii, utrata łącza (RTL), utrata GPS (lądowanie), awaria silnika
(po wykryciu realokacja ciągu na sprawne silniki).
"""
from __future__ import annotations

import math

import numpy as np

from ..math3d import GRAVITY, clamp, cross3, dcm_to_quat, quat_conj, quat_mul
from .pid import PID

DEFAULT_GAINS = {
    "pos_p": 0.95, "xtrack_p": 1.2,
    "vel_p": 2.2, "vel_i": 0.6, "vel_i_limit": 4.0,
    "velz_p": 4.0, "velz_i": 2.0, "velz_i_limit": 12.0,
    "att_p": [7.0, 7.0, 3.0],
    "rate_p": [18.0, 18.0, 9.0], "rate_i": [6.0, 6.0, 2.0], "rate_d": [0.25, 0.25, 0.0],
    "rate_i_limit": 25.0,
    "max_rate": [3.8, 3.8, 2.0],
}


class MultirotorAutopilot:
    def __init__(self, model: dict, mission: dict, gains: dict | None = None):
        self.m = model
        g = dict(DEFAULT_GAINS)
        g.update(gains or {})
        self.g = g
        self.mission = mission
        self.wps = [self._wp(w) for w in mission.get("waypoints", [])]
        self.takeoff_alt = mission.get("takeoff_altitude", 10.0)
        self.cruise = min(mission.get("cruise_speed", 6.0), model["max_speed"])
        self.acc_radius = mission.get("acceptance_radius", 1.5)
        self.land_at_end = mission.get("land", True)
        self.rtl_at_end = mission.get("return_home", False)
        self.rtl_alt = mission.get("rtl_altitude", 30.0)
        self.land_speed = mission.get("land_speed", 1.0)
        self.yaw_sp = math.radians(mission.get("yaw_deg", 0.0))
        self.yaw_to_wp = mission.get("yaw_to_waypoint", False)
        self.low_batt = mission.get("low_battery_soc", 0.2)
        self.low_batt_action = mission.get("low_battery_action", "rtl")
        self.gps_loss_action = mission.get("gps_loss_action", "continue")
        self.gps_loss_timeout = mission.get("gps_loss_timeout", 5.0)
        self.link_loss_action = mission.get("link_loss_action", "rtl")
        self.arm_time = mission.get("arm_delay", 1.0)

        self.phase = "idle"
        self.wp_i = 0
        self.seg_start = None
        self.hold_timer = 0.0
        self.landed_timer = 0.0
        self.land_point = None
        self.home = np.zeros(3)
        self.mission_complete = False
        self.events: list[tuple[float, str]] = []
        self.failed_motors: set[int] = set()
        self.gps_lost_time = 0.0
        self._failsafe = None

        self.vel_n = PID(g["vel_p"], g["vel_i"], i_limit=g["vel_i_limit"])
        self.vel_e = PID(g["vel_p"], g["vel_i"], i_limit=g["vel_i_limit"])
        self.vel_d = PID(g["velz_p"], g["velz_i"], i_limit=g["velz_i_limit"])
        self.rate = [PID(g["rate_p"][i], g["rate_i"][i], g["rate_d"][i], i_limit=g["rate_i_limit"])
                     for i in range(3)]
        self._build_allocation()

        self.pos_sp = np.zeros(3)
        self.vel_sp = np.zeros(3)
        self.track_error = 0.0
        self.thrust_sp = 0.0
        self.saturated = False

    @staticmethod
    def _wp(w):
        if isinstance(w, dict):
            return {"pos": np.array([w["north"], w["east"], -w["alt"]], float), "hold": w.get("hold", 0.0),
                    "speed": w.get("speed")}
        return {"pos": np.array([w[0], w[1], -w[2]], float), "hold": w[3] if len(w) > 3 else 0.0, "speed": None}

    # ---------------------------------------------------------- alokacja
    def _build_allocation(self):
        P = self.m["positions"]
        s = self.m["spins"]
        B = np.vstack([np.ones(len(P)), -P[:, 1], P[:, 0], s * self.m["k_q"]])
        for i in self.failed_motors:
            B[:, i] = 0.0
        self.B = B
        self.Bp = np.linalg.pinv(B)

    def notify_motor_failure(self, idx: int, t: float):
        if idx not in self.failed_motors:
            self.failed_motors.add(idx)
            self._build_allocation()
            self.events.append((t, f"autopilot wykrył awarię silnika {idx + 1} - realokacja ciągu"))
            if self.phase in ("mission", "takeoff", "rtl"):
                self._set_failsafe("land", t, "awaria napędu -> lądowanie awaryjne")

    def allocate(self, T: float, tau: np.ndarray) -> np.ndarray:
        tmax = self.m["t_max"]
        c0 = self.Bp[:, 0]
        t_rp = self.Bp[:, 1:3] @ tau[:2]
        t_y = self.Bp[:, 3] * tau[2]
        healthy = np.array([i not in self.failed_motors for i in range(len(c0))])
        # 1) roll/pitch muszą się zmieścić w zakresie
        span = np.ptp(t_rp[healthy]) if healthy.any() else 0.0
        if span > tmax:
            t_rp *= tmax / span
        t = c0 * T + t_rp
        # 2) przesunięcie kolektywne, aby zmieścić się w [0, tmax] (priorytet: orientacja)
        mask = healthy & (c0 > 1e-6)
        if mask.any():
            lo = np.max((0.0 - t[mask]) / c0[mask])
            hi = np.min((tmax - t[mask]) / c0[mask])
            if lo > 0:
                t = t + c0 * min(lo, max(hi, lo))
            elif hi < 0:
                t = t + c0 * hi
        # 3) odchylenie (yaw) z najniższym priorytetem
        s = 1.0
        for i in np.nonzero(healthy)[0]:
            if t_y[i] > 1e-9:
                s = min(s, max(0.0, (tmax - t[i]) / t_y[i]))
            elif t_y[i] < -1e-9:
                s = min(s, max(0.0, -t[i] / t_y[i]))
        t = np.clip(t + s * t_y, 0.0, tmax)
        # nasycenie: mikser nie jest w stanie zrealizować żądanego ciągu lub momentów
        self.saturated = bool(t.sum() < 0.97 * T or span > tmax or s < 0.5)
        return np.sqrt(t / tmax)

    # ------------------------------------------------------------- misja
    def _set_failsafe(self, action, t, reason):
        if self._failsafe == action or self.phase in ("landed", "idle"):
            return
        if self._failsafe == "land" and action == "rtl":
            return
        self._failsafe = action
        self.events.append((t, f"FAILSAFE: {reason}"))
        if action == "rtl":
            self.phase = "rtl"
            self.seg_start = None
        else:
            self.phase = "land"
            self.land_point = None

    def _check_failsafes(self, t, dt, nav, status):
        if status.get("link_lost") and self.phase in ("mission", "takeoff"):
            self._set_failsafe(self.link_loss_action, t, "utrata łącza z operatorem")
        soc = status.get("battery_soc", 1.0)
        if soc < self.low_batt and self.phase in ("mission", "takeoff"):
            self._set_failsafe(self.low_batt_action, t, f"niski poziom baterii ({soc * 100:.0f}%)")
        if not nav.gps_ok:
            self.gps_lost_time += dt
            if self.gps_loss_action == "land" and self.gps_lost_time > self.gps_loss_timeout \
                    and self.phase in ("mission", "rtl", "takeoff"):
                self._set_failsafe("land", t, "utrata GPS - lądowanie bez pozycji")
        else:
            self.gps_lost_time = 0.0

    def _guidance(self, t, dt, nav):
        p = nav.pos
        g = self.g
        vel_sp = np.zeros(3)
        self.track_error = 0.0
        if self.phase == "idle":
            if t >= self.arm_time:
                self.phase = "takeoff"
                self.home = p.copy()
                self.events.append((t, "uzbrojenie i start"))
            return None
        if self.phase == "takeoff":
            target = np.array([self.home[0], self.home[1], -self.takeoff_alt])
            vel_sp[:2] = g["pos_p"] * (target[:2] - p[:2])
            vel_sp[2] = g["pos_p"] * (target[2] - p[2])
            self.pos_sp = target
            if abs(target[2] - p[2]) < 0.5:
                self.phase = "mission" if self.wps else "land"
                self.seg_start = p.copy()
                self.events.append((t, f"osiągnięto wysokość startu {self.takeoff_alt:.0f} m"))
        elif self.phase in ("mission", "rtl"):
            if self.phase == "mission":
                wp = self.wps[self.wp_i]
                target, hold, speed = wp["pos"], wp["hold"], wp["speed"] or self.cruise
            else:
                alt = max(-p[2], self.rtl_alt)
                if self.seg_start is None:
                    self.seg_start = p.copy()
                    self.rtl_target = np.array([self.home[0], self.home[1], -alt])
                    self.events.append((t, "powrót do punktu startu (RTL)"))
                target, hold, speed = self.rtl_target, 0.0, self.cruise
            if self.seg_start is None:
                self.seg_start = p.copy()
            vel_sp, dist = self._track_segment(self.seg_start, target, p, speed)
            self.pos_sp = target
            if dist < self.acc_radius:
                self.hold_timer += dt
                if self.hold_timer >= hold:
                    self.hold_timer = 0.0
                    self.seg_start = target.copy()
                    if self.phase == "rtl":
                        self.phase = "land"
                        self.land_point = None
                    else:
                        self.events.append((t, f"osiągnięto punkt trasy {self.wp_i + 1}/{len(self.wps)}"))
                        self.wp_i += 1
                        if self.wp_i >= len(self.wps):
                            self.wp_i = len(self.wps) - 1
                            if self.rtl_at_end:
                                self.phase = "rtl"
                                self.seg_start = None
                            elif self.land_at_end:
                                self.phase = "land"
                                self.land_point = None
                            else:
                                self.phase = "hold"
                                self.mission_complete = True
        elif self.phase == "hold":
            vel_sp[:2] = g["pos_p"] * (self.pos_sp[:2] - p[:2])
            vel_sp[2] = g["pos_p"] * (self.pos_sp[2] - p[2])
        elif self.phase == "land":
            if self.land_point is None:
                self.land_point = p.copy()
                self.events.append((t, "lądowanie"))
            vel_sp[:2] = g["pos_p"] * (self.land_point[:2] - p[:2])
            h = nav.alt
            vel_sp[2] = self.land_speed if h > 4.0 else max(0.4, self.land_speed * 0.5)
            self.pos_sp = np.array([self.land_point[0], self.land_point[1], 0.0])
            # detektor lądowania (jak w PX4): kontakt z ziemią -> "może wylądował" -> wylądował
            if nav.alt < 1.5 and abs(nav.vel[2]) < 0.25:
                self.landed_timer += dt
                if self.landed_timer > 1.0:
                    self.phase = "landed"
                    if self._failsafe is None:
                        self.mission_complete = True
                    self.events.append((t, "wylądowano - silniki wyłączone"))
            else:
                self.landed_timer = 0.0
        if self.yaw_to_wp and self.phase == "mission" and math.hypot(*(self.pos_sp[:2] - p[:2])) > 3.0:
            self.yaw_sp = math.atan2(self.pos_sp[1] - p[1], self.pos_sp[0] - p[0])
        return vel_sp

    def true_track_error(self, p_true: np.ndarray) -> float:
        """Odchyłka RZECZYWISTEJ pozycji od aktualnego odcinka trasy (autopilot widzi tylko estymatę)."""
        if self.phase not in ("mission", "rtl") or self.seg_start is None:
            return 0.0
        a, b = self.seg_start, self.pos_sp
        ab = b - a
        L = float(np.linalg.norm(ab))
        if L < 1e-3:
            return float(np.linalg.norm(p_true - b))
        s = clamp(float((p_true - a) @ ab) / L, 0.0, L)
        return float(np.linalg.norm(p_true - (a + ab / L * s)))

    def _track_segment(self, a, b, p, speed):
        """Prędkość zadana do śledzenia odcinka a->b (prowadzenie po linii + hamowanie)."""
        ab = b - a
        L = np.linalg.norm(ab)
        if L < 1e-3:
            return self.g["pos_p"] * (b - p), np.linalg.norm(b - p)
        u = ab / L
        s = clamp(float((p - a) @ u), 0.0, L)
        closest = a + u * s
        d_rem = np.linalg.norm(b - p)
        self.track_error = float(np.linalg.norm(p - closest))
        v_along = min(speed, math.sqrt(2 * 1.5 * max(L - s, 0.0)), self.g["pos_p"] * d_rem)
        v = v_along * u + self.g["xtrack_p"] * (closest - p)
        if L - s < 0.5:
            v = self.g["pos_p"] * (b - p)
        return v, d_rem

    # -------------------------------------------------------------- pętle
    def update(self, t: float, dt: float, nav, status: dict) -> np.ndarray:
        n_units = self.m["n_units"]
        self._check_failsafes(t, dt, nav, status)
        vel_sp = self._guidance(t, dt, nav)
        if vel_sp is None or self.phase == "landed":
            for pid in (self.vel_n, self.vel_e, self.vel_d, *self.rate):
                pid.reset()
            return np.zeros(n_units)

        # ograniczenia prędkości
        vh = math.hypot(vel_sp[0], vel_sp[1])
        vmax = self.m["max_speed"]
        if vh > vmax:
            vel_sp[:2] *= vmax / vh
        vel_sp[2] = clamp(vel_sp[2], -self.m["max_climb"], self.m["max_descent"])
        self.vel_sp = vel_sp

        # prędkość -> przyspieszenie (z całkowaniem kompensującym wiatr i błąd modelu ciągu)
        v = nav.vel
        on_ground = self.phase == "takeoff" and nav.alt < 0.3
        maybe_landed = self.phase == "land" and self.landed_timer > 0.2
        a_sp = np.array([
            self.vel_n.update(vel_sp[0] - v[0], dt, freeze_integral=on_ground or maybe_landed),
            self.vel_e.update(vel_sp[1] - v[1], dt, freeze_integral=on_ground or maybe_landed),
            self.vel_d.update(vel_sp[2] - v[2], dt),
        ])
        if maybe_landed:   # na ziemi: poziomo, bez walki z tarciem
            a_sp[:2] = 0.0
            self.vel_n.integral *= 0.9
            self.vel_e.integral *= 0.9
        a_h_max = GRAVITY * math.tan(self.m["max_tilt"])
        ah = math.hypot(a_sp[0], a_sp[1])
        if ah > a_h_max:
            a_sp[:2] *= a_h_max / ah
        a_sp[2] = clamp(a_sp[2], -GRAVITY * 2.0, GRAVITY * 0.7)
        f = self.m["mass"] * (a_sp - np.array([0.0, 0.0, GRAVITY]))
        # ograniczenie pochylenia wektora ciągu (priorytet: utrzymanie wysokości)
        fh = math.hypot(f[0], f[1])
        fh_max = -f[2] * math.tan(self.m["max_tilt"])
        if fh > fh_max:
            f[:2] *= fh_max / fh
        R = nav.R
        z_b = R[:, 2]
        T = max(0.0, -f @ z_b)
        self.thrust_sp = T

        # orientacja zadana z kierunku wektora ciągu i kursu
        zd = -f / np.linalg.norm(f)
        xc = np.array([math.cos(self.yaw_sp), math.sin(self.yaw_sp), 0.0])
        yd = cross3(zd, xc)
        yd /= np.linalg.norm(yd)
        xd = cross3(yd, zd)
        q_sp = dcm_to_quat(np.column_stack((xd, yd, zd)))
        qe = quat_mul(quat_conj(nav.q), q_sp)
        if qe[0] < 0:
            qe = -qe
        w_sp = 2.0 * np.array(self.g["att_p"]) * qe[1:4]
        w_sp = np.clip(w_sp, -np.array(self.g["max_rate"]), np.array(self.g["max_rate"]))

        # prędkości kątowe -> momenty
        w = nav.omega
        alpha = np.array([self.rate[i].update(w_sp[i] - w[i], dt, measurement=w[i]) for i in range(3)])
        J = self.m["J"]
        tau = J @ alpha + cross3(w, J @ w)
        return self.allocate(T, tau)
