"""Autopilot samolotu:
  * nawigacja: śledzenie odcinków trasy (prawo prowadzenia "vector field", Beard & McLain)
    i krążenie (loiter) po okręgu,
  * kanał podłużny: TECS (Total Energy Control System) - przepustnica steruje całkowitą
    energią, pochylenie - jej rozdziałem między wysokość a prędkość; ochrona przed
    przeciągnięciem (priorytet prędkości przy zbyt małej prędkości),
  * kanał poprzeczny: kurs -> przechylenie -> lotki, koordynacja zakrętu sterem kierunku.
Regulatory pętli wewnętrznych są skalowane ciśnieniem dynamicznym (gain scheduling).
"""
from __future__ import annotations

import math

import numpy as np

from ..math3d import GRAVITY, clamp, wrap_pi
from .pid import PID

DEFAULT_GAINS = {
    "course_p": 1.1, "course_i": 0.05,
    "roll_p": 0.55, "roll_i": 0.08, "roll_d": 0.03,
    "yaw_damper": 0.25,
    "pitch_p": 0.9, "pitch_i": 0.25, "pitch_d": 0.09,
    "h_p": 0.25, "v_p": 0.4,
    "thr_p": 0.012, "thr_i": 0.006,
    "seb_p": 1.0, "seb_i": 0.02,
    "path_k": 0.03, "path_chi_inf_deg": 60.0, "orbit_k": 2.0,
}


class FixedWingAutopilot:
    def __init__(self, model: dict, mission: dict, gains: dict | None = None):
        self.m = model
        g = dict(DEFAULT_GAINS)
        g.update(gains or {})
        self.g = g
        self.mission = mission
        self.wps = [np.array([w[0], w[1], w[2]], float) if not isinstance(w, dict)
                    else np.array([w["north"], w["east"], w["alt"]], float)
                    for w in mission.get("waypoints", [])]
        self.v_cruise = mission.get("airspeed", model["v_trim"])
        self.max_bank = math.radians(mission.get("max_bank_deg", 35.0))
        self.loiter_radius = mission.get("loiter_radius", 60.0)
        self.climb_max = mission.get("max_climb_rate", 3.0)
        self.sink_max = mission.get("max_sink_rate", 3.0)
        self.low_batt = mission.get("low_battery_soc", 0.2)
        self.link_loss_action = mission.get("link_loss_action", "rtl")
        self.end_action = "rtl" if mission.get("return_home", True) else "loiter"
        self.phase = "mission" if self.wps else "loiter"
        self.wp_i = 0
        self.prev_wp = None
        self.home = None
        self.loiter_center = None
        self.loiter_alt = None
        self.mission_complete = False
        self.events: list[tuple[float, str]] = []
        self._failsafe = None

        self.course = PID(g["course_p"], g["course_i"], i_limit=math.radians(10))
        self.roll = PID(g["roll_p"], g["roll_i"], i_limit=math.radians(5))
        self.pitch = PID(g["pitch_p"], g["pitch_i"], i_limit=math.radians(8))
        self.thr_i = 0.0
        self.seb_i = 0.0
        self.vdot_f = 0.0
        self.track_error = 0.0
        self.h_sp = 0.0
        self.v_sp = self.v_cruise
        self.course_sp = 0.0
        self.roll_sp = 0.0
        self.pitch_sp = 0.0
        self.rudder_roll = False
        self.rr = PID(g.get("rudder_roll_p", 1.2), g.get("rudder_roll_i", 0.3), i_limit=math.radians(15))

    # ------------------------------------------------------------ nawigacja
    def _set_failsafe(self, action, t, reason):
        if self._failsafe:
            return
        self._failsafe = action
        self.events.append((t, f"FAILSAFE: {reason}"))
        self._goto_loiter(t, self.home, max(self.loiter_alt or 0.0, 60.0), "powrót nad punkt startu")

    def _goto_loiter(self, t, center, alt, msg):
        self.phase = "loiter"
        self.loiter_center = np.array(center[:2], float)
        self.loiter_alt = alt
        self.events.append((t, msg))

    def _guidance(self, t, nav):
        p = nav.pos
        chi = nav.course
        if self.home is None:
            self.home = p.copy()
            self.prev_wp = np.array([p[0], p[1], nav.alt])
            self.loiter_alt = nav.alt
            if self.phase == "loiter":
                self.loiter_center = p[:2].copy()
        if self.phase == "mission":
            a, b = self.prev_wp, self.wps[self.wp_i]
            q = b[:2] - a[:2]
            Lq = np.linalg.norm(q)
            q = q / Lq if Lq > 1e-6 else np.array([math.cos(chi), math.sin(chi)])
            chi_q = math.atan2(q[1], q[0])
            chi_q = chi + wrap_pi(chi_q - chi)
            e_py = -math.sin(chi_q) * (p[0] - a[0]) + math.cos(chi_q) * (p[1] - a[1])
            self.track_error = abs(e_py)
            chi_inf = math.radians(self.g["path_chi_inf_deg"])
            chi_c = chi_q - chi_inf * 2 / math.pi * math.atan(self.g["path_k"] * e_py)
            # wysokość zadana interpolowana wzdłuż odcinka
            s = clamp(float((p[:2] - a[:2]) @ q) / max(Lq, 1e-6), 0.0, 1.0)
            self.h_sp = a[2] + (b[2] - a[2]) * s
            # przełączenie punktu: przekroczenie półpłaszczyzny w punkcie b
            d_switch = 0.0
            if self.wp_i + 1 < len(self.wps):
                q2 = self.wps[self.wp_i + 1][:2] - b[:2]
                q2 = q2 / max(np.linalg.norm(q2), 1e-6)
                n = q + q2
                n = n / max(np.linalg.norm(n), 1e-6)
                # antycypacja zakrętu: przełączenie wcześniej o R*tan(dchi/2) (łuk wpisany w narożnik)
                dchi = math.acos(clamp(float(q @ q2), -1.0, 1.0))
                vg = max(nav.ground_speed, 8.0)
                r_turn = vg * vg / (GRAVITY * math.tan(0.8 * self.max_bank))
                d_switch = min(r_turn * math.tan(dchi / 2), 3 * r_turn)
            else:
                n = q
            along = float((b[:2] - p[:2]) @ q)
            if (p[:2] - b[:2]) @ n >= 0 or along < d_switch or np.linalg.norm(p[:2] - b[:2]) < 10.0:
                self.events.append((t, f"osiągnięto punkt trasy {self.wp_i + 1}/{len(self.wps)}"))
                self.prev_wp = b.copy()
                self.wp_i += 1
                if self.wp_i >= len(self.wps):
                    self.mission_complete = True
                    if self.end_action == "rtl":
                        self._goto_loiter(t, self.home, b[2], "koniec trasy - powrót i krążenie nad startem")
                    else:
                        self._goto_loiter(t, b, b[2], "koniec trasy - krążenie nad ostatnim punktem")
            return chi_c, 0.0
        # krążenie (loiter)
        c = self.loiter_center
        d = math.hypot(p[0] - c[0], p[1] - c[1])
        phi = math.atan2(p[1] - c[1], p[0] - c[0])
        phi = chi + wrap_pi(phi - chi)
        rho = self.loiter_radius
        lam = 1.0
        chi_c = phi + lam * (math.pi / 2 + math.atan(self.g["orbit_k"] * (d - rho) / rho))
        self.track_error = abs(d - rho)
        self.h_sp = self.loiter_alt
        vg = max(nav.ground_speed, 5.0)
        ff = lam * math.atan(vg * vg / (GRAVITY * rho)) if d < 2 * rho else 0.0
        return chi_c, ff

    def true_track_error(self, p_true: np.ndarray) -> float:
        if self.phase == "mission" and self.prev_wp is not None:
            a, b = self.prev_wp[:2], self.wps[self.wp_i][:2]
            q = b - a
            L = float(np.linalg.norm(q))
            if L < 1e-6:
                return 0.0
            q = q / L
            return abs(-q[1] * (p_true[0] - a[0]) + q[0] * (p_true[1] - a[1]))
        if self.loiter_center is not None:
            return abs(math.hypot(p_true[0] - self.loiter_center[0], p_true[1] - self.loiter_center[1])
                       - self.loiter_radius)
        return 0.0

    def notify_surface_failure(self, idx: int, t: float) -> None:
        """Po wykryciu zablokowanej lotki przechylenie realizowane jest sterem kierunku
        (ślizg + wznios skrzydeł) - prosta rekonfiguracja sterowania."""
        if idx == 0 and not self.rudder_roll:
            self.rudder_roll = True
            self.events.append((t, "autopilot wykrył zablokowaną lotkę - sterowanie przechyleniem sterem kierunku"))

    # ---------------------------------------------------------------- pętle
    def update(self, t: float, dt: float, nav, status: dict) -> np.ndarray:
        g = self.g
        if status.get("link_lost"):
            self._set_failsafe(self.link_loss_action, t, "utrata łącza z operatorem")
        if status.get("battery_soc", 1.0) < self.low_batt:
            self._set_failsafe("rtl", t, f"niski poziom baterii ({status['battery_soc'] * 100:.0f}%)")
        chi_c, roll_ff = self._guidance(t, nav)
        self.course_sp = chi_c
        V = max(nav.airspeed, 3.0)
        sched = clamp((self.m["v_trim"] / V) ** 2, 0.4, 2.5)

        # --- kanał poprzeczny
        chi_err = wrap_pi(chi_c - nav.course)
        roll_c = clamp(roll_ff + self.course.update(chi_err, dt), -self.max_bank, self.max_bank)
        self.roll_sp = roll_c
        roll, pitch, _ = nav.euler
        p, q, r = nav.omega
        da = sched * (self.roll.update(roll_c - roll, dt) - g["roll_d"] * p)
        r_ff = GRAVITY / V * math.tan(roll) * math.cos(pitch)
        dr = sched * g["yaw_damper"] * (r - r_ff)
        if self.rudder_roll:
            roll_c = clamp(roll_c, -math.radians(20), math.radians(20))
            dr = -sched * (self.rr.update(roll_c - roll, dt) - 0.15 * p)

        # --- kanał podłużny (TECS)
        h, hdot = nav.alt, nav.climb_rate
        v_min = self.m["v_min"]
        v_sp = max(self.v_cruise, v_min + 1.0)
        self.v_sp = v_sp
        h_err = self.h_sp - h
        hdot_dem = clamp(g["h_p"] * h_err, -self.sink_max, self.climb_max)
        vdot_dem = clamp(g["v_p"] * (v_sp - V), -1.5, 1.5)
        a_x = nav.accel_body[0] - GRAVITY * math.sin(pitch)
        self.vdot_f += (a_x - self.vdot_f) * min(1.0, dt / 0.3)
        vdot = self.vdot_f
        ste_dem = GRAVITY * hdot_dem + V * vdot_dem
        ste = GRAVITY * hdot + V * vdot
        # wagi rozdziału energii: przy niskiej prędkości priorytet ma prędkość (ochrona przed przeciągnięciem)
        w_h = 1.0 if V > v_min else clamp(1.0 - (v_min - V) / 2.0, 0.0, 1.0)
        seb_dem = w_h * GRAVITY * hdot_dem - (2.0 - w_h) * V * vdot_dem
        seb = w_h * GRAVITY * hdot - (2.0 - w_h) * V * vdot
        thr_trim = self.m["thr_trim"]
        ste_max = GRAVITY * self.m["climb_full_throttle"]
        thr_ff = thr_trim + ste_dem / ste_max * (1 - thr_trim if ste_dem > 0 else thr_trim)
        thr_unsat = thr_ff + g["thr_p"] * (ste_dem - ste) + self.thr_i
        thr = clamp(thr_unsat, 0.0, 1.0)
        if 0.0 < thr_unsat < 1.0 or (thr_unsat >= 1.0) == (ste_dem - ste < 0):
            self.thr_i = clamp(self.thr_i + g["thr_i"] * (ste_dem - ste) * dt, -0.4, 0.4)
        pitch_c = self.m["alpha_trim"] + (seb_dem + g["seb_p"] * (seb_dem - seb)) / (2 * GRAVITY * V) + self.seb_i
        self.seb_i = clamp(self.seb_i + g["seb_i"] * (seb_dem - seb) / (2 * GRAVITY * V) * dt,
                           -math.radians(8), math.radians(8))
        pitch_c = clamp(pitch_c, math.radians(-20), math.radians(20))
        self.pitch_sp = pitch_c
        de = self.m["de_trim"] - sched * (self.pitch.update(pitch_c - pitch, dt) - g["pitch_d"] * q)
        return np.array([da, de, dr, thr])
