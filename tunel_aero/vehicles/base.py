"""Dynamika bryły sztywnej 6-DOF + kontakt z ziemią (model sprężysto-tłumiący z tarciem)."""
from __future__ import annotations

import math

import numpy as np

from ..environment import EnvSample
from ..math3d import GRAVITY, cross3, quat_to_dcm

# indeksy wektora stanu
POS = slice(0, 3)
VEL = slice(3, 6)    # prędkość w NED
QUAT = slice(6, 10)  # ciało -> NED
OMEGA = slice(10, 13)


class Vehicle:
    kind = "base"

    def __init__(self, name: str, mass: float, inertia, contact_points,
                 crash_vertical_speed: float = 3.0, crash_horizontal_speed: float = 6.0,
                 ground_contact_is_crash: bool = False, friction: float = 0.6,
                 max_tilt_on_ground_deg: float = 60.0):
        self.name = name
        self.mass = float(mass)
        J = np.asarray(inertia, dtype=float)
        self.J = np.diag(J) if J.ndim == 1 else J
        self.Jinv = np.linalg.inv(self.J)
        self.contact_points = np.atleast_2d(np.asarray(contact_points, dtype=float))
        n_pts = len(self.contact_points)
        self.k_contact = self.mass * GRAVITY / (n_pts * 0.01)
        self.c_contact = 2 * 0.8 * math.sqrt(self.k_contact * self.mass / n_pts)
        self.mu = friction
        self.crash_vz = crash_vertical_speed
        self.crash_vh = crash_horizontal_speed
        self.contact_is_crash = ground_contact_is_crash
        self.max_tilt_ground = math.radians(max_tilt_on_ground_deg)
        self.battery = None
        self.on_ground = False
        self.crashed = False
        self.crash_reason = ""
        self.impact_speed = 0.0
        self.aux: dict = {}
        self.specific_force = np.array([0.0, 0.0, -GRAVITY])
        self.extra_mass = 0.0

    # ---- do nadpisania w klasach pochodnych --------------------------------
    def body_forces(self, x, env: EnvSample, R, v_air_b, record: bool):
        raise NotImplementedError

    def step_actuators(self, dt: float, cmd, env: EnvSample) -> None:
        raise NotImplementedError

    def icing_exposure_speed(self, airspeed: float) -> float:
        return airspeed

    def initial_state(self, **kw) -> np.ndarray:
        raise NotImplementedError

    # -----------------------------------------------------------------------
    @property
    def total_mass(self) -> float:
        return self.mass + self.extra_mass

    def contact(self, x, R, env: EnvSample, record: bool):
        p, v, w = x[POS], x[VEL], x[OMEGA]
        ground_d = env.ground_d
        pts_ned = p + self.contact_points @ R.T
        pen = pts_ned[:, 2] - ground_d
        if not np.any(pen > 0):
            if record:
                self.on_ground = False
            return np.zeros(3), np.zeros(3)
        F_b = np.zeros(3)
        M_b = np.zeros(3)
        touching = False
        for i in np.nonzero(pen > 0)[0]:
            r = self.contact_points[i]
            v_pt = v + R @ cross3(w, r)
            if record and not self.on_ground:
                vz, vh = v_pt[2], math.hypot(v_pt[0], v_pt[1])
                self.impact_speed = max(self.impact_speed, vz)
                if not self.crashed:
                    if self.contact_is_crash:
                        self._crash(f"kontakt z ziemią (Vz={vz:.1f} m/s, Vh={vh:.1f} m/s)")
                    elif vz > self.crash_vz:
                        self._crash(f"twarde uderzenie w ziemię Vz={vz:.1f} m/s")
                    elif vh > self.crash_vh:
                        self._crash(f"przyziemienie z dużą prędkością poziomą {vh:.1f} m/s")
            fn = max(0.0, self.k_contact * pen[i] + self.c_contact * v_pt[2])
            vt = v_pt[:2]
            ft = -self.mu * fn * vt / math.sqrt(vt @ vt + 0.01)
            f_ned = np.array([ft[0], ft[1], -fn])
            f_b = R.T @ f_ned
            F_b += f_b
            M_b += cross3(r, f_b)
            touching = True
        if record:
            self.on_ground = touching
            if touching and not self.crashed:
                tilt = math.acos(max(-1.0, min(1.0, R[2, 2])))
                if tilt > self.max_tilt_ground:
                    self._crash(f"przewrócenie na ziemi (przechył {math.degrees(tilt):.0f} deg)")
        return F_b, M_b

    def _crash(self, reason: str) -> None:
        self.crashed = True
        self.crash_reason = reason

    def dynamics(self, x: np.ndarray, env: EnvSample, record: bool = False) -> np.ndarray:
        q = x[QUAT]
        w = x[OMEGA]
        R = quat_to_dcm(q)
        v_air_b = R.T @ (x[VEL] - env.wind.total)
        F_b, M_b = self.body_forces(x, env, R, v_air_b, record)
        Fc, Mc = self.contact(x, R, env, record)
        F_b = F_b + Fc
        M_b = M_b + Mc
        m = self.total_mass
        acc = R @ F_b / m
        acc[2] += env.gravity
        w_dot = self.Jinv @ (M_b - cross3(w, self.J @ w))
        qw, qx, qy, qz = q
        wx, wy, wz = w
        q_dot = 0.5 * np.array([
            -qx * wx - qy * wy - qz * wz,
            qw * wx + qy * wz - qz * wy,
            qw * wy - qx * wz + qz * wx,
            qw * wz + qx * wy - qy * wx,
        ])
        if record:
            self.specific_force = F_b / m
            self.aux["load_factor"] = -self.specific_force[2] / GRAVITY
        return np.concatenate((x[VEL], acc, q_dot, w_dot))
