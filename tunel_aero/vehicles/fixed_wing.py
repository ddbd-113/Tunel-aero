"""Samolot (płatowiec) - model aerodynamiczny oparty na pochodnych
stateczności z nieliniowym przeciągnięciem (płynne przejście do opływu
oderwanego), serwomechanizmami sterów, napędem elektrycznym i akumulatorem."""
from __future__ import annotations

import math

import numpy as np

from ..math3d import quat_from_euler
from .base import Vehicle
from .battery import Battery
from .propulsion import PropulsionSet

DEFAULT_AERO = {
    "CL0": 0.25, "CLa": 4.8, "CLq": 4.0, "CLde": 0.30,
    "CD0": 0.035, "e": 0.8,
    "Cm0": 0.02, "Cma": -0.70, "Cmq": -12.0, "Cmde": -1.0,
    "CYb": -0.40, "CYp": 0.0, "CYr": 0.10, "CYdr": 0.15,
    "Clb": -0.08, "Clp": -0.50, "Clr": 0.12, "Clda": 0.20, "Cldr": 0.005,
    "Cnb": 0.08, "Cnp": -0.05, "Cnr": -0.15, "Cnda": -0.01, "Cndr": -0.06,
    "alpha_stall_deg": 15.0, "stall_blend": 40.0, "post_stall_cl": 1.0, "cd90": 1.3,
    "stall_roll_asymmetry": 0.015,
}

SURFACES = ("aileron", "elevator", "rudder")


class FixedWing(Vehicle):
    kind = "fixed_wing"

    def __init__(self, cfg: dict, rng: np.random.Generator | None = None):
        self.cfg = cfg
        rng = rng or np.random.default_rng()
        geo = cfg.get("geometry", {})
        self.S = geo.get("wing_area", 0.44)
        self.b = geo.get("span", 2.0)
        self.c = geo.get("chord", 0.22)
        self.AR = self.b ** 2 / self.S
        J = cfg.get("inertia", [0.30, 0.20, 0.45, 0.01])
        Jx, Jy, Jz = J[:3]
        Jxz = J[3] if len(J) > 3 else 0.0
        inertia = np.array([[Jx, 0, -Jxz], [0, Jy, 0], [-Jxz, 0, Jz]])
        half_b = self.b / 2
        contact = cfg.get("contact_points") or [[0.45, 0.0, 0.08], [-0.7, 0.0, 0.05], [0.0, half_b, 0.0],
                                                [0.0, -half_b, 0.0]]
        crash = cfg.get("crash", {})
        super().__init__(cfg.get("name", "fixed_wing"), cfg["mass"], inertia, contact,
                         crash_vertical_speed=crash.get("max_vertical_speed", 2.5),
                         ground_contact_is_crash=crash.get("ground_contact_is_crash", True))
        self.aero = dict(DEFAULT_AERO)
        self.aero.update(cfg.get("aero", {}))
        self.stall_sign = 1.0 if rng.random() < 0.5 else -1.0
        prop = cfg.get("propulsion", {})
        self.props = PropulsionSet(
            [prop.get("position", [0.4, 0.0, 0.0])], [[1.0, 0.0, 0.0]], [prop.get("spin", 1)],
            diameter=prop.get("diameter", 0.28), ct0=prop.get("ct0", 0.11), j0=prop.get("j0", 0.85),
            eta=prop.get("eta", 0.85), cp0=prop.get("cp0", 0.005), kv=prop.get("kv", 700.0),
            kv_efficiency=prop.get("kv_efficiency", 0.85), motor_efficiency=prop.get("motor_efficiency", 0.82),
            tau=prop.get("tau", 0.08), rng=rng)
        self.battery = Battery(**cfg.get("battery", {}))
        srv = cfg.get("servos", {})
        lim = srv.get("limit_deg", 25.0)      # jedna wartość lub [lotki, ster wys., ster kier.]
        self.servo_limit = np.radians(np.broadcast_to(np.asarray(lim, dtype=float), (3,))).copy()
        self.servo_tau = srv.get("tau", 0.04)
        self.servo_rate = math.radians(srv.get("rate_deg_s", 300.0))
        self.delta = np.zeros(3)                 # [lotka, ster wysokości, ster kierunku] [rad]
        self.stuck: dict[int, float] = {}        # zablokowane stery: indeks -> pozycja
        self.throttle = 0.0
        self._prop_cache = None
        lim = cfg.get("limits", {})
        self.vne = lim.get("vne", 35.0)
        self.n_max = lim.get("max_load_factor", 4.0)
        self.v_stall_hint = lim.get("stall_speed", 10.0)

    # ---------------------------------------------------------------- aero
    def coefficients(self, alpha, beta, p, q, r, Va, delta, deg=None):
        """Współczynniki aerodynamiczne (CL, CD, CY, Cl, Cm, Cn) - używane też przez tunel."""
        a = self.aero
        da, de, dr = delta
        cl_f = deg.cl_factor if deg else 1.0
        cd_add = deg.cd_add if deg else 0.0
        st_red = deg.stall_reduction if deg else 0.0
        cm_f = deg.cm_factor if deg else 1.0
        a0 = math.radians(a["alpha_stall_deg"]) - st_red
        M = a["stall_blend"]
        # funkcja przejścia opływ przylegający -> oderwany
        e1 = math.exp(max(-50.0, min(50.0, -M * (alpha - a0))))
        e2 = math.exp(max(-50.0, min(50.0, M * (alpha + a0))))
        sigma = (1 + e1 + e2) / ((1 + e1) * (1 + e2))
        Va = max(Va, 1.0)
        cl_att = cl_f * (a["CL0"] + a["CLa"] * alpha)
        cl_post = a["post_stall_cl"] * math.sin(2 * alpha)
        CL = (1 - sigma) * cl_att + sigma * cl_post + a["CLq"] * self.c * q / (2 * Va) + a["CLde"] * de
        cd_att = a["CD0"] + cd_add + cl_att ** 2 / (math.pi * a["e"] * self.AR)
        cd_post = a["CD0"] + cd_add + a["cd90"] * math.sin(alpha) ** 2
        CD = (1 - sigma) * cd_att + sigma * cd_post + abs(a.get("CDde", 0.02) * de) + 0.3 * beta * beta
        bv = self.b / (2 * Va)
        CY = a["CYb"] * beta + a["CYp"] * bv * p + a["CYr"] * bv * r + a["CYdr"] * dr
        Cl = (a["Clb"] * beta + a["Clp"] * (1 - 0.7 * sigma) * bv * p + a["Clr"] * bv * r
              + cm_f * a["Clda"] * da + a["Cldr"] * dr + sigma * a["stall_roll_asymmetry"] * self.stall_sign)
        Cm = cm_f * (a["Cm0"] + a["Cma"] * alpha + a["Cmde"] * de) + a["Cmq"] * self.c * q / (2 * Va)
        Cn = (a["Cnb"] * beta + a["Cnp"] * bv * p + a["Cnr"] * bv * r
              + cm_f * a["Cnda"] * da + cm_f * a["Cndr"] * dr)
        return CL, CD, CY, Cl, Cm, Cn, sigma

    def aero_forces(self, v_air_b, w, rho, delta, deg=None, record=False):
        u, v, ww = v_air_b
        Va = math.sqrt(u * u + v * v + ww * ww)
        alpha = math.atan2(ww, u)
        beta = math.asin(max(-1.0, min(1.0, v / Va))) if Va > 1e-3 else 0.0
        p, q, r = w
        CL, CD, CY, Cl, Cm, Cn, sigma = self.coefficients(alpha, beta, p, q, r, Va, delta, deg)
        qbar = 0.5 * rho * Va * Va
        ca, sa = math.cos(alpha), math.sin(alpha)
        F = qbar * self.S * np.array([-CD * ca + CL * sa, CY, -CD * sa - CL * ca])
        M = qbar * self.S * np.array([self.b * Cl, self.c * Cm, self.b * Cn])
        if record:
            self.aux.update(alpha=alpha, beta=beta, airspeed=Va, CL=CL, CD=CD, stall=sigma, qbar=qbar)
        return F, M

    def body_forces(self, x, env, R, v_air_b, record):
        rho = env.atm.density
        w = x[10:13]
        F, M = self.aero_forces(v_air_b, w, rho, self.delta, env.deg, record)
        if record or self._prop_cache is None:
            self._prop_cache = self.props.forces(v_air_b, w, rho, env.h_agl, env.deg.thrust_factor,
                                                 profile_factor=env.deg.prop_drag_factor)
        Fp, Mp, p_elec = self._prop_cache
        if record:
            self.aux["power"] = p_elec
            self.aux["thrust_total"] = float(self.props.thrust[0])
        return F + Fp, M + Mp

    # ----------------------------------------------------------- aktuatory
    def step_actuators(self, dt, cmd, env):
        """cmd = [lotka, ster wys., ster kier.] [rad] + przepustnica 0..1"""
        target = np.clip(np.asarray(cmd[:3], dtype=float), -self.servo_limit, self.servo_limit)
        a = 1.0 - math.exp(-dt / self.servo_tau)
        step = np.clip((target - self.delta) * a, -self.servo_rate * dt, self.servo_rate * dt)
        self.delta = self.delta + step
        for i, val in self.stuck.items():
            self.delta[i] = val
        self.throttle = float(np.clip(cmd[3], 0.0, 1.0))
        self.props.update_speed(dt, np.array([self.throttle]), self.battery.voltage)

    def icing_exposure_speed(self, airspeed: float) -> float:
        return airspeed

    def initial_state(self, north=0.0, east=0.0, altitude=100.0, yaw_deg=0.0, airspeed=17.0,
                      trim=None, wind_ned=None, **_) -> np.ndarray:
        """Start w locie ustalonym (trym). trim = (alpha, elevator, throttle)."""
        alpha, de, thr = trim if trim is not None else (0.03, 0.0, 0.5)
        yaw = math.radians(yaw_deg)
        x = np.zeros(13)
        x[0:3] = [north, east, -altitude]
        q = quat_from_euler(0.0, alpha, yaw)
        x[6:10] = q
        wind = np.zeros(3) if wind_ned is None else np.asarray(wind_ned, dtype=float)
        v_air_ned = airspeed * np.array([math.cos(yaw), math.sin(yaw), 0.0])
        x[3:6] = v_air_ned + wind
        self.delta = np.array([0.0, de, 0.0])
        self.throttle = thr
        self.props.n[:] = thr * self.props.n_max(self.battery.voltage)
        return x
