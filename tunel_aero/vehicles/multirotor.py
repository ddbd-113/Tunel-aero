"""Wielowirnikowiec (quad / hexa / octo) z modelem napędu, oporem kadłuba,
efektem przypowierzchniowym, stanem pierścienia wirowego i akumulatorem."""
from __future__ import annotations

import math

import numpy as np

from ..math3d import quat_from_euler
from .base import Vehicle
from .battery import Battery
from .propulsion import PropulsionSet

# kąty ramion [deg] mierzone od nosa (+x) w stronę prawego boku (+y); spin +1 = CCW widziane z góry
LAYOUTS = {
    "quad_x": ([45, 135, 225, 315], [1, -1, 1, -1]),
    "quad_plus": ([0, 90, 180, 270], [1, -1, 1, -1]),
    "hexa_x": ([30, 90, 150, 210, 270, 330], [1, -1, 1, -1, 1, -1]),
    "octo_x": ([22.5 + 45 * i for i in range(8)], [1, -1] * 4),
}


class Multirotor(Vehicle):
    kind = "multirotor"

    def __init__(self, cfg: dict, rng: np.random.Generator | None = None):
        self.cfg = cfg
        rot = cfg.get("rotors", {})
        layout = rot.get("layout", "quad_x")
        if layout == "custom":
            positions = np.array(rot["positions"], dtype=float)
            spins = rot["spins"]
        else:
            angles, spins = LAYOUTS[layout]
            L = rot.get("arm_length", 0.25)
            zr = rot.get("height", -0.05)
            positions = np.array([[L * math.cos(math.radians(a)), L * math.sin(math.radians(a)), zr]
                                  for a in angles])
        axes = np.tile([0.0, 0.0, -1.0], (len(positions), 1))
        gear = cfg.get("landing_gear", {"half_width": 0.15, "height": 0.12})
        hw, hg = gear["half_width"], gear["height"]
        contact = [[hw, hw, hg], [hw, -hw, hg], [-hw, hw, hg], [-hw, -hw, hg]]
        crash = cfg.get("crash", {})
        super().__init__(cfg.get("name", "multirotor"), cfg["mass"], cfg["inertia"], contact,
                         crash_vertical_speed=crash.get("max_vertical_speed", 3.0),
                         crash_horizontal_speed=crash.get("max_horizontal_speed", 4.0),
                         max_tilt_on_ground_deg=crash.get("max_tilt_on_ground_deg", 60.0))
        self.props = PropulsionSet(
            positions, axes, spins,
            diameter=rot.get("diameter", 0.254), ct0=rot.get("ct0", 0.10), j0=rot.get("j0", 0.75),
            eta=rot.get("eta", 0.75), cp0=rot.get("cp0", 0.005), kv=rot.get("kv", 920.0),
            kv_efficiency=rot.get("kv_efficiency", 0.8), motor_efficiency=rot.get("motor_efficiency", 0.82),
            tau=rot.get("tau", 0.04), h_force_coeff=rot.get("h_force_coeff", 1.0e-3),
            ground_effect=True, vrs=rot.get("vortex_ring_state", True), rng=rng)
        self.battery = Battery(**cfg.get("battery", {}))
        drag = cfg.get("drag", {})
        self.drag_area = np.asarray(drag.get("area", [0.04, 0.04, 0.09]), dtype=float)
        self.drag_cd = drag.get("cd", 1.0)
        self.limits = cfg.get("limits", {})
        self.throttle = np.zeros(self.props.n_units)
        self._prop_cache = None

    # -----------------------------------------------------------------------
    def controller_model(self) -> dict:
        """Nominalny model, który "zna" autopilot (bez wiedzy o warunkach i awariach)."""
        pr = self.props
        v_nom = self.battery.cells * 3.85
        t_max = pr.static_thrust_max(v_nom, 1.225)
        t_hover = self.mass * 9.80665 / pr.n_units
        n_h = math.sqrt(t_hover / (pr.ct0 * 1.225 * pr.D ** 4))
        vh = math.sqrt(t_hover / (2 * 1.225 * pr.A))
        p_shaft = t_hover * vh / pr.eta + pr.cp0 * 1.225 * n_h ** 3 * pr.D ** 5
        k_q = p_shaft / (2 * math.pi * n_h) / t_hover
        return {
            "mass": self.mass, "J": self.J, "positions": pr.pos, "spins": pr.spin,
            "k_q": k_q, "t_max": t_max, "n_units": pr.n_units,
            "max_tilt": math.radians(self.limits.get("max_tilt_deg", 35.0)),
            "max_speed": self.limits.get("max_speed", 12.0),
            "max_climb": self.limits.get("max_climb_rate", 3.0),
            "max_descent": self.limits.get("max_descent_rate", 2.0),
        }

    def initial_state(self, north=0.0, east=0.0, altitude=0.0, yaw_deg=0.0, **_) -> np.ndarray:
        gear_h = self.contact_points[0, 2]
        d = -altitude - (gear_h - 0.002 if altitude <= 0.0 else 0.0)
        x = np.zeros(13)
        x[0:3] = [north, east, d]
        x[6:10] = quat_from_euler(0.0, 0.0, math.radians(yaw_deg))
        if altitude > 0.0:   # start w powietrzu - wirniki na obrotach zawisu
            n_h = math.sqrt(self.mass * 9.80665 / self.props.n_units / (self.props.ct0 * 1.225 * self.props.D ** 4))
            self.props.n[:] = n_h
        return x

    def body_forces(self, x, env, R, v_air_b, record):
        rho = env.atm.density
        w = x[10:13]
        if record or self._prop_cache is None:
            # siły napędu liczone raz na krok (obroty silników są stałe w obrębie kroku)
            self._prop_cache = self.props.forces(v_air_b, w, rho, env.h_agl, env.deg.thrust_factor, R,
                                                 env.deg.prop_drag_factor)
        F, M, p_elec = self._prop_cache
        speed = math.sqrt(v_air_b @ v_air_b)
        cd = self.drag_cd + env.deg.cd_add * 10.0  # mokry/oblodzony kadłub
        F = F - 0.5 * rho * speed * v_air_b * self.drag_area * cd
        if env.deg.rain_mass_flux > 0:
            # pęd kropel deszczu (prędkość opadania ~8 m/s)
            c_w = env.deg.rain_mass_flux / 8.0
            u = R.T @ (env.wind.total + np.array([0.0, 0.0, 8.0]) - x[3:6])
            F = F + c_w * math.sqrt(u @ u) * u * self.drag_area
        if record:
            self.aux["power"] = p_elec
            self.aux["thrust_total"] = float(np.sum(self.props.thrust))
            self.aux["vrs"] = float(np.max(self.props.vrs_level))
        return F, M

    def step_actuators(self, dt, cmd, env):
        self.throttle = np.clip(np.asarray(cmd, dtype=float), 0.0, 1.0)
        self.props.update_speed(dt, self.throttle, self.battery.voltage if self.battery.voltage > 0 else 0.0)

    def icing_exposure_speed(self, airspeed: float) -> float:
        return 0.75 * math.pi * self.props.D * float(np.mean(self.props.n)) * 0.5 + airspeed
