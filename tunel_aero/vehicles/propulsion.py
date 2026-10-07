"""Zespoły napędowe silnik BLDC + śmigło (wektorowo dla wielu wirników).

Ciąg:  T = C_T(J) * rho * n^2 * D^4,   C_T(J) = C_T0 * (1 - J/J0),  J = V_osiowa / (n D)
Moc:   P_wał = T (V + v_i) / eta + c_p0 * rho * n^3 * D^5   (teoria strumieniowa + straty profilowe)
Obroty maksymalne zależą od napięcia akumulatora (KV), więc spadek napięcia = spadek ciągu.

Dodatkowo dla wielowirnikowców: efekt przypowierzchniowy (Cheeseman-Bennett),
stan pierścienia wirowego (VRS) przy szybkim pionowym opadaniu, siła H (rotor drag).
"""
from __future__ import annotations

import math

import numpy as np

from ..math3d import cross_rows


class PropulsionSet:
    def __init__(self, positions, axes, spins, diameter=0.254, ct0=0.10, j0=0.75, eta=0.75,
                 cp0=0.005, kv=920.0, kv_efficiency=0.8, motor_efficiency=0.82, tau=0.04,
                 h_force_coeff=0.0, ground_effect=False, vrs=False, rng=None):
        self.pos = np.atleast_2d(np.asarray(positions, dtype=float))
        self.axis = np.atleast_2d(np.asarray(axes, dtype=float))
        self.axis /= np.linalg.norm(self.axis, axis=1, keepdims=True)
        self.spin = np.asarray(spins, dtype=float)
        self.n_units = len(self.pos)
        self.D = diameter
        self.A = math.pi * diameter ** 2 / 4
        self.ct0, self.j0, self.eta, self.cp0 = ct0, j0, eta, cp0
        self.kv, self.kv_eff = kv, kv_efficiency
        self.motor_eff = motor_efficiency
        self.tau = tau
        self.h_coeff = h_force_coeff
        self.ground_effect = ground_effect
        self.vrs_enabled = vrs
        self.rng = rng or np.random.default_rng()
        self.n = np.zeros(self.n_units)          # obroty [obr/s]
        self.health = np.ones(self.n_units)      # 0 = silnik nie działa, <1 = uszkodzone śmigło
        self.vrs_noise = np.zeros(self.n_units)
        self.vrs_state = np.zeros(self.n_units)   # VRS rozwija się w czasie ~1 s
        self._dt = 0.005
        # wyniki ostatniej ewaluacji (do logów)
        self.thrust = np.zeros(self.n_units)
        self.power_shaft = np.zeros(self.n_units)
        self.vrs_level = np.zeros(self.n_units)

    def n_max(self, voltage: float) -> float:
        return self.kv * self.kv_eff * max(voltage, 0.0) / 60.0

    def static_thrust_max(self, voltage: float, rho: float = 1.225) -> float:
        n = self.n_max(voltage)
        return self.ct0 * rho * n * n * self.D ** 4

    def update_speed(self, dt: float, throttle: np.ndarray, voltage: float) -> None:
        cmd = np.clip(throttle, 0.0, 1.0) * self.n_max(voltage) * np.minimum(self.health * 10.0, 1.0)
        a = math.exp(-dt / self.tau)
        self.n = cmd + (self.n - cmd) * a
        self._dt = dt   # dla health=0 cmd=0 - śmigło wybiega

    def forces(self, v_air_body: np.ndarray, omega: np.ndarray, rho: float, h_agl: float,
               thrust_factor: float = 1.0, R_body_to_ned: np.ndarray | None = None,
               profile_factor: float = 1.0):
        """Zwraca (F_body, M_body, P_elektryczna) dla wszystkich jednostek."""
        n = np.maximum(self.n, 1e-3)
        # prędkość powietrza w miejscu każdego wirnika (uwzględnia obrót kadłuba)
        v_loc = v_air_body + cross_rows(omega, self.pos)
        v_ax = np.einsum("ij,ij->i", v_loc, self.axis)          # napływ osiowy (+ od przodu wirnika)
        v_perp = v_loc - v_ax[:, None] * self.axis
        J = v_ax / (n * self.D)
        ct = np.minimum(np.maximum(self.ct0 * (1.0 - J / self.j0), -0.3 * self.ct0), 1.3 * self.ct0)
        T = ct * rho * n * n * self.D ** 4 * self.health * thrust_factor
        T = np.where(self.n < 1.0, 0.0, T)

        if self.ground_effect and R_body_to_ned is not None:
            # wysokość każdego wirnika nad ziemią
            z = h_agl - (R_body_to_ned @ self.pos.T)[2]
            R = self.D / 2
            z = np.maximum(z, 0.5 * R)
            ge = np.where(z < 4 * R, 1.0 / (1.0 - (R / (4 * z)) ** 2), 1.0)
            T = T * ge

        vh = np.sqrt(np.maximum(T, 0.0) / (2 * rho * self.A)) + 1e-6
        if self.vrs_enabled:
            d = -v_ax / vh                       # względna prędkość opadania
            mu = np.linalg.norm(v_perp, axis=1) / vh
            sev = np.where((d > 0.4) & (d < 1.6) & (mu < 0.6) & (vh > 2.0) & (h_agl > 1.0),
                           np.sin(np.pi * (d - 0.4) / 1.2) * (1.0 - mu / 0.6), 0.0)
            self.vrs_state += (sev - self.vrs_state) * min(1.0, self._dt / 1.0)
            sev = self.vrs_state
            self.vrs_noise = 0.9 * self.vrs_noise + 0.1 * self.rng.standard_normal(self.n_units)
            T = T * (1.0 - 0.25 * sev + 0.15 * sev * self.vrs_noise)
            self.vrs_level = sev
        Tpos = np.maximum(T, 0.0)

        # moc: indukowana (teoria strumieniowa) + profilowa
        vax_pos = np.maximum(v_ax, 0.0)
        vi = -vax_pos / 2 + np.sqrt(vax_pos ** 2 / 4 + Tpos / (2 * rho * self.A))
        p_ind = Tpos * np.maximum(vax_pos + vi, 0.2 * vh)
        p_shaft = p_ind / self.eta + profile_factor * self.cp0 * rho * n ** 3 * self.D ** 5 * (self.n > 1.0)
        p_shaft = p_shaft * np.minimum(self.health * 10.0, 1.0)
        Q = p_shaft / (2 * math.pi * n)

        F_units = T[:, None] * self.axis
        if self.h_coeff > 0:
            F_units = F_units - self.h_coeff * (n * self.health)[:, None] * v_perp
        M_units = cross_rows(self.pos, F_units) - (self.spin * Q)[:, None] * self.axis
        self.thrust = T
        self.power_shaft = p_shaft
        p_elec = float(np.sum(p_shaft)) / self.motor_eff + 0.5 * self.n_units
        return F_units.sum(axis=0), M_units.sum(axis=0), p_elec
