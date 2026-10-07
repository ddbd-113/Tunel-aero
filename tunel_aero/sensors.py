"""Czujniki i nawigacja.

Dwa poziomy:
  1. Surowe czujniki (IMU, GPS, barometr, magnetometr, rurka Pitota) - z szumem, biasem,
     opóźnieniem i awariami; logowane i dostępne w API (np. do testowania własnego filtru).
  2. NavigationSystem - model błędów estymatora (EKF) w postaci "prawda + realistyczny błąd":
     dryf GPS, nawigacja zliczeniowa po utracie GPS, błąd wysokości barometrycznej przy
     niestandardowej temperaturze/zmianie ciśnienia, zakłócenia magnetometru, zablokowany Pitot.
     Autopilot steruje na podstawie TEGO estymatu, nie prawdy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .atmosphere import RHO0_ISA, pressure_altitude
from .math3d import euler_from_quat, quat_from_euler, quat_to_dcm

DEFAULT_SENSORS = {
    "imu": {"gyro_noise": 0.004, "gyro_bias": 0.003, "accel_noise": 0.08, "accel_bias": 0.05,
            "vibration": 0.6},
    "gps": {"rate": 5.0, "sigma_h": 1.2, "sigma_v": 2.0, "sigma_vel": 0.08, "tau": 30.0,
            "latency": 0.15},
    "baro": {"sigma": 0.3, "tau": 8.0, "noise": 0.08},
    "mag": {"sigma_deg": 1.0, "current_interference_deg_per_a": 0.015},
    "airspeed": {"sigma": 0.25, "noise": 0.3, "heater": False},
    "attitude": {"sigma_deg": 0.3, "tau": 20.0},
    "dead_reckoning": {"accel_bias": 0.08},
}


class GaussMarkov:
    """Proces Gaussa-Markowa 1. rzędu (dokładna dyskretyzacja)."""

    def __init__(self, sigma, tau, rng, size=1):
        self.sigma = np.full(size, sigma, dtype=float) if np.isscalar(sigma) else np.asarray(sigma, float)
        self.tau = tau
        self.rng = rng
        self.x = self.sigma * rng.standard_normal(size)

    def step(self, dt, scale=1.0):
        a = math.exp(-dt / self.tau)
        self.x = a * self.x + scale * self.sigma * math.sqrt(1 - a * a) * self.rng.standard_normal(self.x.shape)
        return self.x


@dataclass
class NavState:
    pos: np.ndarray
    vel: np.ndarray
    q: np.ndarray
    R: np.ndarray
    euler: tuple
    omega: np.ndarray
    alt: float               # wysokość nad startem (barometryczna) [m]
    climb_rate: float
    airspeed: float          # prędkość wskazywana IAS [m/s]
    accel_body: np.ndarray
    gps_ok: bool = True
    course: float = 0.0
    ground_speed: float = 0.0


@dataclass
class SensorFaults:
    gps_lost: bool = False
    gps_sigma_factor: float = 1.0
    gps_spoof_rate: np.ndarray = field(default_factory=lambda: np.zeros(2))
    mag_offset_deg: float = 0.0
    pitot_blocked: bool = False
    imu_noise_factor: float = 1.0
    link_lost: bool = False


def _merge(base: dict, over: dict | None) -> dict:
    out = {k: dict(v) for k, v in base.items()}
    for k, v in (over or {}).items():
        out.setdefault(k, {}).update(v)
    return out


class NavigationSystem:
    def __init__(self, cfg: dict | None, rng: np.random.Generator, mode: str = "realistic"):
        self.c = _merge(DEFAULT_SENSORS, cfg)
        self.mode = mode
        self.rng = rng
        c = self.c
        self.gps_err = GaussMarkov([c["gps"]["sigma_h"]] * 2 + [c["gps"]["sigma_v"]], c["gps"]["tau"], rng, 3)
        self.vel_err = GaussMarkov(c["gps"]["sigma_vel"], 2.0, rng, 3)
        self.baro_err = GaussMarkov(c["baro"]["sigma"], c["baro"]["tau"], rng)
        self.att_err = GaussMarkov([math.radians(c["attitude"]["sigma_deg"])] * 2
                                   + [math.radians(c["mag"]["sigma_deg"])], c["attitude"]["tau"], rng, 3)
        self.as_err = GaussMarkov(c["airspeed"]["sigma"], 10.0, rng)
        self.gyro_bias = c["imu"]["gyro_bias"] * rng.standard_normal(3)
        self.acc_bias = c["imu"]["accel_bias"] * rng.standard_normal(3)
        self.dr_bias = c["dead_reckoning"]["accel_bias"] * rng.standard_normal(2)
        self.faults = SensorFaults()
        self.dr_pos = np.zeros(2)       # błąd nawigacji zliczeniowej
        self.dr_vel = np.zeros(2)
        self.spoof = np.zeros(2)
        self.p_home = None
        self.pitot_q = 0.0
        self.alt_f = None
        self.ias_f = None
        self.gps_timer = 0.0
        self.raw: dict = {}
        self.was_lost = False
        self.pitot_iced = False

    def update(self, t, dt, x, vehicle, env) -> NavState:
        pos, vel, q, w = x[0:3], x[3:6], x[6:10], x[10:13]
        f = self.faults
        rng = self.rng
        R_true = quat_to_dcm(q)
        v_air_b = R_true.T @ (vel - env.wind.total)
        tas = float(np.linalg.norm(v_air_b))
        atm = env.atm
        if self.p_home is None:
            # kalibracja barometru na ziemi przed startem (ciśnienie przy ziemi)
            self.p_home = env.atm.pressure * math.exp(-9.80665 * (-env.h_agl) / (287.05 * atm.temperature)) \
                if env.h_agl > 0.5 else atm.pressure
        sf = vehicle.specific_force

        # ---------------- surowe czujniki (logowanie / API)
        ci = self.c["imu"]
        vib = ci["vibration"] * (float(np.mean(vehicle.props.n)) / 150.0 if hasattr(vehicle, "props") else 0.0)
        nf = f.imu_noise_factor
        acc_meas = sf + self.acc_bias + (ci["accel_noise"] * nf + vib) * rng.standard_normal(3)
        gyro_meas = w + self.gyro_bias + ci["gyro_noise"] * nf * 10 * rng.standard_normal(3)
        p_meas = atm.pressure * (1 + 1e-5 * self.c["baro"]["noise"] * rng.standard_normal())
        baro_alt = pressure_altitude(p_meas) - pressure_altitude(self.p_home)
        # Pitot: ciśnienie dynamiczne -> prędkość wskazywana (IAS) przy gęstości wzorcowej
        self.pitot_iced = env.deg.ice > 0.5 and not self.c["airspeed"]["heater"]
        if f.pitot_blocked or self.pitot_iced:
            self.pitot_q *= math.exp(-dt / 5.0)    # zatkany wlot, odpływ drożny -> wskazanie spada do 0
        else:
            self.pitot_q = 0.5 * atm.density * tas * tas
        ias_meas = math.sqrt(2 * max(self.pitot_q, 0.0) / RHO0_ISA)
        self.gps_timer += dt
        gps_valid = not f.gps_lost
        if f.gps_spoof_rate.any():
            self.spoof += f.gps_spoof_rate * dt
        gps_e = self.gps_err.step(dt, f.gps_sigma_factor)
        roll, pitch, yaw = euler_from_quat(q)
        mag_dist = math.radians(f.mag_offset_deg + self.c["mag"]["current_interference_deg_per_a"]
                                * (vehicle.battery.current if vehicle.battery else 0.0))
        self.raw = {
            "imu_acc": acc_meas, "imu_gyro": gyro_meas, "baro_alt": baro_alt, "airspeed": ias_meas,
            "mag_heading": yaw + mag_dist + math.radians(self.c["mag"]["sigma_deg"]) * rng.standard_normal(),
            "gps_valid": gps_valid,
        }
        if self.gps_timer >= 1.0 / self.c["gps"]["rate"]:
            self.gps_timer = 0.0
            if gps_valid:
                lat_shift = vel * self.c["gps"]["latency"]
                gp = pos - lat_shift + gps_e
                gp[:2] += self.spoof
                self.raw["gps_pos"] = gp
                self.raw["gps_vel"] = vel + self.c["gps"]["sigma_vel"] * rng.standard_normal(3)

        if self.mode == "perfect":
            return self._make(pos, vel, q, w, -pos[2], -vel[2], ias_meas, sf, True)

        # ---------------- model błędu estymatora
        ev = self.vel_err.step(dt)
        if gps_valid:
            target_err = gps_e[:2] + self.spoof
            spoof_v = np.array([*f.gps_spoof_rate, 0.0])
            if self.was_lost:
                # po odzyskaniu GPS filtr zbiega do pomiaru (skok pozycji jak w prawdziwym EKF)
                k = min(1.0, dt / 1.5)
                self.dr_pos += (target_err - self.dr_pos) * k
                self.dr_vel *= (1 - k)
                if np.linalg.norm(self.dr_pos - target_err) < 0.3:
                    self.was_lost = False
                pos_err_h = self.dr_pos
                vel_err = ev + np.array([*self.dr_vel, 0.0]) + spoof_v
            else:
                pos_err_h = target_err
                vel_err = ev + spoof_v
        else:
            if not self.was_lost:
                self.was_lost = True
                self.dr_pos = gps_e[:2] + self.spoof
                self.dr_vel = ev[:2].copy()
            att_tilt_err = 9.80665 * np.array([self.att_err.x[1], -self.att_err.x[0]])
            self.dr_vel += (self.dr_bias + att_tilt_err + 0.05 * rng.standard_normal(2)) * dt
            self.dr_pos += self.dr_vel * dt
            pos_err_h = self.dr_pos
            vel_err = np.array([*self.dr_vel, ev[2]])

        be = self.baro_err.step(dt)[0]
        alt_est = baro_alt + be
        self.alt_f = alt_est if self.alt_f is None else self.alt_f + (alt_est - self.alt_f) * min(1.0, dt / 0.1)
        pos_est = np.array([pos[0] + pos_err_h[0], pos[1] + pos_err_h[1], -self.alt_f])

        ae = self.att_err.step(dt)
        q_est = quat_from_euler(roll + ae[0], pitch + ae[1], yaw + ae[2] + mag_dist)
        omega_est = w + 0.1 * self.gyro_bias + 0.2 * ci["gyro_noise"] * nf * rng.standard_normal(3)
        ias = max(0.0, ias_meas + self.as_err.step(dt)[0] + self.c["airspeed"]["noise"] * rng.standard_normal())
        self.ias_f = ias if self.ias_f is None else self.ias_f + (ias - self.ias_f) * min(1.0, dt / 0.15)
        acc_est = sf + self.acc_bias + 0.05 * rng.standard_normal(3)
        return self._make(pos_est, vel + vel_err, q_est, omega_est, self.alt_f, -(vel[2] + vel_err[2]),
                          self.ias_f, acc_est, gps_valid)

    @staticmethod
    def _make(pos, vel, q, w, alt, climb, ias, acc, gps_ok) -> NavState:
        R = quat_to_dcm(q)
        return NavState(pos=pos, vel=vel, q=q, R=R, euler=euler_from_quat(q), omega=w, alt=alt,
                        climb_rate=climb, airspeed=ias, accel_body=acc, gps_ok=gps_ok,
                        course=math.atan2(vel[1], vel[0]), ground_speed=math.hypot(vel[0], vel[1]))
