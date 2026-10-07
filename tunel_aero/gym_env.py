"""API w stylu Gymnasium (reset/step) - testowanie własnych regulatorów, estymatorów lub agentów RL
w realistycznym środowisku. Nie wymaga instalacji gymnasium.

Tryby akcji (action_mode):
  "autopilot" - steruje wbudowany autopilot; step(None) tylko przesuwa czas (obserwacja/rejestracja),
  "direct"    - własne sterowanie: multirotor -> przepustnice silników [0..1] (n wartości),
                samolot -> [lotki, ster wys., ster kier.] w [-1..1] (skalowane do wychyleń) + przepustnica [0..1].

Obserwacja (słownik):
  "sensors": surowe czujniki (IMU, baro, Pitot, magnetometr, GPS gdy dostępny),
  "nav":     estymata nawigacyjna (pozycja, prędkość, kąty) - jak w autopilocie,
  "truth":   stan rzeczywisty (tylko gdy include_truth=True - do uczenia/oceny, nie do sterowania!).
"""
from __future__ import annotations

import numpy as np

from .metrics import compute_metrics, evaluate_criteria
from .scenario import build_simulation, load_scenario


class FlightEnv:
    def __init__(self, scenario, action_mode: str = "direct", control_rate: float = 100.0,
                 include_truth: bool = False, reward_fn=None, overrides: dict | None = None):
        self.sc, _ = load_scenario(scenario, overrides)
        self.action_mode = action_mode
        self.control_rate = control_rate
        self.include_truth = include_truth
        self.reward_fn = reward_fn or self._default_reward
        self.sim = None

    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None):
        self.sim, _ = build_simulation(self.sc, seed)
        self.steps_per_action = max(1, int(round(1.0 / (self.control_rate * self.sim.dt))))
        self.sim.step()   # inicjalizacja środowiska i czujników
        return self._obs(), {"t": self.sim.t}

    @property
    def n_actions(self) -> int:
        v = self.sim.vehicle
        return v.props.n_units if v.kind == "multirotor" else 4

    def step(self, action=None):
        sim = self.sim
        if self.action_mode == "direct":
            a = np.asarray(action, dtype=float)
            if sim.vehicle.kind == "fixed_wing":
                lim = sim.vehicle.servo_limit
                cmd = np.concatenate((np.clip(a[:3], -1, 1) * lim, [np.clip(a[3], 0, 1)]))
            else:
                cmd = np.clip(a, 0.0, 1.0)
            sim.external_cmd = cmd
        alive = True
        for _ in range(self.steps_per_action):
            alive = sim.step()
            if not alive:
                break
        obs = self._obs()
        terminated = sim.vehicle.crashed or getattr(sim.ap, "phase", "") == "landed"
        truncated = (not alive) and not terminated
        reward = self.reward_fn(self, obs)
        info = {"t": sim.t, "crashed": sim.vehicle.crashed, "reason": sim.end_reason,
                "phase": getattr(sim.ap, "phase", "")}
        return obs, reward, terminated, truncated, info

    def evaluate(self) -> tuple[dict, list]:
        """Metryki i kryteria scenariusza dla dotychczasowego lotu."""
        log = self.sim.log
        log.meta.update(crashed=self.sim.vehicle.crashed, crash_reason=self.sim.vehicle.crash_reason,
                        vehicle_kind=self.sim.vehicle.kind, end_reason=self.sim.end_reason,
                        mission_complete=bool(getattr(self.sim.ap, "mission_complete", False)))
        log.finalize()
        m = compute_metrics(log, self.sim, self.sc)
        return m, evaluate_criteria(m, self.sc.get("criteria", {}))

    # ------------------------------------------------------------------
    def _obs(self) -> dict:
        sim = self.sim
        raw = sim.nav.raw
        nav = sim.nav_state
        obs = {
            "t": sim.t,
            "sensors": {
                "accel": np.array(raw.get("imu_acc", np.zeros(3))),
                "gyro": np.array(raw.get("imu_gyro", np.zeros(3))),
                "baro_alt": float(raw.get("baro_alt", 0.0)),
                "airspeed": float(raw.get("airspeed", 0.0)),
                "mag_heading": float(raw.get("mag_heading", 0.0)),
                "gps_valid": bool(raw.get("gps_valid", True)),
                "gps_pos": np.array(raw["gps_pos"]) if "gps_pos" in raw else None,
                "gps_vel": np.array(raw["gps_vel"]) if "gps_vel" in raw else None,
                "battery_voltage": sim.vehicle.battery.voltage,
            },
            "nav": {"pos": nav.pos.copy(), "vel": nav.vel.copy(), "euler": np.array(nav.euler),
                    "omega": nav.omega.copy(), "alt": nav.alt, "airspeed": nav.airspeed},
        }
        if self.include_truth:
            x = sim.x
            obs["truth"] = {"pos": x[0:3].copy(), "vel": x[3:6].copy(), "quat": x[6:10].copy(),
                            "omega": x[10:13].copy(), "wind": sim.env_sample.wind.total.copy()}
        return obs

    @property
    def target(self) -> np.ndarray:
        """Punkt docelowy dla trybu direct: pierwszy punkt trasy lub zawis na wysokości startu."""
        mis = self.sc.get("mission", {}) or {}
        wps = mis.get("waypoints") or []
        if wps:
            w = wps[0]
            n, e, a = (w["north"], w["east"], w["alt"]) if isinstance(w, dict) else w[:3]
            return np.array([n, e, -a], dtype=float)
        return np.array([0.0, 0.0, -mis.get("takeoff_altitude", 10.0)])

    @staticmethod
    def _default_reward(env, obs) -> float:
        """Domyślnie: kara za odległość od celu (direct) lub od trasy (autopilot) i za duże pochylenie."""
        sim = env.sim
        if sim.vehicle.crashed:
            return -100.0
        if env.action_mode == "direct":
            err = float(np.linalg.norm(sim.x[0:3] - env.target))
        else:
            err = sim.ap.true_track_error(sim.x[0:3]) if hasattr(sim.ap, "true_track_error") else 0.0
        roll, pitch, _ = obs["nav"]["euler"]
        return -0.1 * err - 0.5 * (abs(roll) + abs(pitch))
