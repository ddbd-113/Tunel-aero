"""Przykład: własny regulator zawisu dla quadrocoptera podpięty przez API FlightEnv.

Regulator PD wysokości + PD orientacji na podstawie estymaty nawigacyjnej (jak na prawdziwym
dronie - nie zna prawdziwego stanu). Lot w porywistym wietrze miejskim ze scenariusza 02.

    python examples/wlasny_regulator.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tunel_aero.gym_env import FlightEnv  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
env = FlightEnv(ROOT / "scenarios/multirotor/02_wiatr_porywisty_miasto.yaml", action_mode="direct",
                control_rate=100, overrides={"duration": 40, "mission.waypoints": [[0, 0, 15]]})
obs, info = env.reset(seed=1)
target = env.target                      # [N, E, D] - zawis 15 m nad startem
model = env.sim.vehicle.controller_model()
m, g = model["mass"], 9.81
P = model["positions"]
# macierz alokacji [ciąg, moment x, moment y, moment z]
B = np.vstack([np.ones(len(P)), -P[:, 1], P[:, 0], model["spins"] * model["k_q"]])
Bp = np.linalg.pinv(B)

total = 0.0
for k in range(4000):
    nav = obs["nav"]
    pos, vel = nav["pos"], nav["vel"]
    roll, pitch, yaw = nav["euler"]
    p, q, r = nav["omega"]
    # pozycja -> zadane przyspieszenia
    acc = 0.8 * (target - pos) - 1.6 * vel
    acc[:2] = np.clip(acc[:2], -4, 4)
    thrust = m * (g - acc[2]) / max(np.cos(roll) * np.cos(pitch), 0.5)
    # przyspieszenie poziome -> zadane kąty (dla yaw ~ 0)
    pitch_sp = np.clip(-acc[0] / g, -0.5, 0.5)
    roll_sp = np.clip(acc[1] / g, -0.5, 0.5)
    tau = model["J"] @ np.array([
        60 * (roll_sp - roll) - 12 * p,
        60 * (pitch_sp - pitch) - 12 * q,
        10 * (0.0 - yaw) - 4 * r,
    ])
    t_motor = np.clip(Bp @ np.concatenate(([thrust], tau)), 0, model["t_max"])
    throttle = np.sqrt(t_motor / model["t_max"])
    obs, reward, terminated, truncated, info = env.step(throttle)
    total += reward
    if k % 500 == 0:
        print(f"t={info['t']:5.1f} s  wysokość={obs['nav']['alt']:5.1f} m  "
              f"poz. N/E=({pos[0]:5.1f}, {pos[1]:5.1f})  nagroda={reward:6.2f}")
    if terminated or truncated:
        break

print(f"\nkoniec: {info['reason'] or 'limit kroków'}  suma nagród={total:.0f}  katastrofa={info['crashed']}")
