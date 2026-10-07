"""Matematyka 3D: kwaterniony, macierze obrotu, kąty Eulera.

Konwencje (standard lotniczy):
  * układ świata: NED (North-East-Down), początek w punkcie startu na ziemi,
  * układ ciała: FRD (Forward-Right-Down),
  * kwaternion q = [w, x, y, z] (Hamilton) opisuje obrót ciało -> NED,
    tzn. v_ned = R(q) @ v_body.
"""
from __future__ import annotations

import math

import numpy as np

GRAVITY = 9.80665


def quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    """Macierz obrotu ciało -> NED."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def dcm_to_quat(R: np.ndarray) -> np.ndarray:
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    if q[0] < 0:
        q = -q
    return quat_normalize(q)


def quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Kąty Eulera ZYX (yaw, pitch, roll) -> kwaternion."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def euler_from_quat(q: np.ndarray) -> tuple[float, float, float]:
    """Kwaternion -> (roll, pitch, yaw) [rad]."""
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sp = 2 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sp)))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def quat_integrate(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    """Dokładne całkowanie kwaternionu przy stałej prędkości kątowej (układ ciała)."""
    wx, wy, wz = omega
    n = math.sqrt(wx * wx + wy * wy + wz * wz)
    if n < 1e-12:
        return q
    half = 0.5 * n * dt
    s = math.sin(half) / n
    dq = np.array([math.cos(half), wx * s, wy * s, wz * s])
    return quat_normalize(quat_mul(q, dq))


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def cross3(a, b) -> np.ndarray:
    """Szybki iloczyn wektorowy dla wektorów 3D (np.cross jest wielokrotnie wolniejszy)."""
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def cross_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Iloczyn wektorowy wierszami: A (n,3) lub (3,) x B (n,3) lub (3,)."""
    ax, ay, az = A[..., 0], A[..., 1], A[..., 2]
    bx, by, bz = B[..., 0], B[..., 1], B[..., 2]
    return np.stack((ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx), axis=-1)


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def norm3(v) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def wind_from_to_ned(speed: float, direction_from_deg: float) -> tuple[float, float]:
    """Wiatr meteorologiczny (kierunek, Z KTÓREGO wieje) -> składowe N, E wektora prędkości powietrza."""
    a = math.radians(direction_from_deg)
    return -speed * math.cos(a), -speed * math.sin(a)
