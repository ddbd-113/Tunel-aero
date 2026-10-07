from __future__ import annotations


class PID:
    """Regulator PID z ograniczeniem całki (anti-windup) i pochodną z pomiaru."""

    def __init__(self, kp: float, ki: float = 0.0, kd: float = 0.0, i_limit: float = float("inf"),
                 out_limit: float = float("inf")):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.i_limit = i_limit
        self.out_limit = out_limit
        self.integral = 0.0
        self._prev = None

    def reset(self) -> None:
        self.integral = 0.0
        self._prev = None

    def update(self, error: float, dt: float, measurement: float | None = None,
               freeze_integral: bool = False) -> float:
        d = 0.0
        if self.kd and measurement is not None:
            if self._prev is not None and dt > 0:
                d = -(measurement - self._prev) / dt
            self._prev = measurement
        if not freeze_integral:
            self.integral += error * dt * self.ki
            self.integral = max(-self.i_limit, min(self.i_limit, self.integral))
        out = self.kp * error + self.integral + self.kd * d
        return max(-self.out_limit, min(self.out_limit, out))
