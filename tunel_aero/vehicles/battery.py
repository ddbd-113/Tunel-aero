"""Model akumulatora Li-Po/Li-Ion: krzywa OCV(SOC), rezystancja wewnętrzna,
zależność pojemności i rezystancji od temperatury oraz bilans cieplny pakietu.
"""
from __future__ import annotations

import math

import numpy as np

# Typowa krzywa napięcia spoczynkowego ogniwa LiPo
_SOC = np.array([0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00])
_OCV = np.array([3.27, 3.61, 3.69, 3.73, 3.77, 3.79, 3.82, 3.87, 3.92, 3.98, 4.06, 4.20])


class Battery:
    def __init__(self, cells: int = 4, capacity_ah: float = 5.0, r_cell: float = 0.006,
                 mass: float = 0.55, soc: float = 1.0, temperature_c: float = 25.0,
                 cutoff_cell_voltage: float = 3.0, heat_transfer: float = 0.35,
                 specific_heat: float = 1000.0, insulated: bool = False):
        self.cells = cells
        self.capacity_ah = capacity_ah
        self.r_cell = r_cell
        self.mass = mass
        self.soc = soc
        self.temp_c = temperature_c
        self.cutoff = cutoff_cell_voltage
        self.hA = heat_transfer * (0.3 if insulated else 1.0)   # W/K
        self.cp = specific_heat                                 # J/(kg K)
        self.current = 0.0
        self.voltage = self.open_circuit_voltage()
        self.energy_used_wh = 0.0
        self.depleted = False
        self.capacity_factor_fault = 1.0
        self.resistance_factor_fault = 1.0

    # --- wpływ temperatury -------------------------------------------------
    def capacity_factor(self) -> float:
        """Zimny akumulator oddaje mniej energii (ok. 60% przy -20 C)."""
        if self.temp_c >= 25:
            return self.capacity_factor_fault
        return max(0.25, 1.0 - 0.009 * (25.0 - self.temp_c)) * self.capacity_factor_fault

    def resistance_factor(self) -> float:
        """Rezystancja wewnętrzna rośnie wykładniczo przy niskiej temperaturze (~3x przy -20 C)."""
        f = math.exp(0.025 * (25.0 - self.temp_c)) if self.temp_c < 25 else 1.0
        return f * self.resistance_factor_fault

    # -----------------------------------------------------------------------
    def open_circuit_voltage(self) -> float:
        return self.cells * float(np.interp(self.soc, _SOC, _OCV))

    @property
    def resistance(self) -> float:
        return self.cells * self.r_cell * self.resistance_factor()

    @property
    def nominal_voltage(self) -> float:
        return self.cells * 3.7

    def terminal_voltage(self, current: float) -> float:
        if self.depleted:
            return 0.0
        return max(0.0, self.open_circuit_voltage() - current * self.resistance)

    def current_for_power(self, power: float) -> float:
        """Prąd potrzebny do uzyskania mocy P na zaciskach: P = (E - I R) I."""
        if self.depleted:
            return 0.0
        E, R = self.open_circuit_voltage(), self.resistance
        disc = E * E - 4 * R * max(power, 0.0)
        if disc <= 0:
            return E / (2 * R)   # granica mocy - napięcie siada do połowy
        return (E - math.sqrt(disc)) / (2 * R)

    def update(self, dt: float, power: float, ambient_c: float) -> None:
        I = self.current_for_power(power)
        self.current = I
        self.voltage = self.terminal_voltage(I)
        cap_as = self.capacity_ah * 3600.0 * self.capacity_factor()
        self.soc = max(0.0, self.soc - I * dt / cap_as)
        self.energy_used_wh += self.voltage * I * dt / 3600.0
        # bilans cieplny: grzanie Joule'a - oddawanie ciepła do otoczenia
        q_gen = I * I * self.resistance
        self.temp_c += (q_gen - self.hA * (self.temp_c - ambient_c)) * dt / (self.mass * self.cp)
        if self.soc <= 0.0 or (self.voltage > 0 and self.voltage < self.cutoff * self.cells * 0.85):
            self.depleted = True
            self.voltage = 0.0
