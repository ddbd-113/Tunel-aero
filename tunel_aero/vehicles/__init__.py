from .base import Vehicle
from .battery import Battery
from .fixed_wing import FixedWing
from .multirotor import Multirotor
from .propulsion import PropulsionSet


def build_vehicle(cfg: dict, rng=None) -> Vehicle:
    kind = cfg.get("type", "multirotor")
    if kind == "multirotor":
        return Multirotor(cfg, rng)
    if kind == "fixed_wing":
        return FixedWing(cfg, rng)
    raise ValueError(f"Nieznany typ pojazdu: {kind}")


__all__ = ["Vehicle", "Battery", "FixedWing", "Multirotor", "PropulsionSet", "build_vehicle"]
