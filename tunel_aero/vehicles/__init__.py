from .base import Vehicle
from .battery import Battery
from .fixed_wing import FixedWing
from .multirotor import Multirotor
from .propulsion import PropulsionSet


def resolve_vehicle_config(cfg: dict) -> dict:
    """Rozwija konfigurację samolotu z CAD (type: cad_aircraft) do modelu fixed_wing."""
    from ..cad.aircraft import build_aircraft, is_cad_aircraft
    if is_cad_aircraft(cfg):
        return build_aircraft(cfg, cfg.get("_base_dir"), cfg.get("_files"))[0]
    return cfg


def build_vehicle(cfg: dict, rng=None) -> Vehicle:
    cfg = resolve_vehicle_config(cfg)
    kind = cfg.get("type", "multirotor")
    if kind == "multirotor":
        return Multirotor(cfg, rng)
    if kind == "fixed_wing":
        return FixedWing(cfg, rng)
    raise ValueError(f"Nieznany typ pojazdu: {kind}")


__all__ = ["Vehicle", "Battery", "FixedWing", "Multirotor", "PropulsionSet", "build_vehicle",
           "resolve_vehicle_config"]
