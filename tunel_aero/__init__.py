"""Tunel-aero - wirtualne środowisko testowe dla dronów i samolotów w realistycznych warunkach.

Szybki start:
    from tunel_aero import run_scenario
    r = run_scenario("scenarios/multirotor/02_wiatr_porywisty_miasto.yaml")
    print(r.passed, r.metrics["max_track_error"])
"""
from .scenario import ScenarioResult, build_simulation, load_scenario, run_scenario

__version__ = "0.1.0"
__all__ = ["run_scenario", "load_scenario", "build_simulation", "ScenarioResult", "__version__"]
