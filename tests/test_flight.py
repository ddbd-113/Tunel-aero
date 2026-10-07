import math
from pathlib import Path

import numpy as np
import pytest

from tunel_aero import run_scenario, tunnel
from tunel_aero.metrics import evaluate_criteria
from tunel_aero.scenario import build_simulation, load_scenario, load_yaml
from tunel_aero.vehicles import build_vehicle

ROOT = Path(__file__).resolve().parent.parent
QUAD = ROOT / "vehicles/quad_x500.yaml"
PLANE = ROOT / "vehicles/fixed_wing_3kg.yaml"
SCENARIOS = sorted(p for p in (ROOT / "scenarios").rglob("*.yaml") if "montecarlo" not in p.parts)


def test_free_fall_without_thrust():
    sc = {"vehicle": str(QUAD), "duration": 1.0, "navigation": "perfect",
          "mission": {"start": {"altitude": 200}, "waypoints": [[0, 0, 200]]}}
    sim, _ = build_simulation(load_scenario(sc)[0])
    sim.vehicle.props.n[:] = 0.0
    sim.external_cmd = np.zeros(4)
    sim.step()
    sim.step()
    acc = (sim.vehicle.dynamics(sim.x, sim.env_sample))[3:6]
    assert acc[2] == pytest.approx(9.80665, rel=1e-3)


def test_quad_hover_power_is_realistic():
    h = tunnel.multirotor_hover(build_vehicle(load_yaml(QUAD)))
    assert 200 < h["power"] < 300
    assert 2.5 < h["thrust_to_weight"] < 3.3
    hot_high = tunnel.multirotor_hover(build_vehicle(load_yaml(QUAD)), altitude=4000, temperature_offset=25)
    assert hot_high["power"] > 1.2 * h["power"]


def test_fixed_wing_trim_and_stall():
    veh = build_vehicle(load_yaml(PLANE))
    tr = tunnel.trim_fixed_wing(veh, 17.0)
    assert tr.feasible
    assert 0 < math.degrees(tr.alpha) < 5
    assert 9.0 < tunnel.stall_speed(veh) < 10.5
    # cięższy samolot przeciąga się przy większej prędkości
    assert tunnel.stall_speed(veh, mass=4.0) > tunnel.stall_speed(veh)


def test_quad_holds_position_in_calm_air():
    sc = {"vehicle": str(QUAD), "duration": 25, "navigation": "perfect",
          "mission": {"takeoff_altitude": 10, "waypoints": [[0, 0, 10, 30]], "land": False}}
    r = run_scenario(sc)
    d = r.log.data
    late = d["t"] > 15
    assert not r.metrics["crashed"]
    assert np.max(np.abs(d["alt"][late] - 10.1)) < 0.5
    assert np.max(np.hypot(d["pn"][late], d["pe"][late])) < 0.5


def test_plane_holds_altitude_in_calm_air():
    sc = {"vehicle": str(PLANE), "duration": 40, "navigation": "perfect",
          "mission": {"airspeed": 17, "start": {"altitude": 100}, "waypoints": [[2000, 0, 100]]}}
    r = run_scenario(sc)
    d = r.log.data
    assert not r.metrics["crashed"]
    assert np.max(np.abs(d["alt"] - 100)) < 3
    assert np.max(np.abs(d["ias"] - 17)) < 1.5


def test_criteria_evaluation():
    m = {"crashed": False, "mission_complete": True, "max_track_error": 3.0, "final_soc": 0.4}
    res = evaluate_criteria(m, {"no_crash": True, "max_track_error": 2.0, "min_battery_soc": 0.3})
    assert [c["passed"] for c in res] == [True, False, True]
    with pytest.raises(ValueError):
        evaluate_criteria(m, {"nieznane": 1})


@pytest.mark.parametrize("path", SCENARIOS, ids=lambda p: p.stem)
def test_scenario_smoke(path):
    """Każdy scenariusz musi się wczytać i przelecieć kilka sekund bez błędów."""
    r = run_scenario(path, overrides={"duration": 4})
    assert len(r.log.data["t"]) > 10
    assert np.all(np.isfinite(r.log.data["pn"]))


def test_cli_list_and_run(tmp_path):
    from tunel_aero.cli import main
    assert main(["list"]) == 0
    rc = main(["run", str(ROOT / "scenarios/multirotor/01_spokojny_lot.yaml"), "--duration", "5", "-q",
               "--out", str(tmp_path)])
    assert rc in (0, 1)
    assert (tmp_path / "raport.html").exists() and (tmp_path / "replay.html").exists()


def test_gym_api_direct_control():
    from tunel_aero.gym_env import FlightEnv
    env = FlightEnv(ROOT / "scenarios/multirotor/01_spokojny_lot.yaml", overrides={"duration": 3},
                    include_truth=True)
    obs, _ = env.reset(seed=0)
    assert set(obs) >= {"sensors", "nav", "truth"}
    for _ in range(50):
        obs, rew, term, trunc, info = env.step(np.full(4, 0.6))
    assert obs["truth"]["pos"][2] < -0.2   # wzniósł się
