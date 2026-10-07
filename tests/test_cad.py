import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from tunel_aero.cad import build_aircraft, load_mesh, write_stl
from tunel_aero.cad.mesh import Mesh, axes_matrix, parse_obj, parse_stl
from tunel_aero.cad.vlm import Control, Lattice, Section, Surface, stability_derivatives

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "examples/moj_samolot/aircraft.yaml"


def box_mesh(lx, ly, lz, c=(0, 0, 0)):
    v = np.array([[x, y, z] for x in (0, lx) for y in (0, ly) for z in (0, lz)], float) - [lx / 2, ly / 2, lz / 2] + c
    f = [[0, 2, 3], [0, 3, 1], [4, 5, 7], [4, 7, 6], [0, 1, 5], [0, 5, 4],
         [2, 6, 7], [2, 7, 3], [0, 4, 6], [0, 6, 2], [1, 3, 7], [1, 7, 5]]
    return Mesh(v, np.array(f))


def test_box_mass_properties():
    m = box_mesh(2.0, 1.0, 0.5, c=(1.0, 2.0, 3.0))
    assert abs(m.volume()) == pytest.approx(1.0)   # orientacja ścian bywa odwrócona w eksporcie CAD
    assert m.area() == pytest.approx(2 * (2 + 1 + 0.5))
    cg, C = m.mass_properties(3.0)
    assert np.allclose(cg, [1, 2, 3])
    Ccg = C - 3.0 * np.outer(cg, cg)
    I = np.trace(Ccg) * np.eye(3) - Ccg
    assert I[0, 0] == pytest.approx(3.0 / 12 * (1 + 0.25))
    assert I[2, 2] == pytest.approx(3.0 / 12 * (4 + 1))
    assert abs(I[0, 1]) < 1e-9


def test_stl_roundtrip_binary_and_ascii(tmp_path):
    m = box_mesh(1, 2, 3)
    write_stl(tmp_path / "b.stl", m)
    mb = load_mesh(tmp_path / "b.stl")
    assert mb.n_tri == 12 and abs(mb.volume()) == pytest.approx(6.0, rel=1e-5)
    lines = ["solid t"]
    for t in m.tri:
        lines += ["facet normal 0 0 0", "outer loop"] + [f"vertex {x} {y} {z}" for x, y, z in t] + ["endloop", "endfacet"]
    ma = parse_stl("\n".join(lines + ["endsolid t"]).encode())
    assert abs(ma.volume()) == pytest.approx(6.0)


def test_obj_objects():
    txt = "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\no skrzydlo\nf 1 2 3\no kadlub\nf 1 2 4\nf 1 3 4\n"
    assert parse_obj(txt.encode(), obj="kadlub").n_tri == 2
    assert parse_obj(txt.encode()).n_tri == 3


def test_axes_matrix():
    R = axes_matrix("-x", "+z")          # x do tyłu, z w górę (konwencja AVL)
    assert np.allclose(R @ [-1, 0, 0], [1, 0, 0])     # nos -> +x ciała
    assert np.allclose(R @ [0, 0, 1], [0, 0, -1])     # góra -> -z ciała (FRD)
    assert np.allclose(R @ [0, 1, 0], [0, 1, 0])      # prawe skrzydło
    R2 = axes_matrix("+z", "+y")         # np. Fusion 360 (Y w górę)
    assert np.isclose(np.linalg.det(R2), 1.0)


def test_decimation_keeps_shape():
    from tunel_aero.cad.mesh import load_mesh as lm
    m = lm(ROOT / "examples/moj_samolot/skrzydlo.stl")
    d = m.decimated(800)
    assert d.n_tri <= 800
    assert np.allclose(d.extent, m.extent, atol=0.05 * m.extent.max())


def test_vlm_elliptic_wing_matches_theory():
    s, AR = 1.0, 8.0
    c0 = 8 * s / (math.pi * AR)
    ys = s * np.sin(np.linspace(0, math.pi / 2, 25))
    secs = [Section(np.array([0.25 * c0 * math.sqrt(1 - (y / s) ** 2), y, 0.0]),
                    max(c0 * math.sqrt(max(1 - (y / s) ** 2, 0)), 1e-4)) for y in ys]
    S = math.pi * s * c0 / 2
    d = stability_derivatives(Lattice([Surface("wing", secs, n_span=20, n_chord=4)]), S, 2 * s, c0).values
    helmbold = 2 * math.pi * AR / (2 + math.sqrt(AR ** 2 + 4))
    assert d["CLa"] == pytest.approx(helmbold, rel=0.03)


def test_vlm_signs_conventional_aircraft():
    wing = Surface("wing", [Section(np.array([0.05, 0, 0]), 0.2), Section(np.array([0.05, 0.7, -0.03]), 0.15)],
                   controls=[Control("aileron", (0.5, 0.95), 0.25)], n_span=10, n_chord=4)
    ht = Surface("htail", [Section(np.array([-0.55, 0, 0]), 0.12), Section(np.array([-0.55, 0.22, 0]), 0.1)],
                 controls=[Control("elevator", (0, 1), 0.35)], n_span=5, n_chord=3)
    vt = Surface("vtail", [Section(np.array([-0.55, 0, 0]), 0.14), Section(np.array([-0.6, 0, -0.16]), 0.09)],
                 vertical=True, mirror=False, controls=[Control("rudder", (0, 1), 0.4)], n_span=5, n_chord=3)
    d = stability_derivatives(Lattice([wing, ht, vt]), 0.27, 1.4, 0.19, alpha0=math.radians(4)).values
    assert d["Cma"] < 0 and d["Cmq"] < 0             # stateczny podłużnie, tłumienie
    assert d["Cnb"] > 0 and d["Clb"] < 0             # kierunkowo stateczny, wznios
    assert d["Clp"] < 0 and d["Cnr"] < 0
    a = math.radians(4)
    cnda_stab = d["Cnda"] * math.cos(a) - d["Clda"] * math.sin(a)   # pochodne są w osiach ciała
    assert d["Clda"] > 0 and cnda_stab < 0           # lotka w prawo -> przechylenie w prawo, odwrotne odchylenie
    assert d["Cmde"] < 0 and d["CLde"] > 0           # ster wys. w dół -> nos w dół
    assert d["Cndr"] < 0 and d["CYdr"] > 0           # ster kier. w lewo -> nos w lewo


def test_example_aircraft_design():
    cfg = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    veh, des = build_aircraft(cfg, base_dir=EXAMPLE.parent)
    assert veh["type"] == "fixed_wing"
    assert 0.05 < des["static_margin"] < 0.2
    assert des["b"] == pytest.approx(1.39, abs=0.03)
    assert des["S"] == pytest.approx(0.287, rel=0.05)          # pole z przekrojów siatki
    assert 6.0 < des["v_stall"] < 9.0
    assert 0.02 < des["CD0"] < 0.05
    assert des["trim"].converged and des["trim"].feasible
    assert not any(level == "error" for level, _ in des["checks"])
    # przesunięcie akumulatora do tyłu zmniejsza zapas stateczności
    for c in cfg["mass"]["components"]:
        if c["name"] == "akumulator":
            c["position"] = [400, 0, 5]
    _, des2 = build_aircraft(cfg, base_dir=EXAMPLE.parent)
    assert des2["static_margin"] < des["static_margin"] - 0.05


def test_cad_aircraft_flies_scenario():
    from tunel_aero import run_scenario
    r = run_scenario(ROOT / "scenarios/fixed_wing/01_patrol.yaml",
                     overrides={"vehicle": str(EXAMPLE), "mission.airspeed": None, "duration": 40})
    assert not r.metrics["crashed"]
    assert r.metrics["max_altitude_error"] < 10
    assert r.scenario["vehicle"]["_visual"]["f"].shape[1] == 3


def test_cli_cad(tmp_path):
    from tunel_aero.cli import main
    assert main(["cad", str(EXAMPLE), "--out", str(tmp_path)]) == 0
    gen = yaml.safe_load((tmp_path / "vehicle_generated.yaml").read_text(encoding="utf-8"))
    assert gen["type"] == "fixed_wing" and "_visual" not in gen
    assert (tmp_path / "projekt.html").stat().st_size > 10000
