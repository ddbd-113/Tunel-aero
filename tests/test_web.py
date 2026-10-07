import json
import zipfile
from pathlib import Path

from tunel_aero import webapi
from tunel_aero.webapp import build_site

ROOT = Path(__file__).resolve().parent.parent
EX = ROOT / "examples/moj_samolot"


def test_build_site(tmp_path):
    out = build_site(tmp_path / "site")
    for f in ("index.html", "app.js", "worker.js", "style.css", ".nojekyll", "data/examples/moj_samolot/skrzydlo.stl"):
        assert (out / f).exists(), f
    html = (out / "index.html").read_text(encoding="utf-8")
    assert "__VERSION__" not in html
    assert "__VERSION__" not in (out / "worker.js").read_text(encoding="utf-8")
    names = zipfile.ZipFile(out / "py/tunel_aero.zip").namelist()
    assert "tunel_aero/webapi.py" in names and "tunel_aero/cad/vlm.py" in names
    assert "scenarios/fixed_wing/01_patrol.yaml" in names and "examples/moj_samolot/aircraft.yaml" in names


def test_webapi_catalog_and_analysis():
    cat = json.loads(webapi.catalog())
    assert len(cat["scenarios"]) >= 16 and cat["example"]["files"]
    import yaml
    cfg = yaml.safe_load((EX / "aircraft.yaml").read_text(encoding="utf-8"))
    files = {n: (EX / n).read_bytes() for n in cat["example"]["files"]}
    a = json.loads(webapi.analyze_aircraft(json.dumps(cfg), files))
    assert 5 < a["summary"]["static_margin"] < 20
    assert len(a["mesh"]["f"]) % 3 == 0
    assert "fixed_wing" in a["vehicle_yaml"]


def test_webapi_run_with_cad_and_preset():
    import yaml
    cfg = yaml.safe_load((EX / "aircraft.yaml").read_text(encoding="utf-8"))
    files = {p.name: p.read_bytes() for p in EX.glob("*.stl")}
    calls = []
    r = json.loads(webapi.run_test("scenarios/fixed_wing/01_patrol.yaml", json.dumps({"duration": 15}),
                                   json.dumps(cfg), files, calls.append))
    assert "Trener" in r["vehicle"] and r["replay_html"].startswith("<!doctype html>")
    assert calls and calls[-1] == 1.0
    r2 = json.loads(webapi.run_test("scenarios/multirotor/01_spokojny_lot.yaml", json.dumps({"duration": 8}),
                                    json.dumps({"path": "vehicles/hexa_x.yaml"})))
    assert r2["kind"] == "multirotor" and "Hexa" in r2["vehicle"]
    assert webapi.last_csv().startswith("t,")
