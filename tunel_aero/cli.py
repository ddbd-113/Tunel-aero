"""Interfejs wiersza poleceń: tunel-aero <komenda> ...

  run SCENARIUSZ        - pojedynczy lot testowy + raport HTML + odtwarzacz 3D
  suite [KATALOG...]    - zestaw scenariuszy (test regresyjny), tabela PASS/FAIL
  montecarlo KONFIG     - wiele lotów z losowymi warunkami
  tunnel POJAZD         - wirtualny tunel aerodynamiczny i analiza osiągów
  list [KATALOG]        - lista dostępnych scenariuszy
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml

from .scenario import PACKAGE_ROOT, load_yaml, run_scenario


def slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.replace("ł", "l").replace("Ł", "L")).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()[:60] or "wynik"


def _parse_set(items: list[str] | None) -> dict:
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--set wymaga formatu klucz=wartość, otrzymano: {it}")
        k, v = it.split("=", 1)
        out[k.strip()] = yaml.safe_load(v)
    return out


def _find_scenarios(paths: list[str]) -> list[Path]:
    if not paths:
        paths = [str(PACKAGE_ROOT / "scenarios")]
    files = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            files += sorted(f for f in pp.rglob("*.yaml") if "montecarlo" not in f.parts)
        else:
            files.append(pp)
    return files


def _print_result(r, verbose=True):
    expect = r.scenario.get("expect_pass", True)
    mark = "PASS" if r.passed else "FAIL"
    print(f"\n  {r.name}\n  wynik: {mark}  ({r.log.meta.get('end_reason')})"
          + ("" if expect else "   [scenariusz demonstracyjny - oczekiwano FAIL]"))
    for c in r.criteria:
        v = c["value"]
        vs = (("tak" if v else "nie") if isinstance(v, bool) else f"{v:.2f}") if v is not None else "-"
        print(f"    {'[OK]' if c['passed'] else '[XX]'} {c['label']}: {vs}  (próg: {c['threshold']})")
    if verbose:
        print("  zdarzenia:")
        for t, msg in r.log.events[:30]:
            print(f"    t={t:7.2f} s  {msg}")
        if len(r.log.events) > 30:
            print(f"    ... (+{len(r.log.events) - 30})")


def cmd_run(a):
    sc_path = Path(a.scenario)
    overrides = _parse_set(a.set)
    if a.duration:
        overrides["duration"] = a.duration
    if a.perfect_nav:
        overrides["navigation"] = "perfect"
    t0 = time.time()
    r = run_scenario(sc_path, seed=a.seed, overrides=overrides, progress=not a.quiet)
    _print_result(r, verbose=not a.quiet)
    if not a.no_report:
        from .report import generate_report
        out = Path(a.out) if a.out else Path("wyniki") / slug(sc_path.stem)
        files = generate_report(r, out, viewer=not a.no_viewer)
        print(f"\n  raport:     {files['report']}")
        if files["replay"]:
            print(f"  odtwarzacz: {files['replay']}")
        print(f"  log CSV:    {files['csv']}")
    print(f"  czas: {time.time() - t0:.1f} s")
    return 0 if (r.passed or not r.scenario.get("expect_pass", True)) else 1


def _suite_job(args):
    path, seed = args
    r = run_scenario(path, seed=seed)
    return {"path": str(path), "name": r.name, "passed": r.passed, "expect": r.scenario.get("expect_pass", True),
            "end": r.log.meta.get("end_reason", ""), "failed": [c["name"] for c in r.criteria if not c["passed"]],
            "time": r.metrics["flight_time"]}


def cmd_suite(a):
    files = _find_scenarios(a.paths)
    print(f"Zestaw testów: {len(files)} scenariuszy")
    t0 = time.time()
    jobs = [(f, a.seed) for f in files]
    if a.jobs and a.jobs > 1:
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            res = list(ex.map(_suite_job, jobs))
    else:
        res = []
        for j in jobs:
            print(f"  ... {j[0].name}", flush=True)
            res.append(_suite_job(j))
    print(f"\n{'wynik':6s} {'zgodny':7s} scenariusz")
    ok_all = True
    for r in res:
        as_expected = r["passed"] == r["expect"]
        ok_all &= as_expected
        tag = "PASS" if r["passed"] else "FAIL"
        extra = f"  <- niespełnione: {', '.join(r['failed'])}" if r["failed"] else ""
        print(f"{tag:6s} {'tak' if as_expected else 'NIE':7s} {r['name']}  [{r['end']}]{extra}")
    n_ok = sum(r["passed"] == r["expect"] for r in res)
    print(f"\n{n_ok}/{len(res)} scenariuszy z wynikiem zgodnym z oczekiwaniem ({time.time() - t0:.0f} s)")
    return 0 if ok_all else 1


def cmd_montecarlo(a):
    from .montecarlo import run_montecarlo
    out = Path(a.out) if a.out else Path("wyniki") / ("mc_" + slug(Path(a.config).stem))
    s = run_montecarlo(a.config, out, runs=a.runs, jobs=a.jobs)
    lo, hi = s["ci95"]
    print(f"\n  {s['name']}")
    print(f"  zaliczone: {s['passed']}/{s['runs']} = {s['pass_rate'] * 100:.0f}%  (95% CI: {lo * 100:.0f}-{hi * 100:.0f}%)")
    print(f"  raport: {out / 'montecarlo.html'}")
    return 0


def cmd_tunnel(a):
    from .scenario import set_dotted
    from .tunnel import write_tunnel_report
    cfg = load_yaml(a.vehicle)
    for k, v in _parse_set(a.set).items():
        set_dotted(cfg, k, v)
    out = Path(a.out) if a.out else Path("wyniki") / ("tunel_" + slug(Path(a.vehicle).stem))
    p = write_tunnel_report(cfg, out, a.speed)
    print(f"  raport tunelu: {p}")
    return 0


def cmd_list(a):
    for f in _find_scenarios(a.paths):
        try:
            sc = load_yaml(f)
        except Exception as e:  # noqa: BLE001
            print(f"{f}: błąd ({e})")
            continue
        exp = "" if sc.get("expect_pass", True) else "  [demonstracja porażki]"
        try:
            shown = f.resolve().relative_to(Path.cwd())
        except ValueError:
            shown = f
        print(f"{shown}\n    {sc.get('name', '')}{exp}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="tunel-aero",
                                description="Tunel-aero: wirtualne środowisko testowe dla dronów i samolotów")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="uruchom scenariusz testowy")
    r.add_argument("scenario")
    r.add_argument("--seed", type=int)
    r.add_argument("--out", help="katalog wyników (domyślnie wyniki/<scenariusz>)")
    r.add_argument("--set", action="append", metavar="KLUCZ=WARTOŚĆ",
                   help="nadpisz parametr, np. --set environment.wind.speed=12")
    r.add_argument("--duration", type=float)
    r.add_argument("--perfect-nav", action="store_true", help="nawigacja idealna (bez błędów czujników)")
    r.add_argument("--no-report", action="store_true")
    r.add_argument("--no-viewer", action="store_true")
    r.add_argument("-q", "--quiet", action="store_true")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("suite", help="uruchom zestaw scenariuszy")
    s.add_argument("paths", nargs="*")
    s.add_argument("--jobs", type=int, default=1)
    s.add_argument("--seed", type=int)
    s.set_defaults(func=cmd_suite)

    m = sub.add_parser("montecarlo", help="testy Monte Carlo")
    m.add_argument("config")
    m.add_argument("--runs", type=int)
    m.add_argument("--jobs", type=int)
    m.add_argument("--out")
    m.set_defaults(func=cmd_montecarlo)

    t = sub.add_parser("tunnel", help="wirtualny tunel aerodynamiczny / osiągi pojazdu")
    t.add_argument("vehicle")
    t.add_argument("--speed", type=float, help="prędkość przelotowa do trymu [m/s]")
    t.add_argument("--set", action="append", metavar="KLUCZ=WARTOŚĆ")
    t.add_argument("--out")
    t.set_defaults(func=cmd_tunnel)

    ls = sub.add_parser("list", help="lista scenariuszy")
    ls.add_argument("paths", nargs="*")
    ls.set_defaults(func=cmd_list)

    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
