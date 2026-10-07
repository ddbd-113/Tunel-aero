"""Testy Monte Carlo: wiele lotów z losowanymi warunkami i awariami.

Plik konfiguracyjny:
  base_scenario: ../multirotor/02_wiatr_porywisty_miasto.yaml
  runs: 30
  seed: 100
  randomize:                                   # klucz w notacji kropkowej -> rozkład
    environment.wind.speed: {uniform: [3, 13]}
    environment.temperature_c: {normal: [10, 8]}
    environment.wind.turbulence: {choice: [light, moderate, severe]}
  random_failures:
    - probability: 0.2
      failure: {type: gps_loss, t: {uniform: [30, 60]}, duration: {uniform: [5, 15]}}
"""
from __future__ import annotations

import base64
import copy
import csv
import html
import io
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .scenario import load_scenario, load_yaml, run_scenario, set_dotted, _resolve_path


def sample_value(spec, rng: np.random.Generator):
    if isinstance(spec, list):
        return [sample_value(v, rng) for v in spec]
    if not isinstance(spec, dict):
        return spec
    if "uniform" in spec:
        lo, hi = spec["uniform"]
        return float(rng.uniform(lo, hi))
    if "normal" in spec:
        mu, sd = spec["normal"]
        v = float(rng.normal(mu, sd))
        if "min" in spec:
            v = max(v, spec["min"])
        if "max" in spec:
            v = min(v, spec["max"])
        return v
    if "choice" in spec:
        opts = spec["choice"]
        return opts[int(rng.integers(len(opts)))]
    if "randint" in spec:
        lo, hi = spec["randint"]
        return int(rng.integers(lo, hi + 1))
    return {k: sample_value(v, rng) for k, v in spec.items()}


def make_runs(cfg: dict, base_dir: Path) -> tuple[dict, list[dict]]:
    base, _ = load_scenario(_resolve_path(cfg["base_scenario"], base_dir))
    n = int(cfg.get("runs", 20))
    rng = np.random.default_rng(cfg.get("seed", 0))
    runs = []
    for i in range(n):
        sc = copy.deepcopy(base)
        params = {}
        for key, spec in (cfg.get("randomize", {}) or {}).items():
            v = sample_value(spec, rng)
            set_dotted(sc, key, v)
            if isinstance(v, list) and all(isinstance(x, dict) for x in v):
                # lista obiektów (np. mikrobursty) - do analizy wrażliwości zapisz pola liczbowe osobno
                for j, item in enumerate(v):
                    for kk, vv in item.items():
                        if isinstance(spec[j].get(kk), dict):
                            params[f"{key.split('.')[-1]}[{j}].{kk}"] = vv
            else:
                params[key] = v
            if key.startswith("vehicle_overrides."):
                set_dotted(sc, "vehicle." + key[len("vehicle_overrides."):], v)
        fails = list(sc.get("failures", []) or [])
        for rf in cfg.get("random_failures", []) or []:
            if rng.random() < rf.get("probability", 0.1):
                f = sample_value(rf["failure"], rng)
                fails.append(f)
                params[f"awaria {f['type']} - czas [s]"] = round(f.get("t", 0.0), 1)
        sc["failures"] = fails
        sc.pop("vehicle_overrides", None)
        runs.append({"i": i, "seed": int(cfg.get("seed", 0)) + i, "scenario": sc, "params": params})
    return base, runs


def _run_one(job: dict) -> dict:
    res = run_scenario(job["scenario"], seed=job["seed"])
    d = res.log.data
    step = max(1, len(d["t"]) // 400)
    m = {k: v for k, v in res.metrics.items() if isinstance(v, (int, float, bool, str))}
    return {"i": job["i"], "seed": job["seed"], "params": job["params"], "metrics": m, "passed": res.passed,
            "failed_criteria": [c["name"] for c in res.criteria if not c["passed"]],
            "pn": d["pn"][::step].tolist(), "pe": d["pe"][::step].tolist(), "end": res.log.meta.get("end_reason", "")}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def run_montecarlo(cfg_path, out_dir=None, runs: int | None = None, jobs: int | None = None,
                   progress: bool = True) -> dict:
    cfg_path = Path(cfg_path)
    cfg = load_yaml(cfg_path)
    if runs:
        cfg["runs"] = runs
    base, jobs_list = make_runs(cfg, cfg_path.resolve().parent)
    jobs = jobs or max(1, (os.cpu_count() or 2) - 1)
    results = []
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            for r in ex.map(_run_one, jobs_list):
                results.append(r)
                if progress:
                    print(f"\r  Monte Carlo: {len(results)}/{len(jobs_list)}", end="", flush=True)
    else:
        for j in jobs_list:
            results.append(_run_one(j))
            if progress:
                print(f"\r  Monte Carlo: {len(results)}/{len(jobs_list)}", end="", flush=True)
    if progress:
        print()
    k = sum(r["passed"] for r in results)
    lo, hi = wilson(k, len(results))
    summary = {"name": cfg.get("name", base.get("name", "Monte Carlo")), "runs": len(results), "passed": k,
               "pass_rate": k / max(len(results), 1), "ci95": (lo, hi), "results": results, "config": cfg,
               "base": base}
    if out_dir:
        write_mc_report(summary, Path(out_dir))
    return summary


def write_mc_report(s: dict, out: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .report import CSS

    out.mkdir(parents=True, exist_ok=True)
    res = s["results"]
    pkeys = sorted({k for r in res for k in r["params"]})
    mkeys = ["max_track_error", "max_tilt_deg", "final_soc", "energy_wh", "min_altitude", "landing_error",
             "min_airspeed", "max_load_factor", "saturation_time", "flight_time"]
    mkeys = [k for k in mkeys if any(k in r["metrics"] for r in res)]
    with open(out / "montecarlo.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "seed", "passed", "failed_criteria", "end"] + pkeys + mkeys)
        for r in res:
            w.writerow([r["i"], r["seed"], r["passed"], ";".join(r["failed_criteria"]), r["end"]]
                       + [r["params"].get(k, "") for k in pkeys] + [r["metrics"].get(k, "") for k in mkeys])

    imgs = []

    def png(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=105, bbox_inches="tight")
        plt.close(fig)
        imgs.append(base64.b64encode(buf.getvalue()).decode())

    fig, ax = plt.subplots(figsize=(7.5, 7))
    for r in res:
        ax.plot(r["pe"], r["pn"], color="#2e8b57" if r["passed"] else "#c0392b", lw=0.8, alpha=0.6)
        ax.plot(r["pe"][-1], r["pn"][-1], "o" if r["passed"] else "x", color="k", ms=3)
    ax.set_title("Trajektorie wszystkich przebiegów (zielone = PASS, czerwone = FAIL)")
    ax.set_xlabel("Wschód [m]")
    ax.set_ylabel("Północ [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    png(fig)

    num_p = [k for k in pkeys if all(isinstance(r["params"].get(k, 0.0), (int, float)) for r in res)]
    if num_p and mkeys:
        cols = min(3, len(num_p))
        rows_ = math.ceil(len(num_p) / cols)
        fig, axs = plt.subplots(rows_, cols, figsize=(5 * cols, 3.8 * rows_), squeeze=False)
        target = mkeys[0]
        for a, k in zip(axs.flat, num_p):
            xs = [r["params"].get(k, np.nan) for r in res]
            ys = [r["metrics"].get(target, np.nan) for r in res]
            cs = ["#2e8b57" if r["passed"] else "#c0392b" for r in res]
            a.scatter(xs, ys, c=cs, s=22)
            a.set_xlabel(k.split(".")[-1])
            a.set_ylabel(target)
            a.grid(alpha=0.3)
        for a in list(axs.flat)[len(num_p):]:
            a.axis("off")
        fig.suptitle(f"Wrażliwość: parametry losowane vs {target}")
        fig.tight_layout()
        png(fig)
    if mkeys:
        cols = min(3, len(mkeys))
        rows_ = math.ceil(len(mkeys) / cols)
        fig, axs = plt.subplots(rows_, cols, figsize=(5 * cols, 3.2 * rows_), squeeze=False)
        for a, k in zip(axs.flat, mkeys):
            vals = [float(r["metrics"][k]) for r in res if k in r["metrics"]]
            a.hist(vals, bins=min(20, max(5, len(vals) // 2)), color="#1f6fb2", alpha=0.8)
            a.set_title(k, fontsize=9)
            a.grid(alpha=0.3)
        for a in list(axs.flat)[len(mkeys):]:
            a.axis("off")
        fig.suptitle("Rozkłady metryk")
        fig.tight_layout()
        png(fig)

    lo, hi = s["ci95"]
    rows = []
    for r in sorted(res, key=lambda r: (r["passed"], -r["metrics"].get(mkeys[0], 0) if mkeys else 0)):
        ps = ", ".join(f"{k.split('.')[-1]}={v:.2f}" if isinstance(v, float) else f"{k.split('.')[-1]}={v}"
                       for k, v in r["params"].items())
        rows.append(f"<tr><td>{r['i']}</td><td class='{'ok' if r['passed'] else 'bad'}'>{'PASS' if r['passed'] else 'FAIL'}</td>"
                    f"<td>{html.escape(', '.join(r['failed_criteria']))}</td><td>{html.escape(r['end'])}</td>"
                    f"<td>{html.escape(ps)}</td></tr>")
    doc = f"""<!doctype html><html lang="pl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Monte Carlo - {html.escape(s['name'])}</title>
<style>{CSS}</style></head><body><main>
<h1>{html.escape(s['name'])}</h1>
<p class="muted">Scenariusz bazowy: {html.escape(s['base'].get('name', ''))}</p>
<div class="grid">
<div class="kv"><b>{s['runs']}</b><span>przebiegów</span></div>
<div class="kv"><b>{s['pass_rate'] * 100:.0f}%</b><span>zaliczonych ({s['passed']}/{s['runs']})</span></div>
<div class="kv"><b>{lo * 100:.0f}-{hi * 100:.0f}%</b><span>przedział ufności 95% (Wilson)</span></div>
</div>
<h2>Wykresy</h2>{''.join(f"<div class='card'><img src='data:image/png;base64,{i}'></div>" for i in imgs)}
<h2>Przebiegi (najgorsze na górze)</h2><div class="card"><table>
<tr><th>#</th><th>wynik</th><th>niespełnione kryteria</th><th>zakończenie</th><th>wylosowane warunki</th></tr>
{''.join(rows)}</table></div>
<p class="muted">Pełne dane: montecarlo.csv</p></main></body></html>"""
    p = out / "montecarlo.html"
    p.write_text(doc, encoding="utf-8")
    return p
