"""Wykresy (matplotlib) i raport HTML z wynikami testu."""
from __future__ import annotations

import base64
import html
import io
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C = {"true": "#1f6fb2", "est": "#8a8f98", "sp": "#d9822b", "wind": "#2a9d8f", "bad": "#c0392b",
     "ok": "#2e8b57", "a": "#6a4c93", "b": "#e76f51", "c": "#264653"}


def _waypoints(sc: dict) -> list[tuple[float, float, float]]:
    out = []
    for w in (sc.get("mission", {}) or {}).get("waypoints", []) or []:
        out.append((w["north"], w["east"], w["alt"]) if isinstance(w, dict) else (w[0], w[1], w[2]))
    return out


def _event_lines(ax, events, t_max):
    for t, msg in events:
        if t > t_max:
            continue
        crit = any(k in msg for k in ("AWARIA", "KATASTROFA", "FAILSAFE", "PRZECIĄGNIĘCIE", "Pitota"))
        if crit:
            ax.axvline(t, color=C["bad"], lw=0.8, ls="--", alpha=0.6)


def _fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def plot_overview(result) -> bytes:
    d, sc, ev = result.log.data, result.scenario, result.log.events
    kind = result.log.meta.get("vehicle_kind")
    t = d["t"]
    fig, axs = plt.subplots(3, 2, figsize=(13, 11))
    fig.suptitle(result.name, fontsize=13, fontweight="bold")

    # 1. widok z góry
    ax = axs[0, 0]
    ax.plot(d["pe"], d["pn"], color=C["true"], lw=1.6, label="tor rzeczywisty")
    ax.plot(d["est_pe"], d["est_pn"], color=C["est"], lw=0.9, ls="--", label="estymata autopilota")
    wps = _waypoints(sc)
    if wps:
        wp = np.array(wps)
        ax.plot(wp[:, 1], wp[:, 0], "s", color=C["sp"], ms=7, label="punkty trasy")
        for i, w in enumerate(wps):
            ax.annotate(str(i + 1), (w[1], w[0]), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.plot(d["pe"][0], d["pn"][0], "o", color=C["ok"], ms=8, label="start")
    ax.plot(d["pe"][-1], d["pn"][-1], "X" if result.metrics["crashed"] else "o",
            color=C["bad"] if result.metrics["crashed"] else "k", ms=10, label="koniec")
    for mb in ((sc.get("environment", {}) or {}).get("wind", {}) or {}).get("microbursts", []):
        ax.add_patch(plt.Circle((mb.get("east", 0), mb.get("north", 0)), mb.get("radius", 600),
                                color=C["bad"], alpha=0.12, label="mikroburst"))
    for th in ((sc.get("environment", {}) or {}).get("wind", {}) or {}).get("thermals", []):
        ax.add_patch(plt.Circle((th.get("east", 0), th.get("north", 0)), th.get("radius", 80),
                                color=C["sp"], alpha=0.15))
    wn, we = np.mean(d["wind_n"]), np.mean(d["wind_e"])
    ws = math.hypot(wn, we)
    if ws > 0.3:
        xl, yl = ax.get_xlim(), ax.get_ylim()
        L = 0.12 * max(xl[1] - xl[0], yl[1] - yl[0])
        x0, y0 = xl[0] + 0.12 * (xl[1] - xl[0]), yl[1] - 0.12 * (yl[1] - yl[0])
        ax.annotate("", xy=(x0 + we / ws * L, y0 + wn / ws * L), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=C["wind"], lw=2))
        ax.text(x0, y0 - 0.05 * (yl[1] - yl[0]), f"wiatr śr. {ws:.1f} m/s", color=C["wind"], fontsize=8)
    ax.set_xlabel("Wschód [m]")
    ax.set_ylabel("Północ [m]")
    ax.set_title("Trajektoria (widok z góry)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.3)

    # 2. wysokość
    ax = axs[0, 1]
    ax.plot(t, d["alt"], color=C["true"], label="rzeczywista AGL")
    ax.plot(t, d["est_alt"], color=C["est"], ls="--", lw=0.9, label="barometryczna (autopilot)")
    if "h_sp" in d:
        ax.plot(t, d["h_sp"], color=C["sp"], lw=0.9, label="zadana")
    ax.axhline(0, color="k", lw=0.8)
    _event_lines(ax, ev, t[-1])
    ax.set_title("Wysokość")
    ax.set_ylabel("[m]")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # 3. prędkości
    ax = axs[1, 0]
    ax.plot(t, d["tas"], color=C["true"], label="prędkość powietrzna TAS")
    if kind == "fixed_wing":
        ax.plot(t, d["est_ias"], color=C["a"], lw=0.9, label="IAS wskazywana (Pitot)")
        ax.plot(t, d["v_sp"], color=C["sp"], lw=0.9, ls=":", label="zadana")
    ax.plot(t, d["gs"], color=C["c"], lw=0.9, label="prędkość względem ziemi")
    ax.plot(t, d["wind_speed"], color=C["wind"], lw=0.9, label="wiatr (poziomy)")
    _event_lines(ax, ev, t[-1])
    ax.set_title("Prędkości")
    ax.set_ylabel("[m/s]")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # 4. orientacja
    ax = axs[1, 1]
    ax.plot(t, d["roll"], color=C["true"], lw=0.9, label="przechylenie")
    ax.plot(t, d["pitch"], color=C["b"], lw=0.9, label="pochylenie")
    if "roll_sp" in d:
        ax.plot(t, d["roll_sp"], color=C["true"], lw=0.6, ls=":", label="przechylenie zad.")
        ax.plot(t, d["pitch_sp"], color=C["b"], lw=0.6, ls=":", label="pochylenie zad.")
    _event_lines(ax, ev, t[-1])
    ax.set_title("Orientacja")
    ax.set_ylabel("[deg]")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # 5. bateria
    ax = axs[2, 0]
    ax.plot(t, d["batt_soc"] * 100, color=C["ok"], label="stan naładowania [%]")
    ax.set_ylabel("SOC [%]")
    ax.set_ylim(0, 105)
    ax2 = ax.twinx()
    ax2.plot(t, d["batt_v"], color=C["a"], lw=0.8, label="napięcie [V]")
    ax2.set_ylabel("U [V]")
    ax.set_title(f"Akumulator (T: {d['batt_temp'][0]:.0f} -> {d['batt_temp'][-1]:.0f} C)")
    ax.set_xlabel("czas [s]")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower left")
    ax.grid(alpha=0.3)

    # 6. sterowanie
    ax = axs[2, 1]
    if kind == "multirotor":
        i = 1
        while f"m{i}" in d:
            ax.plot(t, d[f"m{i}"], lw=0.7, label=f"silnik {i}")
            i += 1
        ax.set_ylim(0, 1.05)
        ax.set_title("Wysterowanie silników")
    else:
        ax.plot(t, d["aileron"], lw=0.8, label="lotki [deg]")
        ax.plot(t, d["elevator"], lw=0.8, label="ster wys. [deg]")
        ax.plot(t, d["rudder"], lw=0.8, label="ster kier. [deg]")
        ax2 = ax.twinx()
        ax2.plot(t, d["throttle"], color="k", lw=0.8, label="przepustnica")
        ax2.set_ylim(0, 1.05)
        ax.set_title("Stery i przepustnica")
    _event_lines(ax, ev, t[-1])
    ax.set_xlabel("czas [s]")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _fig_to_png(fig)


def plot_environment(result) -> bytes:
    d, ev = result.log.data, result.log.events
    t = d["t"]
    fig, axs = plt.subplots(2, 2, figsize=(13, 7))
    fig.suptitle("Warunki środowiskowe w trakcie lotu", fontsize=12, fontweight="bold")
    ax = axs[0, 0]
    ax.plot(t, d["wind_mean"], color=C["wind"], label="wiatr średni (profil wysokości)")
    ax.plot(t, d["turb"], color=C["a"], lw=0.7, label="|turbulencja|")
    ax.plot(t, d["gust"], color=C["b"], lw=0.9, label="|podmuchy|")
    ax.plot(t, d["updraft"], color=C["sp"], lw=0.9, label="prąd pionowy (+ w górę)")
    if np.any(d["downburst_h"] > 0.01):
        ax.plot(t, d["downburst_h"], color=C["bad"], lw=0.9, label="wypływ mikroburstu")
    _event_lines(ax, ev, t[-1])
    ax.set_title("Wiatr")
    ax.set_ylabel("[m/s]")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax = axs[0, 1]
    ax.plot(t, d["temp_c"], color=C["b"], label="temperatura powietrza [C]")
    ax.plot(t, d["batt_temp"], color=C["a"], lw=0.8, label="temperatura akumulatora [C]")
    ax.set_ylabel("[C]")
    ax2 = ax.twinx()
    ax2.plot(t, d["density_alt"], color=C["c"], lw=0.8, ls="--", label="wysokość gęstościowa [m]")
    ax2.set_ylabel("[m]")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7)
    ax.set_title("Temperatura i gęstość powietrza")
    ax.grid(alpha=0.3)
    ax = axs[1, 0]
    ax.plot(t, d["power"], color=C["c"], lw=0.8, label="moc elektryczna [W]")
    ax.set_ylabel("[W]")
    ax2 = ax.twinx()
    ax2.plot(t, d["batt_i"], color=C["b"], lw=0.6, alpha=0.7, label="prąd [A]")
    ax2.set_ylabel("[A]")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7)
    ax.set_title("Pobór mocy")
    ax.set_xlabel("czas [s]")
    ax.grid(alpha=0.3)
    ax = axs[1, 1]
    ax.plot(t, d["ice"] * 100, color=C["true"], label="oblodzenie [%]")
    ax.plot(t, d["rain"], color=C["wind"], label="opad [mm/h]")
    nav_err = np.hypot(d["est_pn"] - d["pn"], d["est_pe"] - d["pe"])
    ax.plot(t, nav_err, color=C["bad"], lw=0.8, label="błąd nawigacji poziomej [m]")
    gps_off = d["gps_ok"] < 0.5
    if gps_off.any():
        ax.fill_between(t, 0, max(1.0, float(np.max(nav_err))), where=gps_off, color=C["bad"], alpha=0.1,
                        label="brak GPS")
    ax.set_title("Pogoda i nawigacja")
    ax.set_xlabel("czas [s]")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _fig_to_png(fig)


# ------------------------------------------------------------------ HTML
CSS = """
:root{--bg:#f6f7f9;--card:#fff;--fg:#1d232b;--muted:#5d6672;--line:#e1e5ea;--ok:#1e7d4f;--bad:#b3261e;--accent:#1f6fb2}
@media (prefers-color-scheme: dark){:root{--bg:#14171c;--card:#1d2128;--fg:#e8ebef;--muted:#9aa3ae;--line:#2c323b;--ok:#4cc38a;--bad:#ff6b5e;--accent:#6cb0f0}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 10px}
.muted{color:var(--muted)}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:12px 0}
.badge{display:inline-block;padding:4px 12px;border-radius:999px;font-weight:700;color:#fff}
.pass{background:var(--ok)}.fail{background:var(--bad)}
table{border-collapse:collapse;width:100%}td,th{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:600}td.num{font-variant-numeric:tabular-nums}
.ok{color:var(--ok);font-weight:700}.bad{color:var(--bad);font-weight:700}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:8px}
.kv{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 10px}
.kv b{display:block;font-size:17px}.kv span{color:var(--muted);font-size:12px}
img{max-width:100%;border-radius:8px;background:#fff}
a{color:var(--accent)}ul.ev{list-style:none;padding:0;margin:0;max-height:340px;overflow:auto}
ul.ev li{padding:3px 0;border-bottom:1px dashed var(--line);font-variant-numeric:tabular-nums}
li.crit{color:var(--bad)}
"""

KEY_METRICS = [
    ("flight_time", "czas lotu", "s", 0), ("distance_km", "dystans", "km", 2),
    ("energy_wh", "zużyta energia", "Wh", 1), ("final_soc", "bateria na końcu", "%", -1),
    ("min_voltage", "min. napięcie", "V", 2), ("max_battery_temp", "maks. temp. baterii", "C", 1),
    ("max_track_error", "maks. odchyłka od trasy", "m", 1), ("p95_track_error", "odchyłka (p95)", "m", 1),
    ("max_tilt_deg", "maks. pochylenie", "deg", 1), ("min_altitude", "min. wysokość w misji", "m", 1),
    ("max_wind", "maks. wiatr", "m/s", 1), ("max_turbulence", "maks. turbulencja", "m/s", 1),
    ("max_updraft", "maks. noszenie", "m/s", 1), ("max_downdraft", "maks. opadanie powietrza", "m/s", 1),
    ("max_ice", "maks. oblodzenie", "%", -1), ("max_nav_error", "maks. błąd nawigacji", "m", 1),
    ("max_baro_error", "maks. błąd wysokości baro", "m", 1), ("max_density_alt", "wysokość gęstościowa", "m", 0),
    ("saturation_time", "czas nasycenia silników", "s", 1), ("mean_hover_power", "moc w locie (mediana)", "W", 0),
    ("landing_error", "błąd miejsca lądowania", "m", 1), ("min_airspeed", "min. IAS", "m/s", 1),
    ("max_airspeed", "maks. IAS", "m/s", 1), ("stall_time", "czas w przeciągnięciu", "s", 1),
    ("max_load_factor", "maks. przeciążenie", "g", 2), ("max_altitude_error", "maks. błąd wysokości", "m", 1),
]


def _fmt(v, dec):
    if isinstance(v, bool):
        return "TAK" if v else "NIE"
    if v is None:
        return "-"
    if dec == -1:
        return f"{v * 100:.0f}"
    return f"{v:.{dec}f}"


def write_html_report(result, out_dir: Path, images: dict[str, bytes], replay_name: str | None) -> Path:
    m = result.metrics
    expect = result.scenario.get("expect_pass", True)
    verdict = "PASS" if result.passed else "FAIL"
    rows = []
    for c in result.criteria:
        v = c["value"]
        vs = _fmt(v, 2) if not isinstance(v, bool) else ("TAK" if v else "NIE")
        thr = c["threshold"]
        ts = ("TAK" if thr else "-") if isinstance(thr, bool) else f"{thr}"
        cls = "ok" if c["passed"] else "bad"
        rows.append(f"<tr><td>{html.escape(c['label'])}</td><td class='num'>{vs}</td><td class='num'>{ts}</td>"
                    f"<td class='{cls}'>{'OK' if c['passed'] else 'NIE'}</td></tr>")
    kvs = []
    for key, label, unit, dec in KEY_METRICS:
        if key in m:
            kvs.append(f"<div class='kv'><b>{_fmt(m[key], dec)} {unit}</b><span>{label}</span></div>")
    evs = []
    for t, msg in result.log.events:
        crit = any(k in msg for k in ("AWARIA", "KATASTROFA", "FAILSAFE", "PRZECIĄGNIĘCIE", "Pitota", "WYCZERPANA"))
        evs.append(f"<li class='{'crit' if crit else ''}'>t = {t:7.2f} s &nbsp; {html.escape(msg)}</li>")
    imgs = "".join(f"<div class='card'><img alt='{k}' src='data:image/png;base64,"
                   f"{base64.b64encode(v).decode()}'></div>" for k, v in images.items())
    note = ""
    if not expect:
        note = ("<p class='muted'>Scenariusz demonstracyjny - zaprojektowany tak, aby pokazać granice pojazdu "
                f"(oczekiwany wynik: FAIL). {'Wynik zgodny z oczekiwaniem.' if not result.passed else 'Pojazd poradził sobie lepiej niż zakładano.'}</p>")
    replay = (f"<p><a href='{replay_name}'>Otwórz odtwarzacz 3D lotu</a> &middot; "
              f"<a href='log.csv'>pobierz log CSV</a></p>") if replay_name else "<p><a href='log.csv'>log CSV</a></p>"
    desc = html.escape(str(result.scenario.get("description", ""))).strip()
    meta = result.log.meta
    doc = f"""<!doctype html><html lang="pl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Raport testu - {html.escape(result.name)}</title>
<style>{CSS}</style></head><body><main>
<h1>{html.escape(result.name)}</h1>
<p class="muted">{desc}</p>
<p><span class="badge {'pass' if result.passed else 'fail'}">{verdict}</span>
&nbsp; zakończenie: <b>{html.escape(meta.get('end_reason', ''))}</b> &middot; pojazd: {html.escape(meta.get('vehicle', ''))}
&middot; ziarno losowe: {result.seed} &middot; czas obliczeń: {meta.get('wall_time', 0):.1f} s</p>
{note}{replay}
<h2>Kryteria zaliczenia</h2><div class="card"><table><tr><th>kryterium</th><th>wynik</th><th>próg</th><th></th></tr>
{''.join(rows)}</table></div>
<h2>Najważniejsze wielkości</h2><div class="grid">{''.join(kvs)}</div>
<h2>Zdarzenia</h2><div class="card"><ul class="ev">{''.join(evs) or '<li>brak</li>'}</ul></div>
<h2>Wykresy</h2>{imgs}
<p class="muted">Wygenerowano przez Tunel-aero - wirtualne środowisko testowe dla dronów i samolotów.</p>
</main></body></html>"""
    path = out_dir / "raport.html"
    path.write_text(doc, encoding="utf-8")
    return path


def generate_report(result, out_dir, viewer: bool = True) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.log.to_csv(out / "log.csv")
    imgs = {"przeglad": plot_overview(result), "srodowisko": plot_environment(result)}
    for k, v in imgs.items():
        (out / f"{k}.png").write_bytes(v)
    replay = None
    if viewer:
        from .viewer import write_replay
        replay = write_replay(result, out / "replay.html").name
    rep = write_html_report(result, out, imgs, replay)
    return {"report": rep, "csv": out / "log.csv", "replay": out / replay if replay else None}
