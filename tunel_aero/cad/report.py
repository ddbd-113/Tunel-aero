"""Raport projektowy samolotu z CAD: rzuty z CG i punktem neutralnym, stateczność, opór, osiągi."""
from __future__ import annotations

import base64
import html
import io
import math
from pathlib import Path

import numpy as np
import yaml

from .aircraft import build_aircraft, vehicle_to_yaml_dict

DERIV_INFO = [
    ("CLa", "nachylenie siły nośnej CL_alfa", "1/rad", "> 0"),
    ("Cma", "stateczność podłużna Cm_alfa", "1/rad", "< 0"),
    ("Cmq", "tłumienie pochylania Cm_q", "", "< 0"),
    ("Cnb", "stateczność kierunkowa Cn_beta", "1/rad", "> 0"),
    ("Clb", "efekt wzniosu Cl_beta", "1/rad", "< 0"),
    ("Clp", "tłumienie przechylania Cl_p", "", "< 0"),
    ("Cnr", "tłumienie odchylania Cn_r", "", "< 0"),
    ("CYb", "siła boczna CY_beta", "1/rad", "< 0"),
    ("Clr", "Cl_r", "", ""), ("Cnp", "Cn_p", "", ""),
    ("CLde", "skuteczność steru wysokości CL_de", "1/rad", ""),
    ("Cmde", "skuteczność steru wysokości Cm_de", "1/rad", "< 0"),
    ("Clda", "skuteczność lotek Cl_da", "1/rad", "> 0"),
    ("Cnda", "odchylanie od lotek Cn_da (osie ciała)", "1/rad", ""),
    ("Cndr", "skuteczność steru kierunku Cn_dr", "1/rad", "< 0"),
    ("CL0", "CL przy alfa = 0", "", ""), ("Cm0", "Cm przy alfa = 0", "", ""),
]


def three_view_png(vehicle: dict, design: dict) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    vis = vehicle.get("_visual")
    cg = design["cg_body"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1.3, 1.3, 1.0]})
    views = (("Widok z góry", 1, 0, (1, 1)), ("Widok z boku", 0, 2, (-1, -1)), ("Widok z przodu", 1, 2, (1, -1)))
    A, B = design["panels"]["A"], design["panels"]["B"]
    for ax, (title, i, j, (si, sj)) in zip(axs, views):
        if vis is not None:
            v = vis["v"].astype(float) + cg
            tri = v[vis["f"]]
            pts = np.stack((si * tri[:, :, i], sj * tri[:, :, j]), axis=-1)
            ax.add_collection(PolyCollection(pts, facecolor="#cfd6df", edgecolor="#9aa6b2", linewidths=0.15,
                                             alpha=0.8))
        for a, b in zip(A, B):
            ax.plot([si * a[i], si * b[i]], [sj * a[j], sj * b[j]], color="#1f6fb2", lw=0.4, alpha=0.6)
        np_b = cg + np.array([design["derivatives"]["Cma"] / design["derivatives"]["CLa"] * design["mac"], 0, 0])
        ax.plot(si * cg[i], sj * cg[j], "o", ms=11, mfc="white", mec="k", mew=2, label="środek ciężkości")
        ax.plot(si * cg[i], sj * cg[j], "k+", ms=11)
        if i == 1 and j == 0 or (i == 0 and j == 2):
            ax.plot(si * np_b[i], sj * np_b[j], "^", ms=10, color="#c0392b", label="punkt neutralny")
            if i == 1 and j == 0:
                c = design["mac"]
                x_np = np_b[0]
                ax.axhspan(x_np + 0.05 * c, x_np + 0.15 * c, color="#2e8b57", alpha=0.15,
                           label="zalecany zakres CG (5-15% MAC)")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.autoscale()
        ax.grid(alpha=0.25)
        ax.set_xticks([])
        ax.set_yticks([])
    axs[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def design_html(vehicle: dict, design: dict, imgs_extra: list | None = None, facts_extra: list | None = None) -> str:
    from ..report import CSS
    d = design["derivatives"]
    u = design["units"]
    cg, npt = design["cg_cad"], design["np_cad"]
    rng = design["cg_range_cad"]
    sm = design["static_margin"]
    tr = design.get("trim")
    cls = {"error": "bad", "warn": "bad", "info": "muted", "ok": "ok"}
    icon = {"error": "BŁĄD", "warn": "uwaga", "info": "info", "ok": "OK"}
    checks = "".join(f"<tr><td class='{cls[l]}'>{icon[l]}</td><td>{html.escape(m)}</td></tr>" for l, m in design["checks"])
    ok = not any(l == "error" for l, _ in design["checks"])
    kv = [
        (f"{design['mass']:.3f} kg", "masa"), (f"{design['b'] * 1000:.0f} mm", "rozpiętość"),
        (f"{design['S'] * 100:.1f} dm²", "powierzchnia skrzydła"), (f"{design['mac'] * 1000:.0f} mm", "średnia cięciwa (MAC)"),
        (f"{design['AR']:.1f}", "wydłużenie"), (f"{design['wing_loading_g_dm2']:.0f} g/dm²", "obciążenie powierzchni"),
        (f"{sm * 100:.1f}% MAC", "zapas stateczności"), (f"{design['cg_pct_mac'] * 100:.0f}% MAC", "położenie CG"),
        (f"{design['np_pct_mac'] * 100:.0f}% MAC", "punkt neutralny"),
        (f"{design['v_stall']:.1f} m/s", "prędkość przeciągnięcia"), (f"{design['cruise_speed']:.1f} m/s", "prędkość przelotowa"),
        (f"{design['CLmax']:.2f}", "CL max"), (f"{design['CD0']:.4f}", "CD0 (opór pasożytniczy)"),
        (f"{design.get('static_thrust_to_weight', 0):.2f}", "ciąg statyczny / ciężar"),
        (f"{design['Vh']:.2f} / {design['Vv']:.3f}", "objętość usterzenia V_h / V_v"),
    ]
    if tr is not None:
        kv += [(f"{math.degrees(tr.alpha):.1f}° / {math.degrees(tr.elevator):.1f}°", "trym: kąt natarcia / ster wys."),
               (f"{tr.throttle * 100:.0f}% · {tr.power:.0f} W", "trym: przepustnica / moc")]
    kvs = "".join(f"<div class='kv'><b>{a}</b><span>{b}</span></div>" for a, b in kv + (facts_extra or []))
    der = "".join(f"<tr><td>{html.escape(lbl)}</td><td class='num'>{d[k]:.4f}</td><td>{unit}</td><td>{exp}</td></tr>"
                  for k, lbl, unit, exp in DERIV_INFO if k in d)
    cdp = "".join(f"<tr><td>{html.escape(k)}</td><td class='num'>{v:.4f}</td><td class='num'>{v / design['CD0'] * 100:.0f}%</td></tr>"
                  for k, v in design["cd_parts"].items())
    surf = "".join(f"<tr><td>{r}</td><td class='num'>{s['area'] * 100:.2f}</td><td class='num'>{s['span'] * 1000:.0f}</td>"
                   f"<td class='num'>{s['mac'] * 1000:.0f}</td><td class='num'>{s['thickness'] * 100:.1f}%</td></tr>"
                   for r, s in design["surfaces"].items())
    ixx, iyy, izz, ixz = design["inertia"]
    img3 = base64.b64encode(three_view_png(vehicle, design)).decode()
    extra = "".join(f"<div class='card'><img src='data:image/png;base64,{i}'></div>" for i in (imgs_extra or []))
    return f"""<!doctype html><html lang="pl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Projekt - {html.escape(design['name'])}</title>
<style>{CSS}</style></head><body><main>
<h1>{html.escape(design['name'])}</h1>
<p><span class="badge {'pass' if ok else 'fail'}">{'model gotowy do lotu' if ok else 'wymaga poprawek'}</span></p>
<h2>Kontrola projektu</h2><div class="card"><table>{checks}</table></div>
<h2>Wyważenie</h2><div class="card"><table>
<tr><th>środek ciężkości (CAD, {u})</th><td class="num">x = {cg[0]:.1f}, y = {cg[1]:.1f}, z = {cg[2]:.1f}</td></tr>
<tr><th>punkt neutralny (CAD, {u})</th><td class="num">x = {npt[0]:.1f}</td></tr>
<tr><th>zalecany zakres CG (5-15% MAC zapasu)</th><td class="num">x = {rng[1][0]:.1f} ... {rng[0][0]:.1f} {u}</td></tr>
<tr><th>momenty bezwładności [kg m²]</th><td class="num">Ixx = {ixx:.4f}, Iyy = {iyy:.4f}, Izz = {izz:.4f}, Ixz = {ixz:.4f}</td></tr>
</table></div>
<div class="card"><img src="data:image/png;base64,{img3}"></div>
<h2>Najważniejsze parametry</h2><div class="grid">{kvs}</div>
<h2>Powierzchnie nośne (z przekrojów siatki)</h2><div class="card"><table>
<tr><th>część</th><th>pole [dm²]</th><th>rozpiętość [mm]</th><th>MAC [mm]</th><th>grubość profilu</th></tr>{surf}</table></div>
<h2>Pochodne stateczności (metoda siatki wirowej VLM + poprawki kadłuba, osie ciała, CL przelotowy)</h2><div class="card"><table>
<tr><th>pochodna</th><th>wartość</th><th>jedn.</th><th>wymagany znak</th></tr>{der}</table></div>
<h2>Opór pasożytniczy (suma składników)</h2><div class="card"><table><tr><th>składnik</th><th>CD</th><th>udział</th></tr>{cdp}</table></div>
<h2>Osiągi (wirtualny tunel)</h2>{extra}
<p class="muted">Model aerodynamiczny: VLM (jak AVL/XFLR5) + półempiryczne poprawki kadłuba i oporu (Raymer).
Sprawdź wyniki lotami próbnymi - zwłaszcza położenie środka ciężkości.</p>
</main></body></html>"""


def write_design_report(cfg_path, out_dir) -> dict:
    from ..tunnel import tunnel_figures
    cfg_path = Path(cfg_path)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    vehicle, design = build_aircraft(cfg, base_dir=cfg_path.resolve().parent)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        _, imgs, _ = tunnel_figures(vehicle, vehicle["cruise_speed"])
    except Exception:  # noqa: BLE001
        imgs = []
    page = design_html(vehicle, design, imgs)
    (out / "projekt.html").write_text(page, encoding="utf-8")
    (out / "vehicle_generated.yaml").write_text(
        "# Wygenerowane automatycznie z " + cfg_path.name + " - model fixed_wing dla symulatora\n"
        + yaml.safe_dump(vehicle_to_yaml_dict(vehicle), allow_unicode=True, sort_keys=False), encoding="utf-8")
    return {"report": out / "projekt.html", "vehicle": out / "vehicle_generated.yaml", "design": design}
