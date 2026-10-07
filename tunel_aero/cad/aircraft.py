"""Budowa modelu symulacyjnego samolotu z danych CAD.

Wejście (YAML, `type: cad_aircraft`): pliki STL/OBJ poszczególnych części (skrzydło, usterzenie,
kadłub), jednostki i orientacja osi CAD, masa i jej rozkład (lub środek ciężkości/bezwładności
z CAD), profil, stery, napęd, akumulator. Wyjście: konfiguracja `fixed_wing` dla symulatora
oraz raport projektowy (środek ciężkości, punkt neutralny, zapas stateczności, trym, ostrzeżenia).
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .mesh import UNITS, Mesh, axes_matrix, load_mesh
from .vlm import Control, Lattice, Section, Surface, stability_derivatives

ROLES = ("wing", "htail", "vtail", "fuselage", "other")
RHO0, MU0 = 1.225, 1.789e-5

DEFAULT_AIRFOIL = {
    "wing": {"cl_max": 1.25, "zero_lift_alpha_deg": -2.5, "cm0": -0.05},
    "htail": {"cl_max": 1.0, "zero_lift_alpha_deg": 0.0, "cm0": 0.0},
    "vtail": {"cl_max": 1.0, "zero_lift_alpha_deg": 0.0, "cm0": 0.0},
}
DEFAULT_CONTROLS = {
    "wing": [{"type": "aileron", "span": [0.55, 0.95], "chord_fraction": 0.25, "max_deg": 20}],
    "htail": [{"type": "elevator", "span": [0.0, 1.0], "chord_fraction": 0.35, "max_deg": 25}],
    "vtail": [{"type": "rudder", "span": [0.0, 1.0], "chord_fraction": 0.4, "max_deg": 25}],
}

_CACHE: dict = {}


def is_cad_aircraft(cfg: dict) -> bool:
    return isinstance(cfg, dict) and cfg.get("type") in ("cad_aircraft", "aircraft")


# ------------------------------------------------------------------ narzędzia
def _parts(cfg: dict) -> dict:
    out = {}
    for name, p in (cfg.get("parts") or {}).items():
        if isinstance(p, str):
            p = {"file": p}
        p = dict(p)
        p.setdefault("role", name if name in ROLES else "other")
        out[name] = p
    return out


def _read(file: str, base_dir: Path | None, files: dict | None) -> Mesh:
    key = file.split("#")[0]
    if files is not None and (key in files or Path(key).name in files):
        data = files.get(key) or files.get(Path(key).name)
        return load_mesh(file, data)
    p = Path(file)
    if not p.is_absolute() and base_dir is not None:
        p = Path(base_dir) / file
    return load_mesh(str(p))


def _sections_from_mesh(m: Mesh, vertical: bool, n: int = 9) -> tuple[list[Section], float]:
    """Przekroje powierzchni nośnej płaszczyznami (y = const lub z = const). Zwraca też grubość względną."""
    lo, hi = m.bounds
    if vertical:
        axis, s_lo, s_hi = 2, lo[2], hi[2]
    else:
        axis = 1
        if lo[1] < -0.2 * hi[1]:          # pełne skrzydło (obie połówki) - bierzemy prawą
            s_lo, s_hi = 0.0, hi[1]
        elif hi[1] <= 1e-9:                # tylko lewa połówka
            m = Mesh(m.v * np.array([1, -1, 1]), m.f[:, ::-1], m.name)
            lo, hi = m.bounds
            s_lo, s_hi = lo[1], hi[1]
        else:
            s_lo, s_hi = lo[1], hi[1]
    span = s_hi - s_lo
    secs, thick = [], []
    ts = 0.5 * (1 - np.cos(np.linspace(0.0, math.pi, n)))
    for t in ts:
        val = s_lo + span * (0.004 + 0.986 * t)
        pts = m.section_points(axis, val)
        if len(pts) < 4:
            continue
        le = pts[np.argmax(pts[:, 0])]
        te = pts[np.argmin(pts[:, 0])]
        chord = float(math.hypot(le[0] - te[0], le[2] - te[2] if not vertical else le[1] - te[1]))
        if chord < 1e-4:
            continue
        if vertical:
            twist = 0.0
            le = np.array([le[0], float(np.mean(pts[:, 1])), val])
            thk = (pts[:, 1].max() - pts[:, 1].min()) / chord
        else:
            twist = math.atan2(te[2] - le[2], le[0] - te[0])
            d = np.array([-(le[0] - te[0]), 0.0, te[2] - le[2]]) / chord
            nrm = np.array([d[2], 0.0, -d[0]])
            rel = pts - le
            proj = rel @ nrm
            thk = (proj.max() - proj.min()) / chord
            le = np.array([le[0], val, le[2]])
        secs.append(Section(le, chord, twist))
        thick.append(thk)
    if not vertical and secs and secs[0].le[1] > 0.02 and secs[0].le[1] < 0.3 * span:
        s0 = secs[0]   # część skrzydła przykryta kadłubem - przedłużenie do osi symetrii
        secs.insert(0, Section(np.array([s0.le[0], 0.0, s0.le[2]]), s0.chord, s0.twist))
    if vertical:   # nasada usterzenia pionowego na dole (z największe w FRD)
        secs = sorted(secs, key=lambda s: -s.le[2])
    return secs, float(np.median(thick)) if thick else 0.12


def _planform(secs: list[Section], vertical: bool) -> dict:
    """Pole, rozpiętość, średnia cięciwa aerodynamiczna (dla jednej strony, mirror jeśli poziome)."""
    key = (lambda s: -s.le[2]) if vertical else (lambda s: abs(s.le[1]))
    secs = sorted(secs, key=key)
    area = mac2 = xle = 0.0
    for a, b in zip(secs[:-1], secs[1:]):
        dy = float(np.linalg.norm((b.le - a.le) * np.array([0.0, 1.0, 1.0])))
        area += 0.5 * (a.chord + b.chord) * dy
        mac2 += dy * (a.chord ** 2 + a.chord * b.chord + b.chord ** 2) / 3.0
        xle += dy * 0.5 * (a.le[0] * a.chord + b.le[0] * b.chord)
    span_one = abs(key(secs[-1]) - (0.0 if not vertical else key(secs[0])))
    if not vertical:
        span_one = abs(secs[-1].le[1])
    mac = mac2 / area if area > 0 else secs[0].chord
    k = 1.0 if vertical else 2.0
    return {"area": k * area, "span": k * span_one, "mac": mac,
            "x_mac_le": xle / area if area > 0 else secs[0].le[0], "sections": secs}


def _cf(re: float, laminar: float = 0.25) -> float:
    re = max(re, 1e4)
    lam = 1.328 / math.sqrt(re)
    turb = 0.455 / (math.log10(re) ** 2.58)
    return laminar * lam + (1 - laminar) * turb


def _prop(p: dict, R: np.ndarray, scale: float) -> dict:
    d = p.get("diameter") or p.get("diameter_in", 10) * 0.0254
    pitch = p.get("pitch") or p.get("pitch_in", 0.5 * d / 0.0254) * 0.0254
    pd = pitch / d
    pos = p.get("position")
    out = {
        "diameter": d, "ct0": p.get("ct0", 0.075 + 0.06 * pd), "j0": p.get("j0", 1.1 * pd + 0.1),
        "eta": p.get("eta", 0.82), "cp0": p.get("cp0", 0.005), "kv": p.get("kv", 1000),
        "kv_efficiency": p.get("kv_efficiency", 0.8), "motor_efficiency": p.get("motor_efficiency", 0.8),
        "tau": p.get("tau", 0.06), "spin": p.get("spin", 1),
    }
    return out, (R @ (np.asarray(pos, float) * scale) if pos is not None else None)


def _cache_key(cfg, base_dir, files) -> str:
    h = hashlib.sha1(json.dumps({k: v for k, v in cfg.items() if not k.startswith("_")},
                                sort_keys=True, default=str).encode())
    h.update(str(base_dir).encode())
    for p in _parts(cfg).values():
        f = p["file"].split("#")[0]
        if files is not None and (f in files or Path(f).name in files):
            h.update(hashlib.sha1(files.get(f) or files.get(Path(f).name)).digest())
        else:
            fp = Path(base_dir or ".") / f
            if fp.exists():
                h.update(str(fp.stat().st_mtime_ns).encode())
    return h.hexdigest()


# ------------------------------------------------------------------ budowa
def build_aircraft(cfg: dict, base_dir=None, files: dict | None = None) -> tuple[dict, dict]:
    """Zwraca (konfiguracja pojazdu fixed_wing, raport projektowy)."""
    base_dir = base_dir or cfg.get("_base_dir")
    key = _cache_key(cfg, base_dir, files)
    if key in _CACHE:
        v, d = _CACHE[key]
        return copy.deepcopy(v), d
    v, d = _build(cfg, Path(base_dir) if base_dir else None, files)
    _CACHE[key] = (copy.deepcopy(v), d)
    return v, d


def _build(cfg: dict, base_dir: Path | None, files: dict | None) -> tuple[dict, dict]:
    checks: list[tuple[str, str]] = []
    scale = UNITS[cfg.get("units", "mm")]
    ax = cfg.get("axes", {}) or {}
    R = axes_matrix(ax.get("forward", "-x"), ax.get("up", "+z"))
    to_body = (lambda p: R @ (np.asarray(p, float) * scale))

    # ---- części
    parts = _parts(cfg)
    meshes: dict[str, Mesh] = {}
    for name, p in parts.items():
        meshes[name] = _read(p["file"], base_dir, files).transformed(R, scale)
        if meshes[name].n_tri == 0:
            raise ValueError(f"Część '{name}' nie zawiera trójkątów")
    by_role = {}
    for name, p in parts.items():
        by_role.setdefault(p["role"], []).append(name)

    def role_mesh(role):
        names = by_role.get(role, [])
        if not names:
            return None
        m = meshes[names[0]]
        for n_ in names[1:]:
            m = m.merged(meshes[n_])
        return m

    # ---- masa, środek ciężkości, bezwładność
    mcfg = cfg.get("mass", {}) or {}
    comps = mcfg.get("components", []) or []
    m_comp = sum(c["mass"] for c in comps)
    part_mass = {n: float(p["mass"]) for n, p in parts.items() if "mass" in p}
    total = mcfg.get("total")
    if total is None:
        total = sum(part_mass.values()) + m_comp
    rest = total - m_comp - sum(part_mass.values())
    free = [n for n in parts if n not in part_mass]
    shell_default = cfg.get("structure_model", "shell") == "shell"
    if rest > 1e-6 and free:
        # pozostała masa konstrukcji: proporcjonalnie do powierzchni (powłoki) lub objętości (bryły)
        meas = {n: max(meshes[n].area() if parts[n].get("shell", shell_default) else abs(meshes[n].volume()), 1e-12)
                for n in free}
        sv = sum(meas.values())
        for n in free:
            part_mass[n] = rest * meas[n] / sv
    elif rest > 1e-6 and part_mass:
        k = (sum(part_mass.values()) + rest) / sum(part_mass.values())
        part_mass = {n: m_ * k for n, m_ in part_mass.items()}
        checks.append(("info", f"Masy części przeskalowane x{k:.2f}, aby zgadzały się z masą całkowitą"))
    elif rest < -1e-6:
        checks.append(("warn", f"Suma mas części i komponentów ({total - rest:.3f} kg) przekracza masę całkowitą "
                               f"({total:.3f} kg)"))
        total = total - rest
    first = np.zeros(3)
    C = np.zeros((3, 3))
    msum = 0.0
    for n, mm in part_mass.items():
        if mm <= 0:
            continue
        cg_p, C_p = meshes[n].mass_properties(mm, shell=bool(parts[n].get("shell", shell_default)))
        first += mm * cg_p
        C += C_p
        msum += mm
    for c in comps:
        r = to_body(c["position"])
        first += c["mass"] * r
        C += c["mass"] * np.outer(r, r)
        msum += c["mass"]
    if msum <= 0:
        raise ValueError("Podaj masę: mass.total i/lub masy części/komponentów")
    cg_calc = first / msum
    C_cg = C - msum * np.outer(cg_calc, cg_calc)
    I_tensor = np.trace(C_cg) * np.eye(3) - C_cg
    cg = to_body(mcfg["cg"]) if mcfg.get("cg") is not None else cg_calc
    if mcfg.get("cg") is not None and np.linalg.norm(cg - cg_calc) > 0.02:
        checks.append(("info", f"Podany środek ciężkości różni się od wyliczonego z rozkładu mas o "
                               f"{np.linalg.norm(cg - cg_calc) * 1000:.0f} mm - używam podanego"))
    if mcfg.get("inertia_cad") is not None:
        I_tensor = R @ np.asarray(mcfg["inertia_cad"], float) @ R.T
    if mcfg.get("inertia") is not None:
        Ixx, Iyy, Izz, *rest_i = mcfg["inertia"]
        Ixz = rest_i[0] if rest_i else 0.0
        inertia = [Ixx, Iyy, Izz, Ixz]
    else:
        inertia = [float(I_tensor[0, 0]), float(I_tensor[1, 1]), float(I_tensor[2, 2]), float(-I_tensor[0, 2])]

    # ---- powierzchnie nośne (względem środka ciężkości)
    scfg = cfg.get("surfaces", {}) or {}
    surfaces, info = [], {}
    for role in ("wing", "htail", "vtail"):
        sc = scfg.get(role, {}) or {}
        vertical = role == "vtail"
        thick = sc.get("thickness")
        if sc.get("sections"):
            secs = [Section(to_body(s[:3]) - cg, s[3] * scale, math.radians(s[4] if len(s) > 4 else 0.0))
                    for s in sc["sections"]]
        else:
            m = role_mesh(role)
            if m is None:
                continue
            secs, t_auto = _sections_from_mesh(m.transformed(np.eye(3), 1.0, cg), vertical)
            thick = thick or t_auto
            if len(secs) < 2:
                checks.append(("error", f"Nie udało się wyznaczyć przekrojów części '{role}' - sprawdź osie (axes)"))
                continue
        if sc.get("incidence_deg") is not None:
            for s in secs:
                s.twist = math.radians(sc["incidence_deg"])
        af = dict(DEFAULT_AIRFOIL[role])
        af.update(sc.get("airfoil", {}) or {})
        ctrl_cfg = sc.get("controls", DEFAULT_CONTROLS[role])
        pf = _planform(secs, vertical)
        info[role] = {**pf, "thickness": thick or 0.12, "airfoil": af}
        surfaces.append(Surface(
            role, secs, vertical=vertical, mirror=not vertical,
            a0l=math.radians(af["zero_lift_alpha_deg"]), cm0=af["cm0"], cl_max=af["cl_max"],
            controls=[Control(c["type"], tuple(c.get("span", (0, 1))), c.get("chord_fraction", 0.3),
                              c.get("max_deg", 25)) for c in ctrl_cfg],
            n_span={"wing": 14, "htail": 7, "vtail": 6}[role], n_chord={"wing": 5, "htail": 4, "vtail": 4}[role]))
    if "wing" not in info:
        raise ValueError("Brak skrzydła: dodaj część z role: wing (plik STL/OBJ) lub surfaces.wing.sections")
    # usterzenie motylkowe (V) bez statecznika pionowego -> ster wysokości + kierunku na tej samej powierzchni
    if "htail" in info and "vtail" not in info:
        secs = info["htail"]["sections"]
        dz = abs(secs[-1].le[2] - secs[0].le[2])
        dy = abs(secs[-1].le[1] - secs[0].le[1]) + 1e-9
        if math.degrees(math.atan2(dz, dy)) > 20 and not (scfg.get("htail") or {}).get("controls"):
            for s in surfaces:
                if s.name == "htail":
                    s.controls.append(Control("rudder", (0.0, 1.0), 0.35, 25))
            checks.append(("info", "Wykryto usterzenie motylkowe (V) - stery pracują jako wysokości i kierunku"))
    if "htail" not in info and not (scfg.get("wing") or {}).get("controls"):
        for s in surfaces:   # latające skrzydło - elewony
            if s.name == "wing":
                s.controls.append(Control("elevator", (0.3, 0.95), 0.2, 20))
        checks.append(("info", "Brak usterzenia poziomego - lotki pracują jako elewony (latające skrzydło)"))

    w = info["wing"]
    S, b, c = w["area"], w["span"], w["mac"]
    AR = b * b / S
    lat = Lattice(surfaces)
    der0 = stability_derivatives(lat, S, b, c)
    # pochodne boczne i sterowania zależą od siły nośnej (np. odwrotne odchylenie) - licz przy CL przelotowym
    v_guess = float(cfg.get("cruise_speed") or 0.0)
    total_guess = (mcfg.get("total") or (sum(float(p.get("mass", 0)) for p in parts.values()) + m_comp)) or 1.0
    if not v_guess:
        v_guess = 1.45 * math.sqrt(2 * total_guess * 9.80665 / (RHO0 * S * 1.1))
    cl_cruise = total_guess * 9.80665 / (0.5 * RHO0 * v_guess ** 2 * S)
    alpha_cruise = float(np.clip((cl_cruise - der0.values["CL0"]) / der0.values["CLa"], 0.0, math.radians(8)))
    der = stability_derivatives(lat, S, b, c, alpha0=alpha_cruise)
    d = dict(der.values)
    d["CL0"], d["Cm0_vlm"] = der0.values["CL0"], der0.values["Cm0_vlm"]

    # ---- moment profili (wygięcie) i kadłub
    cm0_af = sum(info[r]["airfoil"]["cm0"] * info[r]["area"] * info[r]["mac"] for r in info if r != "vtail") / (S * c)
    d["Cm0"] = d["Cm0_vlm"] + cm0_af
    fus = role_mesh("fuselage")
    fus_info = None
    if fus is not None:
        fm = fus.transformed(np.eye(3), 1.0, cg)
        lo, hi = fm.bounds
        Lf, Wf, Hf = hi - lo
        vol = abs(fm.volume())
        x_wing = w["sections"][0].le[0] - 0.25 * w["sections"][0].chord
        x_rel = float(np.clip((hi[0] - x_wing) / Lf, 0.0, 1.0))
        Kf = 0.0035 + 0.0475 * x_rel ** 2
        dCma = Kf * Wf ** 2 * Lf / (S * c) * 57.3
        dCnb = -1.3 * vol / (S * b) * (Hf / max(Wf, 1e-6))
        dCYb = -2.0 * (math.pi * Wf * Hf / 4) / S
        # wpływ wysokości skrzydła na kadłubie na efekt wzniosu (DATCOM): górnopłat ~ dodatkowy wznios
        z_w = w["sections"][0].le[2] - 0.5 * (lo[2] + hi[2])          # FRD: ujemne = skrzydło nad osią kadłuba
        d_avg = 0.5 * (Wf + Hf)
        dClb = 1.2 * math.sqrt(AR) * (z_w / b) * (2 * d_avg / b)
        d["Cma"] += dCma
        d["Cnb"] += dCnb
        d["CYb"] += dCYb
        d["Clb"] += dClb
        fus_info = {"length": Lf, "width": Wf, "height": Hf, "volume": vol, "wetted": fm.area(),
                    "dCma": dCma, "dCnb": dCnb, "dClb": dClb, "x_wing_rel": x_rel}
    elif parts:
        checks.append(("info", "Brak kadłuba (role: fuselage) - pomijam jego wpływ na stateczność i opór"))

    # ---- opór pasożytniczy (suma składników, Raymer)
    v_ref = float(cfg.get("cruise_speed") or 14.0)
    lam = float(cfg.get("laminar_fraction", 0.25))
    cd_parts = {}
    for role in ("wing", "htail", "vtail"):
        if role not in info:
            continue
        m = role_mesh(role)
        tc = info[role]["thickness"]
        swet = m.area() if m is not None else 2.04 * info[role]["area"]
        if m is not None and role != "vtail" and m.bounds[0][1] > -0.2 * m.bounds[1][1]:
            swet *= 2.0   # zamodelowana tylko połówka
        re = RHO0 * v_ref * info[role]["mac"] / MU0
        ff = 1 + 2.0 * tc + 100 * tc ** 4
        q = 1.0 if role == "wing" else 1.05
        cd_parts[role] = _cf(re, lam) * ff * q * swet / S
    if fus_info:
        f_ratio = fus_info["length"] / max(math.sqrt(fus_info["width"] * fus_info["height"]), 1e-6)
        re = RHO0 * v_ref * fus_info["length"] / MU0
        cd_parts["fuselage"] = _cf(re, lam * 0.6) * (1 + 60 / f_ratio ** 3 + f_ratio / 400) * fus_info["wetted"] / S
    cd_parts["inne (podwozie, szczeliny, anteny)"] = float(cfg.get("cd_extra", 0.008))
    CD0 = sum(cd_parts.values())
    e = float(np.clip(1.78 * (1 - 0.045 * AR ** 0.68) - 0.64, 0.6, 0.9))

    # ---- przeciągnięcie
    wmask = der.strip_owner == [s.name for s in surfaces].index("wing")
    cl0s, clas = der.strip_cl0[wmask], der.strip_cla[wmask]
    clmax = info["wing"]["airfoil"]["cl_max"]
    a_strip = (clmax - cl0s) / np.maximum(clas, 1e-6)
    a_stall = float(np.min(a_strip)) if len(a_strip) else math.radians(12)
    CLmax = d["CL0"] + d["CLa"] * a_stall

    # ---- punkt neutralny i zapas stateczności
    x_np = c * d["Cma"] / d["CLa"]           # względem środka ciężkości (dodatnie = przed CG)
    sm = -d["Cma"] / d["CLa"]
    x_mac_le = w["x_mac_le"]                 # krawędź natarcia MAC względem CG (dodatnie = przed CG)
    lh = (info["htail"]["x_mac_le"] - 0.25 * info["htail"]["mac"]) if "htail" in info else 0.0
    lv = (info["vtail"]["x_mac_le"] - 0.25 * info["vtail"]["mac"]) if "vtail" in info else 0.0
    Vh = info["htail"]["area"] * abs(lh) / (S * c) if "htail" in info else 0.0
    Vv = info["vtail"]["area"] * abs(lv) / (S * b) if "vtail" in info else 0.0

    # ---- napęd, akumulator, punkty kontaktu
    pcfg, ppos = _prop(cfg.get("propulsion", {}) or {}, R, scale)
    all_mesh = None
    for m in meshes.values():
        all_mesh = m if all_mesh is None else all_mesh.merged(m)
    if ppos is None:
        ppos = np.array([(all_mesh.bounds[1][0] if all_mesh is not None else cg[0] + 0.3), cg[1], cg[2]])
    pcfg["position"] = [float(x) for x in ppos - cg]
    bat = dict(cfg.get("battery", {"cells": 3, "capacity_ah": 2.2}))
    bat.setdefault("mass", next((cc["mass"] for cc in comps if "akum" in cc.get("name", "").lower()
                                 or "bat" in cc.get("name", "").lower()), 0.2))
    if all_mesh is not None:
        am = all_mesh.transformed(np.eye(3), 1.0, cg)
        vv = am.v
        contact = [vv[np.argmax(vv[:, 0])], vv[np.argmin(vv[:, 0])], vv[np.argmax(vv[:, 1])],
                   vv[np.argmin(vv[:, 1])], vv[np.argmax(vv[:, 2])]]
        contact = [[float(x) for x in p] for p in contact]
        visual = am.decimated(int(cfg.get("visual_max_triangles", 8000)))
    else:
        contact, visual = None, None

    weight = total * 9.80665
    v_stall = math.sqrt(2 * weight / (RHO0 * S * max(CLmax, 0.3)))
    cruise = float(cfg.get("cruise_speed") or round(1.45 * v_stall, 1))
    def ctrl_lim(kind):
        vals = [cc.max_deg for s_ in surfaces for cc in s_.controls if cc.kind == kind]
        return float(max(vals)) if vals else 25.0
    lim = cfg.get("limits", {}) or {}
    aero = {k: float(d[k]) for k in ("CL0", "CLa", "CLq", "CLde", "Cm0", "Cma", "Cmq", "Cmde", "CYb", "CYp",
                                     "CYr", "CYdr", "Clb", "Clp", "Clr", "Clda", "Cldr", "Cnb", "Cnp", "Cnr",
                                     "Cnda", "Cndr")}
    aero.update({"CD0": CD0, "e": e, "alpha_stall_deg": math.degrees(a_stall) + 1.5, "stall_blend": 40.0,
                 "post_stall_cl": 0.9, "cd90": 1.2, "stall_roll_asymmetry": float(cfg.get("stall_roll_asymmetry", 0.015))})
    vehicle = {
        "type": "fixed_wing",
        "name": cfg.get("name", "Samolot z CAD"),
        "mass": float(total),
        "inertia": inertia,
        "cruise_speed": cruise,
        "geometry": {"wing_area": S, "span": b, "chord": c},
        "aero": aero,
        "propulsion": pcfg,
        "battery": bat,
        "servos": {"limit_deg": [ctrl_lim("aileron"), ctrl_lim("elevator"), ctrl_lim("rudder")],
                   "tau": 0.04, "rate_deg_s": 300},
        "limits": {"vne": float(lim.get("vne", round(2.3 * cruise, 1))),
                   "max_load_factor": float(lim.get("max_load_factor", 6.0)), "stall_speed": v_stall},
        "crash": {"ground_contact_is_crash": True},
    }
    if contact:
        vehicle["contact_points"] = contact
    if visual is not None:
        vehicle["_visual"] = {"v": visual.v.astype(np.float32), "f": visual.f.astype(np.int32)}

    # ---- kontrole projektu
    mm = 1000.0
    if sm < 0:
        checks.append(("error", f"NIESTATECZNY podłużnie (zapas {sm * 100:.0f}% MAC). Przesuń środek ciężkości do "
                                f"przodu o co najmniej {(-sm + 0.05) * c * mm:.0f} mm albo powiększ usterzenie poziome."))
    elif sm < 0.05:
        checks.append(("warn", f"Mały zapas stateczności podłużnej ({sm * 100:.0f}% MAC) - zalecane 5-15%."))
    elif sm > 0.25:
        checks.append(("warn", f"Bardzo duży zapas stateczności ({sm * 100:.0f}% MAC) - samolot 'ciężki na nos', "
                               f"duże wychylenia steru wysokości przy lądowaniu."))
    else:
        checks.append(("ok", f"Zapas stateczności podłużnej {sm * 100:.0f}% MAC (zalecane 5-15%)."))
    if d["Cnb"] <= 0:
        checks.append(("error", f"Niestateczny kierunkowo (Cn_beta = {d['Cnb']:.3f} <= 0) - powiększ statecznik pionowy."))
    else:
        checks.append(("ok", f"Stateczny kierunkowo (Cn_beta = {d['Cnb']:.3f})."))
    if d["Clb"] >= 0:
        checks.append(("warn", f"Brak stateczności poprzecznej (Cl_beta = {d['Clb']:.3f} >= 0) - dodaj wznios skrzydeł."))
    spiral = d["Clb"] * d["Cnr"] - d["Clr"] * d["Cnb"]
    checks.append(("ok" if spiral > 0 else "info",
                   "Spirala stateczna." if spiral > 0 else "Spirala niestateczna (typowe, wolna - autopilot ją koryguje)."))
    if "htail" in info and not (0.3 <= Vh <= 0.9):
        checks.append(("warn", f"Nietypowy współczynnik objętości usterzenia poziomego V_h = {Vh:.2f} (typowo 0.35-0.7)."))
    if "vtail" in info and not (0.02 <= Vv <= 0.08):
        checks.append(("warn", f"Nietypowy współczynnik objętości usterzenia pionowego V_v = {Vv:.3f} (typowo 0.025-0.05)."))
    loading = total / S * 10   # g/dm^2
    design = {
        "name": vehicle["name"], "mass": total, "S": S, "b": b, "mac": c, "AR": AR, "wing_loading_g_dm2": loading,
        "cg_cad": (R.T @ cg) / scale, "cg_calc_cad": (R.T @ cg_calc) / scale,
        "np_cad": (R.T @ (cg + np.array([x_np, 0, 0]))) / scale,
        "cg_range_cad": [(R.T @ (cg + np.array([x_np + k * c, 0, 0]))) / scale for k in (0.15, 0.05)],
        "static_margin": sm, "cg_pct_mac": x_mac_le / c, "np_pct_mac": (x_mac_le - x_np) / c,
        "Vh": Vh, "Vv": Vv, "CLmax": CLmax, "alpha_stall_deg": math.degrees(a_stall), "v_stall": v_stall,
        "cruise_speed": cruise, "CD0": CD0, "cd_parts": cd_parts, "e": e, "derivatives": d, "inertia": inertia,
        "checks": checks, "units": cfg.get("units", "mm"), "fuselage": fus_info,
        "surfaces": {r: {"area": info[r]["area"], "span": info[r]["span"], "mac": info[r]["mac"],
                         "thickness": info[r]["thickness"],
                         "sections": [[*(s.le + cg), s.chord, math.degrees(s.twist)] for s in info[r]["sections"]]}
                     for r in info},
        "panels": {"A": lat.A + cg, "B": lat.B + cg, "CP": lat.CP + cg},
        "cg_body": cg, "R": R, "scale": scale,
    }
    # wymagany trym w locie przelotowym
    try:
        from ..tunnel import trim_fixed_wing
        from ..vehicles.fixed_wing import FixedWing
        fw = FixedWing({k: v for k, v in vehicle.items() if k != "_visual"})
        tr = trim_fixed_wing(fw, cruise)
        design["trim"] = tr
        if not tr.converged:
            checks.append(("error", f"Brak równowagi (trymu) przy {cruise:.1f} m/s - sprawdź środek ciężkości i zaklinowanie."))
        else:
            de_pct = abs(math.degrees(tr.elevator)) / vehicle["servos"]["limit_deg"][1] * 100
            if de_pct > 60:
                checks.append(("warn", f"Trym wymaga {math.degrees(tr.elevator):.1f} deg steru wysokości ({de_pct:.0f}% zakresu) "
                                       f"- zmień zaklinowanie usterzenia lub środek ciężkości."))
            if tr.throttle > 0.9:
                checks.append(("warn", f"Lot przelotowy {cruise:.1f} m/s wymaga {tr.throttle * 100:.0f}% mocy - za słaby napęd."))
        n_max = fw.props.n_max(fw.battery.nominal_voltage)
        t_static = fw.props.ct0 * RHO0 * n_max ** 2 * fw.props.D ** 4
        design["static_thrust_to_weight"] = t_static / weight
        if t_static / weight < 0.35:
            checks.append(("warn", f"Ciąg statyczny {t_static / weight:.2f} x ciężar - trudny start z ręki/wznoszenie."))
    except Exception as ex:  # noqa: BLE001
        checks.append(("warn", f"Nie udało się wyznaczyć trymu: {ex}"))
    order = {"error": 0, "warn": 1, "info": 2, "ok": 3}
    checks.sort(key=lambda x: order[x[0]])
    return vehicle, design


def vehicle_to_yaml_dict(vehicle: dict) -> dict:
    """Konfiguracja do zapisu w YAML (bez siatki wizualizacji i typów numpy)."""
    def conv(x):
        if isinstance(x, dict):
            return {k: conv(v) for k, v in x.items() if not k.startswith("_")}
        if isinstance(x, (list, tuple)):
            return [conv(v) for v in x]
        if isinstance(x, np.ndarray):
            return [conv(v) for v in x.tolist()]
        if isinstance(x, (np.floating, float)):
            return float(round(float(x), 6))
        if isinstance(x, np.integer):
            return int(x)
        return x
    return conv(vehicle)
