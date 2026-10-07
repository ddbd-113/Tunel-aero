"""Wirtualny tunel aerodynamiczny i analiza osiągów.

Samolot:   charakterystyki CL(alfa), CD(alfa), Cm(alfa), biegunowa, doskonałość L/D,
           trym w locie poziomym, prędkość przeciągnięcia, moc i długotrwałość lotu vs prędkość.
Multirotor: moc i przepustnica zawisu, zapas ciągu i czas lotu w funkcji wysokości
           i temperatury, moc w locie postępowym vs prędkość.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .atmosphere import Atmosphere
from .math3d import GRAVITY


# ============================================================ samolot
def aero_sweep(veh, airspeed=17.0, alphas_deg=None, elevator_deg=0.0, rho=1.225, deg=None) -> dict:
    alphas = np.radians(np.arange(-10, 31, 0.5) if alphas_deg is None else np.asarray(alphas_deg, float))
    out = {k: [] for k in ("alpha", "CL", "CD", "Cm", "LD", "L", "D")}
    qS = 0.5 * rho * airspeed ** 2 * veh.S
    for a in alphas:
        CL, CD, CY, Cl, Cm, Cn, _ = veh.coefficients(a, 0.0, 0.0, 0.0, 0.0, airspeed,
                                                     (0.0, math.radians(elevator_deg), 0.0), deg)
        out["alpha"].append(math.degrees(a))
        out["CL"].append(CL)
        out["CD"].append(CD)
        out["Cm"].append(Cm)
        out["LD"].append(CL / CD if CD > 0 else 0.0)
        out["L"].append(qS * CL)
        out["D"].append(qS * CD)
    return {k: np.array(v) for k, v in out.items()}


def cl_max(veh, deg=None) -> tuple[float, float]:
    s = aero_sweep(veh, alphas_deg=np.arange(0, 30, 0.1), deg=deg)
    i = int(np.argmax(s["CL"]))
    return float(s["CL"][i]), float(s["alpha"][i])


def stall_speed(veh, rho=1.225, deg=None, mass=None) -> float:
    clm, _ = cl_max(veh, deg)
    m = mass or veh.total_mass
    return math.sqrt(2 * m * GRAVITY / (rho * veh.S * clm))


@dataclass
class Trim:
    airspeed: float
    alpha: float        # rad
    elevator: float     # rad
    throttle: float     # 0..1 (może być > 1 - lot niemożliwy)
    thrust: float       # N
    power: float        # W (elektryczna)
    prop_speed: float   # obr/s
    converged: bool

    @property
    def feasible(self) -> bool:
        return self.converged and 0.0 <= self.throttle <= 1.0


def trim_fixed_wing(veh, airspeed: float, rho: float = 1.225, gamma: float = 0.0,
                    voltage: float | None = None, deg=None) -> Trim:
    """Trym w locie ustalonym (skrzydła poziomo, bez ślizgu) - metoda Newtona."""
    m = veh.total_mass
    props = veh.props
    D4 = props.D ** 4

    def residual(z):
        alpha, de, n = z
        theta = alpha + gamma
        v_b = airspeed * np.array([math.cos(alpha), 0.0, math.sin(alpha)])
        F, M = veh.aero_forces(v_b, np.zeros(3), rho, np.array([0.0, de, 0.0]), deg)
        J = v_b[0] / (max(n, 1e-3) * props.D)
        ct = props.ct0 * (1 - J / props.j0)
        T = ct * rho * n * abs(n) * D4
        fx = F[0] + T - m * GRAVITY * math.sin(theta)
        fz = F[2] + m * GRAVITY * math.cos(theta)
        return np.array([fx / (m * GRAVITY), fz / (m * GRAVITY), M[1] / (m * GRAVITY * veh.c)])

    z = np.array([0.05, 0.0, 80.0])
    ok = False
    for _ in range(60):
        r = residual(z)
        if np.max(np.abs(r)) < 1e-9:
            ok = True
            break
        Jm = np.zeros((3, 3))
        for j, h in enumerate((1e-6, 1e-6, 1e-3)):
            dz = np.zeros(3)
            dz[j] = h
            Jm[:, j] = (residual(z + dz) - r) / h
        try:
            step = np.linalg.solve(Jm, -r)
        except np.linalg.LinAlgError:
            break
        step = np.clip(step, [-0.05, -0.05, -40.0], [0.05, 0.05, 40.0])
        z = z + step
    alpha, de, n = z
    ok = ok or np.max(np.abs(residual(z))) < 1e-6
    v_batt = voltage if voltage is not None else veh.battery.open_circuit_voltage()
    thr = n / props.n_max(v_batt)
    v_b = airspeed * np.array([math.cos(alpha), 0.0, math.sin(alpha)])
    saved = props.n.copy()
    props.n[:] = n
    F, _, p_elec = props.forces(v_b, np.zeros(3), rho, 100.0)
    props.n[:] = saved
    return Trim(airspeed, float(alpha), float(de), float(thr), float(F[0]), float(p_elec), float(n),
                bool(ok and abs(de) < veh.servo_limit[1] and alpha < math.radians(veh.aero["alpha_stall_deg"])))


def fixed_wing_controller_model(veh, airspeed: float, rho: float = 1.225, deg=None) -> dict:
    tr = trim_fixed_wing(veh, airspeed, rho, deg=deg)
    vs = stall_speed(veh, rho)
    vs_ias = vs * math.sqrt(rho / 1.225)
    # prędkość wznoszenia przy pełnej przepustnicy (z nadmiaru mocy)
    n_full = veh.props.n_max(veh.battery.nominal_voltage)
    J = airspeed / (n_full * veh.props.D)
    T_full = veh.props.ct0 * (1 - J / veh.props.j0) * rho * n_full ** 2 * veh.props.D ** 4
    climb = max(1.0, (T_full - tr.thrust) * airspeed / (veh.total_mass * GRAVITY))
    return {"v_trim": airspeed * math.sqrt(rho / 1.225), "alpha_trim": tr.alpha, "de_trim": tr.elevator,
            "thr_trim": min(max(tr.throttle, 0.05), 0.95), "v_min": 1.25 * vs_ias, "v_stall": vs_ias,
            "climb_full_throttle": climb, "trim": tr}


def fixed_wing_performance(veh, speeds=None, altitude=0.0, temperature_offset=0.0) -> dict:
    atm = Atmosphere(temperature_offset).at(altitude)
    speeds = np.arange(9, 31, 0.5) if speeds is None else np.asarray(speeds, float)
    res = {k: [] for k in ("speed", "alpha", "elevator", "throttle", "power", "LD", "endurance_min", "range_km")}
    energy_wh = veh.battery.capacity_ah * veh.battery.nominal_voltage * 0.85
    for V in speeds:
        tr = trim_fixed_wing(veh, V, atm.density)
        if not tr.converged or tr.throttle > 1.05:
            continue
        res["speed"].append(V)
        res["alpha"].append(math.degrees(tr.alpha))
        res["elevator"].append(math.degrees(tr.elevator))
        res["throttle"].append(tr.throttle)
        res["power"].append(tr.power)
        res["LD"].append(veh.total_mass * GRAVITY / max(tr.thrust, 1e-6))
        t_h = energy_wh / tr.power
        res["endurance_min"].append(60 * t_h)
        res["range_km"].append(V * 3.6 * t_h)
    return {k: np.array(v) for k, v in res.items()}


# ============================================================ multirotor
def multirotor_hover(veh, altitude=0.0, temperature_offset=0.0, payload=0.0) -> dict:
    atm = Atmosphere(temperature_offset).at(altitude)
    pr = veh.props
    m = veh.total_mass + payload
    T_each = m * GRAVITY / pr.n_units
    rho = atm.density
    n = math.sqrt(T_each / (pr.ct0 * rho * pr.D ** 4))
    vh = math.sqrt(T_each / (2 * rho * pr.A))
    p_shaft = T_each * vh / pr.eta + pr.cp0 * rho * n ** 3 * pr.D ** 5
    p_el = pr.n_units * p_shaft / pr.motor_eff
    v_batt = veh.battery.nominal_voltage
    n_max = pr.n_max(v_batt)
    t_max = pr.ct0 * rho * n_max ** 2 * pr.D ** 4 * pr.n_units
    return {"density": rho, "density_altitude": atm.density_altitude, "rpm": n * 60,
            "throttle": n / n_max, "power": p_el, "thrust_to_weight": t_max / (m * GRAVITY)}


def multirotor_endurance(veh, altitude=0.0, temperature_c=None, temperature_offset=0.0,
                         payload=0.0, reserve=0.2) -> float:
    """Czas zawisu [min] z uwzględnieniem temperatury akumulatora."""
    h = multirotor_hover(veh, altitude, temperature_offset, payload)
    b = veh.battery
    t0 = b.temp_c
    if temperature_c is not None:
        b.temp_c = temperature_c
    cap_factor = b.capacity_factor()
    r_factor = b.resistance_factor()
    b.temp_c = t0
    v_load = b.nominal_voltage - (h["power"] / b.nominal_voltage) * b.cells * b.r_cell * r_factor
    energy = b.capacity_ah * cap_factor * (1 - reserve) * max(v_load, 0.1)
    return 60.0 * energy / h["power"]


def multirotor_forward_power(veh, speeds=None, rho=1.225) -> dict:
    """Moc w locie poziomym ze stałą prędkością (bilans: ciąg = ciężar + opór kadłuba)."""
    speeds = np.arange(0, 21, 1.0) if speeds is None else np.asarray(speeds, float)
    pr = veh.props
    m = veh.total_mass
    res = {"speed": [], "pitch_deg": [], "power": []}
    for V in speeds:
        # opór kadłuba - przybliżenie: powierzchnia czołowa z rzutu nachylonego kadłuba
        D = 0.5 * rho * V * V * veh.drag_cd * veh.drag_area[0]
        Hf = pr.h_coeff * pr.n_units * 100.0 * V
        T = math.hypot(m * GRAVITY, D + Hf)
        theta = math.atan2(D + Hf, m * GRAVITY)
        T_each = T / pr.n_units
        v_ax = V * math.sin(theta)
        vi = -v_ax / 2 + math.sqrt(v_ax ** 2 / 4 + T_each / (2 * rho * pr.A))
        # w locie postępowym prędkość indukowana maleje (teoria Glauerta)
        vh = math.sqrt(T_each / (2 * rho * pr.A))
        vt = V * math.cos(theta)
        vi_g = vh ** 2 / math.sqrt(vt ** 2 + (v_ax + vi) ** 2 + 1e-9)
        vi = min(vi, vi_g) if V > 0 else vi
        n = math.sqrt(T_each / (pr.ct0 * rho * pr.D ** 4))
        p_shaft = T_each * (v_ax + vi) / pr.eta + pr.cp0 * rho * n ** 3 * pr.D ** 5 * (1 + 4.65 * (vt / (math.pi * pr.D * n)) ** 2)
        res["speed"].append(V)
        res["pitch_deg"].append(math.degrees(theta))
        res["power"].append(pr.n_units * p_shaft / pr.motor_eff)
    return {k: np.array(v) for k, v in res.items()}


# ============================================================ raport
def tunnel_figures(vehicle_cfg: dict, cruise_speed: float | None = None) -> tuple[list, list, object]:
    """Charakterystyki pojazdu: (lista faktów, obrazy PNG w base64, pojazd)."""
    import base64
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .vehicles import build_vehicle, resolve_vehicle_config

    vehicle_cfg = resolve_vehicle_config(vehicle_cfg)
    veh = build_vehicle(vehicle_cfg, np.random.default_rng(0))
    imgs, facts = [], []

    def png(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=105, bbox_inches="tight")
        plt.close(fig)
        imgs.append(base64.b64encode(buf.getvalue()).decode())

    if veh.kind == "fixed_wing":
        from .weather import Weather
        iced = Weather()
        iced.ice = 1.0
        s = aero_sweep(veh)
        si = aero_sweep(veh, deg=iced.degradation())
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        axs[0, 0].plot(s["alpha"], s["CL"], label="czysty")
        axs[0, 0].plot(si["alpha"], si["CL"], "--", label="oblodzony (100%)")
        axs[0, 0].set(title="Współczynnik siły nośnej CL(alfa)", xlabel="alfa [deg]")
        axs[0, 1].plot(s["CD"], s["CL"], label="czysty")
        axs[0, 1].plot(si["CD"], si["CL"], "--", label="oblodzony")
        axs[0, 1].set(title="Biegunowa CL(CD)", xlabel="CD")
        axs[1, 0].plot(s["alpha"], s["Cm"])
        axs[1, 0].axhline(0, color="k", lw=0.6)
        axs[1, 0].set(title="Moment pochylający Cm(alfa) - ujemne nachylenie = stateczny", xlabel="alfa [deg]")
        axs[1, 1].plot(s["alpha"], s["LD"], label="czysty")
        axs[1, 1].plot(si["alpha"], si["LD"], "--", label="oblodzony")
        axs[1, 1].set(title="Doskonałość aerodynamiczna L/D (sam płat)", xlabel="alfa [deg]")
        for a in axs.flat:
            a.grid(alpha=0.3)
            if a.get_legend_handles_labels()[0]:
                a.legend(fontsize=8)
        fig.tight_layout()
        png(fig)
        perf = fixed_wing_performance(veh)
        perf_hot = fixed_wing_performance(veh, altitude=2000, temperature_offset=20)
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
        axs[0].plot(perf["speed"], perf["power"], label="0 m ISA")
        axs[0].plot(perf_hot["speed"], perf_hot["power"], "--", label="2000 m, ISA+20")
        axs[0].set(title="Moc elektryczna w locie poziomym", xlabel="TAS [m/s]", ylabel="[W]")
        axs[1].plot(perf["speed"], perf["endurance_min"], label="0 m ISA")
        axs[1].plot(perf_hot["speed"], perf_hot["endurance_min"], "--", label="2000 m, ISA+20")
        axs[1].set(title="Długotrwałość lotu (85% energii)", xlabel="TAS [m/s]", ylabel="[min]")
        axs[2].plot(perf["speed"], perf["throttle"] * 100, label="przepustnica [%]")
        axs[2].plot(perf["speed"], perf["alpha"], label="kąt natarcia [deg]")
        axs[2].plot(perf["speed"], perf["elevator"], label="ster wys. [deg]")
        axs[2].set(title="Trym", xlabel="TAS [m/s]")
        for a in axs:
            a.grid(alpha=0.3)
            a.legend(fontsize=8)
        fig.tight_layout()
        png(fig)
        clm, a_clm = cl_max(veh)
        vs = stall_speed(veh)
        V = cruise_speed or vehicle_cfg.get("cruise_speed", 17.0)
        tr = trim_fixed_wing(veh, V)
        i_end = int(np.argmax(perf["endurance_min"])) if len(perf["speed"]) else 0
        i_rng = int(np.argmax(perf["range_km"])) if len(perf["speed"]) else 0
        facts = [
            ("CL max", f"{clm:.2f} przy alfa {a_clm:.1f} deg"),
            ("prędkość przeciągnięcia (ISA, 0 m)", f"{vs:.1f} m/s"),
            ("prędkość przeciągnięcia z pełnym oblodzeniem", f"{stall_speed(veh, deg=iced.degradation()):.1f} m/s"),
            (f"trym przy {V:.0f} m/s", f"alfa {math.degrees(tr.alpha):.1f} deg, ster wys. {math.degrees(tr.elevator):.1f} deg, "
                                       f"przepustnica {tr.throttle * 100:.0f}%, moc {tr.power:.0f} W"),
            ("maks. długotrwałość", f"{perf['endurance_min'][i_end]:.0f} min przy {perf['speed'][i_end]:.1f} m/s"),
            ("maks. zasięg", f"{perf['range_km'][i_rng]:.0f} km przy {perf['speed'][i_rng]:.1f} m/s"),
            ("maks. prędkość (pełny gaz)", f"{perf['speed'][-1]:.1f} m/s" if len(perf['speed']) else "-"),
        ]
    else:
        alts = np.arange(0, 5001, 250)
        fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
        for dT, ls in ((-20, ":"), (0, "-"), (20, "--")):
            h = [multirotor_hover(veh, a, dT) for a in alts]
            axs[0].plot(alts, [x["power"] for x in h], ls, label=f"ISA{dT:+d} K")
            axs[1].plot(alts, [x["thrust_to_weight"] for x in h], ls, label=f"ISA{dT:+d} K")
        axs[0].set(title="Moc zawisu vs wysokość n.p.m.", xlabel="wysokość [m]", ylabel="[W]")
        axs[1].axhline(1.0, color="r", lw=0.8)
        axs[1].axhline(1.5, color="orange", lw=0.8, ls=":")
        axs[1].set(title="Stosunek ciągu do ciężaru (min. ~1.5 do bezpiecznych manewrów)", xlabel="wysokość [m]")
        temps = np.arange(-25, 41, 5)
        for alt in (0, 2000, 4000):
            axs[2].plot(temps, [multirotor_endurance(veh, alt, temperature_c=tc,
                                                     temperature_offset=tc - (15 - 0.0065 * alt)) for tc in temps],
                        label=f"{alt} m n.p.m.")
        axs[2].set(title="Czas zawisu (rezerwa 20%) vs temperatura", xlabel="temperatura [C]", ylabel="[min]")
        for a in axs:
            a.grid(alpha=0.3)
            a.legend(fontsize=8)
        fig.tight_layout()
        png(fig)
        fp = multirotor_forward_power(veh)
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.plot(fp["speed"], fp["power"])
        ax.set(title="Moc w locie postępowym (bez wiatru)", xlabel="prędkość [m/s]", ylabel="[W]")
        ax.grid(alpha=0.3)
        png(fig)
        h0 = multirotor_hover(veh)
        facts = [
            ("masa", f"{veh.mass:.2f} kg"),
            ("ciąg / ciężar (0 m ISA)", f"{h0['thrust_to_weight']:.2f}"),
            ("moc zawisu (0 m ISA)", f"{h0['power']:.0f} W, przepustnica {h0['throttle'] * 100:.0f}%"),
            ("czas zawisu 20 C / -10 C (rezerwa 20%)",
             f"{multirotor_endurance(veh, temperature_c=20, temperature_offset=5):.1f} / "
             f"{multirotor_endurance(veh, temperature_c=-10, temperature_offset=-25):.1f} min"),
            ("ciąg / ciężar na 3000 m, +25 C", f"{multirotor_hover(veh, 3000, 25 - (15 - 19.5))['thrust_to_weight']:.2f}"),
        ]
    return facts, imgs, veh


def write_tunnel_report(vehicle_cfg: dict, out_dir, cruise_speed: float | None = None):
    """Generuje raport HTML z charakterystykami pojazdu (wirtualny tunel + osiągi)."""
    import html as _html
    from pathlib import Path

    from .report import CSS

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    facts, imgs, veh = tunnel_figures(vehicle_cfg, cruise_speed)
    doc = f"""<!doctype html><html lang="pl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Tunel - {_html.escape(veh.name)}</title>
<style>{CSS}</style></head><body><main><h1>Wirtualny tunel aerodynamiczny: {_html.escape(veh.name)}</h1>
<div class="card"><table>{''.join(f'<tr><th>{_html.escape(k)}</th><td>{_html.escape(v)}</td></tr>' for k, v in facts)}</table></div>
{''.join(f"<div class='card'><img src='data:image/png;base64,{i}'></div>" for i in imgs)}
</main></body></html>"""
    p = out / "tunel.html"
    p.write_text(doc, encoding="utf-8")
    return p
