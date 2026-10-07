"""Generuje przykładowe pliki STL samolotu (jak eksport z programu CAD).

Trener z górnym płatem, rozpiętość 1400 mm, profil NACA 2412, usterzenie NACA 0009.
Układ współrzędnych jak w typowym projekcie aerodynamicznym (AVL/XFLR5):
  x - do tyłu (nos w x = 0), y - na prawe skrzydło, z - w górę, jednostki: mm.
W pliku aircraft.yaml odpowiada to:  units: mm,  axes: {forward: -x, up: +z}

    python examples/moj_samolot/generuj_stl.py
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tunel_aero.cad.mesh import Mesh, write_stl  # noqa: E402

OUT = Path(__file__).resolve().parent


def naca4(code: str, n: int = 30) -> np.ndarray:
    """Zamknięty kontur profilu NACA 4-cyfrowego (x od 0 do 1), kolejność: spływ-góra-nos-dół."""
    m, p, t = int(code[0]) / 100, int(code[1]) / 10, int(code[2:]) / 100
    beta = np.linspace(0, math.pi, n)
    x = 0.5 * (1 - np.cos(beta))
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.126 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1036 * x ** 4)
    if m > 0:
        yc = np.where(x < p, m / p ** 2 * (2 * p * x - x ** 2), m / (1 - p) ** 2 * (1 - 2 * p + 2 * p * x - x ** 2))
        dyc = np.where(x < p, 2 * m / p ** 2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x))
    else:
        yc = dyc = np.zeros_like(x)
    th = np.arctan(dyc)
    xu, yu = x - yt * np.sin(th), yc + yt * np.cos(th)
    xl, yl = x + yt * np.sin(th), yc - yt * np.cos(th)
    up = np.column_stack((xu, yu))[::-1]
    lo = np.column_stack((xl, yl))[1:-1]
    return np.vstack((up, lo))


def loft(rings: list[np.ndarray], cap: bool = True) -> Mesh:
    """Powierzchnia rozpięta na kolejnych zamkniętych pierścieniach o tej samej liczbie punktów."""
    k = len(rings[0])
    v = np.vstack(rings)
    f = []
    for i in range(len(rings) - 1):
        a, b = i * k, (i + 1) * k
        for j in range(k):
            j2 = (j + 1) % k
            f += [[a + j, b + j, b + j2], [a + j, b + j2, a + j2]]
    if cap:
        for idx, flip in ((0, True), (len(rings) - 1, False)):
            c = len(v)
            v = np.vstack((v, rings[idx].mean(axis=0)))
            base = idx * k
            for j in range(k):
                j2 = (j + 1) % k
                f.append([c, base + j2, base + j] if flip else [c, base + j, base + j2])
    return Mesh(v, np.array(f))


def wing_surface(code, root_le, root_c, tip_le, tip_c, semi, incidence_deg, dihedral_deg, n=16, both=True,
                 vertical=False):
    af = naca4(code)
    rings = []
    ys = np.linspace(-semi if both else 0.0, semi, 2 * n + 1 if both else n + 1)
    for y in ys:
        f = abs(y) / semi
        le = np.array(root_le, float) + (np.array(tip_le, float) - np.array(root_le, float)) * f
        c = root_c + (tip_c - root_c) * f
        inc = math.radians(incidence_deg)
        x = af[:, 0] * c
        z = af[:, 1] * c
        xr = x * math.cos(inc) + z * math.sin(inc)          # nos w górę = krawędź spływu w dół
        zr = -x * math.sin(inc) + z * math.cos(inc)
        if vertical:
            pts = np.column_stack((le[0] + xr, le[1] + zr, np.full_like(xr, le[2] + y)))
        else:
            zd = abs(y) * math.tan(math.radians(dihedral_deg))
            pts = np.column_stack((le[0] + xr, np.full_like(xr, y), le[2] + zr + zd))
        rings.append(pts)
    return loft(rings)


def fuselage(length=1000.0, n=40, m=24):
    rings = []
    xs = np.linspace(0, length, n)
    for x in xs:
        s = x / length
        # przekrój: zaokrąglony nos, maksimum ok. 25% długości, zwężenie ku ogonowi
        w = 82 * (np.clip(s / 0.18, 0, 1) ** 0.5) * (1 - 0.62 * np.clip((s - 0.4) / 0.6, 0, 1)) + 2
        h = 112 * (np.clip(s / 0.18, 0, 1) ** 0.5) * (1 - 0.6 * np.clip((s - 0.4) / 0.6, 0, 1)) + 2
        zc = 5 + 18 * np.clip((s - 0.4) / 0.6, 0, 1)
        ang = np.linspace(0, 2 * math.pi, m, endpoint=False)
        rings.append(np.column_stack((np.full(m, x), 0.5 * w * np.cos(ang), zc + 0.5 * h * np.sin(ang))))
    return loft(rings)


if __name__ == "__main__":
    parts = {
        "skrzydlo.stl": wing_surface("2412", (220, 0, 62), 240, (235, 0, 62), 170, 700, 2.0, 3.0),
        "ster_wysokosci.stl": wing_surface("0009", (850, 0, 30), 145, (875, 0, 30), 110, 235, 0.0, 0.0, n=8),
        "statecznik.stl": wing_surface("0009", (830, 0, 40), 175, (905, 0, 40), 100, 175, 0.0, 0.0, n=6,
                                       both=False, vertical=True),
        "kadlub.stl": fuselage(),
    }
    for name, mesh in parts.items():
        write_stl(OUT / name, mesh)
        print(f"{name}: {mesh.n_tri} trójkątów, wymiary [mm]: {np.round(mesh.extent, 1)}")
