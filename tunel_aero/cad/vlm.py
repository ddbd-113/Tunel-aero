"""Metoda siatki wirowej (Vortex Lattice Method) - jak w AVL / XFLR5.

Powierzchnie nośne dzielone są na panele z wirem podkowiastym (wir związany na 1/4 cięciwy
panelu, punkt kontrolny na 3/4, wiry swobodne do nieskończoności za samolotem). Z warunku
opływu stycznego wyznacza się cyrkulacje, a z twierdzenia Kutty-Żukowskiego - siły i momenty.
Pochodne stateczności liczone są różnicami skończonymi (alfa, beta, p, q, r, wychylenia sterów).

Układ: ciało FRD (x - do przodu, y - prawe skrzydło, z - w dół), początek w środku masy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

FOUR_PI = 4.0 * math.pi


@dataclass
class Section:
    le: np.ndarray            # krawędź natarcia [m] (FRD, względem środka masy)
    chord: float              # [m]
    twist: float = 0.0        # kąt zaklinowania [rad] (+ = nos w górę)


@dataclass
class Control:
    kind: str                 # aileron | elevator | rudder
    span: tuple = (0.0, 1.0)  # zakres rozpiętości (ułamek półrozpiętości)
    chord_fraction: float = 0.25
    max_deg: float = 25.0


@dataclass
class Surface:
    name: str
    sections: list            # sekcje od nasady do końcówki (jedna strona)
    vertical: bool = False
    mirror: bool = True
    a0l: float = 0.0          # kąt zerowej siły nośnej profilu [rad] (ujemny = profil wygięty)
    cm0: float = 0.0          # moment profilu względem 1/4 cięciwy
    cl_max: float = 1.2
    controls: list = field(default_factory=list)
    n_span: int = 12
    n_chord: int = 5


class Lattice:
    def __init__(self, surfaces: list[Surface]):
        self.surfaces = surfaces
        A, B, CP, N, T, M, owner, xi, span_f, side = [], [], [], [], [], [], [], [], [], []
        self.strips = []      # (powierzchnia, cięciwa, szerokość pasa, indeksy paneli)
        for si, s in enumerate(surfaces):
            secs = s.sections
            spos = np.array([self._span_coord(sec.le, s.vertical) for sec in secs])
            order = np.argsort(spos)
            secs = [secs[i] for i in order]
            spos = spos[order]
            s0, s1 = spos[0], spos[-1]
            ts = 0.5 * (1 - np.cos(np.linspace(0, math.pi, s.n_span + 1)))   # zagęszczenie przy końcówce/nasadzie
            stations = s0 + (s1 - s0) * ts

            def interp(sv):
                k = int(np.clip(np.searchsorted(spos, sv) - 1, 0, len(secs) - 2)) if len(secs) > 1 else 0
                if len(secs) == 1:
                    return secs[0].le, secs[0].chord, secs[0].twist
                a, b = secs[k], secs[k + 1]
                w = 0.0 if spos[k + 1] == spos[k] else (sv - spos[k]) / (spos[k + 1] - spos[k])
                return a.le + (b.le - a.le) * w, a.chord + (b.chord - a.chord) * w, a.twist + (b.twist - a.twist) * w

            st = [interp(v) for v in stations]
            sides = (1, -1) if (s.mirror and not s.vertical) else (1,)
            for sg in sides:
                for j in range(s.n_span):
                    (le0, c0, t0), (le1, c1, t1) = st[j], st[j + 1]
                    if sg < 0:
                        le0 = le0 * np.array([1, -1, 1])
                        le1 = le1 * np.array([1, -1, 1])
                    frac = 0.5 * (ts[j] + ts[j + 1])
                    ids = []
                    for k in range(s.n_chord):
                        x0, x1 = k / s.n_chord, (k + 1) / s.n_chord

                        def pt(le, c, tw, xf):
                            return le + xf * c * np.array([-math.cos(tw), 0.0, math.sin(tw)])

                        p00, p01 = pt(le0, c0, t0, x0), pt(le1, c1, t1, x0)
                        p10, p11 = pt(le0, c0, t0, x1), pt(le1, c1, t1, x1)
                        qa = p00 + 0.25 * (p10 - p00)
                        qb = p01 + 0.25 * (p11 - p01)
                        cp = 0.5 * ((p00 + 0.75 * (p10 - p00)) + (p01 + 0.75 * (p11 - p01)))
                        n = np.cross(p11 - p00, p01 - p10)
                        n /= np.linalg.norm(n)
                        if s.vertical:
                            n = n if n[1] >= 0 else -n
                        else:
                            n = n if n[2] <= 0 else -n
                        t = 0.5 * ((p10 - p00) + (p11 - p01))
                        t /= np.linalg.norm(t)
                        A.append(qa)
                        B.append(qb)
                        CP.append(cp)
                        N.append(n)
                        T.append(t)
                        M.append(0.5 * (qa + qb))
                        owner.append(si)
                        xi.append(0.5 * (x0 + x1))
                        span_f.append(frac)
                        side.append(sg)
                        ids.append(len(A) - 1)
                    width = float(np.linalg.norm((le1 - le0) - ((le1 - le0) @ np.array([1.0, 0, 0])) * np.array([1.0, 0, 0])))
                    self.strips.append((si, 0.5 * (c0 + c1), width, ids))
        self.A, self.B, self.CP = map(np.array, (A, B, CP))
        self.N0, self.T, self.M = np.array(N), np.array(T), np.array(M)
        self.owner = np.array(owner)
        self.xi = np.array(xi)
        self.span_f = np.array(span_f)
        self.side = np.array(side)
        self.W_cp = self._influence(self.CP)
        self.W_mid = self._influence(self.M)
        # zaklinowanie profilem (wygięcie): obrót normalnych o -a0l
        self.base_rot = np.array([-surfaces[o].a0l if not surfaces[o].vertical else 0.0 for o in self.owner])

    @staticmethod
    def _span_coord(le, vertical):
        return -le[2] if vertical else abs(le[1])

    # ------------------------------------------------------- Biot-Savart
    def _influence(self, P: np.ndarray) -> np.ndarray:
        """Prędkość indukowana w punktach P przez jednostkowe wiry podkowiaste: (nP, nW, 3)."""
        A, B = self.A[None, :, :], self.B[None, :, :]
        P = P[:, None, :]
        u = np.array([-1.0, 0.0, 0.0])           # wiry swobodne płyną do tyłu
        r1, r2 = P - A, P - B
        cr = np.cross(r1, r2)
        cr2 = np.einsum("ijk,ijk->ij", cr, cr)
        n1, n2 = np.linalg.norm(r1, axis=2), np.linalg.norm(r2, axis=2)
        r0 = B - A
        L2 = np.einsum("ijk,ijk->ij", r0, r0)
        ok = cr2 > 1e-10 * L2
        dot = np.einsum("ijk,ijk->ij", r0, r1 / np.maximum(n1, 1e-12)[..., None] - r2 / np.maximum(n2, 1e-12)[..., None])
        v = np.where(ok[..., None], cr / np.where(ok, cr2, 1.0)[..., None] * dot[..., None], 0.0)

        def semi(r, n):
            c = np.cross(u, r)
            c2 = np.einsum("ijk,ijk->ij", c, c)
            ok2 = c2 > 1e-10
            f = (1.0 + (r @ u) / np.maximum(n, 1e-12))
            return np.where(ok2[..., None], c / np.where(ok2, c2, 1.0)[..., None] * f[..., None], 0.0)

        v = v + semi(r2, n2) - semi(r1, n1)       # B -> nieskończoność, nieskończoność -> A
        return v / FOUR_PI

    # ------------------------------------------------------- rozwiązanie
    def normals(self, deflections: dict | None = None) -> np.ndarray:
        rot = self.base_rot.copy()
        for kind, delta in (deflections or {}).items():
            if not delta:
                continue
            for si, s in enumerate(self.surfaces):
                for c in s.controls:
                    if c.kind != kind:
                        continue
                    m = (self.owner == si) & (self.xi > 1 - c.chord_fraction) \
                        & (self.span_f >= c.span[0]) & (self.span_f <= c.span[1])
                    if kind == "elevator":
                        d = np.array([0.0, 0.0, 1.0])
                        proj = -(self.N0[m] @ d)
                    elif kind == "rudder":
                        d = np.array([0.0, -1.0, 0.0])
                        proj = -(self.N0[m] @ d)
                    else:   # lotka: prawa krawędź spływu w górę, lewa w dół
                        proj = -np.einsum("ij,ij->i", self.N0[m],
                                          np.column_stack((np.zeros(m.sum()), np.zeros(m.sum()), -self.side[m])))
                    rot[m] += delta * proj
        c, s = np.cos(rot)[:, None], np.sin(rot)[:, None]
        return self.N0 * c + self.T * s

    def solve(self, alpha: float = 0.0, beta: float = 0.0, omega=(0.0, 0.0, 0.0), deflections=None):
        """Siły i momenty dla V=1, rho=1 (współczynniki = F / (0.5 S))."""
        u = math.cos(alpha) * math.cos(beta)
        v = math.sin(beta)
        w = math.sin(alpha) * math.cos(beta)
        vinf = -np.array([u, v, w])
        om = np.asarray(omega, float)
        N = self.normals(deflections)
        Vcp = vinf - np.cross(om, self.CP)
        Amat = np.einsum("ijk,ik->ij", self.W_cp, N)
        rhs = -np.einsum("ij,ij->i", Vcp, N)
        gam = np.linalg.solve(Amat, rhs)
        Vm = vinf - np.cross(om, self.M) + np.einsum("ijk,j->ik", self.W_mid, gam)
        F = gam[:, None] * np.cross(Vm, self.B - self.A)
        Mo = np.cross(self.M, F)
        return F.sum(axis=0), Mo.sum(axis=0), F, gam


@dataclass
class Derivatives:
    S: float
    b: float
    c: float
    values: dict
    strip_cl0: np.ndarray
    strip_cla: np.ndarray
    strip_owner: np.ndarray


def coeffs(F, M, alpha, S, b, c):
    q = 0.5 * S
    ca, sa = math.cos(alpha), math.sin(alpha)
    return {"CL": (F[0] * sa - F[2] * ca) / q, "CD": -(F[0] * ca + F[2] * sa) / q, "CY": F[1] / q,
            "Cl": M[0] / (q * b), "Cm": M[1] / (q * c), "Cn": M[2] / (q * b)}


def stability_derivatives(lat: Lattice, S: float, b: float, c: float, alpha0: float = 0.0) -> Derivatives:
    """Pochodne stateczności i sterowania (na radian / na znormalizowaną prędkość kątową)."""
    da = math.radians(2.0)

    def C(alpha=alpha0, beta=0.0, om=(0.0, 0.0, 0.0), defl=None):
        F, M, Fp, _ = lat.solve(alpha, beta, om, defl)
        return coeffs(F, M, alpha, S, b, c), Fp

    base, Fp0 = C()
    up, Fp1 = C(alpha=alpha0 + da)
    d = {}
    d["CL0"] = base["CL"] - (up["CL"] - base["CL"]) / da * alpha0
    d["CLa"] = (up["CL"] - base["CL"]) / da
    d["Cm0_vlm"] = base["Cm"] - (up["Cm"] - base["Cm"]) / da * alpha0
    d["Cma"] = (up["Cm"] - base["Cm"]) / da
    d["CDi_ref"] = base["CD"]
    qh = 0.01   # znormalizowana prędkość kątowa
    cq, _ = C(om=(0.0, 2 * qh / c, 0.0))
    d["CLq"] = (cq["CL"] - base["CL"]) / qh
    d["Cmq"] = (cq["Cm"] - base["Cm"]) / qh
    cb, _ = C(beta=da)
    d["CYb"], d["Clb"], d["Cnb"] = [(cb[k] - base[k]) / da for k in ("CY", "Cl", "Cn")]
    cp, _ = C(om=(2 * qh / b, 0.0, 0.0))
    d["CYp"], d["Clp"], d["Cnp"] = [(cp[k] - base[k]) / qh for k in ("CY", "Cl", "Cn")]
    cr, _ = C(om=(0.0, 0.0, 2 * qh / b))
    d["CYr"], d["Clr"], d["Cnr"] = [(cr[k] - base[k]) / qh for k in ("CY", "Cl", "Cn")]
    dd = math.radians(5.0)
    kinds = {c_.kind for s in lat.surfaces for c_ in s.controls}
    for kind, keys in (("elevator", ("CLde", "Cmde")), ("aileron", ("Clda", "Cnda")),
                       ("rudder", ("CYdr", "Cldr", "Cndr"))):
        if kind not in kinds:
            for k in keys:
                d[k] = 0.0
            continue
        cd, _ = C(defl={kind: dd})
        mp = {"CLde": "CL", "Cmde": "Cm", "Clda": "Cl", "Cnda": "Cn", "CYdr": "CY", "Cldr": "Cl", "Cndr": "Cn"}
        for k in keys:
            d[k] = (cd[mp[k]] - base[mp[k]]) / dd
    # rozkład siły nośnej na pasach (do wyznaczenia przeciągnięcia)
    lift_dir = np.array([math.sin(alpha0), 0.0, -math.cos(alpha0)])
    cl0, cl1, own = [], [], []
    for si, chord, width, ids in lat.strips:
        if lat.surfaces[si].vertical or width <= 0:
            continue
        L0 = float((Fp0[ids] @ lift_dir).sum())
        L1 = float((Fp1[ids] @ np.array([math.sin(alpha0 + da), 0.0, -math.cos(alpha0 + da)])).sum())
        cl0.append(L0 / (0.5 * chord * width))
        cl1.append((L1 / (0.5 * chord * width) - L0 / (0.5 * chord * width)) / da)
        own.append(si)
    return Derivatives(S, b, c, d, np.array(cl0), np.array(cl1), np.array(own))
