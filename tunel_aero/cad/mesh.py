"""Siatki trójkątne z CAD (STL binarny/ASCII, OBJ): wczytywanie, transformacje jednostek i osi,
właściwości masowe, przekroje płaszczyznami, upraszczanie do wizualizacji."""
from __future__ import annotations

import io
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

UNITS = {"m": 1.0, "cm": 0.01, "mm": 0.001, "in": 0.0254, "ft": 0.3048}
AXES = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0), "+z": (0, 0, 1), "-z": (0, 0, -1)}


@dataclass
class Mesh:
    v: np.ndarray       # (n, 3) wierzchołki
    f: np.ndarray       # (m, 3) indeksy trójkątów
    name: str = ""

    @property
    def tri(self) -> np.ndarray:
        return self.v[self.f]

    @property
    def n_tri(self) -> int:
        return len(self.f)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.v.min(axis=0), self.v.max(axis=0)

    @property
    def extent(self) -> np.ndarray:
        lo, hi = self.bounds
        return hi - lo

    def area(self) -> float:
        t = self.tri
        return float(0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum())

    def volume(self) -> float:
        t = self.tri
        return float(np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6.0)

    def transformed(self, R: np.ndarray, scale: float = 1.0, offset=None) -> "Mesh":
        v = (self.v * scale) @ R.T
        if offset is not None:
            v = v - np.asarray(offset, float)
        return Mesh(v, self.f.copy(), self.name)

    def merged(self, other: "Mesh") -> "Mesh":
        return Mesh(np.vstack((self.v, other.v)), np.vstack((self.f, other.f + len(self.v))), self.name)

    # ------------------------------------------------------- masa
    def second_moments(self, shell: bool = False) -> tuple[float, np.ndarray, np.ndarray]:
        """Zwraca (miara, moment 1. rzędu, macierz momentów 2. rzędu) względem początku układu.

        shell=False - bryła (miara = objętość, siatka musi być zamknięta),
        shell=True  - cienka powłoka (miara = pole powierzchni; np. pianka/kompozyt).
        """
        t = self.tri
        a, b, c = t[:, 0], t[:, 1], t[:, 2]
        s = a + b + c
        if shell:
            w = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
            first = (w[:, None] * s / 3.0).sum(axis=0)
            C = np.einsum("i,ij,ik->jk", w / 12.0, a, a) + np.einsum("i,ij,ik->jk", w / 12.0, b, b) \
                + np.einsum("i,ij,ik->jk", w / 12.0, c, c) + np.einsum("i,ij,ik->jk", w / 12.0, s, s)
            return float(w.sum()), first, C
        w = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0     # objętości czworościanów (0, a, b, c)
        first = (w[:, None] * s / 4.0).sum(axis=0)
        C = np.einsum("i,ij,ik->jk", w / 20.0, a, a) + np.einsum("i,ij,ik->jk", w / 20.0, b, b) \
            + np.einsum("i,ij,ik->jk", w / 20.0, c, c) + np.einsum("i,ij,ik->jk", w / 20.0, s, s)
        meas = float(w.sum())
        if meas < 0:   # odwrócona orientacja trójkątów
            return -meas, -first, -C
        return meas, first, C

    def mass_properties(self, mass: float, shell: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """(środek masy, macierz momentów 2. rzędu masy względem początku układu)."""
        meas, first, C = self.second_moments(shell)
        if meas <= 1e-15 and not shell:   # siatka niezamknięta - przejdź na model powłokowy
            meas, first, C = self.second_moments(True)
        k = mass / max(meas, 1e-15)
        return first / max(meas, 1e-15), C * k

    # ------------------------------------------------------- przekroje
    def section_points(self, axis: int, value: float) -> np.ndarray:
        """Punkty przecięcia siatki z płaszczyzną x_axis = value."""
        t = self.tri
        d = t[:, :, axis] - value
        pts = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            da, db = d[:, i], d[:, j]
            m = da * db < 0
            if m.any():
                a, b = t[m, i], t[m, j]
                r = (da[m] / (da[m] - db[m]))[:, None]
                pts.append(a + (b - a) * r)
        return np.vstack(pts) if pts else np.zeros((0, 3))

    # ------------------------------------------------------- upraszczanie
    def decimated(self, max_tris: int = 6000) -> "Mesh":
        """Upraszczanie przez klasteryzację wierzchołków (do podglądu 3D)."""
        if self.n_tri <= max_tris:
            return self
        diag = float(np.linalg.norm(self.extent)) or 1.0
        res = 200
        best = self
        for _ in range(12):
            cell = diag / res
            keys = np.floor((self.v - self.v.min(axis=0)) / cell).astype(np.int64)
            uniq, inv = np.unique(keys, axis=0, return_inverse=True)
            inv = inv.reshape(-1)
            nv = np.zeros((len(uniq), 3))
            np.add.at(nv, inv, self.v)
            nv /= np.bincount(inv, minlength=len(uniq))[:, None]
            f = inv[self.f]
            ok = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
            m = Mesh(nv, f[ok], self.name)
            best = m
            if m.n_tri <= max_tris:
                return m
            res = int(res * 0.75)
        return best


# ----------------------------------------------------------- odczyt / zapis
def _dedup(tri: np.ndarray, name: str) -> Mesh:
    pts = tri.reshape(-1, 3)
    scale = float(np.abs(pts).max()) or 1.0
    q = np.round(pts / scale * 1e7).astype(np.int64)
    _, idx, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    return Mesh(pts[idx], inv.reshape(-1, 3), name)


def parse_stl(data: bytes, name: str = "") -> Mesh:
    if len(data) >= 84:
        n = struct.unpack("<I", data[80:84])[0]
        if 84 + 50 * n == len(data):
            rec = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]),
                                count=n, offset=84)
            return _dedup(rec["v"].astype(float), name)
    text = data.decode("utf-8", errors="ignore")
    vals = [line.split()[1:4] for line in text.splitlines() if line.strip().startswith("vertex")]
    if not vals:
        raise ValueError("Nie rozpoznano pliku STL (ani binarny, ani ASCII)")
    tri = np.array(vals, dtype=float).reshape(-1, 3, 3)
    return _dedup(tri, name)


def parse_obj(data: bytes, name: str = "", obj: str | None = None) -> Mesh:
    """OBJ z obsługą obiektów/grup (o/g) - obj wybiera konkretną część."""
    verts, faces = [], []
    current, active = "", obj is None
    for line in data.decode("utf-8", errors="ignore").splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == "v":
            verts.append([float(x) for x in p[1:4]])
        elif p[0] in ("o", "g"):
            current = " ".join(p[1:])
            active = obj is None or current == obj
        elif p[0] == "f" and active:
            idx = [int(x.split("/")[0]) for x in p[1:]]
            idx = [i - 1 if i > 0 else len(verts) + i for i in idx]
            for k in range(1, len(idx) - 1):
                faces.append([idx[0], idx[k], idx[k + 1]])
    if not faces:
        raise ValueError(f"Brak ścian w pliku OBJ{' dla obiektu ' + obj if obj else ''}")
    v = np.array(verts, float)
    f = np.array(faces, int)
    used = np.unique(f)
    remap = np.full(len(v), -1)
    remap[used] = np.arange(len(used))
    return Mesh(v[used], remap[f], name or (obj or ""))


def load_mesh(path, data: bytes | None = None) -> Mesh:
    """Wczytuje STL/OBJ. Dla OBJ można wskazać obiekt: 'samolot.obj#skrzydlo'."""
    path = str(path)
    obj = None
    if "#" in path:
        path, obj = path.split("#", 1)
    raw = data if data is not None else Path(path).read_bytes()
    ext = Path(path).suffix.lower()
    if ext == ".obj":
        return parse_obj(raw, Path(path).stem, obj)
    if ext == ".stl":
        return parse_stl(raw, Path(path).stem)
    raise ValueError(f"Nieobsługiwany format '{ext}'. Wyeksportuj z CAD plik STL lub OBJ.")


def write_stl(path, mesh: Mesh, header: str = "Tunel-aero") -> None:
    t = mesh.tri.astype("<f4")
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(t), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    rec["n"] = n
    rec["v"] = t
    buf = io.BytesIO()
    buf.write(header.encode()[:80].ljust(80, b" "))
    buf.write(struct.pack("<I", len(t)))
    buf.write(rec.tobytes())
    Path(path).write_bytes(buf.getvalue())


def axes_matrix(forward: str = "-x", up: str = "+z") -> np.ndarray:
    """Macierz CAD -> układ ciała FRD (x - nos, y - prawe skrzydło, z - w dół)."""
    f = np.array(AXES[forward], float)
    u = np.array(AXES[up], float)
    if abs(f @ u) > 1e-9:
        raise ValueError("Oś 'forward' i 'up' muszą być prostopadłe")
    right = np.cross(-u, f)
    return np.vstack((f, right, -u))
