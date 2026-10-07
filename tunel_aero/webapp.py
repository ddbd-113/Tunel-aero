"""Budowa statycznej aplikacji webowej (GitHub Pages / Codespaces / lokalnie).

Strona zawiera interfejs (HTML/JS) i paczkę Pythona `py/tunel_aero.zip`, którą Pyodide
uruchamia w przeglądarce - symulacja liczy się na komputerze użytkownika, bez serwera.
"""
from __future__ import annotations

import hashlib
import http.server
import io
import shutil
import socketserver
import zipfile
from pathlib import Path

from .scenario import PACKAGE_ROOT

PKG = Path(__file__).resolve().parent
WEB = PKG / "web"


def _zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(PKG.rglob("*.py")):
            if "__pycache__" in f.parts:
                continue
            z.write(f, Path("tunel_aero") / f.relative_to(PKG))
        for sub in ("scenarios", "vehicles", "examples/moj_samolot"):
            d = PACKAGE_ROOT / sub
            for f in sorted(d.rglob("*")):
                if f.is_file() and f.suffix.lower() in (".yaml", ".yml", ".stl", ".obj"):
                    z.write(f, f.relative_to(PACKAGE_ROOT))
    return buf.getvalue()


def build_site(out="site") -> Path:
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    (out / "py").mkdir(parents=True)
    data = _zip_bytes()
    version = hashlib.sha1(data).hexdigest()[:10]
    (out / "py" / "tunel_aero.zip").write_bytes(data)
    for f in WEB.iterdir():
        if f.is_file():
            txt = f.read_text(encoding="utf-8").replace("__VERSION__", version)
            (out / f.name).write_text(txt, encoding="utf-8")
    ex = PACKAGE_ROOT / "examples" / "moj_samolot"
    dst = out / "data" / "examples" / "moj_samolot"
    dst.mkdir(parents=True)
    for f in ex.iterdir():
        if f.suffix.lower() in (".yaml", ".stl", ".obj"):
            shutil.copy(f, dst / f.name)
    (out / ".nojekyll").write_text("")
    return out


def serve(directory, port: int = 8000) -> None:
    directory = str(Path(directory).resolve())

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=directory, **kw)

        def log_message(self, *a):
            pass

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("0.0.0.0", port), Handler) as httpd:
        print(f"  Tunel-aero: http://localhost:{port}  (Ctrl+C - zatrzymaj)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
