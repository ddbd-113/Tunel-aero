"""Import samolotów z CAD: siatki STL/OBJ -> geometria -> model aerodynamiczny (VLM) -> pojazd do symulacji."""
from .aircraft import build_aircraft, is_cad_aircraft
from .mesh import Mesh, load_mesh, write_stl

__all__ = ["Mesh", "load_mesh", "write_stl", "build_aircraft", "is_cad_aircraft"]
