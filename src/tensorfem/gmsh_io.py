"""Gmsh v2 ASCII mesh reader without external dependencies."""
from pathlib import Path

from .modeldb import ElementBlock, ModelDB

_GMSH = {1: ("T2D2", 2), 3: ("CPS4", 4), 4: ("C3D4", 4), 5: ("C3D8", 8)}


def read_msh(path: str | Path) -> ModelDB:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if "$MeshFormat" not in lines or not lines[lines.index("$MeshFormat") + 1].startswith("2."):
        raise ValueError("only Gmsh v2 ASCII is supported")
    db = ModelDB()
    i = lines.index("$Nodes") + 1
    count = int(lines[i]); i += 1
    for line in lines[i:i + count]:
        p = line.split(); db.nodes[int(p[0])] = tuple(map(float, p[1:4]))
    i = lines.index("$Elements") + 1
    count = int(lines[i]); i += 1
    by_type: dict[str, tuple[list[int], list[list[int]]]] = {}
    for line in lines[i:i + count]:
        p = list(map(int, line.split())); eid, code, ntags = p[:3]
        if code not in _GMSH:
            continue
        etype, nnode = _GMSH[code]
        conn = p[3 + ntags:3 + ntags + nnode]
        ids, conns = by_type.setdefault(etype, ([], [])); ids.append(eid); conns.append(conn)
    db.elements = [ElementBlock(t, ids, conn) for t, (ids, conn) in by_type.items()]
    db.validate()
    return db

