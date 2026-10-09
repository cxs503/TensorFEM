"""Dependency-free VTK legacy output and optional HDF5 result output."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch

from .modeldb import ModelDB

_VTK_CELL = {"T2D2": 3, "CPS4": 9, "CPE4": 9, "C3D8": 12}


def _values(value):
    return value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else value


def write_vtk(path: str | Path, db: ModelDB, *, point_data: Mapping[str, object] | None = None,
              cell_data: Mapping[str, object] | None = None) -> None:
    db.validate(); point_data = point_data or {}; cell_data = cell_data or {}
    node_ids = sorted(db.nodes); idx = {n: i for i, n in enumerate(node_ids)}
    cells = [(b.element_type.upper(), c) for b in db.elements for c in b.connectivity]
    unsupported = {t for t, _ in cells} - set(_VTK_CELL)
    if unsupported:
        raise ValueError(f"unsupported VTK cell types {sorted(unsupported)}")
    out = ["# vtk DataFile Version 3.0", "TensorFEM results", "ASCII", "DATASET UNSTRUCTURED_GRID",
           f"POINTS {len(node_ids)} double"]
    for nid in node_ids:
        xyz = list(db.nodes[nid]) + [0.0] * (3 - len(db.nodes[nid])); out.append(" ".join(map(str, xyz[:3])))
    out.append(f"CELLS {len(cells)} {sum(len(c) + 1 for _, c in cells)}")
    out.extend(f"{len(c)} " + " ".join(str(idx[n]) for n in c) for _, c in cells)
    out += [f"CELL_TYPES {len(cells)}"] + [str(_VTK_CELL[t]) for t, _ in cells]
    if point_data:
        out.append(f"POINT_DATA {len(node_ids)}")
        for name, raw in point_data.items():
            data = _values(raw)
            if len(data) != len(node_ids): raise ValueError(f"point field {name} has wrong length")
            if data and isinstance(data[0], (list, tuple)):
                if len(data[0]) not in (2, 3): raise ValueError("VTK vector fields require 2 or 3 components")
                out.append(f"VECTORS {name} double")
                for v in data: out.append(" ".join(map(str, list(v) + [0.0] * (3 - len(v)))))
            else:
                out += [f"SCALARS {name} double 1", "LOOKUP_TABLE default"] + [str(v) for v in data]
    if cell_data:
        out.append(f"CELL_DATA {len(cells)}")
        for name, raw in cell_data.items():
            data = _values(raw)
            if len(data) != len(cells): raise ValueError(f"cell field {name} has wrong length")
            out += [f"SCALARS {name} double 1", "LOOKUP_TABLE default"] + [str(v) for v in data]
    Path(path).write_text("\n".join(out) + "\n", encoding="utf-8")


def write_hdf5(path: str | Path, db: ModelDB, **fields: object) -> bool:
    """Write compact results when h5py is available; return False otherwise."""
    try:
        import h5py
    except ImportError:
        return False
    with h5py.File(path, "w") as h5:
        ids = sorted(db.nodes)
        h5.create_dataset("node_ids", data=ids)
        h5.create_dataset("nodes", data=[db.nodes[i] for i in ids])
        for name, value in fields.items(): h5.create_dataset(name, data=_values(value))
    return True
