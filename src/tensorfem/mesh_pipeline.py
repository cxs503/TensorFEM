"""Dependency-free mesh import, engineering diagnostics and conversion.

The diagnostics are intentionally fail-closed: an invalid mesh produces a
machine-readable report, but cannot be certified with :meth:`require_valid`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Iterable

import torch

from .abaqus_io import write_inp
from .modeldb import ElementBlock, ModelDB
from .result_io import write_vtk


@dataclass(frozen=True)
class Diagnostic:
    severity: str
    code: str
    message: str
    entities: tuple[int, ...] = ()


@dataclass
class MeshReport:
    diagnostics: list[Diagnostic] = field(default_factory=list)
    metrics: dict[str, float | int] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return not any(d.severity == "error" for d in self.diagnostics)

    def require_valid(self) -> "MeshReport":
        if not self.valid:
            codes = sorted({d.code for d in self.diagnostics if d.severity == "error"})
            raise ValueError("mesh/model diagnostics failed: " + ", ".join(codes))
        return self

    def to_dict(self) -> dict[str, object]:
        return {"valid": self.valid, "summary": {
            "errors": sum(d.severity == "error" for d in self.diagnostics),
            "warnings": sum(d.severity == "warning" for d in self.diagnostics)},
            "metrics": self.metrics,
            "diagnostics": [asdict(d) for d in self.diagnostics]}

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")


_NNODE = {"T2D2": 2, "CPS3": 3, "CPE3": 3, "CPS4": 4, "CPE4": 4,
          "C3D4": 4, "C3D8": 8}
_GMSH4 = {1: ("T2D2", 2), 2: ("CPS3", 3), 3: ("CPS4", 4),
          4: ("C3D4", 4), 5: ("C3D8", 8)}


def _xyz(db: ModelDB, conn: Iterable[int]) -> torch.Tensor:
    return torch.tensor([list(db.nodes[n]) + [0.] * (3-len(db.nodes[n])) for n in conn],
                        dtype=torch.float64)[:, :3]


def _hex_dets(x: torch.Tensor) -> list[float]:
    signs = torch.tensor([[-1.,-1.,-1.],[1.,-1.,-1.],[1.,1.,-1.],[-1.,1.,-1.],
                          [-1.,-1.,1.],[1.,-1.,1.],[1.,1.,1.],[-1.,1.,1.]])
    q = 1 / math.sqrt(3)
    out = []
    for r in (-q, q):
        for s in (-q, q):
            for t in (-q, q):
                d = torch.empty((8, 3), dtype=torch.float64)
                d[:, 0] = signs[:, 0]*(1+s*signs[:, 1])*(1+t*signs[:, 2])/8
                d[:, 1] = signs[:, 1]*(1+r*signs[:, 0])*(1+t*signs[:, 2])/8
                d[:, 2] = signs[:, 2]*(1+r*signs[:, 0])*(1+s*signs[:, 1])/8
                out.append(float(torch.linalg.det(x.T @ d)))
    return out


def diagnose_model(db: ModelDB, *, coordinate_tolerance: float = 1e-12,
                   max_aspect: float = 100., max_skew_degrees: float = 75.) -> MeshReport:
    """Inspect topology, geometry and analysis-definition completeness."""
    r = MeshReport(metrics={"nodes": len(db.nodes),
                            "elements": sum(len(b.ids) for b in db.elements)})
    add = lambda sev, code, msg, ids=(): r.diagnostics.append(
        Diagnostic(sev, code, msg, tuple(int(i) for i in ids)))
    if not db.nodes: add("error", "empty_mesh", "model has no nodes")
    if not db.elements: add("error", "empty_mesh", "model has no elements")

    # ID/connectivity integrity (collect rather than relying on ModelDB.validate).
    used: set[int] = set(); eids: set[int] = set(); all_eids: set[int] = set()
    faces: dict[tuple[int, ...], list[int]] = {}
    min_jac = math.inf; max_aspect_seen = 1.; max_skew_seen = 0.
    for b in db.elements:
        et = b.element_type.upper()
        if len(b.ids) != len(b.connectivity):
            add("error", "block_length", f"block {et} has mismatched IDs/connectivity")
        for eid, conn in zip(b.ids, b.connectivity):
            if eid in eids: add("error", "duplicate_element_id", f"duplicate element {eid}", [eid])
            eids.add(eid); all_eids.add(eid); used.update(conn)
            missing = set(conn) - set(db.nodes)
            if missing:
                add("error", "dangling_connectivity", f"element {eid} references missing nodes", [eid, *sorted(missing)])
                continue
            if et not in _NNODE:
                add("warning", "unsupported_quality", f"no quality rules for {et}", [eid]); continue
            if len(conn) != _NNODE[et]:
                add("error", "wrong_node_count", f"element {eid} {et} expects {_NNODE[et]} nodes", [eid]); continue
            if len(set(conn)) != len(conn):
                add("error", "repeated_element_node", f"element {eid} repeats a node", [eid]); continue
            x = _xyz(db, conn)
            if et == "T2D2":
                lengths = [float(torch.linalg.vector_norm(x[1]-x[0]))]; jac = lengths[0]
                edge_faces = []
            elif et in {"CPS3", "CPE3"}:
                jac = float(torch.cross(x[1]-x[0], x[2]-x[0], dim=0)[2])
                lengths = [float(torch.linalg.vector_norm(x[(i+1)%3]-x[i])) for i in range(3)]
                edge_faces = [(conn[i],conn[(i+1)%3]) for i in range(3)]
            elif et in {"CPS4", "CPE4"}:
                # Signed Jacobian at every 2x2 integration point.
                dets=[]; q=1/math.sqrt(3)
                signs=torch.tensor([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]],dtype=torch.float64)
                for rr in (-q,q):
                    for ss in (-q,q):
                        d=torch.empty((4,2),dtype=torch.float64)
                        d[:,0]=signs[:,0]*(1+ss*signs[:,1])/4
                        d[:,1]=signs[:,1]*(1+rr*signs[:,0])/4
                        dets.append(float(torch.linalg.det(x[:,:2].T@d)))
                jac=min(dets)
                lengths = [float(torch.linalg.vector_norm(x[(i+1)%4]-x[i])) for i in range(4)]
                edge_faces = [(conn[i],conn[(i+1)%4]) for i in range(4)]
                angles=[]
                for i in range(4):
                    a=x[(i-1)%4]-x[i]; c=x[(i+1)%4]-x[i]
                    angles.append(math.degrees(math.acos(float(torch.clamp(torch.dot(a,c)/(torch.linalg.vector_norm(a)*torch.linalg.vector_norm(c)),-1,1)))))
                skew=max(abs(a-90.) for a in angles); max_skew_seen=max(max_skew_seen,skew)
                if skew > max_skew_degrees: add("error", "excessive_skew", f"element {eid} skew {skew:.3g} deg", [eid])
            elif et == "C3D4":
                jac = float(torch.linalg.det(torch.stack((x[1]-x[0],x[2]-x[0],x[3]-x[0]),1)))
                lengths=[float(torch.linalg.vector_norm(x[j]-x[i])) for i,j in ((0,1),(0,2),(0,3),(1,2),(1,3),(2,3))]
                edge_faces=[(conn[i],conn[j],conn[k]) for i,j,k in ((0,2,1),(0,1,3),(1,2,3),(2,0,3))]
            else:
                dets=_hex_dets(x); jac=min(dets)
                lengths=[float(torch.linalg.vector_norm(x[j]-x[i])) for i,j in ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))]
                edge_faces=[tuple(conn[i] for i in f) for f in ((0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7))]
            min_jac=min(min_jac,jac)
            if jac <= coordinate_tolerance:
                add("error", "inverted_or_degenerate", f"element {eid} minimum Jacobian {jac:.6g}", [eid])
            mn=min(lengths); aspect=math.inf if mn <= coordinate_tolerance else max(lengths)/mn
            max_aspect_seen=max(max_aspect_seen,aspect)
            if aspect > max_aspect: add("error", "excessive_aspect", f"element {eid} aspect {aspect:.6g}", [eid])
            for f in edge_faces: faces.setdefault(tuple(sorted(f)), []).append(eid)
    orphan = sorted(set(db.nodes)-used)
    if orphan: add("warning", "orphan_nodes", f"{len(orphan)} nodes are unused", orphan)

    # Geometrically coincident nodes using quantized buckets.
    buckets: dict[tuple[int, ...], list[int]] = {}
    if coordinate_tolerance <= 0: raise ValueError("coordinate_tolerance must be positive")
    for nid, p in db.nodes.items():
        key=tuple(round(float(v)/coordinate_tolerance) for v in (*p, *([0.]*(3-len(p)))))
        buckets.setdefault(key,[]).append(nid)
    for ids in buckets.values():
        if len(ids)>1: add("error", "duplicate_coordinates", "coincident nodes", sorted(ids))
    for face, owners in faces.items():
        if len(owners)>2: add("error", "nonmanifold_face", f"face {face} has {len(owners)} owners", owners)

    # Sets, sections, materials and analysis definition.
    for name, ids in db.node_sets.items():
        if not ids: add("warning", "empty_node_set", f"node set {name} is empty")
        missing=ids-set(db.nodes)
        if missing: add("error", "invalid_node_set", f"node set {name} contains missing IDs", missing)
    for name, ids in db.element_sets.items():
        if not ids: add("warning", "empty_element_set", f"element set {name} is empty")
        missing=ids-all_eids
        if missing: add("error", "invalid_element_set", f"element set {name} contains missing IDs", missing)
    assignment: dict[int,int]={}
    if not db.materials: add("warning", "no_materials", "model has no material definitions")
    if not db.sections: add("warning", "no_sections", "model has no section assignments")
    for sec in db.sections:
        key=sec.elset.upper()
        if key not in db.element_sets: add("error", "missing_section_set", f"section refers to unknown set {key}"); continue
        if sec.material.upper() not in db.materials: add("error", "missing_material", f"section refers to unknown material {sec.material}")
        elif db.materials[sec.material.upper()].elastic is None: add("error", "incomplete_material", f"material {sec.material} has no elastic data")
        for eid in db.element_sets[key]: assignment[eid]=assignment.get(eid,0)+1
    if db.sections:
        missing=all_eids-set(assignment)
        if missing: add("error", "unassigned_section", "elements have no section", missing)
        duplicate=[e for e,n in assignment.items() if n>1]
        if duplicate: add("error", "multiple_sections", "elements have multiple sections", duplicate)
    prescribed: dict[tuple[int,int],float]={}
    for bc in db.boundaries:
        if bc.dof_start < 1 or bc.dof_end < bc.dof_start:
            add("error", "invalid_boundary_dof",
                f"invalid boundary DOF range {bc.dof_start}:{bc.dof_end}")
            continue
        try: targets={bc.target} if isinstance(bc.target,int) else db.node_sets[bc.target.upper()]
        except KeyError: add("error","unknown_bc_target",f"unknown boundary target {bc.target}"); continue
        for nid in targets:
            if nid not in db.nodes: add("error","unknown_bc_target",f"boundary references node {nid}",[nid]); continue
            for dof in range(bc.dof_start,bc.dof_end+1):
                key=(nid,dof)
                if key in prescribed and prescribed[key] != bc.value: add("error","conflicting_boundary",f"conflicting boundary at node {nid} DOF {dof}",[nid])
                prescribed[key]=bc.value
    for load in db.loads:
        if not math.isfinite(load.value): add("error","nonfinite_load","load is not finite")
        if load.dof < 1: add("error","invalid_load_dof",f"load DOF must be positive, got {load.dof}")
        if isinstance(load.target,str) and load.target.upper() not in db.node_sets: add("error","unknown_load_target",f"unknown load target {load.target}")
        if isinstance(load.target,int) and load.target not in db.nodes: add("error","unknown_load_target",f"load references node {load.target}",[load.target])
    if not db.boundaries: add("warning", "no_constraints", "model has no boundary conditions")
    if not db.loads: add("warning", "no_loads", "model has no concentrated loads")
    if "length_unit" not in db.metadata: add("warning", "units_unspecified", "metadata.length_unit is not defined")
    elif not isinstance(db.metadata["length_unit"], str) or not db.metadata["length_unit"].strip():
        add("error", "invalid_units", "metadata.length_unit must be a non-empty string")
    r.metrics.update({"orphan_nodes":len(orphan), "minimum_jacobian":min_jac if math.isfinite(min_jac) else 0.,
                      "maximum_aspect":max_aspect_seen, "maximum_skew_degrees":max_skew_seen})
    return r


def read_msh4(path: str | Path) -> ModelDB:
    """Read the common Gmsh 4.1 ASCII nodes/elements subset."""
    lines=Path(path).read_text(encoding="utf-8").splitlines()
    try: fmt=lines[lines.index("$MeshFormat")+1].split()
    except (ValueError,IndexError): raise ValueError("missing Gmsh $MeshFormat") from None
    if not fmt[0].startswith("4.") or fmt[1] != "0": raise ValueError("requires Gmsh v4 ASCII")
    db=ModelDB(); i=lines.index("$Nodes")+1; header=list(map(int,lines[i].split())); i+=1
    for _ in range(header[0]):
        entity_dim,entity_tag,parametric,nblock=map(int,lines[i].split()); i+=1
        tags=[int(lines[i+j].strip()) for j in range(nblock)]; i+=nblock
        for nid in tags:
            vals=list(map(float,lines[i].split())); i+=1
            db.nodes[nid]=tuple(vals[:3])
            if parametric and len(vals)<3+entity_dim: raise ValueError("truncated parametric node")
    i=lines.index("$Elements")+1; header=list(map(int,lines[i].split())); i+=1
    grouped: dict[str,tuple[list[int],list[list[int]]]]={}
    for _ in range(header[0]):
        entity_dim,entity_tag,code,nblock=map(int,lines[i].split()); i+=1
        supported=_GMSH4.get(code)
        for _ in range(nblock):
            row=list(map(int,lines[i].split())); i+=1
            if supported:
                et,n=supported
                if len(row)!=n+1: raise ValueError(f"Gmsh element {row[0]} has wrong node count")
                ids,conns=grouped.setdefault(et,([],[])); ids.append(row[0]); conns.append(row[1:])
    db.elements=[ElementBlock(et,ids,conns) for et,(ids,conns) in grouped.items()]
    if sum(map(len,(b.ids for b in db.elements))) == 0: raise ValueError("no supported Gmsh elements")
    db.validate(); return db


def read_gmsh(path: str | Path) -> ModelDB:
    """Version-dispatching dependency-free Gmsh ASCII reader."""
    text=Path(path).read_text(encoding="utf-8")
    line=text.split("$MeshFormat",1)[1].splitlines()[1].split()[0] if "$MeshFormat" in text else ""
    if line.startswith("4."): return read_msh4(path)
    from .gmsh_io import read_msh
    return read_msh(path)


def read_with_meshio(path: str | Path) -> ModelDB | None:
    """Optional meshio adapter. Return ``None`` when meshio is unavailable."""
    try: import meshio
    except ImportError: return None
    mesh=meshio.read(path); db=ModelDB()
    db.nodes={i+1:tuple(map(float,p)) for i,p in enumerate(mesh.points)}
    mapping={"line":"T2D2","triangle":"CPS3","quad":"CPS4","tetra":"C3D4","hexahedron":"C3D8"}
    eid=1
    for block in mesh.cells:
        if block.type not in mapping: continue
        conns=[[int(n)+1 for n in row] for row in block.data.tolist()]
        ids=list(range(eid,eid+len(conns))); eid+=len(conns)
        db.elements.append(ElementBlock(mapping[block.type],ids,conns))
    db.validate(); return db


def convert_model(db: ModelDB, path: str | Path) -> None:
    """Validate and convert ModelDB to Abaqus INP or legacy VTK."""
    diagnose_model(db).require_valid()
    suffix=Path(path).suffix.lower()
    if suffix == ".inp": write_inp(path,db)
    elif suffix == ".vtk": write_vtk(path,db)
    else: raise ValueError("conversion output must be .inp or .vtk")
