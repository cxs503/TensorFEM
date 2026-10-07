"""Solver-neutral engineering model database and lightweight file adapters."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

SCHEMA = "tensorfem.model-db.v1"


@dataclass
class ElementBlock:
    element_type: str
    ids: list[int]
    connectivity: list[list[int]]
    name: str = ""


@dataclass
class Material:
    name: str
    elastic: tuple[float, float] | None = None


@dataclass
class Section:
    elset: str
    material: str
    kind: str = "solid"
    properties: tuple[float, ...] = ()


@dataclass
class BoundaryCondition:
    target: str | int
    dof_start: int
    dof_end: int
    value: float = 0.0


@dataclass
class ConcentratedLoad:
    target: str | int
    dof: int
    value: float


@dataclass
class AnalysisStep:
    name: str = "Step-1"
    kind: str = "static"


@dataclass
class OutputRequest:
    variables: tuple[str, ...] = ("U", "S")


@dataclass
class ModelDB:
    """Minimal, explicit model representation using external integer IDs."""

    nodes: dict[int, tuple[float, ...]] = field(default_factory=dict)
    elements: list[ElementBlock] = field(default_factory=list)
    node_sets: dict[str, set[int]] = field(default_factory=dict)
    element_sets: dict[str, set[int]] = field(default_factory=dict)
    materials: dict[str, Material] = field(default_factory=dict)
    sections: list[Section] = field(default_factory=list)
    boundaries: list[BoundaryCondition] = field(default_factory=list)
    loads: list[ConcentratedLoad] = field(default_factory=list)
    steps: list[AnalysisStep] = field(default_factory=lambda: [AnalysisStep()])
    output_requests: list[OutputRequest] = field(default_factory=lambda: [OutputRequest()])
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        node_ids = set(self.nodes)
        element_ids: set[int] = set()
        for block in self.elements:
            if len(block.ids) != len(block.connectivity):
                raise ValueError("element IDs and connectivity have different lengths")
            for eid, conn in zip(block.ids, block.connectivity):
                if eid in element_ids:
                    raise ValueError(f"duplicate element ID {eid}")
                element_ids.add(eid)
                missing = set(conn) - node_ids
                if missing:
                    raise ValueError(f"element {eid} references missing nodes {sorted(missing)}")
        if any(not ids <= node_ids for ids in self.node_sets.values()):
            raise ValueError("node set references a missing node")
        if any(not ids <= element_ids for ids in self.element_sets.values()):
            raise ValueError("element set references a missing element")
        for section in self.sections:
            if section.elset.upper() not in self.element_sets:
                raise ValueError(f"unknown section element set {section.elset}")
            if section.material.upper() not in self.materials:
                raise ValueError(f"unknown section material {section.material}")

    def expand_nodes(self, target: str | int) -> set[int]:
        return {target} if isinstance(target, int) else set(self.node_sets[target.upper()])

    def element_lookup(self) -> dict[int, tuple[str, list[int]]]:
        return {eid: (b.element_type, conn) for b in self.elements
                for eid, conn in zip(b.ids, b.connectivity)}


def modeldb_to_dict(db: ModelDB) -> dict[str, Any]:
    """Return a canonical, JSON-safe ModelDB representation.

    External identifiers are encoded as records rather than JSON object keys so
    integer IDs survive a round trip without coercion.
    """
    db.validate()
    return {
        "schema": SCHEMA,
        "nodes": [{"id": i, "coordinates": list(db.nodes[i])} for i in sorted(db.nodes)],
        "elements": [{"element_type": b.element_type, "ids": list(b.ids),
                      "connectivity": [list(c) for c in b.connectivity], "name": b.name}
                     for b in db.elements],
        "node_sets": {k: sorted(v) for k, v in sorted(db.node_sets.items())},
        "element_sets": {k: sorted(v) for k, v in sorted(db.element_sets.items())},
        "materials": {k: {"name": v.name, "elastic": None if v.elastic is None else list(v.elastic)}
                      for k, v in sorted(db.materials.items())},
        "sections": [{"elset": x.elset, "material": x.material, "kind": x.kind,
                      "properties": list(x.properties)} for x in db.sections],
        "boundaries": [{"target": x.target, "dof_start": x.dof_start,
                        "dof_end": x.dof_end, "value": x.value} for x in db.boundaries],
        "loads": [{"target": x.target, "dof": x.dof, "value": x.value} for x in db.loads],
        "steps": [{"name": x.name, "kind": x.kind} for x in db.steps],
        "output_requests": [{"variables": list(x.variables)} for x in db.output_requests],
        "metadata": db.metadata,
    }


def modeldb_from_dict(raw: Mapping[str, Any]) -> ModelDB:
    """Load the versioned representation, rejecting unknown executable input."""
    expected = {"schema", "nodes", "elements", "node_sets", "element_sets", "materials",
                "sections", "boundaries", "loads", "steps", "output_requests", "metadata"}
    if set(raw) != expected or raw.get("schema") != SCHEMA:
        raise ValueError("unknown fields or unsupported ModelDB schema")
    def exact(item, fields, label):
        if set(item) != set(fields): raise ValueError(f"unknown or missing {label} fields")
    for x in raw["nodes"]: exact(x, ("id", "coordinates"), "node")
    for x in raw["elements"]: exact(x, ("element_type", "ids", "connectivity", "name"), "element block")
    for x in raw["materials"].values(): exact(x, ("name", "elastic"), "material")
    maps = (("sections", Section, ("elset", "material", "kind", "properties")),
            ("boundaries", BoundaryCondition, ("target", "dof_start", "dof_end", "value")),
            ("loads", ConcentratedLoad, ("target", "dof", "value")),
            ("steps", AnalysisStep, ("name", "kind")),
            ("output_requests", OutputRequest, ("variables",)))
    made = {}
    for key, cls, fields in maps:
        for x in raw[key]: exact(x, fields, key)
        values = []
        for x in raw[key]:
            x = dict(x)
            tuple_field = "properties" if key == "sections" else "variables" if key == "output_requests" else None
            if tuple_field is not None: x[tuple_field] = tuple(x[tuple_field])
            values.append(cls(**x))
        made[key] = values
    db = ModelDB(
        nodes={int(x["id"]): tuple(x["coordinates"]) for x in raw["nodes"]},
        elements=[ElementBlock(x["element_type"], list(x["ids"]),
                               [list(c) for c in x["connectivity"]], x["name"])
                  for x in raw["elements"]],
        node_sets={str(k): set(v) for k, v in raw["node_sets"].items()},
        element_sets={str(k): set(v) for k, v in raw["element_sets"].items()},
        materials={str(k): Material(v["name"], None if v["elastic"] is None else tuple(v["elastic"]))
                   for k, v in raw["materials"].items()}, sections=made["sections"],
        boundaries=made["boundaries"], loads=made["loads"], steps=made["steps"],
        output_requests=made["output_requests"], metadata=dict(raw["metadata"]))
    db.validate(); return db


def to_truss_model(db: ModelDB, *, dtype: torch.dtype = torch.float64):
    """Adapt a 2-D Abaqus T2D2 database to TensorFEM's verified truss solver."""
    from .model import TrussModel

    db.validate()
    blocks = [b for b in db.elements if b.element_type.upper() == "T2D2"]
    if not blocks or sum(map(len, (b.ids for b in blocks))) != sum(len(b.ids) for b in db.elements):
        raise ValueError("truss adapter requires only T2D2 elements")
    node_ids = sorted(db.nodes)
    index = {nid: i for i, nid in enumerate(node_ids)}
    nodes = torch.tensor([db.nodes[n][:2] for n in node_ids], dtype=dtype)
    conn, young, area = [], [], []
    section_by_eid: dict[int, Section] = {}
    for section in db.sections:
        for eid in db.element_sets[section.elset.upper()]:
            section_by_eid[eid] = section
    for block in blocks:
        for eid, c in zip(block.ids, block.connectivity):
            section = section_by_eid.get(eid)
            if section is None:
                raise ValueError(f"element {eid} has no section")
            material = db.materials[section.material.upper()]
            if material.elastic is None or not section.properties:
                raise ValueError("truss material/section requires elastic E and area")
            conn.append([index[c[0]], index[c[1]]])
            young.append(material.elastic[0])
            area.append(section.properties[0])
    force = torch.zeros(2 * len(node_ids), dtype=dtype)
    for load in db.loads:
        for nid in db.expand_nodes(load.target):
            if load.dof not in (1, 2):
                raise ValueError("T2D2 supports load DOFs 1 and 2")
            force[2 * index[nid] + load.dof - 1] += load.value
    prescribed, fixed = {}, []
    for bc in db.boundaries:
        for nid in db.expand_nodes(bc.target):
            for dof in range(bc.dof_start, bc.dof_end + 1):
                if dof not in (1, 2):
                    raise ValueError("T2D2 supports boundary DOFs 1 and 2")
                prescribed[2 * index[nid] + dof - 1] = bc.value
    if any(v != 0.0 for v in prescribed.values()):
        raise ValueError("current truss adapter supports zero prescribed displacement only")
    fixed.extend(sorted(prescribed))
    return TrussModel(nodes, torch.tensor(conn, dtype=torch.long),
                      torch.tensor(young, dtype=dtype), torch.tensor(area, dtype=dtype),
                      force, torch.tensor(fixed, dtype=torch.long)), node_ids
