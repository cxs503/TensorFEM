"""Reader for a practical, deliberately bounded Abaqus keyword subset."""
from __future__ import annotations

from pathlib import Path

from .modeldb import (BoundaryCondition, ConcentratedLoad, ElementBlock, Material,
                      ModelDB, Section)


def _options(line: str) -> tuple[str, dict[str, str]]:
    parts = [p.strip() for p in line[1:].split(",")]
    opts = {}
    for item in parts[1:]:
        key, sep, value = item.partition("=")
        opts[key.upper()] = value.strip() if sep else ""
    return parts[0].upper(), opts


def _records(lines: list[str]):
    keyword, opts, data = None, {}, []
    for raw in lines + ["*END"]:
        line = raw.strip()
        if not line or line.startswith("**"):
            continue
        if line.startswith("*"):
            if keyword is not None:
                yield keyword, opts, data
            keyword, opts = _options(line)
            data = []
        else:
            data.append([x.strip() for x in line.split(",") if x.strip()])


def read_inp(path: str | Path) -> ModelDB:
    db, active_material = ModelDB(), None
    for keyword, opts, rows in _records(Path(path).read_text(encoding="utf-8").splitlines()):
        if keyword == "NODE":
            for row in rows:
                db.nodes[int(row[0])] = tuple(map(float, row[1:]))
            if "NSET" in opts:
                db.node_sets.setdefault(opts["NSET"].upper(), set()).update(int(r[0]) for r in rows)
        elif keyword == "ELEMENT":
            etype = opts.get("TYPE", "").upper()
            if etype not in {"T2D2", "CPS4", "CPE4", "C3D8"}:
                raise ValueError(f"unsupported Abaqus element type {etype}")
            block = ElementBlock(etype, [int(r[0]) for r in rows],
                                 [[int(x) for x in r[1:]] for r in rows], opts.get("ELSET", ""))
            db.elements.append(block)
            if block.name:
                db.element_sets.setdefault(block.name.upper(), set()).update(block.ids)
        elif keyword in ("NSET", "ELSET"):
            name = opts[keyword].upper()
            values: set[int] = set()
            if "GENERATE" in opts:
                for row in rows:
                    start, stop = int(row[0]), int(row[1]); step = int(row[2]) if len(row) > 2 else 1
                    values.update(range(start, stop + 1, step))
            else:
                values.update(int(x) for row in rows for x in row)
            (db.node_sets if keyword == "NSET" else db.element_sets).setdefault(name, set()).update(values)
        elif keyword == "MATERIAL":
            active_material = opts["NAME"].upper()
            db.materials[active_material] = Material(active_material)
        elif keyword == "ELASTIC":
            if active_material is None:
                raise ValueError("*ELASTIC must follow *MATERIAL")
            db.materials[active_material].elastic = (float(rows[0][0]), float(rows[0][1]))
        elif keyword in ("SOLID SECTION", "TRUSS SECTION"):
            props = tuple(float(x) for row in rows for x in row)
            db.sections.append(Section(opts["ELSET"].upper(), opts["MATERIAL"].upper(),
                                       "truss" if keyword == "TRUSS SECTION" else "solid", props))
        elif keyword == "BOUNDARY":
            for r in rows:
                target = int(r[0]) if r[0].lstrip("+-").isdigit() else r[0].upper()
                db.boundaries.append(BoundaryCondition(target, int(r[1]),
                                         int(r[2]) if len(r) > 2 else int(r[1]),
                                         float(r[3]) if len(r) > 3 else 0.0))
        elif keyword == "CLOAD":
            for r in rows:
                target = int(r[0]) if r[0].lstrip("+-").isdigit() else r[0].upper()
                db.loads.append(ConcentratedLoad(target, int(r[1]), float(r[2])))
    db.validate()
    return db


def write_inp(path: str | Path, db: ModelDB) -> None:
    """Write the supported subset, preserving external IDs and named sets."""
    db.validate()
    out = ["*HEADING", "TensorFEM ModelDB export", "*NODE"]
    for nid in sorted(db.nodes):
        out.append(", ".join([str(nid), *(f"{x:.17g}" for x in db.nodes[nid])]))
    for block in db.elements:
        heading = f"*ELEMENT, TYPE={block.element_type}"
        if block.name:
            heading += f", ELSET={block.name}"
        out.append(heading)
        out.extend(", ".join(map(str, [eid, *conn]))
                   for eid, conn in zip(block.ids, block.connectivity))
    for name, values in db.node_sets.items():
        out += [f"*NSET, NSET={name}", ", ".join(map(str, sorted(values)))]
    # Do not duplicate sets already emitted on *ELEMENT.
    implicit = {b.name.upper() for b in db.elements if b.name}
    for name, values in db.element_sets.items():
        if name.upper() not in implicit:
            out += [f"*ELSET, ELSET={name}", ", ".join(map(str, sorted(values)))]
    for material in db.materials.values():
        out.append(f"*MATERIAL, NAME={material.name}")
        if material.elastic is not None:
            out += ["*ELASTIC", f"{material.elastic[0]:.17g}, {material.elastic[1]:.17g}"]
    for section in db.sections:
        keyword = "TRUSS SECTION" if section.kind == "truss" else "SOLID SECTION"
        out.append(f"*{keyword}, ELSET={section.elset}, MATERIAL={section.material}")
        if section.properties:
            out.append(", ".join(f"{x:.17g}" for x in section.properties))
    if db.boundaries:
        out.append("*BOUNDARY")
        for bc in db.boundaries:
            out.append(f"{bc.target}, {bc.dof_start}, {bc.dof_end}, {bc.value:.17g}")
    if db.loads:
        out.append("*CLOAD")
        for load in db.loads:
            out.append(f"{load.target}, {load.dof}, {load.value:.17g}")
    Path(path).write_text("\n".join(out) + "\n", encoding="utf-8")
