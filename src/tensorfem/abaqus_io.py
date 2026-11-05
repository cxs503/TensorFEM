"""Reader for a practical, deliberately bounded Abaqus keyword subset.

The subset is exactly what :class:`~tensorfem.modeldb.ModelDB` can represent.
Every other keyword is either accepted as a declared *non-model* no-op
(``NON_MODEL_KEYWORDS``) or rejected with a ``ValueError`` that names the
keyword and the line it appeared on.

Silent skipping is deliberately not an option: a dropped ``*INCLUDE`` removes
an arbitrary part of the mesh, and a dropped ``*SHELL SECTION`` removes the
plate thickness, yet both leave a file that still parses into a *plausible*
model.  For a ship panel that is a silent stiffness error of orders of
magnitude, so the reader fails closed instead.
"""

from __future__ import annotations

from pathlib import Path

from .modeldb import BoundaryCondition, ConcentratedLoad, ElementBlock, Material, ModelDB, Section

#: Keywords this reader turns into model data.
SUPPORTED_KEYWORDS = frozenset(
    {
        "NODE",
        "ELEMENT",
        "NSET",
        "ELSET",
        "MATERIAL",
        "ELASTIC",
        "SOLID SECTION",
        "TRUSS SECTION",
        "BOUNDARY",
        "CLOAD",
    }
)

#: Keywords that carry no model data for this bounded reader and are accepted
#: as no-ops.  ``END`` is the internal record terminator, never a real keyword.
NON_MODEL_KEYWORDS = frozenset(
    {
        "HEADING",
        "PREPRINT",
        "STEP",
        "END STEP",
        "STATIC",
        "OUTPUT",
        "NODE OUTPUT",
        "ELEMENT OUTPUT",
        "NODE PRINT",
        "EL PRINT",
        "CONTROLS",
        "RESTART",
        "SYSTEM",
        "PHYSICAL CONSTANTS",
        "END",
    }
)

#: Actionable hints for keywords that are commonly mistaken for supported ones.
_UNSUPPORTED_HINTS = {
    "INCLUDE": "external includes are not resolved; expand the deck before import",
}


def _options(line: str) -> tuple[str, dict[str, str]]:
    parts = [p.strip() for p in line[1:].split(",")]
    opts = {}
    for item in parts[1:]:
        key, sep, value = item.partition("=")
        opts[key.upper()] = value.strip() if sep else ""
    return parts[0].upper(), opts


def _required(opts: dict[str, str], key: str, keyword: str, lineno: int) -> str:
    """Return a required keyword option or fail closed with a located message."""
    if key not in opts:
        raise ValueError(f"*{keyword} on line {lineno} requires {key}=")
    return opts[key]


def _reject_unsupported_keyword(keyword: str, lineno: int) -> None:
    """Gate: only declared keywords may enter the model."""
    if keyword in SUPPORTED_KEYWORDS or keyword in NON_MODEL_KEYWORDS:
        return
    hint = _UNSUPPORTED_HINTS.get(keyword)
    if hint is None and keyword.endswith("SECTION"):
        hint = (
            "section keywords outside the bounded subset would silently lose "
            "their thickness/properties"
        )
    message = f"unsupported Abaqus keyword *{keyword} on line {lineno}"
    if hint:
        message += f" ({hint})"
    raise ValueError(message)


def _records(lines: list[str]):
    keyword, opts, data, at = None, {}, [], 0
    for lineno, raw in enumerate(lines + ["*END"], start=1):
        line = raw.strip()
        if not line or line.startswith("**"):
            continue
        if line.startswith("*"):
            if keyword is not None:
                yield at, keyword, opts, data
            keyword, opts = _options(line)
            data = []
            at = lineno
        else:
            data.append([x.strip() for x in line.split(",") if x.strip()])


def read_inp(path: str | Path) -> ModelDB:
    db, active_material = ModelDB(), None
    for lineno, keyword, opts, rows in _records(
        Path(path).read_text(encoding="utf-8").splitlines()
    ):
        _reject_unsupported_keyword(keyword, lineno)
        if keyword == "NODE":
            for row in rows:
                db.nodes[int(row[0])] = tuple(map(float, row[1:]))
            if "NSET" in opts:
                db.node_sets.setdefault(opts["NSET"].upper(), set()).update(int(r[0]) for r in rows)
        elif keyword == "ELEMENT":
            etype = opts.get("TYPE", "").upper()
            if etype not in {"T2D2", "CPS4", "CPE4", "C3D8"}:
                raise ValueError(f"unsupported Abaqus element type {etype!r} on line {lineno}")
            block = ElementBlock(
                etype,
                [int(r[0]) for r in rows],
                [[int(x) for x in r[1:]] for r in rows],
                opts.get("ELSET", ""),
            )
            db.elements.append(block)
            if block.name:
                db.element_sets.setdefault(block.name.upper(), set()).update(block.ids)
        elif keyword in ("NSET", "ELSET"):
            name = _required(opts, keyword, keyword, lineno).upper()
            values: set[int] = set()
            if "GENERATE" in opts:
                for row in rows:
                    start, stop = int(row[0]), int(row[1])
                    step = int(row[2]) if len(row) > 2 else 1
                    values.update(range(start, stop + 1, step))
            else:
                values.update(int(x) for row in rows for x in row)
            (db.node_sets if keyword == "NSET" else db.element_sets).setdefault(name, set()).update(
                values
            )
        elif keyword == "MATERIAL":
            active_material = _required(opts, "NAME", keyword, lineno).upper()
            db.materials[active_material] = Material(active_material)
        elif keyword == "ELASTIC":
            if active_material is None:
                raise ValueError(f"*ELASTIC on line {lineno} must follow *MATERIAL")
            db.materials[active_material].elastic = (float(rows[0][0]), float(rows[0][1]))
        elif keyword in ("SOLID SECTION", "TRUSS SECTION"):
            props = tuple(float(x) for row in rows for x in row)
            db.sections.append(
                Section(
                    _required(opts, "ELSET", keyword, lineno).upper(),
                    _required(opts, "MATERIAL", keyword, lineno).upper(),
                    "truss" if keyword == "TRUSS SECTION" else "solid",
                    props,
                )
            )
        elif keyword == "BOUNDARY":
            for r in rows:
                target = int(r[0]) if r[0].lstrip("+-").isdigit() else r[0].upper()
                db.boundaries.append(
                    BoundaryCondition(
                        target,
                        int(r[1]),
                        int(r[2]) if len(r) > 2 else int(r[1]),
                        float(r[3]) if len(r) > 3 else 0.0,
                    )
                )
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
        out.extend(
            ", ".join(map(str, [eid, *conn])) for eid, conn in zip(block.ids, block.connectivity)
        )
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
