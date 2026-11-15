"""Pointwise reference regression for finite-element fields.

Motivation
----------
Every acceptance gate in this repository is a *scalar*: tip deflection, peak
von Mises stress, one probe value.  Those gates are necessary but blind to a
whole class of defects: a refactor of the assembly path, a changed quadrature
rule, a swapped sparse solver or a reordered DOF map can perturb the *whole
field* by a fraction of a percent while every scalar gate still passes.  The
pointwise reference-regression practice used by the lattice-Boltzmann codes
(``lbmpy`` keeps a reference dataset and compares individual cell values with a
tolerance, so any change to the kernel is caught at the point where it happens)
is the missing complement: compare the *field itself*, point by point, against
a stored reference captured from a known-good run.

What this module provides
-------------------------
* :class:`Snapshot` - an immutable, JSON-serialisable set of probe samples for
  one case: ``(probe_id, kind, coordinates, {quantity: value})`` plus provenance
  ``meta`` (including a mesh signature, so a probe can never be silently
  compared against a different mesh).
* :func:`build_snapshot` - validated construction (canonical probe ordering,
  unique ids, identical quantity set on every probe, finite values).
* :func:`mesh_signature` - SHA-256 over the float64 node/connectivity bytes;
  pins the discretisation a reference was captured on.
* :func:`compare` - two modes of the same machinery:
  ``regression`` (tight ``rtol``, drift detection against a stored run) and
  ``reference`` (acceptance gate against an analytic/literature value).  The
  aggregation is :func:`tensorfem.certification_evidence.aggregate_verdicts`, so
  a pointwise row carries the same ``PASS`` / ``FAIL`` / ``WITHHELD`` semantics -
  including the rule that rows lacking a well-posed criterion downgrade a clean
  pass to ``PASS_WITH_WITHHELD`` instead of silently passing.  The row rule
  itself is the relative criterion of
  :func:`~tensorfem.certification_evidence.within_tolerance` extended with an
  absolute floor (see "Design rules" below).
* :func:`verify_directory` / :func:`main` - walk a store of reference JSON
  files, rebuild each case from a builder registry, and emit a machine-readable
  ledger.  ``python -m tensorfem.pointwise_regression check --store DIR`` is the
  CI entry point; ``update`` is the deliberate, explicit re-baselining.

Design rules
------------
1. No numbers are hard-coded here: tolerances, probes and values are inputs.
2. A structural mismatch (different case id, probe set, coordinates or
   discretisation signature) is an *error*, not a ``FAIL``; comparing fields on
   different meshes is meaningless, and reporting it as a tolerance failure
   would hide the real problem.
3. A missing/extra quantity is likewise an error; a silently dropped quantity
   would shrink the regression surface without anyone noticing.
4. The row criterion is ``|v - ref| <= rtol*|ref| + atol``, i.e. the
   repository's relative criterion plus an absolute floor.  A purely relative
   criterion cannot test a component that is zero by symmetry (``sigma_yy`` of a
   uniaxial patch test): it would have to be withheld, and a genuine defect
   there - a spurious transverse stress of 1 % of the load scale - would escape
   the regression entirely.  ``atol`` defaults to ``1e-12`` times the largest
   magnitude of *that quantity in the reference data*, so the floor scales with
   the data instead of being an invented constant, and every row reports both
   the band and its utilisation ``|v-ref| / band``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch

from .certification_evidence import FAIL, PASS, WITHHELD, aggregate_verdicts

__all__ = [
    "SCHEMA_VERSION",
    "Tolerance",
    "ProbeSample",
    "Snapshot",
    "build_snapshot",
    "mesh_signature",
    "sample_nodes",
    "sample_element_centres",
    "compare",
    "dumps",
    "loads",
    "save",
    "load",
    "verify_directory",
    "write_ledger",
    "main",
]

SCHEMA_VERSION = "1.0"

#: Comparison modes.  ``regression`` asks "did the field change?" and needs a
#: tight relative tolerance; ``reference`` asks "is the field right?" and uses
#: the acceptance gate of the case.
REGRESSION = "regression"
REFERENCE = "reference"
_MODES = (REGRESSION, REFERENCE)


# ---------------------------------------------------------------------------
# tolerances
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Tolerance:
    """Per-quantity acceptance band ``|value - reference| <= rtol*|reference| + atol``.

    ``atol`` may be ``None``, in which case it is resolved *per quantity* as
    ``atol_scale * scale``, where ``scale`` is the largest reference magnitude
    of that quantity, or - when the quantity is identically zero in the
    reference data (``sigma_yy`` of a uniaxial patch test) - the physical scale
    the case declares in ``meta["quantity_scales"]``.  The second fallback
    matters: a zero field has no intrinsic magnitude, and deriving the floor
    from it would make the component untestable (``WITHHELD``) precisely where a
    spurious transverse stress would be a real defect.
    """

    rtol: float = 1e-9
    atol: float | None = None
    atol_scale: float = 1e-12

    def __post_init__(self) -> None:
        if not 0.0 <= self.rtol < 1.0:
            raise ValueError("rtol must lie in [0, 1)")
        if self.atol is not None and self.atol < 0.0:
            raise ValueError("atol must be >= 0 when given")
        if self.atol_scale < 0.0:
            raise ValueError("atol_scale must be >= 0")
        if self.atol is None and self.atol_scale == 0.0:
            raise ValueError("atol and atol_scale cannot both be zero")

    def resolve_atol(self, reference_scale: float) -> float:
        if self.atol is not None:
            return float(self.atol)
        return float(self.atol_scale * abs(float(reference_scale)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "rtol": self.rtol,
            "atol": self.atol,
            "atol_scale": self.atol_scale,
        }


def _as_tolerance(value: Tolerance | Mapping[str, Any]) -> Tolerance:
    if isinstance(value, Tolerance):
        return value
    if isinstance(value, Mapping):
        return Tolerance(**value)
    raise TypeError("tolerance must be a Tolerance or a mapping with its fields")


# ---------------------------------------------------------------------------
# snapshots
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProbeSample:
    """One probe: a labelled point in the discretisation and its field values."""

    probe_id: str
    kind: str
    coordinates: tuple[float, ...]
    values: Mapping[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "kind": self.kind,
            "coordinates": list(self.coordinates),
            "values": {k: float(self.values[k]) for k in sorted(self.values)},
        }


@dataclass(frozen=True)
class Snapshot:
    """Reference (or freshly computed) probe set for one case."""

    case_id: str
    quantities: tuple[str, ...]
    probes: tuple[ProbeSample, ...]
    meta: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    @property
    def n_values(self) -> int:
        """Number of comparison rows; probes carry only their local quantities."""
        return sum(len(p.values) for p in self.probes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "quantities": list(self.quantities),
            "meta": dict(self.meta),
            "probes": [p.as_dict() for p in self.probes],
        }

    def values(self, quantity: str) -> list[float]:
        """Values of one quantity, in canonical probe order.

        Only the probes that carry the quantity are returned (nodal probes hold
        displacements, element-centre probes hold stresses); a quantity that no
        probe carries cannot be declared, which :func:`build_snapshot` enforces.
        """
        if quantity not in self.quantities:
            raise KeyError(f"{self.case_id!r} has no quantity {quantity!r}")
        return [float(p.values[quantity]) for p in self.probes if quantity in p.values]


def build_snapshot(
    case_id: str,
    quantities: Sequence[str],
    probes: Iterable[ProbeSample],
    meta: Mapping[str, Any] | None = None,
    *,
    schema_version: str = SCHEMA_VERSION,
) -> Snapshot:
    """Validate and canonicalise a snapshot.

    Probes are sorted by ``probe_id`` so a regenerated store diffs only where a
    value actually changed; duplicated ids, missing quantities and non-finite
    numbers are rejected here rather than producing a silently incomplete
    reference.
    """
    if not case_id:
        raise ValueError("case_id must be a non-empty string")
    names = tuple(str(q) for q in quantities)
    if not names:
        raise ValueError("at least one quantity is required")
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate quantity names in {names}")

    samples = tuple(probes)
    if not samples:
        raise ValueError(f"{case_id!r}: snapshot needs at least one probe")
    ids = [p.probe_id for p in samples]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{case_id!r}: duplicate probe ids")
    canonical = tuple(sorted(samples, key=lambda p: p.probe_id))

    expected = set(names)
    covered: set[str] = set()
    for probe in canonical:
        if not probe.probe_id:
            raise ValueError(f"{case_id!r}: empty probe id")
        if probe.kind not in ("node", "element_centre"):
            raise ValueError(f"{probe.probe_id!r}: unknown probe kind {probe.kind!r}")
        got = set(probe.values)
        # A probe carries the quantities that are *defined* at its location:
        # nodal probes hold displacements, element-centre probes hold stresses.
        # A subset is legitimate; an undeclared quantity is not, and neither is
        # a declared quantity that no probe carries.
        if not got:
            raise ValueError(f"{probe.probe_id!r}: probe carries no quantities")
        extra = got - expected
        if extra:
            raise ValueError(
                f"{probe.probe_id!r}: undeclared quantities {sorted(extra)} "
                f"(declared {list(names)})"
            )
        covered |= got
        coords = tuple(float(c) for c in probe.coordinates)
        if not coords or not all(math.isfinite(c) for c in coords):
            raise ValueError(f"{probe.probe_id!r}: coordinates must be finite")
        for name in sorted(got):
            value = float(probe.values[name])
            if not math.isfinite(value):
                raise ValueError(f"{probe.probe_id!r}: {name} is not finite ({value!r})")

    missing = expected - covered
    if missing:
        raise ValueError(
            f"{case_id!r}: declared quantities {sorted(missing)} are not carried by "
            "any probe, so they would never be checked"
        )

    return Snapshot(case_id=case_id, quantities=names, probes=canonical,
                    meta=dict(meta or {}), schema_version=schema_version)


def mesh_signature(nodes: torch.Tensor, elements: torch.Tensor) -> str:
    """SHA-256 over the discretisation, so a probe set is bound to its mesh.

    The digest covers the float64 (little-endian) node coordinates and the int64
    connectivity.  Two runs that discretise a case differently therefore cannot
    compare pointwise values, even if the probe coordinates coincide.
    """
    n = torch.as_tensor(nodes, dtype=torch.float64).detach().cpu().contiguous()
    e = torch.as_tensor(elements, dtype=torch.int64).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(b"tensorfem.pointwise.nodes.float64")
    digest.update(n.numpy().astype("<f8", copy=False).tobytes())
    digest.update(b"tensorfem.pointwise.elements.int64")
    digest.update(e.numpy().astype("<i8", copy=False).tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# sampling helpers
# ---------------------------------------------------------------------------
def _probe_layer(coordinates: torch.Tensor, fields: Mapping[str, torch.Tensor],
                 indices: Sequence[int], kind: str, prefix: str) -> list[ProbeSample]:
    coords = torch.as_tensor(coordinates, dtype=torch.float64).detach().cpu()
    cols = {}
    for name, values in fields.items():
        array = torch.as_tensor(values, dtype=torch.float64).detach().cpu().reshape(-1)
        if array.numel() != coords.shape[0]:
            raise ValueError(
                f"field {name!r} has {array.numel()} samples but {coords.shape[0]} "
                f"{kind} locations were supplied"
            )
        cols[name] = array
    samples = []
    for index in indices:
        if not 0 <= int(index) < coords.shape[0]:
            raise IndexError(f"probe index {index} out of range for {coords.shape[0]} {kind}s")
        samples.append(
            ProbeSample(
                probe_id=f"{prefix}{int(index):05d}",
                kind=kind,
                coordinates=tuple(float(c) for c in coords[int(index)]),
                values={name: float(col[int(index)]) for name, col in cols.items()},
            )
        )
    return samples


def sample_nodes(nodes: torch.Tensor, fields: Mapping[str, torch.Tensor],
                 indices: Sequence[int], *, prefix: str = "n") -> list[ProbeSample]:
    """Probe per-node fields (displacement components, nodal temperature, ...)."""
    return _probe_layer(nodes, fields, indices, "node", prefix)


def sample_element_centres(centres: torch.Tensor, fields: Mapping[str, torch.Tensor],
                           indices: Sequence[int], *, prefix: str = "e") -> list[ProbeSample]:
    """Probe element-centre fields (recovered stress / flux components, ...)."""
    return _probe_layer(centres, fields, indices, "element_centre", prefix)


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------
def _check_compatible(reference: Snapshot, current: Snapshot) -> None:
    if reference.case_id != current.case_id:
        raise ValueError(
            f"case mismatch: reference {reference.case_id!r} vs current {current.case_id!r}"
        )
    if tuple(reference.quantities) != tuple(current.quantities):
        raise ValueError(
            f"{reference.case_id!r}: quantity set changed "
            f"{list(reference.quantities)} -> {list(current.quantities)}"
        )
    ref_ids = [p.probe_id for p in reference.probes]
    cur_ids = [p.probe_id for p in current.probes]
    if ref_ids != cur_ids:
        missing = sorted(set(ref_ids) - set(cur_ids))
        extra = sorted(set(cur_ids) - set(ref_ids))
        raise ValueError(
            f"{reference.case_id!r}: probe set changed (missing={missing}, extra={extra})"
        )
    for ref, cur in zip(reference.probes, current.probes):
        if ref.kind != cur.kind:
            raise ValueError(
                f"{ref.probe_id!r}: probe kind changed {ref.kind!r} -> {cur.kind!r}"
            )
        scale = max(1.0, max(abs(c) for c in ref.coordinates))
        if any(abs(a - b) > 1e-9 * scale for a, b in zip(ref.coordinates, cur.coordinates)):
            raise ValueError(
                f"{ref.probe_id!r}: probe coordinates moved "
                f"{ref.coordinates} -> {cur.coordinates}"
            )
    ref_sig = reference.meta.get("mesh_signature")
    cur_sig = current.meta.get("mesh_signature")
    if ref_sig is not None and cur_sig is not None and ref_sig != cur_sig:
        raise ValueError(
            f"{reference.case_id!r}: discretisation changed "
            f"(mesh signature {ref_sig[:12]} -> {cur_sig[:12]}); pointwise values on "
            "different meshes are not comparable - re-baseline deliberately"
        )


def compare(reference: Snapshot, current: Snapshot, *, mode: str = REGRESSION,
            tolerance: Tolerance | Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Compare two snapshots point by point.

    ``mode`` is ``"regression"`` (default: tight tolerance, asks whether the
    field changed) or ``"reference"`` (acceptance gate, asks whether the field
    is right).  Both modes produce the same row/verdict structure so a report
    can present them side by side.
    """
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
    tol = tolerance if tolerance is None else _as_tolerance(tolerance)
    if tol is None:
        tol = Tolerance(rtol=1e-9) if mode == REGRESSION else Tolerance(rtol=0.03)
    _check_compatible(reference, current)

    hints = reference.meta.get("quantity_scales") or {}
    scales: dict[str, float] = {}
    scale_source: dict[str, str] = {}
    for quantity in reference.quantities:
        observed = max((abs(v) for v in reference.values(quantity)), default=0.0)
        hint = abs(float(hints.get(quantity, 0.0) or 0.0))
        if observed > 0.0 and observed >= hint:
            scales[quantity], scale_source[quantity] = observed, "reference_data"
        elif hint > 0.0:
            scales[quantity], scale_source[quantity] = hint, "meta.quantity_scales"
        else:
            scales[quantity], scale_source[quantity] = observed, "undefined"
    atols = {q: tol.resolve_atol(scales[q]) for q in reference.quantities}

    rows: list[dict[str, Any]] = []
    for ref_probe, cur_probe in zip(reference.probes, current.probes):
        if set(ref_probe.values) != set(cur_probe.values):
            raise ValueError(
                f"{ref_probe.probe_id!r}: quantity coverage changed "
                f"{sorted(ref_probe.values)} -> {sorted(cur_probe.values)}"
            )
        for quantity in sorted(ref_probe.values):
            ref_value = float(ref_probe.values[quantity])
            cur_value = float(cur_probe.values[quantity])
            label = f"{reference.case_id}:{ref_probe.probe_id}:{quantity}"
            if not math.isfinite(cur_value):
                rows.append({
                    "label": label,
                    "case_id": reference.case_id,
                    "probe_id": ref_probe.probe_id,
                    "kind": ref_probe.kind,
                    "coordinates": list(ref_probe.coordinates),
                    "quantity": quantity,
                    "computed": cur_value,
                    "reference": ref_value,
                    "mode": mode,
                    "abs_error": None,
                    "relative_error": None,
                    "atol": atols[quantity],
                    "band": None,
                    "band_utilisation": None,
                    "well_conditioned": None,
                    "threshold_relative": tol.rtol,
                    "verdict": FAIL,
                    "reason": "computed value is not finite",
                })
                continue
            if ref_value == 0.0 and atols[quantity] == 0.0:
                rows.append({
                    "label": label,
                    "case_id": reference.case_id,
                    "probe_id": ref_probe.probe_id,
                    "kind": ref_probe.kind,
                    "coordinates": list(ref_probe.coordinates),
                    "quantity": quantity,
                    "computed": cur_value,
                    "reference": ref_value,
                    "mode": mode,
                    "abs_error": abs(cur_value),
                    "relative_error": None,
                    "relative_error_raw": None,
                    "atol": atols[quantity],
                    "band": 0.0,
                    "band_utilisation": None,
                    "well_conditioned": False,
                    "threshold_relative": tol.rtol,
                    "verdict": WITHHELD,
                    "reason": (
                        "reference is exactly zero and no absolute floor is available: "
                        "a relative comparison is undefined"
                    ),
                })
                continue
            # Decision rule (documented in the module docstring): a band with
            # both a relative and an absolute term,
            #     |computed - reference| <= rtol*|reference| + atol,
            # which is `certification_evidence.within_tolerance` plus the
            # absolute floor.  The floor is what keeps a component that is zero
            # by symmetry inside the regression surface: with a purely relative
            # criterion such a row would have to be withheld (or would fail on
            # round-off), and a real defect there would escape unnoticed.
            abs_error = abs(cur_value - ref_value)
            band = tol.rtol * abs(ref_value) + atols[quantity]
            well_conditioned = abs(ref_value) >= atols[quantity]
            relative_error = abs_error / abs(ref_value) if ref_value != 0.0 else None
            verdict = PASS if abs_error <= band else FAIL
            rows.append({
                "label": label,
                "case_id": reference.case_id,
                "probe_id": ref_probe.probe_id,
                "kind": ref_probe.kind,
                "coordinates": list(ref_probe.coordinates),
                "quantity": quantity,
                "mode": mode,
                "computed": cur_value,
                "reference": ref_value,
                "abs_error": abs_error,
                "relative_error": relative_error if well_conditioned else None,
                "relative_error_raw": relative_error,
                "atol": atols[quantity],
                "band": band,
                "band_utilisation": (abs_error / band) if band > 0.0 else None,
                "well_conditioned": well_conditioned,
                "threshold_relative": tol.rtol,
                "verdict": verdict,
                "reason": None if verdict == PASS else (
                    f"|computed-reference|={abs_error:.3e} exceeds the band "
                    f"{band:.3e} (rtol={tol.rtol:.1e}, atol={atols[quantity]:.3e})"
                ),
            })

    verdict = aggregate_verdicts(rows)
    measured = [r for r in rows if r.get("relative_error") is not None]
    worst = max(measured, key=lambda r: r["relative_error"]) if measured else None
    by_quantity: dict[str, Any] = {}
    for quantity in reference.quantities:
        sub = [r for r in rows if r["quantity"] == quantity]
        sub_measured = [r for r in sub if r.get("relative_error") is not None]
        sub_worst = max(sub_measured, key=lambda r: r["relative_error"]) if sub_measured else None
        by_quantity[quantity] = {
            "n_rows": len(sub),
            "verdicts": aggregate_verdicts(sub)["state"],
            "max_relative_error": None if sub_worst is None else sub_worst["relative_error"],
            "worst_label": None if sub_worst is None else sub_worst["label"],
            "max_abs_error": max(
                (r["abs_error"] for r in sub if r.get("abs_error") is not None),
                default=None,
            ),
            "reference_scale": scales[quantity],
            "scale_source": scale_source[quantity],
            "atol": atols[quantity],
        }

    return {
        "case_id": reference.case_id,
        "mode": mode,
        "schema_version": reference.schema_version,
        "tolerance": tol.as_dict(),
        "probe_count": len(reference.probes),
        "quantity_count": len(reference.quantities),
        "n_values": len(rows),
        "state": verdict["state"],
        "counts": verdict["counts"],
        "failed_labels": verdict["failed_labels"],
        "withheld_labels": verdict["withheld_labels"],
        "max_relative_error": None if worst is None else worst["relative_error"],
        "worst_label": None if worst is None else worst["label"],
        "by_quantity": by_quantity,
        "reference_meta": dict(reference.meta),
        "current_meta": dict(current.meta),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# serialisation
# ---------------------------------------------------------------------------
def dumps(snapshot: Snapshot, *, indent: int = 2) -> str:
    """Canonical JSON (sorted keys, fixed separators) for a stable diff."""
    return json.dumps(snapshot.as_dict(), indent=indent, sort_keys=True,
                      ensure_ascii=False) + "\n"


def loads(text: str | bytes) -> Snapshot:
    payload = json.loads(text)
    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported snapshot schema_version {version!r} (expected {SCHEMA_VERSION!r})"
        )
    probes = [
        ProbeSample(
            probe_id=str(item["probe_id"]),
            kind=str(item["kind"]),
            coordinates=tuple(float(c) for c in item["coordinates"]),
            values={str(k): float(v) for k, v in item["values"].items()},
        )
        for item in payload["probes"]
    ]
    snapshot = build_snapshot(
        str(payload["case_id"]),
        payload["quantities"],
        probes,
        payload.get("meta") or {},
    )
    # `build_snapshot` re-sorts; a store whose on-disk order differs would
    # defeat the point of a diffable reference, so reject it outright.
    on_disk = [p["probe_id"] for p in payload["probes"]]
    canonical = [p.probe_id for p in snapshot.probes]
    if on_disk != canonical:
        raise ValueError(
            f"{snapshot.case_id!r}: probes are not in canonical order; "
            "regenerate the store with the writer instead of editing it by hand"
        )
    return snapshot


def save(snapshot: Snapshot, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(dumps(snapshot), encoding="utf-8")
    return target


def load(path: str | Path) -> Snapshot:
    return loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# store verification
# ---------------------------------------------------------------------------
Builder = Callable[[], Snapshot]


def _store_files(directory: Path) -> list[Path]:
    """Snapshot files in a store: JSON objects carrying ``case_id`` and ``probes``.

    Ledgers, evidence packs and any other JSON artefact a report writes next to
    the baselines are skipped by content, not by a hard-coded name list.
    """
    files = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "case_id" in payload and "probes" in payload:
            files.append(path)
    return files


def verify_directory(
    directory: str | Path,
    builders: Mapping[str, Builder],
    *,
    mode: str = REGRESSION,
    tolerance: Tolerance | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Rebuild every reference in ``directory`` and compare it.

    ``builders`` maps a ``case_id`` to a zero-argument callable returning the
    freshly computed snapshot.  A case present in the store but missing from the
    registry is an error (the reference would silently stop being checked).
    """
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"pointwise reference store not found: {root}")
    files = _store_files(root)
    if not files:
        raise FileNotFoundError(f"no reference snapshots in {root}")

    tol = tolerance if tolerance is None else _as_tolerance(tolerance)
    cases: list[dict[str, Any]] = []
    for path in files:
        reference = load(path)
        builder = builders.get(reference.case_id)
        if builder is None:
            raise KeyError(
                f"{path.name}: no builder registered for case {reference.case_id!r}"
            )
        current = builder()
        report = compare(reference, current, mode=mode, tolerance=tol)
        report["reference_file"] = path.name
        cases.append(report)

    state = FAIL if any(c["state"] == FAIL for c in cases) else PASS
    if state != FAIL and any(c["state"] == "PASS_WITH_WITHHELD" for c in cases):
        state = "PASS_WITH_WITHHELD"
    return {
        "schema_version": SCHEMA_VERSION,
        "store": str(root),
        "mode": mode,
        "tolerance": (tol.as_dict() if tol is not None else None),
        "n_cases": len(cases),
        "n_values": sum(c["n_values"] for c in cases),
        "state": state,
        "cases": cases,
    }


def write_ledger(ledger: Mapping[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _builders() -> dict[str, Builder]:
    """Adapter: the case module returns ``(computed, analytic)`` pairs."""
    from . import pointwise_cases

    return {
        case_id: (lambda builder=builder: builder()[0])
        for case_id, builder in pointwise_cases.CASE_BUILDERS.items()
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tensorfem.pointwise_regression",
        description="Pointwise reference regression over stored FEM field snapshots.",
    )
    parser.add_argument("action", choices=("check", "update"),
                        help="check = recompute and compare; update = re-baseline")
    parser.add_argument("--store", default="benchmarks/pointwise_reference",
                        help="directory holding the reference snapshots")
    parser.add_argument("--rtol", type=float, default=None,
                        help="relative tolerance (default 1e-9 check, 3e-2 update)")
    parser.add_argument("--ledger", default=None,
                        help="write the JSON ledger here (default <store>/ledger.json)")
    args = parser.parse_args(argv)

    root = Path(args.store)
    rtol = args.rtol
    if rtol is None:
        rtol = 1e-9 if args.action == "check" else 0.03
    tolerance = Tolerance(rtol=rtol)

    if args.action == "update":
        root.mkdir(parents=True, exist_ok=True)
        builders = _builders()
        written = []
        for case_id, builder in sorted(builders.items()):
            snapshot = builder()
            written.append(save(snapshot, root / f"{case_id}.json").name)
        print(json.dumps({"action": "update", "store": str(root),
                          "files": written, "rtol": rtol}, indent=2))
        return 0

    ledger = verify_directory(root, _builders(), tolerance=tolerance)
    target = Path(args.ledger) if args.ledger else root / "ledger.json"
    write_ledger(ledger, target)
    summary = {c["case_id"]: {"state": c["state"], "max_relative_error": c["max_relative_error"]}
               for c in ledger["cases"]}
    print(json.dumps({"action": "check", "store": str(root), "state": ledger["state"],
                      "n_values": ledger["n_values"], "cases": summary}, indent=2))
    return 0 if ledger["state"] == PASS else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())