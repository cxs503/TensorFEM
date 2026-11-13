"""Predeclared, machine-checkable *benchmark method selection*.

Anchor: ``XVAL-LBM-BENCHMARK-METHOD-SELECTION`` -- the practice "a numerical
capability record must state **how** its benchmark reference was selected, and
that selection must be reproducible from raw measurements" applied inside
TensorFEM.

Why a dedicated procedure
-------------------------
A benchmark record normally stores a *number* (``error = 1.03 %``) and a
*verdict* (``PASS``).  Neither says which of the available formulations was
allowed to play the role of reference, nor why the others were rejected, nor
how close the winner sits to the acceptance line.  Three failure modes follow
and are all cheap to achieve accidentally:

* **Under-resolved comparison.**  The rejected candidate is measured with a
  probe that confounds its own discretisation error with the model difference,
  so the recorded "model error" is an over-statement (`P4` in the applied
  study below inflates a 2.16 % model deficit to 7.91 %, i.e. 3.7x).
* **Probe mixing.**  Candidate A is scored on a nodal quantity, candidate B on
  an interior field sample; the comparison then ranks sampling schemes rather
  than methods.
* **Silent no-winner.**  No candidate meets the tolerance, the record keeps the
  least-bad one and still prints ``selected``.

This module makes the selection a *value*: candidates, per-probe relative
errors, a cost proxy, a tolerance -- in; a decision plus signed margins and a
Pareto set -- out.  ``certify_selection`` recomputes the decision from the
stored raw probe errors only, so a hand-edited ``selected`` field cannot
survive (same discipline as ``tools/verified_anchor_audit.py``).

Selection rules (predeclared, applied in this order)
---------------------------------------------------
R1 *Comparability.*  Every candidate must be measured on exactly the same probe
   set; a candidate missing a probe, or carrying an extra one, is rejected with
   ``ValueError`` (fail closed -- an unmeasured probe is not "0 % error").
   Errors must be finite and non-negative.
R2 *Admissibility.*  Candidate ``i`` is admissible iff
   ``max_p error[i][p] <= tolerance`` (inclusive; the line is a bound, not a
   target).
R3 *Winner.*  Among admissible candidates the lowest ``cost`` wins; ties are
   broken by lower worst-probe error, then by name (deterministic, no
   dependence on dict ordering).
R4 *Frontier consistency.*  The winner must be Pareto-non-dominated in
   ``(cost, worst error)``.  A violation means the inputs are inconsistent
   (a cheaper candidate is also at least as accurate, so it should have won),
   and is raised rather than written to the record.
R5 *Fail closed on no-winner.*  If no candidate is admissible the decision is
   ``"no_admissible_method"``, ``selected`` is ``None`` and the record must not
   be read as a pass.  ``certify_selection`` reproduces this too.
R6 *Margins are recorded, not implied.*  ``margin[p] = error[p]/tolerance - 1``
   per candidate (negative = inside the line), plus ``worst_probe`` and
   ``worst_error``.  A 3 %-admissible candidate sitting at 99.9 % of the line is
   reported as such instead of being flattened to "PASS".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

__all__ = [
    "MethodCandidate",
    "SelectionRecord",
    "select_method",
    "certify_selection",
    "pareto_frontier",
]

_RECHECK_ABS = 1e-15


def _check_error(name: str, probe: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"candidate {name!r}: probe {probe!r} error is not finite ({value!r})")
    if value < 0.0:
        raise ValueError(f"candidate {name!r}: probe {probe!r} error is negative ({value!r})")
    return value


@dataclass(frozen=True)
class MethodCandidate:
    """One candidate formulation/element revision, scored on a shared probe set.

    ``error`` values are *relative* errors against the declared reference,
    i.e. ``|candidate - reference| / |reference|``, reported per probe.
    ``cost`` is a deterministic, reproducible proxy (dof count is the natural
    choice); wall-clock time is deliberately not accepted as it is not
    reproducible across hosts.
    """

    name: str
    cost: float
    error: Mapping[str, float] = field(default_factory=dict)
    note: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("candidate name must be non-empty")
        if not math.isfinite(float(self.cost)) or float(self.cost) <= 0.0:
            raise ValueError(f"candidate {self.name!r}: cost must be finite and positive")

    @property
    def worst_probe(self) -> str:
        return max(sorted(self.error), key=lambda k: self.error[k])

    @property
    def worst_error(self) -> float:
        return max(self.error.values()) if self.error else 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "cost": float(self.cost),
            "error": {k: float(v) for k, v in self.error.items()},
            "note": self.note,
        }


@dataclass(frozen=True)
class SelectionRecord:
    tolerance: float
    probes: tuple[str, ...]
    candidates: tuple[MethodCandidate, ...]
    admissible: tuple[str, ...]
    selected: str | None
    decision: str
    margin: Mapping[str, Mapping[str, float]]
    worst: Mapping[str, float]
    worst_probe: Mapping[str, str]
    pareto: tuple[str, ...]
    rule: str

    def to_dict(self) -> dict:
        return {
            "tolerance": float(self.tolerance),
            "probes": list(self.probes),
            "candidates": [c.to_dict() for c in self.candidates],
            "admissible": list(self.admissible),
            "selected": self.selected,
            "decision": self.decision,
            "margin": {k: dict(v) for k, v in self.margin.items()},
            "worst": dict(self.worst),
            "worst_probe": dict(self.worst_probe),
            "pareto": list(self.pareto),
            "rule": self.rule,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "SelectionRecord":
        candidates = tuple(
            MethodCandidate(
                name=str(c["name"]),
                cost=float(c["cost"]),
                error={k: float(v) for k, v in dict(c["error"]).items()},
                note=str(c.get("note", "")),
            )
            for c in payload["candidates"]  # type: ignore[index]
        )
        return cls(
            tolerance=float(payload["tolerance"]),  # type: ignore[index]
            probes=tuple(str(p) for p in payload["probes"]),  # type: ignore[index]
            candidates=candidates,
            admissible=tuple(str(n) for n in payload["admissible"]),  # type: ignore[index]
            selected=None if payload.get("selected") is None else str(payload["selected"]),  # type: ignore[index]
            decision=str(payload["decision"]),  # type: ignore[index]
            margin={k: {p: float(v) for p, v in dict(m).items()} for k, m in dict(payload["margin"]).items()},  # type: ignore[index]
            worst={k: float(v) for k, v in dict(payload["worst"]).items()},  # type: ignore[index]
            worst_probe={k: str(v) for k, v in dict(payload["worst_probe"]).items()},  # type: ignore[index]
            pareto=tuple(str(n) for n in payload["pareto"]),  # type: ignore[index]
            rule=str(payload["rule"]),  # type: ignore[index]
        )


def pareto_frontier(candidates: Sequence[MethodCandidate]) -> tuple[str, ...]:
    """Names of the (cost, worst-error) Pareto-non-dominated candidates."""
    keep: list[str] = []
    for cand in candidates:
        dominated = False
        for other in candidates:
            if other.name == cand.name:
                continue
            cheaper_or_equal_cost = other.cost <= cand.cost
            at_least_as_accurate = other.worst_error <= cand.worst_error
            strictly_better = other.cost < cand.cost or other.worst_error < cand.worst_error
            if cheaper_or_equal_cost and at_least_as_accurate and strictly_better:
                dominated = True
                break
        if not dominated:
            keep.append(cand.name)
    return tuple(keep)


def _shared_probe_set(candidates: Sequence[MethodCandidate]) -> tuple[str, ...]:
    if not candidates:
        raise ValueError("at least one candidate is required")
    names = [c.name for c in candidates]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate candidate names: {sorted(names)}")
    reference = set(candidates[0].error)
    if not reference:
        raise ValueError("candidates carry no probes; nothing to select on")
    for cand in candidates[1:]:
        if set(cand.error) != reference:
            missing = sorted(reference - set(cand.error))
            extra = sorted(set(cand.error) - reference)
            raise ValueError(
                f"candidate {cand.name!r} is not comparable with {candidates[0].name!r}: "
                f"missing probes {missing}, extra probes {extra}"
            )
    return tuple(sorted(reference))


def select_method(
    candidates: Iterable[MethodCandidate],
    tolerance: float,
    *,
    rule: str = "cheapest admissible; ties by accuracy then name",
) -> SelectionRecord:
    """Apply rules R1-R6 and return a reproducible selection record."""
    cands = tuple(candidates)
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    probes = _shared_probe_set(cands)

    checked: list[MethodCandidate] = []
    for cand in cands:
        errors = {p: _check_error(cand.name, p, cand.error[p]) for p in probes}
        checked.append(
            MethodCandidate(name=cand.name, cost=cand.cost, error=errors, note=cand.note)
        )

    worst = {c.name: c.worst_error for c in checked}
    worst_probe = {c.name: c.worst_probe for c in checked}
    margin = {
        c.name: {p: c.error[p] / tolerance - 1.0 for p in probes} for c in checked
    }
    admissible = tuple(c.name for c in checked if worst[c.name] <= tolerance)

    if admissible:
        ranked = sorted(
            (c for c in checked if c.name in admissible),
            key=lambda c: (c.cost, worst[c.name], c.name),
        )
        selected: str | None = ranked[0].name
        decision = "selected"
    else:
        selected = None
        decision = "no_admissible_method"

    frontier = pareto_frontier(checked)
    if selected is not None and selected not in frontier:
        raise ValueError(
            f"rule R4 violated: winner {selected!r} is Pareto-dominated; "
            f"frontier is {list(frontier)} (inconsistent cost/accuracy inputs)"
        )

    return SelectionRecord(
        tolerance=tolerance,
        probes=probes,
        candidates=tuple(checked),
        admissible=admissible,
        selected=selected,
        decision=decision,
        margin=margin,
        worst=worst,
        worst_probe=worst_probe,
        pareto=frontier,
        rule=rule,
    )


def certify_selection(record: SelectionRecord, *, rel_tol: float = 1e-12) -> dict:
    """Recompute the decision from stored raw probe errors.

    Returns ``{"certified": bool, "mismatches": [...], "recomputed": {...}}``.
    Nothing is read from the record's claimed ``selected``/``admissible``/
    ``margin`` fields except to compare them against the recomputation.
    """
    try:
        recomputed = select_method(
            record.candidates, record.tolerance, rule=record.rule
        )
    except ValueError as exc:  # R1/R4 refusal is itself a non-certification
        return {"certified": False, "mismatches": [f"recomputation refused: {exc}"],
                "recomputed": None}

    mismatches: list[str] = []
    if tuple(sorted(record.probes)) != tuple(sorted(recomputed.probes)):
        mismatches.append(
            f"probe set {list(record.probes)} != recomputed {list(recomputed.probes)}"
        )
    if record.admissible != recomputed.admissible:
        mismatches.append(
            f"admissible {list(record.admissible)} != recomputed {list(recomputed.admissible)}"
        )
    if record.selected != recomputed.selected:
        mismatches.append(f"selected {record.selected!r} != recomputed {recomputed.selected!r}")
    if record.decision != recomputed.decision:
        mismatches.append(f"decision {record.decision!r} != recomputed {recomputed.decision!r}")
    if tuple(sorted(record.pareto)) != tuple(sorted(recomputed.pareto)):
        mismatches.append(f"pareto {list(record.pareto)} != recomputed {list(recomputed.pareto)}")
    if (record.decision == "selected") != (record.selected is not None):
        mismatches.append("decision/selected inconsistency (R5)")

    for name, per_probe in recomputed.margin.items():
        claimed = record.margin.get(name, {})
        for probe, value in per_probe.items():
            if probe not in claimed:
                mismatches.append(f"margin[{name}][{probe}] missing")
                continue
            if abs(float(claimed[probe]) - value) > rel_tol * max(1.0, abs(value)):
                mismatches.append(
                    f"margin[{name}][{probe}] {float(claimed[probe])!r} != {value!r}"
                )
    for name, value in recomputed.worst.items():
        if name not in record.worst:
            mismatches.append(f"worst[{name}] missing")
        elif abs(float(record.worst[name]) - value) > rel_tol * max(_RECHECK_ABS, abs(value)):
            mismatches.append(f"worst[{name}] {float(record.worst[name])!r} != {value!r}")

    return {
        "certified": not mismatches,
        "mismatches": mismatches,
        "recomputed": {
            "selected": recomputed.selected,
            "decision": recomputed.decision,
            "admissible": list(recomputed.admissible),
        },
    }