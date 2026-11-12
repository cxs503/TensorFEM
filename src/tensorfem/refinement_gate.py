"""Reference-benchmark gates for a mesh-refinement ladder (solver-agnostic).

Why this module exists
----------------------
Two verification anchors (``fem_pinched_cylinder``, ``fem_shell_scordelis_lo``)
graded themselves *only* on the finest-mesh error against a published value.
That criterion is unsound whenever the ladder is pre-asymptotic or
non-monotone: a coarse mesh can overshoot the reference and a finer mesh
undershoot it, so ``|w_finest - ref|`` may be small purely by cancellation
while the discretisation still converges to a value outside the tolerance.

Measured example (kept as a regression test, see
``tests/test_refinement_gate.py``): MacNeal-Harder pinched cylinder gives
errors 2.86 % -> 3.35 % -> 0.17 % -> 1.82 % on 4x -> 32x.  The finest-mesh
reading (1.82 %) "passes" a 3 % gate, yet the Richardson limit of the ladder is
-1.88451e-05, i.e. **3.27 % from the reference** -- the accuracy claim is an
artefact of the mesh ladder, not a property of the element.

Design contract
---------------
* All arithmetic is delegated to :mod:`tensorfem.convergence_verification`
  (observed order, Richardson extrapolation, Roache GCI).  Nothing is
  re-derived here; this module only turns that report into graded gates.
* A gate is ``PASS``, ``FAIL`` or ``WITHHELD``.  ``WITHHELD`` means the ladder
  cannot answer the question (no measurable order / non-monotone) -- it is
  *never* silently promoted to a pass, and an overall verdict is ``PASS`` only
  when every gate passes.
* The *certified accuracy* of an anchor is the asymptotic (extrapolated) error,
  not the finest-mesh error.  The latter is still reported, as
  ``reported_accuracy_finest``, so an over-stated claim stays visible.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .convergence_verification import (
    ConvergenceLevel,
    evaluate_convergence_study,
    refinement_ratio,
    richardson_observed_order,
)

SCHEMA = "tensorfem.refinement-gate/1.0"

PASS = "PASS"
FAIL = "FAIL"
WITHHELD = "WITHHELD"

__all__ = [
    "SCHEMA",
    "PASS",
    "FAIL",
    "WITHHELD",
    "GateVerdict",
    "observed_orders",
    "evaluate_reference_ladder",
    "summarise",
]

#: Gates that carry the accuracy claim, and gates that qualify it.
ACCURACY_GATE_IDS = ("accuracy_finest", "accuracy_asymptotic")
STUDY_GATE_IDS = ("monotone_qoi", "observed_order", "asymptotic_range_consistency",
                  "coarsest_sanity")


def _rollup(gates: list[dict[str, object]]) -> str:
    if any(g["status"] == FAIL for g in gates):
        return FAIL
    if any(g["status"] == WITHHELD for g in gates):
        return WITHHELD
    return PASS


def summarise(report: Mapping[str, object]) -> dict[str, object]:
    """Split a ladder report into an accuracy verdict and a study verdict.

    Keeping the two apart stops an unclosed convergence *study* from being
    read as a failed accuracy *claim* (and vice versa): an element can meet the
    tolerance while the ladder is still pre-asymptotic, and the machine
    contract must say exactly which of the two is unproven.
    """
    gates = list(report["gates"])  # type: ignore[index]
    accuracy = [g for g in gates if g["id"] in ACCURACY_GATE_IDS]
    study = [g for g in gates if g["id"] in STUDY_GATE_IDS]
    accuracy_verdict = _rollup(accuracy)
    study_verdict = _rollup(study)
    if FAIL in (accuracy_verdict, study_verdict):
        overall = FAIL
    elif WITHHELD in (accuracy_verdict, study_verdict):
        overall = WITHHELD
    else:
        overall = PASS
    return {
        "accuracy_verdict": accuracy_verdict,
        "study_verdict": study_verdict,
        "overall_verdict": overall,
        "passed": overall == PASS,
        "unproven": [g["id"] for g in gates if g["status"] != PASS],
    }


@dataclass(frozen=True)
class GateVerdict:
    """One graded criterion of a reference ladder."""

    id: str
    description: str
    status: str
    actual: float | None
    threshold: float | None
    detail: str
    #: Comparison applied to ``(actual, threshold)``; emitted so a certificate
    #: can be recomputed by a reader that never runs the solver.
    operator: str | None = None

    def __post_init__(self) -> None:
        if self.status not in (PASS, FAIL, WITHHELD):
            raise ValueError(f"unsupported status {self.status!r}")

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status,
            "passed": self.status == PASS,
            "actual": self.actual,
            "threshold": self.threshold,
            "operator": self.operator,
            "detail": self.detail,
        }


def _coerce_levels(
    levels: Iterable[ConvergenceLevel | Mapping[str, object] | Sequence[object]],
) -> list[ConvergenceLevel]:
    out: list[ConvergenceLevel] = []
    for level in levels:
        if isinstance(level, ConvergenceLevel):
            out.append(level)
        elif isinstance(level, Mapping):
            out.append(ConvergenceLevel(float(level["mesh_size"]), float(level["value"]),
                                        str(level.get("label", ""))))
        else:
            mesh_size, value, *rest = level  # type: ignore[misc]
            out.append(ConvergenceLevel(float(mesh_size), float(value),
                                        str(rest[0]) if rest else ""))
    if len(out) < 3:
        raise ValueError("at least three refinement levels are required to grade an anchor")
    return out


def observed_orders(levels: Iterable[ConvergenceLevel | Mapping[str, object] | Sequence[object]]) -> list[dict[str, object]]:
    """Observed order of every consecutive *triple*, honestly labelled.

    ``p = ln(|w2-w1| / |w3-w2|) / ln(r)`` on the quantity of interest itself
    (valid because the QoI converges to a finite limit, unlike the error, whose
    limit is zero).  A triple whose differences change sign -- or vanish to
    machine precision -- is reported with ``order = None`` and a reason instead
    of a spurious number.
    """
    study = _coerce_levels(levels)
    out: list[dict[str, object]] = []
    for a, b, c in zip(study, study[1:], study[2:]):
        r = refinement_ratio(a, c) ** 0.5
        entry: dict[str, object] = {"from": a.label, "to": c.label, "ratio": r}
        try:
            entry["order"] = richardson_observed_order(a, b, c)
        except ValueError as exc:
            entry["order"] = None
            entry["reason"] = str(exc)
        out.append(entry)
    return out


def evaluate_reference_ladder(
    levels: Iterable[ConvergenceLevel | Mapping[str, object] | Sequence[object]],
    *,
    reference_value: float,
    tolerance: float = 0.03,
    min_order: float | None = None,
    coarsest_sanity: float | None = None,
    qoi: str = "quantity",
    reference_label: str = "reference",
) -> dict[str, object]:
    """Grade a refinement ladder against an independent reference value.

    ``tolerance`` is the relative accuracy the anchor claims (e.g. 0.03 = 3 %).
    ``min_order``, when given, is the lowest observed order the discretisation
    is expected to reach on the finest triple.  ``coarsest_sanity``, when given,
    caps the relative error the coarsest level is allowed to show (a guard
    against a ladder that starts from a meaningless model).
    """
    study = _coerce_levels(levels)
    if not math.isfinite(reference_value):
        raise ValueError("reference_value must be finite")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")

    study_report = evaluate_convergence_study(
        study,
        qoi=qoi,
        reference_value=reference_value,
        reference_tolerance=tolerance,
        reference_label=reference_label,
    )
    ref = study_report.get("reference") or {}
    status = str(study_report.get("status", "blocked"))
    resolvable = status in ("passed", "failed") and study_report.get("gci") is not None

    finest = study[-1]
    finest_error = float(ref.get("finest_relative_error",
                                 abs(finest.value - reference_value) / abs(reference_value)))
    asymptotic_error = ref.get("extrapolated_relative_error")
    gci = study_report.get("gci")
    order = study_report.get("finest_order")
    orders = observed_orders(study)

    # --- monotonicity of the quantity of interest itself -------------------
    values = [lv.value for lv in study]
    steps = [b - a for a, b in zip(values, values[1:])]
    monotone = all(s > 0.0 for s in steps) or all(s < 0.0 for s in steps)
    # The gate used to be marked ``actual=None`` (pass) / ``1.0`` (fail) with no
    # threshold, i.e. a verdict no reader can recompute.  It now carries the
    # count of ladder steps that break monotonicity (sign reversal or exact
    # stagnation) against a threshold of 0, so PASS/FAIL is machine-checkable.
    reversals = sum(1 for s1, s2 in zip(steps, steps[1:]) if s1 * s2 < 0.0)
    stagnant = sum(1 for s in steps if s == 0.0)
    violations = reversals + stagnant
    worst_flip = ""
    for i, (s1, s2) in enumerate(zip(steps, steps[1:])):
        if s1 * s2 < 0.0:
            worst_flip = (f"sign flips between {study[i].label}->{study[i + 1].label} "
                          f"({s1:.6g}) and {study[i + 1].label}->{study[i + 2].label} ({s2:.6g})")
            break

    gates: list[GateVerdict] = []

    gates.append(GateVerdict(
        id="accuracy_finest",
        description="finest-mesh value within tolerance of the reference",
        status=PASS if abs(finest_error) <= tolerance else FAIL,
        actual=finest_error,
        threshold=tolerance,
        operator="<=",
        detail=f"finest level {finest.label or finest.mesh_size} rel. error {finest_error:.4e}",
    ))

    if resolvable and asymptotic_error is not None:
        gates.append(GateVerdict(
            id="accuracy_asymptotic",
            description="Richardson limit of the ladder within tolerance (certified accuracy)",
            status=PASS if abs(float(asymptotic_error)) <= tolerance else FAIL,
            actual=float(asymptotic_error),
            threshold=tolerance,
            operator="<=",
            detail=(f"extrapolated {float(ref['extrapolated_value']):.12g} vs "
                    f"{reference_label} {reference_value:.12g}: rel. error "
                    f"{float(asymptotic_error):.4e} (order p={float(order):.3f})"),
        ))
    else:
        reason = study_report.get("blocked_reason") or "no measurable order on this ladder"
        gates.append(GateVerdict(
            id="accuracy_asymptotic",
            description="Richardson limit of the ladder within tolerance (certified accuracy)",
            status=WITHHELD,
            actual=None,
            threshold=tolerance,
            operator="<=",
            detail=f"cannot certify: {reason}",
        ))

    gates.append(GateVerdict(
        id="monotone_qoi",
        description="quantity of interest moves monotonically along the ladder "
                    "(count of sign reversals + exact stagnation, must be 0)",
        status=PASS if monotone else FAIL,
        actual=float(violations),
        threshold=0.0,
        operator="<=",
        detail=("monotone" if monotone else f"non-monotone: {worst_flip}")
               + f" [{reversals} sign reversal(s), {stagnant} stagnation step(s)]",
    ))

    if min_order is not None:
        if order is None:
            gates.append(GateVerdict(
                id="observed_order",
                description=f"finest observed order >= {min_order:g}",
                status=WITHHELD,
                actual=None,
                threshold=float(min_order),
                operator=">=",
                detail=f"cannot certify: {study_report.get('blocked_reason', 'order not measurable')}",
            ))
        else:
            gates.append(GateVerdict(
                id="observed_order",
                description=f"finest observed order >= {min_order:g}",
                status=PASS if float(order) >= min_order else FAIL,
                actual=float(order),
                threshold=float(min_order),
                operator=">=",
                detail=", ".join(
                    f"{e['from']}->{e['to']}: "
                    + (f"p={e['order']:.3f}" if e["order"] is not None else f"n/a ({e.get('reason', '')})")
                    for e in orders
                ),
            ))

    if resolvable and gci is not None:
        inside = bool(ref.get("finest_inside_gci", False))
        # A finest-mesh error larger than the extrapolation band means the ladder
        # is still pre-asymptotic: the extrapolation is not conservative, so the
        # study cannot *certify* the value.  That is missing evidence, not a
        # demonstrated violation of the tolerance -- hence WITHHELD, never FAIL.
        band = float(ref.get("relative_gci", float("nan")))
        # ``actual`` is the measured finest-mesh error; ``threshold`` is the GCI
        # band.  Writing the band into both fields made the gate reproduce its
        # own threshold and left the verdict unsupported by its own numbers.
        # The comparison is repeated here without the 1e-12 estimator slack, so
        # a band-edge ladder is reported WITHHELD rather than promoted.
        gates.append(GateVerdict(
            id="asymptotic_range_consistency",
            description="finest-mesh error inside the GCI band (ladder in the asymptotic range)",
            status=PASS if (inside and finest_error <= band) else WITHHELD,
            actual=finest_error,
            threshold=band,
            operator="<=",
            detail=(str(ref.get("gci_detail", "")) if inside else
                    str(ref.get("gci_detail", ""))
                    + "; ladder pre-asymptotic -> accuracy not certifiable from these levels"),
        ))
    else:
        gates.append(GateVerdict(
            id="asymptotic_range_consistency",
            description="finest-mesh error inside the GCI band (ladder in the asymptotic range)",
            status=WITHHELD,
            actual=None,
            threshold=None,
            detail="no GCI band available: " + str(study_report.get("blocked_reason", "unresolvable ladder")),
        ))

    if coarsest_sanity is not None:
        coarsest = study[0]
        coarsest_error = abs(coarsest.value - reference_value) / abs(reference_value)
        gates.append(GateVerdict(
            id="coarsest_sanity",
            description=f"coarsest level within {coarsest_sanity:g} of the reference",
            status=PASS if coarsest_error <= coarsest_sanity else FAIL,
            actual=coarsest_error,
            threshold=float(coarsest_sanity),
            operator="<=",
            detail=f"coarsest level {coarsest.label or coarsest.mesh_size} rel. error {coarsest_error:.4e}",
        ))

    if any(g.status == FAIL for g in gates):
        overall = FAIL
    elif any(g.status == WITHHELD for g in gates):
        overall = WITHHELD
    else:
        overall = PASS

    return {
        "schema": SCHEMA,
        "qoi": qoi,
        "levels": [lv.as_dict() for lv in study],
        "reference": {"label": reference_label, "value": reference_value,
                      "tolerance": tolerance},
        "observed_orders": orders,
        "reported_accuracy_finest": abs(finest_error),
        "certified_accuracy": None if (not resolvable or asymptotic_error is None)
        else abs(float(asymptotic_error)),
        "gates": [g.to_dict() for g in gates],
        "status": overall,
        "passed": overall == PASS,
        "study_status": status,
        "limitations": [
            "Certified accuracy is the Richardson limit of the ladder; reported "
            "finest-mesh accuracy is not evidence of asymptotic accuracy.",
            "A non-monotone ladder cannot certify any accuracy: the gate is WITHHELD, not passed.",
        ],
    }