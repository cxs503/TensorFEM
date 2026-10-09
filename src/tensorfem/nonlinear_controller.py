"""Deterministic nonlinear continuation control policy.

The policy converts accepted-point and Newton diagnostics into an auditable
recommendation.  A caller may explicitly opt into bounded step-size changes;
algorithm switching remains disabled until separately qualified.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Iterable, Mapping, Sequence


SCHEMA = "tensorfem.nonlinear-controller/1.0"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _path_curvature(history: Sequence[Mapping[str, object]]) -> float | None:
    """Return the sine of the angle between the last two response segments."""
    if len(history) < 3:
        return None
    points = history[-3:]
    force_scale = max(abs(float(p["force_n"])) for p in points)
    displacement_scale = max(abs(float(p["edge_shortening_m"])) for p in points)
    force_scale = max(force_scale, 1.0)
    displacement_scale = max(displacement_scale, 1e-15)
    vectors = []
    for left, right in zip(points, points[1:]):
        vectors.append((
            (float(right["edge_shortening_m"])-float(left["edge_shortening_m"]))
            / displacement_scale,
            (float(right["force_n"])-float(left["force_n"])) / force_scale,
        ))
    a, b = vectors
    na, nb = math.hypot(*a), math.hypot(*b)
    return None if na == 0.0 or nb == 0.0 else abs(a[0]*b[1]-a[1]*b[0])/(na*nb)


def _newton_metrics(diagnostics: Iterable[Mapping[str, object]]) -> tuple[float | None, int | None]:
    accepted = [item for item in diagnostics if item.get("reason") == "accepted"
                and item.get("attempt") is not None and item.get("iterations")]
    if not accepted:
        return None, None
    iterations = accepted[-1]["iterations"]
    first = float(iterations[0]["residual_relative"])
    last = float(iterations[-1]["residual_relative"])
    rate = None if first <= 0.0 else (last/first)**(1.0/max(len(iterations)-1, 1))
    return rate, len(iterations)


def recommend_nonlinear_controls(
    point_history: Sequence[Mapping[str, object]],
    diagnostics: Sequence[Mapping[str, object]],
    *,
    prior_decision_sha256: str | None = None,
    automatic_step_control: bool = False,
    current_step_size: float | None = None,
    minimum_step_size: float | None = None,
    maximum_step_size: float | None = None,
) -> dict[str, object]:
    """Produce a fail-closed decision, optionally applying *only* step size.

    Automatic control is deliberately explicit opt-in.  Newton method, line
    search and arc metric remain unchanged even when step control is enabled.
    The applied size and its bounds are part of the hashed decision, making a
    restarted run deterministic and tamper evident.
    """
    if automatic_step_control:
        values = (current_step_size, minimum_step_size, maximum_step_size)
        if any(value is None or not _finite(value) or float(value) <= 0
               for value in values):
            raise ValueError("automatic step control requires finite positive bounds")
        if not float(minimum_step_size) <= float(current_step_size) <= float(maximum_step_size):
            raise ValueError("current step size must lie within automatic-control bounds")
    attempted = [item for item in diagnostics if item.get("attempt") is not None]
    rejected = sum(item.get("reason") != "accepted" for item in attempted)
    rate, iterations = _newton_metrics(diagnostics)
    curvature = _path_curvature(point_history)
    terminal = point_history[-1] if point_history else None
    balance = terminal.get("equilibrium_relative_norm") if terminal else None
    # Whole-path energy remains a qualification result, but cannot guide a
    # controller after one coarse increment has left an irreversible
    # quadrature residual.  Prefer the explicitly reported latest-increment
    # gate when the runner provides it.
    gate = (terminal.get("controller_incremental_energy_balance_gate",
                         terminal.get("energy_balance_gate"))
            if terminal else None)
    energy_passed = gate.get("passed") if isinstance(gate, Mapping) else None

    invalid = []
    for name, value in (("newton_rate", rate), ("path_curvature", curvature),
                        ("equilibrium_relative_norm", balance)):
        if value is not None and not _finite(value):
            invalid.append(name)
    if terminal is None:
        invalid.append("accepted_point")
    if balance is None:
        invalid.append("equilibrium_relative_norm")
    if energy_passed is None:
        invalid.append("energy_balance_gate")

    reasons: list[str] = []
    if invalid:
        classification, factor = "blocked", 1.0
        reasons.append("missing_or_nonfinite_evidence:" + ",".join(sorted(set(invalid))))
    elif not bool(energy_passed) or float(balance) > 1e-6:
        classification, factor = "unsafe", 0.5
        reasons.append("accepted_point_failed_energy_or_equilibrium_gate")
    elif rejected >= 2:
        classification, factor = "difficult", 0.5
        reasons.append("multiple_rejected_attempts")
    elif ((iterations is not None and iterations > 8)
          or (rate is not None and rate > 0.7)
          or (curvature is not None and curvature > 0.35)):
        classification, factor = "difficult", 0.75
        reasons.append("slow_newton_or_high_path_curvature")
    elif (rejected == 0 and iterations is not None and iterations <= 4
          and rate is not None and rate < 0.25
          and (curvature is None or curvature < 0.1)):
        classification, factor = "stable", 1.25
        reasons.append("fast_newton_low_curvature")
    else:
        classification, factor = "nominal", 1.0
        reasons.append("no_policy_threshold_crossed")

    difficult = classification in {"blocked", "unsafe", "difficult"}
    applied_step = None
    if automatic_step_control:
        applied_step = min(float(maximum_step_size), max(float(minimum_step_size),
                           float(current_step_size) * factor))
    evidence = {
        "schema": SCHEMA,
        "mode": "automatic_step_only" if automatic_step_control else "advisory_only",
        "automatic_switching_enabled": automatic_step_control,
        "automatic_algorithm_switching_enabled": False,
        "classification": classification,
        "observations": {
            "accepted_points": len(point_history),
            "attempted_steps": len(attempted),
            "rejected_steps": rejected,
            "terminal_newton_iterations": iterations,
            "newton_residual_contraction_rate": rate,
            "path_curvature": curvature,
            "terminal_equilibrium_relative_norm": balance,
            "terminal_energy_gate_passed": energy_passed,
        },
        "recommendation": {
            "step_size_factor": factor,
            "newton_method": "full_newton" if difficult else "modified_newton_candidate",
            "line_search": "enable" if difficult else "retain",
            "arc_metric": "dimensionally_scaled",
        },
        "step_application": ({
            "current_step_size": float(current_step_size),
            "minimum_step_size": float(minimum_step_size),
            "maximum_step_size": float(maximum_step_size),
            "applied_step_size": applied_step,
        } if automatic_step_control else None),
        "reasons": reasons,
        "prior_decision_sha256": prior_decision_sha256,
        "boundary": (
            ("Only the hashed step-size recommendation is applied; full-Newton, "
             "line-search and dimensionally-scaled arc controls are retained."
             if automatic_step_control else
             "Recommendations are evidence only. The panel runner retains its "
             "qualified full-Newton and dimensionally-scaled arc controls; a "
             "line-search or modified-Newton recommendation is not activated.")
        ),
    }
    evidence["decision_sha256"] = hashlib.sha256(
        _canonical(evidence).encode()).hexdigest()
    return evidence


def verify_decision_chain(decisions: Sequence[Mapping[str, object]]) -> bool:
    """Verify hashes and predecessor links after persistence or restart."""
    previous = None
    for decision in decisions:
        value = dict(decision)
        recorded = value.pop("decision_sha256", None)
        if value.get("prior_decision_sha256") != previous:
            return False
        expected = hashlib.sha256(_canonical(value).encode()).hexdigest()
        if recorded != expected:
            return False
        previous = recorded
    return True
