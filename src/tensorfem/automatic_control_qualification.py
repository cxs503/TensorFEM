"""Fail-closed comparison of automatic and fixed panel continuation paths."""
from __future__ import annotations

import hashlib
import json
from typing import Mapping, Sequence


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def _interpolate(points: Sequence[Mapping[str, object]], x: float,
                 name: str) -> float:
    for left, right in zip(points, points[1:]):
        x0, x1 = float(left["edge_shortening_m"]), float(right["edge_shortening_m"])
        if min(x0, x1) <= x <= max(x0, x1) and x1 != x0:
            weight = (x-x0)/(x1-x0)
            return float(left[name])+weight*(float(right[name])-float(left[name]))
    raise ValueError("automatic terminal shortening is outside fixed-path coverage")


def qualify_automatic_step_path(
    automatic: Mapping[str, object], fixed: Mapping[str, object],
    restarted: Mapping[str, object], *, relative_tolerance: float = .01,
) -> dict[str, object]:
    """Compare a real automatic path with fixed and restarted evidence."""
    if relative_tolerance <= 0:
        raise ValueError("relative_tolerance must be positive")
    auto_points = automatic.get("point_history", [])
    fixed_points = fixed.get("point_history", [])
    if len(auto_points) < 2 or len(fixed_points) < 2:
        raise ValueError("qualification requires nontrivial real paths")
    terminal = auto_points[-1]
    shortening = float(terminal["edge_shortening_m"])
    names = ("force_n", "centre_deflection_m", "external_work_j",
             "internal_energy_j")
    errors = {}
    references = {}
    for name in names:
        reference = _interpolate(fixed_points, shortening, name)
        references[name] = reference
        errors[name] = abs(float(terminal[name])-reference)/max(abs(reference), 1e-30)
    decisions = automatic.get("nonlinear_controller_decisions", [])
    scaled = [decision for decision in decisions
              if decision.get("step_application") is not None
              and decision["step_application"]["applied_step_size"]
              != decision["step_application"]["current_step_size"]]
    algorithms_retained = all(
        not decision.get("automatic_algorithm_switching_enabled", True)
        for decision in decisions
    )
    restart_exact = (
        automatic.get("point_history") == restarted.get("point_history")
        and automatic.get("nonlinear_controller_decisions")
        == restarted.get("nonlinear_controller_decisions")
    )
    def totals(manifest):
        chunks = manifest.get("chunks", [])
        return {
            "attempted": sum(int(chunk["attempted"]) for chunk in chunks),
            "rejected": sum(int(chunk["rejected"]) for chunk in chunks),
            "newton_iterations": sum(int(point["newton_iterations"])
                                     for point in manifest["point_history"]),
        }
    auto_cost, fixed_cost = totals(automatic), totals(fixed)
    payload = {
        "schema": "tensorfem.automatic-step-qualification/1.0",
        "relative_tolerance": relative_tolerance,
        "automatic_terminal_shortening_m": shortening,
        "fixed_interpolated_reference": references,
        "relative_errors": errors,
        "step_scaling_events": len(scaled),
        "restart_exact": restart_exact,
        "algorithms_retained": algorithms_retained,
        "automatic_cost": auto_cost,
        "fixed_cost": fixed_cost,
        "rejected_step_reduction": fixed_cost["rejected"]-auto_cost["rejected"],
        "newton_iteration_reduction": (fixed_cost["newton_iterations"]
                                        - auto_cost["newton_iterations"]),
    }
    payload["passed"] = bool(
        scaled and restart_exact and algorithms_retained
        and all(error < relative_tolerance for error in errors.values())
    )
    payload["evidence_sha256"] = hashlib.sha256(
        _canonical(payload).encode()).hexdigest()
    return payload

