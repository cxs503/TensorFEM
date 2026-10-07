"""Fail-closed TensorFEM 1.0 readiness contract.

The contract deliberately separates a release that is healthy from one whose
industrial P0 claims are complete.  Missing evidence is blocked, never inferred.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping, Sequence

from .panel_mesh_convergence import evaluate_panel_mesh_convergence


SCHEMA = "tensorfem.v1-readiness/1.0"
REQUIRED_CAPABILITIES = (
    "nonlinear_shell_robustness",
    "general_double_deformable_contact_3d",
    "shell_sparse_scalability",
    "release_quality",
)


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def evaluate_v1_readiness(
    capabilities: Mapping[str, Mapping[str, object]],
    panel_manifests: Sequence[Mapping[str, object]],
    *,
    benchmark_tolerance: float = .03,
    minimum_shell_sparse_dofs: int = 10_000,
) -> dict[str, object]:
    """Evaluate the declared 1.0 gates without promoting partial prototypes."""
    if (not math.isfinite(benchmark_tolerance)
            or not 0 < benchmark_tolerance <= .03):
        raise ValueError("benchmark_tolerance must be finite and at most 3%")
    if minimum_shell_sparse_dofs < 1_000:
        raise ValueError("minimum_shell_sparse_dofs must be at least 1000")

    gates: dict[str, dict[str, object]] = {}
    for name in REQUIRED_CAPABILITIES:
        evidence = capabilities.get(name)
        if evidence is None:
            gates[name] = {"status": "blocked", "reason": "missing_evidence"}
            continue
        declared = bool(evidence.get("passed"))
        error = evidence.get("maximum_relative_error")
        finite_error = (error is not None and math.isfinite(float(error))
                        and float(error) <= benchmark_tolerance)
        checks = [declared, finite_error, bool(evidence.get("restart_or_rollback"))]
        if name == "general_double_deformable_contact_3d":
            checks.append(evidence.get("scope") == "general_surface_to_surface")
        elif name == "shell_sparse_scalability":
            checks.extend((int(evidence.get("qualified_dofs", 0))
                           >= minimum_shell_sparse_dofs,
                           float(evidence.get("storage_reduction", 0.0)) > 0.0))
        elif name == "nonlinear_shell_robustness":
            checks.append(bool(evidence.get("difficult_step_improved")))
        elif name == "release_quality":
            checks.extend((bool(evidence.get("full_regression")),
                           bool(evidence.get("offline_install")),
                           bool(evidence.get("api_audit"))))
        passed = all(checks)
        gates[name] = {
            "status": "qualified" if passed else "blocked",
            "reason": "all_gates_pass" if passed else "incomplete_or_out_of_scope",
        }

    panel = evaluate_panel_mesh_convergence(
        panel_manifests, relative_tolerance=benchmark_tolerance,
    )
    gates["marine_panel_4_8_12_peak_postpeak"] = {
        "status": "qualified" if panel["passed"] else panel["status"],
        "reason": "mesh_gate_passed" if panel["passed"] else "mesh_gate_incomplete",
    }
    blockers = sorted(name for name, gate in gates.items()
                      if gate["status"] != "qualified")
    clean = {
        "schema": SCHEMA,
        "ready_for_1_0": not blockers,
        "benchmark_tolerance": benchmark_tolerance,
        "minimum_shell_sparse_dofs": minimum_shell_sparse_dofs,
        "gates": gates,
        "panel_mesh_convergence": panel,
        "blockers": blockers,
    }
    return {**clean, "report_sha256": _digest(clean)}


def validate_v1_readiness(report: Mapping[str, object]) -> Mapping[str, object]:
    """Validate report integrity; readiness itself may honestly be false."""
    if report.get("schema") != SCHEMA:
        raise ValueError("invalid TensorFEM 1.0 readiness schema")
    clean = {key: value for key, value in report.items() if key != "report_sha256"}
    if _digest(clean) != report.get("report_sha256"):
        raise ValueError("TensorFEM 1.0 readiness hash mismatch")
    blockers = report.get("blockers")
    if bool(report.get("ready_for_1_0")) != (blockers == []):
        raise ValueError("TensorFEM 1.0 readiness status is inconsistent")
    return report
