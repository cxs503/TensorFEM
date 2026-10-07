"""Fail-closed qualification view for the industrial P0 phase-3 work.

This report intentionally distinguishes a qualified *axisymmetric* Hertz
verification from the still-blocked general 3-D contact and panel ultimate
strength claims.  It executes production solvers and hashes the canonical
evidence, so a label cannot be promoted without changing auditable data.
"""
from __future__ import annotations

import hashlib
import json

import torch

from .finite_rotation_layered_shell4 import (
    assemble_finite_rotation_layered_shell4, solve_finite_rotation_arc_path,
)
from .hertz_axisymmetric import axisymmetric_hertz_errors, solve_axisymmetric_hertz
from .layered_shell4_plasticity import LayeredShell4Model, LayeredShell4State
from .marine_panel_ultimate_fe import run_panel_ultimate_qualification


SCHEMA = "tensorfem.industrial-p0-phase3/1.0"
_HERTZ_MESHES = (16, 24, 48)
_HERTZ_LIMIT = 0.03
_EQUILIBRIUM_LIMIT = 1.0e-10


def _digest(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _hertz_evidence() -> dict[str, object]:
    rows = []
    for n in _HERTZ_MESHES:
        result = solve_axisymmetric_hertz(nr=n, nz=n)
        errors = axisymmetric_hertz_errors(result)
        rows.append({
            "radial_elements": n,
            "depth_elements": n,
            "converged": result.converged,
            "iterations": result.iterations,
            "relative_errors": errors,
        })

    checked = (
        "load", "contact_radius", "peak_pressure", "pressure_l2",
        "normalized_overlap",
    )
    fine = rows[-1]["relative_errors"]
    monotone = {
        name: all(
            rows[i + 1]["relative_errors"][name]
            < rows[i]["relative_errors"][name]
            for i in range(len(rows) - 1)
        )
        for name in checked
    }
    gates = {
        "all_solves_converged": all(row["converged"] for row in rows),
        "fine_mesh_errors_below_3_percent": all(fine[name] < _HERTZ_LIMIT for name in checked),
        "errors_decrease_on_stated_mesh_sequence": all(monotone.values()),
        "fine_mesh_force_balance_below_1e-10": fine["force_balance"] < _EQUILIBRIUM_LIMIT,
    }
    clean = {
        "status": "qualified_axisymmetric_rigid_indenter_deformable_halfspace",
        "scope": (
            "small-strain axisymmetric Q4 elastic half-space with a rigid "
            "parabolic indenter and frictionless nodal-penalty contact"
        ),
        "thresholds": {
            "relative_error_strict_upper_bound": _HERTZ_LIMIT,
            "force_balance_strict_upper_bound": _EQUILIBRIUM_LIMIT,
        },
        "mesh_sequence": rows,
        "monotone_error_checks": monotone,
        "gates": gates,
        "passed": all(gates.values()),
        "boundary": (
            "This evidence does not qualify arbitrary 3-D geometry, a "
            "deformable sphere, friction, or nonmatching mortar contact."
        ),
    }
    return {**clean, "evidence_hash": _digest(clean)}


def _finite_rotation_evidence() -> dict[str, object]:
    dtype = torch.float64
    nodes = torch.tensor([[0., 0., 0.], [1., 0., 0.],
                          [1., 1., 0.], [0., 1., 0.]], dtype=dtype)
    model = LayeredShell4Model(nodes, torch.tensor([[0, 1, 2, 3]]),
                               200000., .3, .08, 250., 1400., layers=3)
    virgin = LayeredShell4State.virgin(model)
    fixed = torch.tensor([0,1,2,3,4,5, 8,9,10,11, 14,15,16,17,
                          18,19,20,21,22,23])
    load = torch.zeros(model.n_dofs, dtype=dtype); load[6] = load[12] = 2.
    accepted = solve_finite_rotation_arc_path(
        model, load, fixed, steps=1, step_size=.01, maximum_step=.01,
        load_scale=.1, initial_state=virgin, tolerance=2e-7,
    )
    point = accepted.points[0]
    response = assemble_finite_rotation_layered_shell4(
        model, point.displacement, virgin, tangent=False
    )
    free = torch.tensor([6, 7, 12, 13])
    residual = float(torch.linalg.vector_norm(
        response.internal_force[free] - point.load_factor*load[free]
    ))
    before = tuple(p.plastic_strain.clone() for e in point.state.points
                   for q in e for p in q)
    rejected = solve_finite_rotation_arc_path(
        model, 1000*load, fixed, steps=1, step_size=.2, load_scale=.1,
        minimum_step=.15, max_iterations=0, initial_state=point.state,
        initial_displacement=point.displacement,
        initial_load_factor=point.load_factor,
    )
    after = tuple(p.plastic_strain for e in rejected.committed_state.points
                  for q in e for p in q)
    rollback = all(torch.equal(a, b) for a, b in zip(before, after))
    clean = {
        "status": "qualified_integration_primitive",
        "scope": "finite-rotation layered Shell4 arc corrector and transactional J2 commit/rollback",
        "accepted_points": len(accepted.points),
        "accepted_equilibrium_norm": residual,
        "equilibrium_strict_upper_bound": 2e-7,
        "rejected_path_converged": rejected.converged,
        "rejected_path_preserved_committed_history": rollback,
        "passed": bool(accepted.converged and residual < 2e-7
                       and not rejected.converged and rollback),
        "boundary": "This primitive qualification is not panel peak/post-peak evidence.",
    }
    return {**clean, "evidence_hash": _digest(clean)}


def run_industrial_p0_phase3_qualification() -> dict[str, object]:
    """Execute phase-3 evidence and return a deterministic signed report."""
    hertz = _hertz_evidence()
    finite_rotation = _finite_rotation_evidence()
    panel = run_panel_ultimate_qualification()
    categories = {
        "axisymmetric_deformable_halfspace_hertz": {
            "status": "qualified", "reason": "all executable Hertz gates pass"
        },
        "finite_rotation_layered_shell_arc_integration": {
            "status": "qualified",
            "reason": "accepted equilibrium and rejected-step history rollback pass",
        },
        "deformable_to_deformable_contact_3d": {
            "status": "blocked",
            "reason": "no global 3-D sphere/half-space FE solve or convergence evidence",
        },
        "marine_panel_peak_and_postpeak": {
            "status": "blocked",
            "reason": (
                "4/8/12 meshes and two arc-length controls have no executed "
                "peak/post-peak convergence, equilibrium, or energy results"
            ),
        },
    }
    panel_evidence = {
        "status": panel["status"],
        "passed": panel["passed"],
        "model": panel["model"],
        "mesh_matrix": panel["mesh_matrix"],
        "arc_length_controls": panel["arc_length_controls"],
        "acceptance": panel["acceptance"],
        "blocking_capability": categories["marine_panel_peak_and_postpeak"]["reason"],
        "boundary": panel["boundary"],
    }
    panel_evidence["evidence_hash"] = _digest(panel_evidence)
    clean = {
        "schema": SCHEMA,
        # Report success means every positive claim passes and every absent
        # capability remains explicitly fail-closed; it does not mean all P0
        # capabilities are qualified.
        "passed": bool(hertz["passed"] and finite_rotation["passed"]
                       and panel["status"] == "blocked"),
        "hertz_axisymmetric": hertz,
        "finite_rotation_layered_shell": finite_rotation,
        "marine_panel_ultimate": panel_evidence,
        "categories": categories,
        "qualified_count": sum(v["status"] == "qualified" for v in categories.values()),
        "blocked_count": sum(v["status"] == "blocked" for v in categories.values()),
    }
    return {**clean, "report_hash": _digest(clean)}


def validate_industrial_p0_phase3_qualification(report: dict[str, object]):
    """Validate schema, nested evidence hashes, claim boundaries and report hash."""
    if report.get("schema") != SCHEMA or report.get("passed") is not True:
        raise ValueError("invalid or failed phase-3 report")
    hertz = report.get("hertz_axisymmetric", {})
    hertz_clean = {k: v for k, v in hertz.items() if k != "evidence_hash"}
    if _digest(hertz_clean) != hertz.get("evidence_hash"):
        raise ValueError("phase-3 Hertz evidence hash mismatch")
    panel = report.get("marine_panel_ultimate", {})
    panel_clean = {k: v for k, v in panel.items() if k != "evidence_hash"}
    if _digest(panel_clean) != panel.get("evidence_hash"):
        raise ValueError("phase-3 panel evidence hash mismatch")
    finite_rotation = report.get("finite_rotation_layered_shell", {})
    finite_clean = {k: v for k, v in finite_rotation.items() if k != "evidence_hash"}
    if _digest(finite_clean) != finite_rotation.get("evidence_hash"):
        raise ValueError("phase-3 finite-rotation evidence hash mismatch")
    categories = report.get("categories", {})
    if categories.get("deformable_to_deformable_contact_3d", {}).get("status") != "blocked":
        raise ValueError("general 3-D Hertz claim must remain blocked")
    if categories.get("marine_panel_peak_and_postpeak", {}).get("status") != "blocked":
        raise ValueError("panel ultimate claim must remain blocked")
    clean = {k: v for k, v in report.items() if k != "report_hash"}
    if _digest(clean) != report.get("report_hash"):
        raise ValueError("phase-3 qualification hash mismatch")
    return report
