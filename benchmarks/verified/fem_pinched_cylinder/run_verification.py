"""MacNeal–Harder pinched-cylinder verification anchor.

Physics
-------
A thin circular cylinder (R=300 in, L=600 in, t=3 in) is pinched by two
diametrically opposed unit forces at midspan.  The reference displacement
at the load point is **-1.8248×10⁻⁵ in** (MacNeal & Harder, 1985).

This is the canonical test for *inextensional bending* in thin shells:
the membrane strain energy is nearly zero and the response is dominated
by bending.  Elements that suffer from shear or membrane locking produce
displacements orders of magnitude too small.

Oracle
------
Published reference: MacNeal, R. L. & Harder, R. L. (1985),
"A proposed standard set of problems to test finite element accuracy",
*Finite Elements in Analysis and Design*, 1(1), 3–20.
Reference displacement at load point: **-1.8248×10⁻⁵**.

Mesh ladder
-----------
Symmetry octant discretised with N×N cylindrical Shell4 elements
(N = 4, 8, 16, 32).  The cylindrical element retains exact metric
and curvature terms, so geometric error vanishes and only discretisation
error remains.

Gates
-----
G1  Finest-mesh displacement within 3 % of reference.
G2  Quantity of interest monotone along the ladder.
G3  Observed convergence order ≥ 1.0 (bending-dominated thin shell).
G4  Coarsest-mesh displacement within 50 % of reference (sanity).
G5  Richardson limit of the ladder within 3 % — the *certified* accuracy.
G6  Finest-mesh error inside the GCI band (ladder in the asymptotic range).

G5/G6 exist because G1 alone is not evidence of accuracy: the error sequence
here is 2.86 % → 3.35 % → 0.17 % → 1.82 %, which "passes" a 3 % gate on the
finest mesh although the ladder converges to 3.27 % from the reference.  A
gate that cannot be answered is reported as ``WITHHELD``, never as a pass.

Usage
-----
    PYTHONPATH=src python benchmarks/verified/fem_pinched_cylinder/run_verification.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Pin the import to *this* checkout.  The box carries several editable
# tensorfem installs (one per agent worktree); without an explicit pin the
# anchor verifies whichever checkout happened to win on sys.path -- the very
# provenance hole that made this anchor import ../m14687 instead of the tree
# under test.
# ---------------------------------------------------------------------------
_REPO_SRC = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(_REPO_SRC))

import tensorfem  # noqa: E402  (must follow the path pin)

if _REPO_SRC.resolve() not in Path(tensorfem.__file__).resolve().parents:
    raise SystemExit(
        f"provenance error: tensorfem resolved to {tensorfem.__file__}, outside "
        f"this checkout ({_REPO_SRC})"
    )

from tensorfem.pinched_cylinder import (
    REFERENCE_DISPLACEMENT,
    pinched_cylinder_model,
    PinchedCylinderResult,
)
from tensorfem.cylindrical_shell4 import cylindrical_shell4_stiffness
from tensorfem.refinement_gate import evaluate_reference_ladder, summarise

_TENSORFEM_FILE = Path(tensorfem.__file__).resolve()
if _REPO_SRC.resolve() not in _TENSORFEM_FILE.parents:
    raise SystemExit(
        f"provenance error: tensorfem resolved to {_TENSORFEM_FILE}, "
        f"which is outside this checkout ({_REPO_SRC.resolve()})"
    )

# ---------------------------------------------------------------------------
# Harness helpers (minimal, self-contained)
# ---------------------------------------------------------------------------
OUT = Path(__file__).parent


def solve_linear(n: int) -> PinchedCylinderResult:
    """Solve the pinched cylinder on an n×n symmetry-octant mesh."""
    mesh, loads, fixed, probe = pinched_cylinder_model(n)
    ndof = len(loads)
    K = torch.zeros((ndof, ndof), dtype=loads.dtype, device=loads.device)
    for conn in mesh.elements:
        ids = torch.stack(tuple(6 * conn + k for k in range(6)), 1).reshape(-1)
        ke = cylindrical_shell4_stiffness(
            mesh.parameters[conn], mesh.radius, mesh.young,
            mesh.poisson, mesh.thickness)
        K[ids[:, None], ids] += ke
    constrained = torch.tensor(sorted(fixed), dtype=torch.long, device=loads.device)
    mask = torch.ones(ndof, dtype=torch.bool, device=loads.device)
    mask[constrained] = False
    free = torch.nonzero(mask).flatten()
    q = torch.zeros_like(loads)
    q[free] = torch.linalg.solve(K[free[:, None], free], loads[free])
    return PinchedCylinderResult(n, float(q[probe]))


def grade_ladder(results) -> dict:
    """Grade the refinement ladder with the canonical refinement-gate module."""
    levels = [(1.0 / r.mesh_size, r.displacement, f"{r.mesh_size}x{r.mesh_size}")
              for r in results]
    return evaluate_reference_ladder(
        levels,
        reference_value=REFERENCE_DISPLACEMENT,
        tolerance=0.03,
        min_order=1.0,
        coarsest_sanity=0.50,
        qoi="radial displacement at the load point",
        reference_label="MacNeal & Harder (1985)",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    mesh_sizes = [4, 8, 16, 32]
    results = []
    for n in mesh_sizes:
        print(f"  Solving n={n} ...", flush=True)
        r = solve_linear(n)
        results.append(r)
        print(f"    u = {r.displacement:.6e}  (ref={REFERENCE_DISPLACEMENT:.6e}, err={r.relative_error*100:.2f}%)")

    # ---- result.json ----
    result_payload = {
        "benchmark": "MacNeal–Harder pinched cylinder",
        "reference": {
            "value": REFERENCE_DISPLACEMENT,
            "source": "MacNeal & Harder (1985), FEAD 1(1), 3–20",
            "quantity": "Radial displacement at load point (in)",
        },
        "problem": {
            "geometry": "Circular cylinder, R=300 in, L=600 in, t=3 in",
            "material": "E=3e6 psi, nu=0.3",
            "load": "Two diametrically opposed unit forces at midspan",
            "boundary_conditions": "Rigid diaphragms at ends, symmetry octant",
        },
        "element": {
            "name": "Cylindrical Shell4",
            "formulation": "Reissner–Mindlin with exact cylindrical metric",
            "integration": "2x2 membrane+bending, 1-point selective shear",
        },
        "levels": [],
    }
    for r in results:
        result_payload["levels"].append({
            "mesh": f"{r.mesh_size}x{r.mesh_size}",
            "n_elements": r.mesh_size ** 2,
            "displacement": r.displacement,
            "reference": REFERENCE_DISPLACEMENT,
            "relative_error": r.relative_error,
            "relative_error_pct": r.relative_error * 100,
        })

    # Convergence orders (canonical Richardson order per mesh triple)
    ladder = grade_ladder(results)
    convergence = []
    for entry in ladder["observed_orders"]:
        convergence.append({
            "mesh_range": f"{entry['from']} → {entry['to']}",
            "refinement_ratio": entry["ratio"],
            "observed_order": entry["order"],
            "reason": entry.get("reason", ""),
        })
    result_payload["convergence"] = convergence
    result_payload["ladder_grading"] = ladder

    result_path = OUT / "result.json"
    result_path.write_text(json.dumps(result_payload, indent=2) + "\n")
    print(f"  Wrote {result_path}")

    # ---- gates_verified.json ----
    summary = summarise(ladder)
    gates = [
        {
            "id": f"G{index}_{g['id']}",
            "description": g["description"],
            "status": g["status"],
            "passed": bool(g["passed"]),
            "threshold": g["threshold"],
            "actual": g["actual"],
            "detail": g["detail"],
        }
        for index, g in enumerate(ladder["gates"], start=1)
    ]

    gates_payload = {
        "schema": "tensorfem.verified-gates.v1",
        "anchor": "fem_pinched_cylinder",
        "title": "MacNeal–Harder pinched cylinder",
        "mission": "selfstart-m6585",
        "criteria": {
            "finest_error": "<= 3%",
            "asymptotic_error": "<= 3% (Richardson limit, certified accuracy)",
            "convergence": "monotone",
            "order": ">= 1.0",
            "coarsest_sanity": "<= 50%",
            "asymptotic_range": "finest error inside the GCI band",
        },
        "verdicts": summary,
        "reported_accuracy_finest": ladder["reported_accuracy_finest"],
        "certified_accuracy": ladder["certified_accuracy"],
        "gates": gates,
        "overall_verdict": summary["overall_verdict"],
        "passed": bool(summary["passed"]),
        "failed_gates": [g["id"] for g in gates if g["status"] == "FAIL"],
        "withheld_gates": [g["id"] for g in gates if g["status"] == "WITHHELD"],
        "limitations": [
            "Cylindrical Shell4 is specialised for cylindrical geometry only",
            "General curved shells require either specialised elements or improved flat-facet formulations",
            "Reference value is from a published benchmark; no independent analytical solution exists for this configuration",
            "The error ladder is non-monotone (4x -> 8x moves away from the reference), so the "
            "finest-mesh reading understates the asymptotic error: certified accuracy is the "
            "Richardson limit (G5), not the 32x32 error",
        ],
    }
    gates_path = OUT / "gates_verified.json"
    gates_path.write_text(json.dumps(gates_payload, indent=2) + "\n")
    print(f"  Wrote {gates_path}")

    # ---- Convergence plot ----
    fig, ax = plt.subplots(figsize=(7, 5))
    n_vals = [r.mesh_size for r in results]
    errors = [r.relative_error * 100 for r in results]
    ax.semilogy(n_vals, errors, "o-", linewidth=2, markersize=8, label="Cylindrical Shell4")
    ax.axhline(3.0, color="red", linestyle="--", linewidth=1, label="3% gate")
    ax.axhline(0.1, color="gray", linestyle=":", linewidth=1, label="0.1%")
    ax.set_xlabel("Mesh size N (elements per side, symmetry octant)")
    ax.set_ylabel("Relative error in load-point displacement (%)")
    ax.set_title("MacNeal–Harder Pinched Cylinder — Convergence")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(n_vals)
    fig.tight_layout()
    fig.savefig(OUT / "convergence.png", dpi=150)
    plt.close(fig)
    print(f"  Wrote {OUT / 'convergence.png'}")

    # ---- Displacement comparison bar chart ----
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    labels = [f"{r.mesh_size}x{r.mesh_size}" for r in results]
    vals = [r.displacement for r in results] + [REFERENCE_DISPLACEMENT]
    labels.append("Reference")
    colors = ["steelblue"] * len(results) + ["red"]
    ax2.bar(labels, vals, color=colors, alpha=0.8)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylabel("Radial displacement at load point (in)")
    ax2.set_title("Pinched Cylinder — Displacement vs Mesh")
    ax2.tick_params(axis="x", rotation=30)
    fig2.tight_layout()
    fig2.savefig(OUT / "displacement_comparison.png", dpi=150)
    plt.close(fig2)
    print(f"  Wrote {OUT / 'displacement_comparison.png'}")

    # ---- Summary ----
    print(f"\n  Overall verdict: {summary['overall_verdict']}")
    print(f"    accuracy_verdict={summary['accuracy_verdict']} "
          f"(reported finest {ladder['reported_accuracy_finest']*100:.3f}%, "
          f"certified {ladder['certified_accuracy']*100:.3f}%)")
    print(f"    study_verdict={summary['study_verdict']} unproven={summary['unproven']}")
    for g in gates:
        print(f"    {g['id']}: {g['status']} (actual={g['actual']}, threshold={g['threshold']})")

    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
