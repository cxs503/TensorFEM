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
G2  Monotone convergence of displacement with mesh refinement.
G3  Observed convergence order ≥ 1.0 (bending-dominated thin shell).
G4  Coarsest-mesh displacement within 50 % of reference (sanity).

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
# Import the production code
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tensorfem.pinched_cylinder import (
    REFERENCE_DISPLACEMENT,
    pinched_cylinder_model,
    PinchedCylinderResult,
)
from tensorfem.cylindrical_shell4 import cylindrical_shell4_stiffness

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


def observed_order(d1: float, d2: float, d3: float, r: float = 2.0) -> float:
    """Richardson observed order from three successive refinements."""
    e1 = abs(d1 - REFERENCE_DISPLACEMENT)
    e2 = abs(d2 - REFERENCE_DISPLACEMENT)
    e3 = abs(d3 - REFERENCE_DISPLACEMENT)
    if e2 < 1e-30 or e3 < 1e-30 or e1 / e2 < 1e-6:
        return float("inf")
    return math.log(e1 / e2) / math.log(r)


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

    # Convergence orders
    convergence = []
    for i in range(len(results) - 2):
        r1, r2, r3 = results[i], results[i + 1], results[i + 2]
        order = observed_order(r1.displacement, r2.displacement, r3.displacement)
        convergence.append({
            "mesh_range": f"{r1.mesh_size}x → {r3.mesh_size}x",
            "observed_order": order if order != float("inf") else "inf (near-exact)",
        })
    result_payload["convergence"] = convergence

    result_path = OUT / "result.json"
    result_path.write_text(json.dumps(result_payload, indent=2) + "\n")
    print(f"  Wrote {result_path}")

    # ---- gates_verified.json ----
    finest = results[-1]
    coarsest = results[0]
    displacements = [r.displacement for r in results]
    monotone = all(
        abs(displacements[i] - REFERENCE_DISPLACEMENT) > abs(displacements[i + 1] - REFERENCE_DISPLACEMENT)
        for i in range(len(displacements) - 1)
    )
    # Observed order from last three levels
    if len(results) >= 3:
        order_val = observed_order(results[-3].displacement, results[-2].displacement, results[-1].displacement)
    else:
        order_val = float("inf")

    gates = [
        {
            "id": "G1_finest_accuracy",
            "description": "Finest mesh (32x32) displacement within 3% of reference",
            "threshold": 0.03,
            "actual": finest.relative_error,
            "passed": bool(finest.relative_error <= 0.03),
        },
        {
            "id": "G2_monotone_convergence",
            "description": "Monotone convergence of displacement error with mesh refinement",
            "threshold": "monotone",
            "actual": "monotone" if monotone else "non-monotone",
            "passed": monotone,
        },
        {
            "id": "G3_convergence_order",
            "description": "Observed convergence order >= 1.0",
            "threshold": 1.0,
            "actual": order_val if order_val != float("inf") else 999.0,
            "passed": bool(order_val >= 1.0) if order_val != float("inf") else True,
        },
        {
            "id": "G4_coarsest_sanity",
            "description": "Coarsest mesh (4x4) displacement within 50% of reference",
            "threshold": 0.50,
            "actual": coarsest.relative_error,
            "passed": bool(coarsest.relative_error <= 0.50),
        },
    ]

    gates_payload = {
        "schema": "tensorfem.verified-gates.v1",
        "anchor": "fem_pinched_cylinder",
        "title": "MacNeal–Harder pinched cylinder",
        "mission": "selfstart-m6585",
        "criteria": {
            "finest_error": "<= 3%",
            "convergence": "monotone",
            "order": ">= 1.0",
            "coarsest_sanity": "<= 50%",
        },
        "gates": gates,
        "passed": all(g["passed"] for g in gates),
        "failed_gates": [g["id"] for g in gates if not g["passed"]],
        "limitations": [
            "Cylindrical Shell4 is specialised for cylindrical geometry only",
            "General curved shells require either specialised elements or improved flat-facet formulations",
            "Reference value is from a published benchmark; no independent analytical solution exists for this configuration",
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
    all_pass = all(g["passed"] for g in gates)
    print(f"\n  Overall verdict: {'PASS' if all_pass else 'FAIL'}")
    for g in gates:
        status = "PASS" if g["passed"] else "FAIL"
        print(f"    {g['id']}: {status} (actual={g['actual']}, threshold={g['threshold']})")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
