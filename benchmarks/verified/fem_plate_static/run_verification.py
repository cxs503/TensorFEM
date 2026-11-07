"""
L2 Plate Static Verification: Clamped Square Plate
===================================================

This module implements verification for a clamped square plate under uniform load.

Analytical reference:
- Timoshenko & Woinowsky-Krieger: w_max = 0.00126 * q * a^4 / D

Verification criteria:
- Relative error < 2% at center
- Mesh convergence with h-refinement (N=4/6/8)
- Convergence order ≥ 1.5
"""

import json
import math
from pathlib import Path
from datetime import datetime, timezone
import platform

import torch
import numpy as np

# Import TensorFEM plate module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from tensorfem.plate import (
    PlateRigidity,
    q4_mindlin_stiffness,
    structured_square_mesh,
)


def clamped_plate_reference_deflection(q: float, a: float, D: float) -> float:
    """Reference center deflection for clamped square plate.
    
    w_max = 0.00126 * q * a^4 / D
    
    From Timoshenko & Woinowsky-Krieger, "Theory of Plates and Shells"
    """
    return 0.00126 * q * a**4 / D


def solve_clamped_plate(n_per_side: int, q: float, a: float, E: float, nu: float, t: float) -> dict:
    """Solve clamped square plate under uniform load using Mindlin Q4 elements."""
    # Generate structured mesh
    xy, conn = structured_square_mesh(n_per_side, side=a)
    
    n_nodes = xy.shape[0]
    n_elements = conn.shape[0]
    
    # Material and section properties
    rigidity = PlateRigidity.isotropic(E, nu, t)
    
    # Assemble global stiffness matrix
    ndof = 3 * n_nodes  # w, θx, θy per node
    K = torch.zeros((ndof, ndof), dtype=torch.float64)
    
    for e in range(n_elements):
        nodes = conn[e]
        xy_elem = xy[nodes]
        ke = q4_mindlin_stiffness(xy_elem, rigidity=rigidity, shear_treatment="mitc")
        
        # Element DOF mapping
        dofs = []
        for node in nodes:
            dofs.extend([3*node, 3*node+1, 3*node+2])
        dofs = torch.tensor(dofs, dtype=torch.long)
        
        K[dofs[:, None], dofs] += ke
    
    # Assemble load vector (uniform pressure)
    F = torch.zeros(ndof, dtype=torch.float64)
    
    # Distribute pressure to nodes using lumped approach
    A_elem = a**2 / n_per_side**2
    for e in range(n_elements):
        nodes = conn[e]
        for node in nodes:
            F[3*node] += q * A_elem / 4.0
    
    # Apply boundary conditions: clamped on all edges
    tol = 1e-10
    edge_nodes = set()
    
    for i in range(n_nodes):
        x, y = xy[i].tolist()
        if (abs(x) < tol or abs(x - a) < tol or 
            abs(y) < tol or abs(y - a) < tol):
            edge_nodes.add(i)
    
    # Constrain w, θx, θy for all edge nodes
    constrained = []
    for node in edge_nodes:
        constrained.extend([3*node, 3*node+1, 3*node+2])
    
    free_dofs = [i for i in range(ndof) if i not in constrained]
    
    # Solve reduced system
    K_free = K[free_dofs][:, free_dofs]
    F_free = F[free_dofs]
    
    u_free = torch.linalg.solve(K_free, F_free)
    
    # Reconstruct full displacement vector
    u = torch.zeros(ndof, dtype=torch.float64)
    for i, dof in enumerate(free_dofs):
        u[dof] = u_free[i]
    
    # Find center node and extract deflection
    center_x, center_y = a/2, a/2
    min_dist = float('inf')
    center_node = 0
    
    for i in range(n_nodes):
        x, y = xy[i].tolist()
        dist = math.sqrt((x - center_x)**2 + (y - center_y)**2)
        if dist < min_dist:
            min_dist = dist
            center_node = i
    
    center_deflection = u[3*center_node].item()
    
    return {
        "center_deflection": center_deflection,
        "n_elements_per_side": n_per_side,
        "n_elements_total": n_elements,
        "n_nodes": n_nodes,
        "center_node": center_node,
        "center_coordinates": [center_x, center_y],
    }


def run_verification():
    """Run complete L2 plate verification suite."""
    
    # Problem parameters
    q = 1.0  # Uniform pressure
    a = 1.0  # Plate side length
    E = 210e9  # Young's modulus (Pa)
    nu = 0.3  # Poisson's ratio
    t = 0.01  # Thickness (m)
    
    # Bending stiffness
    D = E * t**3 / (12 * (1 - nu**2))
    
    # Mesh levels for convergence study (small for computational efficiency)
    mesh_levels = [4, 6, 8]
    
    results = {
        "benchmark": "L2 Plate Static Verification - Clamped Square Plate",
        "generated_by": "benchmarks/verified/fem_plate_static/run_verification.py",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "machine": platform.machine(),
        },
        "case": {},
    }
    
    print("=" * 70)
    print("L2 Plate Verification: Clamped Square Plate under Uniform Load")
    print("=" * 70)
    print(f"Parameters: q={q} Pa, a={a} m, E={E:.3e} Pa, ν={nu}, t={t} m")
    print(f"Bending stiffness D = {D:.3e} N·m")
    print()
    
    # Analytical reference
    w_ref = clamped_plate_reference_deflection(q, a, D)
    print(f"Reference center deflection (Timoshenko): {w_ref:.6e} m")
    print()
    
    # Mesh convergence study
    conv_results = []
    for n_elem in mesh_levels:
        print(f"Solving with {n_elem}×{n_elem} mesh ({n_elem**2} elements)...")
        fe_sol = solve_clamped_plate(n_elem, q, a, E, nu, t)
        w_fe = fe_sol["center_deflection"]
        rel_error = abs(w_fe - w_ref) / abs(w_ref)
        
        conv_results.append({
            "n_elements_per_side": n_elem,
            "n_elements_total": fe_sol["n_elements_total"],
            "n_nodes": fe_sol["n_nodes"],
            "center_deflection": w_fe,
            "relative_error": rel_error,
        })
        
        print(f"  N={n_elem:2d}: w_center = {w_fe:.6e} m, rel error = {rel_error:.6e} ({rel_error*100:.4f}%)")
    
    # Compute convergence order
    errors = [r["relative_error"] for r in conv_results]
    if len(errors) >= 2 and all(e > 0 for e in errors):
        h_ratios = [mesh_levels[i+1]/mesh_levels[i] for i in range(len(mesh_levels)-1)]
        e_ratios = [errors[i]/errors[i+1] for i in range(len(errors)-1)]
        orders = [math.log(r)/math.log(h) for r, h in zip(e_ratios, h_ratios)]
        avg_order = sum(orders) / len(orders)
    else:
        avg_order = 0.0
    
    print()
    print(f"Observed convergence order: {avg_order:.3f}")
    print()
    
    # Store case results
    results["case"] = {
        "description": "Clamped square plate under uniform pressure",
        "analytical_reference": "Timoshenko & Woinowsky-Krieger: w_max = 0.00126*q*a⁴/D",
        "parameters": {
            "q": q,
            "a": a,
            "E": E,
            "nu": nu,
            "t": t,
            "D": D,
        },
        "reference_center_deflection": w_ref,
        "element_type": "Mindlin Q4 with MITC shear treatment",
        "boundary_conditions": "Fully clamped (w=θx=θy=0) on all edges",
        "mesh_convergence": conv_results,
        "observed_convergence_order": avg_order,
        "max_error_at_finest": conv_results[-1]["relative_error"],
    }
    
    # Verification gates
    print("=" * 70)
    print("Verification Gates")
    print("=" * 70)
    
    gates = []
    
    # Gate 1: Center deflection error < 2%
    error_pass = conv_results[-1]["relative_error"] < 0.02
    gates.append({
        "id": "G1.center_deflection_error",
        "description": "Center deflection error < 2%",
        "value": conv_results[-1]["relative_error"],
        "limit": 0.02,
        "relation": "le",
        "passed": error_pass,
    })
    print(f"G1: Center deflection error = {conv_results[-1]['relative_error']*100:.4f}% (< 2%) -> {'PASS' if error_pass else 'FAIL'}")
    
    # Gate 2: Convergence order ≥ 1.5
    order_pass = avg_order >= 1.5
    gates.append({
        "id": "G2.convergence_order",
        "description": "Convergence order ≥ 1.5",
        "value": avg_order,
        "limit": 1.5,
        "relation": "ge",
        "passed": order_pass,
    })
    print(f"G2: Convergence order = {avg_order:.3f} (≥ 1.5) -> {'PASS' if order_pass else 'FAIL'}")
    
    results["gates"] = gates
    results["verdict"] = "PASS" if all(g["passed"] for g in gates) else "FAIL"
    
    print()
    print("=" * 70)
    print(f"VERDICT: {results['verdict']}")
    print("=" * 70)
    
    # Save results
    output_dir = Path(__file__).parent
    result_file = output_dir / "result.json"
    result_file.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to: {result_file}")
    
    return results


if __name__ == "__main__":
    run_verification()
