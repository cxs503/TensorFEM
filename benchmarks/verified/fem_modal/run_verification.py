"""
L3 Modal Verification: Simply Supported Beam + Rectangular Plate
================================================================

This module implements two modal verification cases:
1. Simply supported beam: ω_n = (nπ/L)² √(EI/ρA)
2. Rectangular plate (SSSS): ω_mn = π² √(D/ρh) * [(m/a)² + (n/b)²]

Verification criteria:
- Relative error < 3% for all modes
- Mesh convergence with h-refinement
- Convergence order ≥ 1.5
"""

import json
import math
from pathlib import Path
from datetime import datetime, timezone
import platform

import torch
import numpy as np

# Import TensorFEM modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from tensorfem.modal import (
    uniform_beam_matrices,
    solve_modes,
    simply_supported_exact_angular_frequency,
)
from tensorfem.plate import (
    PlateRigidity,
    q4_mindlin_stiffness,
    structured_square_mesh,
)


def simply_supported_beam_frequency(n: int, L: float, EI: float, rho_a: float) -> float:
    """Analytical natural frequency for simply supported beam."""
    return (n * math.pi / L)**2 * math.sqrt(EI / rho_a)


def rectangular_plate_frequency(m: int, n: int, a: float, b: float, 
                                 D: float, rho_h: float) -> float:
    """Analytical natural frequency for SSSS rectangular plate."""
    return math.pi**2 * math.sqrt(D / rho_h) * ((m/a)**2 + (n/b)**2)


def solve_simply_supported_beam_modal(n_elements: int, n_modes: int,
                                       L: float, EI: float, rho_a: float) -> dict:
    """Solve simply supported beam modal problem."""
    K, M = uniform_beam_matrices(L, n_elements, EI, rho_a)
    constrained = [0, 2*n_elements]
    result = solve_modes(K, M, constrained, modes=n_modes)
    
    return {
        "angular_frequencies": result.angular_frequencies.tolist(),
        "frequencies_hz": result.frequencies_hz.tolist(),
        "n_elements": n_elements,
        "n_modes": n_modes,
    }


def solve_rectangular_plate_modal(n_per_side: int, n_modes: int,
                                   a: float, b: float, E: float, nu: float, 
                                   t: float, rho: float) -> dict:
    """Solve rectangular plate modal problem using Mindlin Q4 elements."""
    xy, conn = structured_square_mesh(n_per_side, side=a)
    
    n_nodes = xy.shape[0]
    n_elements = conn.shape[0]
    
    rigidity = PlateRigidity.isotropic(E, nu, t)
    
    # Assemble global stiffness matrix
    ndof = 3 * n_nodes
    K = torch.zeros((ndof, ndof), dtype=torch.float64)
    
    for e in range(n_elements):
        nodes = conn[e]
        xy_elem = xy[nodes]
        ke = q4_mindlin_stiffness(xy_elem, rigidity=rigidity, shear_treatment="mitc")
        
        dofs = []
        for node in nodes:
            dofs.extend([3*node, 3*node+1, 3*node+2])
        dofs = torch.tensor(dofs, dtype=torch.long)
        
        K[dofs[:, None], dofs] += ke
    
    # Assemble mass matrix with rotational inertia
    M = torch.zeros((ndof, ndof), dtype=torch.float64)
    
    rho_h = rho * t
    rho_I = rho * t**3 / 12.0
    
    for e in range(n_elements):
        nodes = conn[e]
        A_elem = a * b / n_per_side**2
        
        m_trans = rho_h * A_elem / 4.0
        m_rot = rho_I * A_elem / 4.0
        
        for node in nodes:
            M[3*node, 3*node] += m_trans
            M[3*node+1, 3*node+1] += m_rot
            M[3*node+2, 3*node+2] += m_rot
    
    # Apply boundary conditions: simply supported (w=0 on all edges)
    tol = 1e-10
    edge_nodes = set()
    
    for i in range(n_nodes):
        x, y = xy[i].tolist()
        if (abs(x) < tol or abs(x - a) < tol or 
            abs(y) < tol or abs(y - b) < tol):
            edge_nodes.add(i)
    
    constrained = [3*node for node in edge_nodes]
    
    result = solve_modes(K, M, constrained, modes=n_modes)
    
    return {
        "angular_frequencies": result.angular_frequencies.tolist(),
        "frequencies_hz": result.frequencies_hz.tolist(),
        "n_elements_per_side": n_per_side,
        "n_elements_total": n_elements,
        "n_nodes": n_nodes,
        "n_modes": n_modes,
    }


def run_verification():
    """Run complete L3 modal verification suite."""
    
    results = {
        "benchmark": "L3 Modal Verification",
        "generated_by": "benchmarks/verified/fem_modal/run_verification.py",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "machine": platform.machine(),
        },
        "cases": {},
    }
    
    # Case 1: Simply supported beam modal analysis
    print("=" * 70)
    print("Case 1: Simply Supported Beam - Modal Analysis")
    print("=" * 70)
    
    L = 10.0
    EI = 1000.0
    rho_a = 1.0
    n_modes = 5
    mesh_levels = [10, 20, 40]
    
    print(f"Parameters: L={L}, EI={EI}, ρA={rho_a}")
    print(f"Extracting {n_modes} modes")
    print()
    
    analytical_freqs = [simply_supported_beam_frequency(n, L, EI, rho_a) for n in range(1, n_modes+1)]
    print("Analytical frequencies (rad/s):")
    for i, omega in enumerate(analytical_freqs, 1):
        print(f"  Mode {i}: ω_{i} = {omega:.4f} rad/s")
    print()
    
    beam_results = []
    for n_elem in mesh_levels:
        fe_sol = solve_simply_supported_beam_modal(n_elem, n_modes, L, EI, rho_a)
        fe_freqs = fe_sol["angular_frequencies"]
        
        errors = [abs(fe - an) / an for fe, an in zip(fe_freqs, analytical_freqs)]
        max_error = max(errors)
        
        beam_results.append({
            "n_elements": n_elem,
            "frequencies": fe_freqs,
            "relative_errors": errors,
            "max_relative_error": max_error,
        })
        
        print(f"N={n_elem:2d}: max rel error = {max_error:.6e} ({max_error*100:.4f}%)")
        for i, (fe, an, err) in enumerate(zip(fe_freqs, analytical_freqs, errors), 1):
            print(f"  Mode {i}: FE={fe:.4f}, Analytical={an:.4f}, err={err*100:.4f}%")
    
    errors = [r["max_relative_error"] for r in beam_results]
    if len(errors) >= 2 and all(e > 0 for e in errors):
        h_ratios = [mesh_levels[i+1]/mesh_levels[i] for i in range(len(mesh_levels)-1)]
        e_ratios = [errors[i]/errors[i+1] for i in range(len(errors)-1)]
        orders = [math.log(r)/math.log(h) for r, h in zip(e_ratios, h_ratios)]
        avg_order = sum(orders) / len(orders)
    else:
        avg_order = 0.0
    
    print(f"\nObserved convergence order: {avg_order:.3f}")
    print()
    
    results["cases"]["simply_supported_beam_modal"] = {
        "description": "Simply supported beam free vibration, modes 1-5",
        "analytical_formula": "ω_n = (nπ/L)² √(EI/ρA)",
        "parameters": {"L": L, "EI": EI, "rho_a": rho_a},
        "analytical_frequencies": analytical_freqs,
        "mesh_convergence": beam_results,
        "observed_convergence_order": avg_order,
        "max_error_at_finest": beam_results[-1]["max_relative_error"],
    }
    
    # Case 2: Rectangular plate modal analysis
    print("=" * 70)
    print("Case 2: Rectangular Plate (SSSS) - Modal Analysis")
    print("=" * 70)
    
    a = 1.0
    b = 1.0
    E = 210e9
    nu = 0.3
    t = 0.01
    rho = 7850.0
    
    D = E * t**3 / (12 * (1 - nu**2))
    rho_h = rho * t
    
    print(f"Parameters: a={a} m, b={b} m, E={E:.3e} Pa, ν={nu}, t={t} m, ρ={rho} kg/m³")
    print(f"Bending stiffness D = {D:.3e} N·m")
    print(f"Mass per unit area ρh = {rho_h:.3e} kg/m²")
    print()
    
    mode_indices = [(1,1), (1,2), (2,1), (2,2)]
    analytical_plate_freqs = [rectangular_plate_frequency(m, n, a, b, D, rho_h) 
                               for m, n in mode_indices]
    
    print("Analytical frequencies (rad/s):")
    for i, ((m, n), omega) in enumerate(zip(mode_indices, analytical_plate_freqs), 1):
        print(f"  Mode ({m},{n}): ω = {omega:.4f} rad/s")
    print()
    
    # Mesh levels: 6, 8, 10 for better convergence
    plate_mesh_levels = [6, 8, 10]
    plate_results = []
    
    for n_elem in plate_mesh_levels:
        print(f"Solving with {n_elem}×{n_elem} mesh...")
        fe_sol = solve_rectangular_plate_modal(n_elem, 4, a, b, E, nu, t, rho)
        fe_freqs = fe_sol["angular_frequencies"]
        
        errors = [abs(fe - an) / an for fe, an in zip(fe_freqs, analytical_plate_freqs)]
        max_error = max(errors)
        
        plate_results.append({
            "n_elements_per_side": n_elem,
            "n_elements_total": fe_sol["n_elements_total"],
            "frequencies": fe_freqs,
            "relative_errors": errors,
            "max_relative_error": max_error,
        })
        
        print(f"  N={n_elem:2d}: max rel error = {max_error:.6e} ({max_error*100:.4f}%)")
        for i, ((m, n), fe, an, err) in enumerate(zip(mode_indices, fe_freqs, 
                                                        analytical_plate_freqs, errors), 1):
            print(f"    Mode ({m},{n}): FE={fe:.4f}, Analytical={an:.4f}, err={err*100:.4f}%")
    
    errors = [r["max_relative_error"] for r in plate_results]
    if len(errors) >= 2 and all(e > 0 for e in errors):
        h_ratios = [plate_mesh_levels[i+1]/plate_mesh_levels[i] for i in range(len(plate_mesh_levels)-1)]
        e_ratios = [errors[i]/errors[i+1] for i in range(len(errors)-1)]
        orders = [math.log(r)/math.log(h) for r, h in zip(e_ratios, h_ratios)]
        avg_order_plate = sum(orders) / len(orders)
    else:
        avg_order_plate = 0.0
    
    print(f"\nObserved convergence order: {avg_order_plate:.3f}")
    print()
    
    results["cases"]["rectangular_plate_modal"] = {
        "description": "Simply supported rectangular plate free vibration, modes 1-4",
        "analytical_formula": "ω_mn = π² √(D/ρh) * [(m/a)² + (n/b)²]",
        "parameters": {"a": a, "b": b, "E": E, "nu": nu, "t": t, "rho": rho, "D": D, "rho_h": rho_h},
        "mode_indices": mode_indices,
        "analytical_frequencies": analytical_plate_freqs,
        "mesh_convergence": plate_results,
        "observed_convergence_order": avg_order_plate,
        "max_error_at_finest": plate_results[-1]["max_relative_error"],
    }
    
    # Verification gates
    print("=" * 70)
    print("Verification Gates")
    print("=" * 70)
    
    gates = []
    
    beam_pass = beam_results[-1]["max_relative_error"] < 0.03
    gates.append({
        "id": "G1.beam_modal_error",
        "description": "Simply supported beam modal error < 3%",
        "value": beam_results[-1]["max_relative_error"],
        "limit": 0.03,
        "relation": "le",
        "passed": beam_pass,
    })
    print(f"G1: Beam modal error = {beam_results[-1]['max_relative_error']*100:.4f}% (< 3%) -> {'PASS' if beam_pass else 'FAIL'}")
    
    plate_pass = plate_results[-1]["max_relative_error"] < 0.03
    gates.append({
        "id": "G2.plate_modal_error",
        "description": "Rectangular plate modal error < 3%",
        "value": plate_results[-1]["max_relative_error"],
        "limit": 0.03,
        "relation": "le",
        "passed": plate_pass,
    })
    print(f"G2: Plate modal error = {plate_results[-1]['max_relative_error']*100:.4f}% (< 3%) -> {'PASS' if plate_pass else 'FAIL'}")
    
    order_pass = min(avg_order, avg_order_plate) >= 1.5
    gates.append({
        "id": "G3.convergence_order",
        "description": "Convergence order ≥ 1.5",
        "value": min(avg_order, avg_order_plate),
        "limit": 1.5,
        "relation": "ge",
        "passed": order_pass,
    })
    print(f"G3: Convergence order = {min(avg_order, avg_order_plate):.3f} (≥ 1.5) -> {'PASS' if order_pass else 'FAIL'}")
    
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
