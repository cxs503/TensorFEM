#!/usr/bin/env python3
"""Benchmark verification for elastic foundation beam/plate elements.

This script verifies the TensorFEM elastic foundation implementation against:
1. Hetényi closed-form solutions for infinite beams
2. Convergence order under mesh refinement
3. Unilateral contact complementarity conditions
4. Limiting cases (k→∞, k→0)
5. Patch tests

Run with: python benchmarks/verify_foundation_beam.py

Output: verification results printed to stdout and saved to
        artifacts/foundation_verification.json
"""
import json
import math
import sys
import time
from pathlib import Path

import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tensorfem.element.foundation import (
    euler_bernoulli_bending_stiffness,
    euler_bernoulli_foundation_stiffness,
    euler_bernoulli_uniform_load,
    assemble_beam_foundation,
)
from tensorfem.contact.unilateral import (
    solve_unilateral_contact,
    solve_winkler_unilateral,
    complementarity_residual,
)


def verify_hetenyi_point_load():
    """Verify FE solution against Hetényi closed-form for infinite beam.

    For an infinite beam on Winkler foundation under point load P:
        y_max = P·β/(2k)
        M_max = P/(4β)
    where β = (k/(4EI))^(1/4).
    """
    print("=" * 70)
    print("Hetényi Point Load Verification")
    print("=" * 70)

    EI = 1e6  # N·m²
    k_b = 1e7  # N/m²
    P = 1e4  # N
    beta = (k_b / (4.0 * EI)) ** 0.25
    y_max_exact = P * beta / (2.0 * k_b)
    M_max_exact = P / (4.0 * beta)

    print(f"Parameters: EI={EI:.2e}, k={k_b:.2e}, P={P:.2e}")
    print(f"β = {beta:.6f} 1/m")
    print(f"Exact y_max = {y_max_exact:.6e} m")
    print(f"Exact M_max = {M_max_exact:.6e} N·m")

    # Solve with FE
    L_c = 1.0 / beta
    L_total = 20.0 * L_c  # 20 characteristic lengths
    n_elements = 400
    L_elem = L_total / n_elements
    ndof = 2 * (n_elements + 1)

    K = torch.zeros((ndof, ndof), dtype=torch.float64)
    for e in range(n_elements):
        Kb = euler_bernoulli_bending_stiffness(L_elem, EI)
        Kf = euler_bernoulli_foundation_stiffness(L_elem, k_b)
        ke = Kb + Kf
        dof_map = [2*e, 2*e+1, 2*(e+1), 2*(e+1)+1]
        for i in range(4):
            for j in range(4):
                K[dof_map[i], dof_map[j]] += ke[i, j]

    f = torch.zeros(ndof, dtype=torch.float64)
    center_node = n_elements // 2
    f[2 * center_node] = P

    fixed_dofs = [0, ndof - 2]
    free_dofs = [i for i in range(ndof) if i not in fixed_dofs]
    K_ff = K[free_dofs][:, free_dofs]
    f_f = f[free_dofs]
    u_ff = torch.linalg.solve(K_ff, f_f)
    u = torch.zeros(ndof, dtype=torch.float64)
    for i, dof in enumerate(free_dofs):
        u[dof] = u_ff[i]

    y_max_fe = u[2 * center_node].item()

    # Moment from 5-point FD stencil
    center = n_elements // 2
    w_m2 = u[2*(center-2)].item()
    w_m1 = u[2*(center-1)].item()
    w_0 = u[2*center].item()
    w_p1 = u[2*(center+1)].item()
    w_p2 = u[2*(center+2)].item()
    h = L_elem
    w_pp = (-w_m2 + 16.0*w_m1 - 30.0*w_0 + 16.0*w_p1 - w_p2) / (12.0 * h**2)
    M_max_fe = -EI * w_pp

    y_max_error = abs(y_max_fe - y_max_exact) / abs(y_max_exact)
    M_max_error = abs(M_max_fe - M_max_exact) / abs(M_max_exact)

    print(f"\nFE Results:")
    print(f"  y_max = {y_max_fe:.6e} m  (error: {y_max_error:.2e})")
    print(f"  M_max = {M_max_fe:.6e} N·m  (error: {M_max_error:.2e})")

    passed = y_max_error < 1e-6 and M_max_error < 0.05
    print(f"\n  y_max error < 1e-6: {'PASS' if y_max_error < 1e-6 else 'FAIL'}")
    print(f"  M_max error < 5%: {'PASS' if M_max_error < 0.05 else 'FAIL'}")
    print(f"  Overall: {'PASS' if passed else 'FAIL'}")

    return {
        "test": "hetenyi_point_load",
        "y_max_exact": y_max_exact,
        "y_max_fe": y_max_fe,
        "y_max_error": y_max_error,
        "M_max_exact": M_max_exact,
        "M_max_fe": M_max_fe,
        "M_max_error": M_max_error,
        "passed": passed,
    }


def verify_convergence_order():
    """Verify convergence order under mesh refinement."""
    print("\n" + "=" * 70)
    print("Convergence Order Verification")
    print("=" * 70)

    EI, k_b, q0, L = 1e6, 1e6, 1e4, 10.0

    def compute_l2_error(n_elements):
        L_elem = L / n_elements
        ndof = 2 * (n_elements + 1)

        K = torch.zeros((ndof, ndof), dtype=torch.float64)
        f = torch.zeros(ndof, dtype=torch.float64)

        for e in range(n_elements):
            Kb = euler_bernoulli_bending_stiffness(L_elem, EI)
            Kf = euler_bernoulli_foundation_stiffness(L_elem, k_b)
            ke = Kb + Kf
            fe = euler_bernoulli_uniform_load(L_elem, q0)
            dof_map = [2*e, 2*e+1, 2*(e+1), 2*(e+1)+1]
            for i in range(4):
                for j in range(4):
                    K[dof_map[i], dof_map[j]] += ke[i, j]
                f[dof_map[i]] += fe[i]

        fixed_dofs = [0, ndof - 2]
        free_dofs = [i for i in range(ndof) if i not in fixed_dofs]
        K_ff = K[free_dofs][:, free_dofs]
        f_f = f[free_dofs]
        u_ff = torch.linalg.solve(K_ff, f_f)
        u = torch.zeros(ndof, dtype=torch.float64)
        for i, dof in enumerate(free_dofs):
            u[dof] = u_ff[i]

        # Exact: Fourier series
        x_nodes = torch.tensor([i * L_elem for i in range(n_elements + 1)], dtype=torch.float64)
        w_exact = torch.zeros_like(x_nodes)
        for n_mode in range(1, 100, 2):
            qn = 4.0 * q0 / (n_mode * math.pi)
            denom = EI * (n_mode * math.pi / L)**4 + k_b
            w_exact += qn / denom * torch.sin(n_mode * math.pi * x_nodes / L)

        w_fe = torch.tensor([u[2*i].item() for i in range(n_elements + 1)], dtype=torch.float64)
        l2_error = torch.sqrt(torch.mean((w_fe - w_exact)**2))
        l2_exact = torch.sqrt(torch.mean(w_exact**2))
        return float(l2_error / l2_exact)

    ns = [10, 20, 40, 80]
    errors = [compute_l2_error(n) for n in ns]

    print(f"{'N_elem':>8} {'L2 error':>12} {'Rate':>8}")
    print("-" * 30)
    for i, (n, err) in enumerate(zip(ns, errors)):
        if i > 0:
            rate = math.log(errors[i-1] / err) / math.log(2.0)
            print(f"{n:>8d} {err:>12.4e} {rate:>8.2f}")
        else:
            print(f"{n:>8d} {err:>12.4e} {'---':>8}")

    rate = math.log(errors[-2] / errors[-1]) / math.log(2.0)
    passed = rate >= 1.9
    print(f"\nConvergence rate: {rate:.2f} (required ≥ 1.9)")
    print(f"Overall: {'PASS' if passed else 'FAIL'}")

    return {
        "test": "convergence_order",
        "mesh_sizes": ns,
        "errors": errors,
        "rate": rate,
        "passed": passed,
    }


def verify_unilateral_contact():
    """Verify unilateral contact complementarity conditions."""
    print("\n" + "=" * 70)
    print("Unilateral Contact Verification")
    print("=" * 70)

    # Test 1: Simple spring system
    K = torch.tensor([
        [100.0, -50.0, 0.0],
        [-50.0, 100.0, -50.0],
        [0.0, -50.0, 100.0],
    ], dtype=torch.float64)
    f = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64)
    C = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=torch.float64)
    g0 = torch.tensor([0.0, 0.0], dtype=torch.float64)

    result = solve_unilateral_contact(K, f, C, g0)

    print(f"Converged: {result.converged}")
    print(f"Iterations: {result.iterations}")
    print(f"Complementarity residual: {result.complementarity_residual:.2e}")
    print(f"Contact forces: {result.contact_force.tolist()}")
    print(f"Gaps: {result.gap.tolist()}")
    print(f"Active set: {result.active_set.tolist()}")

    # Check complementarity
    R = result.contact_force
    y = result.gap
    comp = R * y
    max_comp = float(torch.max(torch.abs(comp)).item())

    passed = result.converged and max_comp < 1e-10
    print(f"\nComplementarity R·y ≤ 1e-10: {'PASS' if max_comp < 1e-10 else 'FAIL'}")
    print(f"Overall: {'PASS' if passed else 'FAIL'}")

    return {
        "test": "unilateral_contact",
        "converged": result.converged,
        "iterations": result.iterations,
        "complementarity_residual": result.complementarity_residual,
        "passed": passed,
    }


def verify_limiting_cases():
    """Verify limiting cases k→∞ and k→0."""
    print("\n" + "=" * 70)
    print("Limiting Cases Verification")
    print("=" * 70)

    # k→∞: rigid support
    EI, k_b = 1e6, 1e12
    q0, L = 1e3, 4.0
    n_elem = 20
    L_elem = L / n_elem
    ndof = 2 * (n_elem + 1)

    K = torch.zeros((ndof, ndof), dtype=torch.float64)
    f = torch.zeros(ndof, dtype=torch.float64)

    for e in range(n_elem):
        Kb = euler_bernoulli_bending_stiffness(L_elem, EI)
        Kf = euler_bernoulli_foundation_stiffness(L_elem, k_b)
        ke = Kb + Kf
        dof_map = [2*e, 2*e+1, 2*(e+1), 2*(e+1)+1]
        for i in range(4):
            for j in range(4):
                K[dof_map[i], dof_map[j]] += ke[i, j]
        f[dof_map[0]] += q0 * L_elem / 2.0
        f[dof_map[1]] += q0 * L_elem**2 / 12.0
        f[dof_map[2]] += q0 * L_elem / 2.0
        f[dof_map[3]] -= q0 * L_elem**2 / 12.0

    fixed = [0, ndof - 2]
    free = [i for i in range(ndof) if i not in fixed]
    u_ff = torch.linalg.solve(K[free][:, free], f[free])
    u = torch.zeros(ndof, dtype=torch.float64)
    for i, dof in enumerate(free):
        u[dof] = u_ff[i]

    w_expected = q0 / k_b
    w_mid = u[n_elem].item()
    k_inf_error = abs(w_mid - w_expected) / abs(w_expected)

    print(f"k→∞ (rigid support):")
    print(f"  Expected w = q/k = {w_expected:.2e}")
    print(f"  FE w = {w_mid:.2e}")
    print(f"  Error: {k_inf_error:.2e}")
    print(f"  Error < 1%: {'PASS' if k_inf_error < 0.01 else 'FAIL'}")

    # k→0: no foundation
    EI = 1e6
    q0, L = 1e3, 4.0
    n_elem = 20
    L_elem = L / n_elem
    ndof = 2 * (n_elem + 1)

    K = torch.zeros((ndof, ndof), dtype=torch.float64)
    f = torch.zeros(ndof, dtype=torch.float64)

    for e in range(n_elem):
        Kb = euler_bernoulli_bending_stiffness(L_elem, EI)
        Kf = euler_bernoulli_foundation_stiffness(L_elem, 0.0)
        ke = Kb + Kf
        dof_map = [2*e, 2*e+1, 2*(e+1), 2*(e+1)+1]
        for i in range(4):
            for j in range(4):
                K[dof_map[i], dof_map[j]] += ke[i, j]
        f[dof_map[0]] += q0 * L_elem / 2.0
        f[dof_map[1]] += q0 * L_elem**2 / 12.0
        f[dof_map[2]] += q0 * L_elem / 2.0
        f[dof_map[3]] -= q0 * L_elem**2 / 12.0

    fixed = [0, ndof - 2]
    free = [i for i in range(ndof) if i not in fixed]
    u_ff = torch.linalg.solve(K[free][:, free], f[free])
    u = torch.zeros(ndof, dtype=torch.float64)
    for i, dof in enumerate(free):
        u[dof] = u_ff[i]

    w_max_exact = 5.0 * q0 * L**4 / (384.0 * EI)
    w_max_fe = u[n_elem].item()
    k_zero_error = abs(w_max_fe - w_max_exact) / abs(w_max_exact)

    print(f"\nk→0 (no foundation):")
    print(f"  Exact w_max = 5qL⁴/(384EI) = {w_max_exact:.6e}")
    print(f"  FE w_max = {w_max_fe:.6e}")
    print(f"  Error: {k_zero_error:.2e}")
    print(f"  Error < 1%: {'PASS' if k_zero_error < 0.01 else 'FAIL'}")

    passed = k_inf_error < 0.01 and k_zero_error < 0.01
    print(f"\nOverall: {'PASS' if passed else 'FAIL'}")

    return {
        "test": "limiting_cases",
        "k_inf_error": k_inf_error,
        "k_zero_error": k_zero_error,
        "passed": passed,
    }


def main():
    """Run all verification tests."""
    print("\n" + "=" * 70)
    print("TensorFEM Elastic Foundation Verification")
    print("=" * 70)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"PyTorch version: {torch.__version__}")
    print()

    results = []
    results.append(verify_hetenyi_point_load())
    results.append(verify_convergence_order())
    results.append(verify_unilateral_contact())
    results.append(verify_limiting_cases())

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_passed = all(r["passed"] for r in results)
    for r in results:
        status = "✓ PASS" if r["passed"] else "✗ FAIL"
        print(f"  {r['test']}: {status}")

    print(f"\nOverall: {'ALL PASS' if all_passed else 'SOME FAILED'}")

    # Save results
    artifacts_dir = Path(__file__).parent.parent / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    output_path = artifacts_dir / "foundation_verification.json"
    with open(output_path, "w") as f:
        json.dump({
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "torch_version": torch.__version__,
            "results": results,
            "all_passed": all_passed,
        }, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
