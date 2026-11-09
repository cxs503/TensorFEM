"""Thermal conduction verification anchor for TensorFEM.

Verifies the ``tensorfem.thermal`` module (LINE2 + Q4 conduction, consistent
capacity, prescribed temperature, nodal heat, boundary convection, volumetric
source, transient backward-Euler) against closed-form analytical solutions.

Three graded ladders:
  A. 1D steady rod with prescribed T — machine-precision gate (LINE2 nodally exact)
  B. 1D steady rod with uniform heat generation — machine-precision gate
  C. 2D Q4 transient cooling of a rectangle — <3% L2 error at coarsest mesh,
     h-refinement order >= 1.0

References:
  - Carslaw & Jaeger, Conduction of Heat in Solids, 2nd ed., Sec. 3.4, 4.2
  - Incropera & DeWitt, Fundamentals of Heat and Mass Transfer, Sec. 3.5
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorfem.thermal import ThermalModel, solve_steady_thermal, solve_transient_thermal

FLOAT = torch.float64
DEV = torch.device("cpu")


def exact_rod_prescribed(x, T0, TL, L):
    return T0 + (TL - T0) * x / L


def exact_rod_generation(x, q_dot, k, L, T0):
    return T0 + q_dot / (2 * k) * (L**2 / 4 - x**2)


def exact_rectangle_transient(x, y, t, a, b, alpha, T0, n_terms=80):
    result = torch.zeros_like(x)
    for nx in range(1, n_terms, 2):
        sx = 4.0 / (nx * math.pi) * torch.sin(nx * math.pi * x / a) * math.exp(-nx**2 * math.pi**2 * alpha * t / a**2)
        for ny in range(1, n_terms, 2):
            sy = 4.0 / (ny * math.pi) * torch.sin(ny * math.pi * y / b) * math.exp(-ny**2 * math.pi**2 * alpha * t / b**2)
            result += sx * sy
    return T0 * result


def rod_mesh_1d(n_elem, L=1.0):
    nodes = torch.linspace(0, L, n_elem + 1, dtype=FLOAT, device=DEV).unsqueeze(1)
    elems = torch.arange(n_elem, dtype=torch.long, device=DEV).unsqueeze(1)
    elems = torch.cat([elems, elems + 1], dim=1)
    return nodes, elems


def rect_mesh_2d(nx, ny, a=1.0, b=0.6):
    xs = torch.linspace(0, a, nx + 1, dtype=FLOAT, device=DEV)
    ys = torch.linspace(0, b, ny + 1, dtype=FLOAT, device=DEV)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    nodes = torch.stack([gx.flatten(), gy.flatten()], dim=1)
    conn = []
    for i in range(nx):
        for j in range(ny):
            n0 = i * (ny + 1) + j
            conn.append([n0, n0 + 1, n0 + ny + 2, n0 + ny + 1])
    elems = torch.tensor(conn, dtype=torch.long, device=DEV)
    return nodes, elems


def case_a(n_elem):
    L, T0, TL = 1.0, 100.0, 0.0
    k, rho, cp = 50.0, 7850.0, 500.0
    nodes, elems = rod_mesh_1d(n_elem, L)
    model = ThermalModel(
        nodes=nodes, elements=elems,
        conductivity=torch.tensor(k, dtype=FLOAT),
        density=torch.tensor(rho, dtype=FLOAT),
        specific_heat=torch.tensor(cp, dtype=FLOAT),
        fixed_nodes=torch.tensor([0, n_elem], dtype=torch.long),
        fixed_temperatures=torch.tensor([T0, TL], dtype=FLOAT),
        element_type="line2",
    )
    T = solve_steady_thermal(model)
    x_nodes = nodes.flatten()
    T_exact = exact_rod_prescribed(x_nodes, T0, TL, L)
    max_err = float(torch.max(torch.abs(T - T_exact)))
    l2_err = float(torch.sqrt(torch.mean((T - T_exact) ** 2)))
    return {"n_elements": n_elem, "n_nodes": n_elem + 1, "max_absolute_error": max_err, "l2_error": l2_err}


def case_b(n_elem):
    L, T0, q_dot = 1.0, 0.0, 1e6
    k, rho, cp = 50.0, 7850.0, 500.0
    nodes, elems = rod_mesh_1d(n_elem, L)
    model = ThermalModel(
        nodes=nodes, elements=elems,
        conductivity=torch.tensor(k, dtype=FLOAT),
        density=torch.tensor(rho, dtype=FLOAT),
        specific_heat=torch.tensor(cp, dtype=FLOAT),
        fixed_nodes=torch.tensor([0, n_elem], dtype=torch.long),
        fixed_temperatures=torch.tensor([T0, T0], dtype=FLOAT),
        heat_source=torch.tensor(q_dot, dtype=FLOAT),
        element_type="line2",
    )
    T = solve_steady_thermal(model)
    x_nodes = nodes.flatten() - L / 2
    T_exact = exact_rod_generation(x_nodes, q_dot, k, L, T0)
    max_err = float(torch.max(torch.abs(T - T_exact)))
    l2_err = float(torch.sqrt(torch.mean((T - T_exact) ** 2)))
    center_idx = n_elem // 2
    T_center_fem = float(T[center_idx])
    T_center_exact = float(T_exact[center_idx])
    return {"n_elements": n_elem, "n_nodes": n_elem + 1, "max_absolute_error": max_err, "l2_error": l2_err,
            "T_center_fem": T_center_fem, "T_center_exact": T_center_exact,
            "T_center_rel_error": abs(T_center_fem / T_center_exact - 1) if T_center_exact != 0 else 0}


def case_c(nx, ny):
    a, b = 1.0, 0.6
    k, rho, cp, T0 = 50.0, 7850.0, 500.0, 100.0
    alpha = k / (rho * cp)
    nodes, elems = rect_mesh_2d(nx, ny, a, b)
    n_nodes = len(nodes)
    edge_nodes = set()
    for i in range(nx + 1):
        edge_nodes.add(i * (ny + 1))
        edge_nodes.add(i * (ny + 1) + ny)
    for j in range(ny + 1):
        edge_nodes.add(j)
        edge_nodes.add(nx * (ny + 1) + j)
    edge_nodes = sorted(edge_nodes)
    fixed_nodes = torch.tensor(edge_nodes, dtype=torch.long)
    fixed_temps = torch.zeros(len(edge_nodes), dtype=FLOAT)
    T_init = T0 * torch.ones(n_nodes, dtype=FLOAT)
    T_init[fixed_nodes] = 0.0
    model = ThermalModel(
        nodes=nodes, elements=elems,
        conductivity=torch.tensor(k, dtype=FLOAT),
        density=torch.tensor(rho, dtype=FLOAT),
        specific_heat=torch.tensor(cp, dtype=FLOAT),
        fixed_nodes=fixed_nodes, fixed_temperatures=fixed_temps,
        element_type="q4", thickness=torch.tensor(1.0, dtype=FLOAT),
    )
    t_final, dt = 0.01, 0.001
    n_steps = int(t_final / dt)
    result = solve_transient_thermal(model, T_init, t_final, n_steps=n_steps)
    T = result.temperature
    x_coords = nodes[:, 0]
    y_coords = nodes[:, 1]
    T_exact = exact_rectangle_transient(x_coords, y_coords, t_final, a, b, alpha, T0)
    interior_mask = torch.ones(n_nodes, dtype=torch.bool)
    interior_mask[fixed_nodes] = False
    T_int = T[interior_mask]
    T_exact_int = T_exact[interior_mask]
    max_err = float(torch.max(torch.abs(T_int - T_exact_int)))
    l2_err = float(torch.sqrt(torch.mean((T_int - T_exact_int) ** 2)))
    T_ref = float(torch.max(T_exact_int))
    rel_l2 = l2_err / T_ref if T_ref > 0 else 0.0
    center_idx = int(nx * (ny + 1) / 2 + ny / 2)
    return {"nx": nx, "ny": ny, "n_elements": nx * ny, "n_nodes": n_nodes,
            "t_final": t_final, "n_steps": n_steps,
            "max_absolute_error": max_err, "l2_error": l2_err,
            "relative_l2_error": rel_l2,
            "T_center_fem": float(T[center_idx]), "T_center_exact": float(T_exact[center_idx]),
            "_T_fem": T.tolist()}


def make_figures(results_a, results_b, results_c, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ns = [r["n_elements"] for r in results_a]
    errs = [r["max_absolute_error"] for r in results_a]
    ax.semilogy(ns, errs, "o-")
    ax.set_xlabel("Elements"); ax.set_ylabel("Max abs error")
    ax.set_title("Case A: 1D steady prescribed T (LINE2 exact)")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / "fig_case_a_convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    ns = [r["n_elements"] for r in results_b]
    errs = [r["max_absolute_error"] for r in results_b]
    ax.semilogy(ns, errs, "o-")
    ax.set_xlabel("Elements"); ax.set_ylabel("Max abs error")
    ax.set_title("Case B: 1D steady heat generation (LINE2 exact)")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / "fig_case_b_convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    ns = [r["n_elements"] for r in results_c]
    errs = [r["relative_l2_error"] for r in results_c]
    ax.loglog(ns, errs, "o-", label="Relative L2 error")
    ref_n = [ns[0], ns[-1]]
    ref_e = [errs[0], errs[0] * (ref_n[1] / ref_n[0]) ** (-1.0)]
    ax.loglog(ref_n, ref_e, "k--", alpha=0.5, label="Slope -1.0")
    ax.set_xlabel("Elements"); ax.set_ylabel("Relative L2 error")
    ax.set_title("Case C: 2D Q4 transient cooling (h-refinement)")
    ax.grid(True, alpha=0.3, which="both"); ax.legend()
    fig.savefig(out_dir / "fig_case_c_convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    r = results_c[0]
    nx, ny = r["nx"], r["ny"]
    a, b = 1.0, 0.6
    xs = np.linspace(0, a, nx + 1)
    ys = np.linspace(0, b, ny + 1)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    T_field = np.array(r["_T_fem"]).reshape(nx + 1, ny + 1)
    cf = ax.contourf(gx, gy, T_field, levels=20, cmap="viridis")
    fig.colorbar(cf, ax=ax, label="Temperature")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"Case C: T field at t={r['t_final']}s ({nx}x{ny})")
    ax.set_aspect("equal")
    fig.savefig(out_dir / "fig_case_c_field.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    out_dir = Path(__file__).parent
    t0 = time.time()

    results_a = [case_a(n) for n in [2, 4, 8, 16, 32, 64]]
    results_b = [case_b(n) for n in [2, 4, 8, 16, 32, 64]]
    results_c = [case_c(nx, ny) for nx, ny in [(4, 3), (8, 5), (16, 10), (32, 20)]]

    orders_c = []
    for i in range(1, len(results_c)):
        h_ratio = results_c[i - 1]["n_elements"] / results_c[i]["n_elements"]
        e_ratio = results_c[i - 1]["relative_l2_error"] / results_c[i]["relative_l2_error"]
        orders_c.append(math.log(e_ratio) / math.log(h_ratio) if e_ratio > 0 and h_ratio > 1 else 0.0)

    elapsed = time.time() - t0

    result = {
        "benchmark": "fem_thermal",
        "description": "Thermal conduction verification: steady 1D + transient 2D Q4",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__, "machine": os.uname().machine, "dtype": "torch.float64"},
        "references": {
            "case_a": "T(x) = T0 + (TL-T0)*x/L (LINE2 nodally exact)",
            "case_b": "T(x) = T0 + q_dot/(2k)*(L^2/4 - x^2) (LINE2 nodally exact)",
            "case_c": "T(x,y,t) = T0 * S(x,t;a) * S(y,t;b), Carslaw & Jaeger Sec. 3.4",
        },
        "case_a": {"description": "1D steady rod, prescribed T at both ends", "levels": results_a},
        "case_b": {"description": "1D steady rod, uniform heat generation, T=0 at ends", "levels": results_b},
        "case_c": {"description": "2D Q4 transient cooling of rectangle, T=0 on all edges", "levels": results_c, "convergence_orders": orders_c},
        "elapsed_seconds": round(elapsed, 2),
    }
    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    all_pass_a = all(r["max_absolute_error"] < 1e-10 for r in results_a)
    all_pass_b = all(r["max_absolute_error"] < 1e-10 for r in results_b)
    pass_c_coarse = results_c[0]["relative_l2_error"] < 0.03
    pass_c_order = min(orders_c) >= 1.0 if orders_c else False
    verdict = "PASS" if all([all_pass_a, all_pass_b, pass_c_coarse, pass_c_order]) else "FAIL"

    gates = {
        "schema": "tensorfem.verified-gates.v1",
        "generated_at": result["generated_at"],
        "gates": [
            {"name": "case_a_machine_zero", "description": "Case A: LINE2 nodally exact for linear T", "criterion": "max_absolute_error < 1e-10", "value": max(r["max_absolute_error"] for r in results_a), "threshold": 1e-10, "passed": all_pass_a},
            {"name": "case_b_machine_zero", "description": "Case B: LINE2 nodally exact for quadratic T with source", "criterion": "max_absolute_error < 1e-10", "value": max(r["max_absolute_error"] for r in results_b), "threshold": 1e-10, "passed": all_pass_b},
            {"name": "case_c_coarse_below_3pct", "description": "Case C: coarsest mesh rel L2 < 3%", "criterion": "relative_l2_error < 0.03", "value": results_c[0]["relative_l2_error"], "threshold": 0.03, "passed": pass_c_coarse},
            {"name": "case_c_convergence_order", "description": "Case C: h-refinement order >= 1.0", "criterion": "min(orders) >= 1.0", "value": min(orders_c) if orders_c else 0.0, "threshold": 1.0, "passed": pass_c_order},
        ],
        "verdict": verdict,
    }
    with open(out_dir / "gates_verified.json", "w") as f:
        json.dump(gates, f, indent=2)

    make_figures(results_a, results_b, results_c, out_dir)

    print(f"Thermal verification complete in {elapsed:.1f}s")
    print(f"  Case A: max_err = {max(r['max_absolute_error'] for r in results_a):.2e}")
    print(f"  Case B: max_err = {max(r['max_absolute_error'] for r in results_b):.2e}")
    print(f"  Case C: rel_L2 = {results_c[0]['relative_l2_error']:.4f} (coarsest), orders = {orders_c}")
    print(f"  Verdict: {verdict}")


if __name__ == "__main__":
    main()
