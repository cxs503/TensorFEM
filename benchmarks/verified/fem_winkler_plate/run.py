#!/usr/bin/env python3
"""GAP-FEM-WINKLER anchor: Winkler plate on elastic foundation vs Hetényi.

Physics
-------
An infinite Kirchhoff/Mindlin plate resting on a Winkler elastic foundation,
loaded by a concentrated point force *P* at the origin.  The governing equation
is

    D ∇⁴w + k w = P δ(r) / (2πr)

where *D* = E h³ / [12(1−ν²)] is the flexural rigidity and *k* is the Winkler
foundation modulus (N/m³).  Hetényi (1946) gives the exact closed-form solution

    w(r) = −P / (2π k l²) · kei(r/l),     l = (D/k)^(1/4)

with centre deflection w(0) = P / (8 √(D k)).  The Kelvin kei function is
evaluated via the independent oracle in ``tensorfem.hetenyi_reference``.

For ice-plate applications the parameters are representative of sea-ice
floating on water:

    E = 3 GPa,  ν = 0.3,  h = 0.15 m,  k = ρ_water g = 1025×9.81 ≈ 10055 N/m³

giving l ≈ 1.7 m.  A 10 kN point load (e.g. a small vehicle or equipment)
produces w(0) ≈ 0.5 mm.

Model
-----
The FE model uses TensorFEM's Mindlin-Reissner Q4 element with MITC shear
treatment, augmented by the Winkler foundation stiffness from
``tensorfem.winkler_foundation``.  A quarter-symmetry model is solved with
graded mesh clustering near the load point.  The domain extends 20
characteristic lengths to approximate the infinite plate.

Verification criteria (GAP-FEM-WINKLER acceptance):
    C1  Centre deflection error < 5% vs Hetényi at the finest mesh.
    C2  Equilibrium: foundation reaction ≈ applied load (< 1% residual).
    C3  Deflection profile shape matches kei curve (R² > 0.99).

Outputs: ``result.json``, ``gates_verified.json``,
         ``winkler_deflection_profile.png``, ``winkler_convergence.png``.

Usage::

    PYTHONPATH=src:benchmarks/verified python benchmarks/verified/fem_winkler_plate/run.py
"""

from __future__ import annotations

import math
import platform
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))

from _harness import (
    Gate,
    gate_manifest,
    pyplot,
    utc_now,
    write_json,
)

from tensorfem.foundation_plate_bending import (
    graded_axis,
    infinite_plate_point_load_fem,
    plate_rigidity,
    winkler_characteristic_length,
)
from tensorfem.hetenyi_reference import kelvin_kei

DTYPE = torch.float64

# --- benchmark parameters (GAP-FEM-WINKLER / H3-ICE-PLATE-FEM) -----
YOUNG = 3.0e9          # Pa, ice Young's modulus
POISSON = 0.3          # ice Poisson's ratio
THICKNESS = 0.15       # m, ice thickness
WINKLER = 1025.0 * 9.81  # N/m³, ρ_water × g
LOAD = 10_000.0        # N, point load

LEVELS = [(20, 3.0), (30, 3.5), (40, 4.0), (50, 4.5), (60, 5.0)]


def hetenyi_center_deflection(load, rigidity, winkler):
    return load / (8.0 * math.sqrt(rigidity * winkler))


def hetenyi_profile(r_vals, load, rigidity, winkler):
    ell = winkler_characteristic_length(rigidity, winkler)
    xi = r_vals / ell
    return -(load / (2.0 * math.pi * winkler * ell * ell)) * kelvin_kei(xi)


def extract_xaxis_deflection(sol):
    """Extract w(x, y=0) from the quarter-model FEM solution."""
    nodes = sol.nodes
    length = sol.metadata['domain_half_width']
    tol = 1e-9 * length
    on_xaxis = nodes[:, 1].abs() <= tol
    x_coords = nodes[on_xaxis, 0].numpy()
    node_indices = torch.nonzero(on_xaxis, as_tuple=False).flatten()
    w_dof_indices = 3 * node_indices
    w_fem = sol.deflection[w_dof_indices].numpy()
    order = x_coords.argsort()
    return x_coords[order], w_fem[order]


def run_level(n, beta):
    t0 = time.perf_counter()
    fem_sol = infinite_plate_point_load_fem(
        LOAD, young=YOUNG, poisson=POISSON, thickness=THICKNESS,
        winkler=WINKLER, half_width_in_lengths=20.0,
        n=n, beta=beta, shear_treatment="mitc", dense_limit=8000,
    )
    elapsed = time.perf_counter() - t0
    rigidity = plate_rigidity(YOUNG, POISSON, THICKNESS)
    w0_hetenyi = hetenyi_center_deflection(LOAD, rigidity, WINKLER)
    w0_fem = float(fem_sol.deflection[0].item())
    centre_error = abs(w0_fem - w0_hetenyi) / abs(w0_hetenyi)
    load_applied = float(fem_sol.applied_load)
    reaction = float(fem_sol.foundation_reaction)
    equilibrium_error = abs(reaction - load_applied) / load_applied if load_applied > 0 else 0.0

    ell = winkler_characteristic_length(rigidity, WINKLER)
    x_axis, w_fem_x = extract_xaxis_deflection(fem_sol)
    w_hetenyi_x = hetenyi_profile(torch.as_tensor(x_axis, dtype=DTYPE), LOAD, rigidity, WINKLER).numpy()
    mask = x_axis > 1e-6
    r_squared = 0.0
    if mask.sum() > 2:
        ss_res = ((w_fem_x[mask] - w_hetenyi_x[mask]) ** 2).sum()
        ss_tot = ((w_hetenyi_x[mask] - w_hetenyi_x[mask].mean()) ** 2).sum()
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "n": n, "beta": beta, "ndof": int(fem_sol.ndof),
        "n_elements": int(fem_sol.metadata.get("n_elements", 0)),
        "characteristic_length_m": ell,
        "w0_fem_m": w0_fem, "w0_hetenyi_m": w0_hetenyi,
        "centre_error_pct": centre_error * 100.0,
        "equilibrium_error_pct": equilibrium_error * 100.0,
        "r_squared_profile": r_squared, "elapsed_s": round(elapsed, 2),
        "x_axis": x_axis.tolist(),
        "w_fem_x": w_fem_x.tolist(),
        "w_hetenyi_x": w_hetenyi_x.tolist(),
    }


def make_figures(results):
    figs = {}
    plt = pyplot()
    finest = results[-1]
    x = finest["x_axis"]
    w_fem = finest["w_fem_x"]
    w_ref = finest["w_hetenyi_x"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, [w * 1000 for w in w_fem], 'b-o', markersize=3, label='FEM (Mindlin Q4 + Winkler)')
    ax.plot(x, [w * 1000 for w in w_ref], 'r--', linewidth=1.5, label='Hetényi (kei)')
    ax.set_xlabel('Radial distance r (m)')
    ax.set_ylabel('Deflection w (mm)')
    ax.set_title(f'Winkler Plate: FEM vs Hetényi (n={finest["n"]}, β={finest["beta"]})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, min(x[-1], 10))
    fig.tight_layout()
    figs["winkler_deflection_profile"] = fig

    fig, ax = plt.subplots(figsize=(7, 5))
    ns = [r["n"] for r in results]
    errors = [r["centre_error_pct"] for r in results]
    ax.semilogy(ns, errors, 'b-o', markersize=6, linewidth=1.5)
    ax.axhline(y=5.0, color='r', linestyle='--', alpha=0.7, label='5% acceptance')
    ax.set_xlabel('Elements per side (n)')
    ax.set_ylabel('Centre deflection error (%)')
    ax.set_title('Convergence: FEM vs Hetényi centre deflection')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    if len(results) >= 2:
        orders = []
        for i in range(1, len(results)):
            e1, e2 = results[i - 1]["centre_error_pct"], results[i]["centre_error_pct"]
            n1, n2 = results[i - 1]["n"], results[i]["n"]
            if e1 > 0 and e2 > 0 and n2 > n1:
                orders.append(math.log(e1 / e2) / math.log(n2 / n1))
        if orders:
            ax.text(0.05, 0.95, f'Observed order: {orders[-1]:.2f}',
                    transform=ax.transAxes, va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    fig.tight_layout()
    figs["winkler_convergence"] = fig
    return figs


def main():
    print("=" * 70)
    print("GAP-FEM-WINKLER: Winkler Plate on Elastic Foundation vs Hetényi")
    print("=" * 70)
    rigidity = plate_rigidity(YOUNG, POISSON, THICKNESS)
    ell = winkler_characteristic_length(rigidity, WINKLER)
    w0_ref = hetenyi_center_deflection(LOAD, rigidity, WINKLER)
    print(f"Ice: E={YOUNG/1e9:.1f} GPa, ν={POISSON}, h={THICKNESS:.2f} m")
    print(f"Foundation: k={WINKLER:.1f} N/m³, l={ell:.3f} m")
    print(f"Load: P={LOAD/1e3:.1f} kN, Hetényi w(0)={w0_ref*1000:.4f} mm")
    print()

    results = []
    for n, beta in LEVELS:
        print(f"  Running n={n}, β={beta} ...", end="", flush=True)
        r = run_level(n, beta)
        results.append(r)
        print(f"  w0_FEM={r['w0_fem_m']*1000:.4f} mm, error={r['centre_error_pct']:.2f}%, "
              f"eq={r['equilibrium_error_pct']:.3f}%, R²={r['r_squared_profile']:.6f}, "
              f"t={r['elapsed_s']:.1f}s")

    finest = results[-1]
    gates = [
        Gate(id="C1_centre_error", quantity="Centre deflection relative error",
             reference="Hetényi w(0)=P/(8√(Dk))", value=finest["centre_error_pct"],
             threshold=5.0, operator="<=", note=f"n={finest['n']}"),
        Gate(id="C2_equilibrium", quantity="Foundation reaction residual",
             reference="ΣF_reaction=ΣF_applied", value=finest["equilibrium_error_pct"],
             threshold=1.0, operator="<=", note="Global force equilibrium"),
        Gate(id="C3_profile_R2", quantity="Deflection profile R² vs kei",
             reference="Hetényi kei(r/l)", value=finest["r_squared_profile"],
             threshold=0.99, operator=">=", note="Along x-axis, excludes r=0"),
    ]

    figs = make_figures(results)

    result_data = {
        "benchmark": "fem_winkler_plate",
        "mission": "GAP-FEM-WINKLER / H3-ICE-PLATE-FEM",
        "generated_at": utc_now(),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "machine": platform.machine(), "dtype": "torch.float64"},
        "physics": {
            "young_modulus_Pa": YOUNG, "poisson_ratio": POISSON,
            "thickness_m": THICKNESS, "winkler_coefficient_N_m3": WINKLER,
            "point_load_N": LOAD, "flexural_rigidity_Nm": rigidity,
            "characteristic_length_m": ell,
            "hetenyi_center_deflection_mm": w0_ref * 1000,
        },
        "reference": {
            "source": "Hetényi (1946), Beams on Elastic Foundation",
            "formula": "w(r) = -P/(2π k l²) · kei(r/l),  l = (D/k)^(1/4)",
            "center_deflection": "w(0) = P / (8 √(D k))",
        },
        "levels": [
            {"n": r["n"], "beta": r["beta"], "ndof": r["ndof"],
             "n_elements": r["n_elements"],
             "w0_fem_mm": round(r["w0_fem_m"] * 1000, 6),
             "w0_hetenyi_mm": round(r["w0_hetenyi_m"] * 1000, 6),
             "centre_error_pct": round(r["centre_error_pct"], 4),
             "equilibrium_error_pct": round(r["equilibrium_error_pct"], 4),
             "r_squared_profile": round(r["r_squared_profile"], 6),
             "elapsed_s": r["elapsed_s"]}
            for r in results
        ],
    }
    write_json(HERE / "result.json", result_data)

    gv = gate_manifest(
        anchor="fem_winkler_plate",
        title="Winkler plate on elastic foundation vs Hetényi analytical solution",
        criteria={"centre_error": "< 5%", "equilibrium_residual": "< 1%", "profile_R2": "> 0.99"},
        gates=gates,
        artifacts={"result_json": "result.json",
                   "deflection_profile_png": "winkler_deflection_profile.png",
                   "convergence_png": "winkler_convergence.png"},
        limitations=[
            "Quarter-symmetry model; asymmetric loads not covered",
            "Domain truncation at 20 characteristic lengths",
            "Mindlin Q4 has shear locking in thin-plate limit (h/l < 0.01)",
            "Winkler model neglects shear-layer coupling",
        ],
        mission="GAP-FEM-WINKLER",
    )
    write_json(HERE / "gates_verified.json", gv)

    plt = pyplot()
    for name, fig in figs.items():
        png_path = HERE / f"{name}.png"
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        print(f"  Saved {png_path.name}")

    print()
    print("=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    all_pass = all(g.passed for g in gates)
    for g in gates:
        status = "✅ PASS" if g.passed else "❌ FAIL"
        print(f"  {status}  {g.id}: {g.quantity} = {g.value:.4f} ({g.operator} {g.threshold})")
    print()
    if all_pass:
        print("ALL GATES PASSED — Winkler plate FEM verified against Hetényi < 5%")
    else:
        print("SOME GATES FAILED — review details above")
    print("=" * 70)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
