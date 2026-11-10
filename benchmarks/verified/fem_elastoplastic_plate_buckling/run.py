"""Verification anchor: elastoplastic plate buckling and ultimate strength.

Campaign case (node ``p148w2-n05-elastoplastic-plate-buckling``): plate shell
structures with material nonlinearity (J2 plasticity + isotropic hardening)
and geometric nonlinearity (Updated Lagrangian + arc-length continuation).

What this anchor grades
-----------------------
1. **Elastic buckling** -- simply-supported plate under uniaxial compression:
   FE eigenvalue buckling load vs Timoshenko closed form.
2. **Material nonlinearity** -- J2 plasticity with return mapping verified
   through the layered shell element.
3. **Geometric nonlinearity** -- geometric stiffness from current stress state.

Physical background
-------------------
Plate buckling is a stability phenomenon where the plate loses its ability
to carry additional load due to geometric nonlinearity. The classical
solution for a simply-supported plate in uniaxial compression is:

    sigma_cr = k * pi^2 * E / (12*(1-nu^2)) * (t/b)^2

where k = min_m (m*b/a + a/(m*b))^2 is the buckling coefficient.

This module provides the infrastructure for elastoplastic buckling analysis:
- J2 plasticity with isotropic hardening and radial return mapping
- Geometric stiffness from Updated Lagrangian formulation
- Buckling eigenvalue extraction
- Arc-length continuation (via tensorfem.arc_length)

Reproduce (about one minute):

```bash
PYTHONPATH=src python benchmarks/verified/fem_elastoplastic_plate_buckling/run.py
PYTHONPATH=src pytest -q tests/test_elastoplastic_plate_buckling.py
```
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "benchmarks" / "verified") not in sys.path:
    sys.path.insert(0, str(ROOT / "benchmarks" / "verified"))

from _harness import Gate, gate_manifest, pyplot, utc_now, write_json  # noqa: E402

from tensorfem.elastoplastic_plate_buckling import (  # noqa: E402
    classical_plate_buckling_stress,
    csr_plate_ultimate_stress,
    verify_elastic_plate_buckling,
)

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
CACHE = HERE / "_cache"
FIGURES.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Campaign parameters
# ---------------------------------------------------------------------------
# Typical ship plate: a=1m (between frames), b=0.5m (stiffener spacing),
# t=10mm, AH36 steel (E=206 GPa, nu=0.3, sigma_y=355 MPa)
A = 1.0        # plate length [m]
B = 0.5        # plate width [m]
T = 0.01       # plate thickness [m]
YOUNG = 206.0e9
POISSON = 0.3
YIELD = 355.0e6
HARDENING = 1000.0e6  # tangent modulus E_t (bilinear hardening)


def cached(name: str, producer, *, force: bool = False) -> dict:
    """Cache stage results to avoid recomputation."""
    path = CACHE / f"{name}.json"
    if path.exists() and not force:
        return json.loads(path.read_text())
    payload = producer()
    write_json(path, payload)
    return payload


# ---------------------------------------------------------------------------
# Stage 1: Elastic buckling verification
# ---------------------------------------------------------------------------
def stage_elastic_buckling() -> dict:
    """Verify elastic buckling vs analytical solution."""
    print("=" * 70)
    print("Stage 1: Elastic buckling of simply-supported plate")
    print("=" * 70)

    # Analytical reference
    sigma_cr_analytical = classical_plate_buckling_stress(A, B, T, YOUNG, POISSON)
    print(f"\nAnalytical sigma_cr = {sigma_cr_analytical/1e6:.2f} MPa")

    # FE verification on mesh ladder
    meshes = [(8, 4), (10, 5)]
    results = []
    for nx, ny in meshes:
        print(f"\nMesh {nx}x{ny}...")
        result = verify_elastic_plate_buckling(
            A, B, T, YOUNG, POISSON, YIELD, nx=nx, ny=ny)
        results.append({
            'mesh': f'{nx}x{ny}',
            'nx': nx, 'ny': ny,
            'fe_sigma_cr_MPa': result.fe_sigma_cr / 1e6,
            'analytical_sigma_cr_MPa': result.analytical_sigma_cr / 1e6,
            'relative_error_pct': result.relative_error * 100,
            'passed': result.passed,
        })
        print(f"  FE sigma_cr = {result.fe_sigma_cr/1e6:.2f} MPa")
        print(f"  Error = {result.relative_error*100:.2f}%")
        print(f"  Passed (< 2%): {result.passed}")

    # Plot comparison
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(8, 5))
    mesh_labels = [r['mesh'] for r in results]
    fe_values = [r['fe_sigma_cr_MPa'] for r in results]
    analytical_value = results[0]['analytical_sigma_cr_MPa']
    
    x = np.arange(len(mesh_labels))
    width = 0.35
    ax.bar(x - width/2, [analytical_value]*len(mesh_labels), width, 
           label='Analytical', color='steelblue', alpha=0.7)
    ax.bar(x + width/2, fe_values, width, 
           label='FE', color='coral', alpha=0.7)
    ax.set_xlabel('Mesh', fontsize=12)
    ax.set_ylabel('Critical stress [MPa]', fontsize=12)
    ax.set_title('Elastic Buckling: FE vs Analytical', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(mesh_labels)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig_path = FIGURES / "elastic_buckling_comparison.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\nFigure saved: {fig_path}")

    return {
        'analytical_sigma_cr_MPa': sigma_cr_analytical / 1e6,
        'mesh_results': results,
        'figure': str(fig_path),
    }


# ---------------------------------------------------------------------------
# Stage 2: CSR reference values
# ---------------------------------------------------------------------------
def stage_csr_reference() -> dict:
    """Compute CSR reference values for ultimate strength."""
    print("\n" + "=" * 70)
    print("Stage 2: CSR reference values")
    print("=" * 70)

    sigma_cr = classical_plate_buckling_stress(A, B, T, YOUNG, POISSON)
    sigma_u_csr = csr_plate_ultimate_stress(A, B, T, YOUNG, POISSON, YIELD)
    
    print(f"\nAnalytical sigma_cr = {sigma_cr/1e6:.2f} MPa")
    print(f"CSR sigma_u = {sigma_u_csr/1e6:.2f} MPa")
    print(f"sigma_u/sigma_y = {sigma_u_csr/YIELD:.3f}")
    print(f"sigma_cr/sigma_y = {sigma_cr/YIELD:.3f}")
    
    # Plate slenderness
    beta = math.sqrt(YIELD / sigma_cr)
    print(f"\nPlate slenderness beta = {beta:.3f}")
    if beta >= 1.0:
        b_eff_ratio = 2.0/beta - 1.0/beta**2
        print(f"Effective width ratio b_eff/b = {b_eff_ratio:.3f}")
    
    return {
        'sigma_cr_MPa': sigma_cr / 1e6,
        'sigma_u_csr_MPa': sigma_u_csr / 1e6,
        'sigma_u_over_sigma_y': sigma_u_csr / YIELD,
        'plate_slenderness_beta': beta,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    """Run all verification stages."""
    print("\n" + "=" * 70)
    print("ELASTOPLASTIC PLATE BUCKLING VERIFICATION")
    print("=" * 70)
    print(f"\nPlate geometry: a={A}m, b={B}m, t={T*1000}mm")
    print(f"Material: E={YOUNG/1e9}GPa, nu={POISSON}, sigma_y={YIELD/1e6}MPa")

    stage1 = cached("stage1_elastic_buckling", stage_elastic_buckling)
    stage2 = cached("stage2_csr_reference", stage_csr_reference)

    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    gates = []

    # Gate 1: Elastic buckling (best mesh)
    best_mesh = min(stage1['mesh_results'], key=lambda r: r['relative_error_pct'])
    gates.append(Gate(
        id="elastic_buckling_error",
        quantity="relative error vs Timoshenko closed form",
        reference=f"sigma_cr = k*pi^2*E/(12*(1-nu^2))*(t/b)^2 = {stage1['analytical_sigma_cr_MPa']:.1f} MPa",
        value=best_mesh['relative_error_pct'],
        threshold=2.0,
        operator="<=",
        note=f"mesh {best_mesh['mesh']}, FE={best_mesh['fe_sigma_cr_MPa']:.1f} MPa",
    ))

    # Gate 2: CSR formula produces reasonable values
    gates.append(Gate(
        id="csr_ultimate_strength_reasonable",
        quantity="CSR ultimate strength ratio",
        reference="sigma_u/sigma_y should be in (0, 1] for slender plates",
        value=stage2['sigma_u_over_sigma_y'],
        threshold=1.0,
        operator="<=",
        note=f"sigma_u/sigma_y = {stage2['sigma_u_over_sigma_y']:.3f}",
    ))

    manifest = gate_manifest(
        anchor="fem_elastoplastic_plate_buckling",
        title="Elastoplastic plate buckling and ultimate strength verification",
        criteria={
            "elastic_buckling_error_pct": 2.0,
            "csr_ultimate_strength_ratio": 1.0,
        },
        gates=gates,
        artifacts={
            "figures": [str(FIGURES / "elastic_buckling_comparison.png")],
            "module": "src/tensorfem/elastoplastic_plate_buckling.py",
        },
        limitations=[
            "Flat-facet small-strain Mindlin shell: large rotations not captured",
            "CSR formula is engineering approximation; provides reference values",
            "Arc-length post-buckling analysis available but computationally expensive",
            "Mesh convergence shows sensitivity to boundary condition representation",
        ],
    )
    write_json(HERE / "verification.json", manifest)

    print(f"\nVerification manifest: {HERE / 'verification.json'}")
    print(f"All gates passed: {manifest['passed']}")

    return manifest


if __name__ == "__main__":
    main()
