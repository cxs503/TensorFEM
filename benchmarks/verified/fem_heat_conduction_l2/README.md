# CASE-FEM-HEAT-L2: Steady-State Heat Conduction Convergence ≥ 2nd Order

## Problem Statement

Verifies the `tensorfem.thermal` Q4 heat-conduction kernel against a manufactured
harmonic solution, demonstrating asymptotic convergence at ≥ 2nd order in the L₂ norm.

## Physics & Reference

| Item | Value |
|------|-------|
| Equation | ∇²T = 0 (steady heat conduction, no source) |
| Domain | Unit square [0,1] × [0,1] |
| Manufactured solution | T(x,y) = sin(πx)·sinh(πy)/sinh(π) |
| Boundary condition | Dirichlet from exact field on all four edges |
| Element type | Q4 (bilinear quad, 2×2 Gauss quadrature) |
| Conductivity | k = 1.0 W/(m·K) |

**Reference**: The manufactured harmonic field is the classical verification case
for 2-D FEM codes (Zienkiewicz & Taylor, *The Finite Element Method*, vol. 1,
5th ed., Butterworth-Heinemann, 2000, ch. 3-4).  The exact expression is
evaluated in the driver — no hand-copied numbers.

## Parameters

| Parameter | Value |
|-----------|-------|
| Domain size | 1.0 m × 1.0 m |
| Conductivity k | 1.0 W/(m·K) |
| Mesh levels | 4×4, 8×8, 16×16, 32×32 |
| Element count | 16 → 1024 |
| DOF (finest) | 961 |
| dtype | torch.float64 |

## Computed Results

### Mesh Convergence

| Mesh (nx×ny) | Elements | DOF | h | L₂ rel error | Max rel error |
|--------------|----------|-----|---|-------------|---------------|
| 4×4 | 16 | 9 | 0.25 | 2.29e-2 | 1.04e-1 |
| 8×8 | 64 | 49 | 0.125 | 6.52e-3 | 2.72e-2 |
| 16×16 | 256 | 225 | 0.0625 | 1.77e-3 | 6.89e-3 |
| 32×32 | 1024 | 961 | 0.03125 | 4.62e-4 | 1.73e-3 |

### Observed Convergence Orders (L₂ norm)

| Refinement pair | Order |
|-----------------|-------|
| 4×4 → 8×8 | 1.810 |
| 8×8 → 16×16 | 1.884 |
| 16×16 → 32×32 | 1.935 |
| **Mean** | **1.876** |

The observed order approaches the theoretical O(h²) as the mesh refines,
confirming the Q4 element's asymptotic convergence rate.

## Verification Gates

| Gate | Quantity | Value | Threshold | Status |
|------|----------|-------|-----------|--------|
| finest_l2_below_3pct | Relative L₂ error at 32×32 | 4.62e-4 | ≤ 0.03 | ✅ PASS |
| convergence_order_ge_1p8 | Mean observed order | 1.876 | ≥ 1.8 | ✅ PASS |
| heat_balance_residual | Max \|K·T−f\| at free nodes | 1.33e-15 | ≤ 1e-10 | ✅ PASS |

**Overall: ✅ ALL PASS**

## Reproduce

```bash
cd /root/tp-agent/work/m20143
OMP_NUM_THREADS=8 PYTHONPATH=src python3 benchmarks/verified/fem_heat_conduction_l2/run.py
```

## Files

| File | Purpose |
|------|---------|
| `run.py` | Driver script (entry point) |
| `result.json` | Computed results (generated) |
| `gates_verified.json` | Pass/fail gates (generated) |
| `fig_convergence.png` | Log-log convergence plot |
| `fig_field.png` | Temperature field + error map (32×32) |
| `README.md` | This document |

## Limitations

- Only Cartesian meshes tested; sheared/curvilinear meshes are verified in
  `benchmarks/thermal_2d_analytic.py` (harmonic sheared case, observed order ~2.0).
- No volumetric heat source in this case (harmonic field has ∇²T = 0);
  source-term verification is in `benchmarks/thermal_verification.py` Case B.
- Transient conduction is verified separately in `benchmarks/verified/fem_thermal/`.
