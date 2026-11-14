# m20361: TensorFEM Anchor Certificate Artifact Repair (fem_heat_conduction_l2)

## 1. Case Introduction

**Physical Background**: Steady-state heat conduction verification on the unit square using a manufactured harmonic solution T(x,y) = sin(πx)·sinh(πy)/sinh(π). This is a classical FEM verification case (Zienkiewicz & Taylor, FEM vol.1, ch.3-4).

**Geometry & Parameters**:
- Domain: [0,1]×[0,1] unit square
- Element type: Q4 bilinear quadrilateral
- Mesh levels: 4×4, 8×8, 16×16, 32×32 elements
- Boundary conditions: Dirichlet from exact solution on all four edges
- Material: unit conductivity, density, specific heat, thickness

**Acceptance Gates** (predeclared):
| Gate | Quantity | Threshold | Operator |
|------|----------|-----------|----------|
| G1 | Finest L₂ relative error | ≤ 0.03 | ≤ |
| G2 | Observed convergence order | ≥ 1.8 | ≥ |
| G3 | Heat balance residual | ≤ 1e-10 | ≤ |

## 2. Methodology

The benchmark runner (`benchmarks/verified/fem_heat_conduction_l2/run.py`) performs:
1. Structured Q4 mesh generation at 4 refinement levels
2. Assembly of thermal conductivity matrix with 2×2 Gauss quadrature
3. Solution of steady-state system with Dirichlet BC enforcement
4. L₂ error computation against the manufactured solution
5. Convergence order calculation via log-log slope
6. Figure generation: convergence plot + field visualization

## 3. Computational Results

| Mesh Level | Elements | L₂ Relative Error | Observed Order |
|------------|----------|-------------------|----------------|
| 4×4 | 16 | 2.285e-02 | — |
| 8×8 | 64 | 6.517e-03 | 1.811 |
| 16×16 | 256 | 1.766e-03 | 1.885 |
| 32×32 | 1024 | 4.618e-04 | 1.933 |

**Mean observed order**: 1.876 (target ≥ 1.8, theoretical 2.0)

**Gate Verdicts**:
- G1 (finest L₂ error): 4.618e-04 ≤ 0.03 → ✅ PASS
- G2 (convergence order): 1.876 ≥ 1.8 → ✅ PASS
- G3 (heat balance): 1.332e-15 ≤ 1e-10 → ✅ PASS

## 4. Figures

Two figures are generated on demand:
- `fig_convergence.png`: Log-log plot of L₂ error vs element size with O(h¹) and O(h²) reference slopes
- `fig_field.png`: Side-by-side comparison of FEM temperature field, exact solution, and error distribution at 32×32 mesh

## 5. Key Observations

1. **Q4 convergence**: The observed order (1.876) is close to the theoretical O(h²) for bilinear elements in L₂ norm. The slight deficit is expected due to boundary layer effects near the Dirichlet edges.

2. **Heat balance**: The residual at machine precision (1.33e-15) confirms the discrete system satisfies global conservation exactly (up to floating-point roundoff).

3. **Artifact tracking**: The `gates_verified.json` declares `fig_convergence.png` and `fig_field.png` as artifacts. These must be committed to git to satisfy the `E_ARTIFACT_UNTRACKED` check in the anchor certificate audit.

## 6. Error Assessment

| Quantity | FE Value | Reference | Relative Error | < 3%? |
|----------|----------|-----------|----------------|-------|
| Finest L₂ error | 4.618e-04 | — | — | N/A (absolute gate) |
| Convergence order | 1.876 | 2.0 (theoretical) | 6.2% | N/A (order gate) |
| Heat balance | 1.332e-15 | 0 | — | ✅ (well within 1e-10) |

The anchor passes all three gates. The L₂ error at the finest mesh (0.046%) is well below the 3% threshold.

## 7. Fix Summary

**Problem**: The `fem_heat_conduction_l2` anchor had `E_ARTIFACT_UNTRACKED` defect — its `gates_verified.json` declared figure artifacts (`fig_convergence.png`, `fig_field.png`) that existed on disk but were not tracked by git. This caused `test_no_unregistered_certificate_defect` to fail.

**Fix**: 
1. Re-ran the benchmark to regenerate fresh figures
2. Added both figures to git tracking via `git add`

**Verification**: All 16 anchor certificate tests now pass (previously 15/16).
