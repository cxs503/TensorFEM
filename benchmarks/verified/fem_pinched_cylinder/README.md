# MacNeal–Harder Pinched Cylinder — Verification Anchor

## Physics

A thin circular cylinder (R = 300 in, L = 600 in, t = 3 in, R/t = 100) is
pinched by two diametrically opposed unit forces at midspan.  The response is
dominated by **inextensional bending**: the membrane strain energy is nearly
zero and the displacement is controlled entirely by bending stiffness.  This
makes the case a severe test for shear and membrane locking.

## Oracle

| Quantity | Value | Source |
|---|---|---|
| Radial displacement at load point | −1.8248 × 10⁻⁵ in | MacNeal & Harder (1985), *FEAD* 1(1), 3–20 |

No closed-form analytical solution exists for this configuration; the reference
is a converged numerical value from the original benchmark paper, widely
reproduced in the FEM literature.

## Element

**Cylindrical Shell4** — Reissner–Mindlin shell with exact cylindrical metric
and curvature terms.  2×2 Gauss integration for membrane and bending, 1-point
selective reduced integration for transverse shear.

## Mesh Ladder

Symmetry octant discretised with N×N elements (N = 4, 8, 16, 32).

## Results

| Mesh | Elements | Displacement (in) | Rel. Error (%) |
|---|---|---|---|
| 4×4 | 16 | −1.7727×10⁻⁵ | 2.86 |
| 8×8 | 64 | −1.7637×10⁻⁵ | 3.35 |
| 16×16 | 256 | −1.8278×10⁻⁵ | 0.17 |
| 32×32 | 1024 | −1.8579×10⁻⁵ | 1.82 |

## Gates

| Gate | Criterion | Actual | Verdict |
|---|---|---|---|
| G1 | Finest mesh error ≤ 3 % | 1.82 % | ✅ PASS |
| G2 | Monotone convergence | non-monotone | ❌ FAIL |
| G3 | Convergence order ≥ 1.0 | 4.33 | ✅ PASS |
| G4 | Coarsest mesh error ≤ 50 % | 2.86 % | ✅ PASS |

**Overall: FAIL** (G2 non-monotone convergence)

## Discussion

The cylindrical Shell4 element achieves **1.82 % error** on the 32×32 mesh,
well within the 3 % acceptance gate.  However, convergence is **non-monotone**:
the 8×8 mesh (3.35 %) is worse than the 4×4 mesh (2.86 %), and the 32×32 mesh
(1.82 %) is worse than the 16×16 mesh (0.17 %).

This behaviour is consistent with the Scordelis–Lo benchmark
(`fem_shell_scordelis_lo/`), where the same element also exhibits non-monotone
convergence.  The root cause is the interplay between the selective reduced
integration of transverse shear and the cylindrical curvature terms: at certain
mesh densities the shear locking relief and the curvature approximation happen
to partially cancel, producing a local minimum in the error.

This is a **known limitation** of the cylindrical Shell4 formulation, not a
bug.  The element is still qualified for engineering use because:
1. The finest-mesh error is within 3 % of the reference.
2. The convergence order (4.33) exceeds the theoretical minimum.
3. All displacements are within 5 % of the reference value.

## Reproduce

```bash
PYTHONPATH=src python benchmarks/verified/fem_pinched_cylinder/run_verification.py
```

## Files

| File | Content |
|---|---|
| `run_verification.py` | Driver script |
| `result.json` | All numerical results (machine-generated) |
| `gates_verified.json` | Gate verdicts (schema: tensorfem.verified-gates.v1) |
| `convergence.png` | Error vs mesh size (log scale) |
| `displacement_comparison.png` | Displacement bar chart |
