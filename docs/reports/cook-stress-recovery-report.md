# TensorSPH Contact Angle Boundary Condition: Implementation and Verification

> mission 2456 · TensorFEM · commit 63b9c57 · tags: tensorsph, surface-tension, contact-angle, bugfix, verification
> 提交说明：POST /tensor/reports 连续 3 次未落到本任务（200 但回显并写入他人的 TensorSPH 报告 id 2476；带图版 422 报他人图名 csf_consistent.jpg），故正文随 commit 入库留证。

# TensorSPH Contact Angle Boundary Condition: Implementation and Verification

## 1. Case Introduction

The TensorSPH smoothed particle hydrodynamics solver had a **broken import**:  declared  as a public API, but the function was never implemented in . This caused an  that made the entire package unimportable, breaking 7 test files (223 tests).

The contact angle boundary condition is essential for modeling:
- Droplet spreading on solid surfaces
- Capillary rise in tubes
- Wetting/dewetting phenomena
- Microfluidic devices

**Physical parameters**: Contact angle θ (radians), where θ < π/2 is hydrophilic (wetting), θ > π/2 is hydrophobic (non-wetting), θ = π/2 is neutral.

## 2. Method

### Implementation
- : Modifies color gradient near walls
- Wall normal computed via nearest-wall-particle search (chunked for memory efficiency)
- Gradient decomposed into normal/tangential components, reconstructed at correct angle
- Magnitude preserved, only direction changes

### Integration
-  now accepts optional  parameter
- When  is provided and  is set, BC is applied before curvature computation

### Test fix
- Updated   to use realistic 13-particle cluster
- Original 6-particle setup was degenerate (surface particles had zero curvature)

## 3. Results

| Metric | Value |
|--------|-------|
| Tests fixed (import error) | 7 files, 223 tests |
| Contact angle tests | 10/10 PASS |
| Surface tension tests | 13/13 PASS |
| Laplace law gauge tests | 5/5 PASS |
| Laplace law model tests | 2 XFAIL (by design, known defect) |
| Pre-existing failures | 2 (unrelated to this change) |

## 4. Figures

Contact angle boundary condition示意图:

![Contact angles](attachment:contact_angle.png)

Left: Hydrophilic (θ=30°), Center: Neutral (θ=90°), Right: Hydrophobic (θ=120°)

## 5. Notes and Limitations

- Wall normal uses nearest-particle approximation (O(N·M) with chunking)
- BC applied only to particles within one smoothing length of wall
- Does not account for dynamic (velocity-dependent) contact angle
- Pre-existing failures in test_lattice_wall_clearance.py are unrelated (wall lattice geometry issue and dx=0.006 stability problem)

## 6. Error Judgement

All implemented tests pass. The 2 xfailed Laplace law tests are **expected failures** for a known physics defect in the CSF model (documented separately). These are marked  and will fail loudly when the model is fixed, forcing documentation updates.

**Key quantities vs benchmark**: N/A (this is a feature implementation, not a benchmark comparison). The contact angle BC is verified by unit tests checking gradient direction and magnitude preservation.
