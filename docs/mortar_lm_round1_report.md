# Mortar Lagrange Multiplier Contact Verification - Round 1 Report

## Status: HONEST-FAIL (Round 1)

## Objective
Implement and verify 2D mortar Lagrange multiplier contact for non-matching meshes against Hertz line contact analytical solution.

## Verification Criteria
1. Hertz pressure distribution vs analytical solution: error < 10%
2. Contact constraint violation (penetration): < tolerance
3. Mesh convergence order: > 0.3

## Results Summary

| Mesh Level | n_contact | nx  | Force Error | b Error | Pressure L2 | Penetration | Converged |
|------------|-----------|-----|-------------|---------|-------------|-------------|-----------|
| 1          | 8         | 16  | 32.3%       | 44.1%   | 35.5%       | -2.0e-3     | Yes (2)   |
| 2          | 16        | 32  | 34.3%       | 44.1%   | 30.7%       | -4.2e-3     | Yes (4)   |
| 3          | 24        | 48  | 34.7%       | 44.1%   | 27.7%       | -5.3e-3     | Yes (4)   |
| 4          | 32        | 64  | 35.1%       | 44.1%   | 28.7%       | -5.8e-3     | Yes (5)   |

**Reference values:**
- Contact half-width: b = 0.3162
- Peak pressure: p0 = 11.06
- Total force: F = 5.49

**Verification:**
1. ❌ Pressure error < 10%: **FAIL** (28.7%)
2. ✅ Penetration < tolerance: **PASS** (-5.8e-3)
3. ❌ Convergence rate > 0.3: **FAIL** (0.07)

## Failure Analysis

### Root Causes

1. **Nodal mortar coupling is too crude**
   - Current implementation uses simple nodal integration
   - Does not properly integrate shape function products over contact segments
   - Leads to inaccurate force transfer and pressure distribution

2. **Contact width severely underestimated**
   - FE contact width: b_FE = 0.177 (44% error)
   - Analytical: b_ref = 0.316
   - Cause: Nodal coupling cannot capture the smooth Hertz pressure distribution
   - The pressure drops to zero too quickly at the contact edge

3. **Insufficient mesh resolution in contact zone**
   - Even with grading, the mesh is too coarse to resolve the Hertz pressure profile
   - Need at least 8-10 nodes within the contact zone for reasonable accuracy
   - Current implementation has only 5-6 nodes in contact

4. **Missing mortar segment integration**
   - Proper mortar method requires clipping slave/master facets
   - Integration over mortar segments (not just nodal points)
   - This is essential for non-matching meshes

### What Works

1. ✅ Saddle-point formulation is correct
2. ✅ Active-set strategy converges
3. ✅ Lagrange multipliers enforce constraints (penetration ~ 0)
4. ✅ Force balance is maintained

### What Doesn't Work

1. ❌ Accurate pressure distribution (28% error vs 10% target)
2. ❌ Correct contact width (44% error)
3. ❌ Mesh convergence (rate = 0.07 vs 0.3 target)

## Recommendations for Round 2

### Priority 1: Implement Proper Mortar Segment Integration
- Clip slave facets against master facets
- Integrate over mortar segments using proper quadrature
- Reference: Puso (2004) "A stabilized segmental mortar method"

### Priority 2: Improve Contact Detection
- Use segment-based contact detection (not just nodal)
- Properly handle non-matching mesh interfaces
- Compute mortar D and M matrices correctly

### Priority 3: Mesh Refinement Strategy
- Adaptive mesh refinement in contact zone
- Target: 10+ nodes within contact half-width b
- Consider h-adaptivity based on pressure gradient

### Priority 4: Alternative Formulations
- Consider augmented Lagrangian instead of pure Lagrange multipliers
- Or penalty method with proper parameter selection
- Hybrid approaches (LM in contact zone, penalty outside)

## Literature References

1. **Hertz, H. (1882)**. "Über die Berührung fester elastischer Körper". Journal für die reine und angewandte Mathematik.
   - Analytical solution for line and point contact

2. **Wriggers, P. (2006)**. Computational Contact Mechanics, 2nd ed. Springer.
   - Comprehensive treatment of contact formulations
   - Chapter 5: Mortar methods

3. **Puso, M.A. (2004)**. "A stabilized segmental mortar method for contact". IJNME.
   - Segment-based mortar integration
   - Essential for non-matching meshes

4. **Hild, P. and Renard, Y. (2003)**. "A stabilized augmented Lagrangian method for frictional contact". IJNME.
   - Augmented Lagrangian approach
   - Better conditioning than pure LM

## Code Artifacts

- Module: `src/tensorfem/mortar_lagrange_multiplier_2d.py`
- Benchmark: `benchmarks/mortar_lm_hertz_verification.py`
- Results: `benchmarks/artifacts/mortar_lm_hertz/verification_results.json`

## Conclusion

Round 1 successfully implemented the basic mortar Lagrange multiplier framework and demonstrated that:
- The saddle-point formulation works
- The active-set strategy converges
- Contact constraints are enforced

However, the accuracy is insufficient for engineering use (28% pressure error vs 10% target). The root cause is the simplified nodal mortar coupling, which cannot capture the smooth Hertz pressure distribution.

**Round 2 must implement proper mortar segment integration** to achieve the target accuracy. This is a well-established technique in the contact mechanics literature and should resolve the accuracy issues.

**Honest-FAIL declaration**: The implementation does not meet the verification criteria, but the failure is well-understood and the path forward is clear. No results were falsified or overstated.
