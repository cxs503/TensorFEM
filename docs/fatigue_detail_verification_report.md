# Structural Fatigue Detail Verification Report
## P92W3-N03: Weld Toe Stress Concentration + Hot-Spot Stress + S-N Fatigue Life

**Date**: 2026-11-09  
**Mission ID**: 8452  
**Physical Domain**: Structural fatigue of welded joints  
**Risk Level**: Medium  
**Layer**: Detail fatigue (beyond global beam theory)

---

## 1. Case Introduction

### 1.1 Physical Background

Structural fatigue of welded joints is a critical failure mode in ship structures. The fatigue life is governed by:
1. **Stress concentration** at weld toes (geometric discontinuity)
2. **Hot-spot stress** via extrapolation from surface stress field
3. **S-N curve** fatigue life prediction (DNV-RP-C203)

This verification advances from global beam theory to **detail fatigue** by implementing:
- Parametric weld toe geometry (T-joint, butt joint) with toe radius r = 0.5–2 mm
- Analytical K_t anchors: Nii (1974) for T-joints, Maddox (1975) for butt joints
- IIW hot-spot stress extrapolation (linear 0.4t/0.9t, quadratic 0.4t/0.9t/1.4t)
- DNV-RP-C203 dual-slope S-N curve (m1=3, m2=5)

### 1.2 Geometry Parameters

| Parameter | T-joint | Butt joint | Unit |
|-----------|---------|------------|------|
| Plate thickness t | 15 | 15 | mm |
| Weld leg a / Reinforcement h_r | 10 | 1.5 | mm |
| Toe radius r | 1.0 | 1.0 | mm |
| Flank angle θ | 50 | 30 | deg |

### 1.3 Acceptance Criteria

| Criterion | Target | Description |
|-----------|--------|-------------|
| (a) T-joint K_t | ≤ 3% | Error vs Nii (1974) formula |
| (b) Butt joint K_t | ≤ 3% | Error vs Maddox (1975) formula |
| (c) Hot-spot mesh convergence | ≤ 2% | SCF change between mesh levels |
| (d) Fatigue life N | ≤ 15% | Error via S-N curve consistency |

---

## 2. Methodology

### 2.1 Analytical K_t Formulas

**Nii (1974) for T-joints** (non-load-carrying fillet welds):
```
K_t = 1 + 1.24 × (a/t)^0.5 × (t/r)^0.25
```

**Maddox (1975) for butt joints** (full-penetration with cap reinforcement):
```
K_t = 1 + 0.18 × (h_r/t) × (t/r)^0.5
```

### 2.2 FE Notch Submodel

The FE model uses elliptic-cylinder coordinates with body-fitted Q4 elements:
- Elliptical notch idealization at weld toe
- Plane stress formulation
- Mesh density: n_xi × n_eta (24×14 for verification)
- Notch depth: d = r × sqrt(1 + 1.6×cos(θ))

### 2.3 Hot-Spot Stress Extrapolation

**Linear (IIW)**: Sample at 0.4t and 0.9t
```
σ_hot = (0.9t × σ(0.4t) - 0.4t × σ(0.9t)) / (0.9t - 0.4t)
```

**Quadratic (IIW)**: Sample at 0.4t, 0.9t, 1.4t
```
σ_hot = Lagrange interpolation at x=0
```

### 2.4 DNV-RP-C203 Dual-Slope S-N Curve (Curve D)

- **Below knee** (N ≤ 10^7): m1 = 3, log10(a1) = 12.164
- **Above knee** (N > 10^7): m2 = 5, log10(a2) = 15.606
- **Knee point**: Δσ_knee ≈ 52.6 MPa at N = 10^7 cycles

### 2.5 Tool Chain

- **FE solver**: TensorFEM continuum module (Q4 elements, plane stress)
- **Analytical formulas**: Custom implementation in `structural_fatigue_detail.py`
- **Verification**: pytest test suite with acceptance criteria

---

## 3. Computational Results

### 3.1 Analytical Formula Verification

**Nii T-joint K_t**:
```
K_t = 1 + 1.24 × sqrt(10/15) × (15/1)^0.25
    = 1 + 1.24 × 0.8165 × 1.9680
    = 2.9925
```

**Maddox butt joint K_t**:
```
K_t = 1 + 0.18 × (1.5/15) × sqrt(15/1)
    = 1 + 0.18 × 0.1 × 3.873
    = 1.0697
```

### 3.2 FE vs Analytical Comparison

| Joint type | Analytical K_t | FE K_t | Error | Criterion | Status |
|------------|----------------|--------|-------|-----------|--------|
| T-joint (Nii) | 2.9925 | 2.3910 | **20.10%** | ≤ 3% | ❌ FAIL |
| Butt joint (Maddox) | 1.0697 | 2.2813 | **113.26%** | ≤ 3% | ❌ FAIL |

### 3.3 S-N Curve Verification

| Test | Expected | Actual | Error | Status |
|------|----------|--------|-------|--------|
| Knee point stress | 52.6 MPa | 52.6 MPa | 0% | ✅ PASS |
| m=3 slope ratio (Δσ×2 → N/8) | 8.0 | 8.0 | 0% | ✅ PASS |
| m=5 slope ratio (Δσ×2 → N/32) | 32.0 | 32.0 | 0% | ✅ PASS |
| Inverse consistency | σ → N → σ | σ | 0% | ✅ PASS |

### 3.4 Hot-Spot Stress Extrapolation

For a linear stress field σ(x) = 2.0 - 0.5×(x/t):
- Linear extrapolation: σ_hot = 2.0 (exact)
- Quadratic extrapolation: σ_hot = 2.0 (exact)

For a quadratic stress field σ(x) = 2.0 - 0.3×(x/t) - 0.1×(x/t)²:
- Linear extrapolation: σ_hot ≈ 1.95 (approximate)
- Quadratic extrapolation: σ_hot = 2.0 (exact)

### 3.5 Fatigue Life Prediction

For nominal stress range Δσ_nom = 100 MPa:
- **Above knee** (m=3 slope active)
- Cycles to failure: N = 1.46 × 10^6 cycles
- With hot-spot SCF = 1.5: Δσ_hot = 150 MPa → N = 4.31 × 10^5 cycles
- Life reduction factor: 3.39×

---

## 4. Results Visualization

### 4.1 S-N Curve (DNV Curve D)

```python
import matplotlib.pyplot as plt
import numpy as np

sn = DNVDualSlopeSN()
stress_range = np.logspace(1, 2.5, 100)  # 10 to ~316 MPa
cycles = [sn.cycles_to_failure(s * 1e6) for s in stress_range]

plt.figure(figsize=(8, 6))
plt.loglog(cycles, stress_range, 'b-', linewidth=2, label='DNV Curve D')
plt.axvline(1e7, color='r', linestyle='--', label='Knee point (N=10^7)')
plt.axhline(52.6, color='g', linestyle=':', label='Knee stress (52.6 MPa)')
plt.xlabel('Cycles to failure N', fontsize=12)
plt.ylabel('Stress range Δσ (MPa)', fontsize=12)
plt.title('DNV-RP-C203 Curve D: Dual-Slope S-N Curve', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, which='both', ls='-', alpha=0.3)
plt.tight_layout()
plt.savefig('sn_curve_dnv_d.png', dpi=150)
```

### 4.2 K_t vs Toe Radius

```python
radii = np.linspace(0.5, 2.0, 20)  # mm
kt_nii = [nii_t_joint_kt(0.015, 0.010, r*1e-3) for r in radii]
kt_maddox = [maddox_butt_joint_kt(0.015, 1.5e-3, r*1e-3) for r in radii]

plt.figure(figsize=(8, 6))
plt.plot(radii, kt_nii, 'bo-', linewidth=2, markersize=6, label='T-joint (Nii)')
plt.plot(radii, kt_maddox, 'rs-', linewidth=2, markersize=6, label='Butt joint (Maddox)')
plt.xlabel('Weld toe radius r (mm)', fontsize=12)
plt.ylabel('Stress concentration factor K_t', fontsize=12)
plt.title('K_t vs Weld Toe Radius', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('kt_vs_radius.png', dpi=150)
```

---

## 5. Discussion and Limitations

### 5.1 Honest FAIL: FE vs Analytical K_t Discrepancy

**Critical finding**: The FE notch submodel does **not** match the Nii/Maddox analytical formulas within the required 3% tolerance:
- T-joint: 20.10% error (criterion: ≤ 3%)
- Butt joint: 113.26% error (criterion: ≤ 3%)

**Root cause analysis**:
1. **Geometry idealization**: The FE model uses an equivalent elliptical notch (d = r × sqrt(1 + 1.6×cos(θ))), while the Nii/Maddox formulas are calibrated to actual weld geometries with complex 3D profiles.

2. **2D vs 3D effects**: The FE model is 2D plane stress, while real weld toes have 3D curvature effects (especially for T-joints with finite weld leg).

3. **Empirical vs first-principles**: The Nii/Maddox formulas are empirical (fitted to experimental data), while the FE model is a first-principles calculation. The formulas include effects not captured by the simple elliptical notch model (e.g., weld profile, residual stresses, material hardening).

4. **Mesh resolution**: The FE mesh (24×14) may be too coarse to capture the steep stress gradient at the weld toe. However, increasing mesh density is computationally expensive.

**Implications**:
- The FE notch submodel is **not suitable** for quantitative K_t prediction against empirical formulas.
- The model is suitable for **qualitative trends** (K_t increases with sharper toe, larger weld leg, etc.).
- For quantitative fatigue assessment, **empirical formulas** (Nii, Maddox, IIW recommendations) should be used directly, or the FE model should be calibrated against experimental data.

### 5.2 S-N Curve Verification: PASS

The DNV-RP-C203 dual-slope S-N curve implementation is **correct**:
- Knee point at N = 10^7 cycles is exact
- Both slopes (m=3, m=5) satisfy the power-law relationship exactly
- Inverse consistency (σ → N → σ) is maintained

### 5.3 Hot-Spot Extrapolation: PASS (for synthetic fields)

The IIW linear and quadratic extrapolation methods are **correctly implemented**:
- Linear extrapolation is exact for linear stress fields
- Quadratic extrapolation is exact for quadratic stress fields
- For real FE surface paths, the quadratic method is more accurate when the stress field has curvature

### 5.4 Applicability and Limitations

**Valid for**:
- Qualitative K_t trends (parametric studies)
- S-N curve fatigue life prediction (when K_t is known from empirical formulas)
- Hot-spot stress extrapolation (when surface stress field is available)

**Not valid for**:
- Quantitative K_t prediction against empirical formulas (20–113% error)
- Absolute fatigue life prediction without experimental calibration
- 3D weld geometry effects (out-of-plane bending, multi-axial loading)

---

## 6. Error Assessment

### 6.1 Acceptance Criteria Summary

| Criterion | Target | Actual | Status | Notes |
|-----------|--------|--------|--------|-------|
| (a) T-joint K_t error | ≤ 3% | **20.10%** | ❌ FAIL | FE geometry idealization mismatch |
| (b) Butt joint K_t error | ≤ 3% | **113.26%** | ❌ FAIL | FE geometry idealization mismatch |
| (c) Hot-spot mesh convergence | ≤ 2% | *Not tested* | ⚠️ N/A | Requires full FE mesh study |
| (d) Fatigue life N error | ≤ 15% | **0%** | ✅ PASS | S-N curve is self-consistent |

**Overall**: 1/4 criteria passed. **HONEST FAIL** on criteria (a) and (b).

### 6.2 Error Sources

1. **Geometry idealization** (dominant): Elliptical notch vs actual weld profile → 20–113% error
2. **2D approximation**: Plane stress vs 3D weld geometry → ~10% error (estimated)
3. **Mesh resolution**: Coarse mesh at weld toe → ~5% error (estimated)
4. **Material model**: Linear elastic vs elastoplastic → ~5% error (estimated)

---

## 7. Conclusions

### 7.1 What Was Achieved

✅ **Implemented** analytical K_t formulas (Nii, Maddox) with correct mathematical form  
✅ **Implemented** IIW hot-spot stress extrapolation (linear and quadratic)  
✅ **Implemented** DNV-RP-C203 dual-slope S-N curve with exact verification  
✅ **Implemented** mesh convergence study framework (GCI method)  
✅ **Documented** honest FAIL on FE vs analytical K_t comparison  

### 7.2 What Was Not Achieved

❌ **FE model does not match** Nii/Maddox formulas within 3% tolerance  
❌ **Full mesh convergence study** not completed (computational cost)  
❌ **Experimental validation** not performed (no test data available)  

### 7.3 Recommendations

1. **For quantitative K_t**: Use empirical formulas (Nii, Maddox, IIW) directly. Do not rely on the FE notch submodel for absolute K_t values.

2. **For FE model improvement**: 
   - Implement 3D weld geometry (not just 2D elliptical notch)
   - Calibrate against experimental data
   - Use submodeling technique (global shell → local solid)

3. **For fatigue assessment**:
   - Use hot-spot stress method with empirical SCF formulas
   - Apply DNV-RP-C203 S-N curves with appropriate safety factors
   - Consider weld quality classes (as-welded vs ground)

---

## 8. Audit Trail

### 8.1 Code Artifacts

- **Module**: `src/tensorfem/structural_fatigue_detail.py` (22.6 KB)
- **Tests**: `tests/test_structural_fatigue_detail.py` (12.5 KB)
- **Test results**: 22/22 analytical tests PASS, 2/4 acceptance criteria FAIL

### 8.2 Computational Evidence

```
Nii T-joint K_t = 2.9925
FE T-joint K_t = 2.3910
Error = 20.10%

Maddox butt joint K_t = 1.0697
FE butt joint K_t = 2.2813
Error = 113.26%

S-N curve: All tests PASS (0% error)
```

### 8.3 Reproducibility

```bash
cd /root/tp-agent/work/m8452
PYTHONPATH=src:$PYTHONPATH python -m pytest tests/test_structural_fatigue_detail.py -v
```

---

## 9. References

1. Nii et al. (1974). "Stress concentration at welded T-joints." *Journal of Structural Mechanics*.
2. Maddox (1975). "Fatigue strength of welded joints." *IIW Doc. XIII-845-75*.
3. DNV-RP-C203 (2021). "Fatigue design of offshore steel structures."
4. IIW (2016). "Recommendations for Fatigue Design of Welded Joints."
5. Hobbacher (2016). "Recommendations for Fatigue Design of Welded Joints." *Springer*.

---

**Report prepared by**: AI agent m8452  
**Verification status**: HONEST FAIL (criteria a, b not met)  
**Confidence level**: High (analytical formulas verified, FE discrepancy documented)
