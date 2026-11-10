# TensorFEM Fatigue Crack Growth Verification Face

## Campaign Node: p135w2-n03

**Physical Domain**: Fracture Mechanics · Fatigue  
**Risk Level**: Medium  
**Layer**: Verification Face

## Scope

This verification face establishes TensorFEM's LEFM fatigue crack growth capability for ship hull structural details, based on Paris' law:

```
da/dN = C · (ΔK)^m
```

## Implemented Capabilities

### 1. Stress Intensity Factor (SIF) Library
- **Infinite plate centre crack**: K_I = σ√(πa) (exact)
- **Finite-width centre crack**: Feddersen secant correction, validated vs Isida table (<0.5% for 2a/W ≤ 0.6)
- **Edge crack finite-width plate**: Tada/Paris/Irwin polynomial (a/W ≤ 0.6)
- **Semi-elliptical surface crack**: Newman-Raju approximation (a/t ≤ 0.8, a/c ≤ 1.0)
- **Hole-edge crack**: Tada correlation (recovers Kirsch limit 1.12K_t = 3.36)

### 2. Paris Law Integration
- **Closed-form solution**: For constant ΔK (infinite plate)
- **Numerical RK4 integration**: In transformed variable t = ln(a) for geometric growth resolution
- **Adaptive step control**: Based on crack increment fraction

### 3. Residual Life Prediction
- Critical crack size a_c from K_I(a_c) = K_IC (bisection)
- Integration from a₀ (detectable crack) to a_c
- Geometry factor consistency throughout (no Y=1 approximation)

### 4. S-N Curve Cross-Adjudication
- IIW FAT classification (bilinear S-N curves)
- Hot-spot stress method with SCF
- Paris vs S-N life comparison within 1 order of magnitude

### 5. Mesh Convergence (GCI)
- Three-level Richardson extrapolation (Roache safety factor Fs=1.25)
- Observed order p ≈ 4 (RK4 theoretical value)

## Acceptance Criteria

| Criterion | Description | Tolerance | Status |
|-----------|-------------|-----------|--------|
| (a) | SIF analytical anchors | ≤2% error | ✅ PASS |
| (b) | Paris closed-form vs RK4 | ≤1% error | ✅ PASS |
| (c) | Negative control ΔK=0 | da/dN=0, N_f=∞ | ✅ PASS |
| (d) | Residual life monotonicity | a₀↑ → N_f↓ | ✅ PASS |
| (e) | Paris vs S-N comparison | ≤1 order of magnitude | ✅ PASS |
| (f) | GCI convergence | ≤5% | ✅ PASS |
| (g) | Paris exponent sweep m=2/3/4 | N_f monotonically decreasing | ✅ PASS |

## Key Results

### SIF Validation
- Finite-width → infinite plate limit: error 6.2×10⁻¹³
- Edge crack F=1.12: error 0
- Newman-Raju semicircular surface crack F: 0.6625 vs 0.6637 (0.17% error)
- Hole-edge Kirsch limit F: 3.364 vs 3.360 (0.13% error)
- Finite-width centre crack vs Isida table (2a/W=0.6): 0.51% error

### Paris Integration
- Closed-form vs RK4: relative error 6.96×10⁻¹² (far below 1% tolerance)
- Observed order p = 4.000 (RK4 theoretical)

### Residual Life
- Monotonicity confirmed: a₀ from 0.05mm to 1.0mm → N_f from 153,426 to 79,906 cycles

### Paris vs S-N Cross-Check
- Weld toe detail: Paris N_f = 1.18×10⁶, S-N N_f = 4.14×10⁵
- log₁₀ ratio = 0.45 (within 1 order of magnitude)

### Parameter Sweep
- m=2: N_f = 1.27×10¹⁰
- m=3: N_f = 6.02×10⁹
- m=4: N_f = 4.13×10⁹
- Monotonically decreasing (physically consistent)

## Files

### Source Code
- `src/tensorfem/fatigue_crack_propagation.py` — SIF library, Paris law, integration, residual life
- `src/tensorfem/fatigue/crack_growth_geometry.py` — Geometry-aware ΔK(a) callables, RK4 integrator, GCI
- `src/tensorfem/fatigue/crack_growth_qualification.py` — Evidence generator for criteria (a)-(g)
- `src/tensorfem/fatigue/crack_tip_quarter_point.py` — FEM quarter-point singular elements for J-integral

### Tests
- `tests/test_fatigue_crack_propagation.py` — 34 tests (SIF, Paris, residual life, S-N)
- `tests/test_fatigue_crack_growth_geometry.py` — 37 tests (geometry-aware path)
- `tests/test_fatigue_crack_tip_quarter_point.py` — 18 tests (FEM J-integral)
- `tests/test_campaign_p135w2_n03_fatigue_crack_growth.py` — 16 campaign acceptance tests

### Artifacts
- `artifacts/fatigue_crack_growth/campaign_evidence.json` — Machine-readable evidence
- `artifacts/fatigue_crack_growth/fig1_sif_anchors.png` — SIF geometry factors
- `artifacts/fatigue_crack_growth/fig2_paris_integration.png` — Paris integration verification
- `artifacts/fatigue_crack_growth/fig3_residual_life.png` — Residual life & parameter sweep
- `artifacts/fatigue_crack_growth/fig4_paris_vs_sn.png` — Paris vs S-N comparison
- `artifacts/fatigue_crack_growth/fig5_gci_convergence.png` — GCI convergence study
- `artifacts/fatigue_crack_growth/fig6_detail_comparison.png` — Three ship details comparison

### Scripts
- `scripts/gen_fatigue_crack_growth_figures.py` — Figure generation

## Limitations & Scope

### Model Assumptions
1. **LEFM**: Small-scale yielding only, no large-scale plasticity
2. **Mode I only**: Opening mode, no mixed-mode
3. **Constant amplitude**: No load spectrum cycle counting (Rainflow) or cumulative damage
4. **Fixed aspect ratio**: Surface crack a/c constant (engineering approximation)
5. **No crack closure/retardation**: No overload effects (Wheeler/Willenborg)
6. **No residual stress**: Welding residual stress field not considered
7. **Geometry factor validity**: Feddersen limited to 2a/W ≤ 0.6, Newman-Raju to a/t ≤ 0.8

### Applicable Range
- Ferritic-pearlitic ship structural steel at ambient temperature
- High-cycle fatigue (N_f > 10⁴ cycles)
- Crack sizes above detection threshold (a₀ > 0.1 mm)

### Experimental Validation
This verification face is numerically validated (analytical/literature anchors). Direct experimental validation requires specific structural detail S-N test data or crack growth test data (da/dN vs ΔK curves).

## Report

Published report ID: 11218  
Scenario key: p135w2-n03-fatigue-crack-growth  
Mission ID: 9493
