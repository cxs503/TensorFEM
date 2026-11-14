# GAP-PP-4-ICE-STRUCT-ELASTIC-RESTRAINT: Elastic Rotational Restraint for Ice Structure Plates

## 1. Problem Statement

**Anchor**: GAP-PP-4-ICE-STRUCT-ELASTIC-RESTRAINT

**Why blocked**: The existing plate solver only supports SSSS (simply supported) and CCCC (clamped) boundary conditions — two extreme cases. Real ice-belt panels are bounded by longitudinal stiffeners and transverse frames that provide *finite* rotational stiffness. The SSSS conclusion can only be stated as an upper bound on deflection, without quantifying the actual restraint effect.

**Need**:
1. Rotational spring boundary conditions (Winkler-type rotational constraint) in the plate solver
2. Embedding coefficient beta scan from 0 (simply supported) to 1 (clamped)
3. Validation against literature values for elastically restrained plates

**Reference**: Timoshenko & Woinowsky-Krieger, *Theory of Plates and Shells*, 2nd ed., §33 (pp. 198-202).

## 2. Method & Implementation

### 2.1 Theory

For a Mindlin-Reissner plate, the edge moment–rotation relation with elastic restraint is:

```
M_n = K_rot · θ_n
```

where `M_n` is the bending moment normal to the edge and `θ_n` is the rotation about the edge normal.

The rotational spring stiffness is parameterized as:

```
K_rot = ζ · D / a
```

with `D = E·t³/(12(1-ν²))` the flexural rigidity, `a` the plate dimension, and `ζ` a dimensionless restraint parameter.

The mapping from embedding coefficient `β ∈ [0,1]` to `ζ` is:

```
ζ = β / (1 - β + ε)    with ε = 10⁻⁶
```

- `β = 0` → `ζ = 0` (simply supported, free rotation)
- `β → 1` → `ζ → 1/ε` (clamped, zero rotation)

### 2.2 Boundary Condition Convention (SS2)

Following the existing solver's "hard" simply supported (SS2) convention:

| DOF | x-edges (x=0, x=a) | y-edges (y=0, y=b) |
|-----|-------------------|-------------------|
| w (transverse) | Fixed | Fixed |
| θ_t (tangential) | θ_y fixed | θ_x fixed |
| θ_n (normal) | θ_x → spring K_rot | θ_y → spring K_rot |

### 2.3 Implementation

New module: `src/tensorfem/plate_elastic_restraint.py`

Key functions:
- `beta_to_zeta(beta)`: Map β to dimensionless restraint ζ
- `rotational_stiffness(beta, E, ν, t, a)`: Compute K_rot
- `assemble_rotational_spring_stiffness()`: Assemble global spring matrix
- `solve_plate_elastic_restraint()`: Main solver with β parameter
- `beta_sweep()`: Utility for parameter sweep

## 3. Parameters

### Ice Plate Case

| Parameter | Value | Unit |
|-----------|-------|------|
| Elastic modulus E | 9.0 | GPa |
| Poisson's ratio ν | 0.33 | - |
| Plate thickness t | 0.5 | m |
| Panel size a × a | 2.0 × 2.0 | m |
| Winkler foundation k | 10,055 | N/m³ |
| Uniform pressure q | 50 | kPa |
| Mesh resolution | 12 × 12 | elements |

Flexural rigidity: D = 4.69×10⁸ N·m

## 4. Results

### 4.1 Beta Sweep

| β | ζ | K_rot (N·m/rad) | w_center (mm) | w·D/(q·a⁴) |
|---|---|-----------------|---------------|------------|
| 0.00 | 0.00 | 0.00e+00 | 2.0789 | 3.998e-03 |
| 0.10 | 0.11 | 5.15e+07 | 1.7344 | 3.335e-03 |
| 0.20 | 0.25 | 1.17e+08 | 1.4842 | 2.854e-03 |
| 0.30 | 0.43 | 2.01e+08 | 1.2941 | 2.489e-03 |
| 0.40 | 0.67 | 3.13e+08 | 1.1446 | 2.201e-03 |
| 0.50 | 1.00 | 4.69e+08 | 1.0237 | 1.969e-03 |
| 0.60 | 1.50 | 7.03e+08 | 0.9239 | 1.777e-03 |
| 0.70 | 2.33 | 1.09e+09 | 0.8400 | 1.615e-03 |
| 0.80 | 4.00 | 1.87e+09 | 0.7683 | 1.478e-03 |
| 0.90 | 9.00 | 4.22e+09 | 0.7062 | 1.358e-03 |
| 0.95 | 19.00 | 8.91e+09 | 0.6782 | 1.304e-03 |
| 0.99 | 99.00 | 4.64e+10 | 0.6570 | 1.263e-03 |
| 1.00 | 1.00e+06 | 4.69e+14 | 0.6518 | 1.254e-03 |

### 4.2 Reference Values

| Case | w_center (mm) | w·D/(q·a⁴) |
|------|---------------|------------|
| SSSS (β=0) | 2.0789 | 3.998e-03 |
| CCCC (β=1) | 0.6518 | 1.254e-03 |
| Literature SSSS | - | 4.06e-03 |

### 4.3 Validation Checklist

| Check | Result |
|-------|--------|
| β=0 matches SSSS | ✓ (machine precision) |
| β=1 matches CCCC | ✓ (0.0001% error) |
| Monotonic decrease | ✓ |
| SSSS vs literature | ✓ (1.5% error, <5% threshold) |
| Mesh convergence | ✓ (nx=6,10,16 tested) |

## 5. Convergence & Sensitivity

### 5.1 Mesh Convergence (β=0.5)

| nx | w_center (mm) | w·D/(q·a⁴) |
|----|---------------|------------|
| 6 | 1.0312 | 1.983e-03 |
| 10 | 1.0237 | 1.969e-03 |
| 16 | 1.0215 | 1.964e-03 |

Convergence rate: ~O(h²) expected for Mindlin Q4 element.

### 5.2 Deflection Range

The deflection range from SSSS to CCCC is:
- (2.0789 - 0.6518) / 2.0789 = **68.6%** of the SSSS value

This demonstrates that the boundary condition assumption has a significant impact on the predicted deflection, and the elastic restraint capability is essential for realistic ice structure analysis.

## 6. Artifacts

- `src/tensorfem/plate_elastic_restraint.py`: Implementation module
- `tests/test_plate_elastic_restraint.py`: Validation tests
- `benchmarks/beta_sweep_elastic_restraint.py`: Benchmark script
- `benchmarks/artifacts/beta_sweep_results.json`: Full sweep data
- `benchmarks/artifacts/beta_sweep_figure.png`: Deflection vs β curve
- `benchmarks/artifacts/beta_sweep_report.md`: Detailed report

## 7. Failure Modes & Limitations

- **Current limitation**: Only square plates with uniform edge restraint are supported. Rectangular plates and non-uniform restraint (different β on different edges) would require extension.
- **Mindlin shear locking**: The Q4 SRI element is used; for very thin plates (t/a < 1/50), MITC treatment may be needed.
- **No plasticity**: The implementation is linear elastic. Ice plasticity would require nonlinear solution.

## 8. Reproduction

```bash
# Run tests
python -m pytest tests/test_plate_elastic_restraint.py -v

# Run benchmark
python benchmarks/beta_sweep_elastic_restraint.py
```

---

# 9. Increment 2026-11-14: peak-stress recovery, rectangular panels, per-edge beta

## 9.1 What was missing after the first pass

The first pass (sections 1-8) delivered the rotational spring in a *square*
solver and validated it on **deflections only**, against its own clamped solve.
Against the anchor's acceptance criteria that left three real gaps:

| Acceptance criterion | State before | Why it was not met |
|---|---|---|
| `beta=0` peak **stress** matches `panel_patch_load_results.json` (`<0.1%`) | not met | the solver returned displacement only — there was no moment/stress recovery at all, and `panel_patch_load_results.json` was nowhere in this repo |
| `beta=1` vs clamped *analytic/literature* `<5%` | not met | only compared against the module's own CCCC solve; §4.3's "SSSS vs literature 1.5%" also mixed a Winkler-foundation result with a foundation-free book coefficient |
| `>=5` beta scan as machine-readable JSON | partial | `benchmarks/artifacts/beta_sweep_results.json` holds deflections only (no stress), square plate only |

The paper case is a **0.8 m x 0.7 m rectangular panel** restrained by
longitudinals on one pair of edges and transverse frames on the other, so a
square solver with a single `beta` cannot represent it either.

## 9.2 What was added

All additive — `solve_plate_elastic_restraint` and `beta_sweep` are untouched,
and the new assembly is asserted equivalent to the historical one.

* `structured_rect_mesh(nx, ny, side_x, side_y)` — structured Q4 mesh.
* `solve_restrained_panel(...)` — rectangular Mindlin panel, **independent
  `beta_x` (longitudinals) and `beta_y` (transverse frames)**, optional Winkler
  foundation, vectorised element assembly + sparse direct solve (scipy;
  `RuntimeError` if scipy is absent, matching the library's existing optional
  sparse precedent). 68043 DOF (`nx=160`) solves in a few seconds.
* `moments_at_nodes` / `bending_stress_at_nodes` / `von_mises_bending_at_nodes`
  / `peak_bending_stress` — Gauss-point curvature, extrapolated to nodes and
  averaged, `M = Db * kappa`, `sigma = 6M/t^2`,
  `sigma_vm = 6/t^2 sqrt(Mx^2 - Mx My + My^2 + 3 Mxy^2)` — the same definition
  the paper's `panel_patch_load.py` uses.
* `rect_patch_pressure(...)` — vectorised pressure patch (no per-element Python
  loop).
* `tests/data/panel_patch_load_reference.json` — the baseline excerpt **with
  the source file's SHA-256** (`b234cdf8...`), so the regression is traceable
  back to `papers/jour-ice-struct/data/panel_patch_load_results.json`.
* `benchmarks/beta_restraint_panel_band.py` →
  `benchmarks/artifacts/beta_restraint_panel_band.json` (11 beta points,
  machine readable, git sha + versions recorded).

## 9.3 Results

`beta = 0` (SSSS), paper case (1 MPa, 8 cm x 70 cm patch), centre von Mises
bending stress vs the Navier N=401 baseline of 73.3710 MPa:

| nx x ny | sigma_vM centre [MPa] | error vs baseline |
|---|---|---|
| 20 x 18 | 70.3286 | -4.147% |
| 40 x 35 | 72.5658 | -1.097% |
| 80 x 70 | 73.1803 | -0.260% |
| 160 x 140 | **73.3233** | **-0.065%** |

Observed order p = 2.02 (40/80/160), Richardson extrapolation -0.03%. The
`<0.1%` acceptance criterion is met at `nx=160` and is a *converged* result,
not a mesh coincidence.

`beta` band (diagonal family, nx=160): 73.32 MPa (SSSS) → 47.27 MPa (clamped),
i.e. **-35.5%**. Most of the redistribution happens between beta = 0 and 0.2
(-33%); beyond that the centre stress saturates. Physically this is
cylindrical-bending-like behaviour of a full-breadth patch (a strip with simply
supported ends carries `pL^2/8`, with clamped ends `pL^2/24`; two-way action
puts the ratio below 3).

Literature re-check (`nu=0.3`, uniform load, square, nx=80):

| case | computed | Timoshenko Table 8 | error |
|---|---|---|---|
| SSSS `M/(qa^2)` | 0.047886 | 0.0479 | -0.03% |
| CCCC `w D/(qa^4)` | 0.001277 | 0.00126 | +1.35% |
| CCCC `M_centre/(qa^2)` | 0.022921 | 0.0231 | -0.78% |
| CCCC `M_edge/(qa^2)` | 0.048585 | 0.0513 | -5.3% raw, **-0.4% after Richardson** |

## 9.4 Pitfalls found (worth keeping)

1. **"Peak stress" is ambiguous.** The paper's `sigma_plate` is the **centre**
   value. A pressure patch has a load discontinuity, so the *global* maximum
   sits on the patch edge and oscillates with the mesh (misaligned meshes give
   a spurious +8.6% at nx=36). `peak_bending_stress` returns both; the
   regression only uses the centre.
2. **Patch/mesh alignment.** Patch edges must fall on element faces (nx a
   multiple of 20 here) or the patch area is discretised inconsistently and
   convergence is non-monotonic.
3. **The clamped-edge moment converges at O(h), not O(h^2)** (moment boundary
   layer): 0.0414 / 0.0461 / 0.0486 for nx=20/40/80. Taking the finest grid
   alone gives 5.3%, just outside the 5% criterion; the test extrapolates.
4. **Mindlin vs Kirchhoff model error**: at t/s = 0.031 the shear deformation
   makes the FEM systematically ~0.2% softer than the thin-plate series. This
   is a model difference, not a mesh error, and it is why the criterion is met
   at 0.065% rather than at machine precision.
5. The earlier §4.3 line "SSSS vs literature 1.5%" compared a
   Winkler-foundation SSSS result with a foundation-free coefficient; the
   foundation-free comparison is the 0.03% row in §9.3 above. Treat §4.3's
   literature line as superseded.

## 9.5 Limitations

* Linear elastic, small deflection; no plasticity, buckling or contact.
* `zeta = beta/(1-beta)` is an engineering mapping; a *physical* `beta` must be
  calibrated from the actual stiffener/frame rotational stiffness
  (`K_rot = EI/L` of the attached stiffener plus frame fixity), which is the
  next step for the manuscript.
* Only uniform structured meshes; nodal averaging is unweighted (exact on
  uniform meshes, mildly biased on graded ones).

## 9.6 Reproduction

```bash
python -m pytest tests/test_plate_restraint_panel.py -v   # 12 tests, ~45 s
python benchmarks/beta_restraint_panel_band.py --nx 160    # writes the JSON band
```
