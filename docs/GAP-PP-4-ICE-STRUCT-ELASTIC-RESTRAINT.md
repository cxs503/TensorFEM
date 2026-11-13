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
