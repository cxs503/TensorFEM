# Elastic Rotational Restraint Benchmark: Ice Structure Plate

## Problem Setup

| Parameter | Value | Unit |
|-----------|-------|------|
| Elastic modulus E | 9 | GPa |
| Poisson's ratio ν | 0.33 | - |
| Plate thickness t | 0.5 | m |
| Panel size a × a | 2.0 × 2.0 | m |
| Winkler foundation k | 10055 | N/m³ |
| Uniform pressure q | 50 | kPa |
| Mesh resolution | 12 × 12 | elements |

Flexural rigidity D = E·t³/(12(1-ν²)) = 1.05e+08 N·m

## Results

| β | ζ | K_rot (N·m/rad) | w_center (mm) | w·D/(q·a⁴) | Fraction from SSSS |
|---|---|-----------------|---------------|------------|-------------------|
| 0.00 | 0.00 | 0.00e+00 | 0.0414 | 5.443320e-03 | 0.00% |
| 0.05 | 0.05 | 2.77e+06 | 0.0411 | 5.401583e-03 | 1.54% |
| 0.10 | 0.11 | 5.84e+06 | 0.0407 | 5.356688e-03 | 3.19% |
| 0.15 | 0.18 | 9.28e+06 | 0.0404 | 5.308260e-03 | 4.97% |
| 0.20 | 0.25 | 1.32e+07 | 0.0400 | 5.255867e-03 | 6.89% |
| 0.30 | 0.43 | 2.25e+07 | 0.0391 | 5.137061e-03 | 11.26% |
| 0.40 | 0.67 | 3.51e+07 | 0.0380 | 4.994988e-03 | 16.49% |
| 0.50 | 1.00 | 5.26e+07 | 0.0367 | 4.822067e-03 | 22.85% |
| 0.60 | 1.50 | 7.89e+07 | 0.0350 | 4.607027e-03 | 30.76% |
| 0.70 | 2.33 | 1.23e+08 | 0.0329 | 4.332341e-03 | 40.86% |
| 0.80 | 4.00 | 2.10e+08 | 0.0302 | 3.969177e-03 | 54.22% |
| 0.90 | 9.00 | 4.73e+08 | 0.0264 | 3.466512e-03 | 72.70% |
| 0.95 | 19.00 | 9.99e+08 | 0.0238 | 3.134993e-03 | 84.90% |
| 0.99 | 98.99 | 5.21e+09 | 0.0214 | 2.814159e-03 | 96.70% |
| 1.00 | 1000000.00 | 5.26e+13 | 0.0207 | 2.724355e-03 | 100.00% |

## Reference Values

- **SSSS** (β=0): w_center = 0.0414 mm
- **CCCC** (β=1): w_center = 0.0207 mm
- **Deflection range**: 50.0% of SSSS value

## Validation

- β=0 matches SSSS: ✓ (machine precision)
- β=1 matches CCCC: ✓ (within 0.01%)
- Monotonic decrease: ✓

## Method

The elastic rotational restraint is implemented by adding diagonal spring stiffness
to the normal rotation DOFs at edge nodes:

- Edges x=0, x=a: spring on θ_x (DOF 3i+1)
- Edges y=0, y=b: spring on θ_y (DOF 3i+2)

The tangential rotations are fixed (SS2 convention), matching the existing
simply supported implementation.

Spring stiffness: K_rot = ζ·D/a, where ζ = β/(1-β+ε) with ε=10⁻⁶.

## Reference

Timoshenko, S. & Woinowsky-Krieger, S. (1959). *Theory of Plates and Shells*,
2nd ed., McGraw-Hill. §33: "Rectangular plates with elastically restrained edges".
