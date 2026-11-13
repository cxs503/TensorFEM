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
| 0.05 | 0.05 | 2.77e+06 | 0.0396 | 5.210758e-03 | 8.55% |
| 0.10 | 0.11 | 5.84e+06 | 0.0380 | 4.994987e-03 | 16.49% |
| 0.15 | 0.18 | 9.28e+06 | 0.0365 | 4.794250e-03 | 23.87% |
| 0.20 | 0.25 | 1.32e+07 | 0.0350 | 4.607026e-03 | 30.76% |
| 0.30 | 0.43 | 2.25e+07 | 0.0325 | 4.267997e-03 | 43.23% |
| 0.40 | 0.67 | 3.51e+07 | 0.0302 | 3.969175e-03 | 54.22% |
| 0.50 | 1.00 | 5.26e+07 | 0.0282 | 3.703790e-03 | 63.98% |
| 0.60 | 1.50 | 7.89e+07 | 0.0264 | 3.466508e-03 | 72.70% |
| 0.70 | 2.33 | 1.23e+08 | 0.0247 | 3.253069e-03 | 80.55% |
| 0.80 | 4.00 | 2.10e+08 | 0.0233 | 3.060026e-03 | 87.65% |
| 0.90 | 9.00 | 4.73e+08 | 0.0219 | 2.884562e-03 | 94.11% |
| 0.95 | 19.00 | 9.99e+08 | 0.0213 | 2.802678e-03 | 97.12% |
| 0.99 | 98.99 | 5.21e+09 | 0.0208 | 2.739741e-03 | 99.43% |
| 1.00 | 1000000.00 | 5.26e+13 | 0.0207 | 2.724347e-03 | 100.00% |

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
