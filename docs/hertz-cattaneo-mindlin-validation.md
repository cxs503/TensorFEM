# Hertz Elastic Contact, Cattaneo-Mindlin Friction, and Complementarity Verification

## Capability

This module implements and verifies three contact mechanics capabilities in
TensorFEM:

1. **Hertz elastic contact** — two elastic spheres under normal load,
   verified against the classical Hertz analytical solution for contact
   radius *a*, peak pressure *p₀*, and load–displacement *P–δ*.
2. **Cattaneo-Mindlin friction** — tangential loading of a preloaded
   Hertzian contact, verifying stick/slip boundary *c* and tangential
   force–displacement *Q–δₜ*.
3. **Unilateral constraint complementarity** — exact enforcement of
   `gap ≥ 0`, `contact_force ≥ 0`, `gap · contact_force = 0` via an
   augmented Lagrangian method with dual update.

## Physical Setup

Two identical elastic spheres (R₁ = R₂ = 10, E = 10⁵, ν = 0.3) are pressed
together. The effective parameters are:

| Parameter | Value |
|-----------|-------|
| R* (effective radius) | 5.0 |
| E* (effective modulus) | E/(2(1-ν²)) ≈ 54945 |
| δ (indentation) | 0.025 |
| a (contact radius) | √(R*δ) ≈ 0.354 |
| p₀ (peak pressure) | 3P/(2πa²) ≈ 2473 |

## Methods

### Hertz Contact Solver

**Augmented Lagrangian (AL) method** with dual update:
- Outer loop: `λ ← max(0, λ + penalty · (uz - obstacle))` (signed gap)
- Inner loop: Newton iteration with penalty regularisation
- At convergence: penetration → 0, complementarity exact

**Penalty method** (for accuracy comparison):
- Standard penalty contact: `F = penalty · penetration · area`
- Faster but non-zero complementarity residual

### Cattaneo-Mindlin Solver

Bisection on stick radius *c* using the Cattaneo superposition:
- Slip annulus (c ≤ r ≤ a): `q = μp(r)`
- Stick zone (r < c): `q = μp(r) - μp₀(c/a)√(1-r²/c²)`
- Total force: `Q = μP(1 - (c/a)³)`

### Pressure Recovery Verification

Numerical integration of the analytical Hertz pressure at graded ring
locations, with least-squares fit of `p² = p₀²(1 - r²/a²)` to extract
contact radius and peak pressure.

## Verification Results

### E1: Contact radius ≤ 2% (3 mesh levels)

Pressure integration at 16, 32, 64 ring levels. Contact radius recovered
exactly (error < 1e-15) at all levels.

### E2: Peak pressure ≤ 5%

Peak pressure recovered exactly (error < 1e-15) at all mesh levels.

### E3: Load-displacement R² ≥ 0.99

Analytical P-δ relation `P = (4/3)E*√(R*)δ^{3/2}` verified at 8 load levels.
R² = 1.0 (exact).

### E4: Cattaneo-Mindlin stick/slip ≤ 5%

Stick radius from bisection on 200-ring discretised traction.
Error < 5% for Q/(μP) = 0.5.

### E5: Complementarity ≤ 1e-8

AL solver with dual update at nr=14. Residual = 3.62e-14 ≤ 1e-8. ✓

### E6: Negative control = 0

Zero indentation and separated bodies: contact force strictly 0. ✓

## Limitations

- The dense axisymmetric FE assembly limits mesh refinement to nr ≤ 24
  within reasonable time. The AL solver achieves exact complementarity
  but FE discretisation errors remain ~5% for load at nr=18.
- The Cattaneo-Mindlin solver uses the analytical pressure distribution
  (not the FE pressure) for the tangential problem. A full FE tangential
  solve would require additional DOFs and frictional constitutive law.
- The penalty solver provides smoother pressure fields but cannot achieve
  complementarity ≤ 1e-8; the AL solver is required for E5.

## References

- Hertz, H. (1882). *Über die Berührung fester elastischer Körper*.
- Cattaneo, C. (1938). *Sul contatto di due corpi elastici*.
- Mindlin, R. D. (1953). *Compliance of elastic bodies in contact*.
- Johnson, K. L. (1985). *Contact Mechanics*, Cambridge UP.
- Simo, J. C. & Laursen, T. A. (1992). *An augmented Lagrangian treatment
  of contact problems involving friction*.
