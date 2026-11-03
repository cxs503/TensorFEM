# Free-end Hetényi reference solution: closing the documented gap

## 1. What was missing

`docs/hetanyi-reference-audit.md` records finding **F3**: `finite_beam_series`
silently returned the simply supported solution for `boundary='free'` and now
fails closed, leaving "free-end finite beam on a Winkler foundation (Hetényi
end-conditioning forces)" as an **open gap**.  Every free- or clamped-ended
beam-on-elastic-foundation result in TensorFEM was therefore judged FE-vs-FE,
which is not a verification.

## 2. Method

The ODE `EI w'''' + k_b w = q` is written in the dimensionless state
`z(xi) = [w, w'/lam, w''/lam^2, w'''/lam^3]`, `xi = lam x`, for which the
homogeneous problem is the constant-coefficient system
`dz/dxi = [[0,1,0,0],[0,0,1,0],[0,0,0,1],[-4,0,0,0]] z`.  Its exponential is
elementary in `u1..u4 = cosh*cos, cosh*sin, sinh*cos, sinh*sin`
(`u1''=-2u4`, `u2''=2u3`, `u3''=-2u2`, `u4''=2u1`), i.e. Hetényi's
end-conditioning coefficient table in matrix form.  Two boundary conditions are
imposed at each end (`free`: `w''=w'''=0`, `simply supported`: `w=w''=0`,
`clamped`: `w=w'=0`, `guided`: `w'=w'''=0`) and the resulting linear system is
solved for the nodal states.

Because `exp(A xi)` contains `exp(+lam x)`, a single-ended propagation is
exponentially ill-conditioned once `lam L` exceeds a few units (the first
implementation lost ~1e6 in relative accuracy at `lam L = 20`).  The shipped
solver therefore uses **multiple shooting**: the span is subdivided so that no
propagation is longer than `max_span_xi/lam` (default 1), which makes the
condition number independent of `lam L` (17.7
for the case below against ~1e17 for single-ended propagation).

## 3. Case (ship structure)

Double-bottom longitudinal girder idealised as a plate strip on a Winkler
foundation: `E = 2.06e+11 Pa`, `nu = 0.3`, `t = 14.0 mm`,
`k = 2e+07 N/m^3`, strip width `0.7 m`, central load `P = 1000000 N`.

| quantity | value |
|---|---|
| `EI` | 3.623e+04 N m^2 |
| `k_b` | 1.4e+07 N/m^2 |
| `lam` | 3.13498 1/m |
| `1/lam` | 0.318981 m |
| `L` | 0.637962 m |
| `lam L` | 2.0000 |

Maximum deflection / moment by end condition:

| ends | `w_max` [m] | `max abs M` [N m] |
|---|---|---|
| free | 1.319538e-01 | 7.345486e+04 |
| clamped | 3.322385e-02 | 7.345486e+04 |
| simply supported | 9.093365e-02 | 1.081087e+05 |
| guided | 1.215518e-01 | 5.186570e+04 |
| infinite beam (asymptote) | 1.119637e-01 | -- |

## 4. Figures

![deflection_end_conditions.png](docs/assets/hetanyi_free_end/deflection_end_conditions.png)
![actions_free_end.png](docs/assets/hetanyi_free_end/actions_free_end.png)
![fe_convergence.png](docs/assets/hetanyi_free_end/fe_convergence.png)

## 5. Independent checks

Simply supported BVP vs the pre-existing Fourier-sine series (point load at
midspan, `n_terms = 800`): max relative difference
**5.27e-10**.

`lam L = 60`, evaluation between `10/lam` and `50/lam`: every end condition must
reproduce the infinite-beam Green's function.

| end condition | max relative difference vs infinite beam |
|---|---|
| free | 9.298e-10 |
| clamped | 3.477e-09 |
| simply_supported | 5.481e-10 |
| guided | 5.482e-10 |

Global-equilibrium invariant `integral k_b w dx + V(0) - V(L) = P`
(`V = -EI w'''`), 8001-point trapezoidal quadrature:

| ends | foundation share of `P` | identity residual |
|---|---|---|
| free | 0.999999997 | 3.05e-09 |
| clamped | 0.147551048 | 1.11e-16 |
| simply_supported | 0.501662963 | 6.16e-09 |
| guided | 1.000000000 | 0.00e+00 |

Rigid-beam limit (`lam L = 0.02`, `w -> P/(k_b L)`): max relative deviation
**3.00e-09**.

## 6. FE verification (the actual debt)

Euler--Bernoulli FE with the consistent Winkler matrix, refined uniformly,
compared with the exact oracle at `w(L/2)`:

| elements | free-free rel. error | clamped-clamped rel. error |
|---|---|---|
| 4 | 3.012e-04 | 3.693e-05 |
| 8 | 1.878e-05 | 2.359e-06 |
| 16 | 1.173e-06 | 1.481e-07 |
| 32 | 7.334e-08 | 9.274e-09 |
| 64 | 4.739e-09 | 6.053e-10 |

## 7. Error verdict

| quantity | baseline | relative error | < 3 % ? |
|---|---|---|---|
| simply supported `w(x)` | Fourier series (independent expansion) | 5.27e-10 | yes |
| far-field `w(x)`, 4 end conditions | infinite-beam Green's function | max 3.48e-09 | yes |
| global equilibrium | exact ODE invariant | max 6.16e-09 | yes |
| rigid limit | `P/(k_b L)` | 3.00e-09 | yes |
| FE free-free `w(L/2)`, 64 elements | exact BVP | 4.74e-09 | yes |
| FE clamped-clamped `w(L/2)`, 64 elements | exact BVP | 6.05e-10 | yes |

Worst FE error over all meshes: **3.01e-04**.

## 8. Limitations

* Linear Winkler foundation (no tensionless contact, no subgrade shear);
  Euler--Bernoulli kinematics only, no shear deformation or axial coupling.
* Point loads must lie strictly inside the span; concentrated moments and
  moving loads are not implemented.
* Distributed loads are piecewise constant; a linear (triangular) intensity is
  not supported.
* This closes the *reference* gap.  It does not qualify any ship structural
  capability: the plate-strip reduction ignores two-way plate action, and the
  foundation modulus is an input assumption, not a validated soil/structural
  model.
