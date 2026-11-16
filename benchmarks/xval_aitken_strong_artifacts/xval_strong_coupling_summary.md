# XVAL round 4 — implicit IBM + Aitken in the strong added-mass regime

Anchor `XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION`; reference analytic (Lamb 1932): `m_a = rho_f pi c^2`, `f = sqrt(k/(m_s+m_a))/2pi`.

Configuration: plate half-width c=10, lattice 96x96, target coupled period 300 steps, 800 steps per case, `sub_tol=1e-06`, `max_sub_iter=40`.

## S1 — interface gain / stencil mass (frozen fluid state)

| c | S = -dF/dy beta dt^2/(2 gamma) | S/c | m_a = rho pi c^2 | ratio_crit = m_a/S | linearity residual |
|---|---|---|---|---|---|
| 10 | 43.691 | 4.369 | 314.16 | **7.19** | 5.1e-16 |
| 16 | 67.691 | 4.231 | 804.25 | **11.88** | 0.0e+00 |
| 20 | 83.691 | 4.185 | 1256.64 | **15.02** | 2.7e-16 |

## S2 — coupled sweep through `ratio_crit = 7.19`

| m_a/m_s | mode | omega | steps | mean sub-it | maxit hits | diverged | T | f error | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1.0 | aitken | 1.0 | 800 | 1.03 | 0 | no | 348.6 | -13.94% | PASS |
| 1.0 | fixed | 1.0 | 800 | 1.03 | 0 | no | 348.6 | -13.94% | PASS |
| 1.0 | fixed | 0.5 | 800 | 1.03 | 0 | no | 348.6 | -13.94% | PASS |
| 4.0 | aitken | 1.0 | 800 | 1.21 | 0 | no | 372.4 | -19.45% | **FAIL** |
| 4.0 | fixed | 1.0 | 800 | 1.20 | 0 | no | 372.4 | -19.45% | **FAIL** |
| 4.0 | fixed | 0.5 | 800 | 1.16 | 0 | no | 372.4 | -19.45% | **FAIL** |
| 8.0 | aitken | 1.0 | 800 | 2.02 | 0 | no | 385.5 | -22.18% | **FAIL** |
| 8.0 | fixed | 1.0 | 3 | 40.00 | 3 | yes | nan | +nan% | **DIVERGED** |
| 8.0 | fixed | 0.5 | 800 | 2.12 | 0 | no | 385.5 | -22.18% | **FAIL** |
| 12.0 | aitken | 1.0 | 800 | 2.60 | 0 | no | 389.7 | -23.02% | **FAIL** |
| 12.0 | fixed | 1.0 | 0 | nan | 0 | yes | nan | +nan% | **DIVERGED** |
| 12.0 | fixed | 0.5 | 800 | 3.67 | 0 | no | 389.7 | -23.02% | **FAIL** |

## S3 — Aitken relaxation factor vs analytic optimum `1/(1-g)`

| m_a/m_s | g = -S/m_s | omega* = 1/(1-g) | measured omega | error |
|---|---|---|---|---|
| 8.0 | -1.111 | 0.4736 | 0.4736 | -0.00% |
| 12.0 | -1.666 | 0.3750 | 0.3750 | -0.00% |

## Verdict

* critical added-mass ratio (measured gain crossing |g| = 1): **7.19**
* fixed-point iteration with `omega = 1` above the critical ratio: fails (maxit hits / divergence): **confirmed**
* Aitken-relaxed implicit IBM above the critical ratio: stable: **confirmed**
* max |frequency error| over converged Aitken cases: **23.02 %%** (acceptance 15 %%): **FAIL**

Runtime: 118 s.
