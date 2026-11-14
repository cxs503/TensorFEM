# XVAL round 4 — implicit IBM + Aitken in the strong added-mass regime

Anchor `XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION`; reference analytic (Lamb 1932): `m_a = rho_f pi c^2`, `f = sqrt(k/(m_s+m_a))/2pi`.

Configuration: plate half-width c=10.0, lattice 96x96, target coupled period 300 steps, 400 steps per case, `sub_tol=1e-06`, `max_sub_iter=40`.

## S1 — interface gain / stencil mass (frozen fluid state)

| c | S = -dF/dy beta dt^2/(2 gamma) | S/c | m_a = rho pi c^2 | ratio_crit = m_a/S | linearity residual |
|---|---|---|---|---|---|
| 10 | 43.691 | 4.369 | 314.16 | **7.19** | 5.1e-16 |
| 16 | 67.691 | 4.231 | 804.25 | **11.88** | 0.0e+00 |
| 20 | 83.691 | 4.185 | 1256.64 | **15.02** | 2.7e-16 |

The interface force is exactly linear in the trial displacement at fixed fluid state (second difference / first difference below 1e-12), so the interface map has the *constant* gain `g = -S/m_s`.

## S2 — coupled sweep through `ratio_crit = 7.19`

| m_a/m_s | mode | omega | steps | mean sub-it | maxit hits | diverged | T | f error | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1.0 | aitken | 1.0 | 400 | 1.05 | 0 | no | nan | -26.76% | FAIL |
| 1.0 | fixed | 1.0 | 400 | 1.05 | 0 | no | nan | -26.76% | FAIL |
| 1.0 | fixed | 0.5 | 400 | 1.05 | 0 | no | nan | -26.76% | FAIL |
| 4.0 | aitken | 1.0 | 400 | 1.45 | 0 | no | nan | -41.41% | FAIL |
| 4.0 | fixed | 1.0 | 400 | 1.31 | 0 | no | nan | -41.41% | FAIL |
| 4.0 | fixed | 0.5 | 400 | 1.23 | 0 | no | nan | -41.41% | FAIL |
| 8.0 | aitken | 1.0 | 400 | 3.35 | 0 | no | nan | -41.41% | FAIL |
| 8.0 | fixed | 1.0 | 3 | 40.00 | 3 | yes | nan | +nan% | DIVERGED |
| 8.0 | fixed | 0.5 | 400 | 2.10 | 0 | no | nan | -41.41% | FAIL |
| 12.0 | aitken | 1.0 | 400 | 3.41 | 0 | no | nan | -48.73% | FAIL |
| 12.0 | fixed | 1.0 | 0 | nan | 0 | yes | nan | +nan% | DIVERGED |
| 12.0 | fixed | 0.5 | 400 | 3.35 | 0 | no | nan | -48.73% | FAIL |

## S3 — Aitken relaxation factor vs analytic optimum `1/(1-g)`

| m_a/m_s | g = -S/m_s | omega* = 1/(1-g) | measured omega | error |
|---|---|---|---|---|
| 8.0 | -1.113 | nan | 0.5250 | +nan% |
| 12.0 | -1.669 | nan | 0.4969 | +nan% |

## Verdict

* critical added-mass ratio (measured gain crossing |g| = 1): **7.19**
* fixed-point iteration with `omega = 1` above the critical ratio: fails (maxit hits / divergence): **confirmed**
* Aitken-relaxed implicit IBM above the critical ratio: stable: **confirmed**
* max |frequency error| over converged cases: **48.73 %** (acceptance 15 %)

Runtime: 64 s.
