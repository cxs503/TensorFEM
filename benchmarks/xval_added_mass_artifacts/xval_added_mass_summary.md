# XVAL — implicit IBM + Aitken: added-mass cross-validation

Anchor `XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION`; reference analytic (Lamb 1932): `m_a = rho_f pi c^2`, `f = sqrt(k/(m_s+m_a))/2pi`.

Configuration: plate half-width c=10.0, lattice 96x96, drive/analytic period 300 steps, 600 steps per case.

| experiment | quantity | reference | got | error | verdict |
|---|---|---|---|---|---|
| E0 control | f (structure only) | 0.004216 | 0.004202 | -0.348% | FAIL |
| E1 operator | m_a (force fit, r2=1.000) | 314.2 | -0.0 | -100.0% | FAIL |
| E2 coupled | f, m_a/m_s=0.60, aitken | 0.003333 | 0.002959 | -11.2% | PASS |
| E2 coupled | f, m_a/m_s=0.60, fixed | 0.003333 | 0.002959 | -11.2% | PASS |
| E2 coupled | f, m_a/m_s=1.00, aitken | 0.003333 | nan | +nan% | n/a |
| E2 coupled | f, m_a/m_s=1.00, fixed | 0.003333 | nan | +nan% | n/a |

Max |frequency error| over converged cases: 11.24 % (tolerance 15 %).

Runtime: 47 s.
