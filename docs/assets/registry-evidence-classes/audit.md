| class | checks | min error | max error | meaning |
|---|---:|---:|---:|---|
| `discretized` | 43 | 0.000e+00 | 2.307e-02 | mesh/FD/eigen/time-step solve vs closed form or literature |
| `trial_space` | 9 | 0.000e+00 | 7.550e-04 | numerical quadrature/Ritz/contour integral vs closed form |
| `closed_form` | 16 | 0.000e+00 | 8.971e-16 | closed form vs a second closed-form coding (no numerics) |
| `identity` | 1 | 1.861e-16 | 1.861e-16 | consistency identity enforced by the algorithm itself |

Total registered checks: 69; classified: 69; discretised share: 0.6232.

| thin margin check | class | error/tolerance |
|---|---|---:|
| `shell.spheroid.displacement_equator` | discretized | 0.769 |
| `continuum.cook` | discretized | 0.737 |
