"""Truncation-law probe: indentation error vs finite domain depth D (in units of a).

Holds h/a fixed and scales the outer domain radius/depth.  The existing
certificate has D = 24, 48, 96 with delta errors -2.900, -1.639, -1.011 %,
which is an exact 1/D law with extrapolated infinite-domain value -0.383 %.
This probe tests that extrapolation at D = 144 and 192 (out-of-sample).
"""
from __future__ import annotations

import sys
import numpy as np

from tensorfem.hertz_halfspace_loadcontrol import (
    graded_depth,
    hertz_load_reference,
    solve_axisymmetric_contact_load_control,
    two_zone_axis,
)

R, E, NU = 10.0, 1.0e5, 0.3
ratio = 0.05
h_rel = float(sys.argv[1]) if len(sys.argv) > 1 else 0.015
depths = [float(x) for x in (sys.argv[2:] or ["144", "192"])]

e_star = E / (1.0 - NU**2)
a_ref = ratio * R
load = 4.0 * e_star * a_ref**3 / (3.0 * R)
ref = hertz_load_reference(load, R, E, NU)
a = ref.contact_radius
h = h_rel * a
n_inner = int(round(3.0 * a / h))

rows = []
for depth in depths:
    r_axis = two_zone_axis(3.0 * a, depth * a, n_inner, 26)
    z_axis = graded_depth(depth * a, 4.0 * a, 24, 16)
    r, st, diag = solve_axisymmetric_contact_load_control(
        load=load, radius=R, young=E, poisson=NU, r_axis=r_axis, z_axis=z_axis,
        rho_factor=1.0, load_tol=1e-10)
    d_err = 100.0 * (st.prescribed_indentation / ref.indentation - 1.0)
    a_err = 100.0 * (st.contact_radius_linear / ref.contact_radius - 1.0)
    rows.append((depth, d_err, a_err))
    print(f"D={depth:6.1f}a  n_elem={diag['n_elements']:6d}  delta_err={d_err:+.4f}%  "
          f"a_err={a_err:+.4f}%  alm={diag['alm_iterations']}", flush=True)

# 1/D extrapolation, using the pre-existing D = 24, 48, 96 measurements
base = [(24.0, -2.900), (48.0, -1.639), (96.0, -1.011)]
allrows = base + [(d, e) for d, e, _ in rows]
inv = np.array([1.0 / d for d, _ in allrows])
err = np.array([e for _, e in allrows])
A = np.stack((np.ones_like(inv), inv), axis=1)
coef, *_ = np.linalg.lstsq(A, err, rcond=None)
resid = err - A @ coef
print(f"\nfit err = {coef[0]:+.4f}% + {coef[1]:+.4f}%/D   "
      f"max|resid|={np.abs(resid).max():.4f} pp")
for (d, e), r_ in zip(allrows, resid):
    print(f"  D={d:6.1f}  measured {e:+.4f}%  model {coef[0]+coef[1]/d:+.4f}%  "
          f"resid {r_:+.4f} pp")
print(f"extrapolated infinite-domain indentation error: {coef[0]:+.4f}%")