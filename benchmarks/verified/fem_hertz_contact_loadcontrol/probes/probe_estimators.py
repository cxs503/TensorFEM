"""Probe alternative contact-radius estimators on one stored FE solution.

Runs a single load-controlled solve (a/R and h/a from argv) and then applies a
family of post-processing estimators to the *computed* surface pressure field.
Purpose: decide which estimator is O(h)-robust before it is promoted to the
anchor certificate.
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
ratio = float(sys.argv[1]) if len(sys.argv) > 1 else 0.10
h_rel = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0075
depth = float(sys.argv[3]) if len(sys.argv) > 3 else 96.0

e_star = E / (1.0 - NU**2)
a_ref = ratio * R
load = 4.0 * e_star * a_ref**3 / (3.0 * R)
ref = hertz_load_reference(load, R, E, NU)
a = ref.contact_radius
h = h_rel * a
n_inner = int(round(3.0 * a / h))
r_axis = two_zone_axis(3.0 * a, depth * a, n_inner, 26)
z_axis = graded_depth(depth * a, 4.0 * a, 24, 16)
ref, state, diag = solve_axisymmetric_contact_load_control(
    load=load, radius=R, young=E, poisson=NU, r_axis=r_axis, z_axis=z_axis,
    rho_factor=1.0, load_tol=1e-10)

radii = state.surface_radii
p_l = state.surface_pressure
p_c = state.surface_pressure_consistent
active = state.active_nodes
idx = np.nonzero(active)[0]
rr, pp = radii[idx], p_l[idx]
a_true = ref.contact_radius
print(f"a/R={ratio:g} h/a={h_rel:g} depth={depth:g}  a_ref={a_true:.6f} "
      f"n_active={idx.size} h={h:.5f} a/h={a_true/h:.1f}")


def rel(a_est):
    return 100.0 * (a_est / a_true - 1.0)


def show(name, a_est):
    print(f"  {name:<42} a={a_est:.6f}  err={rel(a_est):+8.4f}%")


# E1 linear zero crossing of the last two active nodes (current primary)
if idx.size >= 2 and pp[-1] < pp[-2]:
    a1 = rr[-1] + (rr[-1] - rr[-2]) * pp[-1] / (pp[-2] - pp[-1])
else:
    a1 = rr[-1]
show("E1 linear zero crossing (current primary)", a1)
show("E1b outermost active node", rr[-1])

# E2 global p^2 = c0 + c1 r^2 on active lumped pressure (current 'fit')
design = np.stack((np.ones_like(rr), rr * rr), axis=1)
coeff, *_ = np.linalg.lstsq(design, pp * pp, rcond=None)
a2 = np.sqrt(-coeff[0] / coeff[1]) if (coeff[1] < 0 and coeff[0] > 0) else np.nan
show("E2 global p^2 fit (lumped)", a2)

# E3 weighted p^2 fit, weights ~ p^2 (emphasise the patch interior)
w = np.clip(pp / pp.max(), 0.0, None) ** 2
Aw = design * w[:, None]
bw = (pp * pp) * w
cw, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
a3 = np.sqrt(-cw[0] / cw[1]) if (cw[1] < 0 and cw[0] > 0) else np.nan
show("E3 weighted (p^2) global fit", a3)

# E4 edge-local p^2 fit on the last k active nodes
for k in (3, 4, 6, 8, 12):
    if idx.size <= k:
        continue
    rk, pk = rr[-k:], pp[-k:]
    Dk = np.stack((np.ones_like(rk), rk * rk), axis=1)
    ck, *_ = np.linalg.lstsq(Dk, pk * pk, rcond=None)
    ak = np.sqrt(-ck[0] / ck[1]) if (ck[1] < 0 and ck[0] > 0) else np.nan
    show(f"E4 edge-local p^2 fit last {k} nodes", ak)

# E5 p^2 fit on the consistent pressure restricted to the lumped patch
mask = radii <= rr[-1]
rc, pc = radii[mask], p_c[mask]
Dc = np.stack((np.ones_like(rc), rc * rc), axis=1)
cc, *_ = np.linalg.lstsq(Dc, pc * pc, rcond=None)
a5 = np.sqrt(-cc[0] / cc[1]) if (cc[1] < 0 and cc[0] > 0) else np.nan
show("E5 consistent field p^2 fit (patch only)", a5)

# E6 nonlinear fit of p = p0 sqrt(1 - (r/a)^2) on the active lumped pressure
from scipy.optimize import least_squares


def resid(par, r, p):
    p0, aa = par
    inside = r < aa
    model = np.where(inside, p0 * np.sqrt(np.clip(1.0 - (r / aa) ** 2, 0.0, None)), 0.0)
    return model - p


sol = least_squares(resid, x0=np.array([pp.max(), rr[-1] * 1.02]), args=(rr, pp),
                    bounds=([0.0, rr[-1] * 0.8], [np.inf, radii[-1]]))
show("E6 nonlinear Hertz profile fit (lumped)", sol.x[1])
print(f"     E6 p0={sol.x[0]:.4f} vs analytic {ref.peak_pressure:.4f} "
      f"({100*(sol.x[0]/ref.peak_pressure-1):+.3f}%)")

sol2 = least_squares(resid, x0=np.array([pc.max(), rr[-1] * 1.02]), args=(rc, pc),
                     bounds=([0.0, rr[-1] * 0.8], [np.inf, radii[-1]]))
show("E7 nonlinear Hertz profile fit (consistent)", sol2.x[1])

# shape self-consistency of the lumped field on the active patch
res2 = pp * pp - design @ coeff
print(f"  E2 shape residual (rel RMS): {np.linalg.norm(res2)/np.linalg.norm(pp*pp):.3e}")
print(f"  contact force check: sum(lam)/P = {state.reaction/load:.9f}  "
      f"penetration/delta = {(state.maximum_penetration/state.prescribed_indentation):.2e}")