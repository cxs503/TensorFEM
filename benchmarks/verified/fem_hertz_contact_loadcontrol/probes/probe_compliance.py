"""Field-level test of the FE half-space compliance (no contact solving).

Prescribe the *exact* Hertz surface tractions on the FE mesh, solve the linear
problem, and compare the central deflection with the closed-form half-space
value

    w(0) = (2 / E*) * int_0^inf p(rho) drho          (Boussinesq, axisymmetric)

which for the Hertz pressure equals delta = a^2 / R exactly.  A mesh-independent
deviation here means the assembled stiffness (not the mesh) is off, because the
same linear system is what the contact solver uses.
"""
from __future__ import annotations

import sys
import numpy as np

from tensorfem.hertz_halfspace_loadcontrol import (
    assemble_axisymmetric_stiffness,
    graded_depth,
    hertz_load_reference,
    mesh_from_axes,
    surface_ring_areas,
    two_zone_axis,
)

R, E, NU = 10.0, 1.0e5, 0.3
ratio = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
h_rel = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0075
depth = float(sys.argv[3]) if len(sys.argv) > 3 else 192.0

e_star = E / (1.0 - NU**2)
a_ref = ratio * R
load = 4.0 * e_star * a_ref**3 / (3.0 * R)
ref = hertz_load_reference(load, R, E, NU)
a = ref.contact_radius
h = h_rel * a
r_axis = two_zone_axis(3.0 * a, depth * a, int(round(3.0 * a / h)), 26)
z_axis = graded_depth(depth * a, 4.0 * a, 24, 16)
nodes, quads = mesh_from_axes(r_axis, z_axis)
nz = len(z_axis) - 1
kmat = assemble_axisymmetric_stiffness(nodes, quads, E, NU)
top = np.arange(0, len(nodes), nz + 1)
radii = nodes[top, 0]
areas = surface_ring_areas(radii)
cz = 2 * top + 1

p_surf = np.where(radii <= a, ref.peak_pressure * np.sqrt(np.clip(1.0 - (radii / a) ** 2, 0, None)), 0.0)
f = np.zeros(2 * len(nodes))
f[cz] = -p_surf * areas                      # traction pushed into the body

bottom = np.arange(nz, len(nodes), nz + 1)
outer = np.arange(len(nodes) - (nz + 1), len(nodes))
fixed = np.unique(np.concatenate([2 * bottom, 2 * bottom + 1, 2 * outer,
                                  2 * top[np.isclose(radii, 0.0)]]))
free = np.setdiff1d(np.arange(2 * len(nodes)), fixed)
from scipy.sparse.linalg import spsolve
import scipy.sparse as sp
u = np.zeros(2 * len(nodes))
ksub = sp.csr_matrix(kmat)[free][:, free]
u[free] = spsolve(ksub.tocsc(), f[free])

w_fe = -float(u[cz[0]])                       # deflection at the axis (downward)
w_exact = 2.0 / e_star * np.sum(p_surf) * h    # (2/E*) int p dr  with uniform surface spacing h
delta = ref.indentation
print(f"a/R={ratio:g} h/a={h_rel:g} D={depth:g}  elements={len(quads)}")
print(f"  total prescribed force / P_ref : {np.sum(p_surf*areas)/load:.9f}")
print(f"  w_FE(0)      = {w_fe:.10f}")
print(f"  w_halfspace  = {w_exact:.10f}   err vs FE = {100*(w_exact/w_fe-1):+.4f}%")
print(f"  delta_Hertz  = {delta:.10f}   FE err    = {100*(w_fe/delta-1):+.4f}%")
print(f"  p0 prescribed= {ref.peak_pressure:.4f}  (analytic)")
print(f"  far-field check: w_FE at r=10a = {-float(u[cz[np.argmin(abs(radii-10*a))]]):.3e}")