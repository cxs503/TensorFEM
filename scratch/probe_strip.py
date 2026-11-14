"""Probe 3: elastically restrained *strip* (cylindrical bending) vs closed form.

A long plate (b >> a) with rotational springs on the edges x = 0, a and simply
supported ends y = 0, b bends cylindrically in the interior.  A unit-width beam
with uniform load q, flexural rigidity D and identical end rotational springs
k = c_r = zeta*D/a has the exact solution

    M_end = (q a^2 / 12) / (1 + 2/zeta)                       (hogging, |.|)
    w_mid = (q a^4 / D) * (5/384 - 1/(96 (1 + 2/zeta)))

which spans simply supported (zeta -> 0: M_end = 0, w = 5 q a^4/(384 D)) and
clamped (zeta -> inf: M_end = q a^2/12, w = q a^4/(384 D)).  This gives an
*analytical* reference for intermediate beta -- the two limits alone (SSSS,
CCCC) cannot detect a wrong beta-to-stiffness mapping.
"""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, "src")

from tensorfem.plate_elastic_restraint import (
    beta_to_zeta, moments_at_nodes, solve_restrained_panel,
)

E, NU, T, Q = 2.1e11, 0.3, 0.01, 1.0e4
A, B = 1.0, 6.0
D = E * T**3 / (12.0 * (1.0 - NU**2))

nx, ny = 16, 96
print(f"strip a={A} b={B} mesh {nx}x{ny}  D={D:.4f}")
print(f"{'beta':>5} {'zeta':>7} {'M_exact':>9} {'M_fem':>9} {'err%':>7} "
      f"{'w_exact':>10} {'w_fem':>10} {'err%':>7}")
for beta in (0.2, 0.5, 0.8):
    r = solve_restrained_panel(
        nx=nx, ny=ny, side_x=A, side_y=B, young=E, poisson=NU, thickness=T,
        pressure=Q, beta_x=beta, beta_y=0.0)
    zeta = beta_to_zeta(beta)
    M_ex = (Q * A**2 / 12.0) / (1.0 + 2.0 / zeta)
    w_ex = (Q * A**4 / D) * (5.0 / 384.0 - 1.0 / (96.0 * (1.0 + 2.0 / zeta)))

    # node nearest (A/2, B/2)
    dist = (r.nodes[:, 0] - A / 2) ** 2 + (r.nodes[:, 1] - B / 2) ** 2
    ic = int(torch.argmin(dist).item())
    w = float(r.displacement[3 * ic])
    M = moments_at_nodes(r)
    # mid-length node on x = 0 edge: normal moment Mx there
    cand = torch.nonzero(r.nodes[:, 0].abs() < 1e-12, as_tuple=False).flatten()
    ie = int(cand[int(torch.argmin((r.nodes[cand, 1] - B / 2).abs()).item())])
    Mn = abs(float(M[ie, 0]))
    print(f"{beta:5.2f} {zeta:7.3f} {M_ex:9.3f} {Mn:9.3f} "
          f"{100*(Mn/M_ex-1):+7.2f} {w_ex:10.6e} {w:10.6e} {100*(w/w_ex-1):+7.2f}")