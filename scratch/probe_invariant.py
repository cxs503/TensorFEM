"""Probe 2: after tributary lumping, check (a) the beta response curve is
mesh-independent and (b) the mechanical invariant M_n = c_r * theta_n holds at
a restrained edge (statics: the plate edge moment equals the spring moment).
"""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, "src")

from tensorfem.plate_elastic_restraint import (
    beta_to_zeta, moments_at_nodes, solve_restrained_panel,
)

E, NU, T, SIDE, Q = 2.1e11, 0.3, 0.01, 1.0, 1.0e4
D = E * T**3 / (12.0 * (1.0 - NU**2))

print(f"{'beta':>5} {'w_norm':>12} {'fix frac':>9} {'M_n/(c_r th)' :>14}")
W_SS, W_CC = 0.00406, 0.00126
for beta in (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0):
    r = solve_restrained_panel(
        nx=16, ny=16, side_x=SIDE, side_y=SIDE, young=E, poisson=NU,
        thickness=T, pressure=Q, beta_x=beta, beta_y=beta)
    c = SIDE / 2
    dist = (r.nodes[:, 0] - c) ** 2 + (r.nodes[:, 1] - c) ** 2
    ic = int(torch.argmin(dist).item())
    wn = float(r.displacement[3 * ic]) * D / (Q * SIDE**4)
    frac = (W_SS - wn) / (W_SS - W_CC)

    # x = 0 edge, mid-span node: theta_n = theta_x (DOF 3i+1)
    cand = torch.nonzero(r.nodes[:, 0].abs() < 1e-12, as_tuple=False).flatten()
    dm = (r.nodes[cand, 1] - c).abs()
    ie = int(cand[int(torch.argmin(dm).item())])
    theta = float(r.displacement[3 * ie + 1])
    M = moments_at_nodes(r)
    Mn = float(M[ie, 0])                      # Mx at the x-edge = normal moment
    cr = r.K_rot_x                            # distributed coefficient
    ratio = Mn / (cr * theta) if (cr * theta) != 0.0 else float("nan")
    print(f"{beta:5.2f} {wn:12.6e} {frac:9.3f} {ratio:14.4f}  theta={theta:+.3e} Mn={Mn:+.1f}")