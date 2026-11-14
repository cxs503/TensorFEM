"""Probe 4: does the recovered edge moment converge to the exact strip value?

w_mid matches the analytic beam-with-end-springs solution to <1 %, and
w_mid = 5qa^4/384D - M_end a^2/(8D) is a *function of* M_end, so the true edge
moment is within ~2 % of (q a^2/12)/(1+2/zeta).  If the recovered Mx at the
edge does not converge to that value, the moment-recovery routine
(moments_at_nodes -> von Mises peak stress) is the defect, not the solve.
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
BETA = 0.5
zeta = beta_to_zeta(BETA)
M_ex = (Q * A**2 / 12.0) / (1.0 + 2.0 / zeta)
w_ex = (Q * A**4 / D) * (5.0 / 384.0 - 1.0 / (96.0 * (1.0 + 2.0 / zeta)))
print(f"beta={BETA} zeta={zeta:.3f}  exact M_end={M_ex:.3f}  w_exact={w_ex:.6e}")

print(f"{'mesh':>10} {'w err%':>8} {'M_edge':>9} {'M err%':>8} {'M_from_w':>10} {'M_spr':>8} {'spr%':>7}")
for nx in (16, 32):
    ny = 4 * nx
    r = solve_restrained_panel(
        nx=nx, ny=ny, side_x=A, side_y=B, young=E, poisson=NU, thickness=T,
        pressure=Q, beta_x=BETA, beta_y=0.0)
    dist = (r.nodes[:, 0] - A / 2) ** 2 + (r.nodes[:, 1] - B / 2) ** 2
    ic = int(torch.argmin(dist).item())
    w = float(r.displacement[3 * ic])
    # M_end implied by the deflection through the exact beam relation (inverse)
    M_from_w = (5.0 * Q * A**4 / 384.0 - w * D) * 8.0 / A**2
    M = moments_at_nodes(r)
    cand = torch.nonzero(r.nodes[:, 0].abs() < 1e-12, as_tuple=False).flatten()
    ie = int(cand[int(torch.argmin((r.nodes[cand, 1] - B / 2).abs()).item())])
    Mn = abs(float(M[ie, 0]))
    theta = float(r.displacement[3 * ie + 1])
    M_spring = abs(r.K_rot_x * theta)
    print(f"{nx:4d}x{ny:<5d} {100*(w/w_ex-1):+8.2f} {Mn:9.3f} "
          f"{100*(Mn/M_ex-1):+8.2f} {M_from_w:10.3f} {M_spring:9.3f} "
          f"{100*(M_spring/M_ex-1):+7.2f}")