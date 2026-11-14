"""Probe: is the beta parameterisation mesh-independent?

Hypothesis (adversarial review of the merged implementation): the nodal
rotational spring is K_rot = zeta*D/a applied to *every* edge node without
lumping over the tributary edge length h.  The physical (continuous) edge
restraint coefficient is c_r = zeta*D/a [N*m/m/rad]; a consistent nodal spring
must be K_rot_i = c_r*h.  If the h factor is missing, the restraint per unit
edge length is c_r/h, i.e. it grows like 1/h under refinement and beta stops
being a meaningful "embedding coefficient".

Test: hold beta fixed, refine the mesh, watch the non-dimensional centre
deflection w*D/(q*a^4) (Kirchhoff limits: SSSS 0.00406, CCCC 0.00126).
"""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, "src")

from tensorfem.plate_elastic_restraint import solve_plate_elastic_restraint

SIDE = 1.0
E, NU, T = 2.1e11, 0.3, 0.01
Q = 1.0e4
D = E * T**3 / (12.0 * (1.0 - NU**2))


def w_norm(nx, beta):
    res = solve_plate_elastic_restraint(
        nx=nx, side=SIDE, young=E, poisson=NU, thickness=T,
        winkler_k=0.0, pressure=Q, beta=beta,
    )
    c = SIDE / 2
    dist = (res.nodes[:, 0] - c) ** 2 + (res.nodes[:, 1] - c) ** 2
    idx = int(torch.argmin(dist).item())
    w = float(res.displacement[3 * idx])
    return w * D / (Q * SIDE**4), float(res.K_rot)


print(f"{'beta':>5} {'nx':>4} {'K_rot [Nm/rad]':>16} {'w*D/(q a^4)':>14}")
for beta in (0.2, 0.5):
    for nx in (8, 12, 16, 24):
        wn, kr = w_norm(nx, beta)
        print(f"{beta:5.2f} {nx:4d} {kr:16.4e} {wn:14.6e}")
    print()