"""Fail-closed verification of the Kerr (three-parameter) foundation plate.

Closes the verification debt for ``tensorfem.kerr_foundation`` (merged m14902,
which shipped with unit-level tests only).  A simply-supported square plate on
a Kerr foundation ``q_f = k w - G_p nabla^2 w`` under a single sinusoidal load
term ``q0 sin(pi x/L) sin(pi y/L)`` has the exact closed-form centre
deflection::

    W = q0 / [ 4 D (pi/L)^4 + k + 2 G_p (pi/L)^2 ],   D = E t^3 / (12 (1-nu^2)).

The FEM side uses the production Q4 Mindlin element plus the production Kerr
foundation stiffness; the analytic side is a scalar formula.  The two share no
code.  These tests lock the full-plate benchmark (not just the element).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tensorfem.plate import (  # noqa: E402
    _shape,
    q4_mindlin_stiffness,
    structured_square_mesh,
)
from tensorfem.kerr_foundation import q4_kerr_foundation_stiffness  # noqa: E402

L = 1.0
E = 2.1e11
NU = 0.3
T = 0.01
Q0 = 1.0e4
D = E * T**3 / (12.0 * (1.0 - NU**2))


def _exact_center(k: float, gp: float) -> float:
    denom = 4.0 * D * (math.pi / L) ** 4 + k + 2.0 * gp * (math.pi / L) ** 2
    return 1.0e4 / denom


def _solve(n: int, k: float, gp: float) -> float:
    nodes, elems = structured_square_mesh(n, L, dtype=torch.float64)
    ndof = 3 * nodes.shape[0]
    K = torch.zeros((ndof, ndof), dtype=torch.float64)
    f = torch.zeros(ndof, dtype=torch.float64)
    gpq = 1.0 / 3.0**0.5
    for conn in elems:
        xy = nodes[conn]
        ke = q4_mindlin_stiffness(xy, E, NU, T)
        ke += q4_kerr_foundation_stiffness(xy, k, gp)
        fe = torch.zeros(12, dtype=torch.float64)
        for xif in (-gpq, gpq):
            for etaf in (-gpq, gpq):
                xi, eta = nodes.new_tensor(xif), nodes.new_tensor(etaf)
                N, _ = _shape(xi, eta)
                J = _[0] if False else None  # placeholder, replaced below
        # recompute properly
        for xif in (-gpq, gpq):
            for etaf in (-gpq, gpq):
                xi, eta = nodes.new_tensor(xif), nodes.new_tensor(etaf)
                N, dN_nat = _shape(xi, eta)
                J = dN_nat.T @ xy
                p = N @ xy
                q = 1.0e4 * math.sin(math.pi * p[0].item() / L) * math.sin(math.pi * p[1].item() / L)
                fe[0::3] += N * q * torch.linalg.det(J)
        dofs = torch.stack((3 * conn, 3 * conn + 1, 3 * conn + 2), dim=1).reshape(-1)
        K[dofs[:, None], dofs] += ke
        f[dofs] += fe
    tol = L * 1e-12
    xedge = (nodes[:, 0].abs() < tol) | ((nodes[:, 0] - L).abs() < tol)
    yedge = (nodes[:, 1].abs() < tol) | ((nodes[:, 1] - L).abs() < tol)
    edge = xedge | yedge
    fixed = torch.cat((
        3 * torch.nonzero(edge, as_tuple=False).flatten(),
        3 * torch.nonzero(yedge, as_tuple=False).flatten() + 1,
        3 * torch.nonzero(xedge, as_tuple=False).flatten() + 2,
    )).unique()
    all_dofs = torch.arange(ndof)
    free = all_dofs[~torch.isin(all_dofs, fixed)]
    u = torch.zeros(ndof, dtype=torch.float64)
    u[free] = torch.linalg.solve(K[free[:, None], free], f[free])
    c = (n // 2) * (n + 1) + n // 2
    return float(u[3 * c])


def test_kerr_full_plate_matches_exact():
    w = _solve(16, 1.0e8, 5.0e4)
    assert abs(w / _exact_center(1.0e8, 5.0e4) - 1.0) < 0.01


def test_winkler_limit_matches_exact():
    w = _solve(16, 1.0e8, 0.0)
    assert abs(w / _exact_center(1.0e8, 0.0) - 1.0) < 0.01


def test_free_plate_matches_exact():
    w = _solve(16, 0.0, 0.0)
    assert abs(w / _exact_center(0.0, 0.0) - 1.0) < 5.0e-3


def test_shear_stiffens_plate():
    # G_p > 0 adds stiffness -> smaller deflection than the Winkler limit.
    w_kerr = _solve(8, 1.0e8, 5.0e4)
    w_wink = _solve(8, 1.0e8, 0.0)
    assert w_kerr < w_wink
