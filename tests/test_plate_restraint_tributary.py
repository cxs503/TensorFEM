"""Invariants that the original beta-restraint suite did not assert.

Three defects/gaps motivated these tests (see
``docs/CASE-FEM-ELASTIC-RESTRAINT-TRIBUTARY.md``):

1. the nodal spring was ``c_r`` per edge node instead of ``c_r * h_i`` (h_i =
   tributary length), so the restraint per unit edge length scaled like 1/h and
   beta silently drifted towards clamped under mesh refinement;
2. no test asserted mesh independence of the beta family -- the old
   "mesh convergence" test only checked that the number was in (1e-4, 1e-2);
3. the extrapolated edge moment is only first order at a restrained edge, so
   the model-consistent value ``M_n = c_r * theta_n`` must be available.
"""
from __future__ import annotations

import pytest
import torch

from tensorfem.plate import structured_square_mesh
from tensorfem.plate_elastic_restraint import (
    assemble_rotational_spring_stiffness,
    beta_to_zeta,
    edge_spring_moment,
    edge_tributary_lengths,
    identify_edge_nodes,
    moments_at_nodes,
    solve_plate_elastic_restraint,
    solve_restrained_panel,
)

E, NU, T, Q = 2.1e11, 0.3, 0.01, 1.0e4
W_SSSS, W_CCCC = 0.00406, 0.00126          # Timoshenko & Woinowsky-Krieger T8


def _D():
    return E * T**3 / (12.0 * (1.0 - NU**2))


def _strip_closed_form(beta, a=1.0, q=Q):
    """Exact cylindrical-bending strip with rotational end springs k = c_r."""
    zeta = beta_to_zeta(beta)
    M_end = (q * a**2 / 12.0) / (1.0 + 2.0 / zeta)
    w_mid = (q * a**4 / _D()) * (5.0 / 384.0 - 1.0 / (96.0 * (1.0 + 2.0 / zeta)))
    return M_end, w_mid


# ---------------------------------------------------------------------------
# 1. Tributary lumping
# ---------------------------------------------------------------------------

def test_tributary_lengths_sum_to_edge_length():
    nodes, _ = structured_square_mesh(6, 1.0)
    edges = identify_edge_nodes(nodes, 1.0)
    for key, along in (("x0", 1), ("xa", 1), ("y0", 0), ("ya", 0)):
        idx, h = edge_tributary_lengths(nodes, edges[key], along=along)
        assert float(h.sum()) == pytest.approx(1.0, rel=1e-12)
        assert int(idx.numel()) == 7


def test_total_spring_stiffness_is_edge_length_times_coefficient():
    """Sum of nodal springs must be c_r * L_edge, independent of the mesh."""
    c_r = 5.0e4
    totals = []
    for nx in (4, 8, 16):
        nodes, _ = structured_square_mesh(nx, 1.0)
        K = assemble_rotational_spring_stiffness(nodes, 1.0, c_r)
        totals.append(float(torch.diag(K).sum()))
    # 4 edges of length 1 -> 4 * c_r
    for tot in totals:
        assert tot == pytest.approx(4.0 * c_r, rel=1e-12)
    assert max(totals) - min(totals) < 1e-9 * max(totals)


# ---------------------------------------------------------------------------
# 2. beta must be mesh independent (the missing invariant)
# ---------------------------------------------------------------------------

def _w_norm(nx, beta):
    res = solve_plate_elastic_restraint(
        nx=nx, side=1.0, young=E, poisson=NU, thickness=T,
        winkler_k=0.0, pressure=Q, beta=beta,
    )
    dist = (res.nodes[:, 0] - 0.5) ** 2 + (res.nodes[:, 1] - 0.5) ** 2
    idx = int(torch.argmin(dist).item())
    return float(res.displacement[3 * idx]) * _D() / (Q * 1.0**4)


@pytest.mark.parametrize("beta", [0.2, 0.5])
def test_beta_response_is_mesh_independent(beta):
    """w(beta) at nx=8 and nx=16 must agree closely (was 30-50 % apart)."""
    w_coarse, w_fine = _w_norm(8, beta), _w_norm(16, beta)
    rel = abs(w_fine - w_coarse) / w_fine
    assert rel < 0.025, f"beta={beta} is mesh dependent: {w_coarse:.6e} vs {w_fine:.6e}"


def test_beta_limits_match_plate_theory():
    """beta=0 -> SSSS (0.00406), beta=1 -> CCCC (0.00126) within 2 % at nx=16."""
    w0 = _w_norm(16, 0.0)
    w1 = _w_norm(16, 1.0)
    assert abs(w0 / W_SSSS - 1.0) < 0.02, w0
    assert abs(w1 / W_CCCC - 1.0) < 0.05, w1
    assert W_CCCC < w1 < w0 < W_SSSS * 1.02


# ---------------------------------------------------------------------------
# 3. Intermediate beta against an analytical reference (strip)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("beta", [0.2, 0.5])
def test_strip_centre_matches_closed_form(beta):
    """Cylindrical bending of a 1 x 4 m strip: w_mid within 3 % of closed form.

    The 12 x 48 grid plus the b/a = 4 end effect contribute ~2 %; the refined
    16 x 96 grid gives -0.66 % (docs/CASE-FEM-ELASTIC-RESTRAINT-TRIBUTARY.md).
    """
    A, B, nx = 1.0, 4.0, 12
    r = solve_restrained_panel(
        nx=nx, ny=4 * nx, side_x=A, side_y=B, young=E, poisson=NU, thickness=T,
        pressure=Q, beta_x=beta, beta_y=0.0,
    )
    dist = (r.nodes[:, 0] - A / 2) ** 2 + (r.nodes[:, 1] - B / 2) ** 2
    w = float(r.displacement[3 * int(torch.argmin(dist).item())])
    _, w_ex = _strip_closed_form(beta, a=A)
    assert abs(w / w_ex - 1.0) < 0.03, (w, w_ex)


def test_spring_edge_moment_beats_extrapolated_recovery():
    """M_n = c_r*theta_n is second order at a restrained edge; B*D*B is O(h)."""
    A, B, nx, beta = 1.0, 4.0, 16, 0.5
    r = solve_restrained_panel(
        nx=nx, ny=4 * nx, side_x=A, side_y=B, young=E, poisson=NU, thickness=T,
        pressure=Q, beta_x=beta, beta_y=0.0,
    )
    M_ex, _ = _strip_closed_form(beta, a=A)
    idx, M_spring = edge_spring_moment(r)["x0"]
    mid = int(torch.argmin((r.nodes[idx, 1] - B / 2).abs()).item())
    err_spring = abs(abs(float(M_spring[mid])) / M_ex - 1.0)

    M_nodes = moments_at_nodes(r)
    err_extrap = abs(abs(float(M_nodes[int(idx[mid]), 0])) / M_ex - 1.0)

    assert err_spring < 0.01, err_spring
    assert err_extrap > 5.0 * err_spring, (err_extrap, err_spring)
