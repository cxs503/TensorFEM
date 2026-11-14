"""Two-way elastic restraint: independent reference + peak-stress口径 guards.

Anchor ``GAP-PP-4-ICE-STRUCT-ELASTIC-RESTRAINT`` (JOUR-ICE-STRUCT §4.2).

The anchor's endpoints (``beta = 0`` SSSS, ``beta = 1`` clamped) are covered by
``tests/test_plate_restraint_panel.py``.  This file covers what those tests
cannot see: the *interior* of the two-way band (``beta_x > 0`` **and**
``beta_y > 0``, i.e. longitudinals *and* frames restraining the panel), where
until now the only evidence was FEM compared against FEM, plus the choice of
口径 used to report the peak bending stress of that band.

References (independent of the Q4 Mindlin solver):

* :func:`tensorfem.plate_levy_restraint.levy_restrained_plate` -- exact
  Kirchhoff Levy solution (two sprung edges, two simple edges);
* :func:`tensorfem.plate_levy_restraint.ritz_restrained_plate` -- double-sine
  Rayleigh-Ritz with four independent edge springs, now with an exact modal
  patch integral.

See ``docs/CASE-FEM-ELASTIC-RESTRAINT-TWO-WAY-PEAK.md``.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorfem.plate_elastic_restraint import (
    beta_to_zeta,
    edge_moment_von_mises_factor,
    edge_spring_stress,
    moments_at_nodes,
    peak_bending_stress,
    solve_restrained_panel,
    von_mises_bending_at_nodes,
)
from tensorfem.plate_levy_restraint import (
    flexural_rigidity,
    levy_restrained_plate,
    ritz_restrained_plate,
)

# verification plate: thin enough that the Mindlin shear contribution is <0.1 %
A = B = 1.0
T = 0.01
E = 200e9
NU = 0.3
Q = 1.0e3


def _solve(nx, beta_x, beta_y, ny=None):
    ny = nx if ny is None else ny
    return solve_restrained_panel(
        nx, ny, A, B, E, NU, T, Q, beta_x=beta_x, beta_y=beta_y)


def _centre_w(result):
    nodes = result.nodes
    k = int(torch.argmin((nodes[:, 0] - A / 2) ** 2 + (nodes[:, 1] - B / 2) ** 2))
    return float(result.w()[k])


def _edge_mid_Mn(result, key='x0', target=None):
    """|M_n| at the mid-point of edge ``key`` (spring口径, sign-free)."""
    data = edge_spring_stress(result)[key]
    nodes = result.nodes
    along = 1 if key in ('x0', 'xa') else 0
    target = (B / 2) if target is None else target
    k = int(torch.argmin((nodes[data['idx'], along] - target).abs()))
    return abs(float(data['M_n'][k]))


def _ritz_two_way(beta, m_modes=40, n_modes=40):
    D = flexural_rigidity(E, NU, T)
    K = beta_to_zeta(beta) * D / A
    return ritz_restrained_plate(A, B, D, NU, K, K, Q, m_modes, n_modes), K


# --------------------------------------------------------------------------
# 1. The reference itself: Ritz (four springs) vs the exact Levy (two springs)
# --------------------------------------------------------------------------

@pytest.mark.parametrize('beta', [0.2, 0.5, 0.8])
def test_ritz_reference_matches_exact_levy(beta):
    """The two-way reference must be accurate where an exact solution exists.

    With ``K_y = 0`` the Ritz model and the Levy solution solve the *same*
    boundary value problem, so any disagreement is the reference's own
    discretisation error.  It is used as the yardstick for the two-way case, so
    it must be at least an order of magnitude better than the 1 % gate applied
    to the FEM below.
    """
    D = flexural_rigidity(E, NU, T)
    K = beta_to_zeta(beta) * D / A
    lev = levy_restrained_plate(A, B, D, NU, K, Q, n_modes=201)
    rit = ritz_restrained_plate(A, B, D, NU, K, 0.0, Q, 40, 40)

    # measured 2e-5 / 1e-4 / 3.4e-3 at beta = 0.2 / 0.5 / 0.8 (40x40)
    assert abs(rit.w_centre() - lev.w_centre()) / lev.w_centre() < 5e-3

    # magnitudes only: the module's spring moment and the series solution
    # carry opposite signs at the same edge (see the probe docstring).
    h = 1e-6
    Mn_rit = abs(K * (rit.w(h, B / 2) - rit.w(0.0, B / 2)) / h)
    Mn_lev = abs(lev.moments(0.0, B / 2)[0])
    # measured reference error at 40x40 is 0.12 % / 0.43 % / 1.16 % for
    # beta = 0.2 / 0.5 / 0.8, so 2 % is the honest envelope of the reference
    # itself (it is the *reference*, not the object under test).
    assert abs(Mn_rit - Mn_lev) / Mn_lev < 2e-2


def test_ritz_patch_integral_reproduces_uniform_load():
    """A patch covering the whole plate must give the uniform-load result.

    Guards the new modal patch integral (``_modal_integral``) against a wrong
    normalisation, which would silently bias every patch reference.
    """
    D = flexural_rigidity(E, NU, T)
    K = beta_to_zeta(0.5) * D / A
    full = ritz_restrained_plate(A, B, D, NU, K, K, Q, 40, 40)
    patch = ritz_restrained_plate(A, B, D, NU, K, K, Q, 40, 40,
                                  patch=(0.0, A, 0.0, B))
    assert patch.w_centre() == pytest.approx(full.w_centre(), rel=1e-12)


# --------------------------------------------------------------------------
# 2. FEM through the two-way band against that reference
# --------------------------------------------------------------------------

@pytest.mark.parametrize('beta', [0.2, 0.4, 0.5, 0.6, 0.8])
def test_two_way_panel_matches_ritz(beta):
    """Interior of the band, both edge pairs restrained -- previously unverified."""
    res = _solve(24, beta, beta)
    rit, K = _ritz_two_way(beta)

    w_err = abs(_centre_w(res) - rit.w_centre()) / rit.w_centre()
    assert w_err < 0.01, f'centre deflection off by {100 * w_err:.2f} %'

    h = 1e-6
    Mn_rit = abs(K * (rit.w(h, B / 2) - rit.w(0.0, B / 2)) / h)
    Mn_fe = _edge_mid_Mn(res)
    assert abs(Mn_fe - Mn_rit) / Mn_rit < 0.02, \
        f'edge spring moment off by {100 * (Mn_fe - Mn_rit) / Mn_rit:.2f} %'


def test_two_way_response_is_monotone_in_beta():
    """Increasing either beta must not increase the centre deflection.

    This is a variational inequality (a stiffer restraint can only raise the
    energy of any admissible field), so it holds for the exact solution
    irrespective of the mesh -- an invariant, not just an execution check.
    """
    betas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    w = [_centre_w(_solve(16, b, b)) for b in betas]
    for lo, hi in zip(w, w[1:]):
        assert hi < lo

    # one-way increase of the *other* pair must also lower the deflection
    base = _centre_w(_solve(16, 0.4, 0.0))
    both = _centre_w(_solve(16, 0.4, 0.4))
    assert both < base


# --------------------------------------------------------------------------
# 3. Peak-stress口径: the extrapolated moment is not the edge moment
# --------------------------------------------------------------------------

def test_edge_von_mises_factor_ss2_kinematics():
    """SS2 edge: M_t = nu M_n, M_nt = 0 -> sigma_vm = 6|M_n|/t^2 * sqrt(1-nu+nu^2)."""
    assert edge_moment_von_mises_factor(0.3) == pytest.approx(math.sqrt(0.79), rel=1e-15)
    res = _solve(16, 0.8, 0.8)
    stress = edge_spring_stress(res)
    for data in stress.values():
        live = data['M_n'].abs() > 0.0        # corner nodes carry zero rotation
        ratio = data['von_mises'][live] / (6.0 * data['M_n'].abs()[live] / T**2)
        assert float(ratio.max()) == pytest.approx(edge_moment_von_mises_factor(NU),
                                                   rel=1e-6)


def test_extrapolated_edge_moment_underpredicts_known_defect():
    """Documented defect bound: `moments_at_nodes` on a restrained edge.

    The Gauss-to-node extrapolation resolves the boundary layer of the edge
    moment only to first order.  At the engineering mesh used for the panel
    band (16 elements across the span) it recovers less than *half* of the
    spring moment.  This test pins the defect so a future "improvement" that
    silently changes the口径 is caught, and so that ``peak_bending_stress``
    keeps exposing the model-consistent value.
    """
    res = _solve(16, 0.5, 0.0, ny=14)
    M = moments_at_nodes(res)
    nodes = res.nodes
    edge = torch.nonzero(nodes[:, 0] == 0.0).flatten()
    k = int(torch.argmin((nodes[edge, 1] - B / 2).abs()))
    node = int(edge[k])
    Mn_spring = _edge_mid_Mn(res, 'x0')
    ratio = abs(float(M[node, 0])) / Mn_spring
    assert ratio < 0.5, f'extrapolated/spring edge moment ratio {ratio:.3f}'

    # ... and the same node set inflates at the corners of a *simple* support
    # (beta = 0): the model edge moment is identically zero there, the
    # extrapolation is not.
    ssss = _solve(16, 0.0, 0.0)
    vm = von_mises_bending_at_nodes(ssss)
    nodes0 = ssss.nodes
    corner = int(torch.argmin(nodes0[:, 0] ** 2 + nodes0[:, 1] ** 2))
    centre = int(torch.argmin((nodes0[:, 0] - A / 2) ** 2 + (nodes0[:, 1] - B / 2) ** 2))
    assert float(vm[corner]) > float(vm[centre])


def test_peak_stress_consistent_is_never_below_either口径():
    """``peak_von_mises_consistent`` bounds both candidate definitions."""
    for beta in (0.0, 0.5, 0.8, 0.9, 1.0):
        res = _solve(16, beta, beta)
        p = peak_bending_stress(res)
        spring_max = max(
            (v['max_von_mises'] for v in p['edge_spring'].values()), default=0.0)
        assert p['peak_von_mises_consistent'] >= spring_max
        assert p['peak_von_mises_consistent'] >= 0.0
        if beta >= 0.9:
            # once the fixity is high the restrained edge carries the peak
            assert p['peak_source'].startswith('edge:')
        else:
            assert p['peak_source'] == 'interior'

    # clamped limit: the extrapolated global maximum under-reports the true
    # peak badly (measured -22 % at 16x16), the consistent value does not.
    res = _solve(16, 1.0, 1.0)
    p = peak_bending_stress(res)
    assert p['peak_von_mises_consistent'] > 1.15 * p['max_von_mises']


def test_patch_load_integration_is_exact():
    """The applied load must equal ``q * patch_area`` on every mesh.

    Regression guard for the load-integration defect found while verifying this
    anchor: a 2x2 Gauss rule on the *whole* element samples a patch narrower
    than the element only through its Gauss points, so the total load (and
    therefore every result) oscillated with the mesh by up to 40 % on the
    manuscript panel.  ``rect_patch_pressure`` now carries its bounds and the
    load is integrated on the element/patch intersection, which is exact.
    """
    from tensorfem.plate_elastic_restraint import (
        _patch_consistent_load_vector, rect_patch_pressure, structured_rect_mesh)

    L, S, p0 = 0.8, 0.7, 1.0e6
    c = 0.1 * L
    press = rect_patch_pressure(p0, 0.5 * (L - c), 0.5 * (L + c), 0.0, S)
    for nx in (8, 12, 16, 24):
        ny = max(1, int(round(nx * S / L)))
        nodes, els = structured_rect_mesh(nx, ny, L, S)
        fe = _patch_consistent_load_vector(nodes, els, press.patch_bounds,
                                           press.intensity)
        assert float(fe.sum()) == pytest.approx(p0 * c * S, rel=1e-12)


def test_spring_edge_moment_converges_on_patch_panel():
    """Converged peak of the manuscript panel band (patch load, beta = 0.8).

    With the load integrated exactly the two edge-moment口径 separate cleanly:
    the spring value is stable to a few percent, the extrapolated one is not.
    """
    from tensorfem.plate_elastic_restraint import rect_patch_pressure

    L, S, TT, EE, NN, p0 = 0.8, 0.7, 0.022, 206e9, 0.3, 1.0e6
    c = 0.1 * L
    press = rect_patch_pressure(p0, 0.5 * (L - c), 0.5 * (L + c), 0.0, S)
    spring, extrap = [], []
    for nx in (16, 32):
        ny = max(1, int(round(nx * S / L)))
        res = solve_restrained_panel(nx, ny, L, S, EE, NN, TT, press,
                                     beta_x=0.8, beta_y=0.8)
        p = peak_bending_stress(res)
        spring.append(max(v['max_von_mises'] for v in p['edge_spring'].values()))
        M = moments_at_nodes(res)
        nodes = res.nodes
        edge = torch.nonzero(nodes[:, 1] == 0.0).flatten()
        k = int(torch.argmin((nodes[edge, 0] - L / 2).abs()))
        extrap.append(abs(float(M[int(edge[k]), 1])))
    assert abs(spring[1] - spring[0]) / spring[1] < 0.05
    assert extrap[1] / spring[1] < 0.9
