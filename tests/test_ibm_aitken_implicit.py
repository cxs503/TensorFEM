"""Unit/contract tests for :mod:`tensorfem.ibm_aitken_implicit`.

Covers the IBM operator invariants (partition of unity, epsilon conservation,
uniform-force transmission), the Aitken update rule, the corotational beam
static stiffness and residual-free state, and the D2Q9 rest state.
"""

import math

import numpy as np
import pytest

from tensorfem.ibm_aitken_implicit import (
    AitkenRelaxation,
    BeamProperties,
    CorotationalBeam2D,
    IBMInterface,
    LBMFluid2D,
    dirac_delta_3pt,
    epsilon_conservation,
    find_support,
    marker_spacing,
)


def _curved_markers(n=41):
    s = np.linspace(0.0, 1.0, n)
    return np.stack((20.0 + 40.0 * s, 32.0 + 3.0 * np.sin(2.0 * np.pi * s)), axis=1)


def test_kernel_partition_of_unity():
    for x in np.linspace(0.0, 1.0, 101):
        assert abs(float(dirac_delta_3pt(x - np.arange(-3, 4)).sum()) - 1.0) < 1e-14


def test_kernel_support_and_symmetry():
    assert dirac_delta_3pt(np.array([1.6]))[0] == 0.0
    assert abs(dirac_delta_3pt(np.array([0.25]))[0]
               - dirac_delta_3pt(np.array([-0.25]))[0]) < 1e-15


def test_support_window_is_three_sites_per_direction():
    i, j, w = find_support(np.array([10.0, 10.2]), 1.0, (32, 32))
    assert len(i) == 9 and len(j) == 9
    assert abs(float(w.sum()) - 1.0) < 1e-14


def test_marker_spacing():
    pos = np.array([[0.0, 0.0], [2.0, 0.0], [5.0, 0.0]])
    assert np.allclose(marker_spacing(pos, 1.0), [2.0, 2.0, 3.0])


def test_epsilon_conservation_system():
    markers = _curved_markers()
    supports = [find_support(p, 1.0, (64, 64)) for p in markers]
    ds = marker_spacing(markers, 1.0)
    eps = epsilon_conservation(markers, supports, ds, 1.0)
    a = np.zeros((len(markers), len(markers)))
    for i, (ii, jj, ww) in enumerate(supports):
        for j, pj in enumerate(markers):
            dj = (dirac_delta_3pt(np.abs(pj[0] - ii))
                  * dirac_delta_3pt(np.abs(pj[1] - jj)))
            a[i, j] = float(np.dot(ww, dj)) * ds[j]
    assert np.max(np.abs(a @ eps - 1.0)) < 1e-10


def test_uniform_force_transmission_with_and_without_epsilon():
    markers = _curved_markers()
    raw = IBMInterface(markers, (64, 64), correct_epsilon=False)
    cor = IBMInterface(markers, (64, 64), correct_epsilon=True)
    assert raw.uniform_force_transmission_error() > 0.02
    assert cor.uniform_force_transmission_error() < 1e-8


def test_spread_conserves_total_momentum():
    markers = _curved_markers()
    ibm = IBMInterface(markers, (64, 64), correct_epsilon=True)
    rng = np.random.default_rng(3)
    f = rng.normal(size=(len(markers), 2))
    grid = ibm.spread(f).sum(axis=(0, 1))
    marker = (f * ibm.eps_used[:, None] * ibm.ds[:, None]).sum(axis=0)
    assert np.linalg.norm(grid - marker) / np.linalg.norm(marker) < 1e-12


def test_direct_forcing_kills_relative_velocity_at_marker():
    rng = np.random.default_rng(11)
    markers = _curved_markers(9)
    ibm = IBMInterface(markers, (64, 64), correct_epsilon=True)
    rho = np.ones((64, 64))
    u = rng.normal(scale=0.05, size=(64, 64, 2))
    u_body = np.zeros((9, 2))
    rho_i, mom_i = ibm.interpolate(rho, u)
    f = ibm.force(rho_i, mom_i, u_body)
    # implicit-IBM force definition: F = 2 (rho u_body - m)
    assert np.allclose(f, 2.0 * (rho_i[:, None] * u_body - mom_i))
    # spreading then interpolating gives back the (corrected) force exactly
    grid = ibm.spread(f)
    back = np.zeros((9, 2))
    for k, (ii, jj, ww) in enumerate(ibm.supports):
        back[k] = [float(np.dot(grid[ii, jj, 0], ww)), float(np.dot(grid[ii, jj, 1], ww))]
    assert np.linalg.norm(back - f) / np.linalg.norm(f) < 1e-8


def test_aitken_update_matches_irons_tuck_formula():
    ait = AitkenRelaxation(relax_max=5.0, relax0=1.0)
    r0 = np.array([1.0, 2.0])
    r1 = np.array([0.4, 1.1])
    omega = ait.omega_for(r0, r1)
    num = float(np.dot(r0, r1 - r0))
    den = float(np.dot(r1 - r0, r1 - r0))
    assert abs(omega - (-1.0 * num / den)) < 1e-14


def test_aitken_converges_where_unit_relaxation_diverges():
    """Scalar interface fixed point ``u -> g u + b`` with ``g > 1``.

    Pure Gauss-Seidel (``omega = 1``) diverges; the production
    :class:`AitkenRelaxation` must drive the same loop to ``tol``.
    """
    g, b, tol = 2.5, 1.0, 1e-12

    def run(use_aitken):
        ait = AitkenRelaxation(relax_max=1.0e6, relax0=1.0)
        u, r_km1, omega = 0.0, None, 1.0
        for it in range(1, 200):
            u_new = g * u + b
            r = u_new - u
            if abs(r) < tol:
                return it, True
            omega = ait.omega_for(r_km1, r) if (use_aitken and it >= 2) else 1.0
            u = u + omega * (u_new - u)
            r_km1 = r
        return 200, False

    assert run(False)[1] is False
    it, ok = run(True)
    assert ok and it < 40


def test_beam_static_tip_deflection_matches_analytic():
    t = 1.0 / 50.0
    props = BeamProperties(E=1e5, A=t, I=t ** 3 / 12.0, rho=1.0, length=1.0,
                           thickness=t)
    beam = CorotationalBeam2D(n_elements=10, props=props)
    load = np.zeros(beam.n_dof)
    load[3 * (beam.n_nodes - 1) + 1] = 1e-3
    u = np.zeros(beam.n_dof)
    for _ in range(20):
        r = beam.internal_force(u) - load
        u[beam.free] += np.linalg.solve(beam.tangent(u)[np.ix_(beam.free, beam.free)],
                                        -r[beam.free])
    w = u[3 * (beam.n_nodes - 1) + 1]
    exact = 1e-3 * props.length ** 3 / (3.0 * props.E * props.I)
    assert abs(w - exact) / exact < 1e-3


def test_beam_predictor_second_order_coefficients():
    t = 1.0 / 50.0
    beam = CorotationalBeam2D(n_elements=4, props=BeamProperties(
        E=1e5, A=t, I=t ** 3 / 12.0, rho=1.0, length=1.0, thickness=t))
    beam.u_n = np.full(beam.n_dof, 1.0)
    beam.u_nm1 = np.full(beam.n_dof, 2.0)
    beam.u_nm2 = np.full(beam.n_dof, 3.0)
    pred = beam.predict(5)
    assert np.allclose(pred[3:], 2.5 - 4.0 + 1.5)
    assert np.allclose(pred[:3], 0.0)


def test_lbm_rest_state_is_preserved():
    fl = LBMFluid2D((16, 16), omega=1.8, rho0=1.0)
    for _ in range(5):
        rho, u = fl.step()
    assert np.allclose(rho, 1.0) and np.allclose(u, 0.0, atol=1e-14)
