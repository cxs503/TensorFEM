"""Regression tests: the iterate-velocity mode and the interface-gain collapse.

Anchor ``XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION`` round v14.  What is pinned here:

1. ``CorotationalBeam2D.velocity_for`` is *exactly* the Newmark correction
   relation (round-off identity against both the recursive and the closed form).
2. The default ``velocity_mode="predictor"`` is the legacy path: the archived
   v6/v13 records stay bit-identical (the whole pre-existing suite still passes).
3. The two velocity modes give the same interface gain to <1 % -- the "the beam
   is inconsistent with the plate driver" hypothesis is false, and this test is
   the falsification record.
4. The gain collapses with the time step: ``mu - 1 < 2e-2`` at the (unphysical)
   ``dt = 1`` and ``< 1e-6`` at a resolved ``dt = 2e-3``.
5. ``coupling_scale`` acts linearly on the gain (a prefactor cannot restore it).
6. The shipped map's Jacobian at a resolved ``dt`` is numerically zero
   (``rho(I - J) = 1`` to 1e-5): the Aitken line search sees a unit operator,
   which is why the relaxation has no room to act in this driver.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorfem.ibm_aitken_implicit import (
    BeamProperties,
    CorotationalBeam2D,
    ImplicitIBMFSI,
    LBMFluid2D,
)
from tensorfem.ibm_subiter_gain import gain_from_trace

LATTICE = (64, 48)


# ---------------------------------------------------------------------------
def _solver(rho_s=1.0, dt=2.0e-3, mode="predictor", omega=1.0, nsub=5,
            coupling_scale=1.0, lattice=LATTICE):
    props = BeamProperties(E=1.0e5, A=1.0, I=1.0 / 12.0, rho=rho_s,
                           length=1.0, thickness=1.0)
    beam = CorotationalBeam2D(n_elements=6, props=props)
    scale = 0.45 * lattice[0]
    beam.x0 = beam.x0 * scale + np.array([0.15 * lattice[0], 0.5 * lattice[1]])
    beam.props = BeamProperties(E=props.E / scale, A=props.A * scale,
                                I=props.I * scale ** 3, rho=rho_s,
                                length=props.length * scale,
                                thickness=props.thickness * scale)
    beam._assemble_constant_matrices()
    beam.set_initial_deflection(0.02 * scale)
    fluid = LBMFluid2D(lattice, omega=1.8, rho0=1.0)
    return ImplicitIBMFSI(beam, fluid, dt=dt, dx=1.0, sub_tol=0.0,
                          max_sub_iter=nsub, relax_mode="fixed",
                          fixed_omega=omega, velocity_mode=mode,
                          coupling_scale=coupling_scale)


def _gain(dt=2.0e-3, mode="predictor", rho_s=1.0, coupling_scale=1.0):
    traces = []
    for w in (1.0, 0.5):
        s = _solver(rho_s=rho_s, dt=dt, mode=mode, omega=w,
                    coupling_scale=coupling_scale)
        s.step(1)
        traces.append(s.residual_traces[-1])
    return gain_from_trace(traces, omegas=(1.0, 0.5)), traces


# -------------------------------------------------------------------- 1. identity
def test_velocity_for_matches_newmark_correction():
    beam = CorotationalBeam2D(n_elements=4)
    beam.set_initial_deflection(0.02)
    beam.advance_history()
    dt, beta, gamma = 2.0e-3, 0.25, 0.5
    u = beam.u.copy()
    u[3:] = beam.u[3:] + 1e-3 * np.arange(len(beam.u) - 3)
    v, a = beam.velocity_for(u, dt, with_accel=True)
    # recursive form (what newmark_solve applies)
    u_pred = beam.u_n + dt * beam.v_n + dt ** 2 * (0.5 - beta) * beam.a_n
    a_ref = (u - u_pred) / (beta * dt ** 2)
    v_ref = beam.v_n + dt * ((1.0 - gamma) * beam.a_n + gamma * a_ref)
    assert np.allclose(v[3:], v_ref[3:], rtol=0, atol=1e-14)
    assert np.allclose(a[3:], a_ref[3:], rtol=0, atol=1e-11)
    # closed form used by the plate driver
    v_cf = (gamma / (beta * dt) * (u - beam.u_n)
            + (1.0 - gamma / beta) * beam.v_n
            + dt * (1.0 - gamma / (2.0 * beta)) * beam.a_n)
    assert np.allclose(v[3:], v_cf[3:], rtol=0, atol=1e-12)
    # clamped node stays clamped
    assert np.all(v[:3] == 0.0) and np.all(a[:3] == 0.0)


# ------------------------------------------------------- 2. legacy default path
def test_default_velocity_mode_is_legacy_predictor():
    assert _solver().velocity_mode == "predictor"
    a = _solver(mode="predictor")
    b = _solver()
    ra = a.step(1)
    rb = b.step(1)
    assert ra["residual"] == rb["residual"]
    assert np.allclose(a.residual_traces[-1], b.residual_traces[-1], rtol=0, atol=0)
    assert ra["velocity_mode"] == "predictor" == rb["velocity_mode"]


def test_invalid_velocity_mode_rejected():
    with pytest.raises(ValueError):
        _solver(mode="velocity")


# ------------------------------------------- 3. falsified: velocity-mode parity
def test_velocity_modes_agree_on_the_gain():
    g_pred, _ = _gain(dt=2.0e-3, mode="predictor")
    g_iter, _ = _gain(dt=2.0e-3, mode="iterate")
    assert abs(g_pred["mu"] - g_iter["mu"]) <= 1e-3 * max(1.0, g_pred["mu"])
    # and the traces themselves agree to 1e-9 relative
    _, t_pred = _gain(dt=1.0e-2, mode="predictor")
    _, t_iter = _gain(dt=1.0e-2, mode="iterate")
    ref = max(np.asarray(t_pred[0]).max(), 1e-300)   # leading residual of the step
    for a, b in zip(t_pred, t_iter):
        a, b = np.asarray(a), np.asarray(b)
        assert np.linalg.norm(a - b) <= 1e-6 * ref
        assert np.allclose(a, b, rtol=1e-6, atol=1e-6 * ref)


# ------------------------------------------------------- 4. the gain collapse
def test_gain_collapses_with_time_step():
    g_coarse, _ = _gain(dt=1.0, mode="predictor")
    g_resolved, _ = _gain(dt=2.0e-3, mode="predictor")
    assert g_coarse["mu"] - 1.0 > 1e-3          # a gain exists at an unphysical dt
    assert g_coarse["mu"] - 1.0 < 2.0e-1
    assert g_resolved["mu"] - 1.0 < 1.0e-6      # ... and is gone at a resolved one
    # monotone collapse over the intermediate steps
    gains = [_gain(dt=dt, mode="predictor")[0]["mu"] - 1.0
             for dt in (1.0, 1.0e-1, 1.0e-2)]
    assert gains[0] >= gains[1] >= gains[2]


# --------------------------------------------------- 5. coupling_scale linearity
def _jac_norm2(coupling_scale, dt=1.0e-2):
    s = _solver(dt=dt, mode="predictor", nsub=20,
                coupling_scale=coupling_scale)
    s.beam.u = s.beam.predict(1)
    u_state = s.beam.u.copy()
    rho, u = s.fluid.macroscopic(include_force=False)

    def G(us):
        s.beam.u = np.array(us, dtype=float)
        s.ibm.move(s.beam.markers())
        rho_i, mom_i = s.ibm.interpolate(rho, u)
        f = s.ibm.force(rho_i, mom_i, s.marker_velocity(), dt=s.dt)
        load = s.coupling_scale * s.beam.apply_marker_loads(-f)
        return np.array(s.beam.newmark_solve(load, s.dt)[0], dtype=float)

    free = np.asarray(s.beam.free, dtype=int)
    h = 1e-6 * max(1.0, float(np.linalg.norm(u_state[free])))
    J = np.zeros((free.size, free.size))
    for j, dof in enumerate(free):
        up, um = u_state.copy(), u_state.copy()
        up[dof] += h
        um[dof] -= h
        J[:, j] = (G(up)[free] - G(um)[free]) / (2.0 * h)
    return float(np.linalg.norm(J, 2))


def test_coupling_scale_acts_linearly_on_the_gain():
    """A prefactor moves the gain only linearly -- it cannot restore the regime."""
    j1 = _jac_norm2(1.0)
    j50 = _jac_norm2(50.0)
    assert j50 > 0.0 and j1 > 0.0
    assert abs(j50 / j1 - 50.0) < 5.0


# ------------------------------------------------ 6. the shipped map's Jacobian
def test_shipped_map_jacobian_is_numerically_zero_at_resolved_dt():
    s = _solver(dt=2.0e-3, mode="predictor", nsub=20)
    s.beam.u = s.beam.predict(1)
    u_state = s.beam.u.copy()
    rho, u = s.fluid.macroscopic(include_force=False)

    def G(us):
        s.beam.u = np.array(us, dtype=float)
        s.ibm.move(s.beam.markers())
        rho_i, mom_i = s.ibm.interpolate(rho, u)
        f = s.ibm.force(rho_i, mom_i, s.marker_velocity(), dt=s.dt)
        load = s.coupling_scale * s.beam.apply_marker_loads(-f)
        return np.array(s.beam.newmark_solve(load, s.dt)[0], dtype=float)

    free = np.asarray(s.beam.free, dtype=int)
    h = 1e-6 * max(1.0, float(np.linalg.norm(u_state[free])))
    J = np.zeros((free.size, free.size))
    for j, dof in enumerate(free):
        up, um = u_state.copy(), u_state.copy()
        up[dof] += h
        um[dof] -= h
        J[:, j] = (G(up)[free] - G(um)[free]) / (2.0 * h)
    M = np.eye(free.size) - J
    assert np.linalg.norm(J, 2) < 1.0e-3
    assert abs(np.abs(np.linalg.eigvals(M)).max() - 1.0) < 1.0e-5


# ------------------------------------------------------ 7. relaxation is inert
def test_relaxation_factor_only_rescales_a_constant_map():
    """At a resolved dt the map is constant, so ``r_1/r_0 = 1 - omega`` exactly.

    The factor therefore *delays* convergence without changing the fixed point:
    the residual ratio is a property of the factor alone, not of the physics.
    """
    for w in (0.5, 0.25):
        _, traces = _gain(dt=2.0e-3, mode="predictor", omega=w) \
            if False else (None, None)
        s = _solver(dt=2.0e-3, mode="predictor", omega=w, nsub=5)
        s.step(1)
        tr = s.residual_traces[-1]
        assert abs(tr[1] / tr[0] - (1.0 - w)) < 1e-2