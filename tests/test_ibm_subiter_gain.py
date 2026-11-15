"""Tests for the sub-iteration interface-gain diagnostics.

Anchor: ``XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION``.  These tests pin the two
things the v13 round claims:

* a *constant* return map (the interface fixed point reached in one sweep) shows
  residual ratios ``1 - omega`` exactly, i.e. ``mu = 1`` and the relaxation
  factor cannot influence anything -- measured on the genuine production driver;
* a map with real gain ``mu`` shows ``|1 - omega mu|`` and the fit recovers it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfem.ibm_aitken_implicit import (
    BeamProperties, CorotationalBeam2D, ImplicitIBMFSI, LBMFluid2D,
)
from tensorfem.ibm_subiter_gain import (
    INERT_TOL, added_mass_ratio, discrete_gain_bound, gain_from_trace,
    marker_local_mass,
)

OMEGAS = (1.0, 0.5, 0.25)


def _synthetic_traces(mu, omegas=OMEGAS, r0=1.0, n=3):
    out = []
    for w in omegas:
        rho = abs(1.0 - w * mu)
        out.append([r0 * rho ** k for k in range(n)])
    return out


@pytest.mark.parametrize("mu", [1.0, 1.25, 1.9, 2.5, 4.0, 60.0])
def test_gain_from_synthetic_map(mu):
    est = gain_from_trace(_synthetic_traces(mu), omegas=OMEGAS)
    assert est["mu"] == pytest.approx(mu, rel=2e-2, abs=2e-2)
    assert est["fit_error"] < 5e-2 * max(1.0, mu)


def test_constant_map_is_gain_one():
    """r1/r0 = 1 - omega is the signature of a map that ignores the iterate."""
    est = gain_from_trace(_synthetic_traces(1.0), omegas=OMEGAS)
    assert est["mu"] == pytest.approx(1.0, abs=1e-9)
    for w, rho in zip(OMEGAS, est["ratios"]):
        assert rho == pytest.approx(1.0 - w, abs=1e-12)
    assert est["fit_error"] < 1e-9
    assert abs(est["mu"] - 1.0) < INERT_TOL


def test_degenerate_trace_reports_nan():
    est = gain_from_trace([[0.0, 0.0, 0.0]], omegas=(1.0,))
    assert math.isnan(est["mu"])
    assert est["method"] == "degenerate_trace"


def test_marker_local_mass_matches_kernel_weights():
    xs = np.linspace(-6.0, 6.0, 13)
    pos = np.stack([xs + 32.0, np.full_like(xs, 32.0)], axis=1)
    from tensorfem.ibm_aitken_implicit import IBMInterface
    ibm = IBMInterface(pos, (64, 64), dx=1.0)
    rho_i = np.full(len(pos), 0.7)
    expected = float(np.sum(rho_i * ibm.eps_used * ibm.ds))
    assert marker_local_mass(ibm, rho_i) == pytest.approx(expected, rel=1e-14)
    # markers one cell apart overlap ~2 kernel supports, so the epsilon system
    # gives ~2 fluid cell masses per marker (not 1): the interface sees the
    # overlap, not the nominal spacing
    per_marker = float(np.sum(ibm.eps_used * ibm.ds)) / len(pos)
    assert 0.5 <= per_marker <= 4.0


def test_discrete_gain_bound_formula_and_monotonicity():
    g = discrete_gain_bound(M_f=10.0, M_s=100.0, k=0.0, dt=1.0)
    assert g == pytest.approx(0.5 * 1.0 * 10.0 / 100.0)
    assert discrete_gain_bound(M_f=20.0, M_s=100.0) > g
    assert discrete_gain_bound(M_f=10.0, M_s=100.0, k=50.0) < g


def test_added_mass_reference_is_lamb_plate():
    ref = added_mass_ratio(rho_f=1.0, c=8.0, M_s=500.0)
    assert ref["m_a"] == pytest.approx(math.pi * 64.0)
    assert ref["ratio"] == pytest.approx(math.pi * 64.0 / 500.0)
    assert ref["mu_predicted"] == pytest.approx(1.0 + ref["ratio"])
    assert added_mass_ratio(0.0, 8.0, 500.0)["mu_predicted"] == pytest.approx(1.0)


def _driver_trace(omega, n_iter=3, fluid_rho=1.0, rho_s=1.0,
                  lattice=(64, 48), n_elements=8, dt=2.0e-3):
    """Residual trace of one time level of the production beam driver."""
    sec = dict(E=1.0e5, A=1.0, I=1.0 / 12.0, rho=rho_s, length=1.0, thickness=1.0)
    beam = CorotationalBeam2D(n_elements=n_elements,
                              props=BeamProperties(**sec))
    scale = 0.45 * float(lattice[0])
    beam.x0 = beam.x0 * scale + np.array([0.15 * lattice[0], 0.5 * lattice[1]])
    beam.props = BeamProperties(E=beam.props.E / scale, A=beam.props.A * scale,
                                I=beam.props.I * scale ** 3, rho=beam.props.rho,
                                length=beam.props.length * scale,
                                thickness=beam.props.thickness * scale)
    beam._assemble_constant_matrices()
    beam.set_initial_deflection(0.05 * scale)
    fluid = LBMFluid2D(lattice, omega=1.8, rho0=fluid_rho)
    solver = ImplicitIBMFSI(beam, fluid, dt=dt, dx=1.0,
                            ref_length=beam.props.length, sub_tol=0.0,
                            max_sub_iter=n_iter, relax_mode="fixed",
                            fixed_omega=omega)
    solver.step(1)
    return solver.residual_traces[-1], beam, solver


def test_production_driver_trace_is_recorded():
    trace, _beam, solver = _driver_trace(1.0)
    assert len(trace) == 3
    assert all(np.isfinite(x) for x in trace)
    assert len(solver.residual_traces) == 1


def test_production_driver_map_is_constant():
    """The shipped loop's return map does not depend on the iterate.

    Measured on the genuine coupled driver: the residual after one relaxation is
    ``(1 - omega) r_0`` to round-off, i.e. one sweep already lands on the fixed
    point and Aitken's factor has nothing to act on.
    """
    traces = []
    for w in OMEGAS:
        tr, _b, _s = _driver_trace(w)
        traces.append(tr)
    r0 = [tr[0] for tr in traces]
    assert max(r0) / min(r0) == pytest.approx(1.0, rel=1e-8)
    for w, tr in zip(OMEGAS, traces):
        assert tr[1] / tr[0] == pytest.approx(1.0 - w, abs=2e-6)
    est = gain_from_trace(traces, omegas=OMEGAS)
    assert abs(est["mu"] - 1.0) < 1e-6
    assert abs(est["mu"] - 1.0) < INERT_TOL


def test_default_relaxation_is_untouched_by_the_trace():
    """Recording the trace must not change the shipped Aitken path."""
    from tensorfem.ibm_aitken_implicit import cantilever_free_vibration
    rec = cantilever_free_vibration(n_steps=4, n_elements=8, dt=2.0e-3,
                                    fluid_rho=0.2, lattice=(64, 48),
                                    sub_tol=1e-6, max_sub_iter=10)
    assert rec["steps_completed"] == 4
    assert all(s <= 2 for s in rec["sub_iterations"])
    assert rec["maxit_hits"] == 0