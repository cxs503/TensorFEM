"""Verification tests for the UNDEX shock chain (Cole -> Taylor -> SRS -> DDAM).

Every test asserts an invariant or an external/closed-form reference, not merely
that the code executes.
"""
import math

import numpy as np
import pytest
import torch

from tensorfem.undex_shock import (
    SEAWATER_DENSITY, SEAWATER_SOUND_SPEED, assemble_ss_plate,
    cole_decay_time, cole_peak_pressure, cole_point, cole_pressure,
    compare_to_taylor_analytic, dimensional_homogeneity_check, interp_srs,
    kirchhoff_ss_plate_frequency, log_frequencies, newmark_equivalence_check,
    plate_modal_unit_stress, q4_top_surface_stress, shock_acceleration_history,
    srs, srs_grid_drift, taylor_analytic, taylor_fsi_fem, yield_margin_ledger,
)
from tensorfem.undex_shock import _element_von_mises_batch, _ss_free_dofs
from tensorfem.undex_shock import _assemble_plate_KM


# ---------------------------------------------------------------------------
# Cole environment
# ---------------------------------------------------------------------------
def test_cole_hopkinson_scaling():
    p = cole_peak_pressure(100.0, 20.0)
    th = cole_decay_time(100.0, 20.0)
    for s in (0.25, 2.0, 5.0):
        assert cole_peak_pressure(s ** 3 * 100.0, s * 20.0) == pytest.approx(p, rel=1e-14)
        assert cole_decay_time(s ** 3 * 100.0, s * 20.0) == pytest.approx(s * th, rel=1e-14)


def test_cole_rejects_non_physical_input():
    with pytest.raises(ValueError):
        cole_peak_pressure(0.0, 10.0)
    with pytest.raises(ValueError):
        cole_peak_pressure(10.0, -1.0)


def test_cole_closed_forms_match_integration():
    pt = cole_point(250.0, 30.0)
    t = torch.linspace(0.0, 40.0 * pt.decay_time_s, 400001, dtype=torch.float64)
    p = cole_pressure(t, 250.0, 30.0)
    impulse = float(torch.trapezoid(p, t))
    energy = float(torch.trapezoid(p * p, t)) / (SEAWATER_DENSITY * SEAWATER_SOUND_SPEED)
    assert impulse == pytest.approx(pt.specific_impulse_pa_s, rel=1e-5)
    assert energy == pytest.approx(pt.energy_flux_j_m2, rel=1e-5)


def test_dimensional_homogeneity_checks_pass():
    result = dimensional_homogeneity_check()
    assert result["pass"], result
    for name, check in result["checks"].items():
        assert check["pass"], (name, check)


# ---------------------------------------------------------------------------
# Taylor air-backed plate
# ---------------------------------------------------------------------------
def test_taylor_heavy_plate_limit():
    """m -> infinity: v -> (2 p_m theta/m)(1 - e^{-t/theta}) (rigid-wall limit)."""
    p_m, theta, m = 1.0e7, 5.0e-4, 1.0e8
    t = torch.linspace(0.0, 3.0 * theta, 2001, dtype=torch.float64)
    ref = taylor_analytic(t, peak_pressure=p_m, decay_time=theta, areal_mass=m)
    v_expect = (2.0 * p_m * theta / m) * (1.0 - torch.exp(-t / theta))
    w_expect = (2.0 * p_m * theta / m) * (t - theta * (1.0 - torch.exp(-t / theta)))
    assert float((ref.velocity - v_expect).abs().max()) < 1e-3 * float(v_expect.abs().max())
    assert float((ref.displacement - w_expect).abs().max()) < 1e-3 * float(w_expect.abs().max())
    # heavy plate: the interface pressure is (almost) the rigid-wall doubling
    assert float(ref.interface_pressure[0]) == pytest.approx(2.0 * p_m, rel=1e-9)


def test_taylor_analytic_satisfies_impedance_relation():
    p_m, theta, m = 1.0e7, 5.5e-4, 94.2
    t = torch.linspace(0.0, 2.0 * theta, 1000, dtype=torch.float64)
    ref = taylor_analytic(t, peak_pressure=p_m, decay_time=theta, areal_mass=m)
    lhs = ref.interface_pressure
    rhs = 2.0 * ref.incident_pressure - SEAWATER_DENSITY * SEAWATER_SOUND_SPEED * \
        ref.velocity
    assert float((lhs - rhs).abs().max()) < 1e-6 * p_m


def test_newmark_factorized_matches_upstream():
    result = newmark_equivalence_check()
    assert result["pass"], result


def test_taylor_fem_matches_analytic():
    """FE air-backed plate vs closed-form Taylor (window: before cavitation)."""
    p_m, theta, m = 1.0e7, 5.5e-4, 94.2
    fem = taylor_fsi_fem(peak_pressure=p_m, decay_time=theta, areal_mass=m,
                         elements=800, dt=5.0e-6, duration=2.0 * theta)
    ref = taylor_analytic(fem.time.numpy(), peak_pressure=p_m, decay_time=theta,
                          areal_mass=m)
    cmp = compare_to_taylor_analytic(fem, ref)
    peak_v = abs(float(fem.plate_velocity.abs().max()) -
                 float(ref.velocity.abs().max())) / float(ref.velocity.abs().max())
    peak_w = abs(float(fem.plate_displacement.abs().max()) -
                 float(ref.displacement.abs().max())) / float(ref.displacement.abs().max())
    assert peak_v < 0.03, peak_v
    assert peak_w < 0.03, peak_w
    assert cmp["velocity"]["relative_l2_error"] < 0.03, cmp["velocity"]
    # sign convention: the plate is driven into the water, Taylor velocity > 0
    assert float(fem.plate_velocity[5]) > 0.0
    # The interface pressure is recovered from one linear element, so the
    # instantaneous pressure doubling at t = 0 (a jump that a linear field cannot
    # represent) is smeared: the residual is O(p_m) in the first few samples and
    # must then collapse.  Check the collapse rather than the jump.
    residual = fem.impedance_residual.abs().numpy()
    p_inc = fem.incident_pressure.abs().max().item()
    assert residual[:5].max() > 0.5 * p_inc
    assert residual[10:].max() < 0.1 * p_inc, residual[10:].max() / p_inc


def test_taylor_fem_converges_under_refinement():
    p_m, theta, m = 1.0e7, 5.5e-4, 94.2
    errors = []
    for elements, dt in ((200, 1.0e-5), (800, 2.5e-6)):
        fem = taylor_fsi_fem(peak_pressure=p_m, decay_time=theta, areal_mass=m,
                             elements=elements, dt=dt, duration=2.0 * theta)
        ref = taylor_analytic(fem.time.numpy(), peak_pressure=p_m, decay_time=theta,
                              areal_mass=m)
        cmp = compare_to_taylor_analytic(fem, ref)
        errors.append(cmp["velocity"]["relative_max_error"])
    # observed first-order (in h) convergence of the sample-wise velocity error
    assert errors[1] < errors[0] / 3.0, errors
    assert errors[1] < 0.04, errors


# ---------------------------------------------------------------------------
# SRS
# ---------------------------------------------------------------------------
def test_sdof_matches_rectangular_pulse_closed_form():
    """zeta = 0, constant base acceleration for t_d: exact SDOF solution."""
    a0, f, td = 10.0, 20.0, 0.2
    dt = td / 2000.0
    n = int(round(td / dt)) + 1
    acc = np.full(n, a0)
    freq = np.array([f])
    res = srs(acc, dt, freq, damping_ratios=(0.0,))
    omega = 2.0 * math.pi * f
    expect = a0 / omega ** 2 * (1.0 - math.cos(omega * td)) if omega * td < math.pi \
        else 2.0 * a0 / omega ** 2
    assert res.sd_m[0.0][0] == pytest.approx(expect, rel=5e-3)


def test_srs_matches_independent_ivp_integration():
    from scipy.integrate import solve_ivp
    cole = cole_point(100.0, 20.0)
    hist = shock_acceleration_history(cole, areal_mass=94.2, dt=2.0e-6)
    t = hist.time.numpy()
    mask = t <= hist.cavitation_time_s
    acc = hist.acceleration.numpy()[mask]
    dt = float(t[1] - t[0])
    freqs = np.array([30.0, 150.0, 800.0])
    res = srs(acc, dt, freqs, damping_ratios=(0.05,))
    times = t[mask]

    def fun(tt, y):
        a = np.interp(tt, times, acc)
        return [y[1], -a - 2 * 0.05 * 2 * np.pi * w_ * y[1] - (2 * np.pi * w_) ** 2 * y[0]]

    for i, f in enumerate(freqs):
        w_ = f
        sol = solve_ivp(fun, (0.0, times[-1]), [0.0, 0.0], t_eval=times,
                        rtol=1e-10, atol=1e-12, max_step=dt)
        assert res.sd_m[0.05][i] == pytest.approx(float(np.abs(sol.y[0]).max()), rel=5e-3)


def test_srs_grid_drift_below_two_percent():
    cole = cole_point(100.0, 20.0)
    hist = shock_acceleration_history(cole, areal_mass=94.2, dt=1.0e-6)
    t = hist.time.numpy()
    mask = t <= hist.cavitation_time_s
    acc = hist.acceleration.numpy()[mask]
    for zeta in (0.0, 0.02, 0.05):
        drift = srs_grid_drift(acc, 1.0e-6, zeta, n_coarse=512, n_fine=1024)
        assert drift["max_relative_drift"] < 0.02, (zeta, drift)


def test_srs_is_monotone_in_damping_at_the_peak():
    cole = cole_point(250.0, 20.0)
    hist = shock_acceleration_history(cole, areal_mass=94.2, dt=1.0e-6)
    t = hist.time.numpy()
    mask = t <= hist.cavitation_time_s
    res = srs(hist.acceleration.numpy()[mask], 1.0e-6,
              log_frequencies(1.0, 20000.0, 256), damping_ratios=(0.0, 0.05))
    assert res.sa_ms2[0.05].max() <= res.sa_ms2[0.0].max() * 1.0000001


# ---------------------------------------------------------------------------
# Plate modal model, stress recovery, DDAM
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def modal():
    return assemble_ss_plate(length=1.5, width=1.0, nx=12, ny=8, thickness=0.012,
                             modes=6)


def test_plate_modal_frequencies_match_kirchhoff(modal):
    # first two modes on the coarse verification mesh; the higher modes carry a
    # genuine Mindlin (rotary inertia / shear) offset against Kirchhoff
    for got, ref in zip(modal.frequencies_hz.numpy()[:2],
                        modal.analytical_frequencies_hz[:2]):
        assert got == pytest.approx(ref, rel=0.03)
    fine = assemble_ss_plate(length=1.5, width=1.0, nx=24, ny=16, thickness=0.012,
                             modes=4)
    assert float(fine.frequency_errors[0]) < float(modal.frequency_errors[0])
    assert float(fine.frequencies_hz[0]) == pytest.approx(
        float(modal.frequencies_hz[0]), rel=0.01)
    assert modal.cumulative_mass_fraction[-1] > 0.6


def test_plate_modal_effective_mass_antisymmetric_modes_vanish(modal):
    # uniform base excitation: only the symmetric (odd, odd) modes participate
    participation = modal.participation.numpy()
    assert abs(participation[0]) > 1.0
    assert abs(participation[1]) < 1e-8


def test_plate_stress_recovery_batch_matches_scalar(modal):
    # margin=0 keeps every element, so the batch values can be compared directly
    unit = plate_modal_unit_stress(modal, margin=0.0)
    for mode in range(2):
        u = modal.modes_full[:, mode]
        for e in (0, modal.elements.shape[0] // 2):
            conn = modal.elements[e]
            dofs = torch.stack((3 * conn, 3 * conn + 1, 3 * conn + 2),
                               dim=1).reshape(-1)
            _, _, _, von = q4_top_surface_stress(modal.nodes[conn, :], u[dofs],
                                                 young=modal.young,
                                                 poisson=modal.poisson,
                                                 thickness=modal.thickness)
            assert float(von.max()) == pytest.approx(float(unit[mode, e]), rel=1e-10)


def test_static_plate_stress_matches_navier_reference():
    """Uniformly loaded SS square plate (nu = 0.3), plate centre.

    Reference (Timoshenko/Woinowsky-Krieger): w_max = 0.00406 q a^4/D and
    sigma_max = 0.2874 q a^2/h^2.  The corner value is *not* used as the
    reference: the idealised SS corner has a singular reaction, so the discrete
    moment there grows under refinement (35.3 -> 38.3 MPa between 16x16 and
    24x24) while the centre is stable.
    """
    side, thickness, q0 = 1.0, 0.01, 1.0e4
    young, poisson = 206.0e9, 0.3
    n = 16
    xs = torch.linspace(0.0, side, n + 1, dtype=torch.float64)
    xx, yy = torch.meshgrid(xs, xs, indexing="xy")
    nodes = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
    elements = []
    for j in range(n):
        for i in range(n):
            n0 = j * (n + 1) + i
            elements.append((n0, n0 + 1, n0 + n + 2, n0 + n + 1))
    elements = torch.tensor(elements, dtype=torch.long)
    ndof = 3 * nodes.shape[0]
    K, _ = _assemble_plate_KM(nodes, elements, ndof, young, poisson, thickness, 7850.0)
    free = _ss_free_dofs(ndof, n, n)
    f = torch.zeros(ndof, dtype=torch.float64)
    area = (side / n) ** 2
    f[torch.arange(0, ndof, 3)] = q0 * area
    u = torch.zeros(ndof, dtype=torch.float64)
    u[free] = torch.linalg.solve(K[free[:, None], free], f[free])
    u_elem = u.numpy()[np.stack((3 * elements.numpy(), 3 * elements.numpy() + 1,
                                 3 * elements.numpy() + 2), axis=-1).reshape(-1, 12)]
    von = _element_von_mises_batch(nodes.numpy(), elements.numpy(), u_elem,
                                   young=young, poisson=poisson, thickness=thickness)
    sigma_ref = 0.2874 * q0 * side ** 2 / thickness ** 2
    centre = int(np.argmin(np.abs(nodes[elements].mean(dim=1).numpy()[:, 0] - side / 2)
                           + np.abs(nodes[elements].mean(dim=1).numpy()[:, 1] - side / 2)))
    assert float(von[centre]) == pytest.approx(sigma_ref, rel=0.02)
    # centre node of the quarter-point ordering is not the centre element node;
    # take the node closest to the plate centre
    node_centre = int(np.argmin(((nodes - torch.tensor([side / 2, side / 2],
                                                      dtype=torch.float64)) ** 2).sum(1)))
    w_num = float(u[3 * node_centre])
    D = young * thickness ** 3 / (12.0 * (1.0 - poisson ** 2))
    w_ref = 0.00406 * q0 * side ** 4 / D
    assert w_num == pytest.approx(w_ref, rel=0.02)


def test_yield_margin_ledger_and_spectrum_consistency(modal):
    unit = plate_modal_unit_stress(modal)
    cole = cole_point(100.0, 20.0)
    hist = shock_acceleration_history(cole, areal_mass=94.2, dt=2.0e-6)
    t = hist.time.numpy()
    mask = t <= hist.cavitation_time_s
    acc = hist.acceleration.numpy()[mask]
    freqs = log_frequencies(1.0, 20000.0, 256)
    res = srs(acc, 2.0e-6, freqs, damping_ratios=(0.05,))
    ledger = yield_margin_ledger(modal, srs_result=res, damping=0.05,
                                 yield_stress=355.0e6, unit_stress=unit)
    assert ledger["nrl_design_stress_pa"] > 0.0
    assert ledger["margin"] > 0.0
    assert ledger["verdict"] in {"PASS", "FAIL"}
    assert ledger["cumulative_mass_fraction_used"] > 0.5
    # the spectral displacement must equal the direct SDOF maximax of the same
    # history at the modal frequency (response-spectrum consistency)
    from tensorfem.undex_shock import _sdof_max_response
    f1 = float(modal.frequencies_hz[0])
    xmax, _ = _sdof_max_response(acc, 2.0e-6, 2.0 * math.pi * f1, 0.05)
    sd = float(interp_srs(res, 0.05, [f1], quantity="sd")[0])
    assert sd == pytest.approx(xmax, rel=0.02)