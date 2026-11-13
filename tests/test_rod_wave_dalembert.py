"""Verification tests for the 1D rod d'Alembert case (CASE-DG-DALEMBERT-ELASTIC-1D).

Independence: the reference is the closed-form d'Alembert solution (Graff 1975
rod limit) and the method of images; it never consumes an FE solve.  The FE side
is the 2-node linear bar, so agreement is a verification statement.
"""
import math

import numpy as np
import pytest

from tensorfem.rod_wave_dalembert import (
    GaussianPulse,
    STEEL,
    assemble_bar,
    bar_element_matrices,
    compare_stress_profile,
    dalembert_components,
    exact_rod_state,
    exact_rod_stress,
    measure_mode_frequency,
    numerical_phase_speed_ratio,
    rod_wave_speed,
    run_case,
    solve_bar_wave,
    stable_time_step,
)

E = STEEL["young"]
RHO = STEEL["density"]
C = rod_wave_speed(E, RHO)
LENGTH = 16.0
PULSE = GaussianPulse(amplitude=1.0, center=5.0, width=0.5)


def _sin_mode(length, index=1):
    k = index * math.pi / length

    class _Mode:
        width = 0.0

        def velocity(self, x):
            return np.sin(k * np.asarray(x, dtype=float))

        def integral(self, a, b):
            a = np.asarray(a, dtype=float)
            b = np.asarray(b, dtype=float)
            return (-np.cos(k * b) + np.cos(k * a)) / k

    return _Mode(), k


# ---------------------------------------------------------------------------
# material / element level
# ---------------------------------------------------------------------------
def test_wave_speed_and_element_matrices():
    assert rod_wave_speed(210e9, 7850.0) == pytest.approx(5172.19, rel=2e-4)
    with pytest.raises(ValueError):
        rod_wave_speed(0.0, 7850.0)
    stiff, mass = bar_element_matrices(0.5, E, RHO, 1.0)
    assert np.allclose(stiff, stiff.T) and np.allclose(mass, mass.T)
    assert np.allclose(stiff.sum(axis=1), 0.0)          # rigid-body translation
    assert mass.sum() == pytest.approx(RHO * 1.0 * 0.5)  # total mass preserved
    assert np.all(np.linalg.eigvalsh(mass) > 0)          # consistent mass is SPD


def test_assembled_spectrum_matches_closed_form():
    system = assemble_bar(200, LENGTH, E, RHO)
    h = system["h"]
    omega_max = stable_time_step(h, E, RHO)[1]
    # sqrt(12) c / h is the infinite-lattice extreme of the consistent-mass bar
    assert omega_max == pytest.approx(math.sqrt(12.0) * C / h, rel=1e-12)
    # long-wave limit of the discrete dispersion law
    assert numerical_phase_speed_ratio(1e-3) == pytest.approx(1.0, rel=1e-6)
    theta = 0.1
    # consistent mass: phase speed slightly ABOVE c (+theta^2/24, earlier arrival)
    assert numerical_phase_speed_ratio(theta) - 1.0 == pytest.approx(theta ** 2 / 24.0, rel=2e-3)
    # lumped mass: same magnitude, opposite sign (-theta^2/24, later arrival)
    assert 1.0 - numerical_phase_speed_ratio(theta, mass_form="lumped") == pytest.approx(
        theta ** 2 / 24.0, rel=2e-3
    )


# ---------------------------------------------------------------------------
# the reference itself: it must solve the PDE and satisfy the end conditions
# ---------------------------------------------------------------------------
def test_exact_reference_solves_wave_equation():
    x = np.linspace(2.0, 12.0, 401)
    t = 4.0e-4
    dt = 2.0e-7
    dx = x[1] - x[0]
    u0 = exact_rod_state(x, t, pulse=PULSE, young=E, density=RHO)["displacement"]
    up = exact_rod_state(x, t + dt, pulse=PULSE, young=E, density=RHO)["displacement"]
    um = exact_rod_state(x, t - dt, pulse=PULSE, young=E, density=RHO)["displacement"]
    u_tt = (up - 2.0 * u0 + um) / dt ** 2
    u_xx = (u0[2:] - 2.0 * u0[1:-1] + u0[:-2]) / dx ** 2
    scale = np.max(np.abs(u_tt))
    assert np.max(np.abs(u_tt[1:-1] - C ** 2 * u_xx)) / scale < 5e-3


def test_exact_reference_satisfies_end_conditions():
    x = np.linspace(0.0, LENGTH, 4001)
    t = 1.5e-3  # a reflection has already happened at both parities
    for boundary, checker in (
        ("free", lambda s: abs(s["strain"][0]) + abs(s["strain"][-1])),
        ("fixed", lambda s: abs(s["displacement"][0]) + abs(s["displacement"][-1])),
    ):
        state = exact_rod_state(x, t, pulse=PULSE, young=E, density=RHO, length=LENGTH, boundary=boundary)
        assert checker(state) < 1e-9 * (1.0 + np.max(np.abs(PULSE.velocity(x))))


def test_travelling_waves_have_no_physical_dispersion():
    x = np.linspace(0.0, LENGTH, 4001)
    t1, t2 = 0.2e-3, 0.6e-3
    shift = C * (t2 - t1)
    early = dalembert_components(x, t1, pulse=PULSE, young=E, density=RHO)
    late = dalembert_components(x, t2, pulse=PULSE, young=E, density=RHO)
    right = dalembert_components(x - shift, t1, pulse=PULSE, young=E, density=RHO)["right"]
    left = dalembert_components(x + shift, t1, pulse=PULSE, young=E, density=RHO)["left"]
    scale = np.max(np.abs(early["right"]))
    # rigid translation of each lobe: a 1D rod has no physical dispersion
    assert np.max(np.abs(late["right"] - right)) / scale < 1e-12
    assert np.max(np.abs(late["left"] - left)) / scale < 1e-12
    # and the decomposition reproduces the full d'Alembert stress
    full = exact_rod_stress(x, t1, pulse=PULSE, young=E, density=RHO)
    assert np.max(np.abs(early["right"] + early["left"] - full)) / scale < 1e-12


# ---------------------------------------------------------------------------
# numerical dispersion law, measured in the time domain
# ---------------------------------------------------------------------------
def test_dispersion_law_matches_time_domain_measurement():
    for n_elem, mode_index in ((100, 16), (200, 32)):
        measured = measure_mode_frequency(
            n_elem=n_elem, mode_index=mode_index, length=LENGTH, young=E, density=RHO,
        )
        assert measured["formula_vs_measured_rel"] < 1e-3
        # consistent mass: the discrete speed is slightly above c (+theta^2/24),
        # so the numerical pulse arrives *early* -- quantify the excess
        excess = measured["c_num_over_c_formula"] - 1.0
        assert 0.0 < excess < 2.0 * measured["theta"] ** 2 / 24.0


# ---------------------------------------------------------------------------
# FE accuracy, convergence and the acceptance criterion
# ---------------------------------------------------------------------------
def test_stress_profile_second_order_and_acceptance():
    rows = []
    for n_elem in (200, 400, 800):
        rows.append(compare_stress_profile(
            n_elem=n_elem, length=LENGTH, young=E, density=RHO, pulse=PULSE, end_time=1.5e-3,
        ))
    errors = [row["rel_l2"] for row in rows]
    # h-refinement must halve the error by ~4 (second order)
    for coarse, fine in zip(errors[:-1], errors[1:]):
        assert coarse / fine == pytest.approx(4.0, rel=0.15)
    # acceptance: the fine profile is inside 2 % of d'Alembert
    assert errors[-1] < 0.02
    # ... and the coarse mesh genuinely fails, so the criterion has teeth
    assert errors[0] > 0.02
    # peak amplitude and leading-lobe position are also reproduced
    assert rows[-1]["lead_amplitude_ratio"] == pytest.approx(1.0, abs=1e-3)
    assert rows[-1]["peak_shift_widths"] < 0.02
    assert rows[-1]["correlation"] > 0.9999


def test_clamped_end_reflection_inverts_the_stress():
    free = compare_stress_profile(
        n_elem=400, length=LENGTH, young=E, density=RHO, pulse=PULSE,
        end_time=1.5e-3, boundary="free",
    )
    fixed = compare_stress_profile(
        n_elem=400, length=LENGTH, young=E, density=RHO, pulse=PULSE,
        end_time=1.5e-3, boundary="fixed",
    )
    assert free["reflected"]["present"] and fixed["reflected"]["present"]
    # free end: compression reflects as compression; clamped end: it inverts
    assert free["reflected"]["sign_exact"] == pytest.approx(-1.0)
    assert fixed["reflected"]["sign_exact"] == pytest.approx(1.0)
    assert free["reflected"]["sign_fem"] == pytest.approx(free["reflected"]["sign_exact"])
    assert fixed["reflected"]["sign_fem"] == pytest.approx(fixed["reflected"]["sign_exact"])
    assert free["rel_l2"] < 0.05 and fixed["rel_l2"] < 0.05


def test_energy_is_conserved_and_decays_with_the_square_of_dt():
    drifts = []
    for n_elem in (200, 400):
        run = solve_bar_wave(
            n_elem=n_elem, length=LENGTH, young=E, density=RHO, pulse=PULSE,
            end_time=1.5e-3, snapshot_times=[1.5e-3], energy_samples=100,
        )
        total = np.asarray(run["energy"]["total"])
        drifts.append(float(np.max(np.abs(total - total[0])) / total[0]))
    assert drifts[-1] < 1e-4
    assert drifts[0] / drifts[-1] == pytest.approx(4.0, rel=0.3)


def test_single_mode_accuracy_is_second_order():
    mode, k = _sin_mode(LENGTH)
    omega = k * C
    t_star = 0.9e-3
    errors = []
    for n_elem in (100, 200):
        run = solve_bar_wave(
            n_elem=n_elem, length=LENGTH, young=E, density=RHO, pulse=mode,
            end_time=t_star, boundary="fixed", snapshot_times=[t_star],
        )
        snap = run["snapshots"][t_star]
        exact = (1.0 / omega) * math.sin(omega * t_star) * np.sin(k * np.asarray(run["nodes"]))
        errors.append(float(np.linalg.norm(np.asarray(snap["displacement"]) - exact) / np.linalg.norm(exact)))
    assert errors[-1] < 5e-7
    assert errors[0] / errors[-1] == pytest.approx(4.0, rel=0.2)


# ---------------------------------------------------------------------------
# fail-closed guards
# ---------------------------------------------------------------------------
def test_fail_closed_inputs():
    with pytest.raises(ValueError):
        solve_bar_wave(n_elem=50, length=LENGTH, young=E, density=RHO, pulse=PULSE,
                       end_time=1e-3, cfl_safety=1.0)  # not below the stability limit
    with pytest.raises(ValueError):
        solve_bar_wave(n_elem=50, length=LENGTH, young=E, density=RHO, pulse=PULSE,
                       end_time=1e-3, boundary="simply-supported")
    with pytest.raises(ValueError):
        solve_bar_wave(n_elem=50, length=LENGTH, young=E, density=RHO, pulse=PULSE,
                       end_time=1e-3, probe_x=LENGTH + 1.0)
    with pytest.raises(ValueError):
        solve_bar_wave(n_elem=50, length=LENGTH, young=E, density=RHO, pulse=PULSE,
                       end_time=1e-3, snapshot_times=[2e-3])
    with pytest.raises(ValueError):
        assemble_bar(0, LENGTH, E, RHO)
    with pytest.raises(ValueError):
        GaussianPulse(width=0.0)


def test_run_case_verdict_pass_and_honest_fail():
    good = run_case(mesh_sequence=(200, 400, 800), end_time=1.5e-3)
    assert good["verdict"] == "pass"
    assert good["finest"]["rel_l2"] < good["tolerance"]
    assert good["translation_error"] < 1e-12
    assert good["decomposition_error"] < 1e-12
    assert good["probe"]["arrival_rel_error"] < 1e-2
    # an under-resolved mesh must be reported as an honest FAIL
    coarse = run_case(mesh_sequence=(100, 200), end_time=1.5e-3)
    assert coarse["verdict"] == "fail"