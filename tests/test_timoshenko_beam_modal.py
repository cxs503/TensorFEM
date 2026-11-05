"""Regression tests for the shear-flexible (Timoshenko) beam vibration module."""

from __future__ import annotations

import math

import pytest
import torch

from tensorfem import modal
from tensorfem.timoshenko_beam_modal import (
    TimoshenkoBeamSection,
    cantilever_timoshenko_modes,
    classical_simply_supported_timoshenko_angular_frequencies,
    euler_bernoulli_simply_supported_angular_frequencies,
    exact_simply_supported_timoshenko_angular_frequencies,
    rectangular_shear_coefficient,
    shear_parameter,
    simply_supported_timoshenko_modes,
    timoshenko_element_matrices,
    timoshenko_equation_eigenvalues,
)


def steel(slenderness: float = 5.0, kappa: float = 5.0 / 6.0) -> TimoshenkoBeamSection:
    length = 1.0
    return TimoshenkoBeamSection(
        length=length,
        height=length / slenderness,
        width=0.1,
        young=210.0e9,
        poisson=0.3,
        density=7850.0,
        shear_coefficient=kappa,
    )


def rel_error(computed, reference) -> float:
    return float(((computed - reference) / reference).abs().max())


def test_section_properties():
    section = steel(5.0)
    assert section.area == pytest.approx(0.1 * 0.2)
    assert section.inertia == pytest.approx(0.1 * 0.2**3 / 12.0)
    assert section.shear_modulus == pytest.approx(210e9 / (2.0 * 1.3))
    assert section.slenderness == pytest.approx(5.0)
    with pytest.raises(ValueError):
        TimoshenkoBeamSection(1.0, 0.2, 0.1, 210e9, 0.5, 7850.0).validate()
    with pytest.raises(ValueError):
        TimoshenkoBeamSection(-1.0, 0.2, 0.1, 210e9, 0.3, 7850.0).validate()


def test_rectangular_shear_coefficient_bounds():
    # 10(1+nu)/(12+11nu) is close to but not identical with the common 5/6
    kappa_third = rectangular_shear_coefficient(1.0 / 3.0)
    assert 0.83 < kappa_third < 0.87
    assert kappa_third == pytest.approx(13.3333333333 / 15.6666666667, rel=1e-9)
    assert rectangular_shear_coefficient(0.0) == pytest.approx(10.0 / 12.0)
    assert rectangular_shear_coefficient(0.3) > rectangular_shear_coefficient(0.0)


def test_shear_parameter_scaling():
    section = steel(5.0)
    phi_a = shear_parameter(section, 0.1)
    phi_b = shear_parameter(section, 0.05)
    # Phi ~ 1/L^2
    assert phi_b == pytest.approx(4.0 * phi_a, rel=1e-12)


def test_element_matrices_symmetry_and_positive_semidefinite():
    section = steel(5.0)
    K, M = timoshenko_element_matrices(section, 0.05)
    assert torch.allclose(K, K.T, atol=0.0)
    assert torch.allclose(M, M.T, atol=0.0)
    eig_m = torch.linalg.eigvalsh(M)
    assert float(eig_m.min()) > 0.0
    eig_k = torch.linalg.eigvalsh(K)
    scale = float(K.abs().max())
    # two rigid-body modes -> two (numerically) zero eigenvalues
    assert float(eig_k[0]) > -1e-10 * scale
    assert float(eig_k[1]) > -1e-10 * scale
    assert float(eig_k[2]) > 1e-6 * scale


def test_rigid_body_modes_carry_correct_mass():
    section = steel(5.0)
    for le in (0.5, 0.05, 0.005):
        K, M = timoshenko_element_matrices(section, le)
        translation = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float64)
        rotation = torch.tensor([0.0, 1.0, le, 1.0], dtype=torch.float64)
        assert float((K @ translation).abs().max()) <= 1e-12 * float(K.abs().max())
        assert float((K @ rotation).abs().max()) <= 1e-12 * float(K.abs().max())
        # rigid translation: total mass only
        assert float(translation @ M @ translation) == pytest.approx(
            section.rho_area * le, rel=1e-12
        )
        # rigid rotation w = x, theta = 1
        expected = section.rho_area * le**3 / 3.0 + section.density * section.inertia * le
        assert float(rotation @ M @ rotation) == pytest.approx(expected, rel=1e-12)


def test_exact_reference_cross_check():
    section = steel(5.0)
    closed = exact_simply_supported_timoshenko_angular_frequencies(section, 4)
    eigen = timoshenko_equation_eigenvalues(section, 4)
    assert rel_error(closed, eigen) < 1e-12


def test_classical_formula_is_a_lower_bound_on_exact_root():
    """The widely quoted formula drops the omega^4 term, so it under-predicts."""
    section = steel(5.0)
    exact = exact_simply_supported_timoshenko_angular_frequencies(section, 4)
    classical = classical_simply_supported_timoshenko_angular_frequencies(section, 4)
    assert torch.all(classical < exact)
    # deviation grows with mode number
    err = ((exact - classical) / exact).tolist()
    assert err[0] < err[1] < err[2] < err[3]
    assert 0.001 < err[0] < 0.002  # 0.13 % for mode 1
    assert 0.04 < err[3] < 0.06  # 5 % for mode 4


def test_slender_limit_recovers_euler_bernoulli():
    section = steel(100.0)
    timo = exact_simply_supported_timoshenko_angular_frequencies(section, 4)
    eb = euler_bernoulli_simply_supported_angular_frequencies(section, 4)
    err = ((timo - eb) / eb).abs().tolist()
    assert err[0] < 1e-3  # 0.02 % for mode 1 at L/h = 100
    assert err[-1] < 5e-3  # 0.27 % for mode 4
    assert err[0] < err[-1]


def test_simply_supported_frequencies_match_exact_reference():
    section = steel(5.0)
    exact = exact_simply_supported_timoshenko_angular_frequencies(section, 4)
    result = simply_supported_timoshenko_modes(section, 32, 4)
    assert rel_error(result.angular_frequencies, exact) < 0.03
    # and the observed error is actually far below the 3 % gate
    assert rel_error(result.angular_frequencies, exact) < 0.01


def test_convergence_rate_is_second_order():
    section = steel(5.0)
    exact = exact_simply_supported_timoshenko_angular_frequencies(section, 4)
    errors = {}
    for elements in (16, 32, 64):
        result = simply_supported_timoshenko_modes(section, elements, 4)
        errors[elements] = ((result.angular_frequencies - exact) / exact).abs()
    for mode in range(4):
        rate = math.log(float(errors[16][mode]) / float(errors[64][mode])) / math.log(4.0)
        assert 1.8 < rate < 2.2, (mode, rate)
    # monotone improvement
    for mode in range(4):
        assert errors[64][mode] < errors[32][mode] < errors[16][mode]


def test_no_shear_locking_for_deep_beam():
    section = steel(2.0)
    exact = exact_simply_supported_timoshenko_angular_frequencies(section, 2)
    result = simply_supported_timoshenko_modes(section, 128, 2)
    assert rel_error(result.angular_frequencies, exact) < 0.01
    # frequencies must not collapse to (near) zero as the beam gets deeper
    assert float(result.angular_frequencies[0]) > 0.5 * float(exact[0])


def test_euler_bernoulli_module_is_inadequate_for_deep_beam():
    """Documented debt: the Euler-Bernoulli module misses the 3 % gate at L/h = 5."""
    section = steel(5.0)
    exact = exact_simply_supported_timoshenko_angular_frequencies(section, 1)
    EI = section.young * section.inertia
    eb_fe = modal.simply_supported_beam_modes(section.length, 32, EI, section.rho_area, modes=1)
    assert float(((eb_fe.angular_frequencies[0] - exact[0]) / exact[0]).abs()) > 0.03
    # while the shear-flexible module passes it
    timo = simply_supported_timoshenko_modes(section, 32, 1)
    assert float(((timo.angular_frequencies[0] - exact[0]) / exact[0]).abs()) < 0.03


def test_cantilever_slender_limit_matches_euler_bernoulli():
    section = steel(100.0)
    EI = section.young * section.inertia
    exact_eb = modal.cantilever_exact_angular_frequency(
        section.length, EI, section.rho_area, mode=1
    )
    result = cantilever_timoshenko_modes(section, 32, 1)
    assert abs(float(result.angular_frequencies[0]) - exact_eb) / exact_eb < 0.03


def test_cantilever_deep_beam_is_below_euler_bernoulli():
    section = steel(5.0)
    EI = section.young * section.inertia
    eb = modal.cantilever_beam_modes(section.length, 64, EI, section.rho_area, modes=4)
    timo = cantilever_timoshenko_modes(section, 64, 4)
    assert torch.all(timo.angular_frequencies < eb.angular_frequencies)
    # the correction grows with mode number
    reduction = ((eb.angular_frequencies - timo.angular_frequencies) / eb.angular_frequencies).tolist()
    assert reduction[0] < reduction[1] < reduction[2] < reduction[3]


def test_reference_json_is_reproducible():
    """The committed reference data must match a fresh computation."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "benchmarks" / "timoshenko_beam_modal_reference.json"
    if not path.exists():
        pytest.skip("reference json not generated")
    data = json.loads(path.read_text())
    section = steel(data["meta"]["canonical_slenderness"])
    exact = exact_simply_supported_timoshenko_angular_frequencies(section, data["meta"]["modes"])
    # agreement is asserted at 7 digits: the stored JSON is decimal text and
    # the comparison must tolerate last-digit round-trip noise
    assert rel_error(torch.tensor(data["convergence"]["exact_angular_frequencies"], dtype=torch.float64), exact) < 1e-6
    result = simply_supported_timoshenko_modes(section, 32, data["meta"]["modes"])
    stored = data["convergence"]["rows"][3]
    assert stored["elements"] == 32
    assert rel_error(result.angular_frequencies, torch.tensor(stored["angular_frequencies"], dtype=torch.float64)) < 1e-9