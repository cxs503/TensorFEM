import math

import pytest
import torch

from tensorfem.buckling import euler_pinned_critical_load, pinned_pinned_column_buckling
from tensorfem.modal import (
    cantilever_beam_modes,
    cantilever_exact_angular_frequency,
    clamped_clamped_beam_modes,
    clamped_clamped_exact_angular_frequency,
    clamped_pinned_beam_modes,
    clamped_pinned_exact_angular_frequency,
    simply_supported_beam_modes,
    simply_supported_exact_angular_frequency,
)


def relative_error(value: float, reference: float) -> float:
    return abs(value - reference) / abs(reference)


# ---------------------------------------------------------------------------
# Cantilever (existing tests, kept for regression)
# ---------------------------------------------------------------------------

def test_cantilever_first_frequency_standard_benchmark_under_three_percent():
    # SI units: 1 m steel-like beam with prescribed section properties.
    L, E, I, rho, area = 1.0, 210e9, 8.0e-8, 7850.0, 4.0e-4
    result = cantilever_beam_modes(L, elements=8, EI=E * I, rho_a=rho * area, modes=3)
    exact = cantilever_exact_angular_frequency(L, E * I, rho * area)
    error = relative_error(result.angular_frequencies[0].item(), exact)
    assert error < 0.03, f"cantilever frequency error {error:.3%} exceeds 3%"
    assert torch.all(result.angular_frequencies[1:] > result.angular_frequencies[:-1])


def test_cantilever_first_mode_is_mass_normalized():
    from tensorfem.modal import uniform_beam_matrices

    K, M = uniform_beam_matrices(1.0, 6, 2.0, 3.0)
    result = cantilever_beam_modes(1.0, 6, 2.0, 3.0, modes=2)
    phi = result.modes[:, 0]
    assert torch.isclose(phi @ M @ phi, torch.tensor(1.0, dtype=phi.dtype), atol=1e-10)


# ---------------------------------------------------------------------------
# Simply supported (pinned--pinned)
# ---------------------------------------------------------------------------

def test_simply_supported_first_four_frequencies_under_three_percent():
    L, E, I, rho, area = 2.0, 70e9, 1.2e-7, 2700.0, 6.0e-4
    result = simply_supported_beam_modes(L, elements=16, EI=E * I, rho_a=rho * area, modes=4)
    for m in range(1, 5):
        exact = simply_supported_exact_angular_frequency(L, E * I, rho * area, mode=m)
        error = relative_error(result.angular_frequencies[m - 1].item(), exact)
        assert error < 0.03, f"SS mode {m} error {error:.3%} exceeds 3%"


def test_simply_supported_convergence_is_monotone_from_above():
    """Stiffness-discretised frequencies are upper bounds."""
    L, EI, rho_a = 1.0, 1.0, 1.0
    exact = simply_supported_exact_angular_frequency(L, EI, rho_a)
    errors = []
    for n in (4, 8, 16, 32):
        result = simply_supported_beam_modes(L, elements=n, EI=EI, rho_a=rho_a, modes=1)
        errors.append(relative_error(result.angular_frequencies[0].item(), exact))
    assert all(e > 0 for e in errors), "FE frequency must be above exact"
    assert errors[-1] < errors[0], "error must decrease with refinement"
    assert errors[-1] < 0.01


def test_simply_supported_analytical_roots_are_exact_multiples_of_pi():
    for n in range(1, 6):
        root = n * math.pi
        omega = simply_supported_exact_angular_frequency(1.0, 1.0, 1.0, mode=n)
        assert omega == pytest.approx(root ** 2, rel=1e-15)


# ---------------------------------------------------------------------------
# Clamped--clamped
# ---------------------------------------------------------------------------

def test_clamped_clamped_first_three_frequencies_under_three_percent():
    L, E, I, rho, area = 1.5, 200e9, 5.0e-8, 7850.0, 3.0e-4
    result = clamped_clamped_beam_modes(L, elements=16, EI=E * I, rho_a=rho * area, modes=3)
    for m in range(1, 4):
        exact = clamped_clamped_exact_angular_frequency(L, E * I, rho * area, mode=m)
        error = relative_error(result.angular_frequencies[m - 1].item(), exact)
        assert error < 0.03, f"CC mode {m} error {error:.3%} exceeds 3%"


def test_clamped_clamped_convergence_monotone():
    L, EI, rho_a = 1.0, 1.0, 1.0
    exact = clamped_clamped_exact_angular_frequency(L, EI, rho_a)
    errors = []
    for n in (4, 8, 16, 32):
        result = clamped_clamped_beam_modes(L, elements=n, EI=EI, rho_a=rho_a, modes=1)
        errors.append(relative_error(result.angular_frequencies[0].item(), exact))
    assert all(e > 0 for e in errors)
    assert errors[-1] < 0.005


# ---------------------------------------------------------------------------
# Clamped--pinned
# ---------------------------------------------------------------------------

def test_clamped_pinned_first_three_frequencies_under_three_percent():
    L, E, I, rho, area = 1.0, 210e9, 6.4e-8, 7850.0, 3.2e-4
    result = clamped_pinned_beam_modes(L, elements=16, EI=E * I, rho_a=rho * area, modes=3)
    for m in range(1, 4):
        exact = clamped_pinned_exact_angular_frequency(L, E * I, rho * area, mode=m)
        error = relative_error(result.angular_frequencies[m - 1].item(), exact)
        assert error < 0.03, f"CP mode {m} error {error:.3%} exceeds 3%"


def test_clamped_pinned_convergence_monotone():
    L, EI, rho_a = 1.0, 1.0, 1.0
    exact = clamped_pinned_exact_angular_frequency(L, EI, rho_a)
    errors = []
    for n in (4, 8, 16, 32):
        result = clamped_pinned_beam_modes(L, elements=n, EI=EI, rho_a=rho_a, modes=1)
        errors.append(relative_error(result.angular_frequencies[0].item(), exact))
    assert all(e > 0 for e in errors)
    assert errors[-1] < 0.005


# ---------------------------------------------------------------------------
# Cross-boundary ordering sanity checks
# ---------------------------------------------------------------------------

def test_frequency_ordering_across_boundary_conditions():
    """For the same beam, clamped--clamped > clamped--pinned > simply supported > cantilever."""
    L, EI, rho_a = 1.0, 1.0, 1.0
    f_cc = clamped_clamped_exact_angular_frequency(L, EI, rho_a)
    f_cp = clamped_pinned_exact_angular_frequency(L, EI, rho_a)
    f_cf = cantilever_exact_angular_frequency(L, EI, rho_a)
    f_ss = simply_supported_exact_angular_frequency(L, EI, rho_a)
    # Clamped-clamped is stiffest, simply supported is most flexible.
    assert f_cc > f_cp > f_ss > f_cf


# ---------------------------------------------------------------------------
# Euler buckling (existing tests, kept for regression)
# ---------------------------------------------------------------------------

def test_euler_pinned_column_standard_benchmark_under_three_percent():
    L, E, I = 3.0, 200e9, 6.0e-6
    result = pinned_pinned_column_buckling(L, elements=8, EI=E * I, modes=3)
    exact = euler_pinned_critical_load(L, E * I)
    error = relative_error(result.load_factors[0].item(), exact)
    assert error < 0.03, f"Euler critical-load error {error:.3%} exceeds 3%"
    assert torch.all(result.load_factors > 0)


def test_first_three_euler_modes_converge_to_analytical_values():
    result = pinned_pinned_column_buckling(2.5, elements=16, EI=5.2e5, modes=3)
    exact = torch.tensor(
        [euler_pinned_critical_load(2.5, 5.2e5, i) for i in range(1, 4)],
        dtype=torch.float64,
    )
    assert torch.all(torch.abs(result.load_factors - exact) / exact < 0.03)
