import math

import torch

from tensorfem.buckling import euler_pinned_critical_load, pinned_pinned_column_buckling
from tensorfem.modal import cantilever_beam_modes, cantilever_exact_angular_frequency


def relative_error(value: float, reference: float) -> float:
    return abs(value - reference) / abs(reference)


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
