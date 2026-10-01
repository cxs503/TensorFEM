import pytest
import torch

from tensorfem.solid3d import structured_hex_mesh
from tensorfem.solid3d_benchmarks import (
    assemble_hex8_consistent_mass,
    distorted_body_force_bar_benchmark,
    longitudinal_bar_frequency_benchmark,
)


def test_consistent_mass_conserves_total_mass_and_is_symmetric():
    nodes, elements = structured_hex_mesh(2.0, 0.5, 0.25, 3, 2, 1)
    density = 7850.0
    mass = assemble_hex8_consistent_mass(nodes, elements, density)
    expected = density * 2.0 * 0.5 * 0.25
    for component in range(3):
        assert float(mass[component::3, component::3].sum()) == pytest.approx(expected, rel=1e-12)
    assert torch.allclose(mass, mass.T, rtol=0, atol=1e-12)
    assert torch.linalg.eigvalsh(mass).min() > 0


def test_distorted_body_force_bar_converges_and_passes_three_percent_gate():
    errors = [distorted_body_force_bar_benchmark(nx)[2] for nx in (2, 4, 6)]
    assert errors[-1] < errors[0]
    assert errors[-1] < 0.03


def test_hex8_longitudinal_frequency_passes_three_percent_gate():
    computed, reference, error = longitudinal_bar_frequency_benchmark(6)
    assert computed > 0 and reference > 0
    assert error < 0.03
