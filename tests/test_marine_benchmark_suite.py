import math

import pytest

from tensorfem.marine_benchmark_suite import run_marine_benchmarks
from tensorfem.marine_hydrodynamics import AiryWave, MorisonMember, morison_base_actions
from tensorfem.marine_structures import solve_hull_girder_uniform_load


def test_all_marine_qualification_evidence_is_traceable_and_below_three_percent():
    evidence = run_marine_benchmarks()
    assert len(evidence) == 8
    assert len({item.id for item in evidence}) == len(evidence)
    assert all(item.source and item.reference != 0.0 for item in evidence)
    assert all(item.passed and item.error < item.tolerance <= 0.03 for item in evidence)


def test_hull_girder_mesh_sequence_converges_and_preserves_reactions():
    errors = []
    for elements in (4, 8, 16, 32):
        result = solve_hull_girder_uniform_load(
            length=120.0, young=2.1e11, area=5.0, inertia=180.0,
            still_water_load=1.4e6, wave_load=0.9e6, elements=elements,
        )
        errors.append(result.maximum_relative_error)
        expected = result.distributed_load * 120.0 / 2.0
        assert float(result.frame.reaction[1]) == pytest.approx(expected, rel=1e-10)
        assert float(result.frame.reaction[-2]) == pytest.approx(expected, rel=1e-10)
    # The consistent-load Euler--Bernoulli element reproduces this polynomial
    # benchmark to roundoff even on the coarsest mesh; do not demand monotonic
    # ordering of floating-point noise.
    assert max(errors) < 1e-8


def test_morison_quadrature_sequence_converges_to_high_order_result():
    wave = AiryWave(3.0, 9.0, 25.0)
    member = MorisonMember(1.2, 1.05, 2.0)
    time = -math.pi / (2.0 * wave.omega)
    reference = morison_base_actions(wave, member, 0.0, time,
                                     current_velocity=0.6, quadrature_order=128)
    errors = []
    for order in (4, 8, 16, 32):
        result = morison_base_actions(wave, member, 0.0, time,
                                     current_velocity=0.6, quadrature_order=order)
        errors.append(max(abs(result[key] / reference[key] - 1.0) for key in reference))
    assert errors[-1] <= errors[0]
    assert errors[-1] < 1e-8
