import math

import pytest

from tensorfem.hull_girder_ultimate import (
    rectangular_section_oracle, run_hull_girder_ultimate_benchmark,
    solve_rectangular_hull_girder,
)


def test_ultimate_benchmark_passes_independent_oracle_and_state_gates():
    report = run_hull_girder_ultimate_benchmark()
    assert report["passed"]
    assert max(report["errors"].values()) < report["tolerance"] == 0.03
    assert all(report["checks"].values())
    assert "no TensorLBM" in report["scope"]
    assert "Not complete hull progressive collapse" in report["limitations"]


def test_oracle_has_elastic_yield_and_plastic_limits():
    b, h, e, sy = 0.8, 2.4, 210e9, 355e6
    ky = 2 * sy / (e * h)
    my, mp = b * sy * h**2 / 6, b * sy * h**2 / 4
    assert rectangular_section_oracle(width=b, depth=h, young=e, yield_stress=sy,
                                      curvature=ky) == pytest.approx(my)
    assert rectangular_section_oracle(width=b, depth=h, young=e, yield_stress=sy,
                                      curvature=1e6 * ky) == pytest.approx(mp, rel=1e-12)


def test_fiber_curve_converges_and_reports_physical_history():
    args = dict(width=1.0, depth=2.0, young=200e9, yield_stress=300e6,
                curvatures=[0.0, 0.0015, 0.003, 0.006, 0.03])
    coarse = solve_rectangular_hull_girder(**args, fibers=20)
    fine = solve_rectangular_hull_girder(**args, fibers=400)
    ref = rectangular_section_oracle(width=1, depth=2, young=200e9,
                                     yield_stress=300e6, curvature=0.006)
    assert abs(fine.points[3].moment - ref) < abs(coarse.points[3].moment - ref)
    assert all(math.isfinite(p.strain_energy) and p.strain_energy >= 0 for p in fine.points)
    assert fine.points[-1].yielded_fraction > fine.points[2].yielded_fraction


@pytest.mark.parametrize("kwargs,exc", [
    ({"width": 0.0}, ValueError), ({"depth": float("nan")}, ValueError),
    ({"young": True}, TypeError), ({"yield_stress": -1.0}, ValueError),
    ({"fibers": 5}, ValueError), ({"curvatures": [0.0, 0.0]}, ValueError),
    ({"curvatures": [0.0, float("inf")]}, ValueError),
])
def test_strict_inputs(kwargs, exc):
    base = dict(width=1.0, depth=2.0, young=200e9, yield_stress=300e6,
                curvatures=[0.0, 0.001], fibers=20)
    base.update(kwargs)
    with pytest.raises(exc):
        solve_rectangular_hull_girder(**base)
