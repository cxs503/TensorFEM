from itertools import pairwise

import pytest

from tensorfem.hull_girder_progressive import (
    HullComponent,
    run_progressive_hull_girder_benchmark,
    solve_component_section,
    symmetric_component_oracle,
)


def test_qualification_passes_oracle_three_percent_and_physics_gates():
    report = run_progressive_hull_girder_benchmark()
    assert report["passed"]
    assert max(report["errors"].values()) < report["tolerance"] == 0.03
    assert all(report["checks"].values())
    assert "not a complete class-rule" in report["limitations"]
    assert "no TensorLBM" in report["scope"]


def test_closed_form_oracle_is_independent_hand_sum():
    e, sy = 200e9, 300e6
    cs = tuple(HullComponent(f"c{i}", 0.01, y, e, sy) for i, y in enumerate((-1.0, -0.5, 0.5, 1.0)))
    k = sy / e
    expected = 2 * 0.01 * sy * 1.0 + 2 * 0.01 * (e * k * 0.5) * 0.5
    assert symmetric_component_oracle(cs, k) == pytest.approx(expected)
    points = solve_component_section(cs, [0, k])
    assert points[-1].moment == pytest.approx(expected, rel=1e-12)


def test_buckling_reduction_changes_path_and_is_auditable():
    e, sy = 200e9, 300e6
    base = [
        HullComponent("deck", 0.02, 1.0, e, sy),
        HullComponent("bottom", 0.02, -1.0, e, sy),
        HullComponent("upper_side", 0.01, 0.5, e, sy),
        HullComponent("lower_side", 0.01, -0.5, e, sy),
    ]
    reduced = list(base)
    reduced[1] = HullComponent("bottom", 0.02, -1.0, e, sy, 0.55 * sy)
    ks = [0, 0.0005, 0.0015, 0.003, 0.006]
    intact = solve_component_section(base, ks)
    buckled = solve_component_section(reduced, ks)
    assert buckled[-1].moment < intact[-1].moment
    assert any(s.mode == "buckling_limited" for s in buckled[-1].states)
    assert abs(buckled[-1].axial_residual) < 1e-8 * sum(c.area * sy for c in reduced)
    assert all(b.work >= a.work for a, b in pairwise(buckled))


@pytest.mark.parametrize(
    "action,exc",
    [
        (lambda: HullComponent("", 1, 0, 1, 1), ValueError),
        (lambda: HullComponent("x", True, 0, 1, 1), TypeError),
        (lambda: HullComponent("x", 1, float("nan"), 1, 1), ValueError),
        (lambda: HullComponent("x", 1, 0, 1, 1, 2), ValueError),
    ],
)
def test_component_inputs_are_strict(action, exc):
    with pytest.raises(exc):
        action()


def test_solver_inputs_are_strict():
    c = HullComponent("x", 1, -1, 1, 1)
    d = HullComponent("y", 1, 1, 1, 1)
    with pytest.raises(ValueError, match="strictly increasing"):
        solve_component_section([c, d], [0, 0])
    with pytest.raises(ValueError, match="unique"):
        solve_component_section([c, HullComponent("x", 1, 1, 1, 1)], [0, 1])
    with pytest.raises(TypeError):
        solve_component_section([c, object()], [0, 1])
