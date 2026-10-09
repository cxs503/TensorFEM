import pytest
import torch

from tensorfem.arc_length import ArcLengthProblem, solve_arc_length
from tensorfem.branch_switch import (
    perturbed_arc_predictor,
    smallest_eligible_symmetric_mode,
)


D = torch.float64


def test_critical_mode_excludes_shell_drilling_gauge():
    # DOF 0 is an artificial drilling gauge; DOFs 1/2 are translations.
    stiffness = torch.diag(torch.tensor([0.0, 2.0e-5, 4.0], dtype=D))
    mode = smallest_eligible_symmetric_mode(
        stiffness, eligible=torch.tensor([False, True, True]),
    )
    assert mode.eigenvalue == pytest.approx(2.0e-5)
    assert mode.relative_eigenvalue == pytest.approx(5.0e-6)
    assert torch.argmax(mode.vector.abs()).item() == 1


def test_predictor_perturbation_preserves_arc_radius_and_selects_sign():
    tangent = torch.tensor([0.0, 1.0], dtype=D)
    mode = torch.tensor([1.0, 0.0], dtype=D)
    positive = perturbed_arc_predictor(tangent, mode, mode_fraction=.8, sign=1)
    negative = perturbed_arc_predictor(tangent, mode, mode_fraction=.8, sign=-1)
    assert float(torch.linalg.vector_norm(positive)) == pytest.approx(1.0)
    assert float(positive[0]) == pytest.approx(float(-negative[0]))
    assert float(positive[1]) == pytest.approx(float(negative[1]))


def test_pitchfork_branch_switch_selects_both_nontrivial_branches():
    # R(u, lambda)=u^3-lambda*u has a trivial path and two branches
    # lambda=u^2.  At the origin K=R_lambda=0, so an ordinary load predictor
    # cannot select either nontrivial branch.
    def residual_tangent(u, load):
        return u**3 - load*u, torch.diag(3*u**2-load), -u

    problem = ArcLengthProblem(
        lambda u: (u**3, torch.diag(3*u**2)),
        torch.zeros(1, dtype=D), residual_tangent,
    )
    terminals = []
    for sign in (-1, 1):
        trace = []
        result = solve_arc_length(
            problem, torch.zeros(1, dtype=D), steps=3, step_size=.05,
            load_scale=1.0, tolerance=1e-11, max_iterations=25,
            branch_switch="critical_mode", branch_mode_fraction=.8,
            branch_sign=sign, critical_eigenvalue_ratio=.01,
            diagnostics=trace,
        )
        assert result.converged
        assert trace[0]["critical_mode"]["activated"] is True
        for point in result.points:
            u = float(point.displacement[0])
            assert point.load_factor == pytest.approx(u*u, rel=2e-8, abs=2e-11)
        terminals.append(float(result.points[-1].displacement[0]))
    assert terminals[0] < 0 < terminals[1]
    assert terminals[0] == pytest.approx(-terminals[1], rel=2e-8)


def test_default_arc_path_remains_unmodified_at_exact_pitchfork():
    def residual_tangent(u, load):
        return u**3-load*u, torch.diag(3*u**2-load), -u
    problem = ArcLengthProblem(lambda u: (u**3, torch.diag(3*u**2)),
                               torch.zeros(1, dtype=D), residual_tangent)
    result = solve_arc_length(problem, torch.zeros(1, dtype=D), steps=1,
                              step_size=.05, load_scale=1.0)
    assert not result.converged and result.points == ()
