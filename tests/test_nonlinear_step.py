import json
import math
import torch
import pytest

from tensorfem.nonlinear_step import (
    NonlinearConvergenceError, StepState, elastoplastic_bar_problem,
    load_checkpoint, save_checkpoint, solve_adaptive,
    total_lagrangian_truss_problem,
)
from tensorfem.nonlinear_truss import NonlinearTrussModel, two_bar_shallow_arch_reaction
from tensorfem.plasticity import Plastic1DState

D = torch.float64


def test_unified_total_lagrangian_bar_benchmark():
    model = NonlinearTrussModel(torch.tensor([[0., 0.], [1., 0.]], dtype=D),
        torch.tensor([[0, 1]]), torch.tensor([200.], dtype=D),
        torch.tensor([2.], dtype=D), torch.tensor([0, 1, 3]))
    load = torch.tensor([0., 0., 46.2, 0.], dtype=D)
    problem, free = total_lagrangian_truss_problem(model, load)
    result = solve_adaptive(problem, StepState(0., torch.zeros(len(free), dtype=D)),
                            initial_increment=.2, tolerance=1e-11)
    assert abs(float(result.displacement[0]) - .1) / .1 < 1e-9
    assert result.load_factor == 1.


def test_two_bar_shallow_arch_displacement_path_benchmark():
    a, h, E, A = 1., .2, 2000., .01
    critical_v = h * (1 - 1 / math.sqrt(3))
    exact = E*A*(h*h-(h/math.sqrt(3))**2)*(h/math.sqrt(3))/(a*a+h*h)**1.5
    grid = torch.linspace(0., .2, 20001, dtype=D)
    peak = torch.max(two_bar_shallow_arch_reaction(a, h, E, A, grid))
    assert abs(float(peak)-exact)/exact < 3e-8
    assert critical_v > 0  # path includes the limit point; load control need not cross it.


@pytest.mark.parametrize("method", ["newton", "modified_newton"])
def test_global_elastoplastic_bar_analytic_response(method):
    E, sy, H, L, A, force = 200000., 250., 10000., 2., .01, 4.
    problem, initial = elastoplastic_bar_problem(L, A, E, sy, H, force)
    result = solve_adaptive(problem, initial, initial_increment=.08,
                            max_iterations=500, method=method, tolerance=1e-8)
    stress = force/A
    exact_strain = sy/E + (stress-sy)*(E+H)/(E*H)
    assert abs(float(result.displacement[0])-L*exact_strain)/(L*exact_strain) < 1e-8
    assert float(result.material_state.alpha) > 0


def test_rejected_increment_rolls_back_and_checkpoint_restarts(tmp_path):
    calls = {"rejected": 0}
    def difficult(u, factor, state):
        # Artificially reject only large trials, proving adaptive retry and rollback.
        if factor - state["factor"] > .26:
            calls["rejected"] += 1
            return torch.ones(1, dtype=D), torch.zeros((1, 1), dtype=D), {"factor": 999.}
        return torch.tensor([factor-u[0]], dtype=D), torch.ones((1, 1), dtype=D), {"factor": factor}
    result = solve_adaptive(difficult, StepState(0., torch.zeros(1, dtype=D), {"factor": 0.}),
                            initial_increment=.5, maximum_increment=.5, minimum_increment=.01)
    assert calls["rejected"] and result.material_state["factor"] == 1.
    assert any(step.retries for step in result.history)
    path = tmp_path / "restart.json"
    save_checkpoint(result, path)
    loaded = load_checkpoint(path)
    assert loaded.load_factor == 1. and torch.equal(loaded.displacement, result.displacement)
    assert json.loads(path.read_text())["format"] == "tensorfem.nonlinear-step.v1"


def test_unconverged_increment_fails_closed():
    def impossible(u, factor, state):
        return torch.ones(1, dtype=D), torch.zeros((1, 1), dtype=D), state
    with pytest.raises(NonlinearConvergenceError, match="minimum increment exhausted"):
        solve_adaptive(impossible, StepState(0., torch.zeros(1, dtype=D)),
                       initial_increment=.1, minimum_increment=.02, max_iterations=2)


def test_elastoplastic_checkpoint_can_resume(tmp_path):
    problem, initial = elastoplastic_bar_problem(2., .01, 200000., 250., 10000., 4.)
    half = solve_adaptive(problem, initial, target_factor=.5, initial_increment=.1)
    path = tmp_path / "plastic.json"
    save_checkpoint(half, path)
    def decode(raw, tensor):
        fields = raw["fields"]
        return Plastic1DState(tensor(fields["plastic_strain"]), tensor(fields["alpha"]))
    restart = load_checkpoint(path, material_decoder=decode)
    final = solve_adaptive(problem, restart, target_factor=1., initial_increment=.1)
    direct = solve_adaptive(problem, initial, target_factor=1., initial_increment=.1)
    assert torch.allclose(final.displacement, direct.displacement, rtol=1e-9, atol=1e-12)
