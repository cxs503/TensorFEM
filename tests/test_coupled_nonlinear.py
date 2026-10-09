import torch
import pytest

from tensorfem.coupled_nonlinear import (
    PlasticBarContact, coupled_bar_contact_problem, coupled_response,
    monotonic_closed_form,
)
from tensorfem.nonlinear_step import NonlinearConvergenceError, solve_adaptive

D = torch.float64


def benchmark():
    return PlasticBarContact(2.0, .01, 200000.0, 250.0, 10000.0,
                             .004, 500.0, 8.0)


def test_plastic_bar_and_contact_share_one_global_equilibrium():
    model = benchmark()
    problem, initial = coupled_bar_contact_problem(model, dtype=D)
    result = solve_adaptive(problem, initial, initial_increment=.1,
                            maximum_increment=.2, tolerance=1e-11)
    exact = monotonic_closed_form(model, model.reference_force)
    error = abs(float(result.displacement[0]) - exact) / exact
    response = coupled_response(model, result.displacement, result.material_state)
    assert error < .03
    assert error < 1e-10
    assert response.active and float(result.material_state.alpha) > 0
    equilibrium = abs(float(response.bar_force + response.contact_force)
                      - model.reference_force) / model.reference_force
    assert equilibrium < 1e-10


def test_automatic_retry_crosses_yield_and_contact_kinks_transactionally():
    model = benchmark()
    physical, initial = coupled_bar_contact_problem(model, dtype=D)
    virgin_plastic_strain = initial.material_state.plastic_strain.clone()
    injected = {"done": False}
    def problem(u, factor, committed):
        # A singular trial emulates a failed constitutive/contact iteration and
        # returns deliberately corrupt trial history. The retry must discard it.
        if factor > .5 and not injected["done"]:
            injected["done"] = True
            corrupt = type(committed)(torch.tensor(999., dtype=D),
                                      torch.tensor(999., dtype=D))
            return torch.ones(1, dtype=D), torch.zeros((1, 1), dtype=D), corrupt
        return physical(u, factor, committed)
    result = solve_adaptive(problem, initial, initial_increment=1.0,
                            maximum_increment=1.0, minimum_increment=1e-4,
                            max_iterations=3, tolerance=1e-10)
    assert any(record.retries > 0 for record in result.history)
    assert torch.equal(initial.material_state.plastic_strain, virgin_plastic_strain)
    assert float(initial.material_state.alpha) == 0.0
    assert result.load_factor == 1.0


def test_coupled_problem_fails_closed_without_committing_trial_state():
    model = benchmark()
    problem, initial = coupled_bar_contact_problem(model, dtype=D)
    with pytest.raises(NonlinearConvergenceError):
        solve_adaptive(problem, initial, initial_increment=1.0,
                       minimum_increment=.6, maximum_increment=1.0,
                       max_iterations=1)
    assert float(initial.material_state.alpha) == 0.0
    assert torch.equal(initial.displacement, torch.zeros(1, dtype=D))


def test_closed_form_covers_elastic_plastic_and_contact_branches():
    model = benchmark()
    for force in (1.0, 3.0, 8.0):
        problem, initial = coupled_bar_contact_problem(model, dtype=D)
        scaled = PlasticBarContact(**{**model.__dict__, "reference_force": force})
        problem, initial = coupled_bar_contact_problem(scaled, dtype=D)
        result = solve_adaptive(problem, initial, initial_increment=.1)
        exact = monotonic_closed_form(scaled, force)
        assert abs(float(result.displacement[0]) - exact) / max(exact, 1e-15) < 1e-9
