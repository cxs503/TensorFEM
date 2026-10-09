import torch
import pytest

from tensorfem.coupled_3d_demo import (
    Coupled3DState, coupled_3d_problem, coupled_3d_response,
    default_coupled_3d_model, monotonic_reference_displacement,
)
from tensorfem.nonlinear_step import NonlinearConvergenceError, solve_adaptive
from tensorfem.plasticity import Plastic1DState

D=torch.float64


def solve(model=None,target=1.):
    model=default_coupled_3d_model() if model is None else model
    problem,initial=coupled_3d_problem(model)
    return model,problem,solve_adaptive(problem,initial,target_factor=target,
        initial_increment=.1,maximum_increment=.2,tolerance=1e-11)


def test_geometric_plastic_contact_equilibrium_against_independent_reference():
    model,_,result=solve(); response=coupled_3d_response(model,result)
    exact=monotonic_reference_displacement(model,model.reference_force)
    error=abs(float(result.displacement[0])-exact)/exact
    assert error < .03
    assert error < 1e-9
    assert abs(float(response.residual))/model.reference_force < 1e-10
    assert result.material_state.plastic.alpha > 0
    assert result.material_state.contact.active
    # The inclined member rotates as well as stretches.
    initial_direction=(model.initial_free_point-model.fixed_point)
    final_direction=response.current_point-model.fixed_point
    cosine=torch.dot(initial_direction,final_direction)/(torch.linalg.vector_norm(initial_direction)*torch.linalg.vector_norm(final_direction))
    assert cosine < 1.


def test_increment_restart_is_identical_to_single_run():
    model,problem,full=solve()
    _,initial=coupled_3d_problem(model)
    half=solve_adaptive(problem,initial,target_factor=.5,initial_increment=.1,maximum_increment=.2,tolerance=1e-11)
    restarted=solve_adaptive(problem,half,target_factor=1.,initial_increment=.1,maximum_increment=.2,tolerance=1e-11)
    assert torch.allclose(restarted.displacement,full.displacement,rtol=1e-11,atol=1e-13)
    assert torch.allclose(restarted.material_state.plastic.plastic_strain,full.material_state.plastic.plastic_strain,rtol=1e-10,atol=1e-13)
    assert torch.allclose(restarted.material_state.contact.normal_multiplier,full.material_state.contact.normal_multiplier,rtol=1e-11,atol=1e-13)


def test_rejected_increment_rolls_back_both_histories():
    model=default_coupled_3d_model(); physical,initial=coupled_3d_problem(model)
    original=initial.clone(); injected={"done":False}
    def unreliable(u,factor,committed):
        if factor>.4 and not injected["done"]:
            injected["done"]=True
            corrupt=Coupled3DState(
                Plastic1DState(torch.tensor(999.,dtype=D),torch.tensor(999.,dtype=D)),
                type(committed.contact)(torch.tensor(999.,dtype=D),torch.ones(3,dtype=D)*999,
                    torch.tensor(999.,dtype=D),torch.ones(3,dtype=D)*999,99,True,False))
            return torch.ones(1,dtype=D),torch.zeros((1,1),dtype=D),corrupt
        return physical(u,factor,committed)
    result=solve_adaptive(unreliable,initial,initial_increment=1.,maximum_increment=1.,
                          minimum_increment=1e-5,max_iterations=5,tolerance=1e-10)
    assert any(x.retries for x in result.history)
    assert torch.equal(initial.displacement,original.displacement)
    assert torch.equal(initial.material_state.plastic.alpha,original.material_state.plastic.alpha)
    assert torch.equal(initial.material_state.contact.normal_multiplier,original.material_state.contact.normal_multiplier)
    exact=monotonic_reference_displacement(model,model.reference_force)
    assert abs(float(result.displacement[0])-exact)/exact < 1e-9


def test_failure_is_closed_and_input_checkpoint_remains_virgin():
    model=default_coupled_3d_model(); problem,initial=coupled_3d_problem(model)
    with pytest.raises(NonlinearConvergenceError):
        solve_adaptive(problem,initial,initial_increment=1.,minimum_increment=.6,
                       maximum_increment=1.,max_iterations=1)
    assert float(initial.displacement[0]) == 0.
    assert float(initial.material_state.plastic.alpha) == 0.
    assert not initial.material_state.contact.active
