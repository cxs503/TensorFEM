import pytest
import torch
from tensorfem.finite_strain_contact3d import (
    _assemble_frictional,balanced_face_load,build_two_block_contact,
    initial_finite_strain_friction_state,solve_finite_strain_friction_path)


def path(model):
    loads=[balanced_face_load(model,normal=p) for p in (5.,10.,15.,20.,25.)]
    loads.extend(balanced_face_load(model,normal=25.,shear=s) for s in (1.,5.,15.))
    return loads,solve_finite_strain_friction_path(model,loads,
        tangential_penalty=2e4,friction=.3,tolerance=2e-9)


def test_finite_strain_symmetric_friction_stick_slip_energy_and_jacobian():
    model=build_two_block_contact();loads,steps=path(model)
    stick=steps[-3].contact.result;slide=steps[-1].contact.result
    assert float(stick.dissipation_increment)==0
    assert float(slide.dissipation_increment)>0
    assert torch.min(steps[-1].jacobian)>0
    expected=.5*(slide.forward.dissipation_increment+slide.reverse.dissipation_increment)
    assert torch.equal(slide.dissipation_increment,expected)
    assert torch.linalg.vector_norm(slide.slave_forces.sum(0)+slide.master_forces.sum(0))<1e-10
    assert steps[-1].residual_norm<1e-7


def test_combined_finite_strain_friction_tangent_and_rollback():
    model=build_two_block_contact();loads,steps=path(model)
    committed=steps[-2].state;trial=steps[-1].state;load=loads[-1]
    r,k,_,_,_=_assemble_frictional(model,trial,committed,load,
        tangential_penalty=2e4,friction=.3)
    direction=torch.zeros_like(trial.displacement);free=torch.ones_like(direction,dtype=torch.bool)
    free[model.solid.fixed_dofs]=False;direction[free]=torch.linspace(-.2,.3,int(free.sum()),dtype=direction.dtype)
    h=1e-7
    def residual(q):
        return _assemble_frictional(model,type(trial)(q,committed.contact),committed,load,
            tangential_penalty=2e4,friction=.3,tangent=False)[0]
    numerical=(residual(trial.displacement+h*direction)-residual(trial.displacement-h*direction))/(2*h)
    assert torch.linalg.vector_norm(numerical-k@direction)/torch.linalg.vector_norm(numerical)<2e-5
    initial=initial_finite_strain_friction_state(model);before=initial.displacement.clone()
    old=initial.contact.forward.points[0].elastic_slip.clone()
    with pytest.raises(RuntimeError,match="committed state unchanged"):
        solve_finite_strain_friction_path(model,[loads[-1]],tangential_penalty=2e4,
            friction=.3,initial_state=initial,max_iterations=1,tolerance=1e-14)
    assert torch.equal(initial.displacement,before)
    assert torch.equal(initial.contact.forward.points[0].elastic_slip,old)
