import pytest
import torch
from tensorfem.finite_strain_contact3d import (
    balanced_face_load,build_two_block_contact,solve_finite_strain_contact_path)


def test_two_hyperelastic_bodies_contact_in_one_newton_path():
    model=build_two_block_contact()
    loads=[balanced_face_load(model,normal=p,shear=0.) for p in (5.,10.,15.,20.,25.)]
    loads.extend(balanced_face_load(model,normal=25.,shear=s) for s in (5.,10.,15.))
    steps=solve_finite_strain_contact_path(model,loads,tolerance=2e-9)
    last=steps[-1];force=torch.cat((last.contact.slave_forces,last.contact.master_forces))
    assert last.residual_norm<1e-7
    assert torch.linalg.vector_norm(force.sum(0))<1e-10
    assert torch.min(last.jacobian)>0
    assert torch.sum(last.strain_energy)>0 and last.contact.penalty_energy>0
    us=last.displacement.reshape(-1,3)[model.slave_nodes]
    um=last.displacement.reshape(-1,3)[model.master_nodes]
    assert torch.mean(us[:,2])<0 and torch.mean(um[:,2])>0
    assert torch.mean(us[:,0]-um[:,0])>.01


def test_failed_finite_strain_contact_increment_rolls_back():
    model=build_two_block_contact();initial=torch.zeros(model.solid.n_dofs,dtype=torch.float64)
    before=initial.clone()
    with pytest.raises(RuntimeError,match="committed displacement unchanged"):
        solve_finite_strain_contact_path(model,[balanced_face_load(model,normal=40.,shear=15.)],
            initial_displacement=initial,max_iterations=1,tolerance=1e-14)
    assert torch.equal(initial,before)
