import math

import pytest
import torch

from tensorfem.corotational_shell import (axis_angle,
    corotational_cylindrical_shell4, cylindrical_nodes)

D=torch.float64


def _param(span=.4):
    return torch.tensor([[0.,-span/2],[2.,-span/2],[2.,span/2],[0.,span/2]],dtype=D)


@pytest.mark.parametrize("degrees",[5.,30.,90.,150.,179.])
def test_finite_rigid_motion_has_machine_zero_energy(degrees):
    p=_param(); X=cylindrical_nodes(p,5.)
    S=axis_angle(torch.tensor([.3,-.7,.2],dtype=D),math.radians(degrees))
    current=X@S.T+torch.tensor([13.,-8.,4.],dtype=D)
    state=corotational_cylindrical_shell4(p,current,S.expand(4,3,3).clone(),5.,70e9,.25,.03)
    scale=70e9*.03*2.*5.*.4
    assert abs(float(state.energy))/scale < 1e-27
    assert float(torch.linalg.vector_norm(state.generalized_deformation)) < 2e-14


def test_superposed_finite_rotation_preserves_deformation_energy():
    p=_param(); X=cylindrical_nodes(p,5.)
    deformation=torch.tensor([[0.,0.,0.],[.001,0.,.002],[.001,-.0005,.002],[0.,-.0005,0.]],dtype=D)
    local_rot=torch.stack([axis_angle(torch.tensor([1.,.2,-.1],dtype=D),a)
                           for a in (.002,.003,.004,.001)])
    base=corotational_cylindrical_shell4(p,X+deformation,local_rot,5.,70e9,.25,.03)
    S=axis_angle(torch.tensor([-.2,.4,1.],dtype=D),1.3)
    transformed=(X+deformation)@S.T+torch.tensor([-4.,3.,11.],dtype=D)
    rotations=S.unsqueeze(0)@local_rot
    moved=corotational_cylindrical_shell4(p,transformed,rotations,5.,70e9,.25,.03)
    assert abs(float(moved.energy/base.energy)-1.) < 2e-11
    assert torch.allclose(moved.generalized_deformation,base.generalized_deformation,atol=2e-12,rtol=2e-11)


def test_small_deformation_energy_is_positive_and_differentiable():
    p=_param(); X=cylindrical_nodes(p,5.); current=X.clone(); current[1:3,2]+=.001
    current.requires_grad_()
    rotations=torch.eye(3,dtype=D).expand(4,3,3).clone()
    state=corotational_cylindrical_shell4(p,current,rotations,5.,70e9,.25,.03)
    assert float(state.energy.detach()) > 0 and torch.isfinite(state.energy)
    state.energy.backward()
    assert torch.all(torch.isfinite(current.grad))
    # Translational objectivity implies self-equilibrated energy gradients.
    assert float(torch.linalg.vector_norm(current.grad.sum(0)).detach()) < 1e-7


def test_invalid_rotation_fails_closed():
    p=_param(); X=cylindrical_nodes(p,5.); rotations=torch.eye(3,dtype=D).expand(4,3,3).clone()
    rotations[2,0,0]=1.01
    with pytest.raises(ValueError,match="proper orthogonal"):
        corotational_cylindrical_shell4(p,X,rotations,5.,70e9,.25,.03)


def test_improper_rotation_fails_closed():
    p=_param(); X=cylindrical_nodes(p,5.); rotations=torch.eye(3,dtype=D).expand(4,3,3).clone()
    rotations[1,2,2]=-1.
    with pytest.raises(ValueError,match="proper orthogonal"):
        corotational_cylindrical_shell4(p,X,rotations,5.,70e9,.25,.03)
