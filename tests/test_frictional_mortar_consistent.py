import math
import pytest
import torch

from tensorfem.frictional_mortar import (
    assemble_frictional_mortar, initial_frictional_mortar_state)

D=torch.float64


def surfaces():
    slave=torch.tensor([[0.,0.,-.001],[1.,0.,-.001],[1.,1.,-.001],[0.,1.,-.001]],dtype=D)
    master=torch.tensor([[0.,0.,0.],[.5,0.,0.],[1.,0.,0.],[0.,1.,0.],[.5,1.,0.],[1.,1.,0.]],dtype=D)
    sf=torch.tensor([[0,1,2,3]],dtype=torch.long)
    mf=torch.tensor([[0,1,4,3],[1,2,5,4]],dtype=torch.long)
    return slave,sf,master,mf


def test_nonmatching_frictional_tangent_matches_directional_difference():
    slave,sf,master,mf=surfaces(); state=initial_frictional_mortar_state(slave,sf,master,mf)
    us=torch.zeros_like(slave);us[:,0]=.01;um=torch.zeros_like(master)
    kwargs=dict(normal_penalty=1e5,tangential_penalty=1e4,friction=.3)
    a=assemble_frictional_mortar(slave,sf,master,mf,us,um,state,**kwargs)
    direction=torch.zeros(a.residual.numel(),dtype=D);direction[:slave.numel()].reshape_as(slave)[:,0]=1
    h=1e-7
    plus=assemble_frictional_mortar(slave,sf,master,mf,
        us+h*direction[:slave.numel()].reshape_as(slave),um,state,tangent=False,**kwargs).residual
    minus=assemble_frictional_mortar(slave,sf,master,mf,
        us-h*direction[:slave.numel()].reshape_as(slave),um,state,tangent=False,**kwargs).residual
    numerical=(plus-minus)/(2*h)
    assert torch.allclose(a.tangent@direction,numerical,rtol=2e-5,atol=2e-5)
    assert a.result.active_points==4


def test_relative_motion_objectivity_balance_and_trial_rollback():
    slave,sf,master,mf=surfaces(); committed=initial_frictional_mortar_state(slave,sf,master,mf)
    us=torch.zeros_like(slave);um=torch.zeros_like(master)
    common=torch.tensor([.2,-.3,-.002],dtype=D);us[:]=common;um[:]=common
    a=assemble_frictional_mortar(slave,sf,master,mf,us,um,committed,
        normal_penalty=1e5,tangential_penalty=1e4,friction=.3,tangent=False)
    assert torch.linalg.vector_norm(a.result.tangential_resultant)<1e-11
    assert torch.linalg.vector_norm(a.result.slave_forces.sum(0)+a.result.master_forces.sum(0))<1e-11
    points=torch.cat((slave+us,master+um));forces=torch.cat((a.result.slave_forces,a.result.master_forces))
    assert torch.linalg.vector_norm(torch.linalg.cross(points,forces).sum(0))<1e-10
    assert all(float(p.dissipated_energy_density)==0 for p in committed.points)
    assert all(torch.equal(p.elastic_slip,torch.zeros(3,dtype=D)) for p in committed.points)


def test_finite_rigid_rotation_is_objective_and_topology_fails_closed():
    slave,sf,master,mf=surfaces(); state=initial_frictional_mortar_state(slave,sf,master,mf)
    base=assemble_frictional_mortar(slave,sf,master,mf,torch.zeros_like(slave),
        torch.zeros_like(master),state,normal_penalty=1e5,tangential_penalty=1e4,
        friction=.3,tangent=False)
    angle=.47;c,s=math.cos(angle),math.sin(angle)
    rotation=torch.tensor([[c,-s,0.],[s,c,0.],[0.,0.,1.]],dtype=D)
    us=slave@rotation.T-slave;um=master@rotation.T-master
    moved=assemble_frictional_mortar(slave,sf,master,mf,us,um,state,
        normal_penalty=1e5,tangential_penalty=1e4,friction=.3,tangent=False)
    assert torch.linalg.vector_norm(moved.result.tangential_resultant)<1e-10
    assert torch.allclose(moved.result.slave_forces,base.result.slave_forces@rotation.T,
                          atol=2e-10,rtol=2e-10)
    with pytest.raises(ValueError,match="topology mismatch"):
        assemble_frictional_mortar(slave,torch.flip(sf,[1]),master,mf,us,um,state,
            normal_penalty=1e5,tangential_penalty=1e4,friction=.3,tangent=False)


def test_assembled_trial_result_and_nested_history_are_fully_detached():
    slave,sf,master,mf=surfaces();state=initial_frictional_mortar_state(slave,sf,master,mf)
    us=torch.zeros_like(slave);us[:,0]=.01
    assembly=assemble_frictional_mortar(slave,sf,master,mf,us,torch.zeros_like(master),state,
        normal_penalty=1e5,tangential_penalty=1e4,friction=.3)
    tensors=[assembly.residual,assembly.tangent,assembly.result.slave_forces,
        assembly.result.master_forces,assembly.result.normal_resultant,
        assembly.result.tangential_resultant,assembly.result.stored_energy,
        assembly.result.dissipation_increment]
    for point in assembly.result.state.points:
        tensors.extend([point.elastic_slip,point.dissipated_energy_density,
            point.projection,point.slave_point,point.master_weights])
    assert all(value is not None and not value.requires_grad for value in tensors)
    assert all(value.grad_fn is None for value in tensors)
