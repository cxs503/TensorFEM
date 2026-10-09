"""Independent resultant, rigid-rotation and work-conjugacy exchange checks."""
import json
from pathlib import Path
import pytest
import torch
from tensorfem.surface_coupling import QuadSurfaceExchange,PlanarEmbedding,validate_physics_ownership
from tensorfem.coupled_hull import LinearHullReceiver

D=torch.float64
ROOT=Path(__file__).resolve().parents[1]

def rectangle(offset=.1):
    return QuadSurfaceExchange([[0.,0.,0.],[2.,0.,0.],[2.,1.,0.],[0.,1.,0.]],[[0,1,2,3]],face_offset_m=offset)

def test_uniform_traction_matches_exact_area_centroid_and_offset_moment():
    s=rectangle();t=torch.tensor([[3.,4.,5.]],dtype=D).expand(4,3)
    load=s.tractions(t,time_s=.2,source='TensorLBM')
    audit=s.audit(load,torch.zeros((4,6),dtype=D),expected_time_s=.2)
    f=torch.tensor([6.,8.,10.],dtype=D);center=torch.tensor([1.,.5,.1],dtype=D)
    torch.testing.assert_close(torch.tensor(audit['force_N'],dtype=D),f)
    torch.testing.assert_close(torch.tensor(audit['moment_N_m'],dtype=D),torch.linalg.cross(center,f))

def test_offset_velocity_rigid_rotation_and_transpose_virtual_power():
    s=rectangle();omega=torch.tensor([.2,-.3,.4],dtype=D);translation=torch.tensor([.5,.1,-.2],dtype=D)
    v=torch.cat((translation+torch.linalg.cross(omega.expand_as(s.nodes),s.nodes),omega.expand_as(s.nodes)),1)
    torch.testing.assert_close(s.velocities(v),translation+torch.linalg.cross(omega.expand_as(s.positions),s.positions))
    load=s.point_loads(torch.arange(12,dtype=D).reshape(4,3),time_s=0.,source='TensorDEM')
    audit=s.audit(load,v,expected_time_s=0.)
    assert audit['power_transfer_error_W']<1e-12
    assert audit['moment_transfer_error_N_m']<1e-12

def test_rotated_surface_traction_and_wrench_are_covariant():
    s=rectangle();R=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]],dtype=D)
    b=QuadSurfaceExchange(s.nodes@R.T,s.elements,face_offset_m=.1)
    f=torch.arange(12,dtype=D).reshape(4,3)
    a=s.audit(s.point_loads(f,time_s=0.,source='verification'),torch.zeros((4,6)),expected_time_s=0.)
    c=b.audit(b.point_loads(f@R.T,time_s=0.,source='verification'),torch.zeros((4,6)),expected_time_s=0.)
    for key in ('force_N','moment_N_m'):torch.testing.assert_close(torch.tensor(c[key],dtype=D),R@torch.tensor(a[key],dtype=D))

@pytest.mark.parametrize('bad',('time','geometry','units','nan','mutated','folded'))
def test_invalid_exchange_rejected(bad):
    s=rectangle();load=s.point_loads(torch.ones((4,3)),time_s=0.,source='verification')
    with pytest.raises(ValueError):
        if bad=='time':s.validate_load(load,expected_time_s=.1)
        elif bad=='geometry':rectangle(.2).validate_load(load,expected_time_s=0.)
        elif bad=='units':s.tractions(torch.ones((4,3)),time_s=0.,source='TensorLBM',units='N')
        elif bad=='nan':s.pressure(torch.full((4,),float('nan')),time_s=0.,source='TensorLBM')
        elif bad=='mutated':s.positions[0,0]+=1.;s.validate_load(load,expected_time_s=0.)
        else:QuadSurfaceExchange([[0.,0.,0.],[1.,1.,0.],[0.,1.,0.],[1.,0.,0.]],[[0,1,2,3]])

def test_no_duplicate_ice_or_water_physics():
    validate_physics_ownership(ice_owners=['TensorDEM'],resolved_fluid=True,simplified_water_terms=[])
    with pytest.raises(ValueError):validate_physics_ownership(ice_owners=['TensorDEM','TensorFEM-reference'],resolved_fluid=False,simplified_water_terms=[])
    with pytest.raises(ValueError):validate_physics_ownership(ice_owners=['TensorDEM'],resolved_fluid=True,simplified_water_terms=['drag'])

def test_explicit_planar_embedding_keeps_force_and_power_and_rejects_lost_component():
    e=PlanarEmbedding([0.,0.,.5],[[1.,0.],[0.,0.],[0.,1.]])
    xy=torch.tensor([[1.,2.],[3.,4.]],dtype=D);f=torch.tensor([[2.,-1.],[5.,.5]],dtype=D)
    assert e.positions(xy).tolist()==[[1.,0.,2.5],[3.,0.,4.5]]
    torch.testing.assert_close(e.project_load(e.vectors(f)),f)
    torch.testing.assert_close((e.vectors(f)*e.vectors(xy)).sum(),(f*xy).sum())
    with pytest.raises(ValueError):e.project_load([[0.,1.,0.]])
    with pytest.raises(ValueError):PlanarEmbedding([0.,0.,0.],[[1.,1.],[0.,0.],[0.,0.]])

def test_vehicle_only_receiver_impulse_and_exact_restart():
    g=json.loads((ROOT/'docs/assets/suboff-ice-v2/appended-16.json').read_text());body=LinearHullReceiver(g)
    dt=.5*body.dt_bound
    def advance(b):
        s=b.surface();t=torch.zeros_like(s.positions);t[:,2]=100.
        b.advance(s,s.tractions(t,time_s=b.time_s,source='verification'),dt)
    for _ in range(2):advance(body)
    snapshot=json.loads(json.dumps(body.snapshot()));other=LinearHullReceiver(g);other.restore(snapshot)
    for _ in range(2):advance(body);advance(other)
    assert torch.equal(body.q,other.q) and torch.equal(body.v,other.v)
    d=body.diagnostics();assert d['ice_elements']==0 and d['maximum_gauss_von_mises_Pa']>0
    assert d['momentum_residual_kg_m_s']<1e-9
    with pytest.raises(ValueError):
        snapshot['geometry_sha256']='0'*64;other.restore(snapshot)
