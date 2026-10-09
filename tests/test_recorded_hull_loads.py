import json
import math
import pytest
import torch
from tensorfem.recorded_hull_loads import embed_wrench,distribute_wrench,LinearMidpointIntegrator
from tensorfem.surface_coupling import PlanarEmbedding,QuadSurfaceExchange
D=torch.float64

def test_axial_moment_embedding_and_origin_shift():
    e=PlanarEmbedding([2.,0.,1.],[[1.,0.],[0.,0.],[0.,1.]])
    r={f'total_{key}':value for key,value in zip(('fx_n','fy_n','fz_n','mx_nm','my_nm','mz_nm'),(3.,4.,0.,0.,0.,5.))}
    torch.testing.assert_close(embed_wrench(r,'total',e),torch.tensor([3.,0.,4.,0.,-10.,0.],dtype=D))
    r['total_fz_n']=1.
    with pytest.raises(ValueError):embed_wrench(r,'total',e)


def test_arbitrary_wrench_mapping_and_rigid_work():
    s=QuadSurfaceExchange([[0.,0.,0.],[2.,0.,0.],[2.,1.,0.],[0.,1.,0.]],[[0,1,2,3]],face_offset_m=.1)
    w=torch.tensor([2.,3.,4.,5.,6.,7.],dtype=D)
    f=distribute_wrench(s,w,torch.ones(4,dtype=torch.bool))
    load=s.point_loads(f,time_s=0.,source='TensorLBM')
    a=s.audit(load,torch.randn((4,6),dtype=D),expected_time_s=0.)
    torch.testing.assert_close(torch.tensor(a['force_N']+a['moment_N_m'],dtype=D),w)
    assert a['power_transfer_error_W']<1e-12
    with pytest.raises(ValueError):distribute_wrench(s,w,torch.zeros(4,dtype=torch.bool))


def test_midpoint_oscillator_energy_accuracy_and_restart():
    errors=[]
    for dt in (.02,.01):
        model=LinearMidpointIntegrator([2.],torch.tensor([[8.]],dtype=D),dt)
        for _ in range(round(.5/dt)):model.step([3.])
        exact=3/8*(1-math.cos(2*.5));errors.append(abs(float(model.q[0])-exact))
        assert abs(model.energy()-model.work_J)<1e-13
        other=LinearMidpointIntegrator([2.],torch.tensor([[8.]],dtype=D),dt)
        other.restore(json.loads(json.dumps(model.snapshot())))
        model.step([2.]);other.step([2.])
        assert model.snapshot()==other.snapshot()
    assert errors[0]/errors[1]>3.9


def test_free_particle_impulse_and_invalid_model():
    model=LinearMidpointIntegrator([2.],torch.zeros((1,1),dtype=D),.1)
    for _ in range(10):model.step([4.])
    assert abs(float(model.v[0])-2.)<1e-14
    assert abs(float(model.q[0])-1.)<1e-14
    with pytest.raises(ValueError):model.step([float('nan')])
    with pytest.raises(ValueError):LinearMidpointIntegrator([-1.],torch.zeros((1,1)),.1)


def test_restart_rejects_same_size_different_material():
    a=LinearMidpointIntegrator([2.],torch.tensor([[8.]],dtype=D),.01)
    b=LinearMidpointIntegrator([2.],torch.tensor([[9.]],dtype=D),.01)
    with pytest.raises(ValueError):b.restore(a.snapshot())
