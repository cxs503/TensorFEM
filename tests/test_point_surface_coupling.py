import pytest
import torch
from tensorfem.point_surface_coupling import PointSurfaceExchange
D=torch.float64

def interface():
    return PointSurfaceExchange([[0.,0.,0.],[2.,0.,0.],[2.,1.,0.],[0.,1.,0.]],[[0,1,2,3]],[[.5,.2,.1],[1.5,.8,-.2]],maximum_offset_m=.3)

def test_original_force_location_wrench_and_rigid_virtual_work():
    s=interface();f=torch.tensor([[1.,2.,3.],[-2.,4.,1.]],dtype=D)
    omega=torch.tensor([.1,.2,.3],dtype=D);translation=torch.tensor([.4,.5,.6],dtype=D)
    v=torch.cat((translation+torch.linalg.cross(omega.expand_as(s.nodes),s.nodes),omega.expand_as(s.nodes)),1)
    torch.testing.assert_close(s.velocities(v),translation+torch.linalg.cross(omega.expand_as(s.points),s.points))
    a=s.audit(s.point_loads(f,time_s=.2,source='TensorLBM'),v,expected_time_s=.2)
    assert max(a['force_error_N'],a['moment_error_N_m'],a['power_error_W'])<1e-12
    assert s.motion_record(v,time_s=.2)['time_s']==.2

def test_natural_coordinates_and_offsets():
    s=interface()
    torch.testing.assert_close(s.anchors,torch.tensor([[.5,.2,0.],[1.5,.8,0.]],dtype=D))
    torch.testing.assert_close(s.weights.sum(1),torch.ones(2,dtype=D))

@pytest.mark.parametrize('bad',['distance','mutated','time','force','source'])
def test_invalid_point_map_or_load_rejected(bad):
    s=interface();f=torch.ones((2,3),dtype=D)
    with pytest.raises(ValueError):
        if bad=='distance':PointSurfaceExchange(s.nodes,s.elements,[[0.,0.,10.]],maximum_offset_m=.1)
        elif bad=='mutated':s.offsets[0,0]+=1.;s.velocities(torch.zeros((4,6)))
        elif bad=='time':s.audit(s.point_loads(f,time_s=.2,source='TensorDEM'),torch.zeros((4,6)),expected_time_s=.3)
        elif bad=='force':s.point_loads([[float('nan')]*3]*2,time_s=.2,source='TensorDEM')
        else:s.point_loads(f,time_s=.2,source='unknown')
