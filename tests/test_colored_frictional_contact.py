import time
import torch

from tensorfem.colored_contact_graph import prepare_colored_contacts
from tensorfem.colored_frictional_contact import solve_colored_frictional_contacts
from tensorfem.rigid_contact_graph import RigidContact,RigidContactState,solve_rigid_contact_graph

D=torch.float64


def chain(n):
    com=torch.zeros((n,3),dtype=D);com[:,0]=torch.arange(n,dtype=D)*2
    mass=torch.ones(n,dtype=D);I=torch.eye(3,dtype=D).repeat(n,1,1)
    v=torch.zeros((n,3),dtype=D);v[:,0]=torch.arange(n-1,-1,-1,dtype=D);v[:,1]=torch.linspace(1.,0.,n);w=torch.zeros_like(v)
    cs=tuple(RigidContact(("c",i),i,i+1,torch.tensor([2*i+1.,0.,0.],dtype=D),torch.tensor([-1.,0.,0.],dtype=D)) for i in range(n-1))
    return com,mass,I,v,w,cs


def test_highly_coupled_friction_chain_matches_scalar_oracle():
    com,m,I,v,w,cs=chain(8);data=prepare_colored_contacts(8,cs,dtype=D)
    colored=solve_colored_frictional_contacts(com,m,I,v,w,data,friction=.25,tolerance=1e-10,max_iterations=3000)
    serial=solve_rigid_contact_graph(com,m,I,v,w,cs,RigidContactState(()),friction=.25,tolerance=1e-10,max_iterations=3000)
    assert colored.converged and serial.converged
    assert torch.allclose(colored.linear_velocity,serial.linear_velocity,rtol=2e-7,atol=2e-8)
    assert torch.allclose(colored.angular_velocity,serial.angular_velocity,rtol=2e-7,atol=2e-8)
    assert colored.complementarity_residual<1e-7 and colored.cone_residual<1e-10
    assert colored.kinetic_after<=colored.kinetic_before and colored.dissipated_energy>=0


def test_stick_slip_history_warm_start_and_rollback():
    com=torch.tensor([[0.,0.,1.]],dtype=D);m=torch.ones(1,dtype=D);I=torch.eye(3,dtype=D).reshape(1,3,3)
    v=torch.tensor([[.05,0.,-1.]],dtype=D);w=torch.zeros_like(v)
    cs=(RigidContact(("g",),0,-1,torch.zeros(3,dtype=D),torch.tensor([0.,0.,1.],dtype=D)),)
    data=prepare_colored_contacts(1,cs,dtype=D);initial=None
    stick=solve_colored_frictional_contacts(com,m,I,v,w,data,initial,friction=1.)
    assert bool(stick.state.sticking[0])
    sliding_v=v.clone();sliding_v[0,0]=2.
    slide=solve_colored_frictional_contacts(com,m,I,sliding_v,w,data,initial,friction=.2)
    assert not bool(slide.state.sticking[0])
    warm=solve_colored_frictional_contacts(com,m,I,sliding_v,w,data,slide.state,friction=.2)
    cold=solve_colored_frictional_contacts(com,m,I,sliding_v,w,data,initial,friction=.2)
    assert warm.iterations<=cold.iterations and torch.allclose(warm.linear_velocity,cold.linear_velocity,atol=1e-10)


def test_1024_independent_friction_constraints_match_serial_and_benchmark():
    n=1024;com=torch.zeros((n,3),dtype=D);com[:,0]=torch.arange(n,dtype=D)*2
    m=torch.ones(n,dtype=D);I=torch.eye(3,dtype=D).repeat(n,1,1)
    v=torch.zeros((n,3),dtype=D);v[:,0]=.4;v[:,2]=-1.;w=torch.zeros_like(v)
    cs=tuple(RigidContact(("g",i),i,-1,com[i]-torch.tensor([0.,0.,1.],dtype=D),torch.tensor([0.,0.,1.],dtype=D)) for i in range(n))
    data=prepare_colored_contacts(n,cs,dtype=D);t=time.perf_counter();colored=solve_colored_frictional_contacts(com,m,I,v,w,data,friction=.3);ct=time.perf_counter()-t
    t=time.perf_counter();serial=solve_rigid_contact_graph(com,m,I,v,w,cs,RigidContactState(()),friction=.3);st=time.perf_counter()-t
    assert colored.converged and torch.allclose(colored.linear_velocity,serial.linear_velocity,atol=1e-11)
    assert torch.allclose(colored.angular_velocity,serial.angular_velocity,atol=1e-11)
    assert colored.complementarity_residual<1e-10 and colored.cone_residual<1e-10
    assert ct<10 and st<10


def test_state_topology_and_device_fail_closed():
    com,m,I,v,w,cs=chain(3);data=prepare_colored_contacts(3,cs,dtype=D)
    result=solve_colored_frictional_contacts(com,m,I,v,w,data,friction=.2)
    bad=type(result.state)((('wrong',),),result.state.normal_impulses,result.state.friction_impulses,result.state.sticking)
    try:solve_colored_frictional_contacts(com,m,I,v,w,data,bad,friction=.2)
    except ValueError:pass
    else:raise AssertionError("wrong warm topology accepted")
    assert all(x.device==com.device for x in (result.state.normal_impulses,result.state.friction_impulses,result.state.sticking))
