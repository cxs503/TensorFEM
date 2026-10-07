import time
import torch

from tensorfem.rigid_contact_graph import (
    RigidContact,RigidContactState,contact_islands,solve_rigid_contact_graph,
)

D=torch.float64


def angular_momentum(com,mass,I,v,w):
    return torch.linalg.cross(com,mass[:,None]*v).sum(0)+torch.stack([I[i]@w[i] for i in range(len(com))]).sum(0)


def test_eccentric_two_body_impact_conserves_momentum_and_cone():
    com=torch.tensor([[-1.,.5,0.],[1.,-.5,0.]],dtype=D); mass=torch.tensor([2.,3.],dtype=D)
    I=torch.stack((torch.diag(torch.tensor([.4,.5,.6],dtype=D)),torch.diag(torch.tensor([.7,.8,.9],dtype=D))))
    v=torch.tensor([[2.,1.,0.],[0.,0.,0.]],dtype=D); w=torch.zeros((2,3),dtype=D)
    c=(RigidContact(("pair",0),0,1,torch.zeros(3,dtype=D),torch.tensor([-1.,0.,0.],dtype=D)),)
    p0=(mass[:,None]*v).sum(0); h0=angular_momentum(com,mass,I,v,w)
    r=solve_rigid_contact_graph(com,mass,I,v,w,c,RigidContactState(()),friction=.3)
    assert r.converged and r.complementarity_residual<1e-9 and r.cone_residual<1e-12
    assert torch.linalg.vector_norm((mass[:,None]*r.linear_velocity).sum(0)-p0)<1e-11
    assert torch.linalg.vector_norm(angular_momentum(com,mass,I,r.linear_velocity,r.angular_velocity)-h0)<1e-11
    assert torch.linalg.vector_norm(r.angular_velocity)>0
    assert r.kinetic_after<=r.kinetic_before


def test_independent_contact_islands_equal_separate_solves():
    com=torch.tensor([[-1.,0.,0.],[1.,0.,0.],[-1.,3.,0.],[1.,3.,0.]],dtype=D)
    mass=torch.ones(4,dtype=D); I=torch.eye(3,dtype=D).repeat(4,1,1)
    v=torch.tensor([[1.,0.,0.],[0.,0.,0.],[2.,0.,0.],[0.,0.,0.]],dtype=D); w=torch.zeros_like(v)
    contacts=(RigidContact(("a",),0,1,torch.tensor([0.,0.,0.],dtype=D),torch.tensor([-1.,0.,0.],dtype=D)),
              RigidContact(("b",),2,3,torch.tensor([0.,3.,0.],dtype=D),torch.tensor([-1.,0.,0.],dtype=D)))
    both=solve_rigid_contact_graph(com,mass,I,v,w,contacts,RigidContactState(()),friction=0.)
    one=solve_rigid_contact_graph(com,mass,I,v,w,contacts[:1],RigidContactState(()),friction=0.)
    two=solve_rigid_contact_graph(com,mass,I,v,w,contacts[1:],RigidContactState(()),friction=0.)
    assert both.islands==((0,1),(2,3))
    assert torch.allclose(both.linear_velocity[:2],one.linear_velocity[:2],atol=1e-12)
    assert torch.allclose(both.linear_velocity[2:],two.linear_velocity[2:],atol=1e-12)


def test_warm_start_is_deterministic_and_rollback_explicit():
    com=torch.tensor([[0.,0.,1.]],dtype=D); mass=torch.ones(1,dtype=D); I=torch.eye(3,dtype=D).reshape(1,3,3)
    v=torch.tensor([[.2,0.,-1.]],dtype=D); w=torch.zeros_like(v)
    c=(RigidContact(("ground",0),0,-1,torch.tensor([0.,0.,0.],dtype=D),torch.tensor([0.,0.,1.],dtype=D)),)
    initial=RigidContactState(()); first=solve_rigid_contact_graph(com,mass,I,v,w,c,initial,friction=.2)
    warm=solve_rigid_contact_graph(com,mass,I,v,w,c,first.state,friction=.2)
    cold=solve_rigid_contact_graph(com,mass,I,v,w,c,initial,friction=.2)
    assert warm.iterations<=cold.iterations and torch.allclose(warm.linear_velocity,cold.linear_velocity,atol=1e-12)
    assert not initial.histories and first.state.histories


def test_thousand_constraint_sparse_islands_benchmark():
    n=1000; com=torch.zeros((n,3),dtype=D); com[:,0]=torch.arange(n,dtype=D)*3
    mass=torch.ones(n,dtype=D); I=torch.eye(3,dtype=D).repeat(n,1,1)
    v=torch.zeros((n,3),dtype=D); v[:,2]=-1.; w=torch.zeros_like(v)
    contacts=tuple(RigidContact(("g",i),i,-1,com[i]-torch.tensor([0.,0.,1.],dtype=D),
                                torch.tensor([0.,0.,1.],dtype=D)) for i in range(n))
    start=time.perf_counter(); r=solve_rigid_contact_graph(com,mass,I,v,w,contacts,RigidContactState(()),friction=0.); elapsed=time.perf_counter()-start
    assert len(r.islands)==n and r.converged and r.complementarity_residual<1e-10
    assert torch.max(torch.abs(r.linear_velocity[:,2]))<1e-12
    assert elapsed<10.


def test_invalid_inertia_or_duplicate_key_fails_closed():
    com=torch.zeros((1,3),dtype=D); mass=torch.ones(1,dtype=D); I=torch.zeros((1,3,3),dtype=D)
    c=RigidContact(("x",),0,-1,torch.zeros(3,dtype=D),torch.tensor([0.,0.,1.],dtype=D))
    try: solve_rigid_contact_graph(com,mass,I,torch.zeros_like(com),torch.zeros_like(com),(c,),RigidContactState(()),friction=0.)
    except torch.linalg.LinAlgError: pass
    else: raise AssertionError("singular inertia accepted")
    I=torch.eye(3,dtype=D).reshape(1,3,3)
    try: solve_rigid_contact_graph(com,mass,I,torch.zeros_like(com),torch.zeros_like(com),(c,c),RigidContactState(()),friction=0.)
    except ValueError: pass
    else: raise AssertionError("duplicate key accepted")
