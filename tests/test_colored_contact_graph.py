import time
import torch

from tensorfem.colored_contact_graph import prepare_colored_contacts,solve_colored_contacts
from tensorfem.rigid_contact_graph import RigidContact,RigidContactState,solve_rigid_contact_graph

D=torch.float64


def chain(n):
    com=torch.zeros((n,3),dtype=D); com[:,0]=torch.arange(n,dtype=D)*2
    mass=torch.ones(n,dtype=D); inertia=torch.eye(3,dtype=D).repeat(n,1,1)
    v=torch.zeros((n,3),dtype=D); v[:,0]=torch.arange(n-1,-1,-1,dtype=D); w=torch.zeros_like(v)
    contacts=tuple(RigidContact(("c",i),i,i+1,torch.tensor([2*i+1.,0.,0.],dtype=D),
                                torch.tensor([-1.,0.,0.],dtype=D)) for i in range(n-1))
    return com,mass,inertia,v,w,contacts


def test_deterministic_coloring_has_no_body_conflicts():
    com,_,_,_,_,contacts=chain(10); data=prepare_colored_contacts(10,contacts,dtype=D)
    assert len(data.colors)==2
    for ids in data.colors:
        bodies=[]
        for i in ids.tolist(): bodies.extend((int(data.body_a[i]),int(data.body_b[i])))
        assert len(bodies)==len(set(bodies))
    again=prepare_colored_contacts(10,tuple(reversed(contacts)),dtype=D)
    assert all(torch.equal(a,b) for a,b in zip(data.colors,again.colors))


def test_colored_solution_matches_serial_pgs_and_conserves_momentum():
    com,mass,I,v,w,contacts=chain(8); data=prepare_colored_contacts(8,contacts,dtype=D)
    colored=solve_colored_contacts(com,mass,I,v,w,data,tolerance=1e-11,max_iterations=2000)
    serial=solve_rigid_contact_graph(com,mass,I,v,w,contacts,RigidContactState(()),friction=0.,tolerance=1e-11,max_iterations=2000)
    assert colored.converged and serial.converged
    assert torch.allclose(colored.linear_velocity,serial.linear_velocity,rtol=1e-8,atol=1e-9)
    assert torch.linalg.vector_norm((mass[:,None]*colored.linear_velocity).sum(0)-(mass[:,None]*v).sum(0))<1e-10
    assert colored.complementarity_residual<1e-8 and colored.kinetic_after<=colored.kinetic_before


def test_multiple_large_islands_match_serial_partition():
    # Four independent 16-body chains in one graph.
    groups=[]; contacts=[]; offset=0
    for g in range(4):
        c,m,I,v,w,cs=chain(16); c[:,1]=g*5; groups.append((c,m,I,v,w))
        for x in cs: contacts.append(RigidContact((g,*x.key),x.body_a+offset,x.body_b+offset,x.point+torch.tensor([0.,g*5.,0.]),x.normal))
        offset+=16
    com=torch.cat([x[0] for x in groups]); mass=torch.cat([x[1] for x in groups]); I=torch.cat([x[2] for x in groups])
    v=torch.cat([x[3] for x in groups]); w=torch.cat([x[4] for x in groups])
    data=prepare_colored_contacts(64,tuple(contacts),dtype=D)
    result=solve_colored_contacts(com,mass,I,v,w,data,tolerance=1e-9,max_iterations=3000)
    assert len(data.islands)==4 and result.converged and result.complementarity_residual<1e-7


def test_1024_constraint_vectorized_benchmark_and_serial_equivalence():
    n=1024; com=torch.zeros((n,3),dtype=D); com[:,0]=torch.arange(n,dtype=D)*2
    mass=torch.ones(n,dtype=D); I=torch.eye(3,dtype=D).repeat(n,1,1)
    v=torch.zeros((n,3),dtype=D); v[:,2]=-1.; w=torch.zeros_like(v)
    contacts=tuple(RigidContact(("g",i),i,-1,com[i]-torch.tensor([0.,0.,1.],dtype=D),
                                torch.tensor([0.,0.,1.],dtype=D)) for i in range(n))
    data=prepare_colored_contacts(n,contacts,dtype=D); t=time.perf_counter()
    colored=solve_colored_contacts(com,mass,I,v,w,data); colored_time=time.perf_counter()-t
    t=time.perf_counter(); serial=solve_rigid_contact_graph(com,mass,I,v,w,contacts,RigidContactState(()),friction=0.); serial_time=time.perf_counter()-t
    assert len(data.colors)==1 and colored.converged
    assert torch.allclose(colored.linear_velocity,serial.linear_velocity,atol=1e-12)
    assert colored_time<10. and serial_time<10.


def test_warm_state_and_device_layout_fail_closed():
    com,mass,I,v,w,contacts=chain(4); data=prepare_colored_contacts(4,contacts,dtype=D)
    cold=solve_colored_contacts(com,mass,I,v,w,data,max_iterations=1000)
    warm=solve_colored_contacts(com,mass,I,v,w,data,cold.state,max_iterations=1000)
    assert warm.iterations<=cold.iterations and torch.allclose(warm.linear_velocity,cold.linear_velocity,atol=1e-9)
    assert all(x.device==com.device for x in (data.body_a,data.points,data.normals,*data.colors))
