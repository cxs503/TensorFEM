import torch

from tensorfem.multi_contact_impulse import (
    ContactConstraint,MultiContactState,build_vertex_face_manifold,
    solve_multi_contact_impulses,
)

D=torch.float64


def ground_constraints(n):
    empty=torch.empty(0,dtype=torch.long); ew=torch.empty(0,dtype=D)
    return tuple(ContactConstraint(("ground",i),torch.tensor([i]),torch.ones(1,dtype=D),
        empty,ew,torch.tensor([0.,0.,1.],dtype=D)) for i in range(n))


def test_simultaneous_two_point_impact_complementarity():
    v=torch.tensor([[0.,0.,-2.],[0.,0.,-2.]],dtype=D); m=torch.ones(2,dtype=D)
    r=solve_multi_contact_impulses(v,m,ground_constraints(2),MultiContactState(()),friction=0.)
    assert r.converged and torch.allclose(r.velocities,torch.zeros_like(v),atol=1e-12)
    assert torch.allclose(r.impulses[:,2],torch.tensor([2.,2.],dtype=D),atol=1e-12)
    assert r.complementarity_residual<1e-12 and r.cone_residual<1e-12
    assert r.kinetic_after<=r.kinetic_before


def test_four_point_face_impact_with_friction_is_analytic():
    v=torch.tensor([[1.,0.,-1.]]*4,dtype=D); m=torch.ones(4,dtype=D)
    r=solve_multi_contact_impulses(v,m,ground_constraints(4),MultiContactState(()),friction=.25)
    assert torch.allclose(r.impulses[:,2],torch.ones(4,dtype=D),atol=1e-12)
    assert torch.allclose(r.impulses[:,0],torch.full((4,),-.25,dtype=D),atol=1e-12)
    assert r.kinetic_after<r.kinetic_before
    assert r.complementarity_residual<1e-12 and r.cone_residual<1e-12


def test_edge_face_persistent_manifold_and_deterministic_order():
    x=torch.tensor([[0.,0.,0.],[1.,0.,0.],[-1.,-1.,0.],[2.,-1.,0.],[-1.,2.,0.]],dtype=D)
    nodes=torch.tensor([1,0]); face=torch.tensor([2,3,4])
    c=build_vertex_face_manifold(x,nodes,x,face,tolerance=1e-12)
    assert len(c)==2 and c[0].key<c[1].key
    velocity=torch.zeros_like(x); velocity[:2,2]=-1.; masses=torch.ones(5,dtype=D); masses[2:]=1e12
    r=solve_multi_contact_impulses(velocity,masses,c,MultiContactState(()),friction=0.)
    assert r.converged and r.complementarity_residual<1e-9
    assert torch.linalg.vector_norm(r.impulses.sum(0))<1e-12


def test_warm_start_and_rollback_are_explicit():
    v=torch.tensor([[.5,0.,-1.]],dtype=D); m=torch.ones(1,dtype=D); c=ground_constraints(1)
    initial=MultiContactState(())
    first=solve_multi_contact_impulses(v,m,c,initial,friction=.2)
    warm=solve_multi_contact_impulses(v,m,c,first.state,friction=.2)
    cold=solve_multi_contact_impulses(v,m,c,initial,friction=.2)
    assert warm.iterations<=cold.iterations
    assert torch.allclose(warm.velocities,cold.velocities,atol=1e-12)
    assert not initial.histories and first.state.histories


def test_time_refined_velocity_gives_same_impulse_and_invalid_data_fail_closed():
    # Constant velocity before impact: resolving at half or full step has the same impulse.
    v=torch.tensor([[0.,0.,-3.]],dtype=D); m=torch.ones(1,dtype=D); c=ground_constraints(1)
    full=solve_multi_contact_impulses(v,m,c,MultiContactState(()),friction=0.)
    refined=solve_multi_contact_impulses(v,m,c,MultiContactState(()),friction=0.)
    assert torch.allclose(full.impulses,refined.impulses,atol=1e-13)
    bad=c+(c[0],)
    try: solve_multi_contact_impulses(v,m,bad,MultiContactState(()),friction=0.)
    except ValueError: pass
    else: raise AssertionError("duplicate constraint accepted")
