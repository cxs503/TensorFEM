import math
import torch

from tensorfem.contact3d import (
    initial_contact_state, project_point_to_facets,
    update_contact_nodes, update_node_facet_contact,
)

D=torch.float64


def test_point_pressing_triangle_action_reaction_and_moment():
    v=torch.tensor([[0.,0.,0.],[2.,0.,0.],[0.,2.,0.]],dtype=D)
    f=torch.tensor([[0,1,2]],dtype=torch.long); p0=torch.tensor([.5,.5,.1],dtype=D)
    state=initial_contact_state(p0,v,f); p=torch.tensor([.5,.5,-.002],dtype=D)
    u=update_node_facet_contact(p,v,f,state,normal_penalty=2e5,tangential_penalty=1e4,friction=.3)
    assert abs(float(u.normal_force)-400.)/400 < .03
    assert torch.linalg.vector_norm(u.slave_force+u.master_forces.sum(0)) < 1e-12
    # Master nodal forces reproduce the resultant moment about the origin.
    mm=torch.linalg.cross(v,u.master_forces).sum(0)
    sm=torch.linalg.cross(p,u.slave_force)
    assert torch.linalg.vector_norm(mm+sm) < 1e-12
    assert torch.allclose(u.projection.weights,torch.tensor([.5,.25,.25],dtype=D))


def test_inclined_triangle_coulomb_stick_then_slide_and_energy():
    a=math.radians(30); t=torch.tensor([math.cos(a),0.,math.sin(a)],dtype=D)
    b=torch.tensor([0.,1.,0.],dtype=D); n=torch.linalg.cross(t,b)
    v=torch.stack((-2*t-b,2*t-b,2*t+b,-2*t+b)); f=torch.tensor([[0,1,2,3]])
    base=.2*t; state=initial_contact_state(base+.01*n,v,f)
    p=base-.002*n
    stick=update_node_facet_contact(p,v,f,state,normal_penalty=1e5,tangential_penalty=2e4,friction=.25,relative_increment=.001*t)
    assert stick.state.sticking
    assert abs(float(stick.normal_force)-200.)/200 < .03
    assert abs(float(torch.linalg.vector_norm(stick.tangential_force))-20.)/20 < .03
    slide=update_node_facet_contact(p,v,f,stick.state,normal_penalty=1e5,tangential_penalty=2e4,friction=.25,relative_increment=.02*t)
    assert not slide.state.sticking
    assert abs(float(torch.linalg.vector_norm(slide.tangential_force))-50.)/50 < .03
    assert slide.dissipation_increment > 0
    # Elastic-plastic split of this increment is exact for perfect Coulomb friction.
    tangential_stored=.5*2e4*torch.dot(slide.state.elastic_slip,slide.state.elastic_slip)
    assert abs(float(torch.dot(slide.tangential_force,slide.state.elastic_slip))+2*tangential_stored) < 1e-10


def test_finite_sliding_crosses_triangles_and_faces():
    v=torch.tensor([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.],
                    [2.,0.,0.],[2.,1.,0.]],dtype=D)
    f=torch.tensor([[0,1,2,3],[1,4,5,2]],dtype=torch.long)
    s=initial_contact_state(torch.tensor([.1,.5,.1],dtype=D),v,f)
    p=torch.tensor([1.7,.4,-.001],dtype=D)
    u=update_node_facet_contact(p,v,f,s,normal_penalty=1e5,tangential_penalty=1e4,friction=0.,relative_increment=torch.zeros(3,dtype=D))
    assert u.projection.face == 1 and u.state.active
    assert torch.linalg.vector_norm(u.projection.point-torch.tensor([1.7,.4,0.],dtype=D)) < 1e-13
    assert abs(float(u.projection.gap)+.001) < 1e-13


def test_augmented_normal_multiple_nodes_and_candidate_filter():
    v=torch.tensor([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.],
                    [2.,0.,0.],[2.,1.,0.]],dtype=D)
    f=torch.tensor([[0,1,2,3],[1,4,5,2]],dtype=torch.long)
    pts0=torch.tensor([[.2,.2,.1],[1.5,.5,.1]],dtype=D)
    states=[initial_contact_state(p,v,f) for p in pts0]
    pts=pts0.clone(); pts[:,2]=-.001
    kw=dict(normal_penalty=1e5,tangential_penalty=1e4,friction=0.,normal_method="augmented_lagrangian",relative_increment=torch.zeros(3,dtype=D))
    us,sf,mf=update_contact_nodes(pts,v,f,states,**kw)
    us2,sf2,mf2=update_contact_nodes(pts,v,f,[u.state for u in us],**kw)
    assert torch.allclose(torch.stack([u.normal_force for u in us2]),torch.tensor([200.,200.],dtype=D))
    assert torch.linalg.vector_norm(sf2.sum(0)+mf2.sum(0)) < 1e-12
    # Adjacency filtering primitive for self-contact: incident face 0 can be excluded.
    pr=project_point_to_facets(torch.tensor([.8,.5,.01],dtype=D),v,f,excluded_faces=torch.tensor([0]))
    assert pr.face == 1


def test_invalid_or_empty_candidates_fail_closed():
    v=torch.zeros((3,3),dtype=D); f=torch.tensor([[0,1,2]])
    try: project_point_to_facets(torch.zeros(3,dtype=D),v,f)
    except ValueError: pass
    else: raise AssertionError("degenerate face accepted")
    v=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],dtype=D)
    try: project_point_to_facets(torch.zeros(3,dtype=D),v,f,excluded_faces=torch.tensor([0]))
    except ValueError: pass
    else: raise AssertionError("empty candidate set accepted")
