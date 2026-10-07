import torch

from tensorfem.self_contact_ccd import (
    DynamicSelfContactState,earliest_self_contact,update_dynamic_self_contact,
    vertex_triangle_ccd,
)

D=torch.float64


def crossing(z0=1.,z1=-1.):
    start=torch.tensor([[.25,.25,z0],[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],dtype=D)
    end=start.clone(); end[0,2]=z1
    faces=torch.tensor([[1,2,3]],dtype=torch.long)
    return start,end,faces


def test_high_speed_vertex_cannot_tunnel_through_thin_face():
    start,end,faces=crossing()
    r=update_dynamic_self_contact(start,end,faces,DynamicSelfContactState(()),dt=.01)
    assert len(r.events)==1
    assert abs(r.accepted_fraction-.5)/.5 < .03
    assert abs(float(r.impact_positions[0,2])) < 1e-8
    assert torch.linalg.vector_norm(r.impulses.sum(0)) < 1e-12
    assert torch.linalg.vector_norm(torch.linalg.cross(r.impact_positions,r.impulses).sum(0)) < 1e-10
    assert r.state.pairs[0].vertex==0 and r.state.pairs[0].face==0


def test_moving_triangle_toi_matches_independent_linear_oracle():
    p0=torch.tensor([.2,.2,1.],dtype=D); p1=torch.tensor([.2,.2,-1.],dtype=D)
    tri0=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],dtype=D)
    tri1=tri0.clone(); tri1[:,2]=.2
    hit=vertex_triangle_ccd(p0,p1,tri0,tri1)
    exact=1/2.2
    assert hit is not None and abs(hit[0]-exact)/exact < .03
    assert abs(hit[0]-exact) < 1e-8


def test_time_step_refinement_and_transactional_pair_lifetime():
    start,end,faces=crossing()
    initial=DynamicSelfContactState(())
    whole=update_dynamic_self_contact(start,end,faces,initial,dt=1.)
    split=start+.4*(end-start)
    first=update_dynamic_self_contact(start,split,faces,initial,dt=.4)
    assert not first.events and not first.state.pairs
    second=update_dynamic_self_contact(split,end,faces,first.state,dt=.6)
    global_t=.4+.6*second.accepted_fraction
    assert abs(global_t-whole.accepted_fraction) < 1e-8
    # Trial update did not mutate the checkpoint; commit is explicit.
    assert not initial.pairs and whole.state.pairs
    separated=start.clone(); separated[0,2]=.1
    farther=separated.clone(); farther[0,2]=.2
    opened=update_dynamic_self_contact(separated,farther,faces,whole.state,dt=.1)
    assert not opened.events and not opened.state.pairs


def test_two_moving_fold_faces_detect_impact_and_balance_impulse():
    # Two nonadjacent triangular flaps move through one another in one step.
    start=torch.tensor([[0.,0.,.2],[1.,0.,.2],[0.,1.,.2],
                        [0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],dtype=D)
    end=start.clone(); end[:3,2]=-.2
    faces=torch.tensor([[0,1,2],[3,4,5]],dtype=torch.long)
    events=earliest_self_contact(start,end,faces)
    assert events and abs(events[0].toi-.5)<1e-8
    r=update_dynamic_self_contact(start,end,faces,DynamicSelfContactState(()),dt=.02)
    assert torch.linalg.vector_norm(r.impulses.sum(0)) < 1e-11
    assert torch.linalg.vector_norm(torch.linalg.cross(r.impact_positions,r.impulses).sum(0)) < 1e-9


def test_invalid_mass_and_degenerate_triangle_fail_closed():
    start,end,faces=crossing()
    try:
        update_dynamic_self_contact(start,end,faces,DynamicSelfContactState(()),dt=1.,masses=torch.zeros(4,dtype=D))
    except ValueError: pass
    else: raise AssertionError("zero masses accepted")
    tri=torch.zeros((3,3),dtype=D)
    try: vertex_triangle_ccd(start[0],end[0],tri,tri)
    except ValueError: pass
    else: raise AssertionError("degenerate moving triangle accepted")
