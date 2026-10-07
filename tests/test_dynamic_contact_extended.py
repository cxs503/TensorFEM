import torch
import pytest

from tensorfem.dynamic_contact_extended import (
    AmbiguousCoplanarContactError,UnifiedEvent,coulomb_impact,edge_edge_ccd,
    earliest_unified_contacts,spatial_hash_edge_pairs,
)

D=torch.float64


def crossing():
    start=torch.tensor([[-1.,0.,0.],[1.,0.,0.],[0.,-1.,1.],[0.,1.,1.]],dtype=D)
    end=start.clone(); end[2:,2]=-1.
    edges=torch.tensor([[0,1],[2,3]],dtype=torch.long)
    return start,end,edges


def test_high_speed_edge_crossing_toi_and_time_refinement():
    start,end,edges=crossing()
    hit=edge_edge_ccd(start[edges[0]],end[edges[0]],start[edges[1]],end[edges[1]])
    assert hit is not None and abs(hit.toi-.5)/.5 < .03
    assert abs(hit.toi-.5)<1e-8
    split=start+.4*(end-start)
    assert edge_edge_ccd(start[edges[0]],split[edges[0]],start[edges[1]],split[edges[1]]) is None
    local=edge_edge_ccd(split[edges[0]],end[edges[0]],split[edges[1]],end[edges[1]])
    assert abs((.4+.6*local.toi)-hit.toi)<1e-8


def test_unified_queue_selects_earliest_and_is_deterministic():
    start,end,edges=crossing(); faces=torch.empty((0,3),dtype=torch.long)
    a=earliest_unified_contacts(start,end,faces,edges,cell_size=.5)
    b=earliest_unified_contacts(start,end,faces,edges,cell_size=.5)
    assert len(a)==len(b)==1 and a[0].kind==b[0].kind=="edge-edge"
    assert torch.equal(a[0].side_a,b[0].side_a) and torch.equal(a[0].side_b,b[0].side_b)
    assert abs(a[0].toi-.5)<1e-8


def test_coulomb_impact_conserves_linear_angular_momentum_and_energy():
    positions=torch.tensor([[-1.,0.,0.],[1.,0.,0.],[0.,-1.,0.],[0.,1.,0.]],dtype=D)
    event=UnifiedEvent("edge-edge",.5,torch.tensor([0,1]),torch.tensor([.5,.5],dtype=D),
        torch.tensor([2,3]),torch.tensor([.5,.5],dtype=D),torch.tensor([0.,0.,1.],dtype=D))
    velocity=torch.tensor([[1.,0.,-1.],[1.,0.,-1.],[0.,0.,0.],[0.,0.,0.]],dtype=D)
    mass=torch.ones(4,dtype=D)
    r=coulomb_impact(event,velocity,mass,friction=.3)
    assert torch.linalg.vector_norm(r.impulses.sum(0))<1e-12
    assert torch.linalg.vector_norm(torch.linalg.cross(positions,r.impulses).sum(0))<1e-12
    assert r.kinetic_after<=r.kinetic_before
    assert torch.linalg.vector_norm(r.friction_impulse)<=.3*torch.linalg.vector_norm(r.normal_impulse)*(1+1e-12)


def test_spatial_hash_deduplicates_and_reduces_large_candidate_set():
    # 40 remote edges plus one crossing pair; only nearby swept boxes pair.
    points=[]; edges=[]
    for i in range(40):
        points.extend(([10.+3*i,0.,0.],[11.+3*i,0.,0.])); edges.append([2*i,2*i+1])
    offset=len(points); points.extend(([-1.,0.,0.],[1.,0.,0.],[0.,-1.,1.],[0.,1.,1.]))
    edges.extend(([offset,offset+1],[offset+2,offset+3]))
    start=torch.tensor(points,dtype=D); end=start.clone(); end[offset+2:,2]=-1.
    edges=torch.tensor(edges,dtype=torch.long)
    pairs=spatial_hash_edge_pairs(start,end,edges,cell_size=1.)
    assert pairs==((40,41),)
    assert pairs==spatial_hash_edge_pairs(start,end,edges,cell_size=1.)


def test_transverse_coplanar_is_defined_parallel_overlap_fails_closed():
    a=torch.tensor([[-1.,0.,0.],[1.,0.,0.]],dtype=D)
    b=torch.tensor([[0.,-1.,0.],[0.,1.,0.]],dtype=D)
    hit=edge_edge_ccd(a,a,b,b)
    assert hit is not None and hit.toi==0.
    parallel=torch.tensor([[-.5,0.,0.],[.5,0.,0.]],dtype=D)
    with pytest.raises(AmbiguousCoplanarContactError): edge_edge_ccd(a,a,parallel,parallel)
