import torch

from tensorfem.frictional_mortar import (
    initial_frictional_mortar_state,update_frictional_mortar,update_self_contact,
)

D=torch.float64


def patch(z=-.001,n=1,reverse=False):
    nodes=torch.tensor([[i/n,j/n,z] for i in range(n+1) for j in range(n+1)],dtype=D)
    faces=[]; ix=lambda i,j:i*(n+1)+j
    for i in range(n):
        for j in range(n): faces.extend(([ix(i,j),ix(i+1,j),ix(i+1,j+1)],
                                         [ix(i,j),ix(i+1,j+1),ix(i,j+1)]))
    f=torch.tensor(faces,dtype=torch.long)
    return nodes,torch.flip(f,(1,)) if reverse else f


def test_sliding_block_matches_coulomb_and_energy_oracle():
    slave,sf=patch(); master,mf=patch(0.,2)
    state=initial_frictional_mortar_state(slave,sf,master,mf)
    inc=torch.zeros_like(slave); inc[:,0]=.01
    r=update_frictional_mortar(slave,sf,master,mf,state,normal_penalty=1e5,
        tangential_penalty=1e4,friction=.3,relative_increments=inc)
    assert abs(float(r.normal_resultant[2])-100.)/100 < .03
    assert abs(float(r.tangential_resultant[0])+30.)/30 < .03
    assert torch.linalg.vector_norm(r.slave_forces.sum(0)+r.master_forces.sum(0)) < 1e-12
    normal_stored=.5*100**2/1e5
    tangential_stored=float(r.stored_energy)-normal_stored
    work=.5*30*.003+30*(.01-.003)
    assert abs(work-tangential_stored-float(r.dissipation_increment))/work < .03


def test_trial_state_rollback_and_commit_are_explicit():
    slave,sf=patch(); master,mf=patch(0.,2)
    initial=initial_frictional_mortar_state(slave,sf,master,mf)
    large=torch.zeros_like(slave); large[:,0]=.01
    rejected=update_frictional_mortar(slave,sf,master,mf,initial,normal_penalty=1e5,
        tangential_penalty=1e4,friction=.3,relative_increments=large)
    small=torch.zeros_like(slave); small[:,0]=.001
    retried=update_frictional_mortar(slave,sf,master,mf,initial,normal_penalty=1e5,
        tangential_penalty=1e4,friction=.3,relative_increments=small)
    committed=update_frictional_mortar(slave,sf,master,mf,rejected.state,normal_penalty=1e5,
        tangential_penalty=1e4,friction=.3,relative_increments=small)
    assert abs(float(retried.tangential_resultant[0])+10.) < 1e-11
    assert abs(float(committed.tangential_resultant[0])+30.) < 1e-11
    assert all(float(p.dissipated_energy_density)==0 for p in initial.points)


def test_nonmatching_frictional_patch_force_and_moment():
    slave,sf=patch(n=3); master,mf=patch(0.,2)
    state=initial_frictional_mortar_state(slave,sf,master,mf)
    inc=torch.zeros_like(slave); inc[:,1]=.0005
    r=update_frictional_mortar(slave,sf,master,mf,state,normal_penalty=2e5,
        tangential_penalty=1e4,friction=.5,relative_increments=inc)
    exact_t=5.
    assert abs(float(r.normal_resultant[2])-200.)/200 < .03
    assert abs(float(r.tangential_resultant[1])+exact_t)/exact_t < .03
    sm=torch.linalg.cross(slave,r.slave_forces).sum(0)
    mm=torch.linalg.cross(master,r.master_forces).sum(0)
    # Small offset creates the physical friction couple; remove r x F offset.
    expected=torch.linalg.cross(torch.tensor([0.,0.,-.001],dtype=D),r.tangential_resultant)
    assert torch.linalg.vector_norm(sm+mm-expected) < 1e-11


def test_folded_self_contact_is_assembled_once_and_conserves_momentum():
    v=torch.tensor([[0.,0.,0.],[0.,1.,0.],[1.,0.,0.],[1.,1.,0.],
                    [.05,0.,.02],[.05,1.,.02]],dtype=D)
    f=torch.tensor([[0,2,1],[1,2,3],[2,4,3],[3,4,5]],dtype=torch.long)
    r=update_self_contact(v,f,clearance=.08,normal_penalty=1e4)
    assert r.candidate_pairs == ((0,3),) and r.active_pairs == ((0,3),)
    assert torch.linalg.vector_norm(r.forces.sum(0)) < 1e-11
    assert torch.linalg.vector_norm(torch.linalg.cross(v,r.forces).sum(0)) < 1e-11
    assert r.maximum_penetration > 0 and r.penalty_energy > 0


def test_topology_and_parameters_fail_closed():
    slave,sf=patch(); master,mf=patch(0.)
    state=initial_frictional_mortar_state(slave,sf,master,mf)
    try:
        update_frictional_mortar(slave,torch.flip(sf,(1,)),master,mf,state,
            normal_penalty=1.,tangential_penalty=1.,friction=0.)
    except ValueError: pass
    else: raise AssertionError("changed topology accepted")
    try: update_self_contact(slave,sf,clearance=0.,normal_penalty=1.)
    except ValueError: pass
    else: raise AssertionError("zero clearance accepted")
