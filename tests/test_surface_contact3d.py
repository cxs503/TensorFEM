import math
import torch

from tensorfem.surface_contact3d import (
    hertz_sphere_halfspace_reference, initial_surface_contact_state,
    integrate_hertz_pressure, tributary_areas, update_surface_contact,
    winkler_sphere_load,
)

D=torch.float64


def surfaces(z=.01):
    slave=torch.tensor([[0.,0.,z],[1.,0.,z],[1.,1.,z],[0.,1.,z]],dtype=D)
    sf=torch.tensor([[0,1,2,3]],dtype=torch.long)
    master=torch.tensor([[-1.,-1.,0.],[2.,-1.,0.],[2.,2.,0.],[-1.,2.,0.]],dtype=D)
    mf=torch.tensor([[0,1,2,3]],dtype=torch.long)
    return slave,sf,master,mf


def test_uniform_surface_pressure_and_consistent_action_reaction():
    slave,sf,master,mf=surfaces(); state=initial_surface_contact_state(slave,sf,master,mf)
    current=slave.clone(); current[:,2]=-.002
    u=update_surface_contact(current,sf,master,mf,state,normal_penalty=2e5,
                             tangential_penalty=1e4,friction=0.,
                             relative_increments=torch.zeros_like(current))
    # unit area x pressure 400
    assert abs(float(u.slave_forces[:,2].sum())-400.)/400 < .03
    assert torch.linalg.vector_norm(u.slave_forces.sum(0)+u.master_forces.sum(0))/400 < 1e-13
    assert abs(float(tributary_areas(current,sf).sum())-1.) < 1e-14
    assert abs(float(u.maximum_penetration)-.002) < 1e-14


def test_surface_coulomb_friction_and_energy_partition():
    slave,sf,master,mf=surfaces(); state=initial_surface_contact_state(slave,sf,master,mf)
    current=slave.clone(); current[:,2]=-.001
    increments=torch.zeros_like(current); increments[:,0]=.01
    u=update_surface_contact(current,sf,master,mf,state,normal_penalty=1e5,
                             tangential_penalty=1e4,friction=.3,
                             relative_increments=increments)
    assert abs(float(u.slave_forces[:,2].sum())-100.)/100 < .03
    assert abs(float(u.slave_forces[:,0].sum())+30.)/30 < .03
    assert bool(torch.all(u.active_nodes)) and u.dissipation_increment > 0
    normal_stored=.5*100.**2/1e5 # pressure energy density, area=1
    tangential_stored=float(u.stored_energy)-normal_stored
    work=.5*30.*(.3*100./1e4)+30.*(.01-.3*100./1e4)
    assert abs(work-(tangential_stored+float(u.dissipation_increment)))/work < .03


def test_surface_search_moves_to_adjacent_master_face():
    slave,sf,_,_=surfaces()
    master=torch.tensor([[0.,-1.,0.],[1.,-1.,0.],[1.,2.,0.],[0.,2.,0.],
                         [2.,-1.,0.],[2.,2.,0.]],dtype=D)
    mf=torch.tensor([[0,1,2,3],[1,4,5,2]],dtype=torch.long)
    state=initial_surface_contact_state(slave,sf,master,mf)
    current=slave.clone(); current[:,0]+=1.; current[:,2]=-.001
    u=update_surface_contact(current,sf,master,mf,state,normal_penalty=1e5,
        tangential_penalty=1e4,friction=0.,relative_increments=torch.zeros_like(current))
    assert abs(float(u.slave_forces[:,2].sum())-100.)/100 < .03
    # Boundary nodes may tie-break to face 0; nodes beyond the seam use face 1.
    assert {s.face for s in u.state.local_states} == {0,1}
    assert all(u.state.local_states[i].face == 1 for i in (1,2))


def test_hertz_pressure_quadrature_recovers_load_and_converges():
    ref=hertz_sphere_halfspace_reference(1000.,.05,210e9,.3,210e9,.3)
    errors=[]
    for n in (8,16,32,64):
        errors.append(abs(integrate_hertz_pressure(ref,radial_cells=n)-1000.)/1000.)
    assert all(b<a for a,b in zip(errors,errors[1:]))
    assert errors[-1] < .03
    assert abs(ref.contact_radius-0.0006875344335370708)/ref.contact_radius < 1e-12


def test_winkler_penalty_sphere_is_explicitly_not_hertz():
    # Calibrate at one indentation, then compare scaling at 4x indentation.
    Eeff=1e6; radius=.1; d0=1e-3
    hertz0=4/3*Eeff*math.sqrt(radius)*d0**1.5
    penalty=hertz0/(math.pi*radius*d0**2)
    hertz4=4/3*Eeff*math.sqrt(radius)*(4*d0)**1.5
    mismatch=abs(winkler_sphere_load(4*d0,radius,penalty)-hertz4)/hertz4
    assert mismatch > .03  # negative-evidence gate: must remain experimental


def test_topology_and_degenerate_surface_fail_closed():
    slave,sf,master,mf=surfaces(); state=initial_surface_contact_state(slave,sf,master,mf)
    try:
        update_surface_contact(slave,torch.tensor([[0,1,2]],dtype=torch.long),master,mf,state,
            normal_penalty=1.,tangential_penalty=1.,friction=0.)
    except ValueError: pass
    else: raise AssertionError("mismatched state topology accepted")
    bad=slave.clone(); bad[1]=bad[0]
    try: tributary_areas(bad,sf)
    except ValueError: pass
    else: raise AssertionError("degenerate slave triangle accepted")
