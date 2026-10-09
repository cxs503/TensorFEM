"""Independent geometry, interface work, action-reaction and stability gates."""
import math
from dataclasses import replace
from pathlib import Path
import json
import pytest
import torch

from tensorfem.cohesive import BilinearCohesiveLaw
from tensorfem.suboff_geometry import audit_closed_quads, closed_revolved_mesh
from tensorfem.suboff_ice_collision import (
    IceImpactConfig, build_impact_system, cohesive_response,
    conservative_step_bound, contact_response,
    simulate_impact,
)

D=torch.float64


def test_closed_quad_caps_match_independent_ellipsoid():
    profile=lambda x: math.sqrt(max(0,1-(2*x-1)**2))
    nodes,cells,audit=closed_revolved_mesh(profile,4.,.5,16,16)
    assert audit['passed'] and audit['euler_characteristic']==2
    assert torch.max(torch.abs(((nodes[:,0]-2)/2)**2+
                               (nodes[:,1:]**2).sum(1)/.25-1))<1e-12
    exact=4/3*math.pi*2*.5**2
    assert abs(audit['enclosed_volume_m3']/exact-1)<.06
    broken=cells.clone();broken[0]=broken[0].flip(0)
    assert not audit_closed_quads(nodes,broken)['passed']


def interface():
    ids=torch.arange(12).reshape(1,12)
    B=torch.zeros((1,3,12),dtype=D)
    B[0,:,:3]=-torch.eye(3,dtype=D);B[0,:,6:9]=torch.eye(3,dtype=D)
    return ids,B,torch.tensor([[1.,0.,0.]],dtype=D),torch.tensor([2.],dtype=D)


def test_cohesive_fracture_work_unloading_and_compression():
    ids,B,n,area=interface();law=BilinearCohesiveLaw(1e9,1e5,100.)
    h=torch.zeros(1,dtype=D);q=torch.zeros(12,dtype=D)
    work=0.;previous_force=0.;previous_gap=0.
    for gap in torch.linspace(0,law.failure_opening,401,dtype=D):
        q[6]=gap
        force,stored,dissipation,damage,h=cohesive_response(q,ids,B,n,area,law,h)
        assert torch.linalg.vector_norm(force[:3]+force[6:9])<1e-10
        work+=.5*(float(force[6])+previous_force)*(float(gap)-previous_gap)
        previous_force=float(force[6]);previous_gap=float(gap)
    assert float(damage)==1 and float(stored)==0
    assert abs(work/(2*100)-1)<1e-4
    assert abs(float(dissipation)-200)<1e-10
    q[6]=law.onset_opening/2
    force,stored,dissipation,damage,h=cohesive_response(q,ids,B,n,area,law,h)
    assert float(force.abs().max())==0 and float(dissipation)==200
    q[6]=-law.onset_opening
    force,stored,dissipation,damage,h=cohesive_response(q,ids,B,n,area,law,h)
    assert float(force[6])==pytest.approx(-2*law.peak_traction)
    assert float(dissipation)==200


def test_frictionless_contact_work_and_action_reaction():
    q=torch.tensor([.03,0.,.01,0.,0.],dtype=D)
    ids=torch.arange(5).reshape(1,5)
    w=torch.tensor([[1.,-.25,-.25,-.25,-.25]],dtype=D)
    gap=torch.tensor([.01],dtype=D);k=torch.tensor([1000.],dtype=D)
    f,energy,force,penetration=contact_response(q,ids,w,gap,k)
    assert float(f.sum())==pytest.approx(0,abs=1e-14)
    assert float(energy)==pytest.approx(.5*1000*.0175**2)
    eps=1e-7
    for i in range(5):
        delta=torch.zeros_like(q);delta[i]=eps
        plus=contact_response(q+delta,ids,w,gap,k)[1]
        minus=contact_response(q-delta,ids,w,gap,k)[1]
        assert float((plus-minus)/(2*eps))==pytest.approx(float(f[i]),rel=1e-8)
    assert float(penetration)==pytest.approx(.0175)


def test_gershgorin_bound_is_conservative_for_independent_eigenvalues():
    K=torch.tensor([[4.,-2.,0.],[-2.,5.,-1.],[0.,-1.,2.]],dtype=D)
    m=torch.tensor([2.,3.,1.],dtype=D);free=torch.tensor([False,True,True])
    dt=conservative_step_bound(K.to_sparse(),m,free)
    scaled=K[free][:,free]/torch.sqrt(m[free,None]*m[None,free])
    exact=2/math.sqrt(float(torch.linalg.eigvalsh(scaled).max()))
    assert 0<dt<=exact
    with pytest.raises(ValueError):
        conservative_step_bound(K.to_sparse(),torch.zeros(3,dtype=D),free)


def test_exported_suboff_and_discontinuous_ice_mass_rigid_motion():
    path=Path(__file__).resolve().parents[1]/'docs/assets/suboff-ice/geometry-16.json'
    geometry=json.loads(path.read_text())
    assert geometry['maximum_source_profile_error_m']<1e-12
    c=IceImpactConfig(ice_nx=4,ice_ny=2)
    s=build_impact_system(geometry,c)
    assert max(s['rigid_translation_errors'])<1e-12
    # The ice grid has no shared corner IDs: failed seams really allow separation.
    assert s['ice_cells'].unique().numel()==s['ice_cells'].numel()
    mass=s['mass'].reshape(-1,6)
    expected=(geometry['length_m']+2*c.ice_margin)*c.ice_width*c.ice_thickness*c.ice_density
    assert float(mass[s['nh']:,2].sum())==pytest.approx(expected,rel=1e-12)
    q=torch.zeros_like(s['mass']);q[2::6]=.001
    f,stored,dissipation,damage,h=cohesive_response(q,s['bid'],s['B'],s['bnormal'],s['barea'],s['law'],
        torch.zeros(len(s['bid']),dtype=D))
    assert float(f.abs().max())<1e-12 and float(stored)==0 and float(damage.max())==0


@pytest.mark.parametrize('kwargs',[{'ice_thickness':float('nan')},{'time_safety':1.},
                                  {'initial_gap':-1},{'ice_nx':2.5}])
def test_invalid_parameters_fail_closed(kwargs):
    with pytest.raises(ValueError):
        IceImpactConfig(**kwargs).validate()


def test_precontact_free_motion_matches_exact_rigid_translation():
    path=Path(__file__).resolve().parents[1]/'docs/assets/suboff-ice/geometry-16.json'
    geometry=json.loads(path.read_text())
    c=replace(IceImpactConfig(),duration=.00005,sample_count=4)
    result=simulate_impact(geometry,c)
    q=torch.tensor(result['final_dofs'],dtype=D)
    v=torch.tensor(result['final_velocity'],dtype=D)
    nh=len(geometry['nodes'])
    assert torch.max(torch.abs(q[:nh,2]-c.upward_speed*c.duration))<1e-13
    assert torch.max(torch.abs(v[:nh,2]-c.upward_speed))<1e-10
    assert float(q[:nh,:2].abs().max())<1e-13
    assert result['metrics']['peak_contact_force_N']==0
    assert result['metrics']['maximum_energy_relative_error']<1e-11
    assert result['metrics']['hull_mass_kg']==pytest.approx(700)
    assert result['status']=='blocked' and not result['checks']['contact_occurred']
