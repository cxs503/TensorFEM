"""Independent floating equilibrium, interface geometry, and fragmentation tests."""
from dataclasses import replace
import pytest
import torch
from tensorfem.ice_sheet import IceSimulationConfig,build_ice_sheet,buoyancy_response,ice_fragments

D=torch.float64


def test_nonuniform_thickness_holes_and_mass_are_explicit():
    c=IceSimulationConfig(ice_nx=3,ice_ny=2,ice_thickness_map=(.01,.02,0,.015,.01,.03),ice_boundary='free',hydrostatic_buoyancy=False)
    s=build_ice_sheet(4.,1.,c)
    assert s['active_grid_ids'].tolist()==[0,1,3,4,5]
    assert s['fixed_dofs'].numel()==0
    assert s['elements'].unique().numel()==s['elements'].numel()
    z=s['nodes'][:,2]+s['node_thickness']/2
    torch.testing.assert_close(z,torch.ones_like(z))
    area=(4+2*c.ice_margin)/3*c.ice_width/2
    mass=c.ice_density*s['node_area']*s['node_thickness']
    assert float(mass.sum())==pytest.approx(c.ice_density*area*(.01+.02+.015+.01+.03))


def test_stepped_thickness_seams_preserve_rigid_rotation():
    c=IceSimulationConfig(ice_nx=2,ice_ny=2,ice_thickness_map=(.01,.02,.03,.015))
    s=build_ice_sheet(2.,1.,c);omega=torch.tensor([.1,.2,-.3],dtype=D)
    q=torch.zeros((len(s['nodes']),6),dtype=D)
    q[:,:3]=torch.linalg.cross(omega.expand_as(s['nodes']),s['nodes'],dim=1);q[:,3:]=omega
    jump=torch.einsum('pij,pj->pi',s['bond_B'],q.reshape(-1)[s['bond_ids']])
    assert float(jump.abs().max())<1e-15


def test_nonuniform_floating_ice_has_one_waterline_and_archimedean_draft():
    c=IceSimulationConfig(ice_nx=2,ice_ny=2,ice_thickness_map=(.01,.02,.03,.015))
    s=build_ice_sheet(2.,1.,c)
    bottom=s['nodes'][:,2]-s['node_thickness']/2
    draft=s['waterline_m']-bottom
    torch.testing.assert_close(draft,c.ice_density/c.water_density*s['node_thickness'])


def test_clipped_buoyancy_matches_archimedes_and_potential_gradient():
    c=IceSimulationConfig();h=torch.tensor([.01,.02,.03],dtype=D);area=torch.tensor([.4,.3,.2],dtype=D)
    w=torch.tensor([.0002,.03,-.04],dtype=D)
    force,energy=buoyancy_response(w,area,h,c)
    draft=c.ice_density/c.water_density*h
    submerged=torch.clamp(draft-w,min=0);submerged=torch.minimum(submerged,h)
    exact=c.gravity*area*(c.ice_density*h-c.water_density*submerged)
    torch.testing.assert_close(force,exact,rtol=1e-12,atol=1e-12)
    assert float(energy)>0
    for i in range(3):
        delta=torch.zeros_like(w);delta[i]=1e-7
        derivative=(buoyancy_response(w+delta,area,h,c)[1]-buoyancy_response(w-delta,area,h,c)[1])/2e-7
        assert float(derivative)==pytest.approx(float(force[i]),rel=1e-8)
    zero=torch.zeros_like(w)
    f,u=buoyancy_response(zero,area,h,c)
    assert float(f.abs().max())==0 and float(u)==0


def test_precrack_separates_two_fragments_without_losing_mass():
    pairs=((0,3),(1,4),(2,5))
    c=IceSimulationConfig(ice_nx=3,ice_ny=2,precracked_pairs=pairs)
    s=build_ice_sheet(4.,1.,c);damage=s['precracked_points'].to(D)
    mass=c.ice_density*s['node_thickness']*s['node_area'];v=torch.zeros_like(s['nodes']);v[:,2]=.1
    r=ice_fragments(s,damage,mass,v,s['nodes'])
    assert r['component_count']==2 and len(r['fully_severed_seams'])==3
    assert r['total_retained_mass_kg']==pytest.approx(float(mass.sum()))
    assert sum(p['translational_kinetic_energy_J'] for p in r['fragments'])==pytest.approx(.5*float(mass.sum())*.1**2)
    assert all(p['mean_velocity_m_s'][2]==pytest.approx(.1) for p in r['fragments'])


def test_free_clamped_and_simply_supported_are_different_boundaries():
    c=IceSimulationConfig(ice_nx=2,ice_ny=2)
    clamped=build_ice_sheet(2.,1.,c)['fixed_dofs']
    simple=build_ice_sheet(2.,1.,replace(c,ice_boundary='simply_supported'))['fixed_dofs']
    free=build_ice_sheet(2.,1.,replace(c,ice_boundary='free'))['fixed_dofs']
    assert len(clamped)==6*len(simple)>0 and len(free)==0
    assert bool((simple%6==2).all())


@pytest.mark.parametrize('kwargs',[{'ice_boundary':'bad'},{'ice_model':'bad'},
    {'water_density':800.},{'ice_thickness_map':(.01,)},{'open_water_cells':(-1,)},
    {'precracked_pairs':((0,5),)},{'water_added_mass_per_area':-1},
    {'ice_model':'elastic','precracked_pairs':((0,1),)}])
def test_invalid_ice_settings_fail_closed(kwargs):
    with pytest.raises(ValueError):IceSimulationConfig(**kwargs).validate()
