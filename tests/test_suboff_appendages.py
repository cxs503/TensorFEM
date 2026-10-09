"""Geometry/load-path and free-motion checks for structural appendages."""
from dataclasses import replace
import json
from pathlib import Path
import pytest
import torch
from tensorfem.ice_sheet import IceSimulationConfig
from tensorfem.suboff_appendages import audit_appended_mesh
from tensorfem.suboff_ice_simulation import build_simulation,simulate

ROOT=Path(__file__).resolve().parents[1]


def geometry():
    return json.loads((ROOT/'docs/assets/suboff-ice-v2/appended-16.json').read_text())


def test_five_appendages_are_welded_to_the_hull_and_have_positive_stiffness():
    g=geometry();x=torch.tensor(g['nodes'],dtype=torch.float64);e=torch.tensor(g['elements'])
    a=audit_appended_mesh(x,e,g['body_node_count'],g['body_element_count'],g['components'])
    assert a['passed'] and a['appendage_count']==5 and a['all_nodes_connected']
    s=build_simulation(g,IceSimulationConfig(duration=.00002,ice_nx=4,ice_ny=2))
    assert max(s['rigid_translation_errors'])<1e-12
    for part in g['components'][1:]:
        assert all(i<g['body_node_count'] for i in part['root_nodes'])
        assert bool((s['labels']==part['id']).any())
        assert float(s['mass'].reshape(-1,6)[s['ns']-1,2])>0
    u=torch.zeros_like(s['mass'])
    sail_nodes=torch.unique(s['structural_cells'][s['labels']==1])
    roots=torch.tensor(g['components'][1]['root_nodes'])
    root_z=float(s['nodes'][roots,2].max())
    u[6*sail_nodes+2]=1e-5*(s['nodes'][sail_nodes,2]-root_z)
    u[6*roots+2]=0
    action=torch.mv(s['shell'],u)
    roots=torch.tensor(g['components'][1]['root_nodes'])
    assert float(action.reshape(-1,6)[roots].abs().max())>0
    assert float(.5*torch.dot(u,action))>0


def test_exact_precontact_vehicle_translation_includes_all_appendages():
    c=IceSimulationConfig(duration=.00002,sample_count=3,ice_nx=4,ice_ny=2,contact_expectation='absent')
    r=simulate(geometry(),c);ns=len(r['outer_vehicle_nodes']);q=torch.tensor(r['final_dofs'])
    assert float((q[:ns,2]-c.duration*c.upward_speed).abs().max())<1e-12
    assert r['metrics']['peak_contact_force_N']==0
    assert r['metrics']['maximum_energy_relative_error']<1e-10
    assert r['status']=='demonstration-audited'


def test_open_water_under_roof_has_no_hidden_ice_contact():
    g=geometry();c=IceSimulationConfig(duration=.00002,ice_nx=4,ice_ny=2,open_water_cells=(0,1,4,5),contact_expectation='absent')
    s=build_simulation(g,c)
    assert not bool((s['contact_components']==1).any())
    assert 0 not in s['sheet']['cell_lookup'] and 4 not in s['sheet']['cell_lookup']
