#!/usr/bin/env python
"""Reassemble published v2 fields and audit hashes, stress, energy and fragments."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from tensorfem.ice_sheet import IceSimulationConfig,ice_fragments
from tensorfem.suboff_ice_simulation import build_simulation,evaluate_state
from tensorfem.suboff_ice_collision import _stress

ROOT=Path(__file__).resolve().parents[1]
D=torch.float64

def validate(path):
    r=json.loads(path.read_text());assert r['schema']=='tensorfem.suboff-ice-simulation/2'
    assert r['physical_accuracy_qualified'] is False
    gp=ROOT/r['geometry_file'];assert hashlib.sha256(gp.read_bytes()).hexdigest()==r['geometry_file_sha256']
    assert len(r['source_sha256'])>=8
    for n,h in r['source_sha256'].items():assert hashlib.sha256((ROOT/'src/tensorfem'/n).read_bytes()).hexdigest()==h,n
    s=build_simulation(json.loads(gp.read_text()),IceSimulationConfig(**r['config']))
    q=torch.tensor(r['final_dofs'],dtype=D).reshape(-1);v=torch.tensor(r['final_velocity'],dtype=D).reshape(-1)
    maximum=torch.tensor(r['cohesive_maximum_opening_m'],dtype=D)
    st=evaluate_state(s,q,maximum);stress,vm=_stress(s,q)
    def close(a,b):torch.testing.assert_close(torch.as_tensor(a,dtype=D),torch.as_tensor(b,dtype=D),rtol=1e-9,atol=1e-9)
    close(stress,r['final_stress_top_bottom_local_Pa']);close(vm,r['final_von_mises_top_bottom_Pa'])
    close(s['mass'],r['mass_diagonal']);close(st['damage'],r['cohesive_damage'])
    close(s['nodes'],r['nodes']);close(s['sheet']['thickness'],r['ice_cell_thickness_m'])
    close(q[s['fixed']],torch.zeros(len(s['fixed']),dtype=D))
    close(v[s['fixed']],torch.zeros(len(s['fixed']),dtype=D))
    close(s['physical_ice_mass'].sum(),r['metrics']['physical_ice_mass_kg'])
    close(s['steel_mass'],r['metrics']['steel_mass_kg'])
    initial=torch.zeros_like(maximum);initial[s['sheet']['precracked_points']]=s['law'].failure_opening
    close(evaluate_state(s,torch.zeros_like(q),initial)['dissipation'],r['metrics']['preexisting_crack_energy_J'])
    ns=s['ns'];fr=ice_fragments(s['sheet'],st['damage'],s['physical_ice_mass'],v.reshape(-1,6)[ns:,:3],s['nodes'][ns:]+q.reshape(-1,6)[ns:,:3])
    assert fr==r['fragments'];assert fr['component_count']==r['metrics']['final_ice_component_count']
    h=r['history'][-1];kin=.5*torch.dot(s['mass'],v*v)
    close(kin,h['kinetic_J']);close(st['shell_energy'],h['shell_strain_J']);close(st['bond_energy'],h['cohesive_stored_J'])
    close(st['water_energy'],h['hydrostatic_potential_J']);close(st['contact_energy'],h['contact_stored_J'])
    close(st['dissipation']-r['metrics']['preexisting_crack_energy_J'],h['new_fracture_dissipation_J'])
    close(s['mass'][2::6]@v[2::6],h['vertical_momentum_kg_m_s'])
    for sample in r['history']:
        total=sum(sample[k] for k in ('kinetic_J','shell_strain_J','cohesive_stored_J','contact_stored_J','hydrostatic_potential_J','new_fracture_dissipation_J','drag_dissipation_J'))
        close(abs(total/r['metrics']['initial_kinetic_energy_J']-1),sample['energy_relative_error'])
        close(abs(sample['vertical_momentum_kg_m_s']-r['metrics']['initial_vertical_momentum_kg_m_s']-sample['external_vertical_impulse_N_s'])/abs(r['metrics']['initial_vertical_momentum_kg_m_s']),sample['momentum_relative_error'])
        close(sample['retained_physical_ice_mass_kg'],r['metrics']['physical_ice_mass_kg'])
    pq=torch.tensor(r['peak_contact_dofs'],dtype=D).reshape(-1)
    close(_stress(s,pq)[1],r['peak_contact_von_mises_top_bottom_Pa'])
    close(evaluate_state(s,pq,maximum)['contact_forces'].sum(),r['metrics']['peak_contact_force_N'])
    m=r['metrics'];c=s['config']
    expectations={'connected_vehicle':s['geometry_audit']['passed'],
        'rigid_translation':max(s['rigid_translation_errors'])<1e-10,
        'energy_balance':m['maximum_energy_relative_error']<.03,
        'vertical_momentum_balance':m['maximum_vertical_momentum_relative_error']<1e-8,
        'contact_expectation':m['peak_contact_force_N']>0 if c.contact_expectation=='required' else m['peak_contact_force_N']==0 if c.contact_expectation=='absent' else True,
        'small_penetration':m['maximum_penetration_m']<c.ice_thickness*.03,
        'small_rotation':m['maximum_rotation_rad']<.1,
        'small_ice_deflection':m['maximum_ice_deflection_over_local_thickness']<.5,
        'elastic_steel_gauss_stress':m['maximum_vehicle_gauss_von_mises_Pa']<c.hull_yield_reference,
        'ice_mass_retained':abs(fr['total_retained_mass_kg']-m['physical_ice_mass_kg'])<1e-9*max(m['physical_ice_mass_kg'],1)}
    assert r['checks']==expectations
    assert r['status']==('demonstration-audited' if all(r['checks'].values()) else 'blocked')
    return {'file':str(path.relative_to(ROOT)),'passed':True,'status':r['status'],'physical_accuracy_qualified':False}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('paths',nargs='*',type=Path);a=p.parse_args()
    paths=a.paths or [ROOT/'docs/assets/suboff-ice-v2'/f'{n}.json' for n in ('baseline','mesh-24','time-half','elastic','open-water','floating-precracked')]
    results=[validate(p.resolve()) for p in paths]
    manifest=ROOT/'docs/assets/suboff-ice-v2/report-manifest.json'
    if not a.paths and manifest.exists():
        for name,expected in json.loads(manifest.read_text())['files'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    print(json.dumps({'passed':all(x['passed'] for x in results),'cases':results},indent=2))

if __name__=='__main__':main()
