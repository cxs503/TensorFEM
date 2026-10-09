#!/usr/bin/env python
"""Recompute constitutive forces, stresses, final energies and impulse balances."""
import hashlib
import json
from pathlib import Path
import torch

from tensorfem.suboff_ice_collision import (
    IceImpactConfig,_stress,build_impact_system,cohesive_response,contact_response,
)

ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'docs/assets/suboff-ice'


def validate_case(record,geometry):
    if record.get('physical_accuracy_qualified') is not False:
        raise ValueError('an artificial collision has no independent physical accuracy qualification')
    c=IceImpactConfig(**record['config']);s=build_impact_system(geometry,c)
    q=torch.tensor(record['final_dofs'],dtype=torch.float64).reshape(-1)
    v=torch.tensor(record['final_velocity'],dtype=torch.float64).reshape(-1)
    maximum=torch.tensor(record['cohesive_maximum_opening_m'],dtype=torch.float64)
    if not bool(torch.isfinite(q).all() and torch.isfinite(v).all() and torch.isfinite(maximum).all()):
        raise ValueError('nonfinite final fields')
    if q.shape!=s['mass'].shape or v.shape!=q.shape or maximum.shape!=(len(s['bid']),):
        raise ValueError('field layout differs from reconstructed system')
    mass=torch.tensor(record['mass_diagonal'],dtype=torch.float64)
    torch.testing.assert_close(mass,s['mass'],rtol=1e-12,atol=1e-12)
    assert float(q[s['fixed']].abs().max())==0 and float(v[s['fixed']].abs().max())==0
    internal=torch.mv(s['shell'],q);se=.5*torch.dot(q,internal)
    bf,be,diss,damage,h=cohesive_response(q,s['bid'],s['B'],s['bnormal'],s['barea'],s['law'],maximum)
    cf,ce,force,pen=contact_response(q,s['cid'],s['cw'],s['gaps'],s['cp'])
    stress,vm=_stress(s,q)
    torch.testing.assert_close(stress,torch.tensor(record['final_stress_top_bottom_local_Pa'],dtype=torch.float64),rtol=1e-10,atol=1e-6)
    torch.testing.assert_close(vm,torch.tensor(record['final_von_mises_top_bottom_Pa'],dtype=torch.float64),rtol=1e-10,atol=1e-6)
    torch.testing.assert_close(damage,torch.tensor(record['cohesive_damage'],dtype=torch.float64),rtol=1e-12,atol=1e-12)
    peak_q=torch.tensor(record['peak_contact_dofs'],dtype=torch.float64).reshape(-1)
    ps,pvm=_stress(s,peak_q)
    torch.testing.assert_close(pvm,torch.tensor(record['peak_contact_von_mises_top_bottom_Pa'],dtype=torch.float64),rtol=1e-10,atol=1e-6)
    kinetic=.5*torch.dot(mass,v*v)
    initial=.5*c.hull_total_mass*c.upward_speed**2
    impulse=record['metrics']['support_vertical_impulse_N_s']
    momentum=float(torch.dot(mass[2::6],v[2::6]))
    last=record['history'][-1]
    for name,value in (('kinetic_J',kinetic),('shell_strain_J',se),('cohesive_stored_J',be),
                       ('contact_stored_J',ce),('fracture_dissipation_J',diss),
                       ('contact_force_N',force.sum()),('vertical_momentum_kg_m_s',momentum)):
        if abs(float(value)-last[name])>1e-8*max(abs(float(value)),1):
            raise ValueError(f'{name} differs from reconstructed raw fields')
    if record['solver']['time_step_s']>c.time_safety*s['step_bound']*(1+1e-12):
        raise ValueError('stored step exceeds conservative stability gate')
    if abs(last['time_s']-c.duration)>1e-12:
        raise ValueError('incomplete simulation duration')
    energy_keys=('kinetic_J','shell_strain_J','cohesive_stored_J','contact_stored_J','fracture_dissipation_J')
    for point in record['history']:
        ee=abs(sum(point[k] for k in energy_keys)/initial-1)
        me=abs(point['vertical_momentum_kg_m_s']-c.hull_total_mass*c.upward_speed-
               point['support_vertical_impulse_N_s'])/(c.hull_total_mass*c.upward_speed)
        if abs(ee-point['energy_relative_error'])>1e-12 or abs(me-point['momentum_relative_error'])>1e-12:
            raise ValueError('history error flags differ from raw energy/impulse values')
    if record['metrics']['maximum_energy_relative_error']<max(p['energy_relative_error'] for p in record['history'])-1e-12:
        raise ValueError('reported maximum error misses a stored sample')
    dissipations=[x['fracture_dissipation_J'] for x in record['history']]
    if any(b<a-1e-12 for a,b in zip(dissipations,dissipations[1:])):
        raise ValueError('fracture dissipation decreased')
    final_energy_error=abs(float(kinetic+se+be+ce+diss)/initial-1)
    momentum_error=abs(momentum-c.hull_total_mass*c.upward_speed-impulse)/(c.hull_total_mass*c.upward_speed)
    if final_energy_error>.03 or momentum_error>1e-8:
        raise ValueError('reconstructed energy/momentum gate failed')
    if bool((h>maximum+1e-12).any()):
        raise ValueError('cohesive state lost maximum opening history')
    if not all(record['checks'].values()) or record['status']!='demonstration-audited':
        raise ValueError(f"case applicability/numerical gates failed: {record['checks']}")
    return {'final_energy_relative_error':final_energy_error,
            'final_vertical_momentum_relative_error':momentum_error,
            'fracture_dissipation_J':float(diss),'fully_failed_points':int((damage>=.999).sum())}


def validate():
    study=json.loads((ASSETS/'study.json').read_text())
    expected_hash=hashlib.sha256((ROOT/'src/tensorfem/suboff_ice_collision.py').read_bytes()).hexdigest()
    if study['solver_source_sha256']!=expected_hash:
        raise ValueError('solver changed since evidence generation')
    results=[]
    for entry in study['cases']:
        path=ROOT/entry['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
            raise ValueError('case hash mismatch')
        record=json.loads(path.read_text());n=int(entry['name'].split('-')[1]) if entry['name'].startswith('hull-') else 32
        gp=ASSETS/f'geometry-{n}.json'
        if hashlib.sha256(gp.read_bytes()).hexdigest()!=record['geometry_file_sha256']:
            raise ValueError('geometry hash mismatch')
        if record['solver_source_sha256']!=expected_hash:
            raise ValueError('case solver hash mismatch')
        results.append({'case':entry['name'],**validate_case(record,json.loads(gp.read_text()))})
    # Numerical sensitivity is evidence, not proof of correctness. Recompute all
    # reported comparisons and explicitly preserve failures in the study.
    by_name={e['name']:e for e in study['cases']}
    for comparison in study['comparisons']:
        a=by_name[comparison['from']]['metrics'];b=by_name[comparison['to']]['metrics']
        fresh={k:abs(b[k]/a[k]-1) for k in comparison['relative_changes']}
        if any(abs(fresh[k]-comparison['relative_changes'][k])>1e-12 for k in fresh):
            raise ValueError('stored convergence comparisons differ')
        if comparison['all_below_3_percent']!=all(v<.03 for v in fresh.values()):
            raise ValueError('stored convergence status differs')
    if study['all_numerical_sensitivity_gates_passed']!=all(x['all_below_3_percent'] for x in study['comparisons']):
        raise ValueError('aggregated convergence status differs')
    manifest=ASSETS/'report-manifest.json'
    if manifest.exists():
        report=json.loads(manifest.read_text())
        if report['physical_accuracy_qualified'] is not False:
            raise ValueError('report falsely claims physical accuracy qualification')
        for relative,digest in report['artifacts'].items():
            if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=digest:
                raise ValueError(f'report artifact hash mismatch: {relative}')
    print(json.dumps({'raw_field_validation':'passed','physical_accuracy_qualified':False,
                      'all_numerical_sensitivity_gates_passed':study['all_numerical_sensitivity_gates_passed'],
                      'cases':results},indent=2))


if __name__=='__main__':validate()
