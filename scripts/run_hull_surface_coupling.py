#!/usr/bin/env python
"""Execute vehicle-only hull exchange verification; no fluid/ice solve implied."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch
from tensorfem.coupled_hull import LinearHullReceiver
from tensorfem.surface_coupling import validate_physics_ownership
from tensorfem.suboff_ice_collision import _stress

ROOT=Path(__file__).resolve().parents[1]

def simulate(g,factor,*,restart=False):
    body=LinearHullReceiver(g);duration=24*body.dt_bound
    steps=math.ceil(duration/(factor*body.dt_bound));dt=duration/steps
    history=[];max_force=max_moment=max_power=0.;peak_vm=0.;peak_energy=0.
    comparison=None
    for step in range(steps):
        surface=body.surface();t=torch.zeros_like(surface.positions)
        t[body.labels.repeat_interleave(4)==1,2]=10000.
        load=surface.tractions(t,time_s=body.time_s,source='verification')
        audit=surface.audit(load,body.v.reshape(-1,6),expected_time_s=body.time_s)
        max_force=max(max_force,audit['force_transfer_error_N'])
        max_moment=max(max_moment,audit['moment_transfer_error_N_m'])
        max_power=max(max_power,audit['power_transfer_error_W'])
        body.advance(surface,load,dt)
        d=body.diagnostics();history.append(d)
        peak_vm=max(peak_vm,d['maximum_gauss_von_mises_Pa'])
        peak_energy=max(peak_energy,abs(d['energy_residual_J']))
        if restart and step+1==steps//2:
            comparison=LinearHullReceiver(g)
            comparison.restore(json.loads(json.dumps(body.snapshot())))
        elif comparison is not None:
            s=comparison.surface();f=s.tractions(t,time_s=comparison.time_s,source='verification')
            comparison.advance(s,f,dt)
    restart_equal=(torch.equal(body.q,comparison.q) and torch.equal(body.v,comparison.v)
                   and body.snapshot()==comparison.snapshot()) if comparison else None
    stress,vm=_stress({'ids':body.ids,'stress_operator':body.stress_operator},body.q)
    final=history[-1];scale=max(abs(final['external_work_J']),1e-15)
    checks={'force_mapping':max_force<1e-9,'moment_mapping':max_moment<1e-9,
            'power_mapping':max_power<1e-9,'momentum_balance':final['momentum_residual_kg_m_s']<1e-9,
            'energy_balance':peak_energy/scale<.03,'small_rotation':final['maximum_rotation_rad']<.1,
            'vehicle_only':final['ice_elements']==0}
    if restart:checks['restart_exact']=restart_equal
    return {'schema':'tensorfem.hull-surface-verification/1','physical_accuracy_qualified':False,
            'status':'interface-verified' if all(checks.values()) else 'blocked','checks':checks,
            'scope':'synthetic sail traction on vehicle-only small-strain shell; not actual CFD/DEM load',
            'time_step_s':dt,'steps':steps,'config':body.snapshot()['config'],
            'metrics':{'maximum_force_transfer_error_N':max_force,'maximum_moment_transfer_error_N_m':max_moment,
                       'maximum_power_transfer_error_W':max_power,'maximum_gauss_von_mises_Pa':peak_vm,
                       'maximum_energy_residual_J':peak_energy,'maximum_energy_relative_residual':peak_energy/scale},
            'nodes_m':body.nodes.tolist(),'elements':body.elements.tolist(),
            'mass_diagonal':body.mass.tolist(),'final_stress_top_bottom_Pa':stress.tolist(),
            'final_gauss_von_mises_Pa':vm.tolist(),'history':history,'restart':body.snapshot(),
            'motion':body.surface().motion_record(body.v.reshape(-1,6),time_s=body.time_s)}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'docs/assets/hull-coupling');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    gp=ROOT/'docs/assets/suboff-ice-v2/appended-16.json';g=json.loads(gp.read_text())
    files=('surface_coupling.py','coupled_hull.py','suboff_ice_collision.py','suboff_ice_simulation.py','spherical_shell.py','plate.py')
    hashes={n:hashlib.sha256((ROOT/'src/tensorfem'/n).read_bytes()).hexdigest() for n in files}
    cases={}
    for name,factor in (('baseline',1.),('time-half',.5)):
        r=simulate(g,factor,restart=name=='baseline');r['source_sha256']=hashes
        r['geometry_file_sha256']=hashlib.sha256(gp.read_bytes()).hexdigest()
        out=a.output/(name+'.json');out.write_text(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n');cases[name]=r
    x=cases['baseline']['metrics']['maximum_gauss_von_mises_Pa'];y=cases['time-half']['metrics']['maximum_gauss_von_mises_Pa']
    study={'schema':'tensorfem.hull-coupling-study/1','physical_accuracy_qualified':False,
           'resolved_fluid_in_this_case':False,'ice_solver_in_this_case':None,
           'intended_coupled_ownership':validate_physics_ownership(ice_owners=['TensorDEM'],resolved_fluid=True,simplified_water_terms=[]),
           'time_half_peak_stress_change_percent':abs(y/x-1)*100,
           'cases':{n:{'status':r['status'],'checks':r['checks'],'metrics':r['metrics']} for n,r in cases.items()}}
    (a.output/'study.json').write_text(json.dumps(study,indent=2)+'\n');print(json.dumps(study,indent=2))

if __name__=='__main__':main()
