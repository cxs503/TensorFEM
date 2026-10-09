#!/usr/bin/env python
"""Reproduce hull mesh, time step, contact penalty and ice-grid sensitivities."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

from tensorfem.suboff_ice_collision import IceImpactConfig,save_result,simulate_impact

ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'docs/assets/suboff-ice'


def run():
    c=IceImpactConfig()
    jobs=[('hull-16',16,c),('hull-24',24,c),('hull-32',32,c),
          ('time-half',32,replace(c,time_safety=c.time_safety/2)),
          ('contact-double',32,replace(c,contact_factor=c.contact_factor*2)),
          ('ice-fine',32,replace(c,ice_nx=24,ice_ny=8))]
    evidence=[]
    solver_hash=hashlib.sha256((ROOT/'src/tensorfem/suboff_ice_collision.py').read_bytes()).hexdigest()
    for name,n,config in jobs:
        path=ASSETS/f'{name}.json';gp=ASSETS/f'geometry-{n}.json'
        start=time.monotonic()
        print(f'Running {name}',flush=True)
        r=simulate_impact(json.loads(gp.read_text()),config)
        r['solver_source_sha256']=solver_hash
        save_result(path,r,gp)
        record={'name':name,'path':path.relative_to(ROOT).as_posix(),
                'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'status':r['status'],'metrics':r['metrics'],'checks':r['checks'],'solver':r['solver']}
        evidence.append(record)
        print(json.dumps({'name':name,'elapsed_s':time.monotonic()-start,
                          'status':r['status'],'metrics':r['metrics'],'checks':r['checks']}),flush=True)
    measures=('peak_contact_force_N','fracture_dissipation_J',
              'maximum_ice_vertical_displacement_m','maximum_hull_gauss_von_mises_Pa')
    comparisons=[]
    for left,right in ((0,1),(1,2),(2,3),(2,4),(2,5)):
        relative={k:abs(evidence[right]['metrics'][k]/evidence[left]['metrics'][k]-1)
                  for k in measures}
        comparisons.append({'from':evidence[left]['name'],'to':evidence[right]['name'],
                            'relative_changes':relative,'all_below_3_percent':all(v<.03 for v in relative.values())})
    record={'schema':'tensorfem.suboff-ice-study/1','physical_accuracy_qualified':False,
            'status':'demonstration; independent physical reference absent',
            'solver_source_sha256':solver_hash,'cases':evidence,'comparisons':comparisons,
            'all_numerical_sensitivity_gates_passed':all(x['all_below_3_percent'] for x in comparisons)}
    (ASSETS/'study.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':run()
