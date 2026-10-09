#!/usr/bin/env python
"""Reassemble vehicle-only fields, stress, impulses, work and source hashes."""
import hashlib,json
from pathlib import Path
import torch
from tensorfem.coupled_hull import LinearHullReceiver,HullCouplingConfig
from tensorfem.suboff_ice_collision import _stress

ROOT=Path(__file__).resolve().parents[1]

def main():
    a=ROOT/'docs/assets/hull-coupling';gp=ROOT/'docs/assets/suboff-ice-v2/appended-16.json'
    cases=[]
    for name in ('baseline','time-half'):
        r=json.loads((a/(name+'.json')).read_text())
        for n,h in r['source_sha256'].items():assert hashlib.sha256((ROOT/'src/tensorfem'/n).read_bytes()).hexdigest()==h,n
        assert hashlib.sha256(gp.read_bytes()).hexdigest()==r['geometry_file_sha256']
        b=LinearHullReceiver(json.loads(gp.read_text()),HullCouplingConfig(**r['config']));b.restore(r['restart'])
        def close(x,y):torch.testing.assert_close(torch.as_tensor(x,dtype=torch.float64),torch.as_tensor(y,dtype=torch.float64),rtol=1e-10,atol=1e-10)
        close(b.mass,r['mass_diagonal']);close(b.nodes,r['nodes_m'])
        stress,vm=_stress({'ids':b.ids,'stress_operator':b.stress_operator},b.q)
        close(stress,r['final_stress_top_bottom_Pa']);close(vm,r['final_gauss_von_mises_Pa'])
        diagnostics=b.diagnostics()
        for key,value in diagnostics.items():
            if isinstance(value,bool):assert value==r['history'][-1][key]
            else:close(value,r['history'][-1][key])
        assert b.surface().motion_record(b.v.reshape(-1,6),time_s=b.time_s)==r['motion']
        assert r['physical_accuracy_qualified'] is False
        assert r['status']=='interface-verified' and all(r['checks'].values())
        for row in r['history']:close(row['kinetic_J']+row['strain_J']-row['external_work_J'],row['energy_residual_J'])
        cases.append({'case':name,'raw_field_audit':'passed','physical_accuracy_qualified':False})
    print(json.dumps({'schema':'tensorfem.hull-coupling-audit/1','cases':cases},indent=2))

if __name__=='__main__':main()
