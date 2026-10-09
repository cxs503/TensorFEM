#!/usr/bin/env python
"""Reject altered fields, stale exports or unsupported full-field claims."""
import json,hashlib
from pathlib import Path
import numpy as np
from tensorfem.sphere_pressure_benchmark import verify_sphere_pressure, sphere_mesh_study
from tensorfem.additional_benchmark_fields import mindlin_fields,cook_fields
from tensorfem.hemisphere_benchmark_fields import hemisphere_fields
ROOT=Path(__file__).resolve().parents[1]


def compare(stored, actual, path=''):
    if isinstance(actual,dict):
        for k,v in actual.items():compare(stored[k],v,path+'/'+k)
    elif isinstance(actual,(list,tuple)):
        a,b=np.asarray(stored),np.asarray(actual)
        if a.dtype.kind in 'biufc' and b.dtype.kind in 'biufc':
            if a.shape!=b.shape or not np.isfinite(a).all() or not np.allclose(a,b,rtol=1e-9,atol=1e-10):
                raise ValueError('unreproduced FE field: '+path)
        else:
            if len(stored)!=len(actual):raise ValueError('incomplete data: '+path)
            for i,(v,w) in enumerate(zip(stored,actual)):compare(v,w,path+'/'+str(i))
    elif isinstance(actual,(int,float)) and not isinstance(actual,bool):
        if not np.isfinite(stored) or not np.isclose(stored,actual,rtol=1e-9,atol=1e-10):raise ValueError('stale metric: '+path)
    elif stored!=actual:raise ValueError('stale qualification: '+path)


def validate(reproduce=True):
    manifest=json.loads((ROOT/'docs/benchmarks/tutorials/additional-evidence-manifest.json').read_text())
    if manifest['all_benchmarks_complete'] or len(manifest['reports'])!=4:raise ValueError('invalid scope')
    for entry in manifest['reports']:
        for filename,digest in entry['files'].items():
            p=ROOT/filename
            if hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError('modified export/evidence: '+filename)
        raw=next(ROOT/f for f in entry['files'] if f.endswith('/results.json'))
        r=json.loads(raw.read_text());cases=r['cases'];name=r['physical_case_id']
        if name=='sphere-pressure-hex8-lame':
            checks=[verify_sphere_pressure(c) for c in cases]
            for c,v in zip(cases,checks):compare(c['verification'],v)
            study=sphere_mesh_study(cases)
            compare(r['mesh_convergence'],study)
            expected='qualified' if study['passed'] else 'blocked'
        elif name in ('mindlin-navier','cook-membrane'):
            if reproduce:
                fn=mindlin_fields if name=='mindlin-navier' else cook_fields
                for c in cases:compare(c,fn(c['mesh']))
            if name=='mindlin-navier':
                trend=all(b['verification']['errors'][k]<a['verification']['errors'][k] for a,b in zip(cases,cases[1:]) for k in a['verification']['errors'])
                expected='qualified' if trend and cases[-1]['verification']['passed'] else 'blocked'
            else:
                if any(c['stress_accuracy_qualified'] for c in cases):raise ValueError('unsupported Cook stress certification')
                errors=[c['verification']['response_relative_error'] for c in cases]
                expected='response-qualified' if errors[-1]<.03 and all(b<a for a,b in zip(errors,errors[1:])) else 'blocked'
        elif name=='hemisphere-hole':
            if r['stress_accuracy_qualified']:raise ValueError('unsupported hemisphere stress certification')
            if reproduce:
                cache={}
                for c in cases:
                    actual=hemisphere_fields(c['mesh'],c['drilling_factor'])
                    cache[(c['mesh'],c['drilling_factor'])]=actual
                    compare(c,actual)
                for mesh,checks in ((40,r['sensitivity']),(24,r['coarse_sensitivity'])):
                    for s in checks:
                        key=(mesh,s['drilling_factor'])
                        if key not in cache:cache[key]=hemisphere_fields(*key)
                        compare(s,cache[key]['validation']|{'drilling_factor':s['drilling_factor']})
            change=abs(cases[-1]['probe']['value']/cases[-2]['probe']['value']-1)
            compare(r['last_response_change'],change)
            expected='response-qualified' if cases[-1]['validation']['passed'] and change<.03 and all(s['passed'] for s in r['sensitivity']) else 'blocked'
        else:raise ValueError('unknown case')
        if r['status']!=expected or entry['status']!=expected:raise ValueError('false report status: '+name)
        print(name,expected,flush=True)
    return True

if __name__=='__main__':validate()
