#!/usr/bin/env python
"""One-way replay of audited real LBM/DEM resultants; not a spatial pressure map."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from tensorfem.coupled_hull import LinearHullReceiver
from tensorfem.recorded_hull_loads import embed_wrench,distribute_wrench,LinearMidpointIntegrator
from tensorfem.surface_coupling import PlanarEmbedding
from tensorfem.suboff_ice_collision import _stress
ROOT=Path(__file__).resolve().parents[1]
D=torch.float64

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def simulate(body,records,source_dt,factor):
    surface=body.surface();mask=body.labels.repeat_interleave(4)==1
    origin=(surface.positions[mask]*surface.areas[mask,None]).sum(0)/surface.areas[mask].sum()
    embedding=PlanarEmbedding(origin,[[1.,0.],[0.,0.],[0.,1.]])
    templates=torch.stack([distribute_wrench(surface,torch.eye(6,dtype=D)[i],mask) for i in range(6)])
    model=LinearMidpointIntegrator(body.mass,body.shell,source_dt/factor)
    history=[];max_wrench=max_power=0.;impulse=torch.zeros(3,dtype=D);restart=None
    for index,record in enumerate(records):
        contributions=[embed_wrench(record,k,embedding) for k in ('fluid','ice_contact','other')]
        w=sum(contributions);total=embed_wrench(record,'total',embedding)
        if not torch.allclose(w,total,atol=1e-11,rtol=1e-11):raise ValueError('duplicated or inconsistent source ledger')
        sample_force=torch.einsum('i,ipj->pj',w,templates)
        for sub in range(factor):
            load=surface.point_loads(sample_force,time_s=model.time_s,source='TensorLBM')
            a=surface.audit(load,model.v.reshape(-1,6),expected_time_s=model.time_s)
            got=torch.tensor(a['force_N']+a['moment_N_m'],dtype=D)
            max_wrench=max(max_wrench,float(torch.linalg.vector_norm(got-w)))
            max_power=max(max_power,a['power_transfer_error_W'])
            f=load.generalized_force.flatten();model.step(f);impulse+=model.dt*w[:3]
            if restart is not None:restart.step(f)
            if index==len(records)//2-1 and sub==factor-1:
                restart=LinearMidpointIntegrator(body.mass,body.shell,model.dt)
                restart.restore(json.loads(json.dumps(model.snapshot())))
        stress,vm=_stress({'ids':body.ids,'stress_operator':body.stress_operator},model.q)
        momentum=(body.mass.reshape(-1,6)[:,:3]*model.v.reshape(-1,6)[:,:3]).sum(0)
        history.append({'source_interval_end_s':record['time_s'],'time_s':model.time_s,
                        'fluid_wrench_N_Nm':contributions[0].tolist(),'ice_wrench_N_Nm':contributions[1].tolist(),
                        'other_wrench_N_Nm':contributions[2].tolist(),'total_wrench_N_Nm':w.tolist(),
                        'kinetic_J':float(.5*body.mass@(model.v*model.v)),
                        'strain_J':float(.5*model.q@(model.K@model.q)),
                        'work_J':model.work_J,'energy_residual_J':model.energy()-model.work_J,
                        'momentum_kg_m_s':momentum.tolist(),'impulse_N_s':impulse.tolist(),
                        'momentum_residual_kg_m_s':float(torch.linalg.vector_norm(momentum-impulse)),
                        'maximum_von_mises_Pa':float(vm.max())})
    maximum_energy=max(abs(h['energy_residual_J']) for h in history)
    energy_scale=max(max(abs(h['work_J']) for h in history),1e-15)
    metrics={'wrench_error_N_Nm':max_wrench,'power_mapping_error_W':max_power,
             'maximum_energy_residual_J':maximum_energy,'relative_energy_residual':maximum_energy/energy_scale,
             'maximum_momentum_residual_kg_m_s':max(h['momentum_residual_kg_m_s'] for h in history),
             'peak_von_mises_Pa':max(h['maximum_von_mises_Pa'] for h in history),
             'maximum_rotation_rad':float(model.q.reshape(-1,6)[:,3:].abs().max()),
             'maximum_displacement_m':float(torch.linalg.vector_norm(model.q.reshape(-1,6)[:,:3],dim=1).max())}
    checks={'wrench':max_wrench<1e-9,'power':max_power<1e-9,'energy':maximum_energy/energy_scale<1e-6,
            'momentum':metrics['maximum_momentum_residual_kg_m_s']<1e-8,
            'restart_exact':restart.snapshot()==model.snapshot(),'small_rotation':metrics['maximum_rotation_rad']<.01}
    return {'schema':'tensorfem.recorded-hull-replay/1','physical_accuracy_qualified':False,
            'checks':checks,'metrics':metrics,'config':body.snapshot()['config'],
            'dt_s':model.dt,'source_interval_dt_s':source_dt,'steps':len(records)*factor,
            'embedding_origin_m':origin.tolist(),'embedding_axes':embedding.axes.tolist(),
            'spatial_load_policy':'area-weighted minimum-norm resultant redistribution on reference sail; not pressure reconstruction',
            'motion_feedback':False,'reference_geometry_frozen':True,'source_interpolation':'piecewise constant interval mean',
            'history':history,'restart':model.snapshot(),'final_stress_Pa':stress.tolist(),
            'final_von_mises_Pa':vm.tolist()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'docs/assets/recorded-hull-replay');a=p.parse_args()
    source=json.loads(a.source.read_text());records=source['state']['history'];dt=source['summary']['config']['fluid_dt_s']
    if not records:raise ValueError('empty load history')
    for i,r in enumerate(records):
        if abs(r['time_s']-(i+1)*dt)>1e-11:raise ValueError('inconsistent source interval')
    gp=ROOT/'docs/assets/suboff-ice-v2/appended-16.json';geometry=json.loads(gp.read_text())
    body=LinearHullReceiver(geometry);a.output.mkdir(parents=True,exist_ok=True)
    # Compact exact source ledger retains every wrench component and sample clock.
    selected=[{k:v for k,v in r.items() if k=='time_s' or any(k.startswith(t+'_') for t in ('fluid','ice_contact','other','total')) and k.endswith(('_n','_nm'))} for r in records]
    ledger={'original_artifact_sha256':sha(a.source),'source_dt_s':dt,'records':selected}
    ledgerpath=a.output/'source-ledger.json';ledgerpath.write_text(json.dumps(ledger,indent=2)+'\n')
    files=['src/tensorfem/recorded_hull_loads.py','src/tensorfem/coupled_hull.py','src/tensorfem/surface_coupling.py',
           'src/tensorfem/suboff_ice_collision.py','src/tensorfem/suboff_ice_simulation.py','src/tensorfem/spherical_shell.py','src/tensorfem/plate.py','scripts/run_recorded_hull_replay.py']
    hashes={name:sha(ROOT/name) for name in files};cases={};artifacts={'source-ledger.json':sha(ledgerpath)}
    for name,factor in (('baseline',1),('time-half',2)):
        result=simulate(body,selected,dt,factor);result['source_sha256']=hashes;result['geometry_sha256']=sha(gp)
        path=a.output/(name+'.json');path.write_text(json.dumps(result,separators=(',',':'),allow_nan=False)+'\n')
        artifacts[path.name]=sha(path);cases[name]={'checks':result['checks'],'metrics':result['metrics']}
        print(name,json.dumps(cases[name]),flush=True)
    x=cases['baseline']['metrics']['peak_von_mises_Pa'];y=cases['time-half']['metrics']['peak_von_mises_Pa']
    difference=abs(y/x-1)*100
    study={'schema':'tensorfem.recorded-hull-replay-study/1','physical_accuracy_qualified':False,
           'source_artifact_sha256':sha(a.source),'source_project':'TensorLBM + TensorDEM',
           'source_commit':'04b247489b87c7f38a4cd95ae64d79903556cebf','cases':cases,'artifacts_sha256':artifacts,
           'time_half_peak_stress_change_percent':difference,'time_sensitivity_pass_3percent':difference<3,
           'scope':'real recorded resultant history, one-way linear shell replay; no actual SUBOFF wet collision prediction'}
    (a.output/'study.json').write_text(json.dumps(study,indent=2)+'\n');print(json.dumps(study,indent=2))
if __name__=='__main__':main()
