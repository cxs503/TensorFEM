#!/usr/bin/env python
"""Reconstruct load origins, impulses and final shell fields from saved evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from tensorfem.coupled_hull import LinearHullReceiver
from tensorfem.recorded_hull_loads import embed_wrench
from tensorfem.surface_coupling import PlanarEmbedding
from tensorfem.suboff_ice_collision import _stress
ROOT=Path(__file__).resolve().parents[1]
D=torch.float64

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(condition,message):
    if not condition:raise ValueError(message)
def close(actual,expected,message,atol=1e-9,rtol=1e-9):
    require(torch.allclose(torch.as_tensor(actual,dtype=D),torch.as_tensor(expected,dtype=D),atol=atol,rtol=rtol),message)

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--evidence',type=Path,default=ROOT/'docs/assets/recorded-hull-replay');a=p.parse_args()
    study=json.loads((a.evidence/'study.json').read_text())
    require(sha(a.source)==study['source_artifact_sha256'],'original source changed')
    for file,digest in study['artifacts_sha256'].items():require(sha(a.evidence/file)==digest,'artifact changed: '+file)
    ledger=json.loads((a.evidence/'source-ledger.json').read_text())
    original=json.loads(a.source.read_text())['state']['history']
    require(len(ledger['records'])==len(original),'source record count')
    for row,raw in zip(ledger['records'],original):require(all(raw[k]==v for k,v in row.items()),'source record differs')
    gpath=ROOT/'docs/assets/suboff-ice-v2/appended-16.json';geometry=json.loads(gpath.read_text());body=LinearHullReceiver(geometry)
    summaries=[]
    for name in ('baseline','time-half'):
        result=json.loads((a.evidence/(name+'.json')).read_text())
        for file,digest in result['source_sha256'].items():require(sha(ROOT/file)==digest,'runtime source changed: '+file)
        require(sha(gpath)==result['geometry_sha256'],'geometry changed')
        require(result['config']==body.snapshot()['config'],'receiver configuration')
        embedding=PlanarEmbedding(result['embedding_origin_m'],result['embedding_axes'])
        surface=body.surface();mask=body.labels.repeat_interleave(4)==1
        origin=(surface.positions[mask]*surface.areas[mask,None]).sum(0)/surface.areas[mask].sum()
        close(embedding.origin,origin,'attachment origin')
        impulse=torch.zeros(3,dtype=D);dt=ledger['source_dt_s']
        require(len(result['history'])==len(ledger['records']),'history length')
        for i,(row,h) in enumerate(zip(ledger['records'],result['history'])):
            contributions=[embed_wrench(row,k,embedding) for k in ('fluid','ice_contact','other')]
            w=sum(contributions);close(w,embed_wrench(row,'total',embedding),'source decomposition')
            for key,value in zip(('fluid_wrench_N_Nm','ice_wrench_N_Nm','other_wrench_N_Nm'),contributions):close(h[key],value,'contribution embedding')
            close(h['total_wrench_N_Nm'],w,'total embedding')
            impulse+=dt*w[:3];close(h['impulse_N_s'],impulse,'impulse integral')
            close(h['time_s'],(i+1)*dt,'integration clock',atol=1e-11)
            close(h['energy_residual_J'],h['kinetic_J']+h['strain_J']-h['work_J'],'history energy identity',atol=1e-12)
            close(h['momentum_residual_kg_m_s'],torch.linalg.vector_norm(torch.tensor(h['momentum_kg_m_s'],dtype=D)-impulse),'history momentum residual',atol=1e-12)
        q=torch.tensor(result['restart']['q'],dtype=D);v=torch.tensor(result['restart']['v'],dtype=D)
        stress,vm=_stress({'ids':body.ids,'stress_operator':body.stress_operator},q)
        close(result['final_stress_Pa'],stress,'final stress');close(result['final_von_mises_Pa'],vm,'final von Mises')
        last=result['history'][-1]
        close(last['kinetic_J'],.5*body.mass@(v*v),'final kinetic energy',atol=1e-12)
        close(last['strain_J'],.5*q@torch.mv(body.shell,q),'final strain energy',atol=1e-12)
        close(last['momentum_kg_m_s'],(body.mass.reshape(-1,6)[:,:3]*v.reshape(-1,6)[:,:3]).sum(0),'final momentum')
        close(result['restart']['work_J'],last['work_J'],'restart work')
        metrics=result['metrics']
        close(metrics['peak_von_mises_Pa'],max(h['maximum_von_mises_Pa'] for h in result['history']),'peak stress')
        require(all(result['checks'].values()),'interface check failed')
        require(result['physical_accuracy_qualified'] is False and result['motion_feedback'] is False,'scope badge')
        summaries.append({'case':name,'raw_field_audit':'passed','physical_accuracy_qualified':False})
    print(json.dumps({'schema':'tensorfem.recorded-hull-replay-audit/1','cases':summaries},indent=2))
if __name__=='__main__':main()
