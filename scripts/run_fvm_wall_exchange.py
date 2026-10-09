#!/usr/bin/env python
"""Actual wall pressure/viscous forces mapped to an offset Q4 carrier."""
import argparse,hashlib,json
from pathlib import Path
import torch
from tensorfem.fvm_wall_loads import read_wall_loads
from tensorfem.point_surface_coupling import PointSurfaceExchange
ROOT=Path(__file__).resolve().parents[1];D=torch.float64

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(path):
    raw=json.loads(path.read_text());wall=read_wall_loads(raw);points=wall.points_m
    lower=points.min(0).values;upper=points.max(0).values;z=float(.5*(lower[2]+upper[2]));pad=.1
    x0,y0=float(lower[0])-pad,float(lower[1])-pad;x1,y1=float(upper[0])+pad,float(upper[1])+pad
    nodes=torch.tensor([[x0,y0,z],[x1,y0,z],[x1,y1,z],[x0,y1,z]],dtype=D)
    carrier=PointSurfaceExchange(nodes,[[0,1,2,3]],points,maximum_offset_m=wall.config['span'])
    v=torch.tensor([.03,-.02,.01],dtype=D);omega=torch.tensor([.02,.01,-.03],dtype=D)
    nv=torch.zeros((4,6),dtype=D);nv[:,:3]=v+torch.linalg.cross(omega.expand_as(nodes),nodes);nv[:,3:]=omega
    audits={};loads={}
    for label,force in [('pressure',wall.pressure_force_N),('viscous',wall.viscous_force_N),('total',wall.total_force_N)]:
        clock=wall.viscous_time_s if label=='viscous' else wall.time_s
        load=carrier.point_loads(force,time_s=clock,source='TensorFVM');audits[label]=carrier.audit(load,nv,expected_time_s=clock);loads[label]=load.generalized_force.tolist()
    returned=carrier.velocities(nv);rigid=v+torch.linalg.cross(omega.expand_as(points),points)
    if any(max(a[k] for k in ['force_error_N','moment_error_N_m','power_error_W'])>1e-10 for a in audits.values()) or float((returned-rigid).abs().max())>1e-12:raise ValueError('actual wall carrier mapping failed')
    if not torch.allclose(torch.tensor(loads['total'],dtype=D),torch.tensor(loads['pressure'],dtype=D)+torch.tensor(loads['viscous'],dtype=D),rtol=1e-12,atol=1e-12):raise ValueError('component force ownership mismatch')
    return {'schema':'tensorfem.actual-fvm-wall-exchange/1','source_raw_sha256':digest(path),'source_time_s':wall.time_s,
            'source_viscous_time_s':wall.viscous_time_s,'force_points_m':points.tolist(),'carrier_nodes_m':nodes.tolist(),
            'pressure_force_N':wall.pressure_force_N.tolist(),'viscous_force_N':wall.viscous_force_N.tolist(),
            'total_force_N':wall.total_force_N.tolist(),'generalized_loads':loads,'audits':audits,
            'nodal_virtual_velocity':nv.tolist(),'returned_point_velocity_m_s':returned.tolist(),
            'rigid_velocity_error_m_s':float((returned-rigid).abs().max()),'physical_accuracy_qualified':False,
            'flexible_response_solved':False,'fluid_feedback':False,'scope':'actual one-way wall force transfer, explicit clocks; virtual motion probes only'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--fvm',type=Path,default=Path('/home/jsyc/tensor-suite-development/TensorFVM'));p.add_argument('--audit',action='store_true');a=p.parse_args()
    out=ROOT/'docs/assets/fvm-wall-exchange';out.mkdir(parents=True,exist_ok=True)
    sources={f'TensorFEM/{name}':digest(ROOT/name) for name in ['src/tensorfem/fvm_wall_loads.py','src/tensorfem/point_surface_coupling.py','src/tensorfem/surface_coupling.py','src/tensorfem/plate.py','scripts/run_fvm_wall_exchange.py']}
    sources.update({f'TensorFVM/{name}':digest(a.fvm/name) for name in ['src/tensorfvm/conservative3d.py','src/tensorfvm/solver3d.py','src/tensorfvm/runtime.py','src/tensorfvm/mesh3d.py','scripts/run_conservative_transport.py']})
    study={'schema':'tensorfem.actual-fvm-wall-study/1','source_sha256':sources,'cases':{},'physical_accuracy_qualified':False}
    for name in ['cpu','cuda']:
        path=a.fvm/'docs/conservative-transport'/name/'wall-facets.json'
        if not path.exists():
            if name=='cpu':raise ValueError('CPU actual wall evidence missing')
            study['cuda']='not executed; actual source absent';continue
        data=run(path);dest=out/(name+'.json')
        if a.audit:
            if json.loads(dest.read_text())!=data:raise ValueError('actual wall exchange replay mismatch')
        else:dest.write_text(json.dumps(data,separators=(',',':'),allow_nan=False)+'\n')
        study['cases'][name]={'artifact_sha256':digest(dest),'source_raw_sha256':digest(path),'audits':data['audits'],'rigid_velocity_error_m_s':data['rigid_velocity_error_m_s']}
    dest=out/'study.json'
    if a.audit:
        if json.loads(dest.read_text())!=study:raise ValueError('wall study sources/fields/hash mismatch')
    else:dest.write_text(json.dumps(study,indent=2)+'\n')
    print(json.dumps(study,indent=2))
if __name__=='__main__':main()
