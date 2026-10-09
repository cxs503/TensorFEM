#!/usr/bin/env python
"""Audit actual calibrated-wet tool fluid forces with a location-preserving map.

The SUBOFF carrier is explicitly registered to the circle fixture. This does
not make that circle flow a SUBOFF pressure solution. Motion is a rigid probe.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from tensorfem.coupled_hull import LinearHullReceiver
from tensorfem.surface_coupling import PlanarEmbedding
from tensorfem.point_surface_coupling import PointSurfaceExchange
from tensorfem.recorded_hull_loads import embed_wrench
ROOT=Path(__file__).resolve().parents[1];D=torch.float64

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def run(source):
    raw=json.loads(source.read_text());state=raw['state']['coupled']
    n=state['config']['ice_nx']*state['config']['ice_ny']
    xy=torch.tensor(state['exchange']['marker_positions'][n:],dtype=D)
    force_xy=-torch.tensor(state['exchange']['marker_force'][n:],dtype=D)
    source_time=state['scalars']['exchange_sample_time']
    gp=ROOT/'docs/assets/suboff-ice-v2/appended-16.json';g=json.loads(gp.read_text())
    body=LinearHullReceiver(g);s=body.surface();mask=body.labels==1
    center=(s.positions[mask.repeat_interleave(4)]*s.areas[mask.repeat_interleave(4),None]).sum(0)/s.areas[mask.repeat_interleave(4)].sum()
    axes=torch.tensor([[1.,0.],[0.,0.],[0.,1.]],dtype=D)
    origin=center-axes@xy.mean(0)
    embedding=PlanarEmbedding(origin,axes)
    points=embedding.positions(xy);force=embedding.vectors(force_xy)
    carrier=PointSurfaceExchange(body.nodes,body.elements[mask],points,maximum_offset_m=.08)
    omega=torch.tensor([0.,.1,0.],dtype=D);translation=torch.tensor([.03,0.,.02],dtype=D)
    velocity=torch.cat((translation+torch.linalg.cross(omega.expand_as(body.nodes),body.nodes),omega.expand_as(body.nodes)),1)
    load=carrier.point_loads(force,time_s=source_time,source='TensorLBM')
    audit=carrier.audit(load,velocity,expected_time_s=source_time)
    expected=embed_wrench(state['history'][-1],'fluid',embedding)
    got=torch.tensor(audit['force_N']+audit['moment_N_m'],dtype=D)
    source_error=float(torch.linalg.vector_norm(got-expected))
    returned=embedding.project_load(carrier.velocities(velocity))
    source_v=torch.tensor(state['exchange']['solid_velocity'][n:],dtype=D)
    mismatch=float(torch.linalg.vector_norm(returned-source_v,dim=1).max())
    metrics={**audit,'source_wrench_error_N_Nm':source_error,'source_vs_probe_velocity_max_difference_m_s':mismatch}
    checks={'force':audit['force_error_N']<1e-9,'moment':audit['moment_error_N_m']<1e-9,
            'power':audit['power_error_W']<1e-9,'source_wrench':source_error<1e-9,
            'motion_roundtrip':bool(torch.allclose(embedding.vectors(returned),carrier.velocities(velocity),atol=1e-12,rtol=0))}
    files=['src/tensorfem/point_surface_coupling.py','src/tensorfem/surface_coupling.py','src/tensorfem/coupled_hull.py',
           'src/tensorfem/recorded_hull_loads.py','src/tensorfem/suboff_ice_collision.py','src/tensorfem/suboff_ice_simulation.py',
           'src/tensorfem/spherical_shell.py','src/tensorfem/plate.py','scripts/run_point_load_exchange.py']
    return {'schema':'tensorfem.point-load-exchange/1','physical_accuracy_qualified':False,
            'feedback_applied_to_fluid':False,'motion_probe_only':True,'pressure_reconstructed':False,
            'scope':'actual circle-tool fluid marker forces on registered equivalent sail carrier; no SUBOFF CFD or hull response solve',
            'source_artifact_sha256':sha(source),'source_file_name':source.name,
            'geometry_file_sha256':sha(gp),'source_sha256':{f:sha(ROOT/f) for f in files},
            'load_sample_time_s':source_time,'source_state_time_s':state['history'][-1]['time_s'],
            'force_hold_interval_end_s':source_time+state['config']['exchange_steps']*state['config']['fluid_dt_s'],
            'embedding_origin_m':origin.tolist(),'embedding_axes':axes.tolist(),
            'force_points_xy_m':xy.tolist(),'force_on_tool_xy_N':force_xy.tolist(),
            'force_points_xyz_m':points.tolist(),'force_on_tool_xyz_N':force.tolist(),
            'carrier_elements':body.elements[mask].tolist(),'carrier_nodes_m':body.nodes.tolist(),
            'carrier_shape_weights':carrier.weights.tolist(),'carrier_natural_coordinates':carrier.natural_coordinates.tolist(),
            'carrier_face_ids':carrier.face_ids.tolist(),'carrier_offsets_m':carrier.offsets.tolist(),
            'nodal_force_and_couple':load.generalized_force.tolist(),'probe_nodal_velocity':velocity.tolist(),
            'returned_planar_velocities_m_s':returned.tolist(),'source_tool_velocity_m_s':source_v.tolist(),
            'motion':carrier.motion_record(velocity,time_s=source_time),'metrics':metrics,'checks':checks}

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'docs/assets/point-load-exchange/case.json')
    p.add_argument('--audit',action='store_true');a=p.parse_args()
    record=run(a.source)
    if a.audit:
        if json.loads(a.output.read_text())!=record:raise ValueError('point map/raw/source exact replay differs')
        print('PASS source force extraction and exact point-map/motion replay')
    else:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'checks':record['checks'],'metrics':record['metrics']},indent=2))
    if not all(record['checks'].values()):raise SystemExit(1)
if __name__=='__main__':main()
