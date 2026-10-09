#!/usr/bin/env python
"""Run actual local/global fluid-to-structural translation feedback and audit."""
import argparse,hashlib,json
from pathlib import Path
import torch
from tensorfem.fluid_rigid_feedback import RigidFluidFeedback
ROOT=Path(__file__).resolve().parents[1]
D=torch.float64

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def run(backend,dt):
    sim=RigidFluidFeedback(backend=backend,thickness_m=.2,center_m=(.162,.161),dt_s=dt,tau=.5+3*.01*dt/.01**2)
    steps=round(.3/dt);restarted=None
    for step in range(steps):
        if step==steps//2:restarted=RigidFluidFeedback.restore(json.loads(json.dumps(sim.snapshot())))
        actual=sim.step()
        if restarted is not None:
            if actual!=restarted.step() or not torch.equal(sim.fluid.solver.f,restarted.fluid.solver.f):raise ValueError('restart trajectory differs')
    h=sim.history
    metrics={'max_momentum_residual_Ns':max(max(abs(x) for x in r['total_momentum_residual_Ns']) for r in h),
             'max_body_energy_identity_error_J':max(abs(r['body_energy_identity_error_J']) for r in h),
             'max_staggered_body_work_defect_J':max(abs(r['staggered_body_work_defect_J']) for r in h),
             'max_force_mapping_error_N':max(r['carrier_mapping']['force_error_N'] for r in h),
             'max_moment_mapping_error_N_m':max(r['carrier_mapping']['moment_error_N_m'] for r in h),
             'max_power_mapping_error_W':max(r['carrier_mapping']['power_error_W'] for r in h),
             'max_motion_feedback_error_m_s':max(r['motion_feedback_error_m_s'] for r in h),
             'covered_events':sum(r['exchange']['covered'] for r in h),'exposed_events':sum(r['exchange']['exposed'] for r in h),
             'final_body_velocity_m_s':sim.velocity.tolist(),'final_body_position_m':sim.fluid.center_m,
             'total_body_impulse_Ns':(sim.mass*(sim.velocity-torch.tensor([0.,.03],dtype=D))).tolist(),
             'global_reservoir_impulse_Ns':sim.reservoir_impulse.tolist(),'total_kinetic_change_J':h[-1]['total_kinetic_change_J']}
    checks={'momentum':metrics['max_momentum_residual_Ns']<1e-9,'body_energy_identity':metrics['max_body_energy_identity_error_J']<1e-12,
            'force_mapping':metrics['max_force_mapping_error_N']<1e-9,'moment_mapping':metrics['max_moment_mapping_error_N_m']<1e-9,
            'power_mapping':metrics['max_power_mapping_error_W']<1e-9,'actual_motion_feedback':metrics['max_motion_feedback_error_m_s']<1e-12,
            'velocity_changes':sim.velocity.tolist()!=[0.,.03],'node_conversion_exercised':metrics['covered_events']+metrics['exposed_events']>0,
            'restart_bitwise':sim.snapshot()==restarted.snapshot()}
    if backend=='local':checks['no_global_reservoir']=sim.reservoir_impulse.tolist()==[0.,0.]
    return {'schema':'tensorfem.live-rigid-fluid-case/1','metrics':metrics,'checks':checks,'state':sim.snapshot(),
            'physical_accuracy_qualified':False,'energy_accuracy_qualified':False,'flexible_suboff_qualified':False,
            'scope':'actual fluid feedback to free rigid translations through a Q4 carrier; no ice, rotations, free surface, or flexible shell'}

def sources():
    import tensorlbm.local_moving_boundary_2d as local
    import tensorlbm.feedback_moving_disk as live
    lbm=Path(local.__file__).parents[2]
    paths={f'TensorFEM/{p}':ROOT/p for p in ['src/tensorfem/fluid_rigid_feedback.py','src/tensorfem/point_surface_coupling.py','src/tensorfem/surface_coupling.py','src/tensorfem/plate.py','scripts/run_live_rigid_feedback.py']}
    paths.update({f'TensorLBM/{p}':lbm/p for p in ['src/tensorlbm/local_moving_boundary_2d.py','src/tensorlbm/feedback_moving_disk.py','src/tensorlbm/moving_boundary_2d.py','src/tensorlbm/d2q9.py','src/tensorlbm/solver.py']})
    return {name:digest(path) for name,path in paths.items()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'docs/assets/live-rigid-feedback');p.add_argument('--audit',action='store_true');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);record={'schema':'tensorfem.live-rigid-feedback-study/1','source_sha256':sources(),
             'physical_accuracy_qualified':False,'cases':{},'artifacts_sha256':{},
             'sensitivity_type':'combined time-step/compressibility; dx and SI nu fixed but sound speed changes'}
    for name,backend,dt in [('local','local',.001),('local-time-half','local',.0005),('global-reference','global',.001)]:
        result=run(backend,dt);path=a.output/(name+'.json')
        if a.audit:
            if json.loads(path.read_text())!=result:raise ValueError('exact case replay differs: '+name)
        else:path.write_text(json.dumps(result,separators=(',',':'),allow_nan=False)+'\n')
        if not all(result['checks'].values()):raise ValueError('exchange check failed: '+name)
        record['cases'][name]={'metrics':result['metrics'],'checks':result['checks']};record['artifacts_sha256'][path.name]=digest(path)
        print(name,'exact replay PASS' if a.audit else 'executed',json.dumps(record['cases'][name]),flush=True)
    x=record['cases']['local']['metrics']['total_body_impulse_Ns'][1];y=record['cases']['local-time-half']['metrics']['total_body_impulse_Ns'][1]
    record['time_eos_half_impulse_change_percent']=abs(y/x-1)*100
    record['time_eos_sensitivity_pass_3percent']=record['time_eos_half_impulse_change_percent']<3
    study=a.output/'study.json'
    if a.audit:
        if json.loads(study.read_text())!=record:raise ValueError('source/summary/hash replay differs')
    else:study.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'time_eos_half_impulse_change_percent':record['time_eos_half_impulse_change_percent'],'physical_accuracy_qualified':False},indent=2))
if __name__=='__main__':main()
