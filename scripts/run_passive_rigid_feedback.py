#!/usr/bin/env python
"""Generate or exactly replay rejected real fluid steps, without force clipping."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from tensorfem.passive_rigid_feedback import PassiveRigidFeedback, EnergyBudgetExceeded, macroscopic_energy
from tensorfem.fluid_rigid_feedback import RigidFluidFeedback
from run_live_rigid_feedback import sources
ROOT=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(dt):
    s=PassiveRigidFeedback(thickness_m=.2,center_m=(.162,.161),dt_s=dt,tau=.5+3*.01*dt/.01**2)
    initial=s.snapshot();resumed=None
    for i in range(round(.3/dt)):
        before=s.snapshot()
        if i==8:resumed=PassiveRigidFeedback.restore(copy.deepcopy(before))
        try:
            actual=s.step()
        except EnergyBudgetExceeded as exc:
            with_rejection=False
            try:resumed.step()
            except EnergyBudgetExceeded as other:with_rejection=other.report==exc.report
            candidate=RigidFluidFeedback.restore(copy.deepcopy(before['solver']));candidate.step()
            return {'schema':'tensorfem.energy-budget-case/1','initial':initial,'accepted':s.snapshot(),
                    'rejected_candidate':candidate.snapshot(),'rejection':exc.report,
                    'rollback_exact':s.snapshot()==before,'restart_exact':resumed.snapshot()==s.snapshot() and with_rejection,
                    'independently_replayed_energy':macroscopic_energy(candidate),
                    'physical_accuracy_qualified':False,'completed_requested_duration':False,'requested_duration_s':.3}
        else:
            if resumed is not None and actual!=resumed.step():raise ValueError('gate restart mismatch')
    raise ValueError('expected moving-mask energy failure did not occur')
def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',action='store_true');a=p.parse_args()
    out=ROOT/'docs/assets/passive-rigid-feedback';out.mkdir(parents=True,exist_ok=True)
    bound=sources();bound.update({f'TensorFEM/{name}':digest(ROOT/name) for name in ['src/tensorfem/passive_rigid_feedback.py','scripts/run_passive_rigid_feedback.py']})
    study={'schema':'tensorfem.energy-budget-study/1','source_sha256':bound,'cases':{},'physical_accuracy_qualified':False,
           'scope':'unforced local rigid translation; kinetic plus isothermal density free energy budget; integer mask volume error unresolved'}
    for name,dt in [('local',.001),('local-time-half',.0005)]:
        raw=run(dt);path=out/(name+'.json')
        if not raw['rollback_exact'] or not raw['restart_exact']:raise ValueError('transactional failure')
        if a.audit:
            if json.loads(path.read_text())!=raw:raise ValueError('raw exact replay mismatch')
        else:path.write_text(json.dumps(raw,separators=(',',':'),allow_nan=False)+'\n')
        study['cases'][name]={'artifact_sha256':digest(path),'rejection':raw['rejection'],'accepted_time_s':raw['accepted']['solver']['fluid']['solver']['time']*dt,'rollback_exact':True,'restart_exact':True}
    path=out/'study.json'
    if a.audit:
        if json.loads(path.read_text())!=study:raise ValueError('bound study replay mismatch')
    else:path.write_text(json.dumps(study,indent=2)+'\n')
    print(json.dumps(study,indent=2))
if __name__=='__main__':main()
