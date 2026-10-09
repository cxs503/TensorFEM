import json
import pytest
import torch
from tensorfem.fluid_rigid_feedback import RigidFluidFeedback

@pytest.mark.parametrize('backend',['local','global'])
def test_real_motion_feedback_and_complete_restart(backend):
    sim=RigidFluidFeedback(backend=backend,thickness_m=.2,center_m=(.162,.161))
    for _ in range(20):sim.step()
    clone=RigidFluidFeedback.restore(json.loads(json.dumps(sim.snapshot())))
    for _ in range(20):
        assert sim.step()==clone.step()
        assert torch.equal(sim.fluid.solver.f,clone.fluid.solver.f)
    assert sim.velocity.tolist()!=[0.,.03]
    assert max(abs(x) for x in sim.history[-1]['total_momentum_residual_Ns'])<1e-10
    assert sim.history[-1]['motion_feedback_error_m_s']<1e-12
    assert sim.history[-1]['body_energy_identity_error_J']<1e-12
    if backend=='local':assert sim.reservoir_impulse.tolist()==[0.,0.]

def test_bad_restart_and_stale_exchange_rejected():
    sim=RigidFluidFeedback(backend='local')
    sim.step();snapshot=sim.snapshot();snapshot['history']=[]
    with pytest.raises(ValueError):RigidFluidFeedback.restore(snapshot)
    with pytest.raises(ValueError):sim.fluid.advance(time_s=0.,center_m=sim.fluid.center_m,velocity_m_s=[0.,.03])
    with pytest.raises(ValueError):RigidFluidFeedback(backend='unknown')
