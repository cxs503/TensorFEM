import copy
import json
import pytest
import torch
from tensorfem.passive_rigid_feedback import PassiveRigidFeedback, EnergyBudgetExceeded, macroscopic_energy

@pytest.mark.parametrize('dt',[.001,.0005])
def test_real_node_conversion_rejected_atomically_and_restart(dt):
    s=PassiveRigidFeedback(thickness_m=.2,center_m=(.162,.161),dt_s=dt,tau=.5+3*.01*dt/.01**2)
    restored=None
    for i in range(100):
        before=s.snapshot()
        if i==8:restored=PassiveRigidFeedback.restore(json.loads(json.dumps(before)))
        try:
            entry=s.step()
        except EnergyBudgetExceeded as exc:
            assert exc.report['density_free_energy_J']>100*s.initial_energy_J
            assert s.snapshot()==before
            assert restored.snapshot()==before
            with pytest.raises(EnergyBudgetExceeded):restored.step()
            assert restored.snapshot()==before
            assert torch.equal(s.sim.fluid.solver.f,restored.sim.fluid.solver.f)
            break
        else:
            assert entry['energy_budget']['accepted']
            if restored is not None:assert entry==restored.step()
    else:pytest.fail('expected actual node-conversion energy rejection')


def test_quiescent_fluid_body_remains_accepted():
    s=PassiveRigidFeedback(initial_velocity_m_s=(0.,0.))
    for _ in range(20):s.step()
    assert s.sim.velocity.tolist()==[0.,0.]
    assert macroscopic_energy(s.sim)['total_J']<1e-12


def test_failure_after_mutation_rolls_back(monkeypatch):
    s=PassiveRigidFeedback();before=s.snapshot();original=s.sim.step
    def fail():
        original()
        raise ValueError('forced downstream mapping failure')
    monkeypatch.setattr(s.sim,'step',fail)
    with pytest.raises(ValueError,match='downstream'):s.step()
    assert s.snapshot()==before


def test_restart_rejects_tampered_budget_and_nonfinite_history():
    s=PassiveRigidFeedback();s.step();raw=s.snapshot()
    for mutate in [lambda d:d.update(initial_energy_J=1.),lambda d:d['accepted'][0].update(total_J=float('nan')),lambda d:d['accepted'][0].update(accepted=False),lambda d:d['accepted'][0].update(candidate_time_s=.5)]:
        bad=copy.deepcopy(raw);mutate(bad)
        with pytest.raises(ValueError):PassiveRigidFeedback.restore(bad)


@pytest.mark.parametrize('config',[{'relative_tolerance':-1},{'absolute_tolerance_J':float('nan')},{'relative_tolerance':True},{'backend':'global'}])
def test_invalid_gate_config(config):
    with pytest.raises(ValueError):PassiveRigidFeedback(**config)


def test_restore_does_not_mutate_input_checkpoint():
    s=PassiveRigidFeedback();s.step();raw=s.snapshot();saved=copy.deepcopy(raw)
    r=PassiveRigidFeedback.restore(raw);r.step()
    assert raw==saved
