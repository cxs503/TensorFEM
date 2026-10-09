import pytest
import torch

from tensorfem.global_contact2d import (
    GlobalContactModel, NodePolylinePair, assemble_global_contact,
    initial_global_contact_state, solve_global_contact_path,
)


D = torch.float64


def model(*, kn=1.0e4, mu=.2, split=True):
    master = [[0., 0.], [.5, 0.], [1.5, 0.]] if split else [[0., 0.], [1.5, 0.]]
    x = torch.tensor(master + [[.2, .05]], dtype=D)
    ndof = 2 * len(x); k = torch.zeros((ndof, ndof), dtype=D)
    slave = len(x) - 1; k[2 * slave, 2 * slave] = 1000.; k[2 * slave + 1, 2 * slave + 1] = 1000.
    return GlobalContactModel(x, k, (NodePolylinePair(slave, tuple(range(slave))),),
                              tuple(range(2 * slave)), kn, 1000., mu)


def load(m, fx, fy=-100.):
    f = torch.zeros(len(m.reference_nodes) * 2, dtype=D)
    f[-2:] = torch.tensor([fx, fy], dtype=D)
    return f


def test_contact_is_in_global_residual_and_balances_slave_master_forces():
    m = model(); step = solve_global_contact_path(m, [load(m, 0.)], tolerance=1e-11)[0]
    a = assemble_global_contact(m, step.displacement, step.external,
                                initial_global_contact_state(m))
    assert torch.linalg.vector_norm(a.residual[-2:]) < 1e-10
    assert torch.linalg.vector_norm(a.force_imbalance) < 1e-12
    assert float(a.updates[0].normal_traction) == pytest.approx(45.45454545)
    # Master reactions plus the free-node spring and applied force close globally.
    assert torch.linalg.vector_norm(a.residual.reshape(-1, 2).sum(0) -
                                    (m.stiffness @ step.displacement-step.external).reshape(-1, 2).sum(0)) < 1e-12


def test_large_slide_researches_segment_and_commits_coulomb_history():
    m = model(); loads = [load(m, f) for f in (0., 200., 500., 900., 1200.)]
    steps = solve_global_contact_path(m, loads, tolerance=1e-11)
    assert [s.state.histories[0].segment for s in steps] == [0, 0, 1, 1, 1]
    assert not steps[-1].state.histories[0].sticking
    assert steps[-1].state.histories[0].dissipated_energy > 0
    assert abs(steps[-1].displacement[-2] - 1.1909090909) < 1e-9
    assert max(s.residual_norm for s in steps) < 1e-9


def test_master_segmentation_and_load_step_refinement_converge():
    coarse, fine = model(split=False), model(split=True)
    uc = solve_global_contact_path(coarse, [load(coarse, 1200.)], tolerance=1e-11)[-1]
    refined_loads = [load(fine, 1200. * i / 12) for i in range(1, 13)]
    uf = solve_global_contact_path(fine, refined_loads, tolerance=1e-11)[-1]
    assert torch.allclose(uc.displacement[-2:], uf.displacement[-2:], atol=2e-10)
    assert abs(float(uc.state.histories[0].dissipated_energy -
                     uf.state.histories[0].dissipated_energy)) < 2e-10


def test_penalty_reaction_converges_to_rigid_contact_limit():
    errors = []
    for kn in (5e3, 2e4, 5e4):
        m = model(kn=kn, mu=0.); s = solve_global_contact_path(m, [load(m, 0.)])[-1]
        state = s.state.histories[0]
        errors.append(abs(float(state.normal_multiplier) / 50. - 1.))
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < .03


def test_failed_increment_does_not_mutate_committed_history():
    m = model(); state = initial_global_contact_state(m)
    before = state.histories[0]
    with pytest.raises(RuntimeError, match="committed state was not changed"):
        solve_global_contact_path(m, [load(m, 1200.)], initial_state=state,
                                  max_iterations=1, tolerance=1e-14)
    after = state.histories[0]
    assert after.segment == before.segment
    assert after.dissipated_energy == before.dissipated_energy
    assert after.elastic_slip == before.elastic_slip


def test_common_rigid_translation_does_not_create_friction():
    m = model(); u = torch.zeros(2 * len(m.reference_nodes), dtype=D)
    u[0::2] = .4; u[-1] = -.055
    a = assemble_global_contact(m, u, torch.zeros_like(u),
                                initial_global_contact_state(m), tangent=False)
    assert a.updates[0].normal_traction > 0
    assert abs(float(a.updates[0].tangential_traction)) < 1e-12
    assert abs(float(a.updates[0].dissipation_increment)) < 1e-12


def test_algorithmic_global_tangent_matches_centered_difference_in_sliding_branch():
    m = model(); previous = solve_global_contact_path(m, [load(m, 200.)])[-1]
    current = solve_global_contact_path(m, [load(m, 500.)],
                                        initial_displacement=previous.displacement,
                                        initial_state=previous.state)[-1]
    u, state = current.displacement, previous.state
    f = load(m, 700.); a = assemble_global_contact(m, u, f, state)
    direction = torch.tensor([.31, -.27], dtype=D); full = torch.zeros_like(u); full[-2:] = direction
    h = 1e-6
    plus = assemble_global_contact(m, u+h*full, f, state, tangent=False).residual
    minus = assemble_global_contact(m, u-h*full, f, state, tangent=False).residual
    fd = (plus-minus)/(2*h)
    assert torch.allclose(a.tangent @ full, fd, rtol=2e-6, atol=2e-6)
