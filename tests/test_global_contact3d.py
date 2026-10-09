import math

import pytest
import torch

from tensorfem.global_contact3d import (
    GlobalSurfaceContactModel,
    NodeTrianglePair,
    assemble_global_surface_contact,
    initial_global_surface_state,
    solve_global_surface_contact_path,
)


D = torch.float64


def model(*, kn=1.0e4, kt=1.0e3, mu=0.2, split=True):
    masters = [[-2., -2., 0.], [3., -2., 0.], [3., 3., 0.], [-2., 3., 0.]]
    x = torch.tensor(masters + [[.2, .4, .05]], dtype=D)
    ndof = 3 * len(x)
    k = torch.zeros((ndof, ndof), dtype=D)
    k[-3:, -3:] = 1000. * torch.eye(3, dtype=D)
    faces = ((0, 1, 2), (0, 2, 3)) if split else ((0, 1, 3),)
    return GlobalSurfaceContactModel(
        x, k, (NodeTrianglePair(4, faces),), tuple(range(12)), kn, kt, mu)


def load(m, fx=0., fy=0., fz=-100.):
    force = torch.zeros(3 * len(m.reference_nodes), dtype=D)
    force[-3:] = torch.tensor([fx, fy, fz], dtype=D)
    return force


def test_contact_enters_fe_residual_and_shape_reactions_balance_force_and_moment():
    m = model(mu=0.)
    step = solve_global_surface_contact_path(m, [load(m)], tolerance=1e-11)[0]
    assembly = assemble_global_surface_contact(
        m, step.displacement, step.external, initial_global_surface_state(m))
    assert torch.linalg.vector_norm(assembly.residual[-3:]) < 1e-9
    assert torch.linalg.vector_norm(assembly.force_imbalance) < 1e-12
    assert torch.linalg.vector_norm(assembly.moment_imbalance) < 1e-12
    assert float(assembly.updates[0].normal_force) == pytest.approx(45.45454545)
    master = assembly.contact_force.reshape(-1, 3)[:4]
    assert torch.count_nonzero(torch.linalg.vector_norm(master, dim=1) > 0) == 3


def test_large_slide_researches_current_triangles_and_commits_friction_history():
    m = model(mu=.2)
    loads = [load(m, fx=value) for value in (0., 300., 800., 1400.)]
    steps = solve_global_surface_contact_path(m, loads, tolerance=1e-10)
    assert [s.state.histories[0].face for s in steps] == [1, 0, 0, 0]
    assert not steps[-1].state.histories[0].sticking
    assert steps[-1].state.histories[0].dissipated_energy > 0
    assert max(s.residual_norm for s in steps) < 2e-8


def test_master_triangulation_and_load_step_refinement_converge():
    coarse, split = model(mu=0., split=False), model(mu=0., split=True)
    coarse_step = solve_global_surface_contact_path(coarse, [load(coarse)])[-1]
    loads = [load(split, fz=-100. * i / 8) for i in range(1, 9)]
    refined_step = solve_global_surface_contact_path(split, loads)[-1]
    assert torch.allclose(coarse_step.displacement[-3:], refined_step.displacement[-3:],
                          atol=2e-10)

    direct_model, refined_model = model(), model()
    direct = solve_global_surface_contact_path(
        direct_model, [load(direct_model, fx=1400.)])[-1]
    refined_loads = [
        load(refined_model, fx=1400. * i / 10, fz=-100. * i / 10)
        for i in range(1, 11)
    ]
    refined = solve_global_surface_contact_path(refined_model, refined_loads)[-1]
    assert torch.allclose(direct.displacement[-3:], refined.displacement[-3:], atol=2e-10)


def test_penalty_reaction_converges_monotonically_to_rigid_limit():
    errors = []
    for penalty in (5e3, 2e4, 5e4):
        m = model(kn=penalty, mu=0.)
        step = solve_global_surface_contact_path(m, [load(m)])[-1]
        reaction = step.state.histories[0].normal_multiplier
        errors.append(abs(float(reaction) / 50. - 1.))
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < .03


def test_frictional_penetration_couple_decreases_with_penalty():
    moments = []
    for penalty in (5e3, 2e4, 5e4):
        m = model(kn=penalty)
        step = solve_global_surface_contact_path(m, [load(m, fx=1400.)])[-1]
        assembly = assemble_global_surface_contact(
            m, step.displacement, step.external, initial_global_surface_state(m),
            tangent=False)
        moments.append(float(torch.linalg.vector_norm(assembly.moment_imbalance)))
    assert moments[0] > moments[1] > moments[2]
    assert moments[-1] < .03


def test_failed_increment_preserves_committed_displacement_and_history():
    m = model()
    state = initial_global_surface_state(m)
    before = state.histories[0]
    with pytest.raises(RuntimeError, match="committed state was not changed"):
        solve_global_surface_contact_path(
            m, [load(m, fx=1400.)], initial_state=state,
            max_iterations=1, tolerance=1e-14)
    after = state.histories[0]
    assert torch.equal(state.displacement, torch.zeros_like(state.displacement))
    assert after.face == before.face
    assert torch.equal(after.elastic_slip, before.elastic_slip)
    assert after.dissipated_energy == before.dissipated_energy


def test_common_rigid_translation_creates_no_spurious_friction():
    m = model()
    u = torch.zeros(3 * len(m.reference_nodes), dtype=D).reshape(-1, 3)
    u[:] = torch.tensor([.7, -.4, .3], dtype=D)
    u[-1, 2] -= .355
    assembly = assemble_global_surface_contact(
        m, u.flatten(), torch.zeros_like(u).flatten(),
        initial_global_surface_state(m), tangent=False)
    update = assembly.updates[0]
    assert update.normal_force > 0
    assert torch.linalg.vector_norm(update.tangential_force) < 1e-12
    assert abs(float(update.dissipation_increment)) < 1e-12


def test_contact_response_is_covariant_under_finite_rigid_rotation():
    base = model(mu=0.)
    u = torch.zeros(15, dtype=D)
    u[-1] = -.055
    original = assemble_global_surface_contact(
        base, u, torch.zeros_like(u), initial_global_surface_state(base), tangent=False)
    angle = .71
    rotation = torch.tensor([
        [math.cos(angle), -math.sin(angle), 0.],
        [math.sin(angle), math.cos(angle), 0.],
        [0., 0., 1.],
    ], dtype=D)
    xr = base.reference_nodes @ rotation.T
    transform = torch.kron(torch.eye(5, dtype=D), rotation.contiguous())
    kr = transform @ base.stiffness @ transform.T
    rotated = GlobalSurfaceContactModel(
        xr, kr, base.pairs, base.fixed_dofs,
        base.normal_penalty, base.tangential_penalty, base.friction)
    ur = (u.reshape(-1, 3) @ rotation.T).flatten()
    response = assemble_global_surface_contact(
        rotated, ur, torch.zeros_like(ur), initial_global_surface_state(rotated), tangent=False)
    expected = original.contact_force.reshape(-1, 3) @ rotation.T
    assert torch.allclose(response.contact_force.reshape(-1, 3), expected, atol=2e-12)


def test_algorithmic_tangent_matches_centered_difference_in_sliding_branch():
    m = model(mu=.2)
    previous = solve_global_surface_contact_path(m, [load(m, fx=300.)])[-1]
    current = solve_global_surface_contact_path(
        m, [load(m, fx=700.)], initial_displacement=previous.displacement,
        initial_state=previous.state)[-1]
    u, state, force = current.displacement, previous.state, load(m, fx=900.)
    assembly = assemble_global_surface_contact(m, u, force, state)
    direction = torch.tensor([.31, -.17, -.27], dtype=D)
    full = torch.zeros_like(u)
    full[-3:] = direction
    h = 1e-6
    plus = assemble_global_surface_contact(
        m, u + h * full, force, state, tangent=False).residual
    minus = assemble_global_surface_contact(
        m, u - h * full, force, state, tangent=False).residual
    difference = (plus - minus) / (2 * h)
    assert torch.allclose(assembly.tangent @ full, difference, rtol=3e-6, atol=3e-6)
