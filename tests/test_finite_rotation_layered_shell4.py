import math

import pytest
import torch

from tensorfem.corotational_shell import axis_angle
from tensorfem.finite_rotation_layered_shell4 import (
    assemble_finite_rotation_layered_shell4,
    finite_rotation_element_response,
    model_with_imperfection,
    solve_finite_rotation_arc_path,
    solve_finite_rotation_increment,
)
from tensorfem.layered_shell4_plasticity import (
    LayeredShell4Model,
    LayeredShell4State,
    assemble_layered_shell4,
    layered_shell4_local_frame,
    plane_stress_j2_update,
)

D = torch.float64


def model(nodes=None, elements=None, *, residual=None):
    if nodes is None:
        nodes = torch.tensor([[0., 0., 0.], [1., 0., 0.],
                              [1., 1., 0.], [0., 1., 0.]], dtype=D)
    if elements is None:
        elements = torch.tensor([[0, 1, 2, 3]])
    return LayeredShell4Model(nodes, elements, 200000., .3, .08, 250., 1400.,
                              layers=3, residual_stress=residual)


def test_large_rigid_rotation_is_objective_and_does_not_update_history():
    shell = model(); state = LayeredShell4State.virgin(shell)
    axis = torch.tensor([.3, -.5, .8], dtype=D)
    angle = 1.17; rotation = axis_angle(axis, angle)
    translation = torch.tensor([2.1, -1.4, .7], dtype=D)
    dofs = torch.zeros(shell.n_dofs, dtype=D)
    for node, X in enumerate(shell.nodes):
        dofs[6*node:6*node+3] = rotation @ X + translation - X
        dofs[6*node+3:6*node+6] = axis / torch.linalg.vector_norm(axis) * angle
    response = assemble_finite_rotation_layered_shell4(shell, dofs, state, tangent=False)
    assert float(torch.linalg.vector_norm(response.internal_force)) < 3e-7
    assert float(response.stress.abs().max()) < 3e-7
    assert max(float(p.alpha) for e in response.trial_state.points for q in e for p in q) == 0.


def test_small_motion_matches_layered_material_point_and_full_tangent_direction():
    shell = model(); state = LayeredShell4State.virgin(shell)
    q = torch.zeros(24, dtype=D)
    q[6] = q[12] = .0018  # plastic membrane extension
    finite = finite_rotation_element_response(shell, 0, q, state.points[0])
    small = assemble_layered_shell4(shell, q, state)
    assert torch.allclose(finite[2], small[2][0], rtol=2e-5, atol=3e-4)
    direction = torch.tensor([math.sin(i * .71) for i in range(24)], dtype=D)
    h = 2e-6
    plus = finite_rotation_element_response(shell, 0, q+h*direction,
                                             state.points[0], tangent=False)[0]
    minus = finite_rotation_element_response(shell, 0, q-h*direction,
                                              state.points[0], tangent=False)[0]
    numerical = (plus-minus)/(2*h)
    assert torch.allclose(finite[1] @ direction, numerical, rtol=3e-3, atol=.2)


def test_imperfection_is_stress_free_reference_and_residual_stress_is_active():
    shell = model()
    imperfection = torch.zeros_like(shell.nodes); imperfection[:, 2] = torch.tensor([0., .01, .01, 0.])
    imperfect = model_with_imperfection(shell, imperfection)
    zero = torch.zeros(imperfect.n_dofs, dtype=D)
    response = assemble_finite_rotation_layered_shell4(
        imperfect, zero, LayeredShell4State.virgin(imperfect), tangent=False
    )
    assert float(response.stress.abs().max()) < 1e-10
    residual = torch.zeros((1, 4, 3, 3), dtype=D)
    residual[..., 0] = torch.tensor([60., -120., 60.], dtype=D)
    stressed = model(nodes=imperfect.nodes, residual=residual)
    response = assemble_finite_rotation_layered_shell4(
        stressed, zero, LayeredShell4State.virgin(stressed), tangent=False
    )
    assert torch.allclose(response.stress[..., 0], residual[..., 0], atol=2e-7)


def test_shallow_warped_reference_is_stress_free_objective_and_deep_warp_rejected():
    nodes = torch.tensor([[0., 0., 0.], [1., 0., .012],
                          [1., 1., -.007], [0., 1., .004]], dtype=D)
    shell = model(nodes=nodes)
    state = LayeredShell4State.virgin(shell)
    zero = assemble_finite_rotation_layered_shell4(
        shell, torch.zeros(shell.n_dofs, dtype=D), state, tangent=False
    )
    assert float(zero.internal_force.abs().max()) < 1e-10
    assert float(zero.stress.abs().max()) < 1e-10

    axis = torch.tensor([.2, -.7, .4], dtype=D)
    angle = .83
    rotation = axis_angle(axis, angle)
    shift = torch.tensor([.6, -1.2, 2.], dtype=D)
    dofs = torch.zeros(shell.n_dofs, dtype=D)
    for node, X in enumerate(nodes):
        dofs[6*node:6*node+3] = rotation @ X + shift - X
        dofs[6*node+3:6*node+6] = axis / torch.linalg.vector_norm(axis) * angle
    moved = assemble_finite_rotation_layered_shell4(shell, dofs, state, tangent=False)
    assert float(torch.linalg.vector_norm(moved.internal_force)) < 4e-7
    assert float(moved.stress.abs().max()) < 4e-7

    folded = nodes.clone(); folded[:, 2] = torch.tensor([0., 2., -2., 2.], dtype=D)
    with pytest.raises(ValueError, match="warp ratio"):
        layered_shell4_local_frame(folded)


def test_force_only_material_update_skips_nested_numerical_tangent(monkeypatch):
    # The outer finite-rotation difference needs stresses, not the small-strain
    # material Jacobian.  Count actual return-map calls to lock in that speedup.
    import tensorfem.layered_shell4_plasticity as layered
    shell = model(); state = LayeredShell4State.virgin(shell).points[0][0][0]
    strain = torch.tensor([2e-4, -1e-4, 3e-5], dtype=D)
    original = layered.update_j2
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(layered, "update_j2", counted)
    fast = plane_stress_j2_update(strain, shell, state, tangent=False)
    fast_calls = calls
    calls = 0
    full = plane_stress_j2_update(strain, shell, state, tangent=True)
    full_calls = calls
    assert torch.equal(fast[0], full[0])
    assert fast[1] is None
    assert full_calls >= 7 * fast_calls


def test_plastic_loading_then_unloading_keeps_committed_material_history():
    shell = model(); virgin = LayeredShell4State.virgin(shell)
    loaded = torch.zeros(shell.n_dofs, dtype=D)
    loaded[6] = loaded[12] = .0025
    forward = assemble_finite_rotation_layered_shell4(shell, loaded, virgin, tangent=False)
    peak_alpha = max(float(p.alpha) for e in forward.trial_state.points for q in e for p in q)
    assert peak_alpha > 0.
    unloaded = assemble_finite_rotation_layered_shell4(
        shell, torch.zeros_like(loaded), forward.trial_state, tangent=False
    )
    final_alpha = max(float(p.alpha) for e in unloaded.trial_state.points for q in e for p in q)
    assert final_alpha == pytest.approx(peak_alpha, abs=1e-12)
    assert float(unloaded.stress.abs().max()) > 1.


def test_two_facet_global_balance_and_failed_step_rolls_back():
    nodes = torch.tensor([[0.,0.,0.], [1.,0.,0.], [2.,0.,0.],
                          [0.,1.,0.], [1.,1.,0.], [2.,1.,0.]], dtype=D)
    shell = model(nodes, torch.tensor([[0,1,4,3], [1,2,5,4]]))
    state = LayeredShell4State.virgin(shell)
    fixed = []
    for node, xyz in enumerate(nodes):
        fixed.extend((6*node+2, 6*node+3, 6*node+4, 6*node+5))
        if float(xyz[0]) == 0.:
            fixed.extend((6*node, 6*node+1))
    fixed = torch.tensor(sorted(set(fixed)))
    load = torch.zeros(shell.n_dofs, dtype=D); load[12] = load[30] = 5.
    result = solve_finite_rotation_increment(shell, load, 1., fixed, state,
                                              torch.zeros_like(load), tolerance=2e-6)
    free = torch.tensor([i for i in range(shell.n_dofs) if i not in set(fixed.tolist())])
    assert float(torch.linalg.vector_norm(result.reaction[free])) < 2e-5
    before = tuple(p.plastic_strain.clone() for e in state.points for q in e for p in q)
    with pytest.raises(RuntimeError):
        solve_finite_rotation_increment(shell, load, 20., fixed, state,
                                        result.displacement, max_iterations=0)
    after = tuple(p.plastic_strain for e in state.points for q in e for p in q)
    assert all(torch.equal(a, b) for a, b in zip(before, after))


def test_arc_correctors_commit_only_accepted_material_state_and_failure_is_closed():
    shell = model(); state = LayeredShell4State.virgin(shell)
    fixed = torch.tensor([0,1,2,3,4,5, 8,9,10,11, 14,15,16,17, 18,19,20,21,22,23])
    load = torch.zeros(shell.n_dofs, dtype=D); load[6] = load[12] = 2.
    result = solve_finite_rotation_arc_path(
        shell, load, fixed, steps=1, step_size=.01, maximum_step=.01,
        load_scale=.1, initial_state=state, tolerance=2e-7
    )
    assert result.converged and len(result.points) == 1
    point = result.points[0]
    response = assemble_finite_rotation_layered_shell4(
        shell, point.displacement, state, tangent=False
    )
    free = torch.tensor([6,7,12,13])
    assert torch.linalg.vector_norm(
        response.internal_force[free] - point.load_factor*load[free]
    ) < 2e-7
    assert point.state is result.committed_state

    before = tuple(p.plastic_strain.clone() for e in point.state.points for q in e for p in q)
    rejected = solve_finite_rotation_arc_path(
        shell, 1000*load, fixed, steps=1, step_size=.2, load_scale=.1,
        minimum_step=.15, max_iterations=0, initial_state=point.state,
        initial_displacement=point.displacement,
        initial_load_factor=point.load_factor,
    )
    after = tuple(p.plastic_strain for e in rejected.committed_state.points for q in e for p in q)
    assert not rejected.converged and all(torch.equal(a,b) for a,b in zip(before,after))
