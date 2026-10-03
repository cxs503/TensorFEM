import pytest
import torch

from tensorfem.layered_shell4_plasticity import (
    LayeredShell4Model, LayeredShell4State, assemble_layered_shell4,
    plane_stress_j2_update, solve_layered_shell4_increment,
    solve_layered_shell4_path,
)

D = torch.float64


def one_element(*, layers=5, residual=None, hardening=1200.):
    nodes = torch.tensor([[0.,0.,0.], [1.,0.,0.], [1.,1.,0.], [0.,1.,0.]], dtype=D)
    return LayeredShell4Model(nodes, torch.tensor([[0,1,2,3]]), 200000., .3,
                              .1, 250., hardening, layers, residual_stress=residual)


def test_plane_stress_material_point_matches_uniaxial_j2_oracle_and_tangent():
    model = one_element()
    state = LayeredShell4State.virgin(model).points[0][0][0]
    target = 310.
    alpha = (target-model.yield_stress)/model.hardening
    strain = torch.tensor([target/model.young+alpha,
                           -model.poisson*target/model.young-alpha/2, 0.], dtype=D)
    stress, tangent, trial = plane_stress_j2_update(strain, model, state)
    assert torch.allclose(stress, torch.tensor([target, 0., 0.], dtype=D), rtol=3e-9, atol=2e-7)
    assert float(trial.alpha) == pytest.approx(alpha, rel=2e-9)
    direction = torch.tensor([.7, -.2, .3], dtype=D)
    h = 2e-7
    plus = plane_stress_j2_update(strain+h*direction, model, state)[0]
    minus = plane_stress_j2_update(strain-h*direction, model, state)[0]
    numerical = (plus-minus)/(2*h)
    assert torch.allclose(tangent@direction, numerical, rtol=2e-4, atol=2e-2)
    assert torch.allclose(tangent, tangent.T, rtol=2e-5, atol=.1)


def test_pure_membrane_and_pure_bending_reach_independent_layers():
    model = one_element(layers=5)
    state = LayeredShell4State.virgin(model)
    membrane = torch.zeros(model.n_dofs, dtype=D)
    membrane[6] = membrane[12] = .002
    _, _, stress_m, trial_m = assemble_layered_shell4(model, membrane, state)
    assert torch.allclose(stress_m[0,0,:,0], stress_m[0,0,0,0].expand(5), atol=1e-8)
    assert all(float(point.alpha) > 0 for point in trial_m.points[0][0])

    bending = torch.zeros_like(membrane)
    # theta_x = ry = kappa*x gives constant kappa_x.
    bending[6+4] = bending[12+4] = .05
    _, _, stress_b, trial_b = assemble_layered_shell4(model, bending, state)
    sx = stress_b[0,0,:,0]
    assert torch.allclose(sx, -sx.flip(0), rtol=1e-9, atol=1e-8)
    assert abs(float(sx[2])) < 1e-10
    assert float(trial_b.points[0][0][0].alpha) > 0
    assert float(trial_b.points[0][0][2].alpha) == 0


def test_residual_stress_is_a_real_self_equilibrated_initial_material_field():
    residual = torch.zeros((1,4,3,3), dtype=D)
    residual[:,:,:,0] = torch.tensor([80., -160., 80.], dtype=D)
    model = one_element(layers=3, residual=residual)
    state = LayeredShell4State.virgin(model)
    internal, _, stress, _ = assemble_layered_shell4(model, torch.zeros(model.n_dofs, dtype=D), state)
    assert torch.allclose(stress[...,0], residual[...,0], rtol=2e-10, atol=2e-8)
    assert float(torch.linalg.vector_norm(internal)) < 2e-8


def two_element_model(layers=3):
    nodes = torch.tensor([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.],
                          [0.,1.,0.],[1.,1.,0.],[2.,1.,0.]], dtype=D)
    elements = torch.tensor([[0,1,4,3],[1,2,5,4]])
    return LayeredShell4Model(nodes, elements, 200000., .3, .1, 250., 1500., layers)


def membrane_constraints(model):
    fixed = []
    for node, xyz in enumerate(model.nodes):
        fixed.extend((6*node+2, 6*node+3, 6*node+4, 6*node+5))
        if float(xyz[0]) == 0.:
            fixed.extend((6*node, 6*node+1))
    return torch.tensor(sorted(set(fixed)))


def test_multi_element_loading_unloading_global_balance_and_history_commit():
    model = two_element_model()
    fixed = membrane_constraints(model)
    force = torch.zeros(model.n_dofs, dtype=D)
    force[6*2] = force[6*5] = 18.
    path = solve_layered_shell4_path(model, force, fixed, [.5, 1., .25, 0.], tolerance=2e-8)
    for step in path:
        free = torch.tensor([i for i in range(model.n_dofs) if i not in set(fixed.tolist())])
        assert float(torch.linalg.vector_norm(step.reaction[free])) < 2e-6
    assert float(path[1].alpha.max()) > 0
    assert torch.allclose(path[-1].alpha, path[1].alpha, atol=1e-11)
    assert float(path[-1].displacement[12]) > 0  # permanent extension after unloading


def test_rejected_increment_cannot_mutate_committed_history():
    model = two_element_model(); state = LayeredShell4State.virgin(model)
    before = tuple(p.plastic_strain.clone() for e in state.points for q in e for p in q)
    with pytest.raises(RuntimeError):
        solve_layered_shell4_increment(model, torch.ones(model.n_dofs, dtype=D), 1.,
                                       membrane_constraints(model), state,
                                       torch.zeros(model.n_dofs, dtype=D), max_iterations=0)
    after = tuple(p.plastic_strain for e in state.points for q in e for p in q)
    assert all(torch.equal(a,b) for a,b in zip(before,after))


def test_layer_count_convergence_for_elastic_bending_resultant():
    values = []
    for layers in (2,4,8,16):
        model = one_element(layers=layers)
        u = torch.zeros(model.n_dofs, dtype=D); u[10] = u[16] = .002
        internal = assemble_layered_shell4(model, u, LayeredShell4State.virgin(model))[0]
        values.append(float(internal[10]+internal[16]))
    exact = values[-1]/(1-1/(4*16**2))
    errors = [abs(v-exact) for v in values]
    assert errors[1] < errors[0] and errors[2] < errors[1]
    assert abs(values[-1]-exact)/abs(exact) < .001
