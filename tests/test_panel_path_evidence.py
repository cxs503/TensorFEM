from types import SimpleNamespace

import pytest
import torch

from tensorfem.finite_rotation_layered_shell4 import assemble_finite_rotation_layered_shell4
from tensorfem.layered_shell4_plasticity import LayeredShell4Model, LayeredShell4State
from tensorfem.panel_path_evidence import (
    energy_balance_gate, energy_residual_metrics, evaluate_panel_path,
    plastic_dissipation_increment,
    shell_stored_energy,
    yielded_fraction,
)
from tensorfem.plasticity import J2State


D = torch.float64


def model(layers=1):
    nodes = torch.tensor([[0., 0., 0.], [1., 0., 0.],
                          [1., 1., 0.], [0., 1., 0.]], dtype=D)
    return LayeredShell4Model(nodes, torch.tensor([[0, 1, 2, 3]]),
                              200000., .3, .1, 250., 1000., layers=layers)


def state_with_alpha(shell, alpha):
    virgin = LayeredShell4State.virgin(shell)
    points = tuple(tuple(tuple(J2State(p.plastic_strain.clone(), p.alpha.new_tensor(alpha))
                                     for p in q) for q in e) for e in virgin.points)
    return LayeredShell4State(points)


def test_elastic_affine_path_closes_external_and_internal_energy():
    shell = model(); virgin = LayeredShell4State.virgin(shell)
    u = torch.zeros(shell.n_dofs, dtype=D)
    # Uniform membrane extension has a linear response and exact trapezoidal work.
    u[6] = u[12] = 2e-4
    response = assemble_finite_rotation_layered_shell4(shell, u, virgin, tangent=False)
    point = SimpleNamespace(step=1, load_factor=1., displacement=u,
                            state=response.trial_state)
    history = evaluate_panel_path(shell, response.internal_force, [point])
    evidence = history.points[0]
    assert evidence.yielded_fraction == 0.
    assert evidence.plastic_dissipation == 0.
    assert evidence.relative_energy_residual < 2e-9
    assert abs(evidence.external_work-evidence.recoverable_energy) < 1e-10


def test_imperfect_facet_stored_energy_gradient_is_assembled_internal_force():
    """The energy observer must use the same tilted local basis as Shell4."""
    shell = model()
    shell = LayeredShell4Model(
        shell.nodes + torch.tensor([[0., 0., 0.00], [0., 0., 0.02],
                                    [0., 0., 0.03], [0., 0., 0.01]], dtype=D),
        shell.elements, shell.young, shell.poisson, shell.thickness,
        shell.yield_stress, shell.hardening, layers=shell.layers,
    )
    virgin = LayeredShell4State.virgin(shell)
    u = torch.tensor([
        0., 0., 0., 0., 0., 0.,
        2e-4, -1e-4, 3e-4, 2e-4, -1e-4, 1e-4,
        3e-4, 2e-4, -2e-4, -1e-4, 2e-4, -2e-4,
        -1e-4, 1e-4, 2e-4, 1e-4, 1e-4, 2e-4,
    ], dtype=D)
    response = assemble_finite_rotation_layered_shell4(
        shell, u, virgin, tangent=False,
    )
    direction = torch.linspace(-1., 1., shell.n_dofs, dtype=D)
    direction /= torch.linalg.vector_norm(direction)
    h = 1e-7
    plus = shell_stored_energy(shell, u+h*direction, response.trial_state).recoverable
    minus = shell_stored_energy(shell, u-h*direction, response.trial_state).recoverable
    gradient_action = (plus-minus)/(2*h)
    force_action = float(torch.dot(response.internal_force, direction))
    assert gradient_action == pytest.approx(force_action, rel=2e-7, abs=2e-7)


def test_plastic_dissipation_is_hand_integrated_sigma_y_delta_alpha():
    shell = model(layers=2); before = state_with_alpha(shell, .01)
    after = state_with_alpha(shell, .013)
    # unit area x thickness, all integration points/layers carry same increment
    expected = shell.yield_stress*.003*shell.thickness
    assert plastic_dissipation_increment(shell, before, after) == pytest.approx(expected)
    assert yielded_fraction(shell, after, before) == pytest.approx(1.)
    with pytest.raises(ValueError, match="decreased"):
        plastic_dissipation_increment(shell, after, before)


def test_energy_observer_does_not_mutate_committed_history():
    shell = model(); state = state_with_alpha(shell, .02)
    before = tuple(p.alpha.clone() for e in state.points for q in e for p in q)
    shell_stored_energy(shell, torch.zeros(shell.n_dofs, dtype=D), state)
    after = tuple(p.alpha for e in state.points for q in e for p in q)
    assert all(torch.equal(a, b) for a, b in zip(before, after))


def test_peak_requires_a_material_load_drop_and_increasing_shortening():
    shell = model(); state = LayeredShell4State.virgin(shell)
    load = torch.zeros(shell.n_dofs, dtype=D); load[6] = 1.
    points = []
    for step, (factor, shortening) in enumerate(((1., .01), (2., .02), (1.8, .03)), 1):
        u = torch.zeros(shell.n_dofs, dtype=D); u[6] = shortening
        points.append(SimpleNamespace(step=step, load_factor=factor,
                                      displacement=u, state=state))
    history = evaluate_panel_path(shell, load, points, post_peak_drop=.05)
    assert history.peak_step == 2
    assert history.post_peak_confirmed
    assert history.failure_mode == "elastic_buckling"
    assert history.points[1].is_peak and history.points[2].is_post_peak


def test_no_false_postpeak_claim_from_noise_or_reversed_shortening():
    shell = model(); state = LayeredShell4State.virgin(shell)
    load = torch.zeros(shell.n_dofs, dtype=D); load[6] = 1.
    points = []
    for step, (factor, shortening) in enumerate(((1., .01), (2., .02), (1.99, .019)), 1):
        u = torch.zeros(shell.n_dofs, dtype=D); u[6] = shortening
        points.append(SimpleNamespace(step=step, load_factor=factor,
                                      displacement=u, state=state))
    assert not evaluate_panel_path(shell, load, points).post_peak_confirmed


def test_chunked_energy_continuation_matches_single_path_exactly():
    shell = model(); virgin = LayeredShell4State.virgin(shell)

    def extension(value):
        u = torch.zeros(shell.n_dofs, dtype=D)
        u[6] = u[12] = value
        state = assemble_finite_rotation_layered_shell4(
            shell, u, virgin, tangent=False,
        ).trial_state
        return u, state

    u0, s0 = extension(1e-4)
    values = (2e-4, 3e-4, 4e-4)
    raw = [extension(value) for value in values]
    points = [SimpleNamespace(step=i, load_factor=float(i), displacement=u,
                              state=state)
              for i, (u, state) in enumerate(raw, 1)]
    load = assemble_finite_rotation_layered_shell4(
        shell, points[-1].displacement, virgin, tangent=False,
    ).internal_force / points[-1].load_factor
    baseline_energy = shell_stored_energy(shell, u0, s0).recoverable

    whole = evaluate_panel_path(
        shell, load, points, initial_displacement=u0, initial_state=s0,
        reference_recoverable_energy=baseline_energy,
    )
    first = evaluate_panel_path(
        shell, load, points[:2], initial_displacement=u0, initial_state=s0,
        reference_recoverable_energy=baseline_energy,
    )
    tail = first.points[-1]
    second = evaluate_panel_path(
        shell, load, points[2:],
        initial_displacement=points[1].displacement,
        initial_state=points[1].state,
        initial_load_factor=points[1].load_factor,
        initial_external_work=tail.external_work,
        initial_plastic_dissipation=tail.plastic_dissipation,
        reference_recoverable_energy=baseline_energy,
    )

    expected = whole.points[-1]
    actual = second.points[-1]
    for name in ("external_work", "recoverable_energy", "plastic_dissipation",
                 "internal_energy", "energy_residual", "relative_energy_residual"):
        assert getattr(actual, name) == pytest.approx(getattr(expected, name), abs=1e-12)


def test_first_point_step_sequence_keeps_absolute_and_mixed_diagnostics():
    # Regression evidence from the 4x4 fresh runs: the absolute defect falls
    # with the arc increment, while division by near-zero first-point work
    # alone reverses that trend.  A dimensionful floor exposes both facts.
    samples = (
        (.10, .2271762794622421, .22990113501035836),
        (.02, .009915024923340192, .010439508372151352),
    )
    measures = [energy_residual_metrics(w, u, absolute_scale_floor=.25)
                for _, w, u in samples]
    assert measures[1][0] < measures[0][0]  # absolute defect converges
    assert measures[1][1] > measures[0][1]  # raw near-zero percentage is ill-conditioned
    assert measures[1][2] < measures[0][2]  # mixed dimensional scale restores the trend
    assert measures[1][0] == pytest.approx(5.244834488111593e-4)
    fine_gate = energy_balance_gate(
        samples[1][1], samples[1][2], characteristic_energy=25_000.,
    )
    assert fine_gate["criterion"] == "absolute"
    assert fine_gate["absolute_tolerance_j"] == pytest.approx(.025)
    assert fine_gate["absolute_residual_j"] == pytest.approx(.0005244834488111593)
    assert fine_gate["passed"] is True


def test_energy_gate_switches_to_relative_above_characteristic_scale():
    passed = energy_balance_gate(25_001., 25_002., characteristic_energy=25_000.,
                                 relative_tolerance=1e-4)
    failed = energy_balance_gate(25_001., 25_011., characteristic_energy=25_000.,
                                 relative_tolerance=1e-4)
    assert passed["criterion"] == failed["criterion"] == "relative"
    assert passed["passed"] is True
    assert failed["passed"] is False


def test_energy_scale_floor_is_explicit_and_does_not_hide_raw_relative_error():
    shell = model(); state = LayeredShell4State.virgin(shell)
    load = torch.zeros(shell.n_dofs, dtype=D); load[6] = 1.
    u = torch.zeros(shell.n_dofs, dtype=D); u[6] = .01
    point = SimpleNamespace(step=1, load_factor=1., displacement=u, state=state)
    raw = evaluate_panel_path(shell, load, [point]).points[0]
    mixed = evaluate_panel_path(shell, load, [point], energy_scale_floor=1.).points[0]
    assert mixed.relative_energy_residual == raw.relative_energy_residual
    assert mixed.mixed_energy_residual <= mixed.relative_energy_residual
    assert mixed.energy_scale == 1.
