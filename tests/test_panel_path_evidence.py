from types import SimpleNamespace

import pytest
import torch

from tensorfem.finite_rotation_layered_shell4 import assemble_finite_rotation_layered_shell4
from tensorfem.layered_shell4_plasticity import LayeredShell4Model, LayeredShell4State
from tensorfem.panel_path_evidence import (
    evaluate_panel_path, plastic_dissipation_increment, shell_stored_energy,
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
