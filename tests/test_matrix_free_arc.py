import torch

from tensorfem.arc_length import ArcLengthProblem, solve_arc_length
from tensorfem.matrix_free_arc import MatrixFreeResponse, solve_matrix_free_arc
from tensorfem.matrix_free_arc import solve_matrix_free_finite_rotation_shell4
from tensorfem.layered_shell4_plasticity import LayeredShell4Model, LayeredShell4State


D = torch.float64


def test_matrix_free_path_matches_dense_crisfield_on_smooth_limit_point():
    load = torch.ones(1, dtype=D)

    def force(u):
        return u - u**3

    dense = solve_arc_length(
        ArcLengthProblem(lambda u: (force(u), torch.diag(1 - 3*u**2)), load),
        torch.zeros(1, dtype=D), steps=12, step_size=.04, load_scale=.2,
        tolerance=1e-10, minimum_step=1e-7,
    )
    virgin = ("virgin",)
    matrix_free = solve_matrix_free_arc(
        lambda u, state: MatrixFreeResponse(force(u), ("trial", float(u[0]))),
        load, torch.zeros(1, dtype=D), virgin, steps=12, step_size=.04,
        load_scale=.2, tolerance=1e-9, minimum_step=1e-7,
        difference_step=1e-6, krylov_rtol=1e-10,
    )
    assert dense.converged and matrix_free.converged
    dense_path = torch.tensor([[p.load_factor, float(p.displacement[0])] for p in dense.points])
    free_path = torch.tensor([[p.load_factor, float(p.displacement[0])] for p in matrix_free.points])
    assert torch.allclose(free_path, dense_path, rtol=2e-5, atol=2e-7)
    assert matrix_free.committed_state is matrix_free.points[-1].state


def test_rejected_step_preserves_exact_committed_state_identity():
    virgin = object()
    trial_states = []

    def response(u, committed):
        assert committed is virgin
        trial = object()
        trial_states.append(trial)
        # A constant force has a singular zero tangent, so every predictor is
        # rejected before any trial history can be committed.
        return MatrixFreeResponse(torch.ones_like(u), trial)

    result = solve_matrix_free_arc(
        response, torch.ones(2, dtype=D), torch.zeros(2, dtype=D), virgin,
        steps=1, step_size=.1, load_scale=1., minimum_step=.02,
        krylov_maxiter=4,
    )
    assert not result.converged
    assert result.points == ()
    assert result.committed_state is virgin
    assert torch.equal(result.displacement, torch.zeros(2, dtype=D))
    assert result.load_factor == 0.
    assert all(value is not virgin for value in trial_states)


def test_linear_identity_uses_directions_not_dense_tangent_columns():
    n = 32
    state = object()
    result = solve_matrix_free_arc(
        lambda u, committed: MatrixFreeResponse(u, committed),
        torch.linspace(1., 2., n, dtype=D), torch.zeros(n, dtype=D), state,
        steps=1, step_size=.05, load_scale=.5, tolerance=1e-11,
        krylov_rtol=1e-12, krylov_maxiter=10,
    )
    assert result.converged
    # Identity needs one Arnoldi direction.  A centred dense numerical tangent
    # would require 2*n force assemblies before its linear solve.
    # GMRES performs a few true-residual checks in addition to Arnoldi.
    assert result.tangent_actions <= 4
    assert result.response_evaluations < 2*n


def test_payload_and_state_come_from_accepted_equilibrium_evaluation():
    state = ("committed", 0)

    def response(u, committed):
        value = float(u[0])
        return MatrixFreeResponse(2*u, ("accepted", value), {"u": value})

    result = solve_matrix_free_arc(
        response, torch.ones(1, dtype=D), torch.zeros(1, dtype=D), state,
        steps=1, step_size=.1, load_scale=.3, tolerance=1e-11,
        krylov_rtol=1e-12,
    )
    assert result.converged
    point = result.points[0]
    assert point.state == ("accepted", float(point.displacement[0]))
    assert point.payload == {"u": float(point.displacement[0])}


def test_dimensional_force_and_arc_constraint_have_separate_normalized_metrics():
    # lambda is a force because the reference resultant is one, matching the
    # panel convention.  The convergence diagnostic must not add N and m^2.
    state = object()
    stiffness = torch.diag(torch.tensor([2.e8, 3.e8], dtype=D))
    result = solve_matrix_free_arc(
        lambda u, committed: MatrixFreeResponse(stiffness @ u, committed),
        torch.tensor([-.5, -.5], dtype=D), torch.zeros(2, dtype=D), state,
        steps=1, step_size=2.e-4, load_scale=2.e-9,
        relative_equilibrium_tolerance=1e-6,
        normalized_constraint_tolerance=1e-8,
        tolerance=1e-12, krylov_rtol=1e-11,
    )
    assert result.converged
    point = result.points[0]
    assert point.load_factor > 4.9e4  # approximately a 50 kN first increment
    assert point.equilibrium_relative_norm <= 1e-6
    assert point.constraint_relative_error <= 1e-8
    assert point.residual_norm == max(point.equilibrium_relative_norm,
                                      point.constraint_relative_error)


def test_initial_equilibrium_gate_fails_before_tangent_or_gmres():
    state = object()
    tangent_calls = 0

    def tangent(u, committed):
        nonlocal tangent_calls
        tangent_calls += 1
        return torch.eye(2, dtype=D)

    result = solve_matrix_free_arc(
        lambda u, committed: MatrixFreeResponse(
            torch.tensor([139229., 0.], dtype=D), committed),
        torch.tensor([1., 0.], dtype=D), torch.zeros(2, dtype=D), state,
        steps=1, step_size=.1, load_scale=1., step_tangent=tangent,
        initial_equilibrium_relative_tolerance=1e-6,
    )
    assert not result.converged
    assert result.termination_reason == "initial_equilibrium_failed"
    assert result.initial_equilibrium_norm == 139229.
    assert result.initial_equilibrium_relative_norm == 1.
    assert result.response_evaluations == 1
    assert result.krylov_iterations == 0 and tangent_calls == 0


def test_nonzero_initial_load_factor_can_pass_initial_equilibrium_gate():
    state = object()
    result = solve_matrix_free_arc(
        lambda u, committed: MatrixFreeResponse(2*u, committed),
        torch.ones(1, dtype=D), torch.ones(1, dtype=D), state,
        steps=1, step_size=.01, load_scale=.5, initial_load_factor=2.,
        initial_equilibrium_relative_tolerance=1e-12,
        relative_equilibrium_tolerance=1e-8,
        normalized_constraint_tolerance=1e-8,
    )
    assert result.converged
    assert result.points[0].load_factor > 2.


def test_frozen_step_tangent_uses_hybrid_quasi_newton_without_jvps():
    state = object()
    tangent_calls = 0

    def tangent(u, committed):
        nonlocal tangent_calls
        tangent_calls += 1
        return torch.diag(torch.tensor([2., 3.], dtype=D))

    result = solve_matrix_free_arc(
        lambda u, committed: MatrixFreeResponse(
            torch.tensor([2., 3.], dtype=D) * u, committed),
        torch.ones(2, dtype=D), torch.zeros(2, dtype=D), state,
        steps=1, step_size=.02, load_scale=.5, step_tangent=tangent,
        relative_equilibrium_tolerance=1e-9,
        normalized_constraint_tolerance=1e-9,
    )
    assert result.converged
    assert tangent_calls == 1
    assert result.tangent_actions == 0
    assert result.termination_reason == "completed"


def test_shell_adapter_returns_full_dofs_and_equilibrium_without_dense_tangent():
    nodes = torch.tensor([[0., 0., 0.], [1., 0., 0.],
                          [1., 1., 0.], [0., 1., 0.]], dtype=D)
    shell = LayeredShell4Model(nodes, torch.tensor([[0, 1, 2, 3]]),
                               200000., .3, .08, 250., 1400., layers=3)
    virgin = LayeredShell4State.virgin(shell)
    fixed = torch.tensor([0,1,2,3,4,5, 8,9,10,11, 14,15,16,17,
                          18,19,20,21,22,23])
    load = torch.zeros(shell.n_dofs, dtype=D)
    load[6] = load[12] = 2.
    result = solve_matrix_free_finite_rotation_shell4(
        shell, load, fixed, steps=1, step_size=.01, maximum_step=.01,
        load_scale=.1, initial_state=virgin, tolerance=3e-7,
        krylov_rtol=2e-7, krylov_maxiter=30,
    )
    assert result.converged and result.displacement.shape == (shell.n_dofs,)
    point = result.points[0]
    assert point.displacement.shape == (shell.n_dofs,)
    assert point.stress.shape[:2] == (1, 4)
    assert point.state is result.committed_state
