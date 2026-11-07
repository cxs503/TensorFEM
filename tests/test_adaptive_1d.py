"""Acceptance tests for the 1D h-adaptive continuation (analytic orders + cost baseline).

Anchors
-------
* **Exact energy error**: with an analytic ``u`` the discrete error of a P1 solution is a closed
  form per element, so the measured slope can be compared with theory instead of with itself.
* **Consistent load**: the analytic ``int f N_i dx`` agrees with an independent high-order
  quadrature on the smooth problem, and the assembled load balances the boundary flux.
* **Patch test**: a linear field is reproduced exactly by the P1 space, i.e. assembly, Dirichlet
  condensation and the Thomas solve agree with the interpolation.
* **Orders**: ``||e||_E = O(N^-1)`` for the smooth solution and ``O(N^-(alpha-1/2))`` for the
  singular one -- both against theory, not against a stored number.
* **Adaptivity**: the indicator-driven loop beats uniform refinement on the singular problem and
  is a *negative control* on the smooth problem (no fabricated gain).
* **Fail closed**: a marking that is not a partition, an inadmissible ``alpha`` and an exact
  error of zero are rejected instead of silently accepted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest

from tensorfem.adaptive_1d import (
    KAPPA,
    Mesh1D,
    PowerSolution,
    SinusoidalSolution,
    adaptive_study,
    cost_baseline,
    element_flux,
    exact_error_energy_sq,
    graded_study,
    optimal_grading_exponent,
    recover_nodal_flux,
    refine,
    solve,
    solve_level,
    study,
    uniform_mesh,
    uniform_study,
)


@dataclass(frozen=True)
class LinearSolution:
    """``u = x``: the P1 space contains it, so the patch test must be exact."""

    kappa: float = KAPPA
    name: str = "linear"

    def u(self, x):
        return np.asarray(x, dtype=np.float64)

    def flux(self, x):
        return np.full_like(np.asarray(x, dtype=np.float64), self.kappa)

    def source(self, x):
        return np.zeros_like(np.asarray(x, dtype=np.float64))

    def flux_singular_at(self):
        return None

    def boundaries(self):
        return 0.0, 1.0

    def derivative_energy_sq(self, a, b):
        return self.kappa * (b - a)

    def source_hat(self, a, b):
        return 0.0, 0.0


@pytest.fixture(scope="module")
def smooth_uniform():
    return uniform_study(SinusoidalSolution(), (8, 16, 32, 64, 128, 256))


@pytest.fixture(scope="module")
def singular_uniform():
    return uniform_study(PowerSolution(0.6), (8, 16, 32, 64, 128, 256, 512))


@pytest.fixture(scope="module")
def singular_adaptive():
    return adaptive_study(PowerSolution(0.6), intervals=8, rounds=8, fraction=0.5)


def test_analytic_derivative_energy_matches_independent_quadrature():
    """The closed-form ``int u'^2`` must agree with a Gauss rule the module does not use."""
    solution = SinusoidalSolution()
    mesh = uniform_mesh(4)
    points, weights = np.polynomial.legendre.leggauss(40)
    quadrature = 0.0
    closed = 0.0
    for a, b in zip(mesh.nodes[:-1], mesh.nodes[1:]):
        half = 0.5 * (b - a)
        for x, w in zip(points, weights):
            physical = 0.5 * (a + b) + half * x
            quadrature += w * half * solution.kappa * (solution.flux(physical) / solution.kappa) ** 2
        closed += solution.derivative_energy_sq(float(a), float(b))
    assert quadrature == pytest.approx(closed, rel=1e-12)


def test_source_hat_matches_quadrature_and_balances_the_boundary_flux():
    """``int f N_i dx`` (closed form) agrees with composite Gauss, and the load balances."""
    solution = SinusoidalSolution()
    points, weights = np.polynomial.legendre.leggauss(60)
    for a, b in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)):
        first, second = solution.source_hat(a, b)
        half = 0.5 * (b - a)
        reference = [0.0, 0.0]
        for x, w in zip(points, weights):
            physical = 0.5 * (a + b) + half * x
            n0 = (b - physical) / (b - a)
            n1 = (physical - a) / (b - a)
            f = float(solution.source(physical))
            reference[0] += w * half * f * n0
            reference[1] += w * half * f * n1
        assert first == pytest.approx(reference[0], abs=1e-13)
        assert second == pytest.approx(reference[1], abs=1e-13)
    # the hats are a partition of unity, so the total load must reproduce int f dx exactly
    mesh = uniform_mesh(16)
    total = 0.0
    for a, b in zip(mesh.nodes[:-1], mesh.nodes[1:]):
        first, second = solution.source_hat(float(a), float(b))
        total += first + second
    # int_0^1 kappa pi^2 sin(pi x) dx = 2 kappa pi
    assert total == pytest.approx(2.0 * solution.kappa * math.pi, rel=1e-13)


def test_patch_test_reproduces_a_linear_field_exactly():
    solution = LinearSolution()
    mesh = uniform_mesh(7)
    displacement = solve(mesh, solution)
    assert np.allclose(displacement, mesh.nodes, atol=1e-13)
    assert np.allclose(element_flux(mesh, displacement, solution.kappa), solution.kappa, atol=1e-12)
    assert exact_error_energy_sq(mesh, displacement, solution) == pytest.approx(0.0, abs=1e-24)
    with pytest.raises(ValueError):
        solve_level(mesh, solution)  # effectivity of a zero error is undefined, not zero


def test_uniform_rate_of_the_smooth_solution_matches_theory(smooth_uniform):
    rate = smooth_uniform["rate"]
    assert rate == pytest.approx(-1.0, abs=0.05), rate
    errors = [record["true_energy_norm"] for record in smooth_uniform["records"]]
    assert all(later < earlier for earlier, later in zip(errors, errors[1:]))


def test_uniform_rate_of_the_singular_solution_matches_theory(singular_uniform):
    """Uniform P1 on ``u = x^0.6`` may only reach ``N^-(alpha-1/2) = N^-0.1``."""
    rate = singular_uniform["rate"]
    assert rate == pytest.approx(-0.1, abs=0.02), rate
    assert rate > -0.2  # the ladder is nearly stagnant by theory, not by accident


def test_adaptive_rate_exceeds_the_uniform_rate_by_a_factorial_margin(singular_adaptive, singular_uniform):
    adaptive_rate = singular_adaptive["rate"]
    uniform_rate = singular_uniform["rate"]
    assert adaptive_rate <= -0.4, adaptive_rate
    assert abs(adaptive_rate) >= 3.0 * abs(uniform_rate)


def test_adaptive_rounds_reach_a_lower_error_at_matched_unknowns(singular_adaptive, singular_uniform):
    rows = [row for row in (cost_baseline(singular_adaptive, singular_uniform)["matched_dof_table"])]
    compared = [row for row in rows if row.get("uniform_dofs") is not None]
    assert compared, "the uniform ladder must contain a mesh with at least as many unknowns"
    final = compared[-1]
    assert final["uniform_error"] > final["adaptive_error"]
    assert final["uniform_over_adaptive_dof_ratio"] >= 1.0


def test_effectivity_is_asymptotically_exact_for_the_smooth_problem(smooth_uniform):
    effectivity = [record["effectivity"] for record in smooth_uniform["records"]]
    assert effectivity[0] == pytest.approx(1.0, abs=0.02)
    assert effectivity[-1] == pytest.approx(1.0, abs=0.005)


def test_singular_effectivity_is_a_stable_under_estimate(singular_uniform):
    """Reporting the bias is the requirement: the estimator must not be silently rescaled."""
    effectivity = [record["effectivity"] for record in singular_uniform["records"]]
    assert all(0.05 < value < 1.0 for value in effectivity)
    spread = max(effectivity) - min(effectivity)
    assert spread < 0.1 * max(effectivity), effectivity


def test_adaptive_is_a_negative_control_on_the_smooth_problem(smooth_uniform):
    """On a smooth solution, marking cannot buy accuracy: uniform must win at matched unknowns."""
    adaptive = adaptive_study(SinusoidalSolution(), intervals=8, rounds=5, fraction=0.5)
    cost = cost_baseline(adaptive, smooth_uniform)
    compared = [row for row in cost["matched_dof_table"] if row.get("uniform_dofs") is not None]
    assert compared
    ratio = compared[-1]["uniform_over_adaptive_error_ratio"]
    assert ratio < 1.0, ratio  # the uniform error is at least as small: adaptivity gained nothing
    assert abs(adaptive["rate"]) < abs(smooth_uniform["rate"])


def test_graded_reference_ladder_is_recorded_with_its_conditioning_limit():
    solution = PowerSolution(0.6)
    graded = graded_study(solution, (8, 16, 32))
    assert optimal_grading_exponent(solution) == pytest.approx(10.0)
    assert graded["rate"] < 0.0
    # at equal element count the analytically graded mesh is already the better mesh
    uniform = uniform_study(solution, (8, 16))
    assert graded["levels"][0].true_norm < uniform["records"][0]["true_energy_norm"]


def test_cost_baseline_records_finite_measured_costs():
    solution = PowerSolution(0.6)
    uniform = uniform_study(solution, (8, 16, 32))
    adaptive = adaptive_study(solution, intervals=8, rounds=3, fraction=0.5)
    cost = cost_baseline(adaptive, uniform)
    assert cost["adaptive_dofs"] > 0
    assert cost["adaptive_seconds"] >= 0.0
    assert cost["adaptive_cumulative_dofs"] > cost["adaptive_dofs"]
    assert cost["time_per_dof_adaptive"] >= 0.0
    assert cost["uniform_dofs_at_equal_error"] is None or (
        cost["uniform_dofs_at_equal_error"] > 0.0
    )


def test_fail_closed_on_bad_input():
    mesh = uniform_mesh(4)
    with pytest.raises(ValueError):
        refine(mesh, np.zeros(mesh.n_elements, dtype=bool))
    with pytest.raises(ValueError):
        refine(mesh, np.ones(mesh.n_elements + 1, dtype=bool))
    with pytest.raises(ValueError):
        PowerSolution(alpha=1.5)
    with pytest.raises(ValueError):
        Mesh1D(np.array([0.0, 0.5, 0.5, 1.0]))
    with pytest.raises(ValueError):
        recover_nodal_flux(mesh, np.array([np.nan, 1.0, 1.0, 1.0]))
    with pytest.raises(ValueError):
        adaptive_study(SinusoidalSolution(), intervals=4, rounds=1, fraction=1.5)


def test_study_payload_carries_both_problems_and_their_theory_rates():
    payload = study(
        smooth_intervals=(8, 16, 32, 64),
        singular_intervals=(8, 16, 32, 64),
        rounds=3,
    )
    assert payload["schema"] == "tensorfem.adaptive_1d.study/v1"
    for name in ("sinusoidal", "power"):
        block = payload["problems"][name]
        assert block["uniform"]["theory_rate"] < 0.0
        # no closed-form rate is claimed for the adaptive ladder on purpose
        assert block["adaptive"]["theory_rate"] is None
        # the short ladder used here is still pre-asymptotic; the artifact study measures < 0.02
        assert block["criteria"]["uniform_rate_gap_to_theory"] < 0.1
        assert block["criteria"]["final_adaptive_dofs"] > 0
        assert len(block["adaptive"]["markings"]) == 3