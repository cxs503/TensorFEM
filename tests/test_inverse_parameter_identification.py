"""Verification of the inverse-identification infrastructure (``tensorfem.inverse``).

Each test binds one property to an *independent* number -- an analytic recovery, a
real :class:`~tensorfem.model.TrussModel` solve, an analytic Jacobian, or a
recomputed hash -- rather than to the optimiser's own self-reported loss.
"""

from __future__ import annotations

import json
import math

import pytest
import torch

from tensorfem.inverse import (
    SCHEMA_CONVERGENCE,
    SCHEMA_SUMMARY,
    InverseProblem,
    OptimizerOptions,
    Parameter,
    SeedControl,
    Tikhonov,
    fd_jacobian,
    run_inversion,
    sha256_of,
)
from tensorfem.model import TrussModel
from tensorfem.solvers import solve_linear_static

FLOAT = torch.float64

# ---------------------------------------------------------------------------
# case builders
# ---------------------------------------------------------------------------
DECAY_TRUTH = {"A": 60.0, "tau": 4.0, "Tinf": 20.0}
DECAY_TIMES = torch.linspace(0.0, 20.0, 41, dtype=FLOAT)

BAR_TRUTH = {"E": 210.0e9}
BAR_AREA = 1.0e-3
BAR_LENGTH = 2.0
BAR_NODES = 4
BAR_LOAD = 1.0e4


def _decay_problem(seed: int = 7, noise: float = 0.01, sigma: float = 0.8) -> InverseProblem:
    def forward(parameters):
        return parameters["Tinf"] + parameters["A"] * torch.exp(
            -DECAY_TIMES / parameters["tau"]
        )

    clean = forward(DECAY_TRUTH)
    generator = torch.Generator().manual_seed(seed)
    data = clean + noise * clean.abs() * torch.randn(
        clean.shape, dtype=FLOAT, generator=generator
    )
    return InverseProblem(
        forward=forward,
        data=data,
        parameters=[
            Parameter("A", 50.0, lower=1.0, upper=200.0),
            Parameter("tau", 2.0, lower=0.1, upper=50.0, log=True),
            Parameter("Tinf", 0.0, lower=-50.0, upper=100.0),
        ],
        sigma=sigma,
        name="exponential-decay",
    )


def _bar_setup():
    x = torch.linspace(0.0, BAR_LENGTH, BAR_NODES + 1, dtype=FLOAT)
    nodes = torch.stack([x, torch.zeros_like(x)], dim=1)
    elements = torch.tensor([[i, i + 1] for i in range(BAR_NODES)])
    forces = torch.zeros(2 * (BAR_NODES + 1), dtype=FLOAT)
    forces[2 * BAR_NODES] = BAR_LOAD
    # fix the left node and every transverse dof, leaving the axial chain free
    fixed = torch.tensor([0] + list(range(1, 2 * (BAR_NODES + 1), 2)))
    return nodes, elements, forces, fixed


def _bar_displacements(parameters) -> torch.Tensor:
    nodes, elements, forces, fixed = _bar_setup()
    area = parameters.get("A", BAR_AREA)
    model = TrussModel(
        nodes=nodes,
        elements=elements,
        young_modulus=torch.tensor([parameters["E"]], dtype=FLOAT),
        area=torch.tensor([area], dtype=FLOAT),
        forces=forces,
        fixed_dofs=fixed,
    )
    result = solve_linear_static(model)
    return result.displacement[torch.arange(1, BAR_NODES + 1) * 2]


def _bar_problem(*, unknown_area: bool = False, seed: int = 11) -> InverseProblem:
    clean = _bar_displacements(BAR_TRUTH)
    generator = torch.Generator().manual_seed(seed)
    data = clean + 0.005 * clean.abs() * torch.randn(
        clean.shape, dtype=FLOAT, generator=generator
    )
    parameters = [Parameter("E", 150.0e9, lower=1.0e9, upper=1.0e12, log=True)]
    if unknown_area:
        parameters.append(Parameter("A", 5.0e-4, lower=1.0e-6, upper=1.0e-1, log=True))
    return InverseProblem(
        forward=_bar_displacements,
        data=data,
        parameters=parameters,
        sigma=0.005 * float(clean[-1]),
        name="axial-bar",
    )


# ---------------------------------------------------------------------------
# 1. the numerical kernel itself
# ---------------------------------------------------------------------------
def test_fd_jacobian_matches_analytic_jacobian():
    def func(vector):
        return torch.stack(
            [
                vector[0] ** 2,
                torch.exp(vector[1]),
                vector[0] * vector[1],
                torch.sin(vector[2]),
            ]
        )

    def analytic(vector):
        return torch.tensor(
            [
                [2.0 * vector[0], 0.0, 0.0],
                [0.0, math.exp(float(vector[1])), 0.0],
                [vector[1], vector[0], 0.0],
                [0.0, 0.0, math.cos(float(vector[2]))],
            ],
            dtype=FLOAT,
        )

    point = torch.tensor([1.3, 0.4, 0.9], dtype=FLOAT)
    forward = fd_jacobian(func, point, rel_step=1e-6, scheme="forward")
    central = fd_jacobian(func, point, rel_step=1e-6, scheme="central")
    exact = analytic(point)
    forward_error = float(
        torch.max(torch.abs(forward - exact) / (torch.abs(exact) + 1e-12))
    )
    central_error = float(
        torch.max(torch.abs(central - exact) / (torch.abs(exact) + 1e-12))
    )
    # forward differences are first order in the step (truncation ~ rel_step), so
    # the reachable accuracy is ~1e-6; central differences are second order.
    assert forward_error < 5e-6, forward_error
    assert central_error < 1e-9, central_error


# ---------------------------------------------------------------------------
# 2. analytic recovery: exponential decay time constant
# ---------------------------------------------------------------------------
def test_decay_time_constant_recovered_within_three_percent():
    problem = _decay_problem()
    run = run_inversion(problem, OptimizerOptions(), SeedControl(seed=1, starts=3))

    assert run.status == "converged", (run.status, run.stop_reason)
    tau = run.best_parameters["tau"]
    error = abs(tau - DECAY_TRUTH["tau"]) / DECAY_TRUTH["tau"]
    assert error < 0.03, error
    assert abs(run.best_parameters["A"] - DECAY_TRUTH["A"]) / DECAY_TRUTH["A"] < 0.05
    assert abs(run.best_parameters["Tinf"] - DECAY_TRUTH["Tinf"]) < 1.0
    assert run.diagnostics["identifiable"] is True
    # within each start, an accepted step must never increase the loss
    for start in {record["start"] for record in run.iterations}:
        accepted = [
            record["loss"]
            for record in run.iterations
            if record["accepted"] and record["start"] == start
        ]
        assert accepted, start
        assert all(later < earlier for earlier, later in zip(accepted, accepted[1:])), (
            start,
            accepted,
        )


# ---------------------------------------------------------------------------
# 3. seed control
# ---------------------------------------------------------------------------
def test_same_seed_reproduces_log_hash_and_estimates():
    problem = _decay_problem()
    first = run_inversion(problem, OptimizerOptions(), SeedControl(seed=3, starts=3))
    second = run_inversion(problem, OptimizerOptions(), SeedControl(seed=3, starts=3))
    assert first.log_sha256() == second.log_sha256()
    assert first.best_parameters == second.best_parameters


def test_three_seeds_all_recover_tau_within_three_percent():
    problem = _decay_problem()
    taus = []
    hashes = set()
    for seed in (1, 2, 3):
        run = run_inversion(problem, OptimizerOptions(), SeedControl(seed=seed, starts=3))
        taus.append(run.best_parameters["tau"])
        hashes.add(run.log_sha256())
        # the recorded start points are the reproducible part of the seed
        recorded = [start["initial_parameters"]["tau"] for start in run.starts]
        assert len(recorded) == 3
    for tau in taus:
        assert abs(tau - DECAY_TRUTH["tau"]) / DECAY_TRUTH["tau"] < 0.03, taus
    assert len(hashes) == 3  # different seeds must not share a provenance hash


# ---------------------------------------------------------------------------
# 4. real FEM forward model: recover Young's modulus from nodal displacements
# ---------------------------------------------------------------------------
def test_fem_bar_young_modulus_recovery_and_calibrated_stderr():
    problem = _bar_problem()
    run = run_inversion(problem, OptimizerOptions(), SeedControl(seed=0, starts=1))

    assert run.status == "converged", (run.status, run.stop_reason)
    recovered = run.best_parameters["E"]
    error = abs(recovered - BAR_TRUTH["E"]) / BAR_TRUTH["E"]
    assert error < 0.03, error
    # the linearised standard error must be the same order as the realised error
    reported = run.diagnostics["relative_standard_error"]["E"]
    assert reported is not None
    assert error / 5.0 < reported < error * 5.0, (error, reported)
    assert run.n_forward_evals > 0


# ---------------------------------------------------------------------------
# 5. non-identifiability must be reported, not hidden
# ---------------------------------------------------------------------------
def test_product_non_identifiability_is_reported():
    problem = _bar_problem(unknown_area=True)
    run = run_inversion(problem, OptimizerOptions(), SeedControl(seed=0, starts=1))

    diagnostics = run.diagnostics
    assert diagnostics["effective_rank"] == 1
    assert diagnostics["identifiable"] is False
    assert diagnostics["relative_standard_error"]["E"] is None  # not estimable

    weak = diagnostics["weak_directions"]
    assert len(weak) == 1
    direction = weak[0]
    assert abs(abs(direction["E"]) - math.sqrt(0.5)) < 1e-2, direction
    assert direction["E"] * direction["A"] < 0.0  # anti-correlated null direction

    assert any("not identifiable" in warning for warning in run.warnings)
    # the product itself *is* constrained and must still be right
    product = run.best_parameters["E"] * run.best_parameters["A"]
    truth = BAR_TRUTH["E"] * BAR_AREA
    assert abs(product - truth) / truth < 0.03


# ---------------------------------------------------------------------------
# 6. loss composition and the regularised trade-off
# ---------------------------------------------------------------------------
def test_loss_components_sum_to_total_and_tikhonov_biases_towards_target():
    problem = _bar_problem()
    run = run_inversion(problem, OptimizerOptions(), SeedControl(seed=0, starts=1))
    total = sum(value for key, value in run.best_components.items() if key != "total")
    assert total == pytest.approx(run.best_components["total"], rel=1e-12)

    target = 1.2 * BAR_TRUTH["E"]
    regularised = InverseProblem(
        forward=problem.forward,
        data=problem.data,
        parameters=[
            Parameter("E", 150.0e9, lower=1.0e9, upper=1.0e12, log=True)
        ],
        sigma=float(problem.sigma[0]),
        regularizers=[
            Tikhonov(targets={"E": target}, weight=1.0e6, scales={"E": BAR_TRUTH["E"]})
        ],
        name="axial-bar-regularised",
    )
    pulled = run_inversion(regularised, OptimizerOptions(), SeedControl(seed=0, starts=1))
    plain_distance = abs(run.best_parameters["E"] - target)
    pulled_distance = abs(pulled.best_parameters["E"] - target)
    assert pulled_distance < plain_distance
    assert pulled.best_components["reg0:tikhonov"] > 0.0


# ---------------------------------------------------------------------------
# 7. machine-readable log
# ---------------------------------------------------------------------------
def test_convergence_log_is_machine_readable_and_hash_verified(tmp_path):
    problem = _decay_problem()
    run = run_inversion(
        problem,
        OptimizerOptions(),
        SeedControl(seed=5, starts=2),
        provenance={"git_commit": "0" * 40, "note": "pytest"},
    )
    paths = run.write(tmp_path)
    assert set(paths) == {"log", "summary"}

    text = (tmp_path / "convergence.jsonl").read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line.strip()]
    records = [json.loads(line) for line in lines]
    assert records[0]["kind"] == "header"
    assert records[0]["schema"] == SCHEMA_CONVERGENCE
    assert records[-1]["kind"] == "final"
    body = records[1:-1]
    assert body, "no iteration records were written"
    required = {
        "kind",
        "schema",
        "start",
        "seed",
        "iteration",
        "loss",
        "components",
        "lambda",
        "step_inf",
        "grad_inf",
        "accepted",
        "parameters",
    }
    for record in body:
        assert required <= set(record)
        assert isinstance(record["accepted"], bool)
        assert set(record["parameters"]) == {"A", "tau", "Tinf"}
    assert all(record["accepted"] for record in body) or any(
        not record["accepted"] for record in body
    )

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["schema"] == SCHEMA_SUMMARY
    assert summary["convergence_log_sha256"] == sha256_of(text)
    assert summary["convergence_log_sha256"] == run.log_sha256()
    assert summary["data_sha256"] == problem.data_sha256()
    assert summary["parameters"]["tau"] == pytest.approx(run.best_parameters["tau"])
    assert summary["provenance"]["note"] == "pytest"
    # strict JSON: no Infinity/NaN tokens anywhere
    assert "Infinity" not in text and "NaN" not in text


# ---------------------------------------------------------------------------
# 8. bounds
# ---------------------------------------------------------------------------
def test_bounds_are_respected_and_active_bound_is_reported():
    truth = {"E": 50.0e9}
    clean = _bar_displacements(truth)
    problem = InverseProblem(
        forward=_bar_displacements,
        data=clean,
        parameters=[Parameter("E", 20.0e9, lower=1.0e9, upper=40.0e9, log=True)],
        sigma=1.0e-12,
        name="bounded-bar",
    )
    run = run_inversion(problem, OptimizerOptions(), SeedControl(seed=0, starts=1))
    assert run.best_parameters["E"] <= 40.0e9 * (1.0 + 1e-12)
    assert run.best_parameters["E"] > 0.0
    summary = run.summary()
    assert summary["parameters_at_bound"] == ["E"]
    assert any("bound" in warning for warning in run.warnings)


# ---------------------------------------------------------------------------
# 9. honest failure: data the forward model cannot represent
# ---------------------------------------------------------------------------
def test_unrepresentable_data_reports_residual_excess_not_a_good_fit():
    problem = _decay_problem(noise=0.0, sigma=1e-4)
    # inject a curvature term the exponential-decay forward model cannot express
    contaminated = problem.data + 0.5 * torch.linspace(0.0, 1.0, problem.data.numel(), dtype=FLOAT)
    problem.data = contaminated
    run = run_inversion(problem, OptimizerOptions(), SeedControl(seed=0, starts=2))

    assert run.status in {"converged", "stalled", "max_iter"}
    assert run.diagnostics["reduced_chi2"] is not None
    assert run.diagnostics["reduced_chi2"] > 100.0
    assert any("reduced chi2" in warning for warning in run.warnings)
    # the fit must not be presented as noise-consistent
    assert run.best_loss > 100.0