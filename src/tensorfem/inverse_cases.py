"""Reusable identification cases driving :mod:`tensorfem.inverse`.

The optimiser loop in :mod:`tensorfem.inverse` is forward-model agnostic; the
cases that exercise it live here so that the command line, the traceable
benchmark registry and the worked example all drive the *same* code path
instead of three private copies of the same setup (the situation that made the
old ad-hoc calibrations incomparable).

Two families, each with an independent ground truth:

``decay``
    A single exponential plus offset ``T(t) = Tinf + A exp(-t/tau)`` -- the same
    mathematical object as a decaying LBM velocity signal.  ``tau_true = 4.0 s``
    is known by construction, the data carry 1 % relative noise.
``axial-bar``
    A real :class:`tensorfem.model.TrussModel` solve (4 axial elements,
    ``E_true = 210 GPa``), observed through the free-node displacements with
    0.5 % relative noise.  This is the "solver in the loop" case: every forward
    evaluation assembles and solves a finite-element system.

``axial-bar-product`` is the negative control: it asks for ``E`` *and* ``A``
although the data only contain the product ``E A``.  A correct infrastructure
reports ``identifiable: False`` together with the weak direction instead of a
confident answer.

All numbers produced here are returned as plain dicts that are JSON
serialisable, so callers (CLI, registry gate, benchmark runner) never hand-copy
a value.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch

from .inverse import InverseProblem, OptimizerOptions, Parameter, SeedControl, run_inversion
from .model import TrussModel
from .solvers import solve_linear_static

__all__ = [
    "BAR_AREA",
    "BAR_LENGTH",
    "BAR_LOAD",
    "BAR_NODES",
    "BAR_TRUTH",
    "DECAY_TIMES",
    "DECAY_TRUTH",
    "axial_bar_case",
    "axial_bar_problem",
    "bar_displacements",
    "decay_case",
    "decay_data",
    "decay_problem",
    "decay_times",
    "identify",
]

_FLOAT = torch.float64

#: truth of the exponential-decay case [s]
DECAY_TRUTH = {"A": 60.0, "tau": 4.0, "Tinf": 20.0}
#: truth of the axial-bar case [Pa]
BAR_TRUTH = {"E": 210.0e9}
BAR_AREA = 1.0e-3  # m^2
BAR_LENGTH = 2.0  # m
BAR_NODES = 4
BAR_LOAD = 1.0e4  # N

#: acceptance for every recovered scalar: 3 % relative, the platform standard.
TOLERANCE = 0.03

DECAY_TIMES = torch.linspace(0.0, 20.0, 41, dtype=_FLOAT)


def decay_times() -> torch.Tensor:
    """Sample times of the decay case (single source of truth for clock/plot)."""
    return DECAY_TIMES.clone()


def decay_forward(parameters: Mapping[str, float]) -> torch.Tensor:
    """Forward model ``Tinf + A exp(-t/tau)``."""
    return parameters["Tinf"] + parameters["A"] * torch.exp(-DECAY_TIMES / parameters["tau"])


_DECAY_CLEAN = decay_forward(DECAY_TRUTH)


def decay_data(seed: int = 7) -> torch.Tensor:
    """Synthetic observations: exact curve + 1 % relative Gaussian noise."""
    generator = torch.Generator().manual_seed(seed)
    return _DECAY_CLEAN + 0.01 * _DECAY_CLEAN.abs() * torch.randn(
        _DECAY_CLEAN.shape, dtype=_FLOAT, generator=generator
    )


def decay_problem(seed: int = 7) -> InverseProblem:
    """The decay inverse problem: recover ``(A, tau, Tinf)`` from 41 samples."""
    return InverseProblem(
        forward=decay_forward,
        data=decay_data(seed),
        parameters=[
            Parameter("A", 50.0, lower=1.0, upper=200.0),
            Parameter("tau", 2.0, lower=0.1, upper=50.0, log=True),
            Parameter("Tinf", 0.0, lower=-50.0, upper=100.0),
        ],
        sigma=0.8,
        name="exponential-decay",
    )


# ---------------------------------------------------------------------------
# finite-element case: solver in the loop
# ---------------------------------------------------------------------------
def bar_displacements(parameters: Mapping[str, float]) -> torch.Tensor:
    """Free-node axial displacements of the bar for the given material values."""
    x = torch.linspace(0.0, BAR_LENGTH, BAR_NODES + 1, dtype=_FLOAT)
    nodes = torch.stack([x, torch.zeros_like(x)], dim=1)
    elements = torch.tensor([[i, i + 1] for i in range(BAR_NODES)])
    forces = torch.zeros(2 * (BAR_NODES + 1), dtype=_FLOAT)
    forces[2 * BAR_NODES] = BAR_LOAD
    fixed = torch.tensor([0] + list(range(1, 2 * (BAR_NODES + 1), 2)))
    model = TrussModel(
        nodes=nodes,
        elements=elements,
        young_modulus=torch.tensor([parameters["E"]], dtype=_FLOAT),
        area=torch.tensor([parameters.get("A", BAR_AREA)], dtype=_FLOAT),
        forces=forces,
        fixed_dofs=fixed,
    )
    return solve_linear_static(model).displacement[torch.arange(1, BAR_NODES + 1) * 2]


_BAR_CLEAN = bar_displacements(BAR_TRUTH)


def axial_bar_problem(*, unknown_area: bool = False, seed: int = 11) -> InverseProblem:
    """Inverse problem on the FE forward solve (0.5 % relative noise)."""
    generator = torch.Generator().manual_seed(seed)
    data = _BAR_CLEAN + 0.005 * _BAR_CLEAN.abs() * torch.randn(
        _BAR_CLEAN.shape, dtype=_FLOAT, generator=generator
    )
    parameters = [Parameter("E", 150.0e9, lower=1.0e9, upper=1.0e12, log=True)]
    if unknown_area:
        parameters.append(Parameter("A", 5.0e-4, lower=1.0e-6, upper=1.0e-1, log=True))
    return InverseProblem(
        forward=bar_displacements,
        data=data,
        parameters=parameters,
        sigma=0.005 * float(_BAR_CLEAN[-1]),
        name="axial-bar-product" if unknown_area else "axial-bar",
    )


# ---------------------------------------------------------------------------
# case drivers
# ---------------------------------------------------------------------------
def _summary_row(run, *, reference: Mapping[str, float], keys: tuple[str, ...]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "status": run.status,
        "converged": run.status == "converged",
        "n_iterations": len(run.iterations),
        "log_sha256": run.log_sha256(),
        "seed": run.seed_control.seed,
        "starts": run.seed_control.starts,
        "identifiable": run.diagnostics["identifiable"],
        "condition_number": run.diagnostics["condition_number"],
    }
    for key in keys:
        value = float(run.best_parameters[key])
        row[key] = value
        row[f"{key}_reference"] = float(reference[key])
        row[f"{key}_relative_error"] = abs(value - reference[key]) / abs(reference[key])
        row[f"{key}_within_tolerance"] = row[f"{key}_relative_error"] < TOLERANCE
    return row


def decay_case(
    *,
    seed: int = 1,
    starts: int = 3,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Recover ``tau`` (and ``A``, ``Tinf``) from 1 %-noise synthetic data."""
    run = run_inversion(
        decay_problem(),
        OptimizerOptions(),
        SeedControl(seed=seed, starts=starts),
        provenance=dict(provenance or {}),
    )
    row = _summary_row(run, reference=DECAY_TRUTH, keys=("tau", "A", "Tinf"))
    row["case"] = "decay"
    row["passed"] = bool(row["converged"] and row["tau_within_tolerance"])
    return {"row": row, "run": run}


def axial_bar_case(*, seed: int = 0, provenance: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Recover Young's modulus from a finite-element forward solve."""
    run = run_inversion(
        axial_bar_problem(),
        OptimizerOptions(),
        SeedControl(seed=seed, starts=1),
        provenance=dict(provenance or {}),
    )
    row = _summary_row(run, reference=BAR_TRUTH, keys=("E",))
    row["case"] = "axial-bar"
    row["passed"] = bool(row["converged"] and row["E_within_tolerance"])
    return {"row": row, "run": run}


def product_case(*, seed: int = 0, provenance: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Negative control: ``E`` and ``A`` are not identifiable from ``E A``.

    The gate is inverted on purpose -- this case passes only when the
    infrastructure *refuses* to report two confident values.
    """
    run = run_inversion(
        axial_bar_problem(unknown_area=True),
        OptimizerOptions(),
        SeedControl(seed=seed, starts=1),
        provenance=dict(provenance or {}),
    )
    product = float(run.best_parameters["E"]) * float(run.best_parameters["A"])
    product_reference = BAR_TRUTH["E"] * BAR_AREA
    row = {
        "case": "axial-bar-product",
        "status": run.status,
        "converged": run.status == "converged",
        "identifiable": run.diagnostics["identifiable"],
        "effective_rank": run.diagnostics["effective_rank"],
        "condition_number": run.diagnostics["condition_number"],
        "product": product,
        "product_reference": product_reference,
        "product_relative_error": abs(product - product_reference) / product_reference,
        "weak_direction": run.diagnostics.get("weak_directions", []),
        "warnings": run.warnings,
        "log_sha256": run.log_sha256(),
        "seed": run.seed_control.seed,
    }
    row["passed"] = bool(
        not row["identifiable"]
        and row["effective_rank"] < 2
        and row["product_relative_error"] < TOLERANCE
    )
    return {"row": row, "run": run}


_CASES = {
    "decay": decay_case,
    "axial-bar": axial_bar_case,
    "axial-bar-product": product_case,
}


def identify(
    case: str = "decay",
    *,
    seed: int = 1,
    starts: int = 3,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one named case and return the JSON-serialisable outcome.

    ``case`` is one of ``decay``, ``axial-bar``, ``axial-bar-product``.
    Raises :class:`KeyError` for an unknown case so a typo cannot silently run
    the wrong physics.
    """
    if case not in _CASES:
        raise KeyError(f"unknown inverse case {case!r}; choose from {sorted(_CASES)}")
    driver = _CASES[case]
    if case == "decay":
        return driver(seed=seed, starts=starts, provenance=provenance)
    return driver(seed=seed, provenance=provenance)