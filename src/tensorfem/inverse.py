"""Inverse parameter identification: optimizer loop, loss composition,
machine-readable convergence logs and seed control.

Why this module exists
----------------------
Every TensorFEM workflow that is *calibrated* rather than merely *solved* needs the
same four pieces: an unknown-parameter description, a forward model, a loss with
regularisation, and an auditable record of what the optimiser did.  Those pieces
used to be re-implemented ad hoc inside individual benchmark scripts, which made
two things impossible: comparing two calibrations without hand-copying numbers,
and stating honestly whether a recovered parameter was identifiable at all.

:func:`run_inversion` is a Levenberg-Marquardt loop over a bounded, optionally
log-transformed parameter space, driven by a user-supplied forward callable
(typically a TensorFEM solver).  Every trial is written to a JSON-lines
convergence log whose SHA-256 is part of the run summary, so a run can be
re-verified instead of trusted.

Design notes
------------
* **Non-dimensional search space.**  ``Parameter.to_internal`` maps each unknown to
  the coordinates the optimiser moves in (``log`` for strictly positive
  quantities, ``x / scale`` otherwise), so a stiffness ``2.1e11`` and a decay
  constant ``0.3`` take comparable Gauss-Newton steps.
* **Composable loss.**  ``loss = sum(data residuals**2) + sum(regulariser terms)``;
  each term is reported separately per iteration so a regularisation weight can be
  audited instead of buried in one scalar.
* **Honest diagnostics.**  The weighted Jacobian at the estimate gives its rank,
  condition number, per-parameter linearised relative standard error and the null
  directions.  A converged run on a rank-deficient problem reports ``converged``
  *plus* ``identifiable: False`` plus an explicit warning -- never plain success.
* **Seed control.**  A single start is deterministic.  With ``starts > 1`` the
  jittered starts come from ``torch.Generator().manual_seed(seed + 1009 * i)`` and
  every start point is recorded in the log, so the same seed reproduces the same
  log hash.

Schemas
-------
``tensorfem.inverse-convergence/1``  one JSON object per line
``tensorfem.inverse-summary/1``      run summary written next to the log
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

SCHEMA_CONVERGENCE = "tensorfem.inverse-convergence/1"
SCHEMA_SUMMARY = "tensorfem.inverse-summary/1"
_FLOAT = torch.float64
_TWO_POW_63 = 2**63


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _as_vector(value: Any, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=_FLOAT).reshape(-1).clone()
    if not bool(torch.all(torch.isfinite(tensor))):
        raise ValueError(f"{name} contains a non-finite entry")
    return tensor


def _finite(value: Any) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def canonical_json(payload: Any) -> str:
    """Deterministic, spec-compliant JSON (sorted keys, no NaN/Infinity)."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    """Recursively coerce to JSON-legal Python (non-finite floats -> None)."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        return _finite(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, torch.Tensor):
        return _jsonable(value.reshape(-1).tolist())
    return value


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Parameter:
    """One unknown.

    ``to_internal`` is the coordinate the optimiser moves in: ``log(x)`` when
    ``log`` is set (strictly positive unknowns), ``x / scale`` otherwise.  Bounds
    are stated in physical units and transformed alongside.
    """

    name: str
    x0: float
    lower: float | None = None
    upper: float | None = None
    log: bool = False
    scale: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Parameter.name must be non-empty")
        if not math.isfinite(float(self.x0)):
            raise ValueError(f"Parameter({self.name}).x0 must be finite")
        if not (math.isfinite(float(self.scale)) and float(self.scale) > 0.0):
            raise ValueError(f"Parameter({self.name}).scale must be finite and positive")
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError(f"Parameter({self.name}) has lower > upper")
        if self.log:
            if self.x0 <= 0.0:
                raise ValueError(f"Parameter({self.name}) is log-scaled so x0 must be > 0")
            if self.lower is not None and self.lower <= 0.0:
                raise ValueError(f"Parameter({self.name}) is log-scaled so lower must be > 0")

    def to_internal(self, value: float) -> float:
        value = float(value)
        if self.log:
            if value <= 0.0:
                raise ValueError(f"Parameter({self.name}) received a non-positive value")
            return math.log(value)
        return value / float(self.scale)

    def to_external(self, internal: float) -> float:
        if self.log:
            return math.exp(float(internal))
        return float(internal) * float(self.scale)

    def internal_bounds(self) -> tuple[float, float]:
        low = self.to_internal(self.lower) if self.lower is not None else -math.inf
        high = self.to_internal(self.upper) if self.upper is not None else math.inf
        return low, high

    def at_bound(self, value: float, *, rtol: float = 1e-9) -> bool:
        for bound in (self.lower, self.upper):
            if bound is None:
                continue
            if abs(float(value) - float(bound)) <= rtol * (abs(float(bound)) + 1e-300):
                return True
        return False

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "x0": float(self.x0),
            "lower": None if self.lower is None else float(self.lower),
            "upper": None if self.upper is None else float(self.upper),
            "log": bool(self.log),
            "scale": float(self.scale),
        }


# ---------------------------------------------------------------------------
# loss composition
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Tikhonov:
    """Quadratic pull of selected parameters towards target values.

    Contribution ``sqrt(weight) * (x_i - target_i) / scale_i`` so ``weight`` is
    comparable to a data weight ``1 / sigma**2``.
    """

    targets: Mapping[str, float]
    weight: float = 1.0
    scales: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if float(self.weight) < 0.0:
            raise ValueError("Tikhonov.weight must be non-negative")
        if not self.targets:
            raise ValueError("Tikhonov.targets must not be empty")
        for name, value in self.targets.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"Tikhonov target for {name!r} must be finite")
        for name, value in self.scales.items():
            if not (math.isfinite(float(value)) and float(value) > 0.0):
                raise ValueError(f"Tikhonov scale for {name!r} must be positive")

    def scaled_residual(self, external: Mapping[str, float]) -> torch.Tensor:
        missing = [name for name in self.targets if name not in external]
        if missing:
            raise KeyError(f"Tikhonov targets unknown parameters: {missing}")
        entries = [
            (float(external[name]) - float(target)) / float(self.scales.get(name, 1.0))
            for name, target in self.targets.items()
        ]
        return _as_vector(entries, name="tikhonov") * math.sqrt(float(self.weight))

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": "tikhonov",
            "weight": float(self.weight),
            "targets": {str(k): float(v) for k, v in self.targets.items()},
            "scales": {str(k): float(v) for k, v in self.scales.items()},
        }


# ---------------------------------------------------------------------------
# problem definition
# ---------------------------------------------------------------------------
@dataclass
class InverseProblem:
    """Forward callable + observations + unknowns + loss weights.

    ``forward`` receives ``{parameter name: physical value}`` and returns an
    observable vector (a solver result reduced to a vector, a time history, an
    eigenfrequency table, ...).  ``data`` are the matching measurements and
    ``sigma`` the per-observable standard deviation weighting
    ``(model - data) / sigma``.
    """

    forward: Callable[[Mapping[str, float]], Any]
    data: Sequence[float] | torch.Tensor
    parameters: Sequence[Parameter]
    sigma: float | Sequence[float] = 1.0
    regularizers: Sequence[Any] = ()
    name: str = "inverse-problem"

    def __post_init__(self) -> None:
        self.parameters = tuple(self.parameters)
        if not self.parameters:
            raise ValueError("InverseProblem needs at least one parameter")
        names = [p.name for p in self.parameters]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate parameter names: {names}")
        self.data = _as_vector(self.data, name="data")
        if self.data.numel() == 0:
            raise ValueError("InverseProblem.data must not be empty")
        sigma = torch.as_tensor(self.sigma, dtype=_FLOAT).reshape(-1)
        if sigma.numel() == 1:
            sigma = sigma.expand(self.data.numel()).clone()
        if sigma.numel() != self.data.numel():
            raise ValueError("sigma must be scalar or match the data length")
        if not bool(torch.all(torch.isfinite(sigma))) or bool(torch.any(sigma <= 0.0)):
            raise ValueError("sigma entries must be positive and finite")
        self.sigma = sigma
        self.regularizers = tuple(self.regularizers)
        self._u0 = torch.tensor(
            [p.to_internal(p.x0) for p in self.parameters], dtype=_FLOAT
        )

    # -- accessors ---------------------------------------------------------
    @property
    def names(self) -> list[str]:
        return [p.name for p in self.parameters]

    @property
    def start_internal(self) -> torch.Tensor:
        return self._u0.clone()

    @property
    def n_parameters(self) -> int:
        return len(self.parameters)

    def to_external(self, internal: torch.Tensor) -> dict[str, float]:
        internal = torch.as_tensor(internal, dtype=_FLOAT).reshape(-1)
        if internal.numel() != self.n_parameters:
            raise ValueError("internal vector length does not match the parameter count")
        return {p.name: p.to_external(float(internal[i])) for i, p in enumerate(self.parameters)}

    def to_internal(self, external: Mapping[str, float]) -> torch.Tensor:
        missing = [p.name for p in self.parameters if p.name not in external]
        if missing:
            raise KeyError(f"missing parameter values: {missing}")
        return torch.tensor(
            [p.to_internal(external[p.name]) for p in self.parameters], dtype=_FLOAT
        )

    def clip_internal(self, internal: torch.Tensor) -> torch.Tensor:
        out = torch.as_tensor(internal, dtype=_FLOAT).reshape(-1).clone()
        for i, parameter in enumerate(self.parameters):
            low, high = parameter.internal_bounds()
            out[i] = min(max(float(out[i]), low), high)
        return out

    # -- forward / residuals ----------------------------------------------
    def predict(self, internal: torch.Tensor) -> torch.Tensor:
        return _as_vector(
            self.forward(self.to_external(internal)), name=f"{self.name}:forward"
        )

    def data_residuals(
        self,
        internal: torch.Tensor,
        predict: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        model = predict(internal) if predict is not None else self.predict(internal)
        if model.numel() != self.data.numel():
            raise ValueError(
                f"forward returned {model.numel()} observables, data has {self.data.numel()}"
            )
        return (model - self.data) / self.sigma

    def regularizer_terms(self, internal: torch.Tensor) -> dict[str, torch.Tensor]:
        external = self.to_external(internal)
        terms: dict[str, torch.Tensor] = {}
        for index, regularizer in enumerate(self.regularizers):
            residual = regularizer.scaled_residual(external)
            label = "tikhonov" if isinstance(regularizer, Tikhonov) else type(regularizer).__name__.lower()
            terms[f"reg{index}:{label}"] = _as_vector(residual, name=f"regularizer-{index}")
        return terms

    def residuals(
        self,
        internal: torch.Tensor,
        predict: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        parts = [self.data_residuals(internal, predict)]
        parts.extend(self.regularizer_terms(internal).values())
        return torch.cat(parts)

    def loss_components(self, internal: torch.Tensor) -> dict[str, float]:
        data = self.data_residuals(internal)
        components = {"data": float(torch.sum(data**2))}
        for key, term in self.regularizer_terms(internal).items():
            components[key] = float(torch.sum(term**2))
        components["total"] = float(sum(components.values()))
        return components

    def loss(self, internal: torch.Tensor) -> float:
        return self.loss_components(internal)["total"]

    # -- provenance --------------------------------------------------------
    def configuration(self) -> dict[str, Any]:
        sigma_min = float(torch.min(self.sigma))
        sigma_max = float(torch.max(self.sigma))
        return {
            "name": self.name,
            "n_observations": int(self.data.numel()),
            "parameters": [p.as_dict() for p in self.parameters],
            "sigma": {
                "min": sigma_min,
                "max": sigma_max,
                "kind": "scalar" if sigma_min == sigma_max else "per-observable",
            },
            "regularizers": [
                reg.as_dict() if hasattr(reg, "as_dict") else {"type": type(reg).__name__}
                for reg in self.regularizers
            ],
        }

    def data_sha256(self) -> str:
        payload = _jsonable(
            {"data": self.data, "sigma": self.sigma, "config": self.configuration()}
        )
        return sha256_of(canonical_json(payload))


# ---------------------------------------------------------------------------
# optimiser options / seeding
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OptimizerOptions:
    """Levenberg-Marquardt controls."""

    max_iter: int = 100
    gtol: float = 1e-10
    xtol: float = 1e-12
    ftol: float = 1e-14
    stall_patience: int = 8
    lm_lambda0: float = 1e-3
    lm_nu: float = 10.0
    max_inner: int = 25
    fd_rel_step: float = 1e-6
    fd_scheme: str = "forward"
    cond_limit: float = 1e10

    def __post_init__(self) -> None:
        if self.max_iter < 1:
            raise ValueError("max_iter must be >= 1")
        if self.max_inner < 1:
            raise ValueError("max_inner must be >= 1")
        if self.lm_lambda0 <= 0.0:
            raise ValueError("lm_lambda0 must be positive")
        if self.lm_nu <= 1.0:
            raise ValueError("lm_nu must be > 1")
        if self.fd_rel_step <= 0.0:
            raise ValueError("fd_rel_step must be positive")
        if self.fd_scheme not in ("forward", "central"):
            raise ValueError("fd_scheme must be 'forward' or 'central'")

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_iter": self.max_iter,
            "gtol": self.gtol,
            "xtol": self.xtol,
            "ftol": self.ftol,
            "stall_patience": self.stall_patience,
            "lm_lambda0": self.lm_lambda0,
            "lm_nu": self.lm_nu,
            "max_inner": self.max_inner,
            "fd_rel_step": self.fd_rel_step,
            "fd_scheme": self.fd_scheme,
            "cond_limit": self.cond_limit,
        }


@dataclass(frozen=True)
class SeedControl:
    """Reproducible multi-start seeding.

    ``starts == 1`` is deterministic (nominal ``x0``, no random draw).  Otherwise
    start ``i`` draws from ``torch.Generator().manual_seed(seed + 1009 * i)``.
    """

    seed: int = 0
    starts: int = 1
    jitter: float = 0.5

    def __post_init__(self) -> None:
        if int(self.seed) < 0:
            raise ValueError("seed must be non-negative")
        if int(self.starts) < 1:
            raise ValueError("starts must be >= 1")
        if float(self.jitter) < 0.0:
            raise ValueError("jitter must be non-negative")

    def start_seed(self, index: int) -> int:
        return (int(self.seed) + 1009 * int(index)) % _TWO_POW_63

    def generator(self, index: int) -> torch.Generator:
        return torch.Generator().manual_seed(self.start_seed(index))

    def start_points(self, problem: InverseProblem) -> list[tuple[int, torch.Tensor]]:
        base = problem.start_internal
        points: list[tuple[int, torch.Tensor]] = [(0, base.clone())]
        for index in range(1, int(self.starts)):
            if float(self.jitter) == 0.0:
                points.append((index, base.clone()))
                continue
            noise = torch.randn(
                problem.n_parameters, dtype=_FLOAT, generator=self.generator(index)
            )
            points.append((index, problem.clip_internal(base + float(self.jitter) * noise)))
        return points

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed": int(self.seed),
            "starts": int(self.starts),
            "jitter": float(self.jitter),
        }


# ---------------------------------------------------------------------------
# numerical kernels
# ---------------------------------------------------------------------------
def fd_jacobian(
    func: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    *,
    rel_step: float = 1e-6,
    scheme: str = "forward",
) -> torch.Tensor:
    """Finite-difference Jacobian of ``func`` at ``x`` in float64."""
    x = torch.as_tensor(x, dtype=_FLOAT).reshape(-1).clone()
    base = torch.as_tensor(func(x), dtype=_FLOAT).reshape(-1)
    n = x.numel()
    m = base.numel()
    jac = torch.empty((m, n), dtype=_FLOAT)
    step = rel_step * (x.abs() + 1.0)
    for j in range(n):
        xp = x.clone()
        xp[j] += step[j]
        forward = torch.as_tensor(func(xp), dtype=_FLOAT).reshape(-1)
        if scheme == "central":
            xm = x.clone()
            xm[j] -= step[j]
            backward = torch.as_tensor(func(xm), dtype=_FLOAT).reshape(-1)
            jac[:, j] = (forward - backward) / (2.0 * step[j])
        else:
            jac[:, j] = (forward - base) / step[j]
    return jac


def _solve_normal(matrix: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    try:
        return torch.linalg.solve(matrix, rhs)
    except Exception:  # singular normal-equation matrix
        return torch.linalg.lstsq(matrix, rhs).solution


def _iteration_record(
    problem: InverseProblem,
    internal: torch.Tensor,
    components: Mapping[str, float],
    *,
    start: int,
    seed: int,
    iteration: int,
    lam: float | None,
    step_inf: float | None,
    grad_inf: float | None,
    accepted: bool,
) -> dict[str, Any]:
    return {
        "kind": "iteration",
        "schema": SCHEMA_CONVERGENCE,
        "start": int(start),
        "seed": int(seed),
        "iteration": int(iteration),
        "loss": _finite(components["total"]),
        "components": {key: _finite(value) for key, value in components.items()},
        "lambda": None if lam is None else _finite(lam),
        "step_inf": None if step_inf is None else _finite(step_inf),
        "grad_inf": None if grad_inf is None else _finite(grad_inf),
        "accepted": bool(accepted),
        "parameters": {k: _finite(v) for k, v in problem.to_external(internal).items()},
    }


def _lm_solve(
    problem: InverseProblem,
    options: OptimizerOptions,
    u0: torch.Tensor,
    predict: Callable[[torch.Tensor], torch.Tensor],
    *,
    start: int,
    seed: int,
) -> tuple[list[dict[str, Any]], str, str, torch.Tensor, dict[str, float]]:
    """One Levenberg-Marquardt run from ``u0``."""

    def residual(u: torch.Tensor) -> torch.Tensor:
        return problem.residuals(u, predict=predict)

    u = problem.clip_internal(u0)
    components = problem.loss_components(u)
    records: list[dict[str, Any]] = []
    if not math.isfinite(components["total"]):
        records.append(
            _iteration_record(
                problem, u, components, start=start, seed=seed, iteration=0,
                lam=None, step_inf=None, grad_inf=None, accepted=False,
            )
        )
        return records, "invalid", "non-finite initial loss", u, components

    lam = float(options.lm_lambda0)
    status, reason = "max_iter", "iteration budget exhausted"
    stall = 0
    for iteration in range(1, options.max_iter + 1):
        residuals = residual(u)
        jac = fd_jacobian(
            residual, u, rel_step=options.fd_rel_step, scheme=options.fd_scheme
        )
        normal = jac.T @ jac
        gradient = jac.T @ residuals
        grad_inf = float(torch.max(torch.abs(gradient))) if gradient.numel() else 0.0
        if grad_inf <= options.gtol:
            status, reason = "converged", "gradient infinity norm below gtol"
            break

        accepted = False
        delta = torch.zeros_like(u)
        best_trial_loss: float | None = None
        best_trial_step = 0.0
        for _ in range(options.max_inner):
            diagonal = torch.diagonal(normal).clone()
            diagonal = torch.where(diagonal > 0, diagonal, torch.ones_like(diagonal))
            step = _solve_normal(normal + lam * torch.diag(diagonal), -gradient)
            trial = problem.clip_internal(u + step)
            trial_components = problem.loss_components(trial)
            trial_delta = trial - u
            trial_step_inf = float(torch.max(torch.abs(trial_delta)))
            moved = trial_step_inf > 0.0
            if math.isfinite(trial_components["total"]) and moved:
                if best_trial_loss is None or trial_components["total"] < best_trial_loss:
                    best_trial_loss = float(trial_components["total"])
                    best_trial_step = trial_step_inf
            if (
                math.isfinite(trial_components["total"])
                and trial_components["total"] < components["total"]
                and moved
            ):
                accepted = True
                delta = trial_delta
                break
            lam = min(lam * options.lm_nu, 1e16)

        if not accepted:
            # Safeguard for ill-conditioned valleys: the damped normal equations
            # keep proposing steps that overshoot, λ escalation then makes every
            # trial too small to change the loss in float64, and a strict-descent
            # test stalls while the gradient is still non-zero.  Fall back to a
            # backtracking steepest-descent step in optimiser coordinates.
            descent = -gradient
            grad_max = float(torch.max(torch.abs(descent))) if descent.numel() else 0.0
            if grad_max > 0.0:
                trial_step = descent / grad_max
                for _ in range(40):
                    trial = problem.clip_internal(u + trial_step)
                    trial_components = problem.loss_components(trial)
                    if math.isfinite(trial_components["total"]):
                        if best_trial_loss is None or trial_components["total"] < best_trial_loss:
                            best_trial_loss = float(trial_components["total"])
                    trial_delta = trial - u
                    if (
                        math.isfinite(trial_components["total"])
                        and trial_components["total"] < components["total"]
                        and float(torch.max(torch.abs(trial_delta))) > 0.0
                    ):
                        accepted = True
                        delta = trial_delta
                        lam = float(options.lm_lambda0)
                        break
                    trial_step = trial_step * 0.5

        step_inf = float(torch.max(torch.abs(delta)))
        records.append(
            _iteration_record(
                problem, u, components, start=start, seed=seed, iteration=iteration,
                lam=lam, step_inf=step_inf, grad_inf=grad_inf, accepted=accepted,
            )
        )
        if not accepted:
            # Distinguish "at the objective's float64 resolution" (a plateau) from
            # "cannot move at all": if a trial did move and its loss is within
            # ftol*|loss| of the current one, no further reduction is representable.
            plateau = (
                best_trial_loss is not None
                and best_trial_step > 0.0
                and abs(best_trial_loss - components["total"])
                <= options.ftol * max(abs(components["total"]), 1e-300)
            )
            if plateau:
                status, reason = "converged", "loss plateau at float64 resolution"
            else:
                status, reason = "stalled", "no trial step reduced the loss within max_inner"
            break

        previous_loss = components["total"]
        u = problem.clip_internal(u + delta)
        components = problem.loss_components(u)
        lam = max(lam / options.lm_nu, 1e-14)

        u_absmax = float(torch.max(torch.abs(u)))
        if step_inf <= options.xtol * (1.0 + u_absmax):
            status, reason = "converged", "step infinity norm below xtol"
            break
        if components["total"] > 0.0:
            relative = (previous_loss - components["total"]) / max(previous_loss, 1e-300)
            stall = stall + 1 if relative <= options.ftol else 0
            if stall >= options.stall_patience:
                # A plateau of accepted-but-tiny steps is convergence in the
                # objective, not a failure: reserve "stalled" for runs that
                # cannot find any loss-reducing step at all.
                status, reason = "converged", "relative loss reduction below ftol"
                break
    return records, status, reason, u, components


# ---------------------------------------------------------------------------
# diagnostics
# ---------------------------------------------------------------------------
def identifiability_diagnostics(
    problem: InverseProblem,
    internal: torch.Tensor,
    *,
    options: OptimizerOptions | None = None,
    predict: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Rank, condition number, null directions, linearised relative stderr."""

    options = options or OptimizerOptions()
    u = torch.as_tensor(internal, dtype=_FLOAT).reshape(-1).clone()

    def residual(vector: torch.Tensor) -> torch.Tensor:
        return problem.residuals(vector, predict=predict)

    jac = fd_jacobian(residual, u, rel_step=options.fd_rel_step, scheme=options.fd_scheme)
    residuals = residual(u)
    m, n = jac.shape
    _, singular, vh = torch.linalg.svd(jac, full_matrices=False)
    s_max = float(singular[0]) if singular.numel() else 0.0
    # Numerical rank at machine tolerance, and the *engineering* rank at the
    # condition-number limit: a direction whose singular value is smaller than
    # s_max / cond_limit cannot be resolved to useful precision even though it is
    # formally non-zero.  ``identifiable`` uses the second (conservative) notion.
    tol = max(m, n) * float(torch.finfo(_FLOAT).eps) * s_max
    rank = int(torch.sum(singular > tol)) if singular.numel() else 0
    weak_tol = s_max / max(float(options.cond_limit), 1.0)
    effective_rank = int(torch.sum(singular > weak_tol)) if singular.numel() else 0
    condition = float(s_max / float(singular[-1])) if rank == n and s_max > 0 else math.inf

    def _direction(index: int) -> dict[str, float]:
        vector = vh[index] if index < vh.shape[0] else torch.zeros(n, dtype=_FLOAT)
        return {name: float(vector[i]) for i, name in enumerate(problem.names)}

    null_directions = [
        _direction(index)
        for index in range(n)
        if not (index < singular.numel() and float(singular[index]) > tol)
    ]
    weak_directions = [
        _direction(index)
        for index in range(n)
        if not (index < singular.numel() and float(singular[index]) > weak_tol)
    ]

    column_norms = torch.linalg.norm(jac, dim=0)
    total = float(torch.sum(column_norms)) or 1.0
    sensitivity = {
        name: float(column_norms[i]) / total for i, name in enumerate(problem.names)
    }

    rel_stderr: dict[str, float | None] = {name: None for name in problem.names}
    correlation: dict[str, dict[str, float | None]] = {}
    if effective_rank == n and m >= n and s_max > 0.0:
        dof = max(m - n, 1)
        variance = float(torch.sum(residuals**2)) / dof
        cov = torch.linalg.inv(jac.T @ jac) * variance
        scales = torch.sqrt(torch.diagonal(cov).clamp(min=0.0))
        for i, name in enumerate(problem.names):
            parameter = problem.parameters[i]
            external = parameter.to_external(float(u[i]))
            # The covariance lives in the optimiser coordinates.  For a log
            # parameter one log unit *is* a relative change, so nothing else is
            # needed; for a scaled linear parameter an internal step dp maps to
            # dp * scale in physical units.
            if parameter.log:
                # A standard deviation of one log unit is a 100 % relative change,
                # so the log-coordinate stderr *is* the relative stderr.
                rel_stderr[name] = _finite(float(scales[i]))
            else:
                rel_stderr[name] = (
                    _finite(float(scales[i]) * float(parameter.scale) / abs(external))
                    if external != 0.0
                    else None
                )
        denominator = torch.outer(scales, scales)
        corr = cov / torch.where(
            denominator > 0, denominator, torch.ones_like(denominator)
        )
        for i, name in enumerate(problem.names):
            correlation[name] = {
                other: _finite(float(corr[i, j])) for j, other in enumerate(problem.names)
            }

    return {
        "rank": rank,
        "effective_rank": effective_rank,
        "n_parameters": n,
        "n_observations": m,
        "identifiable": bool(effective_rank == n and condition <= options.cond_limit),
        "condition_number": _finite(condition),
        "singular_values": [_finite(v) for v in singular.tolist()],
        "null_directions": null_directions,
        "weak_directions": weak_directions,
        "sensitivity": sensitivity,
        "relative_standard_error": rel_stderr,
        "correlation": correlation,
        "residual_rms": float(torch.sqrt(torch.mean(residuals**2))),
        "reduced_chi2": (
            float(torch.sum(residuals**2)) / max(m - n, 1) if m > n else None
        ),
    }


# ---------------------------------------------------------------------------
# run object
# ---------------------------------------------------------------------------
@dataclass
class ConvergenceRun:
    """Auditable result of :func:`run_inversion`."""

    problem: InverseProblem
    options: OptimizerOptions
    seed_control: SeedControl
    best_internal: torch.Tensor
    best_parameters: dict[str, float]
    best_loss: float
    best_components: dict[str, float]
    status: str
    stop_reason: str
    start_index: int
    starts: list[dict[str, Any]]
    iterations: list[dict[str, Any]]
    diagnostics: dict[str, Any]
    warnings: list[str]
    provenance: dict[str, Any]
    n_forward_evals: int
    elapsed_s: float

    # -- log ---------------------------------------------------------------
    def header_record(self) -> dict[str, Any]:
        return {
            "kind": "header",
            "schema": SCHEMA_CONVERGENCE,
            "problem": self.problem.configuration(),
            "data_sha256": self.problem.data_sha256(),
            "options": self.options.as_dict(),
            "seed_control": self.seed_control.as_dict(),
            "starts": self.starts,
            "provenance": self.provenance,
        }

    def final_record(self) -> dict[str, Any]:
        return {
            "kind": "final",
            "schema": SCHEMA_CONVERGENCE,
            "status": self.status,
            "stop_reason": self.stop_reason,
            "selected_start": int(self.start_index),
            "loss": _finite(self.best_loss),
            "components": {k: _finite(v) for k, v in self.best_components.items()},
            "parameters": {k: _finite(v) for k, v in self.best_parameters.items()},
            "identifiable": bool(self.diagnostics["identifiable"]),
            "n_forward_evals": int(self.n_forward_evals),
            "warnings": list(self.warnings),
        }

    def records(self) -> list[dict[str, Any]]:
        return [self.header_record(), *self.iterations, self.final_record()]

    def log_jsonl(self) -> str:
        return "".join(canonical_json(record) + "\n" for record in self.records())

    def log_sha256(self) -> str:
        return sha256_of(self.log_jsonl())

    # -- summary -----------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        external = self.problem.to_external(self.best_internal)
        at_bound = [
            p.name for i, p in enumerate(self.problem.parameters) if p.at_bound(external[p.name])
        ]
        payload = {
            "kind": "summary",
            "schema": SCHEMA_SUMMARY,
            "problem": self.problem.configuration(),
            "data_sha256": self.problem.data_sha256(),
            "options": self.options.as_dict(),
            "seed_control": self.seed_control.as_dict(),
            "starts": self.starts,
            "selected_start": int(self.start_index),
            "status": self.status,
            "stop_reason": self.stop_reason,
            "parameters": {k: _finite(v) for k, v in self.best_parameters.items()},
            "loss": _finite(self.best_loss),
            "loss_components": {k: _finite(v) for k, v in self.best_components.items()},
            "diagnostics": self.diagnostics,
            "warnings": list(self.warnings),
            "parameters_at_bound": at_bound,
            "n_forward_evals": int(self.n_forward_evals),
            "n_iterations": len([r for r in self.iterations if r["accepted"]]),
            "elapsed_s": _finite(self.elapsed_s),
            "provenance": self.provenance,
            "convergence_log_sha256": self.log_sha256(),
        }
        return _jsonable(payload)

    def write(self, directory: str | Path) -> dict[str, str]:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        log_path = target / "convergence.jsonl"
        summary_path = target / "summary.json"
        log_path.write_text(self.log_jsonl(), encoding="utf-8")
        summary_path.write_text(
            json.dumps(self.summary(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return {"log": str(log_path), "summary": str(summary_path)}


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run_inversion(
    problem: InverseProblem,
    options: OptimizerOptions | None = None,
    seed_control: SeedControl | None = None,
    *,
    provenance: Mapping[str, Any] | None = None,
) -> ConvergenceRun:
    """Identify ``problem.parameters`` from ``problem.data`` and return the audit trail."""

    options = options or OptimizerOptions()
    seed_control = seed_control or SeedControl()
    started = time.perf_counter()
    counter = {"n": 0}

    def predict(internal: torch.Tensor) -> torch.Tensor:
        counter["n"] += 1
        return problem.predict(internal)

    iterations: list[dict[str, Any]] = []
    starts: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for index, u0 in seed_control.start_points(problem):
        records, status, reason, u_final, components = _lm_solve(
            problem,
            options,
            u0,
            predict,
            start=index,
            seed=seed_control.start_seed(index),
        )
        iterations.extend(records)
        start_record = {
            "start": int(index),
            "seed": int(seed_control.start_seed(index)),
            "initial_parameters": {
                k: _finite(v) for k, v in problem.to_external(u0).items()
            },
            "status": status,
            "stop_reason": reason,
            "final_loss": _finite(components["total"]),
            "final_parameters": {k: _finite(v) for k, v in problem.to_external(u_final).items()},
            "n_accepted": len([r for r in records if r["accepted"]]),
        }
        starts.append(start_record)
        if not math.isfinite(components["total"]):
            continue
        if best is None or components["total"] < best["components"]["total"]:
            best = {
                "index": index,
                "internal": u_final,
                "components": components,
                "status": status,
                "reason": reason,
            }

    if best is None:
        raise RuntimeError("every start produced a non-finite loss; check the forward model")

    diagnostics = identifiability_diagnostics(
        problem, best["internal"], options=options, predict=predict
    )
    best_parameters = problem.to_external(best["internal"])
    warnings: list[str] = []
    if not diagnostics["identifiable"]:
        warnings.append(
            "parameter set is not identifiable from these data "
            f"(rank {diagnostics['rank']} of {diagnostics['n_parameters']}, "
            f"condition {diagnostics['condition_number']}); the fitted combination is "
            "constrained but individual values are not"
        )
    at_bound = [
        p.name
        for i, p in enumerate(problem.parameters)
        if p.at_bound(best_parameters[p.name])
    ]
    if at_bound:
        warnings.append(f"parameters at a bound: {at_bound}")
    reduced = diagnostics["reduced_chi2"]
    if reduced is not None and reduced > 100.0:
        warnings.append(
            f"residuals exceed the assumed noise level: reduced chi2 = {reduced:.3g}"
        )

    return ConvergenceRun(
        problem=problem,
        options=options,
        seed_control=seed_control,
        best_internal=best["internal"],
        best_parameters=best_parameters,
        best_loss=float(best["components"]["total"]),
        best_components=dict(best["components"]),
        status=best["status"],
        stop_reason=best["reason"],
        start_index=int(best["index"]),
        starts=starts,
        iterations=iterations,
        diagnostics=diagnostics,
        warnings=warnings,
        provenance=dict(provenance or {}),
        n_forward_evals=int(counter["n"]),
        elapsed_s=time.perf_counter() - started,
    )


__all__ = [
    "SCHEMA_CONVERGENCE",
    "SCHEMA_SUMMARY",
    "Parameter",
    "Tikhonov",
    "InverseProblem",
    "OptimizerOptions",
    "SeedControl",
    "ConvergenceRun",
    "run_inversion",
    "fd_jacobian",
    "identifiability_diagnostics",
    "canonical_json",
    "sha256_of",
]