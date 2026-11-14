"""Observed-order-of-accuracy and Roache GCI evidence for refinement studies.

This module is deliberately solver-agnostic and *evidence-only*: it never
invents a fine-mesh result.  It consumes a sequence of scalar quantities of
interest (QoI) measured on successively refined meshes and returns

* the observed order of accuracy ``p`` of every consecutive mesh triple,
* the Richardson extrapolated value of the finest level,
* the Roache Grid Convergence Index (GCI) error band of the finest pair,
* an optional comparison against an independent reference (analytical or
  published value) supplied by the caller.

Nothing here proves a solver is *physically* correct; it quantifies how fast
the discretisation error decays and how large the remaining error band is.
A study that cannot demonstrate a monotone, constant-ratio refinement sequence
is reported as ``blocked`` -- never extrapolated.

References
----------
Roache, P.J. (1994) "Perspective: a method for uniform reporting of grid
refinement studies", *J. Fluids Eng.* 116, 405-413.
Celik, I.B. et al. (2008) "Procedure for estimation and reporting of
uncertainty due to discretization", *J. Fluids Eng.* 130, 078001.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

SCHEMA = "tensorfem.convergence-verification/1.0"

__all__ = [
    "SCHEMA",
    "ConvergenceLevel",
    "refinement_ratio",
    "richardson_observed_order",
    "observed_order_from_errors",
    "richardson_extrapolate",
    "grid_convergence_index",
    "evaluate_convergence_study",
]


@dataclass(frozen=True)
class ConvergenceLevel:
    """One refinement level: representative mesh size ``h`` and QoI value."""

    mesh_size: float
    value: float
    label: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(self.mesh_size) or self.mesh_size <= 0.0:
            raise ValueError("mesh_size must be finite and positive")
        if not math.isfinite(self.value):
            raise ValueError("value must be finite")

    def as_dict(self) -> dict[str, object]:
        return {"label": self.label, "mesh_size": self.mesh_size, "value": self.value}


def _coerce(level: "ConvergenceLevel | Mapping[str, object] | Sequence[object]") -> ConvergenceLevel:
    if isinstance(level, ConvergenceLevel):
        return level
    if isinstance(level, Mapping):
        return ConvergenceLevel(
            float(level["mesh_size"]), float(level["value"]), str(level.get("label", ""))
        )
    mesh_size, value, *rest = level  # type: ignore[misc]
    return ConvergenceLevel(float(mesh_size), float(value), str(rest[0]) if rest else "")


def refinement_ratio(coarse: ConvergenceLevel, fine: ConvergenceLevel) -> float:
    """Mesh-size ratio ``r = h_coarse / h_fine`` (``r > 1`` for refinement)."""
    ratio = coarse.mesh_size / fine.mesh_size
    if not math.isfinite(ratio) or ratio <= 1.0 + 1e-12:
        raise ValueError(
            f"levels must be ordered coarse -> fine with decreasing mesh size, got "
            f"h={coarse.mesh_size:g} then h={fine.mesh_size:g}"
        )
    return ratio


def richardson_observed_order(
    coarse: ConvergenceLevel, mid: ConvergenceLevel, fine: ConvergenceLevel
) -> float:
    """Observed order of accuracy from three levels at a constant ratio ``r``.

    ``p = ln|(f_fine - f_mid) / (f_mid - f_coarse)| / ln r``

    Raises ``ValueError`` when the differences vanish (QoI converged to machine
    precision) or have the same sign (non-monotone / diverging sequence), so a
    caller can never mistake noise for a measured order.
    """
    r1 = refinement_ratio(coarse, mid)
    r2 = refinement_ratio(mid, fine)
    if abs(r1 - r2) > 1e-6 * max(r1, r2):
        raise ValueError(
            f"constant refinement ratio required, got r1={r1:.6g} and r2={r2:.6g}"
        )
    d_coarse = mid.value - coarse.value
    d_fine = fine.value - mid.value
    scale = max(abs(coarse.value), abs(mid.value), abs(fine.value), 1e-300)
    if abs(d_coarse) <= 1e-12 * scale and abs(d_fine) <= 1e-12 * scale:
        raise ValueError("QoI is identical on all three levels (machine-precision convergence)")
    if d_coarse * d_fine <= 0.0:
        raise ValueError(
            f"non-monotone convergence: |df| = {abs(d_coarse):.6g} then {abs(d_fine):.6g}"
        )
    return math.log(abs(d_coarse) / abs(d_fine)) / math.log(r1)


def observed_order_from_errors(
    errors: Sequence[float], ratios: Sequence[float]
) -> list[float]:
    """Observed order from a sequence of errors with known refinement ratios.

    ``p_ij = ln(e_i / e_j) / ln(r_ij)`` for every consecutive pair.  This is the
    right estimator when the QoI is the discretisation error itself (its exact
    value is zero, so three-level Richardson on the QoI is not available).
    """
    if len(errors) < 2:
        raise ValueError("at least two error levels are required")
    if len(ratios) != len(errors) - 1:
        raise ValueError("ratios must contain len(errors)-1 entries")
    orders: list[float] = []
    for coarse, fine, ratio in zip(errors, errors[1:], ratios):
        if not (math.isfinite(coarse) and math.isfinite(fine)) or coarse <= 0.0 or fine <= 0.0:
            raise ValueError("errors must be finite and positive")
        if not math.isfinite(ratio) or ratio <= 1.0:
            raise ValueError("ratios must be finite and greater than one")
        orders.append(math.log(coarse / fine) / math.log(ratio))
    return orders


def richardson_extrapolate(
    coarse: ConvergenceLevel, fine: ConvergenceLevel, order: float, ratio: float | None = None
) -> float:
    """Richardson extrapolation to ``h -> 0`` from the two finest levels."""
    r = refinement_ratio(coarse, fine) if ratio is None else float(ratio)
    if not math.isfinite(order) or order <= 0.0:
        raise ValueError("order of accuracy must be finite and positive")
    denominator = r**order - 1.0
    if denominator <= 0.0:
        raise ValueError(f"degenerate refinement ratio r={r:g} for order p={order:g}")
    return fine.value + (fine.value - coarse.value) / denominator


def grid_convergence_index(
    coarse: ConvergenceLevel,
    fine: ConvergenceLevel,
    order: float,
    *,
    ratio: float | None = None,
    safety_factor: float = 1.25,
) -> dict[str, float]:
    """Roache GCI of the ``coarse -> fine`` pair, relative to the fine value.

    ``GCI_fine = Fs * |e_relative| / (r^p - 1)`` with the standard safety
    factor ``Fs = 1.25`` when the observed order is used (3+ levels).
    """
    if not math.isfinite(safety_factor) or safety_factor <= 0.0:
        raise ValueError("safety_factor must be finite and positive")
    r = refinement_ratio(coarse, fine) if ratio is None else float(ratio)
    denominator = r**order - 1.0
    if denominator <= 0.0:
        raise ValueError(f"degenerate refinement ratio r={r:g} for order p={order:g}")
    absolute_change = abs(fine.value - coarse.value)
    reference = abs(fine.value) if fine.value != 0.0 else 1.0
    relative_gci = safety_factor * (absolute_change / reference) / denominator
    extrapolated = richardson_extrapolate(coarse, fine, order, r)
    return {
        "refinement_ratio": r,
        "observed_order": order,
        "safety_factor": safety_factor,
        "relative_gci": relative_gci,
        "absolute_gci": safety_factor * absolute_change / denominator,
        "richardson_value": extrapolated,
        "band_lower": fine.value - safety_factor * absolute_change / denominator,
        "band_upper": fine.value + safety_factor * absolute_change / denominator,
    }


def evaluate_convergence_study(
    levels: Iterable["ConvergenceLevel | Mapping[str, object] | Sequence[object]"],
    *,
    qoi: str = "quantity",
    expected_order: float | None = None,
    order_band: float = 0.15,
    safety_factor: float = 1.25,
    reference_value: float | None = None,
    reference_tolerance: float = 0.03,
    reference_label: str = "",
) -> dict[str, object]:
    """Evaluate an h-refinement study and return a machine-readable report.

    ``expected_order`` is the theoretical order the discretisation should
    attain (e.g. 2 for a linear HEX8 in the L2 displacement norm).  When given,
    the finest observed order must lie within ``order_band`` (relative) of it.

    ``reference_value`` is an *independent* reference (analytical solution or
    published value).  Its comparison is reported separately from the
    self-consistency checks so that a converged-but-wrong result is still
    flagged as failed.

    Status values: ``passed`` / ``failed`` (data complete, a check failed) /
    ``blocked`` (the sequence cannot be interpreted, e.g. non-monotone).
    """
    study = [_coerce(level) for level in levels]
    if len(study) < 3:
        raise ValueError("at least three refinement levels are required")
    if not math.isfinite(order_band) or order_band <= 0.0:
        raise ValueError("order_band must be finite and positive")
    ratios = [refinement_ratio(a, b) for a, b in zip(study, study[1:])]
    r_mean = sum(ratios) / len(ratios)
    if any(abs(r - r_mean) > 1e-6 * r_mean for r in ratios):
        raise ValueError(f"constant refinement ratio required, got {ratios}")
    if any(b.mesh_size >= a.mesh_size for a, b in zip(study, study[1:])):
        raise ValueError("mesh sizes must decrease strictly")

    report: dict[str, object] = {
        "schema": SCHEMA,
        "qoi": qoi,
        "levels": [level.as_dict() for level in study],
        "refinement_ratios": ratios,
        "checks": [],
        "unevaluable_checks": [],
    }
    checks: list[dict[str, object]] = report["checks"]  # type: ignore[assignment]
    unevaluable: list[dict[str, object]] = report["unevaluable_checks"]  # type: ignore[assignment]

    def check(name: str, passed: bool, detail: str) -> bool:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})
        return bool(passed)

    def record_unevaluable(name: str, detail: str) -> None:
        """A check the ladder cannot answer: neither a pass nor a violation."""
        unevaluable.append({"name": name, "detail": detail})

    # Machine-precision convergence: all levels identical.  No order exists.
    values = [level.value for level in study]
    span = max(values) - min(values)
    scale = max(abs(v) for v in values) or 1.0
    if span <= 1e-12 * scale:
        report["mean_observed_order"] = None
        report["finest_order"] = None
        report["richardson_value"] = study[-1].value
        check(
            "machine_precision_convergence",
            True,
            f"all levels agree to {span / scale:.3g} relative; no order can be measured",
        )
        reference = _reference_block(
            study, study[-1].value, reference_value, reference_tolerance, reference_label
        )
        report["reference"] = reference
        if reference is not None:
            check("reference_tolerance", bool(reference["within_tolerance"]),
                  reference["detail"])
        report["status"] = "failed" if any(not c["passed"] for c in checks) else "passed"
        report["passed"] = report["status"] == "passed"
        return report

    orders: list[float] = []
    order_failures: list[str] = []
    for i in range(len(study) - 2):
        try:
            orders.append(richardson_observed_order(study[i], study[i + 1], study[i + 2]))
        except ValueError as exc:  # non-monotone / no signal
            order_failures.append(str(exc))
    report["observed_orders"] = orders

    if not orders:
        report["status"] = "blocked"
        report["passed"] = False
        report["blocked_reason"] = order_failures[0] if order_failures else "no measurable order"
        check("measured_order", False, report["blocked_reason"])
        return report

    mean_order = sum(orders) / len(orders)
    finest_order = orders[-1]
    report["mean_observed_order"] = mean_order
    report["finest_order"] = finest_order
    check("measured_order", True,
          "observed order per triple: " + ", ".join(f"{p:.3f}" for p in orders))

    drift = abs(orders[-1] - orders[0]) / abs(orders[-1]) if orders[-1] else math.inf
    check("asymptotic_range_drift", drift <= order_band,
          f"order drift between first and last triple = {drift:.3f} "
          f"(band {order_band:.3f})")

    if expected_order is not None:
        if not math.isfinite(expected_order) or expected_order <= 0.0:
            raise ValueError("expected_order must be finite and positive")
        deviation = abs(finest_order - expected_order) / expected_order
        check("expected_order", deviation <= order_band,
              f"finest observed order {finest_order:.3f} vs theoretical "
              f"{expected_order:.3f} (rel. deviation {deviation:.3f}, band {order_band:.3f})")
        report["expected_order"] = expected_order

    # A non-positive observed order means the *increments* of the QoI grow (or
    # stay constant) under refinement: the ladder is not in the asymptotic
    # range, so ``r**p - 1 <= 0`` and neither a GCI band nor a Richardson limit
    # is defined.  That is an attribute of the data, not an error in the
    # caller, so it is reported as ``blocked`` and never raised out of the
    # estimator (a raising estimator cannot be used to grade a stored anchor
    # ladder, and callers would have to guess the verdict).
    extrapolable = math.isfinite(finest_order) and finest_order > 0.0
    gci: dict[str, float] | None
    if extrapolable:
        gci = grid_convergence_index(
            study[-2], study[-1], finest_order, ratio=ratios[-1], safety_factor=safety_factor
        )
        report["gci"] = gci
        report["richardson_value"] = gci["richardson_value"]
    else:
        gci = None
        report["gci"] = None
        report["richardson_value"] = None
        report["blocked_reason"] = (
            f"non-positive observed order p={finest_order:.4g} on the finest triple: the "
            f"increments of the QoI do not shrink under refinement, so no GCI band and no "
            f"Richardson limit exist (pre-asymptotic / non-converging ladder)"
        )
        check("asymptotic_order", False, report["blocked_reason"])

    # Roache's band must shrink with refinement: GCI_fine <= GCI_coarse.  The
    # contract is only defined for pairs whose *own* order is positive; a
    # coarser pair with a non-positive order is recorded as unevaluable rather
    # than compared (``r**p - 1 <= 0`` is degenerate) or silently passed.
    coarser_gci: dict[str, float] | None = None
    if not extrapolable:
        record_unevaluable(
            "gci_band_contracts", "no GCI band on the finest pair (see blocked_reason)"
        )
    elif len(study) >= 3 and len(orders) >= 2 and orders[-2] > 0.0:
        coarser_gci = grid_convergence_index(
            study[-3], study[-2], orders[-2], ratio=ratios[-2], safety_factor=safety_factor
        )
    if coarser_gci is None:
        if extrapolable:
            check("gci_band_contracts", True,
                  "single evaluable pair, band contract not applicable")
    else:
        report["gci_coarser_pair"] = coarser_gci
        check(
            "gci_band_contracts",
            gci["absolute_gci"] <= coarser_gci["absolute_gci"] * (1.0 + 1e-9),
            f"absolute GCI {gci['absolute_gci']:.3e} (fine pair) <= "
            f"{coarser_gci['absolute_gci']:.3e} (coarser pair)",
        )
    if extrapolable and coarser_gci is None and len(study) >= 3 and len(orders) >= 2:
        record_unevaluable(
            "gci_band_contracts",
            f"coarser triple order p={orders[-2]:.4g} is non-positive; band contract "
            f"not evaluable",
        )

    reference = _reference_block(
        study,
        gci["richardson_value"] if gci is not None else study[-1].value,
        reference_value,
        reference_tolerance,
        reference_label,
        gci=gci,
        extrapolation_available=extrapolable,
    )
    report["reference"] = reference
    if reference is not None:
        check("reference_tolerance", bool(reference["within_tolerance"]), reference["detail"])
        check(
            "finest_error_inside_gci_band",
            bool(reference["finest_inside_gci"]),
            reference["gci_detail"],
        )

    if not extrapolable:
        report["status"] = "blocked"
    else:
        report["status"] = "failed" if any(not c["passed"] for c in checks) else "passed"
    report["passed"] = report["status"] == "passed"
    return report


def _relative_difference(value: float, reference: float) -> float:
    if reference == 0.0:
        return abs(value)
    return abs(value - reference) / abs(reference)


def _reference_block(
    study: Sequence[ConvergenceLevel],
    extrapolated: float,
    reference_value: float | None,
    tolerance: float,
    label: str,
    *,
    gci: Mapping[str, float] | None = None,
    extrapolation_available: bool = True,
) -> dict[str, object] | None:
    """Compare the extrapolated/finest values against an independent reference.

    ``extrapolation_available=False`` records that no Richardson limit exists
    for this ladder: the block then carries only the finest-level comparison,
    and ``extrapolated_value`` / ``extrapolated_relative_error`` stay ``None``
    so no caller can read a finest-mesh number as a certified limit.
    """
    if reference_value is None:
        return None
    if not math.isfinite(reference_value):
        raise ValueError("reference_value must be finite")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("reference_tolerance must be finite and positive")
    finest = study[-1]
    name = label or "reference"
    finest_error = _relative_difference(finest.value, reference_value)
    block: dict[str, object] = {
        "label": name,
        "value": reference_value,
        "extrapolated_value": None,
        "extrapolated_relative_error": None,
        "finest_relative_error": finest_error,
        "tolerance": tolerance,
        "detail": "",
    }
    if not extrapolation_available:
        block["within_tolerance"] = False
        block["detail"] = (
            f"no Richardson limit (ladder pre-asymptotic); finest level "
            f"{finest.value:.12g} vs {name} {reference_value:.12g}: rel. error "
            f"{finest_error:.3e} (> {tolerance:.3e} at best)"
        )
        block["extrapolated_value"] = None
    else:
        extrapolated_error = _relative_difference(extrapolated, reference_value)
        block["extrapolated_value"] = extrapolated
        block["extrapolated_relative_error"] = extrapolated_error
        block["within_tolerance"] = extrapolated_error <= tolerance
        block["detail"] = (
            f"Richardson {extrapolated:.12g} vs {name} {reference_value:.12g}: "
            f"rel. error {extrapolated_error:.3e} (<= {tolerance:.3e}); "
            f"finest-level error {finest_error:.3e}"
        )
    if gci is None:
        block["finest_inside_gci"] = True
        block["gci_detail"] = "no GCI band available (machine-precision levels)"
        return block
    band = float(gci["relative_gci"])
    inside = finest_error <= band * (1.0 + 1e-12)
    block["finest_inside_gci"] = inside
    block["relative_gci"] = band
    block["gci_detail"] = (
        f"finest-mesh true error {finest_error:.3e} vs GCI band {band:.3e} -> "
        f"{'inside' if inside else 'outside'} the band"
    )
    return block
