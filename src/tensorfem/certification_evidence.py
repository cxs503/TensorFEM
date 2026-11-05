"""Uncertainty-aware evidence helpers for the structural certification dossier.

The module holds the *decision* logic that turns solver output into a
certification table, so the numbers and the verdicts cannot drift apart:

* :func:`classify_pointwise` turns ``(finite element value, reference value)``
  into a row carrying the numerator ``|fe - ref|``, the denominator ``|ref|``,
  the relative error and a three-state verdict ``PASS`` / ``FAIL`` /
  ``WITHHELD``.  A row is ``WITHHELD`` when the denominator is so small that a
  relative-error criterion is ill-conditioned (the pathology reported for the
  near-wall Poisson probes of the elliptic-tube benchmark); such a row is never
  silently reported as a pass.
* :func:`grid_uncertainty` is a Roache grid-convergence-index estimator over an
  h-refinement ladder.  Its preconditions (monotone error decay, positive
  observed order) are checked; when they fail the estimate is *withheld* and the
  min/max envelope of the ladder is returned instead of an invented band.
* :func:`envelope_uncertainty` and :func:`combine_envelopes` build the
  conservative, non-statistical uncertainty envelope used for the error bars.
* :func:`aggregate_verdicts` collapses row verdicts into one section state.

Nothing here is a statistical confidence interval: the envelope combination is a
linear (worst-case) sum with an explicit method label so a report cannot mistake
it for a 95 % interval.  No numbers are hard-coded; every quantity is computed
from the values handed in.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

__all__ = [
    "Criterion",
    "classify_pointwise",
    "envelope_uncertainty",
    "grid_uncertainty",
    "combine_envelopes",
    "aggregate_verdicts",
    "within_tolerance",
]

PASS = "PASS"
FAIL = "FAIL"
WITHHELD = "WITHHELD"


@dataclass(frozen=True)
class Criterion:
    """Acceptance criterion for a relative-error comparison.

    ``relative_tolerance`` is the acceptance threshold (``0.03`` for the 3 %
    gate).  ``denominator_floor`` is the smallest ``|reference|``, expressed as a
    fraction of the problem's stress scale, at which a relative-error statement
    is considered well conditioned; below it the row is withheld.
    """

    relative_tolerance: float = 0.03
    denominator_floor: float = 0.01

    def __post_init__(self) -> None:
        if not 0.0 < self.relative_tolerance < 1.0:
            raise ValueError("relative_tolerance must lie in (0, 1)")
        if not 0.0 <= self.denominator_floor < 1.0:
            raise ValueError("denominator_floor must lie in [0, 1)")

    def as_dict(self) -> dict[str, float]:
        return {
            "relative_tolerance": self.relative_tolerance,
            "denominator_floor_fraction_of_scale": self.denominator_floor,
        }


def within_tolerance(value: float, reference: float, tolerance: float = 0.03) -> bool:
    """``|value - reference| <= tolerance * |reference|`` with a nonzero reference."""
    if reference == 0.0:
        raise ValueError("relative comparison against a zero reference is undefined")
    return abs(value - reference) <= tolerance * abs(reference)


def classify_pointwise(
    component: str,
    value: float,
    reference: float,
    *,
    scale: float,
    criterion: Criterion,
    label: str = "",
) -> dict[str, object]:
    """Build one row of the pointwise comparison table.

    ``scale`` is the problem stress scale (here the far-field tension) against
    which the denominator floor is measured.
    """
    numerator = abs(float(value) - float(reference))
    denominator = abs(float(reference))
    row: dict[str, object] = {
        "label": label,
        "component": component,
        "fe": float(value),
        "reference": float(reference),
        "numerator_abs_error": numerator,
        "denominator_abs_reference": denominator,
        "threshold_relative": criterion.relative_tolerance,
    }
    floor = criterion.denominator_floor * abs(float(scale))
    row["denominator_floor"] = floor
    if denominator < floor:
        row["relative_error"] = None
        row["verdict"] = WITHHELD
        row["withheld_reason"] = (
            f"|reference|={denominator:.3e} is below {criterion.denominator_floor:.0%} of the "
            f"stress scale {abs(float(scale)):.3e}: the relative criterion is ill-conditioned"
        )
        return row
    relative = numerator / denominator
    row["relative_error"] = relative
    row["verdict"] = PASS if relative <= criterion.relative_tolerance else FAIL
    row["withheld_reason"] = None
    return row


def envelope_uncertainty(values: Sequence[float], *, label: str = "") -> dict[str, object]:
    """Min/max envelope of a set of exchangeable estimates of one quantity."""
    data = [float(v) for v in values]
    if not data:
        raise ValueError("envelope needs at least one value")
    if not all(math.isfinite(v) for v in data):
        raise ValueError("envelope values must be finite")
    low, high = min(data), max(data)
    centre = 0.5 * (low + high)
    half_width = 0.5 * (high - low)
    relative = half_width / abs(centre) if centre != 0.0 else float("nan")
    return {
        "label": label,
        "method": "min/max envelope (non-statistical)",
        "count": len(data),
        "min": low,
        "max": high,
        "centre": centre,
        "half_width": half_width,
        "relative_half_width": relative,
    }


def grid_uncertainty(
    ladder: Sequence[Mapping[str, object]],
    *,
    reference: float,
    safety_factor: float = 1.25,
    label: str = "",
) -> dict[str, object]:
    """Roache GCI of an h-refinement ladder, with explicit precondition checks.

    ``ladder`` is a sequence of ``{"level": <refinement level>, "value": <qoi>}``
    ordered coarse to fine.  The observed order is taken from the error sequence
    ``|value - reference|`` (a verification study has a known exact value), and
    the GCI uses the standard ``Fs * |v_f - v_c| / (r**p - 1)``.

    When the ladder does not show monotone error decay with a positive observed
    order the estimator's precondition is violated: the result is ``WITHHELD``
    and the min/max envelope of the ladder is returned instead.
    """
    levels = [float(item["level"]) for item in ladder]
    values = [float(item["value"]) for item in ladder]
    if len(values) < 3:
        raise ValueError("a GCI study needs at least three levels")
    if not all(math.isfinite(v) for v in values):
        raise ValueError("ladder values must be finite")
    errors = [abs(v - float(reference)) for v in values]
    ratios = [levels[i + 1] / levels[i] for i in range(len(levels) - 1)]
    if any(r <= 1.0 for r in ratios):
        raise ValueError("refinement levels must increase")

    orders: list[float] = []
    for i, ratio in enumerate(ratios):
        coarse, fine = errors[i], errors[i + 1]
        if coarse <= 0.0 or fine <= 0.0 or fine >= coarse:
            orders.append(float("nan"))
        else:
            orders.append(math.log(coarse / fine) / math.log(ratio))
    monotone = all(errors[i + 1] < errors[i] for i in range(len(errors) - 1))

    envelope = envelope_uncertainty(values, label=label)
    base: dict[str, object] = {
        "label": label,
        "levels": levels,
        "values": values,
        "errors": errors,
        "refinement_ratios": ratios,
        "observed_orders": orders,
        "observed_order": orders[-1],
        "safety_factor": safety_factor,
        "reference": float(reference),
        "monotone_error_decay": monotone,
        "envelope": envelope,
    }
    if not monotone or not math.isfinite(orders[-1]) or orders[-1] <= 0.0:
        base["status"] = WITHHELD
        base["reason"] = (
            "the error sequence |value - reference| = "
            f"{['%.3e' % e for e in errors]} is not monotone decreasing with positive "
            f"observed order ({['%.3f' % o for o in orders]}): the GCI precondition is "
            "violated, so no analytic error band is claimed; the min/max envelope of the "
            "ladder is reported instead"
        )
        base["half_width"] = envelope["half_width"]
        base["relative_half_width"] = envelope["relative_half_width"]
        base["band"] = [envelope["min"], envelope["max"]]
        base["richardson_value"] = None
        base["relative_gci"] = None
        base["band_covers_reference"] = bool(
            envelope["min"] <= reference <= envelope["max"]
        )
        return base

    order = orders[-1]
    ratio = ratios[-1]
    denominator = ratio**order - 1.0
    delta = abs(values[-1] - values[-2])
    gci_abs = safety_factor * delta / denominator
    fine = values[-1]
    richardson = fine + (fine - values[-2]) / denominator
    base["status"] = "ESTIMATED"
    base["reason"] = None
    base["half_width"] = gci_abs
    base["relative_half_width"] = gci_abs / abs(fine) if fine != 0.0 else float("nan")
    base["band"] = [fine - gci_abs, fine + gci_abs]
    base["richardson_value"] = richardson
    base["relative_gci"] = gci_abs / abs(fine) if fine != 0.0 else float("nan")
    base["band_covers_reference"] = bool(fine - gci_abs <= reference <= fine + gci_abs)
    return base


def combine_envelopes(
    contributions: Sequence[Mapping[str, object]], *, label: str = "combined"
) -> dict[str, object]:
    """Linear (worst-case) sum of independent half-widths.

    The result is deliberately *not* a root-sum-square and *not* a statistical
    interval: it is the conservative envelope of the listed contributions.
    """
    widths = []
    for item in contributions:
        width = item.get("half_width")
        if width is None or not math.isfinite(float(width)):
            raise ValueError(f"contribution {item.get('label')!r} has no finite half width")
        widths.append(float(width))
    total = float(sum(widths))
    return {
        "label": label,
        "method": "linear sum of half-widths (worst case, non-statistical)",
        "half_width": total,
        "contributions": [
            {"label": item.get("label"), "half_width": float(item["half_width"])}
            for item in contributions
        ],
    }


def aggregate_verdicts(rows: Iterable[Mapping[str, object]]) -> dict[str, object]:
    """Collapse row verdicts into a section state.

    Any ``FAIL`` makes the section ``FAIL``; withheld rows downgrade a clean
    ``PASS`` to ``PASS_WITH_WITHHELD`` so a report can never present an
    unqualified pass while rows are missing from the criterion.
    """
    counts = {PASS: 0, FAIL: 0, WITHHELD: 0}
    failed: list[str] = []
    withheld: list[str] = []
    for row in rows:
        verdict = row["verdict"]
        if verdict not in counts:
            raise ValueError(f"unknown verdict {verdict!r}")
        counts[verdict] += 1
        if verdict == FAIL:
            failed.append(str(row.get("label", "")))
        elif verdict == WITHHELD:
            withheld.append(str(row.get("label", "")))
    if counts[FAIL]:
        state = FAIL
    elif counts[WITHHELD]:
        state = "PASS_WITH_WITHHELD"
    else:
        state = PASS
    return {
        "state": state,
        "counts": counts,
        "failed_labels": failed,
        "withheld_labels": withheld,
    }