"""Weld hot-spot stress and S-N fatigue qualification (SI units).

This module is deliberately solver-only: it contains no CFD or TensorLBM coupling.
The calculations are engineering assessment primitives, not class approval.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence


QUALIFICATION_TOLERANCE = 0.03


def _positive(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real scalar")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return value


def hot_spot_stress_pa(
    surface_stresses_pa: Sequence[float], distances_m: Sequence[float], *, method: str
) -> float:
    """Extrapolate surface stress to the weld toe at zero distance.

    ``method`` is ``"linear"`` (two samples) or ``"quadratic"`` (three samples).
    Distances are metres and stresses are Pa. Lagrange extrapolation supports any
    distinct, positive sample locations, including the common 0.4t/1.0t and
    0.4t/0.9t/1.4t layouts.
    """
    required = {"linear": 2, "quadratic": 3}
    if method not in required:
        raise ValueError("method must be 'linear' or 'quadratic'")
    if len(surface_stresses_pa) != required[method] or len(distances_m) != required[method]:
        raise ValueError(f"{method} extrapolation requires {required[method]} samples")
    xs = [_positive(x, f"distances_m[{i}]") for i, x in enumerate(distances_m)]
    ys = [_positive(y, f"surface_stresses_pa[{i}]") for i, y in enumerate(surface_stresses_pa)]
    if len(set(xs)) != len(xs):
        raise ValueError("distances_m must be distinct")
    result = 0.0
    for i, yi in enumerate(ys):
        weight = 1.0
        for j, xj in enumerate(xs):
            if i != j:
                weight *= -xj / (xs[i] - xj)
        result += weight * yi
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("extrapolated hot-spot stress must be finite and > 0")
    return result


@dataclass(frozen=True)
class SNLogLogCurve:
    """Single-slope S-N curve defined by one point and log-log slope.

    Stress range is in Pa and cycle count is dimensionless. The relation is
    ``N = N_ref * (stress_ref / stress_range)**m``.
    """

    reference_stress_pa: float
    reference_cycles: float
    slope_m: float

    def __post_init__(self) -> None:
        _positive(self.reference_stress_pa, "reference_stress_pa")
        _positive(self.reference_cycles, "reference_cycles")
        _positive(self.slope_m, "slope_m")

    def cycles_to_failure(self, stress_range_pa: float) -> float:
        stress = _positive(stress_range_pa, "stress_range_pa")
        cycles = self.reference_cycles * (self.reference_stress_pa / stress) ** self.slope_m
        if not math.isfinite(cycles) or cycles <= 0.0:
            raise ValueError("computed fatigue life is outside finite range")
        return cycles


def thickness_corrected_stress_pa(
    stress_range_pa: float, thickness_m: float, *, reference_thickness_m: float, exponent: float
) -> float:
    """Return optional thickness-corrected stress range in Pa.

    No reduction is applied below the reference thickness. The exponent is a
    user-selected assessment parameter; this function does not encode a class rule.
    """
    stress = _positive(stress_range_pa, "stress_range_pa")
    thickness = _positive(thickness_m, "thickness_m")
    reference = _positive(reference_thickness_m, "reference_thickness_m")
    if isinstance(exponent, bool) or not isinstance(exponent, (int, float)):
        raise TypeError("exponent must be a real scalar")
    exponent = float(exponent)
    if not math.isfinite(exponent) or exponent < 0.0:
        raise ValueError("exponent must be finite and >= 0")
    return stress * max(1.0, (thickness / reference) ** exponent)


@dataclass(frozen=True)
class StressBlock:
    stress_range_pa: float
    cycles: float

    def __post_init__(self) -> None:
        _positive(self.stress_range_pa, "stress_range_pa")
        _positive(self.cycles, "cycles")


@dataclass(frozen=True)
class MinerResult:
    damage: float
    block_damage: tuple[float, ...]
    passed: bool


def miner_damage(blocks: Iterable[StressBlock], curve: SNLogLogCurve) -> MinerResult:
    """Calculate Palmgren-Miner damage for multiple constant-amplitude blocks."""
    if not isinstance(curve, SNLogLogCurve):
        raise TypeError("curve must be SNLogLogCurve")
    rows = tuple(blocks)
    if not rows:
        raise ValueError("at least one stress block is required")
    if not all(isinstance(row, StressBlock) for row in rows):
        raise TypeError("blocks must contain StressBlock values")
    parts = tuple(row.cycles / curve.cycles_to_failure(row.stress_range_pa) for row in rows)
    total = math.fsum(parts)
    if not math.isfinite(total):
        raise ValueError("computed Miner damage is outside finite range")
    return MinerResult(damage=total, block_damage=parts, passed=total <= 1.0)


def run_fatigue_qualification() -> dict[str, object]:
    """Run independent hand-calculation oracles with a strict <3% gate."""
    curve = SNLogLogCurve(100e6, 2e6, 3.0)
    # Oracles are evaluated independently before calling production functions.
    cases = (
        (
            "hot_spot.linear",
            hot_spot_stress_pa((132e6, 105e6), (0.004, 0.010), method="linear"),
            150e6,
        ),
        (
            "hot_spot.quadratic",
            hot_spot_stress_pa(
                (121.2e6, 99.2e6, 87.2e6), (0.004, 0.009, 0.014), method="quadratic"
            ),
            146e6,
        ),
        ("sn.single_block", curve.cycles_to_failure(125e6), 1_024_000.0),
        (
            "miner.multi_block",
            miner_damage(
                (StressBlock(100e6, 5e5), StressBlock(125e6, 2.56e5)), curve
            ).damage,
            0.5,
        ),
    )
    evidence = []
    for case_id, actual, expected in cases:
        error = abs(actual - expected) / abs(expected)
        evidence.append({"id": case_id, "actual": actual, "oracle": expected,
                         "relative_error": error, "tolerance": QUALIFICATION_TOLERANCE,
                         "passed": error < QUALIFICATION_TOLERANCE})
    return {
        "scope": "TensorFEM only; no TensorLBM or CFD execution",
        "units": {"stress": "Pa", "distance_and_thickness": "m", "cycles": "1"},
        "certification": "Engineering verification evidence; not class-society certification",
        "evidence": evidence,
        "passed": all(row["passed"] for row in evidence),
    }
