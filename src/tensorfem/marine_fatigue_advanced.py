"""Advanced structural-fatigue assessment primitives (SI units).

The routines in this module are deterministic TensorFEM-side verification tools.
They are not a substitute for a class-rule fatigue assessment or certification.
No TensorLBM or CFD code is imported or executed.
"""
from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .marine_fatigue_qualification import SNLogLogCurve

QUALIFICATION_TOLERANCE = 0.03


def _real(value: float, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real scalar")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0.0):
        condition = "finite and > 0" if positive else "finite"
        raise ValueError(f"{name} must be {condition}")
    return value


@dataclass(frozen=True)
class PathStressSample:
    """A surface-path node, measured from the weld toe (m, Pa)."""

    distance_m: float
    stress_pa: float

    def __post_init__(self) -> None:
        _real(self.distance_m, "distance_m")
        _real(self.stress_pa, "stress_pa")
        if self.distance_m < 0.0:
            raise ValueError("distance_m must be >= 0")


def _linear_path_value(samples: Sequence[PathStressSample], location: float) -> float:
    for left, right in itertools.pairwise(samples):
        if left.distance_m <= location <= right.distance_m:
            fraction = (location - left.distance_m) / (right.distance_m - left.distance_m)
            return left.stress_pa + fraction * (right.stress_pa - left.stress_pa)
    raise ValueError("path does not bracket every required sampling location")


def hot_spot_stress_from_path_pa(
    samples: Sequence[PathStressSample], thickness_m: float, *, method: str = "linear"
) -> float:
    """Interpolate nodal path stresses and extrapolate them to the weld toe.

    ``linear`` samples at 0.4t and 1.0t. ``quadratic`` samples at 0.4t,
    0.9t and 1.4t. Path nodes must be strictly increasing. Stresses may be
    signed because a membrane/bending component can cross zero.
    """
    thickness = _real(thickness_m, "thickness_m", positive=True)
    factors = {"linear": (0.4, 1.0), "quadratic": (0.4, 0.9, 1.4)}
    if method not in factors:
        raise ValueError("method must be 'linear' or 'quadratic'")
    rows = tuple(samples)
    if len(rows) < 2 or not all(isinstance(row, PathStressSample) for row in rows):
        raise TypeError("samples must contain at least two PathStressSample values")
    distances = [row.distance_m for row in rows]
    if any(b <= a for a, b in itertools.pairwise(distances)):
        raise ValueError("path distances must be strictly increasing")
    locations = tuple(factor * thickness for factor in factors[method])
    stresses = tuple(_linear_path_value(rows, x) for x in locations)
    result = 0.0
    for i, yi in enumerate(stresses):
        weight = 1.0
        for j, xj in enumerate(locations):
            if i != j:
                weight *= -xj / (locations[i] - xj)
        result += weight * yi
    if not math.isfinite(result):
        raise ValueError("extrapolated hot-spot stress is not finite")
    return result


@dataclass(frozen=True)
class RainflowCycle:
    range_pa: float
    mean_pa: float
    count: float

    def __post_init__(self) -> None:
        _real(self.range_pa, "range_pa", positive=True)
        _real(self.mean_pa, "mean_pa")
        _real(self.count, "count", positive=True)
        if self.count not in (0.5, 1.0):
            raise ValueError("count must be 0.5 or 1.0")


def _reversals(history: Sequence[float]) -> tuple[float, ...]:
    unique: list[float] = []
    for value in history:
        value = _real(value, "stress_history_pa value")
        if not unique or value != unique[-1]:
            unique.append(value)
    if len(unique) < 2:
        raise ValueError("stress history must contain at least two distinct values")
    result = [unique[0]]
    for previous, current, following in zip(unique, unique[1:], unique[2:]):
        if (current - previous) * (following - current) < 0.0:
            result.append(current)
    result.append(unique[-1])
    return tuple(result)


def rainflow_cycles(stress_history_pa: Sequence[float]) -> tuple[RainflowCycle, ...]:
    """Count cycles with the ASTM E1049 four-point stack algorithm.

    The returned records preserve full cycles (count 1.0) and open-history
    residual half cycles (count 0.5); zero-range cycles are discarded.
    """
    reversals = _reversals(stress_history_pa)
    stack: list[float] = []
    cycles: list[RainflowCycle] = []
    for reversal in reversals:
        stack.append(reversal)
        while len(stack) >= 3:
            older_range = abs(stack[-2] - stack[-3])
            newer_range = abs(stack[-1] - stack[-2])
            if older_range > newer_range:
                break
            mean = 0.5 * (stack[-3] + stack[-2])
            if older_range > 0.0:
                if len(stack) == 3:
                    cycles.append(RainflowCycle(older_range, mean, 0.5))
                    stack.pop(0)
                else:
                    cycles.append(RainflowCycle(older_range, mean, 1.0))
                    del stack[-3:-1]
            else:
                del stack[-3:-1]
    for first, second in itertools.pairwise(stack):
        cycle_range = abs(second - first)
        if cycle_range > 0.0:
            cycles.append(RainflowCycle(cycle_range, 0.5 * (first + second), 0.5))
    return tuple(cycles)


@dataclass(frozen=True)
class SpectrumDamageResult:
    damage: float
    cycle_damage: tuple[float, ...]
    passed: bool


def spectrum_miner_damage(
    cycles: Iterable[RainflowCycle], curve: SNLogLogCurve
) -> SpectrumDamageResult:
    """Accumulate variable-amplitude rainflow cycles using Miner's rule."""
    if not isinstance(curve, SNLogLogCurve):
        raise TypeError("curve must be SNLogLogCurve")
    rows = tuple(cycles)
    if not rows:
        raise ValueError("at least one rainflow cycle is required")
    if not all(isinstance(row, RainflowCycle) for row in rows):
        raise TypeError("cycles must contain RainflowCycle values")
    for row in rows:
        _real(row.range_pa, "range_pa", positive=True)
        _real(row.mean_pa, "mean_pa")
        if row.count not in (0.5, 1.0):
            raise ValueError("rainflow count must be 0.5 or 1.0")
    parts = tuple(row.count / curve.cycles_to_failure(row.range_pa) for row in rows)
    damage = math.fsum(parts)
    return SpectrumDamageResult(damage, parts, damage <= 1.0)


@dataclass(frozen=True)
class ParisLaw:
    """Paris relation da/dN = C (delta K)^m in SI units."""

    coefficient_c: float
    exponent_m: float
    geometry_factor: float = 1.0

    def __post_init__(self) -> None:
        _real(self.coefficient_c, "coefficient_c", positive=True)
        _real(self.exponent_m, "exponent_m", positive=True)
        _real(self.geometry_factor, "geometry_factor", positive=True)


@dataclass(frozen=True)
class CrackGrowthBlock:
    stress_range_pa: float
    cycles: float

    def __post_init__(self) -> None:
        _real(self.stress_range_pa, "stress_range_pa", positive=True)
        _real(self.cycles, "cycles", positive=True)


@dataclass(frozen=True)
class CrackGrowthResult:
    final_crack_m: float
    crack_history_m: tuple[float, ...]
    reached_critical: bool


def paris_crack_growth(
    initial_crack_m: float,
    blocks: Iterable[CrackGrowthBlock],
    law: ParisLaw,
    *,
    critical_crack_m: float | None = None,
) -> CrackGrowthResult:
    """Integrate constant-geometry Paris growth analytically over load blocks."""
    crack = _real(initial_crack_m, "initial_crack_m", positive=True)
    if not isinstance(law, ParisLaw):
        raise TypeError("law must be ParisLaw")
    rows = tuple(blocks)
    if not rows or not all(isinstance(row, CrackGrowthBlock) for row in rows):
        raise TypeError("blocks must contain at least one CrackGrowthBlock")
    critical = None
    if critical_crack_m is not None:
        critical = _real(critical_crack_m, "critical_crack_m", positive=True)
        if critical <= crack:
            raise ValueError("critical_crack_m must exceed initial_crack_m")
    history = [crack]
    power = 0.5 * law.exponent_m
    for row in rows:
        scale = law.coefficient_c * (
            law.geometry_factor * row.stress_range_pa * math.sqrt(math.pi)
        ) ** law.exponent_m
        if math.isclose(power, 1.0, rel_tol=0.0, abs_tol=1e-14):
            crack *= math.exp(scale * row.cycles)
        else:
            transformed = crack ** (1.0 - power) + (1.0 - power) * scale * row.cycles
            if transformed <= 0.0:
                crack = math.inf
            else:
                crack = transformed ** (1.0 / (1.0 - power))
        history.append(crack)
        if critical is not None and crack >= critical:
            return CrackGrowthResult(crack, tuple(history), True)
    if not math.isfinite(crack):
        raise ValueError("Paris-law crack growth became unbounded")
    return CrackGrowthResult(crack, tuple(history), False)


def run_advanced_fatigue_qualification() -> dict[str, object]:
    """Run independent analytic/hand-calculation oracles with a strict <3% gate."""
    path = tuple(
        PathStressSample(x, 150e6 - 4e9 * x)
        for x in (0.0, 0.003, 0.007, 0.012, 0.016)
    )
    cycles = rainflow_cycles((0.0, 100e6, 0.0))
    curve = SNLogLogCurve(100e6, 2e6, 3.0)
    paris = ParisLaw(1.0e-24, 2.0, 1.0)
    growth = paris_crack_growth(
        0.01, (CrackGrowthBlock(100e6, 1000.0),), paris
    ).final_crack_m
    expected_growth = 0.01 * math.exp(1.0e-24 * math.pi * (100e6) ** 2 * 1000.0)
    cases = (
        ("hot_spot.path_linear", hot_spot_stress_from_path_pa(path, 0.01), 150e6),
        ("rainflow.triangle_count", sum(row.count for row in cycles), 1.0),
        ("miner.variable_amplitude", spectrum_miner_damage(cycles, curve).damage, 0.5e-6),
        ("paris.m2_closed_form", growth, expected_growth),
    )
    evidence = []
    for case_id, actual, oracle in cases:
        error = abs(actual - oracle) / abs(oracle)
        evidence.append({
            "id": case_id, "actual": actual, "oracle": oracle,
            "relative_error": error, "tolerance": QUALIFICATION_TOLERANCE,
            "passed": error < QUALIFICATION_TOLERANCE,
        })
    return {
        "scope": "TensorFEM structural fatigue only; no TensorLBM or CFD execution",
        "certification": "Verification evidence only; not class or regulatory certification",
        "evidence": evidence,
        "passed": all(row["passed"] for row in evidence),
    }
