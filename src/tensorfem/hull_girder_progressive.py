"""Auditable component model for hull-girder progressive strength qualification.

The model enforces plane sections and zero resultant axial force.  Each component
is elastic until its tensile yield stress or (optionally lower) compressive
buckling strength is reached.  It is intentionally a section qualification
model, not a class-rule progressive-collapse method.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise


def _positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real scalar")
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return out


@dataclass(frozen=True)
class HullComponent:
    """A plate, stiffener, or other longitudinal area lumped at ``y``."""

    name: str
    area: float
    y: float
    young: float
    yield_stress: float
    compression_strength: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("component name must be a non-empty string")
        object.__setattr__(self, "area", _positive("area", self.area))
        object.__setattr__(self, "young", _positive("young", self.young))
        object.__setattr__(self, "yield_stress", _positive("yield_stress", self.yield_stress))
        if isinstance(self.y, bool) or not isinstance(self.y, (int, float)):
            raise TypeError("y must be a real scalar")
        if not math.isfinite(float(self.y)):
            raise ValueError("y must be finite")
        object.__setattr__(self, "y", float(self.y))
        if self.compression_strength is None:
            object.__setattr__(self, "compression_strength", self.yield_stress)
        else:
            cs = _positive("compression_strength", self.compression_strength)
            if cs > self.yield_stress:
                raise ValueError("compression_strength cannot exceed yield_stress")
            object.__setattr__(self, "compression_strength", cs)


@dataclass(frozen=True)
class ComponentState:
    name: str
    strain: float
    stress: float
    axial_force: float
    moment: float
    mode: str


@dataclass(frozen=True)
class ProgressivePoint:
    curvature: float
    axial_strain: float
    moment: float
    axial_residual: float
    work: float
    states: tuple[ComponentState, ...]


def _stress(component: HullComponent, strain: float) -> tuple[float, str]:
    trial = component.young * strain
    if trial >= component.yield_stress:
        return component.yield_stress, "tensile_yield"
    if trial <= -component.compression_strength:
        mode = (
            "compressive_yield"
            if component.compression_strength == component.yield_stress
            else "buckling_limited"
        )
        return -component.compression_strength, mode
    return trial, "elastic"


def solve_component_section(
    components: tuple[HullComponent, ...] | list[HullComponent],
    curvatures: tuple[float, ...] | list[float],
    *,
    equilibrium_tolerance: float = 1e-11,
) -> tuple[ProgressivePoint, ...]:
    """Trace positive-curvature response while solving axial equilibrium."""
    if not isinstance(components, (tuple, list)) or len(components) < 2:
        raise ValueError("components must contain at least two entries")
    if not all(isinstance(c, HullComponent) for c in components):
        raise TypeError("components must contain HullComponent objects")
    if len({c.name for c in components}) != len(components):
        raise ValueError("component names must be unique")
    if max(c.y for c in components) == min(c.y for c in components):
        raise ValueError("components must span at least two y coordinates")
    if not isinstance(curvatures, (tuple, list)) or len(curvatures) < 2:
        raise ValueError("curvatures must contain at least two values")
    ks = []
    for value in curvatures:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("curvatures must contain real scalars")
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError("curvatures must be finite and non-negative")
        ks.append(value)
    if ks[0] != 0.0 or any(b <= a for a, b in pairwise(ks)):
        raise ValueError("curvatures must start at zero and be strictly increasing")
    tol = _positive("equilibrium_tolerance", equilibrium_tolerance)
    force_scale = sum(c.area * c.yield_stress for c in components)

    def force(e0: float, k: float) -> float:
        return sum(c.area * _stress(c, e0 + k * c.y)[0] for c in components)

    points: list[ProgressivePoint] = []
    work = previous_m = previous_k = 0.0
    for k in ks:
        span = max(abs(k * c.y) + c.yield_stress / c.young for c in components)
        lo, hi = -2.0 * span - 1e-15, 2.0 * span + 1e-15
        while force(lo, k) > 0.0:
            lo *= 2.0
        while force(hi, k) < 0.0:
            hi *= 2.0
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if force(mid, k) > 0.0:
                hi = mid
            else:
                lo = mid
        e0 = 0.5 * (lo + hi)
        states = []
        for c in components:
            strain = e0 + k * c.y
            stress, mode = _stress(c, strain)
            f = c.area * stress
            states.append(ComponentState(c.name, strain, stress, f, f * c.y, mode))
        residual = sum(s.axial_force for s in states)
        moment = sum(s.moment for s in states)
        if points:
            work += 0.5 * (previous_m + moment) * (k - previous_k)
        if abs(residual) > tol * force_scale:
            raise RuntimeError("axial equilibrium did not converge")
        points.append(ProgressivePoint(k, e0, moment, residual, work, tuple(states)))
        previous_k, previous_m = k, moment
    return tuple(points)


def symmetric_component_oracle(components: tuple[HullComponent, ...], curvature: float) -> float:
    """Closed-form oracle for mirrored components with identical pair properties."""
    k = float(curvature)
    if not math.isfinite(k) or k < 0.0:
        raise ValueError("curvature must be finite and non-negative")
    by_y = {c.y: c for c in components}
    for c in components:
        mate = by_y.get(-c.y)
        if mate is None or (
            mate.area,
            mate.young,
            mate.yield_stress,
            mate.compression_strength,
        ) != (c.area, c.young, c.yield_stress, c.compression_strength):
            raise ValueError("oracle requires identical mirrored component pairs")
        if c.compression_strength != c.yield_stress:
            raise ValueError("oracle requires symmetric tensile/compressive strength")
    return sum(c.area * min(c.yield_stress, c.young * k * abs(c.y)) * abs(c.y) for c in components)


def run_progressive_hull_girder_benchmark() -> dict[str, object]:
    """Qualify a multi-component section against a closed-form hand oracle."""
    e, sy = 210e9, 355e6
    components = tuple(
        HullComponent(f"{kind}_{side}", area, sign * y, e, sy)
        for kind, area, y in (
            ("plate", 0.030, 1.20),
            ("longitudinal", 0.012, 1.05),
            ("side", 0.018, 0.55),
            ("inner", 0.014, 0.30),
        )
        for side, sign in (("top", 1.0), ("bottom", -1.0))
    )
    ky = sy / (e * 1.20)
    ks = tuple(ky * x for x in (0, 0.25, 0.5, 1, 1.5, 2, 3, 5, 10, 20))
    points = solve_component_section(components, ks)
    refs = tuple(symmetric_component_oracle(components, k) for k in ks)
    scale = max(refs)
    error = max(abs(p.moment - r) for p, r in zip(points, refs)) / scale
    moments = tuple(p.moment for p in points)
    works = tuple(p.work for p in points)
    residual = max(abs(p.axial_residual) for p in points) / sum(c.area * sy for c in components)
    checks = {
        "moment_monotone": all(b >= a for a, b in pairwise(moments)),
        "work_nonnegative_monotone": works[0] >= 0 and all(b >= a for a, b in pairwise(works)),
        "axial_equilibrium": residual < 1e-10,
        "multiple_failure_events": len(
            {s.name for p in points for s in p.states if s.mode != "elastic"}
        )
        >= 4,
    }
    return {
        "schema": "tensorfem.hull-girder-progressive/1.0",
        "scope": "TensorFEM component-section qualification; no TensorLBM",
        "method": "plane-section component integration with axial-force equilibrium",
        "oracle": "closed-form mirrored-component elastic-perfect-plastic sum",
        "limitations": (
            "Qualification prototype only; not a complete class-rule method. "
            "It excludes shell instability interaction, imperfections, residual "
            "stress, unloading, cyclic damage and fracture."
        ),
        "tolerance": 0.03,
        "errors": {"moment_curvature": error},
        "checks": checks,
        "passed": error < 0.03 and all(checks.values()),
        "curve": [
            {
                "curvature": p.curvature,
                "moment": p.moment,
                "reference_moment": r,
                "axial_residual": p.axial_residual,
                "work": p.work,
                "events": {s.name: s.mode for s in p.states if s.mode != "elastic"},
            }
            for p, r in zip(points, refs)
        ],
    }
