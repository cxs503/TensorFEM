"""Auditable ideal-elastic-plastic hull-girder section qualification.

This is a section-level material-spreading benchmark.  It deliberately excludes
plate/stiffener buckling, residual stress, imperfections, fracture and class-rule
ultimate-strength assessment.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class MomentCurvaturePoint:
    curvature: float
    moment: float
    yielded_fraction: float
    strain_energy: float


@dataclass(frozen=True)
class HullGirderSectionResult:
    points: tuple[MomentCurvaturePoint, ...]
    initial_yield_curvature: float
    initial_yield_moment: float
    full_plastic_moment: float
    elastic_rigidity: float
    max_abs_stress: float


def _positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real scalar")
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return out


def rectangular_section_oracle(
    *, width: float, depth: float, young: float, yield_stress: float, curvature: float
) -> float:
    """Closed-form moment for a rectangular ideal-elastic-plastic section."""
    b, h = _positive("width", width), _positive("depth", depth)
    e, sy = _positive("young", young), _positive("yield_stress", yield_stress)
    if isinstance(curvature, bool) or not isinstance(curvature, (int, float)):
        raise TypeError("curvature must be a real scalar")
    kappa = float(curvature)
    if not math.isfinite(kappa) or kappa < 0.0:
        raise ValueError("curvature must be finite and non-negative")
    ky = 2.0 * sy / (e * h)
    inertia = b * h**3 / 12.0
    if kappa <= ky:
        return e * inertia * kappa
    core = sy / (e * kappa)
    return b * sy * h**2 / 4.0 - b * sy * core**2 / 3.0


def solve_rectangular_hull_girder(
    *, width: float, depth: float, young: float, yield_stress: float,
    curvatures: tuple[float, ...] | list[float], fibers: int = 400,
) -> HullGirderSectionResult:
    """Integrate an ideal-elastic-plastic rectangular section using midpoint fibers."""
    b, h = _positive("width", width), _positive("depth", depth)
    e, sy = _positive("young", young), _positive("yield_stress", yield_stress)
    if isinstance(fibers, bool) or not isinstance(fibers, int):
        raise TypeError("fibers must be an integer")
    if fibers < 4 or fibers % 2:
        raise ValueError("fibers must be an even integer of at least 4")
    if not isinstance(curvatures, (tuple, list)) or len(curvatures) < 2:
        raise ValueError("curvatures must contain at least two values")
    ks: list[float] = []
    for value in curvatures:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("curvatures must contain real scalars")
        k = float(value)
        if not math.isfinite(k) or k < 0.0:
            raise ValueError("curvatures must be finite and non-negative")
        ks.append(k)
    if ks[0] != 0.0 or any(right <= left for left, right in zip(ks, ks[1:])):
        raise ValueError("curvatures must start at zero and be strictly increasing")

    dy, area = h / fibers, b * h / fibers
    ys = tuple(-h / 2.0 + (i + 0.5) * dy for i in range(fibers))
    discrete_i = sum(area * y * y for y in ys)
    points: list[MomentCurvaturePoint] = []
    work = 0.0
    previous_k = previous_m = 0.0
    max_stress = 0.0
    for kappa in ks:
        stresses = tuple(max(-sy, min(sy, e * kappa * y)) for y in ys)
        moment = sum(area * stress * y for stress, y in zip(stresses, ys))
        yielded = sum(abs(stress) >= sy * (1.0 - 1e-14) for stress in stresses) / fibers
        if points:
            work += 0.5 * (previous_m + moment) * (kappa - previous_k)
        points.append(MomentCurvaturePoint(kappa, moment, yielded, work))
        previous_k, previous_m = kappa, moment
        max_stress = max(max_stress, *(abs(value) for value in stresses))

    ymax = max(abs(y) for y in ys)
    ky_num = sy / (e * ymax)
    my_num = e * discrete_i * ky_num
    mp_num = sum(area * sy * abs(y) for y in ys)
    return HullGirderSectionResult(tuple(points), ky_num, my_num, mp_num,
                                   e * discrete_i, max_stress)


def run_hull_girder_ultimate_benchmark(*, fibers: int = 400) -> dict[str, object]:
    """Run the SI-unit qualification case and enforce a strict three-percent gate."""
    b, h, e, sy = 0.80, 2.40, 210e9, 355e6
    ky = 2.0 * sy / (e * h)
    curvatures = tuple(ky * factor for factor in (0.0, 0.25, 0.5, 1.0, 1.5, 2, 3, 5, 10, 20))
    result = solve_rectangular_hull_girder(
        width=b, depth=h, young=e, yield_stress=sy,
        curvatures=curvatures, fibers=fibers,
    )
    oracle = tuple(rectangular_section_oracle(
        width=b, depth=h, young=e, yield_stress=sy, curvature=k,
    ) for k in curvatures)
    my_ref, mp_ref = b * sy * h**2 / 6.0, b * sy * h**2 / 4.0
    curve_error = max(abs(p.moment - ref) / mp_ref for p, ref in zip(result.points, oracle))
    yield_error = abs(result.initial_yield_moment - my_ref) / my_ref
    plastic_error = abs(result.full_plastic_moment - mp_ref) / mp_ref
    moments = tuple(p.moment for p in result.points)
    fractions = tuple(p.yielded_fraction for p in result.points)
    energies = tuple(p.strain_energy for p in result.points)
    checks = {
        "moment_monotone": all(b >= a for a, b in zip(moments, moments[1:])),
        "yielded_fraction_monotone": all(b >= a for a, b in zip(fractions, fractions[1:])),
        "work_nonnegative_monotone": energies[0] >= 0
        and all(b >= a for a, b in zip(energies, energies[1:])),
        "stress_bounded": result.max_abs_stress <= sy * (1.0 + 1e-12),
    }
    errors = {"moment_curvature": curve_error, "initial_yield_moment": yield_error,
              "full_plastic_moment": plastic_error}
    return {
        "schema": "tensorfem.hull-girder-ultimate/1.0",
        "scope": "TensorFEM section-fiber ideal-elastic-plastic qualification; no TensorLBM",
        "units": "SI (m, Pa, N m, 1/m)",
        "method": "independent midpoint-fiber stress integration",
        "oracle": "closed-form rectangular ideal-elastic-plastic bending solution",
        "limitations": (
            "Not complete hull progressive collapse or a class-rule assessment; excludes "
            "local buckling, imperfections, residual stress and fracture."
        ),
        "errors": errors, "tolerance": 0.03, "checks": checks,
        "passed": max(errors.values()) < 0.03 and all(checks.values()),
        "initial_yield": {"computed_curvature": result.initial_yield_curvature,
                          "computed_moment": result.initial_yield_moment,
                          "reference_curvature": ky, "reference_moment": my_ref},
        "full_plastic": {"computed_moment": result.full_plastic_moment,
                         "reference_moment": mp_ref},
        "curve": [{"curvature": p.curvature, "computed_moment": p.moment,
                   "reference_moment": ref, "yielded_fraction": p.yielded_fraction,
                   "work": p.strain_energy} for p, ref in zip(result.points, oracle)],
    }
