"""Auditable plate prestress and imperfect postbuckling qualification kernels.

This module deliberately implements classical simply-supported plate models,
not a general shell finite element.  The eigenproblem uses central-difference
wave numbers.  The nonlinear path is a one-mode von Karman--Koiter reduction
integrated by load continuation and checked against an independent root solve.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .marine_plate_buckling import (_positive_finite, _positive_integer,
                                    isotropic_plate_rigidities)


@dataclass(frozen=True)
class PrestressBucklingResult:
    load_factor: float
    reference_load_factor: float
    longitudinal_half_waves: int
    transverse_half_waves: int
    relative_error: float
    grid: tuple[int, int]


@dataclass(frozen=True)
class PostbucklingPoint:
    load_factor: float
    amplitude: float
    reference_amplitude: float
    relative_error: float


def _nonnegative_finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be nonnegative and finite")
    return result


def navier_prestress_buckling_factor(
    *, length: float, width: float, rigidity_x: float, rigidity_y: float,
    coupling_rigidity: float, prestress_x: float, prestress_y: float,
    maximum_half_waves: int = 64,
) -> tuple[float, int, int]:
    """Return the multiplier on prescribed uniform compressive line loads."""
    a, b = _positive_finite("length", length), _positive_finite("width", width)
    dx = _positive_finite("rigidity_x", rigidity_x)
    dy = _positive_finite("rigidity_y", rigidity_y)
    h = _positive_finite("coupling_rigidity", coupling_rigidity)
    nx = _nonnegative_finite("prestress_x", prestress_x)
    ny = _nonnegative_finite("prestress_y", prestress_y)
    if nx == 0.0 and ny == 0.0:
        raise ValueError("at least one prestress component must be positive")
    limit = _positive_integer("maximum_half_waves", maximum_half_waves)
    best = (math.inf, 0, 0)
    for m in range(1, limit + 1):
        alpha2 = (m * math.pi / a) ** 2
        for n in range(1, limit + 1):
            beta2 = (n * math.pi / b) ** 2
            numerator = dx * alpha2**2 + 2.0 * h * alpha2 * beta2 + dy * beta2**2
            factor = numerator / (nx * alpha2 + ny * beta2)
            if factor < best[0]:
                best = factor, m, n
    if best[1] == limit or best[2] == limit:
        raise ValueError("maximum_half_waves is too small to establish the minimum")
    return best


def finite_difference_prestress_buckling(
    *, length: float, width: float, rigidity_x: float, rigidity_y: float,
    coupling_rigidity: float, prestress_x: float, prestress_y: float,
    grid_x: int, grid_y: int, maximum_half_waves: int = 64,
) -> PrestressBucklingResult:
    """Central-difference generalized eigenvalue for a prestressed plate."""
    a, b = _positive_finite("length", length), _positive_finite("width", width)
    dx = _positive_finite("rigidity_x", rigidity_x)
    dy = _positive_finite("rigidity_y", rigidity_y)
    h = _positive_finite("coupling_rigidity", coupling_rigidity)
    px = _nonnegative_finite("prestress_x", prestress_x)
    py = _nonnegative_finite("prestress_y", prestress_y)
    if px == 0.0 and py == 0.0:
        raise ValueError("at least one prestress component must be positive")
    gx = _positive_integer("grid_x", grid_x, 2)
    gy = _positive_integer("grid_y", grid_y, 2)
    reference, _, _ = navier_prestress_buckling_factor(
        length=a, width=b, rigidity_x=dx, rigidity_y=dy, coupling_rigidity=h,
        prestress_x=px, prestress_y=py, maximum_half_waves=maximum_half_waves)
    hx, hy = a / (gx + 1), b / (gy + 1)
    best = (math.inf, 0, 0)
    for m in range(1, gx + 1):
        lx = 4.0 * math.sin(m * math.pi / (2.0 * (gx + 1))) ** 2 / hx**2
        for n in range(1, gy + 1):
            ly = 4.0 * math.sin(n * math.pi / (2.0 * (gy + 1))) ** 2 / hy**2
            factor = (dx * lx**2 + 2.0 * h * lx * ly + dy * ly**2) / (px * lx + py * ly)
            if factor < best[0]:
                best = factor, m, n
    return PrestressBucklingResult(best[0], reference, best[1], best[2],
                                   abs(best[0] / reference - 1.0), (gx, gy))


def _postbuckling_residual(q: float, load: float, imperfection: float,
                           coefficient: float) -> float:
    # Stationarity of V=(q-w0)^2/2-load*q^2/2
    #                  +coefficient*(q^2-w0^2)^2/4.
    return (1.0 - load) * q - imperfection + coefficient * (q * q - imperfection**2) * q


def postbuckling_reference_amplitude(*, load_factor: float, imperfection: float,
                                     coefficient: float) -> float:
    """Independent bracketed root oracle for the positive equilibrium branch."""
    load = _nonnegative_finite("load_factor", load_factor)
    w0 = _positive_finite("imperfection", imperfection)
    c = _positive_finite("coefficient", coefficient)
    lo, hi = 0.0, max(w0, 1.0)
    while _postbuckling_residual(hi, load, w0, c) <= 0.0:
        hi *= 2.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _postbuckling_residual(mid, load, w0, c) > 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def imperfect_postbuckling_path(
    *, maximum_load_factor: float, steps: int, imperfection: float,
    coefficient: float = 1.0, samples: int = 7,
) -> list[PostbucklingPoint]:
    """Integrate the stable positive branch with second-order continuation.

    Amplitude and imperfection are nondimensional modal amplitudes (for
    example, physical centre deflection divided by plate thickness).
    """
    end = _positive_finite("maximum_load_factor", maximum_load_factor)
    count = _positive_integer("steps", steps, 2)
    sample_count = _positive_integer("samples", samples, 2)
    w0 = _positive_finite("imperfection", imperfection)
    c = _positive_finite("coefficient", coefficient)
    if sample_count > count + 1:
        raise ValueError("samples must not exceed steps + 1")
    q, step = w0, end / count
    states = [(0.0, q)]
    for index in range(count):
        load = index * step
        tangent = 1.0 - load + c * (3.0 * q * q - w0 * w0)
        slope = q / tangent
        predicted = q + step * slope
        next_load = load + step
        predicted_tangent = 1.0 - next_load + c * (3.0 * predicted**2 - w0 * w0)
        q += 0.5 * step * (slope + predicted / predicted_tangent)
        states.append((next_load, q))
    indices = [round(i * count / (sample_count - 1)) for i in range(sample_count)]
    points = []
    for index in indices:
        load, amplitude = states[index]
        reference = postbuckling_reference_amplitude(
            load_factor=load, imperfection=w0, coefficient=c)
        points.append(PostbucklingPoint(load, amplitude, reference,
                                        abs(amplitude / reference - 1.0)))
    return points


def run_plate_postbuckling_qualification() -> dict[str, object]:
    """Run prestress grid and imperfect-path step convergence gates."""
    d, dy, h = isotropic_plate_rigidities(210.0e9, 0.3, 0.012)
    eigen_rows = []
    for name, px, py in (("uniaxial", 1.0e6, 0.0), ("biaxial", 1.0e6, 0.5e6)):
        sequence = [finite_difference_prestress_buckling(
            length=1.0, width=1.0, rigidity_x=d, rigidity_y=dy,
            coupling_rigidity=h, prestress_x=px, prestress_y=py,
            grid_x=n, grid_y=n) for n in (8, 16, 32)]
        errors = [item.relative_error for item in sequence]
        eigen_rows.append({"case": name, "grids": [item.grid for item in sequence],
                           "factors": [item.load_factor for item in sequence],
                           "errors": errors,
                           "mode": [sequence[-1].longitudinal_half_waves,
                                    sequence[-1].transverse_half_waves],
                           "reference_factor": sequence[-1].reference_load_factor,
                           "passed": errors[2] < errors[1] < errors[0] and errors[2] < 0.03})
    paths = [imperfect_postbuckling_path(maximum_load_factor=1.2, steps=n,
                                         imperfection=0.1, coefficient=1.0)
             for n in (40, 80, 160)]
    errors = [max(point.relative_error for point in path) for path in paths]
    path_row = {"imperfection": 0.1, "coefficient": 1.0, "step_counts": [40, 80, 160],
                "maximum_errors": errors,
                "final_amplitudes": [path[-1].amplitude for path in paths],
                "reference_final_amplitude": paths[-1][-1].reference_amplitude,
                "passed": errors[2] < errors[1] < errors[0] and errors[2] < 0.03}
    passed = all(row["passed"] for row in eigen_rows) and path_row["passed"]
    if not passed:
        raise AssertionError("plate postbuckling qualification failed")
    scope = ("TensorFEM classical plate and one-mode von Karman-Koiter reduction; "
             "no general shell FE or TensorLBM")
    return {"schema": "tensorfem.plate-postbuckling/1.0", "tolerance": 0.03,
            "scope": scope,
            "prestress_eigenbuckling": eigen_rows, "imperfect_postbuckling": path_row,
            "passed": passed}
