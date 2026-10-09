"""Qualification kernels for local buckling of simply supported marine panels.

The numerical result is a finite-difference eigenvalue of the classical
orthotropic plate equation.  The oracle is the independent continuous Navier
solution.  This deliberately does not claim shell-element prebuckling stress
or postbuckling capability.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PlateBucklingResult:
    critical_line_load: float
    critical_stress: float
    longitudinal_half_waves: int
    transverse_half_waves: int
    reference_line_load: float
    relative_error: float
    grid: tuple[int, int]


def _positive_finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _positive_integer(name: str, value: int, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def isotropic_plate_rigidities(young: float, poisson: float, thickness: float) -> tuple[float, float, float]:
    """Return ``Dx, Dy, H=D12+2D66`` in N m for an isotropic thin plate."""
    young = _positive_finite("young", young)
    thickness = _positive_finite("thickness", thickness)
    if isinstance(poisson, bool) or not isinstance(poisson, (int, float)):
        raise TypeError("poisson must be a real number")
    poisson = float(poisson)
    if not math.isfinite(poisson) or not -1.0 < poisson < 0.5:
        raise ValueError("poisson must be finite and lie between -1 and 0.5")
    rigidity = young * thickness**3 / (12.0 * (1.0 - poisson**2))
    return rigidity, rigidity, rigidity


def navier_uniaxial_buckling_load(
    *, length: float, width: float, rigidity_x: float, rigidity_y: float,
    coupling_rigidity: float, maximum_half_waves: int = 64,
) -> tuple[float, int, int]:
    """Continuous Navier oracle for a simply supported orthotropic plate.

    ``coupling_rigidity`` is ``D12 + 2 D66``.  Compression is a positive line
    load in N/m along ``x``.  Positive definite bending rigidities are required.
    """
    a = _positive_finite("length", length)
    b = _positive_finite("width", width)
    dx = _positive_finite("rigidity_x", rigidity_x)
    dy = _positive_finite("rigidity_y", rigidity_y)
    h = _positive_finite("coupling_rigidity", coupling_rigidity)
    limit = _positive_integer("maximum_half_waves", maximum_half_waves)
    best = (math.inf, 0, 0)
    # Under x-only compression the minimum transverse half-wave is n=1, but
    # enumerate n as an audit against accidental hard-coding of that fact.
    for m in range(1, limit + 1):
        alpha2 = (m * math.pi / a) ** 2
        for n in range(1, limit + 1):
            beta2 = (n * math.pi / b) ** 2
            load = (dx * alpha2**2 + 2.0 * h * alpha2 * beta2 + dy * beta2**2) / alpha2
            if load < best[0]:
                best = (load, m, n)
    if best[1] == limit or best[2] == limit:
        raise ValueError("maximum_half_waves is too small to establish the minimum")
    return best


def finite_difference_uniaxial_buckling(
    *, length: float, width: float, thickness: float, rigidity_x: float,
    rigidity_y: float, coupling_rigidity: float, grid_x: int,
    grid_y: int, maximum_half_waves: int = 64,
) -> PlateBucklingResult:
    """Solve the simply supported plate eigenproblem by central differences.

    The separable eigenvalues of the second-order central-difference operators
    are used directly; no continuous wave numbers enter the numerical result.
    ``grid_x`` and ``grid_y`` are the numbers of interior nodes.
    """
    a = _positive_finite("length", length)
    b = _positive_finite("width", width)
    t = _positive_finite("thickness", thickness)
    dx = _positive_finite("rigidity_x", rigidity_x)
    dy = _positive_finite("rigidity_y", rigidity_y)
    h = _positive_finite("coupling_rigidity", coupling_rigidity)
    nx = _positive_integer("grid_x", grid_x, 2)
    ny = _positive_integer("grid_y", grid_y, 2)
    limit = _positive_integer("maximum_half_waves", maximum_half_waves)
    reference, reference_m, reference_n = navier_uniaxial_buckling_load(
        length=a, width=b, rigidity_x=dx, rigidity_y=dy,
        coupling_rigidity=h, maximum_half_waves=limit,
    )
    hx, hy = a / (nx + 1), b / (ny + 1)
    best = (math.inf, 0, 0)
    for m in range(1, nx + 1):
        lx = 4.0 * math.sin(m * math.pi / (2.0 * (nx + 1))) ** 2 / hx**2
        for n in range(1, ny + 1):
            ly = 4.0 * math.sin(n * math.pi / (2.0 * (ny + 1))) ** 2 / hy**2
            load = (dx * lx**2 + 2.0 * h * lx * ly + dy * ly**2) / lx
            if load < best[0]:
                best = (load, m, n)
    error = abs(best[0] / reference - 1.0)
    return PlateBucklingResult(best[0], best[0] / t, best[1], best[2], reference,
                               error, (nx, ny))


def smeared_longitudinal_stiffener_rigidity(
    *, plate_rigidity: float, stiffener_young: float, stiffener_area: float,
    eccentricity: float, spacing: float,
) -> float:
    """Equivalent longitudinal ``Dx`` for identical discrete stiffeners."""
    base = _positive_finite("plate_rigidity", plate_rigidity)
    young = _positive_finite("stiffener_young", stiffener_young)
    area = _positive_finite("stiffener_area", stiffener_area)
    spacing = _positive_finite("spacing", spacing)
    if isinstance(eccentricity, bool) or not isinstance(eccentricity, (int, float)):
        raise TypeError("eccentricity must be a real number")
    eccentricity = float(eccentricity)
    if not math.isfinite(eccentricity):
        raise ValueError("eccentricity must be finite")
    return base + young * area * eccentricity**2 / spacing


def run_local_plate_buckling_qualification() -> dict[str, object]:
    """Run isotropic and smeared-stiffener convergence gates below 3 percent."""
    e, nu, t = 210.0e9, 0.3, 0.012
    d, _, h = isotropic_plate_rigidities(e, nu, t)
    cases = (("square", 1.0, d), ("long_panel", 2.0, d),
             ("longitudinally_stiffened", 2.0,
              smeared_longitudinal_stiffener_rigidity(
                  plate_rigidity=d, stiffener_young=e, stiffener_area=3.0e-4,
                  eccentricity=0.04, spacing=0.5)))
    rows: list[dict[str, object]] = []
    for name, length, dx in cases:
        sequence = [finite_difference_uniaxial_buckling(
            length=length, width=1.0, thickness=t, rigidity_x=dx,
            rigidity_y=d, coupling_rigidity=h, grid_x=n, grid_y=n,
        ) for n in (8, 16, 32)]
        final = sequence[-1]
        convergent = sequence[-1].relative_error < sequence[-2].relative_error < sequence[-3].relative_error
        passed = convergent and final.relative_error < 0.03
        rows.append({"case": name, "grids": [item.grid for item in sequence],
                     "loads": [item.critical_line_load for item in sequence],
                     "errors": [item.relative_error for item in sequence],
                     "mode": [final.longitudinal_half_waves, final.transverse_half_waves],
                     "reference_load": final.reference_line_load, "passed": passed})
    passed = all(bool(row["passed"]) for row in rows)
    if not passed:
        raise AssertionError("local plate buckling qualification failed")
    return {"schema": "tensorfem.local-plate-buckling/1.0",
            "scope": "TensorFEM finite-difference plate equation; no TensorLBM/CFD",
            "tolerance": 0.03, "cases": rows, "passed": passed}
