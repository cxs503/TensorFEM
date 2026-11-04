"""Combined in-plane loading of simply supported marine panels.

Why this module exists
----------------------
``tensorfem.marine_plate_buckling`` qualifies a simply supported panel under
*uniaxial* compression.  Its numerical side exploits the fact that a
rectangular panel under ``N_x`` only is separable: the eigenvalues of the
second-order central-difference operators are written down in closed form as
``4 sin^2(m pi / 2 (n+1)) / h^2``.  Ship panels are never loaded that way.
Classification-society screening uses the simultaneous action of longitudinal
compression, transverse compression and in-plane shear, and the shear term

    - 2 N_xy w_,xy

introduces a mixed derivative that couples every ``(m, n)`` pair, so the
separable shortcut stops applying.  This module supplies the missing part:

* an **exact Navier oracle** for combined biaxial compression, which stays
  closed form because biaxial loading keeps the double-sine eigenfunctions;
* a **non-separable finite-difference eigensolver** for the general
  ``(N_x, N_y, N_xy)`` combination, assembled as explicit Kronecker matrices;
* the classical **shear buckling coefficient** ``k_s`` of Timoshenko & Gere as
  the literature anchor for the pure-shear branch.

Formulation
-----------
Simply supported rectangular panel ``a x b``, orthotropic bending rigidities
``Dx``, ``Dy`` and ``H = D12 + 2 D66``.  The buckling equation is

    Dx w_,xxxx + 2 H w_,xxyy + Dy w_,yyyy
        = N_x w_,xx + 2 N_xy w_,xy + N_y w_,yy

with ``w = 0`` and zero normal bending moment on the edges.  One reference line
load ``N`` is factored out of the three components,

    N_x = N * longitudinal_share,   N_y = N * transverse_share,
    N_xy = N * shear_share,

so the eigensolve returns the single scalar ``N`` at which the panel buckles.
Loading shares may be negative (tension); the solver then reports the positive
buckling load if one exists and raises otherwise.

Exact oracle (biaxial)
----------------------
For ``N_x > 0``, ``N_y = r N_x`` the double-sine solution is exact and the
critical load is the smallest member of

    N_x(m, n) = [Dx alpha^4 + 2 H alpha^2 beta^2 + Dy beta^4] / (alpha^2 + r beta^2)

with ``alpha = m pi / a``, ``beta = n pi / b``.  With ``r = 0`` this
reproduces :func:`tensorfem.marine_plate_buckling.navier_uniaxial_buckling_load`
term by term, which the tests use as a consistency gate.

Finite-difference discretisation
--------------------------------
Interior nodes only, ``w = 0`` on all four edges.  With ``Sx``, ``Sy`` the
one-dimensional three-point second differences and ``Dx1``, ``Dy1`` the central
first differences,

    K  = Dx (Sx^2 kron I) + 2 H (Sx kron Sy) + Dy (I kron Sy^2)
    Gx = Sx kron I,   Gy = I kron Sy,   Gxy = Dx1 kron Dy1
    G  = lx Gx + ty Gy + 2 sxy Gxy

and the critical load solves ``K v = N G v``.  ``K`` is symmetric positive
definite and every geometric term is self-adjoint, so the pencil is reduced
through the Cholesky factor of ``K`` to ``L^-1 G L^-T mu = (1/N) mu``.  The
geometric operator is indefinite (that is what makes shear able to buckle in
either sense), so the critical load is the largest **positive** member of the
reduced spectrum.  The shear coupling adds no non-symmetry -- ``D1 kron D1`` of
two antisymmetric first differences is symmetric -- but it does destroy
separability, which is why the closed-form eigenvalue of the separable case
cannot be reused.

References
----------
* S. Timoshenko, J.M. Gere, *Theory of Elastic Stability*, 2nd ed.,
  McGraw-Hill 1961, Art. 9.3 (shear buckling of simply supported plates,
  ``k_s = 5.34 + 4 (b/a)^2``).
* R. Szilard, *Theories and Applications of Plate Analysis*, Wiley 2004,
  Sec. 10.4 (combined in-plane loading, Navier reduction).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch


@dataclass(frozen=True)
class PanelBucklingResult:
    """Critical load of a simply supported panel under a load combination."""

    critical_line_load: float
    """Reference line load ``N`` at buckling (N/m)."""

    critical_stress: float
    """``N / thickness`` (Pa)."""

    grid: tuple[int, int]
    """Interior node counts ``(grid_x, grid_y)``."""

    shares: tuple[float, float, float]
    """``(longitudinal, transverse, shear)`` multipliers of ``N``."""

    reference_line_load: float | None = None
    relative_error: float | None = None
    mode_shape: torch.Tensor | None = None
    """Buckling mode, shape ``(grid_x, grid_y)``, when ``keep_mode=True``."""


@dataclass(frozen=True)
class BiaxialBucklingResult:
    """Exact Navier result for combined longitudinal/transverse compression."""

    critical_line_load: float
    """Critical *longitudinal* line load ``N_x`` (N/m)."""

    longitudinal_half_waves: int
    transverse_half_waves: int


def _finite(name: str, value, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _integer(name: str, value, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def navier_biaxial_buckling_load(
    *, length: float, width: float, rigidity_x: float, rigidity_y: float,
    coupling_rigidity: float, biaxial_ratio: float = 0.0,
    maximum_half_waves: int = 64,
) -> BiaxialBucklingResult:
    """Exact critical ``N_x`` for ``N_y = biaxial_ratio * N_x`` (both compressive).

    ``biaxial_ratio = 0`` is uniaxial compression; ``biaxial_ratio = 1`` is equal
    biaxial compression.  A negative ratio (transverse tension) is admissible as
    long as a positive buckling load remains.
    """
    a = _finite("length", length, positive=True)
    b = _finite("width", width, positive=True)
    dx = _finite("rigidity_x", rigidity_x, positive=True)
    dy = _finite("rigidity_y", rigidity_y, positive=True)
    h = _finite("coupling_rigidity", coupling_rigidity, positive=True)
    ratio = _finite("biaxial_ratio", biaxial_ratio)
    limit = _integer("maximum_half_waves", maximum_half_waves)
    best = (math.inf, 0, 0)
    for m in range(1, limit + 1):
        alpha2 = (m * math.pi / a) ** 2
        for n in range(1, limit + 1):
            beta2 = (n * math.pi / b) ** 2
            loading = alpha2 + ratio * beta2
            if loading <= 0.0:
                # The transverse tension is large enough to make this mode
                # non-buckling; it cannot be the critical one.
                continue
            load = (dx * alpha2**2 + 2.0 * h * alpha2 * beta2 + dy * beta2**2) / loading
            if load < best[0]:
                best = (load, m, n)
    if not math.isfinite(best[0]):
        raise ValueError("biaxial_ratio leaves no compressive mode; no buckling load exists")
    if best[1] == limit or best[2] == limit:
        raise ValueError("maximum_half_waves is too small to establish the minimum")
    return BiaxialBucklingResult(best[0], best[1], best[2])


def classical_shear_buckling_coefficient(aspect_ratio: float) -> float:
    """Timoshenko & Gere shear buckling coefficient ``k_s`` for ``a/b >= 1``.

    ``tau_cr = k_s pi^2 D / (b^2 t)`` with ``k_s = 5.34 + 4 (b/a)^2``.  The
    expression is asymptotic in ``a/b``; it is quoted as the literature anchor
    for the square panel (``k_s = 9.34``) and is *not* used as a reference for
    intermediate aspect ratios, where the fit carries an error of a few percent.
    """
    aspect = _finite("aspect_ratio", aspect_ratio, positive=True)
    if aspect < 1.0:
        raise ValueError("aspect_ratio must be at least 1 (longitudinal direction is x)")
    return 5.34 + 4.0 / aspect**2


def _second_difference(n: int, h: float, dtype) -> torch.Tensor:
    matrix = torch.zeros((n, n), dtype=dtype)
    index = torch.arange(n)
    matrix[index, index] = -2.0
    matrix[index[:-1], index[1:]] = 1.0
    matrix[index[1:], index[:-1]] = 1.0
    return matrix / h**2


def _central_first_difference(n: int, h: float, dtype) -> torch.Tensor:
    matrix = torch.zeros((n, n), dtype=dtype)
    index = torch.arange(n)
    matrix[index[:-1], index[1:]] = 1.0
    matrix[index[1:], index[:-1]] = -1.0
    return matrix / (2.0 * h)


def _pencil_matrices(
    length: float, width: float, grid_x: int, grid_y: int,
    rigidity_x: float, rigidity_y: float, coupling_rigidity: float, dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hx, hy = length / (grid_x + 1), width / (grid_y + 1)
    sx = _second_difference(grid_x, hx, dtype)
    sy = _second_difference(grid_y, hy, dtype)
    d1x = _central_first_difference(grid_x, hx, dtype)
    d1y = _central_first_difference(grid_y, hy, dtype)
    ix, iy = torch.eye(grid_x, dtype=dtype), torch.eye(grid_y, dtype=dtype)
    stiffness = (
        rigidity_x * torch.kron(sx @ sx, iy)
        + 2.0 * coupling_rigidity * torch.kron(sx, sy)
        + rigidity_y * torch.kron(ix, sy @ sy)
    )
    return stiffness, torch.kron(sx, iy), torch.kron(ix, sy), torch.kron(d1x, d1y)


def panel_buckling_load(
    *, length: float, width: float, thickness: float, rigidity_x: float,
    rigidity_y: float, coupling_rigidity: float, grid_x: int, grid_y: int,
    longitudinal_share: float = 1.0, transverse_share: float = 0.0,
    shear_share: float = 0.0, keep_mode: bool = False,
    dtype=torch.float64,
) -> PanelBucklingResult:
    """Critical reference line load for a general in-plane load combination.

    The returned load ``N`` corresponds to ``N_x = longitudinal_share * N``,
    ``N_y = transverse_share * N`` and ``N_xy = shear_share * N``.  The
    eigensolve is non-separable, so the shear term is admissible.
    """
    a = _finite("length", length, positive=True)
    b = _finite("width", width, positive=True)
    t = _finite("thickness", thickness, positive=True)
    dx = _finite("rigidity_x", rigidity_x, positive=True)
    dy = _finite("rigidity_y", rigidity_y, positive=True)
    h = _finite("coupling_rigidity", coupling_rigidity, positive=True)
    nx = _integer("grid_x", grid_x, 2)
    ny = _integer("grid_y", grid_y, 2)
    lx = _finite("longitudinal_share", longitudinal_share)
    ty = _finite("transverse_share", transverse_share)
    sxy = _finite("shear_share", shear_share)
    if lx == 0.0 and ty == 0.0 and sxy == 0.0:
        raise ValueError("at least one load share must be non-zero")
    if lx <= 0.0 and ty <= 0.0 and sxy == 0.0:
        raise ValueError("tension-only loading does not buckle; no compressive load state")

    stiffness, gx, gy, gxy = _pencil_matrices(a, b, nx, ny, dx, dy, h, dtype)
    # ``Sx`` and ``Sy`` are negative semidefinite (three-point second difference
    # with zero edge values), so the compressive branch of ``K v = N G v`` is the
    # negative branch.  Flipping the geometric operator once makes ``N > 0`` the
    # smallest positive eigenvalue, which is the quantity a rule check wants.
    geometry = -(lx * gx + ty * gy + 2.0 * sxy * gxy)

    chol = torch.linalg.cholesky(stiffness)
    left = torch.linalg.solve_triangular(chol, geometry, upper=False)
    reduced = torch.linalg.solve_triangular(chol, left.T, upper=False).T
    # Every geometric term is self-adjoint under ``w = 0`` on the boundary, so the
    # pencil is symmetric even with shear; the reduction can only perturb that by
    # round-off.  The indefinite geometric operator makes the spectrum two-sided,
    # so the buckling load is the largest *positive* eigenvalue of the reduced
    # problem (i.e. the smallest positive ``N``).
    reduced = 0.5 * (reduced + reduced.T)
    values, vectors = torch.linalg.eigh(reduced)
    real = values.real
    if not bool(torch.any(real > 0.0)):
        raise ValueError("load combination admits no positive buckling load")
    masked = torch.where(real > 0.0, real, torch.zeros_like(real))
    index = int(torch.argmax(masked))
    load = 1.0 / real[index].item()
    mode = vectors[:, index]

    shape = None
    if keep_mode:
        shape = torch.linalg.solve_triangular(chol.T, mode.unsqueeze(1), upper=True)[:, 0]
        shape = shape.reshape(nx, ny)
        norm = shape.abs().max()
        shape = shape / norm if norm > 0 else shape
    return PanelBucklingResult(
        critical_line_load=load, critical_stress=load / t, grid=(nx, ny),
        shares=(lx, ty, sxy), mode_shape=shape,
    )


def panel_buckling_error(result: PanelBucklingResult) -> PanelBucklingResult:
    """Attach the relative error against the stored reference load."""
    if result.reference_line_load is None:
        raise ValueError("result carries no reference load")
    if result.reference_line_load == 0.0:
        raise ValueError("reference load must be non-zero")
    error = abs(result.critical_line_load / result.reference_line_load - 1.0)
    return replace(result, relative_error=error)


def run_combined_panel_buckling_qualification() -> dict[str, object]:
    """Qualify the combined-loading kernels and raise on any failed gate.

    Gates (all fail-closed):

    1. equal biaxial compression is exactly ``2 pi^2 D / b^2`` for the square
       isotropic panel -- the Navier oracle must hit it to rounding;
    2. the biaxial oracle with ``ratio = 0`` must reproduce the uniaxial oracle of
       :mod:`tensorfem.marine_plate_buckling` to rounding;
    3. the non-separable finite-difference solver under uniaxial load must
       reproduce the separable solver to rounding, which proves the Kronecker
       assembly is the same operator;
    4. biaxial compression at ``a/b = 2`` and ratio ``0.5`` must converge
       monotonically to the exact Navier load and end below 3 %;
    5. pure shear on the square panel must land within 3 % of the literature
       coefficient ``k_s = 9.34`` (Timoshenko & Gere).
    """
    e, nu, t = 210.0e9, 0.3, 0.012
    d = e * t**3 / (12.0 * (1.0 - nu**2))
    h = d
    rows: list[dict[str, object]] = []

    # Gate 1: equal biaxial, square, isotropic.
    exact_square = navier_biaxial_buckling_load(
        length=1.0, width=1.0, rigidity_x=d, rigidity_y=d,
        coupling_rigidity=h, biaxial_ratio=1.0)
    expected_square = 2.0 * math.pi**2 * d
    rows.append({
        "gate": "equal_biaxial_square_exact",
        "measured": exact_square.critical_line_load,
        "expected": expected_square,
        "relative_error": abs(exact_square.critical_line_load / expected_square - 1.0),
        "tolerance": 1e-12,
        "passed": abs(exact_square.critical_line_load / expected_square - 1.0) < 1e-12,
        "note": "Navier oracle vs 2 pi^2 D / b^2 for Nx = Ny",
    })

    # Gate 2: biaxial ratio 0 reproduces the uniaxial oracle.
    from .marine_plate_buckling import navier_uniaxial_buckling_load

    uniaxial, m, n = navier_uniaxial_buckling_load(
        length=2.0, width=1.0, rigidity_x=d, rigidity_y=d, coupling_rigidity=h)
    biaxial_zero = navier_biaxial_buckling_load(
        length=2.0, width=1.0, rigidity_x=d, rigidity_y=d,
        coupling_rigidity=h, biaxial_ratio=0.0)
    rows.append({
        "gate": "biaxial_reduces_to_uniaxial",
        "measured": biaxial_zero.critical_line_load,
        "expected": uniaxial,
        "relative_error": abs(biaxial_zero.critical_line_load / uniaxial - 1.0),
        "tolerance": 1e-12,
        "passed": biaxial_zero.critical_line_load == uniaxial,
        "note": f"mode ({m}, {n}) shared with marine_plate_buckling",
    })

    # Gate 3: non-separable solver reproduces the separable one under N_x only.
    from .marine_plate_buckling import finite_difference_uniaxial_buckling

    separable = finite_difference_uniaxial_buckling(
        length=2.0, width=1.0, thickness=t, rigidity_x=d, rigidity_y=d,
        coupling_rigidity=h, grid_x=16, grid_y=16)
    assembled = panel_buckling_load(
        length=2.0, width=1.0, thickness=t, rigidity_x=d, rigidity_y=d,
        coupling_rigidity=h, grid_x=16, grid_y=16, longitudinal_share=1.0)
    rows.append({
        "gate": "assembled_matches_separable_uniaxial",
        "measured": assembled.critical_line_load,
        "expected": separable.critical_line_load,
        "relative_error": abs(assembled.critical_line_load / separable.critical_line_load - 1.0),
        "tolerance": 1e-10,
        "passed": abs(assembled.critical_line_load / separable.critical_line_load - 1.0) < 1e-10,
        "note": "Kronecker assembly vs closed-form sin^2 eigenvalues, 16x16",
    })

    # Gate 4: biaxial ratio 0.5 at a/b = 2, grid refinement against Navier.
    ratio = 0.5
    reference = navier_biaxial_buckling_load(
        length=2.0, width=1.0, rigidity_x=d, rigidity_y=d,
        coupling_rigidity=h, biaxial_ratio=ratio).critical_line_load
    sequence = [
        panel_buckling_error(replace(
            panel_buckling_load(
                length=2.0, width=1.0, thickness=t, rigidity_x=d, rigidity_y=d,
                coupling_rigidity=h, grid_x=n, grid_y=n,
                longitudinal_share=1.0, transverse_share=ratio),
            reference_line_load=reference))
        for n in (8, 16, 32)
    ]
    errors = [item.relative_error for item in sequence]
    convergent = errors[2] < errors[1] < errors[0]
    passed = convergent and errors[2] < 0.03
    rows.append({
        "gate": "biaxial_ratio_0.5_convergence",
        "grids": [item.grid for item in sequence],
        "loads": [item.critical_line_load for item in sequence],
        "errors": errors,
        "reference_load": reference,
        "tolerance": 0.03,
        "passed": passed,
        "note": "Nx + 0.5 Ny, a/b = 2, exact Navier reference",
    })

    # Gate 5: pure shear, square panel, literature coefficient.
    ks_literature = 9.34
    shear_reference = ks_literature * math.pi**2 * d / 1.0**2
    shear_sequence = [
        panel_buckling_error(replace(
            panel_buckling_load(
                length=1.0, width=1.0, thickness=t, rigidity_x=d, rigidity_y=d,
                coupling_rigidity=h, grid_x=n, grid_y=n,
                longitudinal_share=0.0, shear_share=1.0),
            reference_line_load=shear_reference))
        for n in (16, 24, 32)
    ]
    shear_errors = [item.relative_error for item in shear_sequence]
    shear_convergent = shear_errors[2] < shear_errors[1] < shear_errors[0]
    shear_passed = shear_convergent and shear_errors[2] < 0.03
    rows.append({
        "gate": "pure_shear_square_literature",
        "grids": [item.grid for item in shear_sequence],
        "loads": [item.critical_line_load for item in shear_sequence],
        "coefficients": [item.critical_line_load / (math.pi**2 * d) for item in shear_sequence],
        "errors": shear_errors,
        "reference_coefficient": ks_literature,
        "reference_load": shear_reference,
        "tolerance": 0.03,
        "passed": shear_passed,
        "note": "tau_cr = k_s pi^2 D / b^2 from Timoshenko & Gere",
    })

    if not all(bool(row["passed"]) for row in rows):
        raise AssertionError("combined panel buckling qualification failed")
    return {
        "schema": "tensorfem.combined-panel-buckling/1.0",
        "scope": (
            "thin-plate eigenproblem for simply supported panels; the shear branch is "
            "anchored on the classical square-panel coefficient, not on a closed form"
        ),
        "tolerance": 0.03,
        "young": e,
        "poisson": nu,
        "thickness": t,
        "cases": rows,
        "passed": True,
    }