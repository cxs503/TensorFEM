"""Two-dimensional analytic convergence order for the Q4 indicator ladder.

Why this module exists
----------------------
:mod:`tensorfem.adaptive_zz` verifies the ZZ indicator on a *singular* plane-elasticity field
(a Williams re-entrant corner).  That problem cannot demonstrate an asymptotically exact
effectivity ``theta -> 1``: the exact stress is unbounded at the corner, and the report itself
records a few percent bias.  The missing rung of that ledger is a **smooth** 2D problem whose
solution, gradient and energy norm are closed forms, so that

* the h-convergence order of the Q4 solution can be checked against theory,
* the effectivity of a recovered-gradient (ZZ) indicator can be checked for asymptotic
  exactness, and
* the uniform ladder's cost can be recorded as a baseline.

Manufactured solution
---------------------
On ``Omega = (0, 1)^2`` take the bi-quartic bubble

    u(x, y) = A * X(x) * Y(y),   X(x) = x^2 (1 - x)^2,   Y(y) = y^2 (1 - y)^2,   A = 256,

so ``u = 0`` and ``grad u = 0`` on the whole boundary, ``u(1/2, 1/2) = 1``, and

    f = -laplacian(u) = -A * (X'' Y + X Y''),    X'' = 2 - 12 x + 12 x^2,

is a polynomial.  Every integral of this problem is evaluated exactly by a Gauss rule of
sufficient order, so no quadrature error is mixed into the measured order.  ``u`` is not
representable by bilinear Q4 elements, hence the discrete error is genuine and theory predicts

===============================  =================  ==================
quantity                         asymptotic order   slope vs unknowns
===============================  =================  ==================
energy norm ``||e||_E``          ``O(h)``           ``-1/2``
L2 norm ``||e||_0``              ``O(h^2)``         ``-1``
recovered gradient ``||g* - g||``  ``O(h^2)``        ``-1``
===============================  =================  ==================

The implementation also reports the **measured** exponent of the averaged recovered gradient,
because a plain nodal average of element-centre gradients is *not* the superconvergent patch
recovery of ``adaptive_zz``: it is a cheap ZZ variant whose superconvergence is measured here
rather than asserted.

Scope and honesty
-----------------
* Scalar Poisson, homogeneous Dirichlet on the full boundary, structured Q4 meshes, **uniform**
  refinement only.  Local refinement with hanging nodes lives in
  :mod:`tensorfem.adaptive_zz` (plane elasticity) and is not re-implemented here.
* All arithmetic is *vectorised over the tensor-product grid*: the element Jacobian of a
  rectangular Q4 element is diagonal and constant, so gradients are computed with 1D
  differences instead of per-element 2x2 solves.  ``tests/test_adaptive_poisson_2d.py`` checks
  this shortcut against the per-element formulation on a distorted (non-rectangular) element.
* Effectivity ``theta -> 1`` is verified for *this* smooth field on *uniform* meshes only; on
  graded meshes, or for singular fields, no asymptotic exactness is claimed.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .adaptive_zz import convergence_rate, effectivity_index

#: amplitude that normalises the bubble so that ``u(1/2, 1/2) = 1``
AMPLITUDE = 256.0
#: Gauss order for the error integrals (exact for these polynomial integrands)
GAUSS_ERROR = 6
#: Gauss order for the indicators (exact: recovered and FE gradients are both bilinear)
GAUSS_INDICATOR = 2


def gauss_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    points, weights = np.polynomial.legendre.leggauss(int(order))
    return points, weights


# ---------------------------------------------------------------------------
# manufactured solution
# ---------------------------------------------------------------------------
def _factor(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(X, X', X'')`` for ``X(x) = x^2 (1-x)^2``."""
    x = np.asarray(x, dtype=np.float64)
    return x * x * (1.0 - x) ** 2, 2.0 * x * (1.0 - x) * (1.0 - 2.0 * x), 2.0 - 12.0 * x + 12.0 * x * x


@dataclass(frozen=True)
class BubbleSolution:
    """``u = A x^2 (1-x)^2 y^2 (1-y)^2``: zero Dirichlet trace, polynomial source."""

    amplitude: float = AMPLITUDE
    name: str = "biquartic-bubble"

    def u(self, x, y) -> np.ndarray:
        X, _, _ = _factor(x)
        Y, _, _ = _factor(y)
        return self.amplitude * X * Y

    def gradient(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        X, dX, _ = _factor(x)
        Y, dY, _ = _factor(y)
        return self.amplitude * dX * Y, self.amplitude * X * dY

    def source(self, x, y) -> np.ndarray:
        """``f = -laplacian(u)``."""
        X, _, ddX = _factor(x)
        Y, _, ddY = _factor(y)
        return -self.amplitude * (ddX * Y + X * ddY)

    def boundaries(self) -> float:
        return 0.0


# ---------------------------------------------------------------------------
# structured Q4 mesh
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Mesh2D:
    """Structured Q4 mesh of ``[0, 1]^2``; node ``(j, i)`` has id ``j (nx+1) + i``."""

    nx: int
    ny: int

    def __post_init__(self) -> None:
        if int(self.nx) < 1 or int(self.ny) < 1:
            raise ValueError("nx and ny must be >= 1")
        object.__setattr__(self, "nx", int(self.nx))
        object.__setattr__(self, "ny", int(self.ny))

    @property
    def nnodes(self) -> int:
        return (self.nx + 1) * (self.ny + 1)

    @property
    def nelements(self) -> int:
        return self.nx * self.ny

    @property
    def hx(self) -> float:
        return 1.0 / self.nx

    @property
    def hy(self) -> float:
        return 1.0 / self.ny

    @property
    def h(self) -> float:
        return max(self.hx, self.hy)

    def x_coords(self) -> np.ndarray:
        return np.linspace(0.0, 1.0, self.nx + 1)

    def y_coords(self) -> np.ndarray:
        return np.linspace(0.0, 1.0, self.ny + 1)

    def grid(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.size != self.nnodes:
            raise ValueError("values must hold one entry per node")
        return values.reshape(self.ny + 1, self.nx + 1)

    def elements(self) -> np.ndarray:
        stride = self.nx + 1
        j, i = np.meshgrid(np.arange(self.ny), np.arange(self.nx), indexing="ij")
        n0 = (j * stride + i).ravel()
        return np.column_stack((n0, n0 + 1, n0 + 1 + stride, n0 + stride)).astype(np.int64)

    def boundary_mask(self) -> np.ndarray:
        mask = np.zeros((self.ny + 1, self.nx + 1), dtype=bool)
        mask[0, :] = mask[-1, :] = True
        mask[:, 0] = mask[:, -1] = True
        return mask.ravel()

    def interior_nodes(self) -> np.ndarray:
        return np.nonzero(~self.boundary_mask())[0]


def shape_functions(xi: float, eta: float) -> np.ndarray:
    return 0.25 * np.array(
        [(1 - xi) * (1 - eta), (1 + xi) * (1 - eta), (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)],
        dtype=np.float64,
    )


def shape_gradients(xi: float, eta: float) -> np.ndarray:
    """``dN/dxi`` of the Q4 element, node order ``(-1,-1) (1,-1) (1,1) (-1,1)``."""
    return 0.25 * np.array(
        [[-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)],
         [-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)]],
        dtype=np.float64,
    )


def element_jacobian(coords: np.ndarray, xi: float, eta: float) -> tuple[np.ndarray, float]:
    """``dX/dxi`` (2x2) and its determinant for one element with corner coordinates ``coords``."""
    jacobian = coords.T @ shape_gradients(xi, eta).T
    det = float(np.linalg.det(jacobian))
    if det <= 0.0:
        raise ValueError("non-positive Jacobian: element orientation or shape is invalid")
    return jacobian, det


# ---------------------------------------------------------------------------
# vectorised gradients on the tensor-product grid
# ---------------------------------------------------------------------------
def _corners(field: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The four corner values of every element, ordered as the Q4 node order."""
    return field[:-1, :-1], field[:-1, 1:], field[1:, 1:], field[1:, :-1]


def gradient_from_nodal(mesh: Mesh2D, nodal: np.ndarray, xi: float, eta: float) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear gradient ``(du/dx, du/dy)`` of a nodal field at one reference point.

    On a rectangular tensor-product mesh the Jacobian is constant and diagonal, so the physical
    gradient reduces to two 1D differences averaged over the element -- identical to
    ``J^{-1} dN/dxi . u_e`` but without a 2x2 solve per element.
    """
    field = mesh.grid(nodal)
    v00, v10, v11, v01 = _corners(field)
    # du/dxi = 0.25 [(1-eta)(u10-u00) + (1+eta)(u11-u01)], likewise for eta (Q4 node order)
    dxi = 0.25 * ((v10 - v00) * (1.0 - eta) + (v11 - v01) * (1.0 + eta))
    deta = 0.25 * ((v01 - v00) * (1.0 - xi) + (v11 - v10) * (1.0 + xi))
    return dxi * (2.0 / mesh.hx), deta * (2.0 / mesh.hy)


def interpolate_nodal(mesh: Mesh2D, field: np.ndarray, xi: float, eta: float) -> np.ndarray:
    """Bilinear interpolation of a *nodal field* at one reference point, shape ``(ny, nx)``.

    The recovered gradient is a nodal field (a vector of nodal values), so it enters the indicator
    by interpolation -- not by differentiation, which would return the derivative of the recovered
    field instead of the recovered field itself.
    """
    v00, v10, v11, v01 = _corners(mesh.grid(field))
    N = shape_functions(xi, eta)
    return N[0] * v00 + N[1] * v10 + N[2] * v11 + N[3] * v01


def point_coordinates(mesh: Mesh2D, xi: float, eta: float) -> tuple[np.ndarray, np.ndarray]:
    """Physical coordinates of one reference point in every element."""
    x = mesh.x_coords()[:-1] + 0.5 * mesh.hx * (1.0 + xi)
    y = mesh.y_coords()[:-1] + 0.5 * mesh.hy * (1.0 + eta)
    # element arrays are (ny, nx): j (y) outermost, i (x) innermost, matching ``elements()``
    return np.broadcast_to(x[None, :], (mesh.ny, mesh.nx)), np.broadcast_to(
        y[:, None], (mesh.ny, mesh.nx)
    )


def assemble(mesh: Mesh2D, solution: BubbleSolution) -> tuple[np.ndarray, np.ndarray]:
    """Dense Q4 Laplacian and load vector, integrated with 4x4 Gauss (exact here).

    Every element of a regular mesh has the same 4x4 stiffness, so the local matrix is formed
    once and scattered with ``np.add.at``; the load is integrated per element by vectorised Gauss
    quadrature because ``f`` varies inside an element.
    """
    points, weights = gauss_rule(4)
    det = mesh.hx * mesh.hy / 4.0  # constant Jacobian determinant of a rectangular Q4 element
    local = np.zeros((4, 4))
    for xi, wi in zip(points, weights):
        for eta, wj in zip(points, weights):
            dxi = shape_gradients(xi, eta)
            # dN/dX = diag(2/hx, 2/hy) dN/dxi for the rectangular element
            dN = np.vstack((dxi[0] * (2.0 / mesh.hx), dxi[1] * (2.0 / mesh.hy)))
            local += wi * wj * det * (dN.T @ dN)
    quads = mesh.elements()
    rows = np.repeat(quads, 4, axis=1).ravel()
    cols = np.tile(quads, (1, 4)).ravel()
    values = np.tile(local.ravel(), mesh.nelements)
    K = np.zeros((mesh.nnodes, mesh.nnodes))
    np.add.at(K, (rows, cols), values)

    F = np.zeros(mesh.nnodes)
    for xi, wi in zip(points, weights):
        for eta, wj in zip(points, weights):
            N = shape_functions(xi, eta)
            gx, gy = point_coordinates(mesh, xi, eta)
            f = solution.source(gx, gy)
            weight = wi * wj * det
            for a in range(4):
                np.add.at(F, quads[:, a], (weight * N[a]) * f.ravel())
    return K, F


def solve(mesh: Mesh2D, K: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Solve with homogeneous Dirichlet data (the manufactured field vanishes on the boundary)."""
    interior = mesh.interior_nodes()
    if interior.size == 0:
        raise ValueError("the mesh has no interior node")
    u = np.zeros(mesh.nnodes)
    u[interior] = np.linalg.solve(K[np.ix_(interior, interior)], F[interior])
    return u


# ---------------------------------------------------------------------------
# recovery, indicators and exact errors
# ---------------------------------------------------------------------------
def element_centre_gradient(mesh: Mesh2D, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """FE gradient at the element centre, returned as ``(ny, nx)`` arrays."""
    return gradient_from_nodal(mesh, u, 0.0, 0.0)


def recover_nodal_gradient(mesh: Mesh2D, u: np.ndarray) -> np.ndarray:
    """Area-weighted nodal average of the element-centre FE gradient.

    Returns a nodal vector of length ``2 * nnodes`` (``gx`` then ``gy``).  On a uniform mesh the
    area weights are equal, and each node simply receives the mean of its adjacent elements.
    """
    gx, gy = element_centre_gradient(mesh, u)
    quads = mesh.elements()
    accumulator = np.zeros((mesh.nnodes, 2))
    counts = np.zeros(mesh.nnodes)
    flat_x = gx.ravel()
    flat_y = gy.ravel()
    for a in range(4):
        np.add.at(accumulator[:, 0], quads[:, a], flat_x)
        np.add.at(accumulator[:, 1], quads[:, a], flat_y)
        np.add.at(counts, quads[:, a], 1.0)
    if np.any(counts == 0.0):
        raise ValueError("every node must belong to at least one element")
    return accumulator / counts[:, None]


def indicator_norms(mesh: Mesh2D, u: np.ndarray, *, order: int = GAUSS_INDICATOR) -> np.ndarray:
    """Per-element ``eta_e = ||grad u* - grad u_h||_Omega_e`` (ZZ recovery-based indicator).

    Both gradients are sampled at the same Gauss points of the same rule, so the integrand is the
    squared difference of two bilinear fields -- a quadratic that the 2x2 rule integrates exactly.
    """
    nodal = recover_nodal_gradient(mesh, u)
    points, weights = gauss_rule(order)
    squared = np.zeros((mesh.ny, mesh.nx))
    for xi, wi in zip(points, weights):
        for eta, wj in zip(points, weights):
            fe_x, fe_y = gradient_from_nodal(mesh, u, xi, eta)
            rec_x = interpolate_nodal(mesh, nodal[:, 0], xi, eta)
            rec_y = interpolate_nodal(mesh, nodal[:, 1], xi, eta)
            squared += wi * wj * (mesh.hx * mesh.hy / 4.0) * (
                (rec_x - fe_x) ** 2 + (rec_y - fe_y) ** 2
            )
    if np.any(squared < -1e-14 * max(1.0, float(np.max(squared)))):
        raise ValueError("negative element indicator energy: recovered gradient not admissible")
    return np.sqrt(np.maximum(squared, 0.0)).ravel()


def _gradient_integrals(mesh: Mesh2D, u: np.ndarray, solution: BubbleSolution, *, order: int = GAUSS_ERROR):
    """Energy-norm error, L2 error and recovered-gradient error against the exact field."""
    nodal = recover_nodal_gradient(mesh, u)
    points, weights = gauss_rule(order)
    energy = 0.0
    l2 = 0.0
    recovered = 0.0
    for xi, wi in zip(points, weights):
        for eta, wj in zip(points, weights):
            gx, gy = point_coordinates(mesh, xi, eta)
            exact_x, exact_y = solution.gradient(gx, gy)
            exact_u = solution.u(gx, gy)
            fe_x, fe_y = gradient_from_nodal(mesh, u, xi, eta)
            rec_x = interpolate_nodal(mesh, nodal[:, 0], xi, eta)
            rec_y = interpolate_nodal(mesh, nodal[:, 1], xi, eta)
            fe_u = interpolate_nodal(mesh, u, xi, eta)
            weight = wi * wj * (mesh.hx * mesh.hy / 4.0)
            energy += weight * float(np.sum((fe_x - exact_x) ** 2 + (fe_y - exact_y) ** 2))
            l2 += weight * float(np.sum((fe_u - exact_u) ** 2))
            recovered += weight * float(np.sum((rec_x - exact_x) ** 2 + (rec_y - exact_y) ** 2))
    return math.sqrt(max(energy, 0.0)), math.sqrt(max(l2, 0.0)), math.sqrt(max(recovered, 0.0))


def gradient_error(mesh: Mesh2D, u: np.ndarray, solution: BubbleSolution):
    """``(energy norm error, L2 error, recovered-gradient L2 error)`` against the exact field."""
    return _gradient_integrals(mesh, u, solution)


# ---------------------------------------------------------------------------
# one level and the ladder
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Level2D:
    label: str
    mesh: Mesh2D
    displacement: np.ndarray
    estimated: float
    true_norm: float
    l2_error: float
    effectivity: float
    recovered_error: float
    seconds: float

    @property
    def dofs(self) -> int:
        return int(self.mesh.interior_nodes().size)

    def record(self) -> dict:
        return {
            "label": self.label,
            "divisions": self.mesh.nx,
            "elements": self.mesh.nelements,
            "nodes": self.mesh.nnodes,
            "dofs": self.dofs,
            "h": float(self.mesh.h),
            "true_energy_norm": float(self.true_norm),
            "l2_error": float(self.l2_error),
            "estimated_energy_norm": float(self.estimated),
            "effectivity": float(self.effectivity),
            "recovered_gradient_l2_error": float(self.recovered_error),
            "seconds": float(self.seconds),
        }


def solve_level(mesh: Mesh2D, solution: BubbleSolution, *, label: str = "") -> Level2D:
    start = time.perf_counter()
    K, F = assemble(mesh, solution)
    u = solve(mesh, K, F)
    eta = indicator_norms(mesh, u)
    estimated = float(np.sqrt(np.sum(eta**2)))
    energy, l2, recovered = _gradient_integrals(mesh, u, solution)
    elapsed = time.perf_counter() - start
    return Level2D(
        label=label,
        mesh=mesh,
        displacement=u,
        estimated=estimated,
        true_norm=energy,
        l2_error=l2,
        effectivity=effectivity_index(estimated, energy),
        recovered_error=recovered,
        seconds=elapsed,
    )


#: theoretical slopes against the number of unknowns for Q4 on a smooth solution.
THEORY_RATES = {
    "energy": -0.5,
    "l2": -1.0,
    "recovered_gradient": -1.0,
}


def asymptotic_rate(dofs: Sequence[float], errors: Sequence[float], levels: int = 3) -> float:
    """Slope over the last ``levels`` rungs only (the pre-asymptotic rungs are excluded)."""
    dofs = np.asarray(dofs, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)
    count = min(int(levels), dofs.size)
    if count < 2:
        raise ValueError("two rungs are needed for a rate")
    return convergence_rate(dofs[-count:], errors[-count:])


def uniform_study(solution: BubbleSolution, divisions: Sequence[int]) -> dict:
    if len(divisions) < 2:
        raise ValueError("a rate needs at least two levels")
    levels = [
        solve_level(Mesh2D(int(n), int(n)), solution, label=f"uniform-{int(n)}") for n in divisions
    ]
    dofs = [level.dofs for level in levels]
    return {
        "levels": levels,
        "records": [level.record() for level in levels],
        "energy_rate": convergence_rate(dofs, [level.true_norm for level in levels]),
        "l2_rate": convergence_rate(dofs, [level.l2_error for level in levels]),
        "recovered_gradient_rate": convergence_rate(
            dofs, [level.recovered_error for level in levels]
        ),
        "energy_rate_asymptotic": asymptotic_rate(dofs, [level.true_norm for level in levels]),
        "l2_rate_asymptotic": asymptotic_rate(dofs, [level.l2_error for level in levels]),
        "recovered_gradient_rate_asymptotic": asymptotic_rate(
            dofs, [level.recovered_error for level in levels]
        ),
    }


def study(*, divisions: Sequence[int] = (4, 8, 12, 16, 24, 32)) -> dict:
    """One payload with the order table, the effectivity range and the cost baseline."""
    solution = BubbleSolution()
    uniform = uniform_study(solution, divisions)
    final = uniform["levels"][-1]
    rates = {
        "energy": uniform["energy_rate"],
        "l2": uniform["l2_rate"],
        "recovered_gradient": uniform["recovered_gradient_rate"],
    }
    last_two = {
        "energy": asymptotic_rate([row["dofs"] for row in uniform["records"]],
                                  [row["true_energy_norm"] for row in uniform["records"]], levels=2),
        "l2": asymptotic_rate([row["dofs"] for row in uniform["records"]],
                              [row["l2_error"] for row in uniform["records"]], levels=2),
        "recovered_gradient": asymptotic_rate(
            [row["dofs"] for row in uniform["records"]],
            [row["recovered_gradient_l2_error"] for row in uniform["records"]],
            levels=2,
        ),
    }
    asymptotic = {
        "energy": uniform["energy_rate_asymptotic"],
        "l2": uniform["l2_rate_asymptotic"],
        "recovered_gradient": uniform["recovered_gradient_rate_asymptotic"],
    }
    return {
        "schema": "tensorfem.adaptive_poisson_2d.study/v1",
        "problem": {
            "name": solution.name,
            "domain": "(0, 1)^2",
            "operator": "-laplacian(u) = f",
            "dirichlet": "u = 0 on the full boundary",
            "amplitude": float(solution.amplitude),
            "quadrature": {
                "stiffness_and_load": "4x4 Gauss (exact for this polynomial problem)",
                "error_integrals": f"{GAUSS_ERROR}x{GAUSS_ERROR} Gauss (exact here)",
                "indicators": f"{GAUSS_INDICATOR}x{GAUSS_INDICATOR} Gauss (exact for bilinear differences)",
            },
        },
        "levels": uniform["records"],
        "rates": rates,
        "rates_asymptotic": asymptotic,
        "rates_last_two": last_two,
        "theory_rates": dict(THEORY_RATES),
        "rate_gap_asymptotic": {
            key: float(asymptotic[key] - THEORY_RATES[key]) for key in THEORY_RATES
        },
        "criteria": {
            "final_effectivity": float(final.effectivity),
            "effectivity_range": [
                float(min(level.effectivity for level in uniform["levels"])),
                float(max(level.effectivity for level in uniform["levels"])),
            ],
            "final_energy_norm": float(final.true_norm),
            "final_dofs": int(final.dofs),
            "final_seconds": float(final.seconds),
            "final_effectivity_gap": float(abs(final.effectivity - 1.0)),
            "energy_rate_gap_asymptotic": float(abs(asymptotic["energy"] - THEORY_RATES["energy"])),
            "l2_rate_gap_asymptotic": float(abs(asymptotic["l2"] - THEORY_RATES["l2"])),
            "energy_rate_gap_last_two": float(abs(last_two["energy"] - THEORY_RATES["energy"])),
            "l2_rate_gap_last_two": float(abs(last_two["l2"] - THEORY_RATES["l2"])),
        },
    }