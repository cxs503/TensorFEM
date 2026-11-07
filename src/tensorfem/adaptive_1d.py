"""One-dimensional h-adaptive continuation: analytic convergence orders and a cost baseline.

Why this module exists
----------------------
:mod:`tensorfem.adaptive_zz` closes the ZZ (Zienkiewicz-Zhu) loop for plane elasticity on a
re-entrant corner.  It necessarily measures the *true* error by quadrature, because the exact
field is only available as a Williams eigen-function and the exact error of a discrete solution
is not a closed form there.  That leaves one accounting hole in the indicator direction: the
convergence-order claim of the estimator-driven loop is asserted against ``-lambda/2`` (an
asymptotic bracket) rather than against an *exactly integrable* error.

This module supplies the missing rung of that ladder in 1D, where the discrete energy norm error
of a P1 solution has a **closed form per element**:

    ||e||_E^2 = sum_e int_a^b kappa (u' - c_e)^2 dx,   c_e = (u(b) - u(a)) / (b - a),

and ``int_a^b u'^2 dx`` is analytic for both test solutions below.  The reported error is
therefore exact to round-off, with no quadrature subcells, so the measured slope of
``log||e||_E`` against ``log Ndof`` can be compared with theory without a discretisation of the
error itself.

Two calibrated problems (both on ``Omega = (0, 1)``, ``-kappa u'' = f``, Dirichlet at both ends)

=================================  ==========================  =============================
problem                            exact solution              regularity
=================================  ==========================  =============================
``SinusoidalSolution``             ``u = sin(pi x)``           analytic (smooth reference)
``PowerSolution(alpha=0.6)``       ``u = x**alpha``            ``u in H^(1+alpha-eps)``
=================================  ==========================  =============================

Theory (see ``docs/adaptive-continuation-order-cost.md`` for the derivation and references)

* smooth solution, uniform P1 ladder: ``||e||_E = O(N^-1)`` -- verified to a few percent;
* power ``u = x^alpha``, ``1/2 < alpha < 1``, **uniform** ladder: ``||e||_E =
  O(N^-(alpha-1/2))`` -- for ``alpha = 0.6`` only ``N^-0.1``, i.e. uniform refinement is almost
  stagnant, and the measured exponent matches that closed form;
* **adaptive** ladders: no closed-form rate is claimed.  The optimal-mesh rate quoted in the
  literature presumes an equidistributing mesh whose element sizes vanish faster than any power
  at the singular end; that grading exhausts double precision before it reaches the asymptotic
  regime (see ``artifacts/adaptive_continuation/grading_scan.json``), so the measured adaptive
  exponent is reported and compared with the uniform one, never with an unverifiable number.

Scope and honesty
-----------------
* 1D, P1 (linear) elements, a single diffusion coefficient ``kappa``, Dirichlet data at both
  ends only, isotropic bisection refinement (in 1D bisection is conforming -- no hanging nodes).
* The error indicator is the L2-projection of the element flux, which is the 1D member of the
  ZZ family (nodal averaging is the lumped-mass variant of the same projection).  The
  effectivity index is **measured**, never assumed: for the singular problem it is not expected
  to tend to one, and the module reports it instead of clipping it.
* Adaptivity gain is reported at matched unknowns and at matched error; the uniform reference is
  interpolated in log-log between ladder rungs, and that interpolation is labelled as such.
* Cost is recorded (wall-clock seconds, unknowns, tridiagonal bandwidth, marked counts) but the
  wall-clock is *not* used as an acceptance criterion: it is machine-dependent.  Acceptance uses
  unknowns and the closed-form error.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from .adaptive_zz import convergence_rate, dorfler_mark, effectivity_index

KAPPA = 1.0

#: nodes closer than this to the singular endpoint are excluded from flux comparisons
SINGULAR_GUARD = 1e-9


class Solution1D(Protocol):
    """Analytic solution of ``-kappa u'' = f`` on ``(0, 1)`` with Dirichlet end data."""

    kappa: float
    name: str

    def u(self, x: np.ndarray) -> np.ndarray: ...

    def flux(self, x: np.ndarray) -> np.ndarray: ...

    def source(self, x: np.ndarray) -> np.ndarray: ...

    def flux_singular_at(self) -> float | None: ...

    def derivative_energy_sq(self, a: float, b: float) -> float: ...

    def source_hat(self, a: float, b: float) -> tuple[float, float]: ...

    def boundaries(self) -> tuple[float, float]: ...


# ---------------------------------------------------------------------------
# exact test solutions; their load integrals are analytic (no quadrature error)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PowerSolution:
    """``u = x**alpha`` with ``1/2 < alpha < 1``: an integrable flux singularity at ``x = 0``.

    ``f = -kappa u'' = kappa alpha (1 - alpha) x**(alpha - 2)`` is integrable but unbounded, so
    the load vector is built from the closed form of ``int f N_i dx`` rather than by quadrature:
    a Gauss rule on the element touching the singularity converges only algebraically and would
    pollute the very quantity under verification.
    """

    alpha: float = 0.6
    kappa: float = KAPPA
    name: str = "power"

    def __post_init__(self) -> None:
        if not 0.5 < float(self.alpha) < 1.0:
            raise ValueError("alpha must lie in (0.5, 1) for an H^1 solution with a singular flux")
        if not (math.isfinite(self.kappa) and self.kappa > 0.0):
            raise ValueError("kappa must be finite and positive")

    def u(self, x: np.ndarray) -> np.ndarray:
        return np.power(np.asarray(x, dtype=np.float64), self.alpha)

    def flux(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        with np.errstate(divide="ignore"):
            return self.kappa * self.alpha * np.power(x, self.alpha - 1.0)

    def source(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        with np.errstate(divide="ignore"):
            return self.kappa * self.alpha * (1.0 - self.alpha) * np.power(x, self.alpha - 2.0)

    def flux_singular_at(self) -> float | None:
        return 0.0

    def boundaries(self) -> tuple[float, float]:
        return 0.0, 1.0

    def derivative_energy_sq(self, a: float, b: float) -> float:
        """``int_a^b kappa u'^2 dx`` in closed form."""
        p = 2.0 * self.alpha - 1.0
        coefficient = self.kappa * self.alpha**2 / p
        return float(coefficient * (b**p - a**p))

    def source_hat(self, a: float, b: float) -> tuple[float, float]:
        """``(int_a^b f N_0 dx, int_a^b f N_1 dx)`` for the two hats of ``[a, b]``."""
        if not b > a:
            raise ValueError("element must have positive length")
        gamma = self.alpha - 2.0
        scale = self.kappa * self.alpha * (1.0 - self.alpha) / (b - a)
        if a <= SINGULAR_GUARD:
            # the individual moments diverge at x = 0; the combinations do not
            weighted = b ** (self.alpha) * (1.0 / (self.alpha - 1.0) - 1.0 / self.alpha)
            pure = b ** (self.alpha) / self.alpha
            return float(scale * weighted), float(scale * pure)
        power = (b ** (gamma + 1.0) - a ** (gamma + 1.0)) / (gamma + 1.0)
        higher = (b ** (gamma + 2.0) - a ** (gamma + 2.0)) / (gamma + 2.0)
        return float(scale * (b * power - higher)), float(scale * (higher - a * power))


@dataclass(frozen=True)
class SinusoidalSolution:
    """``u = sin(pi x)``: analytic reference with ``f = kappa pi^2 sin(pi x)``."""

    kappa: float = KAPPA
    name: str = "sinusoidal"

    def __post_init__(self) -> None:
        if not (math.isfinite(self.kappa) and self.kappa > 0.0):
            raise ValueError("kappa must be finite and positive")

    def u(self, x: np.ndarray) -> np.ndarray:
        return np.sin(math.pi * np.asarray(x, dtype=np.float64))

    def flux(self, x: np.ndarray) -> np.ndarray:
        return self.kappa * math.pi * np.cos(math.pi * np.asarray(x, dtype=np.float64))

    def source(self, x: np.ndarray) -> np.ndarray:
        return self.kappa * math.pi**2 * np.sin(math.pi * np.asarray(x, dtype=np.float64))

    def flux_singular_at(self) -> float | None:
        return None

    def boundaries(self) -> tuple[float, float]:
        return 0.0, 1.0

    def derivative_energy_sq(self, a: float, b: float) -> float:
        def primitive(x: float) -> float:
            return 0.5 * x + math.sin(2.0 * math.pi * x) / (4.0 * math.pi)

        return float(self.kappa * math.pi**2 * (primitive(b) - primitive(a)))

    def source_hat(self, a: float, b: float) -> tuple[float, float]:
        if not b > a:
            raise ValueError("element must have positive length")

        def sine(x: float) -> float:
            return -math.cos(math.pi * x) / math.pi

        def x_sine(x: float) -> float:
            return -x * math.cos(math.pi * x) / math.pi + math.sin(math.pi * x) / math.pi**2

        scale = self.kappa * math.pi**2 / (b - a)
        first = b * (sine(b) - sine(a)) - (x_sine(b) - x_sine(a))
        second = (x_sine(b) - x_sine(a)) - a * (sine(b) - sine(a))
        return float(scale * first), float(scale * second)


# ---------------------------------------------------------------------------
# mesh, assembly and the closed-form error
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Mesh1D:
    """Sorted node coordinates; every neighbouring pair is a P1 element."""

    nodes: np.ndarray

    def __post_init__(self) -> None:
        nodes = np.asarray(self.nodes, dtype=np.float64).ravel()
        if nodes.size < 2:
            raise ValueError("a 1D mesh needs at least two nodes")
        if not np.all(np.isfinite(nodes)):
            raise ValueError("node coordinates must be finite")
        if np.any(np.diff(nodes) <= 0.0):
            raise ValueError("node coordinates must be strictly increasing")
        object.__setattr__(self, "nodes", nodes)

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.size)

    @property
    def n_elements(self) -> int:
        return int(self.nodes.size - 1)

    def lengths(self) -> np.ndarray:
        return np.diff(self.nodes)

    def midpoints(self) -> np.ndarray:
        return 0.5 * (self.nodes[:-1] + self.nodes[1:])

    def dofs(self) -> int:
        """Free unknowns: one per interior node (both end nodes are Dirichlet)."""
        return max(self.n_elements - 1, 0)


def uniform_mesh(intervals: int, length: float = 1.0) -> Mesh1D:
    if intervals < 1:
        raise ValueError("intervals must be >= 1")
    return Mesh1D(np.linspace(0.0, length, intervals + 1))


def refine(mesh: Mesh1D, marked) -> Mesh1D:
    """Bisect every marked element; in 1D this stays conforming (no hanging nodes)."""
    flagged = np.asarray(marked, dtype=bool).ravel()
    if flagged.size != mesh.n_elements:
        raise ValueError("marking vector must have one entry per element")
    if not flagged.any():
        raise ValueError("nothing marked: refinement would return the same mesh")
    midpoints = mesh.midpoints()[flagged]
    nodes = np.union1d(mesh.nodes, midpoints)
    return Mesh1D(nodes)


def assemble(mesh: Mesh1D, solution: Solution1D) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """P1 stiffness (tridiagonal) and consistent load from the analytic ``source_hat``.

    Returns ``(lower, diag, upper, rhs)`` with the Dirichlet end values already condensed into
    ``rhs`` for the free (interior) unknowns.  ``lower``/``upper`` have length ``n_free - 1``.
    """
    kappa = float(solution.kappa)
    n = mesh.n_nodes
    diag = np.zeros(n)
    off = np.zeros(n - 1)
    rhs = np.zeros(n)
    lengths = mesh.lengths()
    for e in range(mesh.n_elements):
        h = float(lengths[e])
        k = kappa / h
        diag[e] += k
        diag[e + 1] += k
        off[e] += -k
        first, second = solution.source_hat(float(mesh.nodes[e]), float(mesh.nodes[e + 1]))
        rhs[e] += first
        rhs[e + 1] += second
    left, right = solution.boundaries()
    g_left = float(solution.u(np.array([left]))[0])
    g_right = float(solution.u(np.array([right]))[0])
    # condense the two Dirichlet ends (nodes 0 and n-1) onto the interior unknowns
    rhs[1] -= off[0] * g_left
    rhs[n - 2] -= off[n - 2] * g_right
    interior_diag = diag[1:-1].copy()
    interior_off = off[1 : n - 2].copy() if n > 2 else np.zeros(0)
    return interior_off.copy(), interior_diag, interior_off, rhs[1:-1]


def _solve_tridiagonal(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Thomas algorithm; ``lower[i]`` multiplies ``x[i]`` in row ``i+1``."""
    n = diag.size
    if n == 0:
        return np.zeros(0)
    if lower.size != n - 1 or upper.size != n - 1:
        raise ValueError("tridiagonal arrays are inconsistent with the diagonal length")
    c = upper.copy()
    d = rhs.copy()
    b = diag.copy()
    for i in range(1, n):
        factor = lower[i - 1] / b[i - 1]
        b[i] -= factor * c[i - 1]
        d[i] -= factor * d[i - 1]
    x = np.zeros(n)
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
    return x


def solve(mesh: Mesh1D, solution: Solution1D) -> np.ndarray:
    """P1 displacement with the exact analytic Dirichlet data at both ends."""
    lower, diag, upper, rhs = assemble(mesh, solution)
    interior = _solve_tridiagonal(lower, diag, upper, rhs)
    left, right = solution.boundaries()
    full = np.empty(mesh.n_nodes)
    full[0] = float(solution.u(np.array([left]))[0])
    full[-1] = float(solution.u(np.array([right]))[0])
    full[1:-1] = interior
    return full


def element_flux(mesh: Mesh1D, u: np.ndarray, kappa: float = KAPPA) -> np.ndarray:
    """Piecewise-constant P1 flux ``sigma_h = kappa du_h/dx`` per element."""
    u = np.asarray(u, dtype=np.float64).ravel()
    if u.size != mesh.n_nodes:
        raise ValueError("displacement vector does not match the mesh")
    return kappa * np.diff(u) / mesh.lengths()


def mass_matrix_tridiagonal(mesh: Mesh1D) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """P1 consistent mass matrix (``lower``, ``diag``, ``upper``) -- no boundary constraints."""
    n = mesh.n_nodes
    diag = np.zeros(n)
    off = np.zeros(n - 1)
    for e, h in enumerate(mesh.lengths()):
        diag[e] += h / 3.0
        diag[e + 1] += h / 3.0
        off[e] += h / 6.0
    return off.copy(), diag, off


def recover_nodal_flux(mesh: Mesh1D, sigma_element: np.ndarray) -> np.ndarray:
    """L2 projection of the piecewise-constant flux onto the P1 space.

    This is the 1D member of the ZZ recovery family: minimise ``int (sigma* - sigma_h)^2`` over
    the nodal values.  No Dirichlet constraint is imposed (the flux is not prescribed anywhere),
    so the consistent mass matrix -- which is SPD -- is solved as is.
    """
    sigma = np.asarray(sigma_element, dtype=np.float64).ravel()
    if sigma.size != mesh.n_elements:
        raise ValueError("one flux value per element is required")
    if not np.all(np.isfinite(sigma)):
        raise ValueError("element flux contains non-finite entries")
    lower, diag, upper = mass_matrix_tridiagonal(mesh)
    rhs = np.zeros(mesh.n_nodes)
    for e, h in enumerate(mesh.lengths()):
        rhs[e] += 0.5 * h * sigma[e]
        rhs[e + 1] += 0.5 * h * sigma[e]
    return _solve_tridiagonal(lower, diag, upper, rhs)


def element_indicators(mesh: Mesh1D, sigma_element: np.ndarray, sigma_nodal: np.ndarray, kappa: float = KAPPA) -> np.ndarray:
    """``eta_e = ||sigma_h - sigma*||_Omega_e / sqrt(kappa)`` with the integral in closed form.

    The recovered flux is linear on the element, so ``int (sigma* - sigma_h)^2 = h (d_a^2 + d_a
    d_b + d_b^2) / 3`` exactly; the indicator is therefore not a quadrature of the estimator.
    """
    sigma = np.asarray(sigma_element, dtype=np.float64).ravel()
    if sigma.size != mesh.n_elements:
        raise ValueError("one flux value per element is required")
    if not np.all(np.isfinite(sigma)):
        raise ValueError("element flux contains non-finite entries")
    nodal = np.asarray(sigma_nodal, dtype=np.float64).ravel()
    if nodal.size != mesh.n_nodes:
        raise ValueError("nodal flux vector does not match the mesh")
    delta_a = nodal[:-1] - sigma
    delta_b = nodal[1:] - sigma
    squared = mesh.lengths() * (delta_a**2 + delta_a * delta_b + delta_b**2) / 3.0
    if np.any(squared < -1e-14 * max(1.0, float(np.max(np.abs(squared))))):
        raise ValueError("negative element indicator energy: recovered flux not admissible")
    return np.sqrt(np.maximum(squared, 0.0) / float(kappa))


def exact_error_energy_sq(mesh: Mesh1D, u: np.ndarray, solution: Solution1D) -> float:
    """``sum_e int_a^b kappa (u' - c_e)^2 dx`` in closed form (no error quadrature)."""
    u = np.asarray(u, dtype=np.float64).ravel()
    if u.size != mesh.n_nodes:
        raise ValueError("displacement vector does not match the mesh")
    total = 0.0
    for e, h in enumerate(mesh.lengths()):
        a = float(mesh.nodes[e])
        b = float(mesh.nodes[e + 1])
        # int_a^b (u' - c_e)^2 dx with c_e the *discrete* gradient: u_h is the Galerkin
        # solution, not the interpolant, so int u' dx = u(b) - u(a) differs from c_e h and the
        # three-term expansion is required.  The terms are O(energy) while the result is the
        # (much smaller) element error, so a term-relative tolerance absorbs the cancellation on
        # elements that are already very well resolved instead of reporting a negative energy.
        c = (u[e + 1] - u[e]) / h
        drop = float(solution.u(np.array([b]))[0] - solution.u(np.array([a]))[0])
        energy = solution.derivative_energy_sq(a, b)
        cross = 2.0 * solution.kappa * c * drop
        discrete = solution.kappa * c * c * h
        element = energy - cross + discrete
        tolerance = 1e-12 * (abs(energy) + abs(cross) + abs(discrete))
        if element < -tolerance:
            raise ValueError(
                f"negative element error energy on [{a:.3e}, {b:.3e}]: "
                "the analytic derivative energy and the discrete gradient are inconsistent"
            )
        total += max(element, 0.0)
    return float(total)


# ---------------------------------------------------------------------------
# one level of the study
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Level1D:
    """Everything measured on one mesh; arrays stay out of the serialised record."""

    label: str
    mesh: Mesh1D
    displacement: np.ndarray
    indicators: np.ndarray
    recovered_flux: np.ndarray
    estimated: float
    true_norm: float
    effectivity: float
    recovery_error: float
    seconds: float

    @property
    def dofs(self) -> int:
        return self.mesh.dofs()

    @property
    def nnz(self) -> int:
        return max(3 * self.mesh.n_nodes - 4, 0)

    def record(self) -> dict:
        lengths = self.mesh.lengths()
        return {
            "label": self.label,
            "elements": self.mesh.n_elements,
            "nodes": self.mesh.n_nodes,
            "dofs": self.dofs,
            "nnz": self.nnz,
            "min_element": float(lengths.min()),
            "max_element": float(lengths.max()),
            "true_energy_norm": float(self.true_norm),
            "estimated_energy_norm": float(self.estimated),
            "effectivity": float(self.effectivity),
            "recovered_flux_relative_error": float(self.recovery_error),
            "seconds": float(self.seconds),
        }


def solve_level(mesh: Mesh1D, solution: Solution1D, *, label: str = "") -> Level1D:
    """Assemble, solve, recover, estimate and measure the exact error on one mesh."""
    start = time.perf_counter()
    displacement = solve(mesh, solution)
    sigma = element_flux(mesh, displacement, solution.kappa)
    nodal = recover_nodal_flux(mesh, sigma)
    eta = element_indicators(mesh, sigma, nodal, solution.kappa)
    elapsed = time.perf_counter() - start
    estimated = float(np.sqrt(np.sum(eta**2)))
    true_norm = math.sqrt(exact_error_energy_sq(mesh, displacement, solution))
    singular = solution.flux_singular_at()
    guard = SINGULAR_GUARD if singular is not None else 0.0
    usable = mesh.nodes > max(guard, SINGULAR_GUARD)
    exact_flux = solution.flux(mesh.nodes[usable])
    difference = nodal[usable] - exact_flux
    recovery_error = float(np.linalg.norm(difference) / np.linalg.norm(exact_flux))
    return Level1D(
        label=label,
        mesh=mesh,
        displacement=displacement,
        indicators=eta,
        recovered_flux=nodal,
        estimated=estimated,
        true_norm=true_norm,
        effectivity=effectivity_index(estimated, true_norm),
        recovery_error=recovery_error,
        seconds=elapsed,
    )


# ---------------------------------------------------------------------------
# ladders, rates and the cost baseline
# ---------------------------------------------------------------------------
def uniform_study(solution: Solution1D, intervals: Sequence[int]) -> dict:
    if len(intervals) < 2:
        raise ValueError("a rate needs at least two uniform levels")
    levels = [
        solve_level(uniform_mesh(int(n)), solution, label=f"uniform-{int(n)}") for n in intervals
    ]
    return {
        "levels": levels,
        "records": [level.record() for level in levels],
        "rate": convergence_rate(
            [level.dofs for level in levels], [level.true_norm for level in levels]
        ),
    }


def graded_mesh(intervals: int, exponent: float = 1.0, length: float = 1.0) -> Mesh1D:
    """Algebraic grading ``x_k = length * (k / N) ** exponent`` that clusters at ``x = 0``.

    For ``u = x**alpha`` the energy-optimal 1D mesh equidistributes ``h^2 |u''|``, which in this
    family means ``exponent = 2 / (2 alpha - 1)`` (see :func:`optimal_grading_exponent`).  The
    graded ladder is the *reference* against which the indicator-driven loop is graded: it is
    built from the analytic regularity, not from the estimator, so it cannot inherit the
    estimator's blind spots.
    """
    if intervals < 1:
        raise ValueError("intervals must be >= 1")
    if not (math.isfinite(exponent) and exponent > 0.0):
        raise ValueError("grading exponent must be finite and positive")
    k = np.arange(intervals + 1, dtype=np.float64)
    return Mesh1D(length * (k / intervals) ** exponent)


def optimal_grading_exponent(solution: Solution1D) -> float:
    """Analytic grading exponent of the energy-optimal mesh for ``solution``."""
    if isinstance(solution, PowerSolution):
        return 2.0 / (2.0 * solution.alpha - 1.0)
    if isinstance(solution, SinusoidalSolution):
        return 1.0
    raise TypeError(f"no grading exponent is registered for {type(solution).__name__}")


def graded_study(
    solution: Solution1D, intervals: Sequence[int], exponent: float | None = None
) -> dict:
    """Reference ladder on analytically graded meshes (no indicator involved)."""
    gamma = optimal_grading_exponent(solution) if exponent is None else float(exponent)
    levels = [
        solve_level(graded_mesh(int(n), gamma), solution, label=f"graded-{int(n)}")
        for n in intervals
    ]
    return {
        "exponent": gamma,
        "levels": levels,
        "records": [level.record() for level in levels],
        "rate": convergence_rate(
            [level.dofs for level in levels], [level.true_norm for level in levels]
        ),
    }


def adaptive_study(
    solution: Solution1D, *, intervals: int = 8, rounds: int = 10, fraction: float = 0.5
) -> dict:
    if rounds < 1:
        raise ValueError("at least one refinement round is required")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("marking fraction must lie in (0, 1]")
    mesh = uniform_mesh(int(intervals))
    levels = [solve_level(mesh, solution, label="start")]
    markings: list[dict] = []
    for step in range(1, rounds + 1):
        eta = levels[-1].indicators
        marked = dorfler_mark(eta, fraction)
        energy = eta**2
        markings.append(
            {
                "round": step,
                "marked_elements": int(marked.sum()),
                "marked_fraction": float(marked.sum() / eta.size),
                "marked_energy_fraction": float(energy[marked].sum() / energy.sum()),
                "min_marked_element": float(levels[-1].mesh.lengths()[marked].min()),
            }
        )
        mesh = refine(mesh, marked)
        levels.append(solve_level(mesh, solution, label=f"adaptive-{step}"))
    return {
        "levels": levels,
        "records": [level.record() for level in levels],
        "markings": markings,
        "rate": convergence_rate(
            [level.dofs for level in levels], [level.true_norm for level in levels]
        ),
    }


def theory_rate(solution: Solution1D) -> float:
    """Closed-form slope of ``log||e||_E`` against ``log Ndof`` for a **uniform** P1 ladder.

    * smooth solution: ``-1`` (energy error ``O(h)`` with ``N ~ 1/h`` in 1D);
    * ``u = x**alpha``: ``-(alpha - 1/2)``.  The derivation: the element adjacent to the corner
      carries ``int_0^h (u' - h^(alpha-1))^2 dx ~ h^(2 alpha - 1)``, so ``||e||_E ~ h^(alpha - 1/2)``
      and ``N = 1/h``.  For ``alpha = 0.6`` this is only ``-0.1``: the uniform ladder is almost
      stagnant, which is exactly what the singular test measures.

    **No value is returned for an adaptive ladder.**  The literature's optimal-mesh rate
    (``O(N^-p)``) presumes an equidistributing mesh whose element sizes vanish faster than any
    power at the singular end; that grading drives the tridiagonal condition number past double
    precision (measured in ``artifacts/adaptive_continuation/grading_scan.json``), so this module
    reports the measured adaptive rate and compares it with the uniform rate instead of quoting a
    number it cannot verify.
    """
    if isinstance(solution, PowerSolution):
        return -0.5 * (2.0 * solution.alpha - 1.0)
    if isinstance(solution, SinusoidalSolution):
        return -1.0
    raise TypeError(f"no theoretical rate is registered for {type(solution).__name__}")


def matched_dof_row(adaptive_level: Level1D, uniform: dict) -> dict:
    """Compare an adaptive level with the cheapest uniform level holding at least its unknowns."""
    records = uniform["records"]
    dofs = np.array([row["dofs"] for row in records], dtype=float)
    errors = np.array([row["true_energy_norm"] for row in records], dtype=float)
    candidates = np.nonzero(dofs >= adaptive_level.dofs)[0]
    row = {
        "label": adaptive_level.label,
        "adaptive_dofs": int(adaptive_level.dofs),
        "adaptive_error": float(adaptive_level.true_norm),
        "adaptive_seconds": float(adaptive_level.seconds),
    }
    if candidates.size == 0:
        row["uniform_dofs"] = None
        return row
    index = int(candidates[0])
    row.update(
        {
            "uniform_dofs": int(dofs[index]),
            "uniform_error": float(errors[index]),
            "uniform_seconds": float(records[index]["seconds"]),
            "uniform_over_adaptive_error_ratio": float(errors[index] / adaptive_level.true_norm),
            "uniform_over_adaptive_dof_ratio": float(dofs[index] / adaptive_level.dofs),
        }
    )
    return row


def uniform_dofs_for_error(uniform: dict, error: float) -> float | None:
    """Log-log interpolation of the uniform ladder at a given error (labelled as interpolation)."""
    records = uniform["records"]
    dofs = np.array([row["dofs"] for row in records], dtype=float)
    errors = np.array([row["true_energy_norm"] for row in records], dtype=float)
    order = np.argsort(errors)
    errors = errors[order]
    dofs = dofs[order]
    if not (errors.min() <= error <= errors.max()):
        return None
    return float(np.exp(np.interp(np.log(error), np.log(errors), np.log(dofs))))


def cost_baseline(adaptive: dict, uniform: dict) -> dict:
    """Cost comparison: unknowns and measured wall-clock at matched error.

    The baseline is the *single* uniform solve that reaches the adaptive error, not the ladder
    that was used to interpolate it; the ladder seconds are reported because they are measured,
    but only the unknown counts are used as an acceptance quantity.
    """
    final = adaptive["levels"][-1]
    target = float(final.true_norm)
    interpolated = uniform_dofs_for_error(uniform, target)
    cumulative_seconds = float(sum(level.seconds for level in adaptive["levels"]))
    cumulative_dofs = int(sum(level.dofs for level in adaptive["levels"]))
    rows = [matched_dof_row(level, uniform) for level in adaptive["levels"]]
    comparison = [row for row in rows if row.get("uniform_dofs") is not None]
    uniform_errors = [row["true_energy_norm"] for row in uniform["records"]]
    uniform_max_dofs = int(max(row["dofs"] for row in uniform["records"]))
    # if the adaptive error lies below the whole uniform ladder, interpolation is not defined:
    # report the conservative bound that the ladder itself provides instead of extrapolating
    saving_lower_bound = (
        float(uniform_max_dofs / final.dofs)
        if interpolated is None and target < min(uniform_errors)
        else None
    )
    return {
        "target_error": target,
        "adaptive_dofs": int(final.dofs),
        "adaptive_seconds": float(final.seconds),
        "adaptive_cumulative_seconds": cumulative_seconds,
        "adaptive_cumulative_dofs": cumulative_dofs,
        "uniform_dofs_at_equal_error": interpolated,
        "dof_saving_at_equal_error": (
            float(interpolated / final.dofs) if interpolated is not None else None
        ),
        "dof_saving_lower_bound": saving_lower_bound,
        "uniform_ladder_max_dofs": uniform_max_dofs,
        "uniform_ladder_min_error": float(min(uniform_errors)),
        "matched_dof_table": rows,
        "final_matched_row": comparison[-1] if comparison else None,
        "time_per_dof_adaptive": float(final.seconds / max(final.dofs, 1)),
        "time_per_dof_uniform": (
            float(uniform["records"][-1]["seconds"] / max(uniform["records"][-1]["dofs"], 1))
        ),
    }


def study(
    *,
    alpha: float = 0.6,
    smooth_intervals: Sequence[int] = (8, 16, 32, 64, 128, 256, 512),
    singular_intervals: Sequence[int] = (8, 16, 32, 64, 128, 256, 512, 1024),
    start_intervals: int = 8,
    rounds: int = 10,
    fraction: float = 0.5,
) -> dict:
    """Both problems, both ladders, the rate table and the cost baseline in one payload."""
    payload: dict = {"schema": "tensorfem.adaptive_1d.study/v1", "problems": {}}
    for solution, uniform_intervals in (
        (SinusoidalSolution(), smooth_intervals),
        (PowerSolution(alpha=alpha), singular_intervals),
    ):
        uniform = uniform_study(solution, uniform_intervals)
        adaptive = adaptive_study(
            solution, intervals=start_intervals, rounds=rounds, fraction=fraction
        )
        cost = cost_baseline(adaptive, uniform)
        payload["problems"][solution.name] = {
            "parameters": {
                "kappa": float(solution.kappa),
                "dirichlet": "exact analytic trace at x = 0 and x = 1",
                "elements": "P1 (linear), isotropic bisection",
                "indicator": "L2-projection flux recovery (ZZ family), Doerfler marking",
                "alpha": getattr(solution, "alpha", None),
            },
            "uniform": {
                "intervals": [int(n) for n in uniform_intervals],
                "records": uniform["records"],
                "rate": uniform["rate"],
                "theory_rate": theory_rate(solution),
            },
            "adaptive": {
                "records": adaptive["records"],
                "markings": adaptive["markings"],
                "rate": adaptive["rate"],
                "theory_rate": None,
            },
            "cost": cost,
            "criteria": {
                "uniform_rate_gap_to_theory": abs(uniform["rate"] - theory_rate(solution)),
                "final_effectivity": float(adaptive["levels"][-1].effectivity),
                "final_adaptive_error": float(adaptive["levels"][-1].true_norm),
                "final_adaptive_dofs": int(adaptive["levels"][-1].dofs),
                "dof_saving_at_equal_error": cost["dof_saving_at_equal_error"],
                "dof_saving_lower_bound": cost["dof_saving_lower_bound"],
                "adaptive_over_uniform_rate_ratio": (
                    adaptive["rate"] / uniform["rate"] if uniform["rate"] != 0.0 else None
                ),
            },
        }
    return payload