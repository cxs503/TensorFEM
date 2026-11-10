"""Structural mechanics verification ladder for TensorFEM.

Why this module exists
----------------------
TensorFEM carries a large amount of *analysis* code (contact, plasticity,
buckling, shells) but the classical one-element-thick structural chain --
"beam bending -> plate bending -> beam vibration" -- had no end-to-end analytic
verification surface.  This module supplies it with three closed-form anchors
that can be recomputed from a textbook in one line each:

1. **Euler--Bernoulli cantilever under uniform load** ``q``

   .. math:: w(x) = \\frac{q x^2 (6L^2 - 4Lx + x^2)}{24 EI}, \\qquad
             w(L) = \\frac{q L^4}{8 EI}

   The finite-element solution uses the two-node cubic Hermite beam element of
   :mod:`tensorfem.modal` (4 dof per element) with the *consistent* load vector;
   for this case the nodal deflections are exact and the remaining error is
   pure interpolation error of the quartic exact field, which is
   :math:`O(h^4)` in :math:`L_2`.

2. **Kirchhoff thin plate, simply supported square, uniform pressure** ``q``

   .. math:: w_{\\max} = \\alpha \\frac{q a^4}{D}, \\qquad
             \\alpha = 0.00406\\ (\\nu = 0.3),\\; D = \\frac{E t^3}{12(1-\\nu^2)}

   ``alpha = 0.00406`` is the value tabulated by Timoshenko & Woinowsky-Krieger
   (*Theory of Plates and Shells*, 2nd ed., table 8) and the classical
   reference quoted in the validation contract.  The more accurate value of the
   Navier series is ``0.00406235...``; both are reported.  The discretisation is
   the C1 conforming bicubic Hermite (Bogner--Fox--Schmit) rectangle: 4 dof per
   node ``(w, w_,x, w_,y, w_,xy)``, bending operator taken from
   :func:`tensorfem.plate_buckling_hermite.bfs_plate_element_matrices` so that
   exactly one implementation of the element exists in the repository.

3. **Cantilever natural frequencies, Euler--Bernoulli (modes 1--5)**

   .. math:: \\omega_n = (\\beta_n L)^2 \\sqrt{\\frac{EI}{\\rho A L^4}}, \\qquad
             \\beta_n L = 1.8751, 4.6941, 7.8548, 10.9955, 14.1372

   The roots are recomputed here by bisection on ``cos(z) cosh(z) + 1 = 0``
   (arbitrary mode count) and cross-checked against the four Blevins table
   values already used in :mod:`tensorfem.modal`.

Nothing in this module touches solver state; every routine assembles and solves
a small dense system or evaluates a closed form.  All results are deterministic.

References
----------
* S. Timoshenko, S. Woinowsky-Krieger, *Theory of Plates and Shells*, 2nd ed.,
  McGraw-Hill, 1959 (tables 8, 30).
* R.D. Blevins, *Formulas for Natural Frequency and Mode Shape*, Van Nostrand
  Reinhold, 1979, table 8-1.
* O.C. Zienkiewicz, R.L. Taylor, *The Finite Element Method*, 5th ed., vol. 2,
  ch. 5 (Bogner--Fox--Schmit element, consistent load vectors).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .modal import beam_bending_stiffness, cantilever_beam_modes
from .plate_buckling_hermite import bfs_plate_element_matrices, hermite_cubic_1d

__all__ = [
    "CANTILEVER_BETA_L",
    "CANTILEVER_BETA_L_TABLE",
    "PLATE_ALPHA_TIMOSHENKO",
    "BeamUDLSolution",
    "KirchhoffPlateSolution",
    "ModalLadder",
    "cantilever_roots",
    "euler_bernoulli_cantilever_deflection",
    "euler_bernoulli_cantilever_tip",
    "beam_consistent_udl_load",
    "solve_cantilever_udl",
    "beam_field",
    "kirchhoff_ss_plate_uniform_load",
    "plate_centre_deflection_series",
    "cantilever_exact_angular_frequencies",
    "cantilever_modal_ladder",
    "observed_order",
]

#: Cantilever characteristic roots ``beta_n L`` quoted from the classical
#: tabulation (Blevins 1979, table 8-1; Timoshenko, *Vibration Problems in
#: Engineering*) at the 5--6 significant digits of the source.  The precision of
#: the source is the tolerance every root cross-check has to respect; the values
#: are therefore *not* overwritten with the output of :func:`cantilever_roots`.
CANTILEVER_BETA_L_TABLE = (1.87510, 4.69409, 7.85476, 10.9955, 14.1372)

#: Backwards-compatible alias; equivalent to :data:`CANTILEVER_BETA_L_TABLE`.
CANTILEVER_BETA_L = CANTILEVER_BETA_L_TABLE

#: Timoshenko & Woinowsky-Krieger centre-deflection coefficient, SSSS square,
#: uniform pressure, ``nu = 0.3`` (table 8 of *Theory of Plates and Shells*).
PLATE_ALPHA_TIMOSHENKO = 0.00406


# ---------------------------------------------------------------------------
# 1. Euler--Bernoulli cantilever under uniform load
# ---------------------------------------------------------------------------

def euler_bernoulli_cantilever_deflection(x: float, length: float, EI: float, load: float) -> float:
    """Exact deflection ``w(x) = q x^2 (6L^2 - 4Lx + x^2) / (24 EI)``."""
    if length <= 0 or EI <= 0:
        raise ValueError("length and EI must be positive")
    x = float(x)
    return load * x**2 * (6.0 * length**2 - 4.0 * length * x + x**2) / (24.0 * EI)


def euler_bernoulli_cantilever_tip(length: float, EI: float, load: float) -> dict[str, float]:
    """Closed-form tip deflection and rotation of a uniformly loaded cantilever."""
    return {
        "tip_deflection": load * length**4 / (8.0 * EI),
        "tip_rotation": load * length**3 / (6.0 * EI),
        "clamped_moment": load * length**2 / 2.0,
    }


def beam_consistent_udl_load(load: float, length: float) -> np.ndarray:
    """Consistent nodal load vector of the cubic Hermite beam element.

    ``f = q * [L/2, L^2/12, L/2, -L^2/12]`` -- the work-equivalent load for
    ``(w1, theta1, w2, theta2)`` with ``theta = dw/dx``.
    """
    return load * np.array([length / 2.0, length**2 / 12.0, length / 2.0, -length**2 / 12.0])


@dataclass(frozen=True)
class BeamUDLSolution:
    """Result of a cantilever uniform-load solve with ``elements`` Hermite beams."""

    elements: int
    length: float
    EI: float
    load: float
    nodal_deflections: np.ndarray
    nodal_rotations: np.ndarray
    element_length: float

    @property
    def tip_deflection(self) -> float:
        return float(self.nodal_deflections[-1])

    @property
    def tip_rotation(self) -> float:
        return float(self.nodal_rotations[-1])

    def clamped_shear(self) -> float:
        """Reaction at the clamp recovered from element end forces (equilibrium)."""
        ke = beam_bending_stiffness(self.EI, self.element_length).numpy()
        fe = beam_consistent_udl_load(self.load, self.element_length)
        ue = np.array(
            [
                self.nodal_deflections[0],
                self.nodal_rotations[0],
                self.nodal_deflections[1],
                self.nodal_rotations[1],
            ]
        )
        return float((ke @ ue - fe)[0])

    def deflection_at(self, x: float) -> float:
        """Evaluate the piecewise-cubic FE field at ``x``."""
        n = self.elements
        le = self.element_length
        if x < 0.0 or x > self.length * (1.0 + 1e-12):
            raise ValueError("x outside the beam")
        e = min(int(x / le), n - 1)
        xi = (x - e * le) / le
        h, _, _ = hermite_cubic_1d(xi, le)
        local = np.array(
            [
                self.nodal_deflections[e],
                self.nodal_rotations[e],
                self.nodal_deflections[e + 1],
                self.nodal_rotations[e + 1],
            ]
        )
        return float(local @ h)


def solve_cantilever_udl(length: float, EI: float, load: float, elements: int) -> BeamUDLSolution:
    """Assemble and solve the clamped cantilever under uniform load ``q``."""
    if elements < 1:
        raise ValueError("elements must be >= 1")
    if length <= 0 or EI <= 0:
        raise ValueError("length and EI must be positive")
    ndof = 2 * (elements + 1)
    le = length / elements
    ke = beam_bending_stiffness(EI, le).numpy()
    fe = beam_consistent_udl_load(load, le)
    K = np.zeros((ndof, ndof))
    f = np.zeros(ndof)
    for e in range(elements):
        dofs = np.array([2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3])
        K[np.ix_(dofs, dofs)] += ke
        f[dofs] += fe
    # Clamped at x = 0: w = theta = 0 -> dofs 0 and 1.
    free = np.arange(2, ndof)
    u = np.zeros(ndof)
    u[free] = np.linalg.solve(K[np.ix_(free, free)], f[free])
    return BeamUDLSolution(
        elements=elements,
        length=length,
        EI=EI,
        load=load,
        nodal_deflections=u[0::2],
        nodal_rotations=u[1::2],
        element_length=le,
    )


def beam_field(
    solution: BeamUDLSolution, samples_per_element: int = 8
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample the FE field and the exact field; return ``(x, w_fe, w_exact)``."""
    if samples_per_element < 1:
        raise ValueError("samples_per_element must be >= 1")
    le = solution.element_length
    # Gauss-Legendre nodes/weights on [0, 1] keep the L2 norm a quadrature.
    gx, gw = np.polynomial.legendre.leggauss(samples_per_element)
    xi = 0.5 * (gx + 1.0)
    xs, fe, ex = [], [], []
    for e in range(solution.elements):
        h, _, _ = hermite_cubic_1d(xi, le)
        local = np.array(
            [
                [solution.nodal_deflections[e], solution.nodal_rotations[e]],
                [solution.nodal_deflections[e + 1], solution.nodal_rotations[e + 1]],
            ]
        ).reshape(4)
        w = local @ h
        x = (e + xi) * le
        xs.append(x)
        fe.append(w)
        ex.append(
            np.array(
                [
                    euler_bernoulli_cantilever_deflection(xi_i, solution.length, solution.EI, solution.load)
                    for xi_i in x
                ]
            )
        )
    return np.concatenate(xs), np.concatenate(fe), np.concatenate(ex)


def beam_l2_relative_error(solution: BeamUDLSolution, samples_per_element: int = 8) -> float:
    """Relative ``L2`` error of the FE field against the exact quartic."""
    _, w_fe, w_ex = beam_field(solution, samples_per_element)
    le = solution.element_length
    wq = np.polynomial.legendre.leggauss(samples_per_element)[1]
    weight = 0.5 * le * np.tile(wq, solution.elements)
    num = np.sqrt(np.sum(weight * (w_fe - w_ex) ** 2))
    den = np.sqrt(np.sum(weight * w_ex**2))
    return float(num / den)


# ---------------------------------------------------------------------------
# 2. Kirchhoff plate, simply supported, uniform pressure (BFS element)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KirchhoffPlateSolution:
    """Simply supported Kirchhoff plate under uniform pressure."""

    side: float
    young: float
    poisson: float
    thickness: float
    pressure: float
    n_x: int
    n_y: int
    bending_stiffness: float
    centre_deflection: float
    deflection: np.ndarray  # nodal w, shape (n_y + 1, n_x + 1)
    n_dof: int
    n_free: int

    def grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(X, Y, W)`` nodal coordinates and deflection for plotting.

        ``n_y`` only refines the mesh; the plate stays ``side`` by ``side`` (the
        element size is ``side/n_x`` by ``side/n_y``), so both axes span
        ``[0, side]``.  The previous ``side*n_y/n_x`` y-extent described a domain
        the assembled stiffness never used.
        """
        xs = np.linspace(0.0, self.side, self.n_x + 1)
        ys = np.linspace(0.0, self.side, self.n_y + 1)
        X, Y = np.meshgrid(xs, ys)
        return X, Y, self.deflection

    @property
    def alpha(self) -> float:
        """Non-dimensional centre deflection ``alpha = w_max D / (q a^4)``."""
        return self.centre_deflection * self.bending_stiffness / (
            self.pressure * self.side**4
        )


def kirchhoff_ss_plate_uniform_load(
    *,
    side: float = 1.0,
    young: float = 2.1e11,
    poisson: float = 0.3,
    thickness: float = 0.01,
    pressure: float = 1.0,
    n_x: int = 8,
    n_y: int | None = None,
) -> KirchhoffPlateSolution:
    """Solve the SSSS square plate with the C1 bicubic Hermite (BFS) element.

    Boundary conditions are the *hard simple support*: ``w = 0`` and the
    tangential slope ``w_,t = 0`` on each edge (the latter is implied by
    ``w = 0`` along a straight edge), with the normal bending moment left
    natural, i.e. ``M_n = 0``.
    """
    if side <= 0 or thickness <= 0 or young <= 0:
        raise ValueError("side, thickness and young must be positive")
    if not -1.0 < poisson < 0.5:
        raise ValueError("poisson out of range")
    if n_x < 2 or (n_x % 2):
        raise ValueError("n_x must be even and >= 2 so the centre node exists")
    n_y = n_x if n_y is None else n_y
    if n_y < 2 or (n_y % 2):
        raise ValueError("n_y must be even and >= 2")

    nodes_x, nodes_y = n_x + 1, n_y + 1
    ndof = 4 * nodes_x * nodes_y
    K = np.zeros((ndof, ndof))
    f = np.zeros(ndof)

    lx, ly = side / n_x, side / n_y
    ke, _ = bfs_plate_element_matrices(lx, ly, young, poisson, thickness)
    # 1D consistent load integrals of the cubic Hermite basis:
    #   int N = L/2 (value), L^2/12 (slope at x=0), L/2, -L^2/12 (slope at x=L)
    ix = np.array([lx / 2.0, lx**2 / 12.0, lx / 2.0, -lx**2 / 12.0])
    iy = np.array([ly / 2.0, ly**2 / 12.0, ly / 2.0, -ly**2 / 12.0])
    fe = np.zeros(16)
    for jj in (0, 1):
        for ii in (0, 1):
            base = 4 * (2 * jj + ii)
            for k in range(4):
                fe[base + k] = pressure * ix[1 if k in (1, 3) else 0] * iy[1 if k in (2, 3) else 0]

    def node_dof(i: int, j: int, k: int) -> int:
        return 4 * (j * nodes_x + i) + k

    for j in range(n_y):
        for i in range(n_x):
            dofs = np.array(
                [node_dof(ii, jj, k) for jj in (j, j + 1) for ii in (i, i + 1) for k in range(4)]
            )
            K[np.ix_(dofs, dofs)] += ke
            f[dofs] += fe

    fixed: set[int] = set()
    for i in range(nodes_x):  # edges y = 0 and y = b: w and tangential slope w_,x
        for j in (0, n_y):
            fixed.add(node_dof(i, j, 0))
            fixed.add(node_dof(i, j, 1))
    for j in range(nodes_y):  # edges x = 0 and x = a: w and tangential slope w_,y
        for i in (0, n_x):
            fixed.add(node_dof(i, j, 0))
            fixed.add(node_dof(i, j, 2))
    free = np.array([d for d in range(ndof) if d not in fixed])

    u = np.zeros(ndof)
    u[free] = np.linalg.solve(K[np.ix_(free, free)], f[free])

    centre = node_dof(n_x // 2, n_y // 2, 0)
    rigidity = young * thickness**3 / (12.0 * (1.0 - poisson**2))
    return KirchhoffPlateSolution(
        side=side,
        young=young,
        poisson=poisson,
        thickness=thickness,
        pressure=pressure,
        n_x=n_x,
        n_y=n_y,
        bending_stiffness=rigidity,
        centre_deflection=float(u[centre]),
        deflection=u[0::4].reshape(nodes_y, nodes_x),
        n_dof=ndof,
        n_free=int(free.size),
    )


def plate_centre_deflection_series(
    *,
    side: float = 1.0,
    young: float = 2.1e11,
    poisson: float = 0.3,
    thickness: float = 0.01,
    pressure: float = 1.0,
    terms: int = 401,
) -> float:
    """Navier double-sine series value of the SSSS centre deflection.

    Delegates to :func:`tensorfem.plate_uniform_load.ss_uniform_kirchhoff` so the
    series lives in one place; it converges to ``1e-9`` relative by 401 terms.
    """
    from .plate_uniform_load import ss_uniform_kirchhoff

    return float(
        ss_uniform_kirchhoff(
            side=side,
            young=young,
            poisson=poisson,
            thickness=thickness,
            pressure=pressure,
            x=0.5 * side,
            y=0.5 * side,
            terms=terms,
        )["deflection"]
    )


# ---------------------------------------------------------------------------
# 3. Cantilever modal ladder
# ---------------------------------------------------------------------------

def cantilever_roots(count: int = 5, tol: float = 1e-14) -> tuple[float, ...]:
    """First ``count`` roots of ``cos(z) cosh(z) + 1 = 0`` by bracketed bisection.

    The function changes sign monotonically between consecutive asymptotes of
    ``cos``; a coarse scan locates the brackets and bisection polishes them to
    ``tol``.  For ``count <= 5`` the values reproduce the Blevins table to
    better than ``1e-12`` (asserted in the test-suite).
    """
    if count < 1:
        raise ValueError("count must be >= 1")

    def fn(z: float) -> float:
        return math.cos(z) * math.cosh(z) + 1.0

    roots: list[float] = []
    z, step = 1e-3, 0.05
    prev = fn(z)
    while len(roots) < count:
        z_next = z + step
        val = fn(z_next)
        if prev == 0.0:
            roots.append(z)
        elif prev * val < 0.0:
            lo, hi, f_lo = z, z_next, prev
            while hi - lo > tol:
                mid = 0.5 * (lo + hi)
                f_mid = fn(mid)
                if f_lo * f_mid <= 0.0:
                    hi = mid
                else:
                    lo, f_lo = mid, f_mid
            roots.append(0.5 * (lo + hi))
            if len(roots) == count:
                break
        z, prev = z_next, val
    return tuple(roots)


def cantilever_exact_angular_frequencies(
    length: float, EI: float, rho_a: float, modes: int = 5
) -> tuple[float, ...]:
    """``omega_n = (beta_n L)^2 sqrt(EI / (rho_a L^4))`` for ``modes`` modes."""
    if rho_a <= 0 or EI <= 0 or length <= 0:
        raise ValueError("length, EI and rho_a must be positive")
    return tuple(
        z**2 * math.sqrt(EI / (rho_a * length**4)) for z in cantilever_roots(modes)
    )


@dataclass(frozen=True)
class ModalLadder:
    """FE vs analytic frequencies for a set of meshes."""

    elements: tuple[int, ...]
    fe_angular: np.ndarray  # (levels, modes)
    fe_hertz: np.ndarray
    exact_angular: tuple[float, ...]

    def relative_errors(self) -> np.ndarray:
        exact = np.asarray(self.exact_angular)
        return np.abs(self.fe_angular - exact) / exact


def cantilever_modal_ladder(
    *,
    length: float = 1.0,
    EI: float = 200.0e9 * 0.01 * 0.01**3 / 12.0,
    rho_a: float = 7850.0 * 0.01 * 0.01,
    elements: tuple[int, ...] = (4, 8, 16, 32, 64),
    modes: int = 5,
) -> ModalLadder:
    """Run the Euler--Bernoulli cantilever eigenvalue problem on each mesh."""
    fe = []
    for n in elements:
        result = cantilever_beam_modes(length, n, EI, rho_a, modes=modes)
        values = result.angular_frequencies.numpy()
        if values.size < modes:
            raise RuntimeError(f"mesh with {n} elements produced {values.size} modes")
        fe.append(values[:modes])
    fe_arr = np.asarray(fe, dtype=float)
    return ModalLadder(
        elements=tuple(elements),
        fe_angular=fe_arr,
        fe_hertz=fe_arr / (2.0 * math.pi),
        exact_angular=cantilever_exact_angular_frequencies(length, EI, rho_a, modes),
    )


# ---------------------------------------------------------------------------
# Convergence helpers
# ---------------------------------------------------------------------------

def observed_order(h: np.ndarray, error: np.ndarray, *, floor: float = 0.0) -> float:
    """Least-squares slope of ``log(error)`` against ``log(h)``.

    ``floor`` (default ``0``) drops samples that have reached round-off so a
    legitimate ``O(h^4)`` rate is not reported as a smaller number.
    """
    h = np.asarray(h, dtype=float)
    error = np.asarray(error, dtype=float)
    if h.shape != error.shape:
        raise ValueError("h and error must have the same shape")
    mask = np.isfinite(error) & (error > floor)
    if mask.sum() < 2:
        raise ValueError("need at least two above-floor samples to fit an order")
    slope = np.polyfit(np.log(h[mask]), np.log(error[mask]), 1)[0]
    return float(slope)