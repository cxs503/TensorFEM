"""Membrane (large-deflection) capacity of the ice-belt plate field.

Why this module exists
----------------------
``grillage_ice_ultimate`` decides the ice-belt ultimate pressure of the
JOUR-ICE-STRUCT case study with three limit states: local plate bending, the
longitudinal beam-column and the transverse frame.  In the round-1..8 evidence
bundle the **binding** branch is ``plate_local_bending`` in every acceptance
case, and that branch is built on a *linear* (small-deflection) plate field: the
peak Navier bending moment is compared with the fully plastic moment
``sigma_y t^2 / 4``.  Linear plate theory carries no membrane action, so for a
plate field whose edges are held in plane -- the normal situation in way of the
ice belt, where the bay edges are the longitudinals and the neighbouring bays
continue the plating -- the real capacity is *larger* than the linear value.

The question this module answers, quantitatively and with independent methods,
is: **how large is that reserve, how much does it depend on the in-plane edge
restraint, and is the committed mapping table therefore conservative by a
stated factor?**  It is the item the independent audit of the anchor listed as
not covered (``docs/AUDIT-GAP-PP-5-GRILLAGE-ULTIMATE-independent-review.md``:
the plate branch is a linear bending criterion and no large-deflection check
was made).

Mechanical model
----------------
The plate field between two longitudinals is treated as a **unit-width strip**
spanning the bay spacing across the bay, loaded by the ice patch plus the
static head.  The reduction is exact-in-the-governing-direction whenever the
patch spans the full bay width, which holds for all three acceptance scenarios
(the patch width is clipped to the bay spacing in
``grillage_ice_ultimate.ultimate_ice_pressure``); the load is then uniform
across the strip span and the strip carries the same peak bending as the plate.

Kinematics are von Karman shallow-strip (moderate rotations, small strain)::

    eps = u' + (1/2) w'^2,      N = EA eps,      EI w'''' - N w'' = q

with ``N`` constant along the strip because the axial force is carried by the
in-plane restraint only.  The in-plane edge condition is parameterised by a
restraint stiffness ``k_s`` (force per unit edge length per unit end
shortening, ``N/m^2``)::

    N L / (EA) + N / k_s = (1/2) * int_0^L w'^2 dx                (*)

``k_s -> inf`` gives the **immovable** (fully restrained) edge, ``k_s -> 0`` the
**free** edge in which no membrane exists and the classic linear plate field is
recovered.  A real structure sits between the two, so the module reports the
bracket instead of a single number.

Three independent solvers for the same equations
------------------------------------------------
No route is checked against itself:

``strip_membrane_exact``
    For a *given* axial force ``N`` the ODE is linear with constant
    coefficients, so ``w(x)`` is closed form (particular ``q x^2 / 2N`` plus
    ``A + Bx + C cosh kx + D sinh kx``, ``k = sqrt(N/EI)``) with the four
    simply-supported conditions fixing ``A..D`` analytically.  The
    compatibility (*) is then a **scalar** root find in ``N``: no series, no
    mesh, no iteration on the field.
``strip_membrane_series``
    Ritz/Galerkin on ``w = sum a_m sin(m pi x / L)`` (odd ``m`` only; the
    problem is symmetric) with the membrane energy written in terms of the end
    shortening; the reduced nonlinear system is solved by damped Newton.  Its
    small-deflection limit is the textbook series solution of
    ``w_max = (5/384) q L^4 / EI``.
``strip_membrane_fe``
    Corotational finite elements, reusing the verified element energy and
    AD-exact tangent of :mod:`tensorfem.beam_large_deflection`
    (``_assemble``): both ends are axially *and* transversely supported, so the
    membrane force appears on its own from the element axial stretch energy.
    This route carries finite rotations and never uses (*) explicitly.

Capacity criteria
-----------------
Three load levels are reported per case, all on the same strip:

``p_nomembrane``
    ``max_x M = M_p sqrt(1 - eta^2)`` with ``M_p = sigma_y t^2 / 4``.  This
    deliberately reproduces the *convention of the committed module* (elastic
    moment field against the plastic moment, no membrane) on the strip
    reduction, so that it can be differenced against the membrane results
    without mixing conventions.  Closed form: ``p = 8 M_p sqrt(1-eta^2)/L^2``.
``p_first_yield``
    membrane included, extreme-fibre stress reaching yield:
    ``sigma_x + N/t + M/Z = sigma_y``, ``Z = t^2/6``.
``p_plastic``
    membrane included, rectangular-section full-plasticity interaction at the
    peak-moment section: ``M/M_p + eta_tot^2 = 1`` with
    ``eta_tot = (sigma_x t + N) / (sigma_y t)``.

The ratio ``p_plastic / p_nomembrane`` is the **membrane reserve** that the
committed mapping table leaves unused.  ``w/t`` is reported with every load
level because the von Karman kinematics -- and the ductility of a welded panel
-- only justify the number while ``w/t`` stays of order one; the load at
``w = t`` and ``w = 2t`` is therefore reported as a ductility cap.

Verification (``tests/test_grillage_plate_membrane.py``)
--------------------------------------------------------
``M1`` exact route in the ``N -> 0`` limit recovers the linear strip field
``w_max = 5 q L^4 / 384 EI`` and ``M_max = q L^2 / 8`` (1 %).
``M2`` series route in the same limit (1 %) -- the two routes share no code.
``M3`` series vs exact over four load levels spanning ``w/t = 0.1 .. 1.6``
(1 %); the exact route is the reference.
``M4`` corotational FE vs exact over the same load levels with a mesh study
(20/40/80/160 elements), monotone convergence and 3 % at the finest mesh.
``M5`` the axial force recovered from the FE element stretch matches the
root-found ``N`` of the exact route (2 %).
``M6`` restraint bracketing: the free edge reproduces ``p_nomembrane``, the
immovable edge gives the largest reserve, and the reserve is monotone in
``k_s``.
``M7`` energy: the Ritz coefficients sit at a minimum of the total potential
and the axial force of the three routes agrees with the compatibility (*).
``M8`` the three acceptance scenarios give bounded reserves and the committed
linear criterion stays *below* the membrane capacities, i.e. the mapping table
is conservative -- the claim the round-9 note makes.

References
----------
Timoshenko & Woinowsky-Krieger, *Theory of Plates and Shells*, 2nd ed.,
ch. 8-9 (large deflection, membrane stress in restrained plates); von Karman
(1910) for the strain measure; Chia, *Nonlinear Analysis of Plates* (1980) for
the Ritz formulation; Crisfield, *Non-linear Finite Element Analysis* (1991)
ch. 7/17 for the corotational element used as the third route.

Honest scope
------------
* The strip reduction ignores two-dimensional redistribution across the bay.
  The module reports the strip-vs-2D difference against the committed module
  (``membrane_reserve_table``) instead of assuming it away.
* The membrane reserve is an upper-bound-type statement on capacity: it needs
  the edge restraint and the panel ductility.  Both are parameterised
  (``restraint``, ``w/t`` caps) and never silently chosen.
* The committed design criterion is *not* changed here.  This module supplies
  the bracket and the machine-readable record of the reserve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .beam_large_deflection import _assemble
from .elastica_cantilever import newton_converged

__all__ = [
    "StripMembraneGeometry",
    "MembraneSolution",
    "StripCapacity",
    "strip_membrane_exact",
    "strip_membrane_series",
    "strip_membrane_fe",
    "strip_capacity",
    "membrane_reserve_table",
    "case_strip_geometry",
    "RESTRAINT_LEVELS",
]

_trapezoid = getattr(np, "trapezoid", None) or np.trapz


# --------------------------------------------------------------------------
# geometry / unit-width section
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StripMembraneGeometry:
    """Unit-width plate strip of span ``span`` and thickness ``thickness``.

    ``restraint`` is the in-plane edge stiffness ``k_s`` in ``N/m^2`` (force per
    unit edge length per unit end shortening); ``math.inf`` is an immovable edge
    and ``0.0`` a free edge.  ``axial_stress`` is the coexisting hull-girder
    axial stress carried by the strip (positive in compression); it enters the
    capacity criteria only, never the closing equations.
    """

    span: float
    thickness: float
    young: float = 206.0e9
    yield_stress: float = 355.0e6
    poisson: float = 0.3
    restraint: float = math.inf
    axial_stress: float = 0.0
    name: str = "strip"

    def __post_init__(self) -> None:
        for name in ("span", "thickness", "young", "yield_stress"):
            value = float(getattr(self, name))
            if not (value > 0.0 and math.isfinite(value)):
                raise ValueError(f"{name} must be positive and finite")
        if not (float(self.restraint) >= 0.0):
            raise ValueError("restraint must be non-negative")
        if not abs(float(self.axial_stress)) < float(self.yield_stress):
            raise ValueError("coexisting axial stress must stay below yield")

    @property
    def area(self) -> float:
        return self.thickness

    @property
    def inertia(self) -> float:
        return self.thickness ** 3 / 12.0

    @property
    def section_modulus(self) -> float:
        return self.thickness ** 2 / 6.0

    @property
    def bending_rigidity(self) -> float:
        return self.young * self.inertia

    @property
    def axial_rigidity(self) -> float:
        return self.young * self.area

    @property
    def plastic_moment(self) -> float:
        return self.yield_stress * self.thickness ** 2 / 4.0

    @property
    def axial_yield(self) -> float:
        return self.yield_stress * self.area

    @property
    def eta(self) -> float:
        return abs(self.axial_stress) / self.yield_stress


@dataclass(frozen=True)
class MembraneSolution:
    """One converged strip solution at a given lateral pressure."""

    pressure: float
    axial_force: float          # N [N/m], tension positive
    max_moment: float           # |M| peak [N m/m]
    max_deflection: float       # |w| peak [m]
    solver: str = "exact"
    converged: bool = True


# --------------------------------------------------------------------------
# route 1: closed-form ODE for a given N + scalar compatibility root find
# --------------------------------------------------------------------------


def _linear_strip(geom: StripMembraneGeometry, pressure: float,
                  nodes: int = 2001) -> MembraneSolution:
    """Classic linear (no-membrane) simply-supported strip."""
    L = geom.span
    EI = geom.bending_rigidity
    x = np.linspace(0.0, L, nodes)
    w = pressure * x * (L ** 3 - 2.0 * L * x ** 2 + x ** 3) / (24.0 * EI)
    return MembraneSolution(pressure=pressure, axial_force=0.0,
                            max_moment=pressure * L ** 2 / 8.0,
                            max_deflection=float(np.max(np.abs(w))),
                            solver="linear")


def _exact_field(geom: StripMembraneGeometry, pressure: float,
                 axial_force: float) -> tuple[Callable, Callable]:
    """Closed-form ``w(x)`` and ``w''(x)`` for a prescribed axial force."""
    L = geom.span
    EI = geom.bending_rigidity
    N = max(axial_force, 1.0e-30)

    k = math.sqrt(N / EI)
    kL = k * L
    C = -pressure * EI / N ** 2
    D = C * (1.0 - math.cosh(kL)) / math.sinh(kL)
    A = -C
    B = -(pressure * L ** 2 / (2.0 * N) + A + C * math.cosh(kL)
          + D * math.sinh(kL)) / L

    def w(x: np.ndarray) -> np.ndarray:
        return (pressure * x ** 2 / (2.0 * N) + A + B * x
                + C * np.cosh(k * x) + D * np.sinh(k * x))

    def curvature(x: np.ndarray) -> np.ndarray:
        return pressure / N + k ** 2 * (C * np.cosh(k * x) + D * np.sinh(k * x))

    return w, curvature


def _shortening(geom: StripMembraneGeometry, pressure: float,
                axial_force: float, nodes: int = 4001) -> float:
    """``(1/2) int w'^2 dx`` for the closed-form field."""
    w, _ = _exact_field(geom, pressure, axial_force)
    x = np.linspace(0.0, geom.span, nodes)
    dw = np.gradient(w(x), x)
    return 0.5 * float(_trapezoid(dw ** 2, x))


def _elastic_shortening(geom: StripMembraneGeometry, axial_force: float) -> float:
    """``N L/(EA) + N/k_s``: the end shortening absorbed by structure."""
    N = max(axial_force, 0.0)
    total = N * geom.span / geom.axial_rigidity
    if geom.restraint == 0.0:
        return math.inf
    if math.isfinite(geom.restraint):
        total += N / geom.restraint
    return total


def strip_membrane_exact(geom: StripMembraneGeometry, pressure: float, *,
                         nodes: int = 4001) -> MembraneSolution:
    """Solve the von Karman strip by closed-form ODE + scalar root find."""
    if pressure < 0.0:
        raise ValueError("pressure must be non-negative")
    if geom.restraint == 0.0:
        return _linear_strip(geom, pressure, nodes=nodes)
    L = geom.span
    EI = geom.bending_rigidity
    scale = EI / L ** 2                      # force-per-length scale of N

    def residual(N: float) -> float:
        return _shortening(geom, pressure, N, nodes=nodes) - _elastic_shortening(geom, N)

    # residual = geometric shortening - absorbed shortening is positive for a
    # small axial force (the field is the linear one) and negative for a large
    # one (the membrane stiffens the strip), so the bracket is [lo, hi] with
    # residual(lo) > 0 > residual(hi).
    lo = 1.0e-10 * scale
    hi = 4.0 * scale
    if residual(lo) < 0.0:
        # the load is too small for any membrane force to build up: the field is
        # the linear one and N -> 0 (exact limit, not a numerical fallback)
        return _linear_strip(geom, pressure, nodes=nodes)
    for _ in range(200):
        if residual(lo) >= 0.0 >= residual(hi):
            break
        hi *= 4.0
    else:  # pragma: no cover - bracket failure
        raise RuntimeError("could not bracket the membrane force")

    from scipy.optimize import brentq

    N = float(brentq(residual, lo, hi, xtol=1e-13 * scale,
                     rtol=8.9e-16, maxiter=300))
    w, curvature = _exact_field(geom, pressure, N)
    x = np.linspace(0.0, L, nodes)
    return MembraneSolution(pressure=pressure, axial_force=N,
                            max_moment=float(EI * np.max(np.abs(curvature(x)))),
                            max_deflection=float(np.max(np.abs(w(x)))),
                            solver="exact")


# --------------------------------------------------------------------------
# route 2: Ritz / Galerkin spectral solution
# --------------------------------------------------------------------------


def strip_membrane_series(geom: StripMembraneGeometry, pressure: float, *,
                          terms: int = 21, tolerance: float = 1e-12,
                          max_iterations: int = 300,
                          nodes: int = 4001) -> MembraneSolution:
    """Ritz solution ``w = sum a_m sin(m pi x / L)``, odd ``m``, Newton solve."""
    if pressure < 0.0:
        raise ValueError("pressure must be non-negative")
    if terms < 1:
        raise ValueError("terms must be positive")
    L = geom.span
    EI = geom.bending_rigidity
    EA = geom.axial_rigidity
    modes = np.arange(1, 2 * terms, 2, dtype=float)
    k_m = modes * math.pi / L
    spring = geom.restraint if math.isfinite(geom.restraint) else 0.0
    stiff = EA / L + spring

    def delta_of(a: np.ndarray) -> float:
        return 0.25 * float(np.sum(a ** 2 * k_m ** 2)) * L

    def potential(a: np.ndarray) -> float:
        bend = 0.25 * EI * L * float(np.sum((a * k_m ** 2) ** 2))
        delta = delta_of(a)
        return (bend + 0.5 * stiff * delta ** 2
                - pressure * float(np.sum(a * 2.0 * L / (modes * math.pi))))

    def gradient(a: np.ndarray) -> np.ndarray:
        delta = delta_of(a)
        return (0.5 * EI * L * k_m ** 4 * a
                + stiff * delta * 0.5 * a * k_m ** 2 * L
                - 2.0 * pressure * L / (modes * math.pi))

    def hessian(a: np.ndarray) -> np.ndarray:
        delta = delta_of(a)
        d_delta = 0.5 * a * k_m ** 2 * L
        H = np.diag(0.5 * EI * L * k_m ** 4 + stiff * delta * 0.5 * k_m ** 2 * L)
        return H + stiff * np.outer(d_delta, d_delta)

    # linear start vector
    a = 4.0 * pressure * L ** 4 / EI / (modes * math.pi) ** 5
    k4_max = float(np.max(k_m ** 4))
    converged = False
    for _ in range(max_iterations):
        g = gradient(a)
        # the residual is compared with the size of the terms that build it, not
        # with an absolute force: the bending term scales with EI k^4 a and is
        # O(1e2) here, so an absolute tolerance would sit under the round-off
        # floor of the assembly.
        scale_g = (0.5 * EI * L * k4_max * float(np.max(np.abs(a)))
                   + stiff * float(np.max(np.abs(a * k_m ** 2))) ** 2 * L
                   + pressure * L + 1.0)
        if float(np.max(np.abs(g))) <= tolerance * scale_g:
            converged = True
            break
        step = np.linalg.solve(hessian(a), -g)
        alpha = 1.0
        base = potential(a)
        while alpha > 1e-3 and potential(a + alpha * step) > base:
            alpha *= 0.5
        a = a + alpha * step
        if float(np.max(np.abs(step))) <= 1e-15 * max(float(np.max(np.abs(a))), 1e-30):
            converged = True
            break
    if not converged:  # pragma: no cover - diagnostic only
        raise RuntimeError("Ritz Newton did not converge")

    delta = delta_of(a)
    N = delta / (L / EA + (1.0 / geom.restraint
                           if math.isfinite(geom.restraint) and geom.restraint > 0.0
                           else (math.inf if geom.restraint == 0.0 else 0.0)))
    x = np.linspace(0.0, L, nodes)
    shape = np.sin(np.outer(x, k_m))
    w = shape @ a
    curvature = np.abs(shape @ (k_m ** 2 * a))
    return MembraneSolution(pressure=pressure, axial_force=float(N),
                            max_moment=float(EI * np.max(curvature)),
                            max_deflection=float(np.max(np.abs(w))),
                            solver="series")


# --------------------------------------------------------------------------
# route 3: corotational finite elements (reuse of the verified element)
# --------------------------------------------------------------------------


def strip_membrane_fe(geom: StripMembraneGeometry, pressure: float, *,
                      n_elements: int = 80, load_steps: int = 8,
                      tolerance: float = 1e-10,
                      max_iterations: int = 60) -> MembraneSolution:
    """Simply-supported strip with immovable ends by corotational FE.

    Both ends are supported in ``u`` and ``v``; rotations are free (simply
    supported).  The membrane force comes out of the element axial stretch
    energy, i.e. this route never uses the compatibility relation (*).
    """
    if pressure < 0.0:
        raise ValueError("pressure must be non-negative")
    if n_elements < 2:
        raise ValueError("n_elements must be at least 2")
    L = geom.span
    EI = geom.bending_rigidity
    EA = geom.axial_rigidity

    n_nodes = n_elements + 1
    x = np.linspace(0.0, L, n_nodes)
    coords = np.stack((x, np.zeros_like(x)), axis=1)
    connectivity = np.array([[i, i + 1] for i in range(n_elements)])
    element_nodes = np.concatenate((coords[connectivity[:, 0]],
                                    coords[connectivity[:, 1]]), axis=1)
    lengths = np.full(n_elements, L / n_elements)

    dofs = np.arange(3 * n_nodes).reshape(n_nodes, 3)
    fixed = np.array([dofs[0, 0], dofs[0, 1], dofs[-1, 0], dofs[-1, 1]])
    free = np.setdiff1d(np.arange(3 * n_nodes), fixed)

    # consistent nodal loads for a uniform pressure (unit width)
    base = np.zeros(3 * n_nodes)
    for e in range(n_elements):
        half = pressure * lengths[e] / 2.0
        base[dofs[e, 1]] += half
        base[dofs[e + 1, 1]] += half

    u = np.zeros(3 * n_nodes)
    iterations = 0
    converged = True
    for step in range(1, load_steps + 1):
        load = base * (step / load_steps)
        step_converged = False
        for _ in range(max_iterations):
            iterations += 1
            _, f_int, k_tan = _assemble(element_nodes, u, EA, EI,
                                        connectivity, lengths)
            g = f_int - load
            g_norm = float(np.linalg.norm(g[free]))
            force_norm = float(np.linalg.norm(load))
            if newton_converged(g_norm, 0.0, float(np.linalg.norm(u[free])),
                                force_norm, ftol=tolerance * max(force_norm, 1.0)):
                step_converged = True
                break
            delta = np.linalg.solve(k_tan[np.ix_(free, free)], -g[free])
            u[free] += delta
            if newton_converged(g_norm, float(np.linalg.norm(delta)),
                                float(np.linalg.norm(u[free])), force_norm,
                                ftol=tolerance * max(force_norm, 1.0)):
                step_converged = True
                break
        if not step_converged:  # pragma: no cover - diagnostic only
            converged = False
            break

    # post-processing: element axial forces and end moments
    u_nodal = u.reshape(n_nodes, 3)
    axial = np.empty(n_elements)
    moment_end = np.empty((n_elements, 2))
    for e in range(n_elements):
        n0, n1 = connectivity[e]
        x0, y0, x1, y1 = element_nodes[e]
        dx = (x1 - x0) + u_nodal[n1, 0] - u_nodal[n0, 0]
        dy = (y1 - y0) + u_nodal[n1, 1] - u_nodal[n0, 1]
        length = math.hypot(dx, dy)
        phi = math.atan2(dy, dx)
        t1 = u_nodal[n0, 2] - phi
        t2 = u_nodal[n1, 2] - phi
        axial[e] = EA * (length - lengths[e]) / lengths[e]
        moment_end[e, 0] = 2.0 * EI / lengths[e] * (2.0 * t1 + t2)
        moment_end[e, 1] = 2.0 * EI / lengths[e] * (2.0 * t2 + t1)

    return MembraneSolution(
        pressure=pressure,
        axial_force=float(np.mean(axial)),
        max_moment=float(np.max(np.abs(moment_end))),
        max_deflection=float(np.max(np.abs(u_nodal[:, 1]))),
        solver=f"fe{n_elements}", converged=converged)


# --------------------------------------------------------------------------
# capacity criteria
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StripCapacity:
    """Load levels of the three plate-strip criteria plus ductility caps."""

    name: str
    restraint: float
    pressure_nomembrane: float
    pressure_nomembrane_same_law: float
    pressure_first_yield: float
    pressure_plastic: float
    deflection_first_yield: float
    deflection_plastic: float
    thickness: float
    axial_force_plastic: float
    axial_force_first_yield: float
    pressure_at_w_eq_t: float
    pressure_at_w_eq_2t: float
    solver: str = "exact"

    @property
    def first_yield_over_committed(self) -> float:
        """First yield relative to the committed (sqrt-interaction) reference."""
        return self.pressure_first_yield / self.pressure_nomembrane

    @property
    def plastic_over_committed(self) -> float:
        """Plastic limit relative to the committed (sqrt-interaction) reference.

        Mixes the membrane effect with the difference between the sqrt and the
        parabolic axial interaction law, so it is *not* the membrane reserve.
        """
        return self.pressure_plastic / self.pressure_nomembrane

    @property
    def reserve_membrane(self) -> float:
        """Pure membrane reserve: plastic limit against the same interaction law.

        Both numerator and denominator use ``M/M_p + eta^2 = 1`` and the same
        strip, so the only difference is the membrane force.
        """
        return self.pressure_plastic / self.pressure_nomembrane_same_law

    @property
    def w_over_t_plastic(self) -> float:
        return self.deflection_plastic / self.thickness


def _bisect_pressure(criterion: Callable[[MembraneSolution], float],
                     solver: Callable[[float], MembraneSolution],
                     lo: float, hi: float, *, tol: float = 1e-10,
                     max_iter: int = 200) -> tuple[float, MembraneSolution]:
    """Load factor where ``criterion(state) = 1``, criterion monotone in p."""
    f_lo = criterion(solver(lo)) - 1.0
    state_hi = solver(hi)
    f_hi = criterion(state_hi) - 1.0
    expand = 0
    while f_hi < 0.0 and expand < 60:
        hi *= 2.0
        state_hi = solver(hi)
        f_hi = criterion(state_hi) - 1.0
        expand += 1
    if f_lo > 0.0:
        return lo, solver(lo)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        state = solver(mid)
        f_mid = criterion(state) - 1.0
        if abs(f_mid) < tol or (hi - lo) < tol * max(hi, 1.0):
            return mid, state
        if f_mid < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), solver(0.5 * (lo + hi))


def strip_capacity(geom: StripMembraneGeometry, *, solver: str = "exact",
                   terms: int = 21, n_elements: int = 80) -> StripCapacity:
    """Load levels of the three criteria for one strip.

    ``solver`` selects the route used for the membrane-inclusive criteria
    (``"exact"``, ``"series"`` or ``"fe"``); ``p_nomembrane`` is closed form, so
    every membrane number is differenced against the same reference.
    """
    sigma_y = geom.yield_stress
    Mp = geom.plastic_moment
    Z = geom.section_modulus
    eta = geom.eta
    # closed form linear strip: M_max = p L^2/8 = Mp sqrt(1-eta^2)
    p_nomem = 8.0 * Mp * math.sqrt(max(1.0 - eta ** 2, 0.0)) / geom.span ** 2
    # same interaction law as the membrane criteria, still without membrane:
    # the reference that isolates the membrane effect
    p_nomem_same_law = 8.0 * Mp * max(1.0 - eta ** 2, 0.0) / geom.span ** 2

    def run(p: float) -> MembraneSolution:
        if p <= 0.0:
            return MembraneSolution(0.0, 0.0, 0.0, 0.0, solver=solver)
        if solver == "exact":
            return strip_membrane_exact(geom, p)
        if solver == "series":
            return strip_membrane_series(geom, p, terms=terms)
        if solver == "fe":
            return strip_membrane_fe(geom, p, n_elements=n_elements)
        raise ValueError(f"unknown solver {solver!r}")

    # the membrane force is *tension* (a sagging restrained strip pulls its
    # edges in) while ``axial_stress`` is compression, so the two subtract: the
    # membrane relieves the compression fibre and adds to the tension one.
    def first_yield(state: MembraneSolution) -> float:
        sigma_compression_fibre = (geom.axial_stress
                                   - state.axial_force / geom.thickness
                                   + state.max_moment / Z)
        return sigma_compression_fibre / sigma_y

    def plastic(state: MembraneSolution) -> float:
        eta_tot = (geom.axial_stress * geom.thickness - state.axial_force) / (
            sigma_y * geom.thickness)
        return state.max_moment / Mp + eta_tot ** 2

    hi = max(4.0 * p_nomem, 1.0e-6)
    p_fy, s_fy = _bisect_pressure(first_yield, run, 1.0e-12, hi)
    p_pl, s_pl = _bisect_pressure(plastic, run, 1.0e-12, max(hi, p_fy * 2.0))
    p_t, _ = _bisect_pressure(lambda s: s.max_deflection / geom.thickness,
                              run, 1.0e-12, max(hi, p_pl * 2.0))
    p_2t, _ = _bisect_pressure(lambda s: 0.5 * s.max_deflection / geom.thickness,
                               run, 1.0e-12, max(hi, p_pl * 2.0))

    return StripCapacity(
        name=geom.name, restraint=geom.restraint,
        pressure_nomembrane=p_nomem,
        pressure_nomembrane_same_law=p_nomem_same_law,
        pressure_first_yield=p_fy,
        pressure_plastic=p_pl,
        deflection_first_yield=s_fy.max_deflection,
        deflection_plastic=s_pl.max_deflection,
        thickness=geom.thickness,
        axial_force_plastic=s_pl.axial_force,
        axial_force_first_yield=s_fy.axial_force,
        pressure_at_w_eq_t=p_t,
        pressure_at_w_eq_2t=p_2t,
        solver=solver,
    )


# --------------------------------------------------------------------------
# application to the anchor's acceptance cases
# --------------------------------------------------------------------------

#: in-plane restraint levels reported for every acceptance case.
#: ``inf`` = immovable edge (upper bound on the membrane reserve);
#: ``1/2`` and ``1/4`` = fractions of the *plate axial stiffness* EA/L offered
#: by the surrounding structure, i.e. a partially restrained edge; ``0`` = free
#: edge (no membrane, recovers the committed linear criterion).
RESTRAINT_LEVELS: tuple[tuple[str, float], ...] = (
    ("immovable", math.inf),
    ("half_ea_over_l", 0.5),
    ("quarter_ea_over_l", 0.25),
    ("free", 0.0),
)


def case_strip_geometry(scenario: str, restraint: float = math.inf, *,
                        plate_thickness: float | None = None
                        ) -> StripMembraneGeometry:
    """Build the plate-field strip of an acceptance scenario.

    The strip spans the bay spacing (the short direction of the plate field, in
    which the patch covers the full width) and carries the coexisting
    hull-girder axial stress of the scenario.  ``restraint`` given as a fraction
    (``0 < restraint <= 1``) is scaled to that fraction of the plate axial
    stiffness ``EA/L``.
    """
    from .grillage_ice_ultimate import case_study_geometry
    from .grillage_ice_verification import CASE_STUDY_SCENARIOS

    geom = case_study_geometry()
    t = geom.plate_thickness if plate_thickness is None else plate_thickness
    lookup = {row[0]: row for row in CASE_STUDY_SCENARIOS}
    if scenario not in lookup:
        raise KeyError(f"unknown scenario {scenario!r}; have {sorted(lookup)}")
    name, eta, _p_ice, _lx, _ly, _p_h = lookup[scenario]
    reference = StripMembraneGeometry(span=geom.bay_spacing, thickness=t,
                                      young=geom.young,
                                      yield_stress=geom.yield_stress,
                                      poisson=geom.poisson)
    if restraint == math.inf or restraint == 0.0:
        k_s = float(restraint)
    else:
        k_s = float(restraint) * reference.axial_rigidity / reference.span
    return StripMembraneGeometry(
        span=geom.bay_spacing, thickness=t, young=geom.young,
        yield_stress=geom.yield_stress, poisson=geom.poisson,
        restraint=k_s, axial_stress=eta * geom.yield_stress, name=name)


def membrane_reserve_table(*, scenarios: Sequence[str] | None = None,
                           solver: str = "exact",
                           restraint_levels: Sequence[tuple[str, float]] | None = None
                           ) -> list[dict[str, object]]:
    """Membrane reserve of the plate-field branch for the acceptance cases.

    Each row carries the committed linear plate-field value of the anchor
    (recomputed from ``grillage_ice_ultimate``, not transcribed), the strip
    ``p_nomembrane`` reference and the membrane capacities, so the reserve and
    the strip-vs-2D reduction error are both machine readable.
    """
    from .grillage_ice_ultimate import (CombinedLoadCase, case_study_geometry,
                                        ultimate_ice_pressure)
    from .grillage_ice_verification import CASE_STUDY_SCENARIOS

    geom = case_study_geometry()
    lookup = {row[0]: row for row in CASE_STUDY_SCENARIOS}
    names = list(scenarios) if scenarios is not None else [r[0] for r in CASE_STUDY_SCENARIOS]
    levels = tuple(restraint_levels) if restraint_levels is not None else RESTRAINT_LEVELS

    rows: list[dict[str, object]] = []
    for scenario in names:
        _name, eta, p_ice, lx, ly, p_h = lookup[scenario]
        committed = float(ultimate_ice_pressure(
            geom, CombinedLoadCase(axial_stress=eta * geom.yield_stress,
                                   ice_pressure=p_ice, patch_length=lx,
                                   patch_width=ly,
                                   hydrostatic_pressure=p_h))["ice_pressure_ultimate"])
        for label, level in levels:
            strip = case_strip_geometry(scenario, level)
            cap = strip_capacity(strip, solver=solver)
            rows.append({
                "scenario": scenario,
                "restraint_label": label,
                "restraint_k_s": cap.restraint,
                "plate_thickness": strip.thickness,
                "span": strip.span,
                "axial_stress_MPa": strip.axial_stress / 1e6,
                "committed_p_ult_2D_MPa": committed / 1e6,
                "strip_p_nomembrane_MPa": cap.pressure_nomembrane / 1e6,
                "strip_p_nomembrane_same_law_MPa":
                    cap.pressure_nomembrane_same_law / 1e6,
                "strip_vs_2D_error_pct": 100.0 * (cap.pressure_nomembrane / committed - 1.0),
                "p_first_yield_MPa": cap.pressure_first_yield / 1e6,
                "p_plastic_MPa": cap.pressure_plastic / 1e6,
                "first_yield_over_committed": cap.first_yield_over_committed,
                "plastic_over_committed": cap.plastic_over_committed,
                "reserve_membrane": cap.reserve_membrane,
                "w_over_t_at_plastic": cap.w_over_t_plastic,
                "p_at_w_eq_t_MPa": cap.pressure_at_w_eq_t / 1e6,
                "p_at_w_eq_2t_MPa": cap.pressure_at_w_eq_2t / 1e6,
                "solver": cap.solver,
            })
    return rows