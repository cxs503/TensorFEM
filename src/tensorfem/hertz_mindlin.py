"""Cattaneo--Mindlin partial-slip contact on a rigid indenter (influence-coefficient form).

What this module owns
---------------------
Verification-grade contact mechanics for a rigid circular indenter pressed into
an elastic half-space:

* **Hertz** -- contact half-width ``b``, peak pressure ``p0``, load ``P``;
* **Cattaneo--Mindlin** -- after a normal preload ``P`` a tangential load
  ``Q < mu P`` is applied, producing a central stick zone of half-width ``c``
  bounded by two slip strips;
* **unilateral constraint** -- ``gap >= 0``, ``pressure >= 0``,
  ``gap * pressure = 0`` enforced *exactly* (active set on Lagrange
  multipliers, no penalty), so the complementarity residual is solver
  round-off rather than penalty-limited.

Contact operator
----------------
The operator is the exact 2-D plane-strain Flamant compliance of a half-plane
surface.  For a surface pressure ``p(x)``::

    u_z(x) = -(2 / (pi E*)) int p(s) ln|x - s| ds + const,   E* = E/(1-nu^2),

and the tangential surface displacement under a tangential traction ``q(x)``
obeys the *same* kernel with the *same* constant (the classical 2-D Flamant
correspondence, which is what makes the 2-D partial-slip law a square root).
Tractions are piecewise constant on a segment grid and collocated at the
segment centres, giving a symmetric positive-definite compliance matrix and a
linear-complementarity problem of the same size as the grid (a few hundred
unknowns).

Because the kernel is exact, the only discretisation error is the traction
interpolation, and the recovered traction field is smooth.  This is deliberate:
a node-to-node Q4 finite-element contact model of the same problem produces a
node-to-node pressure sawtooth (alternating signs of the nodal contact force)
which corrupts ``p0`` and the stick boundary -- measured in
``docs/cattaneo-mindlin-partial-slip-validation.md``.  The module therefore
carries a small plane-strain Q4 finite element *only* to audit the kernel: for
a prescribed traction field the FE surface settlement must reproduce the
Flamant convolution.

Scope and limitations
---------------------
Linear elasticity, small sliding, one convex rigid indenter, monotonic normal
then tangential loading, no normal/tangential kernel coupling (the 2-D
cross-terms are constants and steps of order ``1-2nu`` that drop out of the
gap equation), two-dimensional plane strain.  Consequently the partial-slip
law is the 2-D square-root law, not the 3-D ``2/3`` power law; the 3-D Hertz
load--approach law is provided analytically (``hertz_sphere_reference``) and is
checked against the axisymmetric finite-element solver in
``benchmarks/mindlin_contact/run.py``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.special import ellipe, ellipk
import scipy.sparse.linalg as spla

__all__ = [
    "HertzCylinderReference", "HertzSphereReference", "NormalSolution",
    "TangentialSolution", "hertz_cylinder_reference", "hertz_sphere_reference",
    "plane_strain_modulus", "half_plane_influence", "segment_grid",
    "solve_hertz_2d", "solve_hertz_sphere_3d", "solve_cattaneo_mindlin_2d",
    "stick_half_width_2d",
    "block_mesh", "plane_strain_stiffness", "ElasticSolver",
    "surface_nodes", "dirichlet_dofs", "solve_indentation_fe",
]


# --------------------------------------------------------------------------- #
# closed-form references (independent oracles)
# --------------------------------------------------------------------------- #
def plane_strain_modulus(young, poisson):
    """``E* = E/(1-nu^2)``: plane-strain modulus of a single elastic body."""
    if young <= 0 or not -1.0 < poisson < 0.5:
        raise ValueError("invalid material")
    return young / (1.0 - poisson ** 2)


@dataclass(frozen=True)
class HertzCylinderReference:
    """2-D (plane-strain) Hertz solution for a rigid cylinder of radius ``R``."""

    load: float
    half_width: float
    peak_pressure: float

    def pressure(self, x):
        x = np.asarray(x, dtype=float)
        return self.peak_pressure * np.sqrt(
            np.clip(1.0 - (x / self.half_width) ** 2, 0.0, None))


def hertz_cylinder_reference(*, load, radius, effective_modulus):
    """``b = sqrt(4 P R/(pi E*))``, ``p0 = 2 P/(pi b)`` (load-controlled form)."""
    if min(load, radius, effective_modulus) <= 0:
        raise ValueError("positive load, radius and modulus required")
    b = math.sqrt(4.0 * load * radius / (math.pi * effective_modulus))
    return HertzCylinderReference(load=load, half_width=b,
                                  peak_pressure=2.0 * load / (math.pi * b))


@dataclass(frozen=True)
class HertzSphereReference:
    """3-D Hertz solution for a rigid sphere on an elastic half-space."""

    load: float
    contact_radius: float
    peak_pressure: float
    approach: float


def hertz_sphere_reference(*, load=None, approach=None, radius, effective_modulus):
    """3-D Hertz: ``a = (3FR/(4E*))^(1/3)``, ``p0 = 3F/(2 pi a^2)``, ``delta = a^2/R``."""
    if radius <= 0 or effective_modulus <= 0:
        raise ValueError("invalid Hertz parameters")
    if (load is None) == (approach is None):
        raise ValueError("give exactly one of load/approach")
    if approach is not None:
        a = math.sqrt(radius * approach)
        load = 4.0 * effective_modulus * a ** 3 / (3.0 * radius)
    else:
        a = (3.0 * load * radius / (4.0 * effective_modulus)) ** (1.0 / 3.0)
    return HertzSphereReference(load=load, contact_radius=a,
                                peak_pressure=3.0 * load / (2.0 * math.pi * a * a),
                                approach=a * a / radius)


def stick_half_width_2d(contact_half_width, tangential, normal, mu):
    """2-D Cattaneo result ``c = b sqrt(1 - Q/(mu P))``."""
    if normal <= 0 or mu <= 0:
        raise ValueError("normal load and friction must be positive")
    ratio = min(max(tangential / (mu * normal), 0.0), 1.0)
    return contact_half_width * math.sqrt(1.0 - ratio)


# --------------------------------------------------------------------------- #
# exact half-plane (Flamant) compliance operator
# --------------------------------------------------------------------------- #
def _log_segment_integral(x, a, b):
    """``int_a^b ln|x-s| ds`` evaluated analytically and vectorised.

    Antiderivative ``F(t) = (t-x)(ln|t-x| - 1)``, with the limit
    ``d ln|d| -> 0`` at ``d = 0``.
    """
    x = np.asarray(x, dtype=float)[:, None]
    a = np.asarray(a, dtype=float)[None, :]
    b = np.asarray(b, dtype=float)[None, :]

    def _antiderivative(t):
        d = t - x
        ad = np.abs(d)
        safe = np.where(ad > 0.0, ad, 1.0)
        return np.where(ad > 0.0, d * (np.log(safe) - 1.0), 0.0)

    return _antiderivative(b) - _antiderivative(a)


def segment_grid(extent, segments):
    """Uniform segment edges on ``[-extent, extent]``."""
    if extent <= 0 or segments < 4:
        raise ValueError("invalid grid parameters")
    return np.linspace(-extent, extent, segments + 1)


def half_plane_influence(centres, edges, effective_modulus, reference):
    """Settlement compliance ``S`` (``w = S @ p``) of a half-plane surface.

    ``centres`` are the collocation points (segment centres), ``edges`` the
    segment edges and ``reference`` the optional x-coordinate(s) whose
    settlement defines the rigid-body datum (``None`` keeps the raw, symmetric
    Flamant kernel and lets the indenter depth absorb the datum).  ``w`` is the
    settlement *into* the half-space, so a compressive pressure gives ``w > 0``.
    """
    a = edges[:-1]
    b = edges[1:]
    x = np.asarray(centres, dtype=float)
    integral = _log_segment_integral(x, a, b)
    if reference is not None:
        ref = np.atleast_1d(np.asarray(reference, dtype=float))
        integral = integral - _log_segment_integral(ref, a, b).mean(axis=0)
    return -(2.0 / (math.pi * effective_modulus)) * integral


# --------------------------------------------------------------------------- #
# solutions
# --------------------------------------------------------------------------- #
@dataclass
class NormalSolution:
    pressure: np.ndarray
    gap: np.ndarray
    active: np.ndarray
    load: float
    half_width: float
    peak_pressure: float
    indentation: float
    complementarity_residual: float
    iterations: int
    converged: bool
    grid: dict = field(default_factory=dict)


@dataclass
class TangentialSolution:
    pressure: np.ndarray
    traction: np.ndarray
    gap: np.ndarray
    slip: np.ndarray
    active: np.ndarray
    stick: np.ndarray
    normal_load: float
    tangential_load: float
    contact_half_width: float
    stick_half_width: float
    peak_pressure: float
    tangential_translation: float
    complementarity_residual: float
    stick_residual: float
    cone_residual: float
    iterations: int
    converged: bool
    grid: dict = field(default_factory=dict)


def _lcp_normal_load_control(S, profile, lengths, target_load, *, max_iterations=200):
    """Exact active set for the *load-controlled* Signorini problem.

    For a fixed active set the pressure is affine in the indentation,
    ``p = delta A^-1 1 - A^-1 d``, so the indentation reproducing the requested
    contact load is obtained in closed form and no bisection is required.
    """
    n = len(profile)
    active = np.zeros(n, dtype=bool)
    active[int(np.argmin(profile))] = True
    pressure = np.zeros(n)
    gap = profile.copy()
    indentation = 0.0
    converged = False
    it = 0
    for it in range(1, max_iterations + 1):
        idx = np.nonzero(active)[0]
        A = S[np.ix_(idx, idx)]
        unit = np.linalg.solve(A, np.ones(len(idx)))
        shift = np.linalg.solve(A, profile[idx])
        span = float(np.sum(unit * lengths[idx]))
        if span <= 0.0:
            raise RuntimeError("degenerate compliance block")
        indentation = (target_load + float(np.sum(shift * lengths[idx]))) / span
        pressure = np.zeros(n)
        pressure[idx] = indentation * unit - shift
        gap = S @ pressure + profile - indentation
        ptol = 1e-12 * max(1.0, float(np.max(np.abs(pressure))))
        g_tol = 1e-13 * max(float(np.max(np.abs(profile))), abs(indentation), 1e-30)
        release = active & (pressure < -ptol)
        engage = (~active) & (gap < -g_tol)
        if not release.any() and not engage.any():
            converged = True
            break
        active = (active & ~release) | engage
    pressure = np.where(active, pressure, 0.0)
    gap = S @ pressure + profile - indentation
    return pressure, gap, active, it, converged, indentation


def _half_width_from_gap(centres, gap, active):
    """Contact half-width from the edge behaviour ``gap ~ (x^2-b^2)^(3/2)``.

    Outside the contact ``gap^(2/3)`` is linear in ``x^2``; the zero intercept
    gives ``b^2``.  Falls back to the outermost loaded segment centre.
    """
    if not active.any():
        return 0.0
    b_fallback = float(np.max(np.abs(centres[active])))
    outside = np.nonzero((~active) & (np.abs(centres) > 0.0))[0]
    if outside.size < 3:
        return b_fallback
    pick = outside[np.argsort(np.abs(centres[outside]))[:6]]
    xo = np.abs(centres[pick])
    go = np.clip(gap[pick], 0.0, None)
    if np.any(go <= 0.0) or xo.size < 3:
        return b_fallback
    slope, intercept = np.polyfit(xo ** 2, go ** (2.0 / 3.0), 1)
    if slope <= 0.0:
        return b_fallback
    b_squared = -intercept / slope
    if not np.isfinite(b_squared) or b_squared <= 0.0:
        return b_fallback
    b = math.sqrt(b_squared)
    if not (0.5 * b_fallback <= b <= 2.0 * max(float(xo.max()), b_fallback)):
        return b_fallback
    return b


def _stick_half_width(centres, active, stick, traction, pressure, mu):
    """Stick-zone half-width from the crossing of ``|q| / (mu p)`` through one."""
    idx = np.nonzero(active)[0]
    if idx.size == 0:
        return 0.0
    idx = idx[np.argsort(np.abs(centres[idx]))]
    x = np.abs(centres[idx])
    ratio = np.abs(traction[idx]) / np.maximum(mu * pressure[idx], 1e-300)
    last = None
    for k, i in enumerate(idx):
        if stick[i] and ratio[k] <= 1.0 + 1e-12:
            last = k
        elif not stick[i] and last is not None:
            r0, r1 = ratio[last], ratio[k]
            t = 1.0 if r1 <= r0 else (1.0 - r0) / (r1 - r0)
            return float(x[last] + t * (x[k] - x[last]))
    if last is not None:
        return float(x[last])
    return float(x[0])


def _bisect_load(evaluate, target, *, bracket_hi=1.0, iterations=200, rtol=1e-10):
    """Monotone bisection: find the control variable giving ``evaluate == target``."""
    lo, hi = 0.0, bracket_hi
    while evaluate(hi) < target:
        hi *= 2.0
        if hi > 1e6:
            raise RuntimeError("load bracket failed")
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        value = evaluate(mid)
        if abs(value / target - 1.0) < rtol:
            return mid
        if value < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def solve_hertz_2d(*, load, radius, young, poisson, extent_factor=3.0, segments=400,
                   reference_extent=None, indentation=None):
    """2-D Hertz contact of a rigid cylinder on an elastic half-space.

    With ``indentation=None`` the contact load is prescribed exactly; otherwise
    the rigid indenter depth is prescribed (used for the no-contact negative
    control, where a retracted indenter must produce a strictly zero force).
    """
    e_star = plane_strain_modulus(young, poisson)
    target = hertz_cylinder_reference(load=load, radius=radius, effective_modulus=e_star)
    edges = segment_grid(extent_factor * target.half_width, segments)
    centres = 0.5 * (edges[:-1] + edges[1:])
    ref = reference_extent
    S = half_plane_influence(centres, edges, e_star, reference=None if ref is None else [-ref, ref])
    profile = centres ** 2 / (2.0 * radius)
    lengths = np.diff(edges)

    if indentation is None:
        pressure, gap, active, it, converged, depth = _lcp_normal_load_control(
            S, profile, lengths, load)
    else:
        depth = indentation
        pressure = np.zeros(len(centres))
        active = np.zeros(len(centres), dtype=bool)
        gap = profile - depth
        converged = bool(np.all(gap >= 0.0))
        it = 1
        if not converged:
            raise ValueError("indentation-control path only supports open contact")
    total = float(np.sum(pressure * lengths))
    b = _half_width_from_gap(centres, gap, active)
    return NormalSolution(
        pressure=pressure, gap=gap, active=active, load=total, half_width=b,
        peak_pressure=float(np.max(pressure)) if pressure.size else 0.0,
        indentation=depth,
        complementarity_residual=float(np.max(np.abs(gap * pressure))),
        iterations=it, converged=converged,
        grid={"centres": centres, "edges": edges, "compliance": S,
              "reference": ref, "lengths": lengths, "target": target},
    )


def solve_cattaneo_mindlin_2d(*, normal_load, tangential_load, friction, radius,
                              young, poisson, extent_factor=3.0, segments=400,
                              max_iterations=200):
    """2-D Cattaneo--Mindlin partial slip: Hertz preload, then tangential load.

    For a fixed stick set the traction is affine in the rigid tangential
    translation ``u_t``, ``q = u_t A^-1 1 - A^-1 (S q_slip)``, so the ``u_t``
    that reproduces the requested tangential load follows in closed form --
    the same load-controlled active set as the normal problem.
    """
    if friction <= 0.0:
        raise ValueError("friction coefficient must be positive")
    if tangential_load >= friction * normal_load:
        raise ValueError("tangential load must stay inside the friction limit")
    normal = solve_hertz_2d(load=normal_load, radius=radius, young=young,
                            poisson=poisson, extent_factor=extent_factor,
                            segments=segments)
    grid = normal.grid
    S, centres, lengths = grid["compliance"], grid["centres"], grid["lengths"]
    active, pressure = normal.active, normal.pressure
    cone = friction * pressure
    traction = np.zeros_like(pressure)
    stick = active.copy()
    u_t = 0.0
    converged = False
    it = 0
    for it in range(1, max_iterations + 1):
        idx = np.nonzero(stick)[0]
        slip = np.nonzero(active & ~stick)[0]
        traction = np.zeros_like(pressure)
        traction[slip] = cone[slip]
        if idx.size:
            A = S[np.ix_(idx, idx)]
            unit = np.linalg.solve(A, np.ones(len(idx)))
            shift = np.linalg.solve(A, S[np.ix_(idx, slip)] @ traction[slip]) if slip.size else np.zeros(len(idx))
            span = float(np.sum(unit * lengths[idx]))
            slip_load = float(np.sum(traction[slip] * lengths[slip])) if slip.size else 0.0
            u_t = (tangential_load - slip_load + float(np.sum(shift * lengths[idx]))) / span
            traction[idx] = u_t * unit - shift
        else:
            u_t = 0.0
        scale = max(1.0, float(np.max(cone[active])))
        over = stick & (traction > cone + 1e-10 * scale)
        negative = stick & (traction < -1e-10 * scale)
        if not over.any() and not negative.any():
            converged = True
            break
        stick = stick & ~(over | negative)
    traction = np.where(active, traction, 0.0)
    total = float(np.sum(traction * lengths))
    slip_gap = S @ traction - u_t
    c = _stick_half_width(centres, active, stick, traction, pressure, friction)
    return TangentialSolution(
        pressure=pressure, traction=traction, gap=normal.gap, slip=slip_gap,
        active=active, stick=stick, normal_load=normal.load,
        tangential_load=total, contact_half_width=normal.half_width,
        stick_half_width=c, peak_pressure=normal.peak_pressure,
        tangential_translation=u_t,
        complementarity_residual=float(np.max(np.abs(normal.gap * pressure))),
        stick_residual=float(np.max(np.abs(slip_gap[stick]))) if stick.any() else 0.0,
        cone_residual=float(np.max(np.abs(traction[active]) - cone[active])),
        iterations=it, converged=converged,
        grid={"centres": centres, "edges": grid["edges"], "lengths": lengths,
              "target": grid["target"]},
    )


# --------------------------------------------------------------------------- #
# finite-element node-to-node contact companion (documented contrast case)
# --------------------------------------------------------------------------- #
def solve_indentation_fe(nodes, elems, K, surface_nodes, radius, indentation,
                         *, solver=None, fixed_dofs=None, max_iterations=200):
    """Plane-strain Q4 contact of a rigid parabolic indenter, node-to-node.

    The unilateral constraint is enforced exactly with Lagrange multipliers, so
    this route shares the complementarity property of the influence solver.
    Its purpose in this module is the documented *contrast*: bilinear elements
    with pointwise contact constraints reproduce the contact half-width but not
    the smooth pressure field -- the nodal contact forces alternate in sign
    (``sawtooth``), which corrupts the peak pressure and the stick boundary.
    Returns the settlement/pressure field and a ``sawtooth`` indicator.
    """
    x = np.asarray(nodes[surface_nodes, 0], dtype=float)
    profile = x ** 2 / (2.0 * radius)
    welded = x[1:] - x[:-1]
    tributary = np.zeros(len(x))
    tributary[1:-1] = 0.5 * (x[2:] - x[:-2])
    tributary[0] = 0.5 * welded[0]
    tributary[-1] = 0.5 * welded[-1]
    if solver is None:
        fixed = fixed_dofs if fixed_dofs is not None else dirichlet_dofs(nodes, _infer_ny(nodes))
        solver = ElasticSolver(K, fixed)
    # gap = settlement into the body minus the rigid indenter surface:
    #   g = -u_y + profile - indentation >= 0,   g < 0  =>  penetration
    active = profile < indentation
    forces = np.zeros(len(x))
    gap = profile - indentation
    converged = False
    it = 0
    loads = np.zeros(len(solver.free))
    for it in range(1, max_iterations + 1):
        dofs = [2 * int(surface_nodes[k]) + 1 for k in np.nonzero(active)[0]]
        targets = (profile - indentation)[active]
        u, lam = solver.solve_constrained(dofs, targets, loads)
        gap = -u[2 * surface_nodes + 1] - indentation + profile
        forces = np.zeros(len(x))
        forces[active] = lam
        scale = max(1.0, float(np.max(np.abs(forces))))
        release = active & (forces < -1e-9 * scale)
        engage = (~active) & (gap < -1e-14)
        if not release.any() and not engage.any():
            converged = True
            break
        active = (active & ~release) | engage
    pressure = forces / tributary
    adjacent = np.nonzero(active[:-1] & active[1:])[0]
    flips = int(np.sum(np.sign(pressure[adjacent]) * np.sign(pressure[adjacent + 1]) < 0))
    return {
        "displacement": u,
        "gap": gap,
        "pressure": pressure,
        "forces": forces,
        "active": active,
        "load": float(np.sum(forces)),
        "contact_extent": float(np.max(np.abs(x[active]))) if active.any() else 0.0,
        "peak_pressure": float(np.max(pressure)) if active.any() else 0.0,
        "complementarity_residual": float(np.max(np.abs(gap * forces))),
        "sawtooth_pairs": flips,
        "adjacent_pairs": int(adjacent.size),
        "sawtooth_fraction": float(flips / max(adjacent.size, 1)),
        "iterations": it,
        "converged": converged,
    }


def _infer_ny(nodes):
    """Recover the structured block ``ny`` from a :func:`block_mesh` node array."""
    ys = np.unique(np.round(nodes[:, 1], 15))
    return len(ys) - 1


# --------------------------------------------------------------------------- #
# 3-D Hertz by axisymmetric disc superposition (Boussinesq / Love kernel)
# --------------------------------------------------------------------------- #
def _disc_settlement(radii, disc_radii, young, poisson):
    """Settlement at ``radii`` under unit uniform pressure on discs of ``disc_radii``.

    Love's closed form for a uniformly loaded circular area (plane surface of a
    half-space)::

        r <= a :  u_z = (4(1-nu^2) a/(pi E)) E(r/a)
        r >  a :  u_z = (4(1-nu^2) r/(pi E)) [E(a/r) - (1-(a/r)^2) K(a/r)]

    with ``E``/``K`` the complete elliptic integrals (scipy's parameter
    convention ``m = k^2``).
    """
    base = 4.0 * (1.0 - poisson ** 2) / (math.pi * young)
    r = np.asarray(radii, dtype=float)[:, None]
    a = np.asarray(disc_radii, dtype=float)[None, :]
    safe_a = np.where(a > 0.0, a, 1.0)
    safe_r = np.where(r > 0.0, r, 1.0)
    inside = r <= a
    m_in = np.clip((r / safe_a) ** 2, 0.0, 1.0)
    m_out = np.clip((a / safe_r) ** 2, 0.0, 1.0 - 1e-15)
    outside_value = base * r * (ellipe(m_out) - (1.0 - m_out) * ellipk(m_out))
    return np.where(inside, base * a * ellipe(m_in), outside_value)


def _fit_hertz_disc(radii, pressure, guess_radius):
    """Least-squares fit of ``p0 sqrt(1-(r/a)^2)`` to the axisymmetric pressure."""
    keep = pressure > 0.0
    if keep.sum() < 4:
        raise ValueError("not enough loaded rings for a Hertz fit")
    rs, ps = radii[keep], pressure[keep]
    best = None
    for a in np.linspace(0.7 * guess_radius, 1.4 * guess_radius, 281):
        shape = np.sqrt(np.clip(1.0 - (rs / a) ** 2, 0.0, None))
        denom = float(shape @ shape)
        if denom <= 0.0:
            continue
        p0 = float(shape @ ps) / denom
        residual = float(np.sum((ps - p0 * shape) ** 2))
        if best is None or residual < best[0]:
            best = (residual, a, p0)
    return {"residual": best[0], "contact_radius": best[1], "peak_pressure": best[2]}


def solve_hertz_sphere_3d(*, radius, young, poisson, load=None, indentation=None,
                          extent_factor=3.0, rings=300, scale_load=5.0e5):
    """3-D Hertz contact of a rigid sphere on an elastic half-space.

    Axisymmetric disc superposition with Love's closed-form disc compliance, and
    the same exact-complementarity active set as the 2-D solver.  Prescribe
    either the load (``load``) or the rigid indenter depth (``indentation``); the
    latter is used for the load--approach law ``P(delta)``.
    """
    if (load is None) == (indentation is None):
        raise ValueError("give exactly one of load/indentation")
    e_star = plane_strain_modulus(young, poisson)
    if load is not None:
        target = hertz_sphere_reference(load=load, radius=radius,
                                        effective_modulus=e_star)
        guess_radius = target.contact_radius
    elif indentation > 0.0:
        target = hertz_sphere_reference(approach=indentation, radius=radius,
                                        effective_modulus=e_star)
        guess_radius = target.contact_radius
    else:                       # open contact: mesh scaled by a nominal load
        target = None
        guess_radius = hertz_sphere_reference(load=scale_load, radius=radius,
                                              effective_modulus=e_star).contact_radius
    edges = np.linspace(0.0, extent_factor * guess_radius, rings + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    discs = edges[1:]
    previous = np.concatenate(([0.0], edges[1:-1]))
    S = (_disc_settlement(centres, discs, young, poisson)
         - _disc_settlement(centres, previous, young, poisson))
    profile = centres ** 2 / (2.0 * radius)
    areas = math.pi * (discs ** 2 - previous ** 2)

    if load is not None:
        pressure, gap, active, it, converged, depth = _lcp_normal_load_control(
            S, profile, areas, load)
    else:
        depth = indentation
        pressure = np.zeros(len(centres))
        active = np.zeros(len(centres), dtype=bool)
        gap = profile - depth
        converged = bool(np.all(gap >= 0.0))
        it = 1
        if not converged:
            raise ValueError("indentation control only supports open contact here")
    total = float(np.sum(pressure * areas))
    fit = _fit_hertz_disc(centres, pressure, guess_radius) if pressure.any() else {
        "contact_radius": 0.0, "peak_pressure": 0.0, "residual": 0.0}
    return NormalSolution(
        pressure=pressure, gap=gap, active=active, load=total,
        half_width=fit["contact_radius"], peak_pressure=fit["peak_pressure"],
        indentation=depth,
        complementarity_residual=float(np.max(np.abs(gap * pressure))),
        iterations=it, converged=converged,
        grid={"centres": centres, "edges": edges, "areas": areas,
              "compliance": S, "target": target},
    )


# --------------------------------------------------------------------------- #
# plane-strain Q4 finite element (kernel audit companion)
# --------------------------------------------------------------------------- #
_GAUSS = ((-1.0 / math.sqrt(3.0), -1.0 / math.sqrt(3.0)),
          (1.0 / math.sqrt(3.0), -1.0 / math.sqrt(3.0)),
          (1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)),
          (-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)))


def block_mesh(nx: int, ny: int, half_width: float, depth: float, *, grading: float = 2.0):
    """Structured Q4 block ``x in [-half_width, half_width]``, ``y in [-depth, 0]``.

    Both directions are graded (``|u|**grading``) so nodes cluster towards the
    contact centre ``x=0`` and the contact surface ``y=0``.  Node ``j=0`` is the
    top (free) surface; node ordering is ``i*(ny+1)+j``.
    """
    if nx < 4 or ny < 4 or half_width <= 0 or depth <= 0 or grading < 1.0:
        raise ValueError("invalid mesh parameters")
    u = np.linspace(-1.0, 1.0, nx + 1)
    x = np.sign(u) * np.abs(u) ** grading * half_width
    v = np.linspace(0.0, 1.0, ny + 1)
    y = -depth * v ** grading
    nodes = np.stack([np.repeat(x, ny + 1), np.tile(y, nx + 1)], axis=1)

    def nid(i, j):
        return i * (ny + 1) + j

    # counter-clockwise in the physical (x, y) plane: y decreases with j, so the
    # physical bottom-left corner of cell (i, j) is node (i, j+1)
    elems = np.array([[nid(i, j + 1), nid(i + 1, j + 1), nid(i + 1, j), nid(i, j)]
                      for i in range(nx) for j in range(ny)], dtype=np.int64)
    return nodes, elems


def plane_strain_stiffness(nodes, elems, young, poisson):
    """Assemble the linear plane-strain stiffness matrix (sparse CSR).

    Vectorised over elements; the 2x2 Jacobian inverse is written out in closed
    form because ``np.linalg.solve`` on 2x2 blocks dominates the runtime
    otherwise on this platform.
    """
    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    mu = young / (2.0 * (1.0 + poisson))
    D = np.array([[lam + 2.0 * mu, lam, 0.0],
                  [lam, lam + 2.0 * mu, 0.0],
                  [0.0, 0.0, mu]])
    nel = len(elems)
    coord = nodes[elems]
    Ke = np.zeros((nel, 8, 8))
    for xi, eta in _GAUSS:
        dN_dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
        dN_deta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
        J00 = coord[:, :, 0] @ dN_dxi
        J01 = coord[:, :, 1] @ dN_dxi
        J10 = coord[:, :, 0] @ dN_deta
        J11 = coord[:, :, 1] @ dN_deta
        detJ = J00 * J11 - J01 * J10
        inv00, inv01 = J11 / detJ, -J01 / detJ
        inv10, inv11 = -J10 / detJ, J00 / detJ
        dN = np.empty((nel, 2, 4))
        dN[:, 0, :] = inv00[:, None] * dN_dxi + inv01[:, None] * dN_deta
        dN[:, 1, :] = inv10[:, None] * dN_dxi + inv11[:, None] * dN_deta
        B = np.zeros((nel, 3, 8))
        B[:, 0, 0::2] = dN[:, 0, :]
        B[:, 1, 1::2] = dN[:, 1, :]
        B[:, 2, 0::2] = dN[:, 1, :]
        B[:, 2, 1::2] = dN[:, 0, :]
        Ke += np.einsum("nki,kl,nlj->nij", B, D, B, optimize=True) * detJ[:, None, None]
    dofs = np.empty((nel, 8), dtype=np.int64)
    dofs[:, 0::2] = 2 * elems
    dofs[:, 1::2] = 2 * elems + 1
    ndof = 2 * len(nodes)
    return sp.coo_matrix((Ke.ravel(),
                          (np.repeat(dofs, 8, axis=1).ravel(), np.tile(dofs, (1, 8)).ravel())),
                         shape=(ndof, ndof)).tocsr()


def surface_nodes(nodes, ny):
    """Indices of the Q4 top-surface nodes (``j = 0``)."""
    return np.arange(0, len(nodes), ny + 1)


def dirichlet_dofs(nodes, ny):
    """Zero-displacement supports: bottom face plus vertical side faces."""
    idx = np.arange(len(nodes)).reshape(-1, ny + 1)   # idx[i, j] = i*(ny+1)+j
    dofs = []
    dofs.extend((2 * idx[:, ny]).tolist())            # bottom face, u_x
    dofs.extend((2 * idx[:, ny] + 1).tolist())        # bottom face, u_y
    dofs.extend((2 * idx[0, :]).tolist())             # left face (i = 0), u_x
    dofs.extend((2 * idx[-1, :]).tolist())            # right face (i = nx), u_x
    return np.array(sorted(set(int(d) for d in dofs)), dtype=np.int64)


class ElasticSolver:
    """Cached sparse factorisation of the free stiffness block."""

    def __init__(self, K, fixed):
        ndof = K.shape[0]
        mask = np.ones(ndof, dtype=bool)
        mask[fixed] = False
        self.free = np.nonzero(mask)[0]
        self.ndof = ndof
        self.remap = -np.ones(ndof, dtype=np.int64)
        self.remap[self.free] = np.arange(len(self.free))
        self.factor = spla.splu(K[self.free][:, self.free].tocsc())

    def solve(self, loads):
        """Solve ``K u = f`` with zero Dirichlet data; returns the full vector."""
        u = np.zeros(self.ndof)
        u[self.free] = self.factor.solve(loads)
        return u

    def solve_constrained(self, constraints, targets, loads):
        """Solve ``K u = f - C' lam`` with ``C u = t`` (exact Lagrange multipliers)."""
        nc = len(constraints)
        u = np.zeros(self.ndof)
        if nc == 0:
            u[self.free] = self.factor.solve(loads)
            return u, np.zeros(0)
        cols = np.empty((len(self.free), nc))
        for j, dof in enumerate(constraints):
            if self.remap[dof] < 0:
                raise ValueError("constraint applied to a fixed dof")
            e = np.zeros(len(self.free))
            e[self.remap[dof]] = 1.0
            cols[:, j] = self.factor.solve(e)
        y = self.factor.solve(loads)
        rows = [self.remap[c] for c in constraints]
        S = cols[rows, :]
        rhs = y[rows] - np.asarray(targets, dtype=float)
        lam = np.linalg.solve(S, rhs)
        u[self.free] = y - cols @ lam
        return u, lam