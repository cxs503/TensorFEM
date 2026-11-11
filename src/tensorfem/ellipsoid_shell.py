"""Ellipsoid-of-revolution shell under uniform internal pressure.

Purpose (case-pool benchmark, anchor CASE-NAFEMS-LE11)
------------------------------------------------------
A closed spheroidal pressure vessel is the canonical curved-shell membrane case
that is *not* a surface of constant curvature: along the meridian both principal
radii ``R1`` (meridional) and ``R2`` (second principal, ``= r / sin(phi)``)
vary, and for an oblate spheroid the hoop force changes **sign** between the
pole and the equator.  That sign change is what makes the case diagnostic: a
formulation that only reproduces the constant-curvature sphere (``N = pR/2``)
can still be wrong here.

Geometry
--------
Spheroid ``r^2/a^2 + z^2/b^2 = 1`` with equatorial radius ``a`` (``r``) and
polar semi-axis ``b`` (``z``), parametrised by the eccentric angle
``theta in [0, pi/2]``::

    r(theta) = a sin(theta),    z(theta) = b cos(theta)
    ds/dtheta = W = sqrt(a^2 cos^2 theta + b^2 sin^2 theta)
    cos(phi) = dr/ds = a cos(theta)/W      sin(phi) = -dz/ds = b sin(theta)/W
    R1 = W^3 / (a b)                       R2 = r / sin(phi) = a W / b

``b > a`` is prolate, ``b < a`` oblate, ``b = a`` the sphere (``R1 = R2 = a``).

Membrane reference (derived here, not transcribed)
--------------------------------------------------
Axial equilibrium of the cap cut at ``r`` gives the meridional force exactly

    N_phi = p r / (2 sin(phi)) = p R2 / 2,

and normal equilibrium ``N_phi/R1 + N_theta/R2 = p`` then gives

    N_theta = p R2 (1 - R2 / (2 R1)).

Substituting the spheroid expressions::

    N_phi   = p a W / (2 b)
    N_theta = p (a W / b) (1 - a^2 / (2 W^2))

Both formulas are checked in :mod:`tests.test_ellipsoid_shell` against the exact
sphere limit (``p a / 2``), against the pole/equator closed forms
(``p a^2/(2b)``, ``p a(1 - a^2/(2b^2))``) and against the FE axial reaction.

The *displacement* reference is not closed-form for a spheroid.  It comes from
an independent Rayleigh-Ritz minimisation of the membrane potential over
polynomial-in-``theta`` trial spaces (see :func:`membrane_shape`); that reference
also reproduces the closed-form forces above, which self-checks it.  It shares
no discretisation with the two-node ring element, so agreement is not tautology.

Numerical model
---------------
The meridian pole -> equator is meshed with nodes exactly on the ellipse; each
:class:`~tensorfem.axisym_shell.Meridian` receives its own osculating radius
``R1`` evaluated at the element midpoint (the 2-node ring element carries a
constant curvature per element, so a spheroid is approximated piecewise by
circular arcs -- second-order geometric error, measured by the convergence
study).  Symmetry boundary conditions: ``u = beta = 0`` at both ends.
"""
from __future__ import annotations

import math

import torch

from .axisym_shell import AxisymShell

__all__ = [
    "meridian_curvature_radius",
    "spheroid_theta",
    "spheroid_nodes",
    "spheroid_element_radii",
    "membrane_forces",
    "membrane_shape",
    "ellipsoid_pressure",
]


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def _wrap(a: float, b: float, theta: float) -> float:
    """``W = ds/dtheta``."""
    return math.hypot(a * math.cos(theta), b * math.sin(theta))


def meridian_curvature_radius(a: float, b: float, theta: float) -> float:
    """Meridional radius of curvature ``R1 = W^3/(a b)``."""
    return _wrap(a, b, theta) ** 3 / (a * b)


def spheroid_theta(a: float, b: float, facets: int, grading: float = 1.0) -> list[float]:
    """Eccentric angles of the meridional nodes (``grading`` clusters to the equator)."""
    if facets < 1:
        raise ValueError("facets must be >= 1")
    if grading <= 0.0:
        raise ValueError("grading must be positive")
    return [0.5 * math.pi * (i / facets) ** grading for i in range(facets + 1)]


def spheroid_nodes(a: float, b: float, facets: int,
                   grading: float = 1.0) -> list[tuple[float, float]]:
    """Meridional nodes ``(r, z)`` from the pole (theta=0) to the equator."""
    return [(a * math.sin(t), b * math.cos(t)) for t in spheroid_theta(a, b, facets, grading)]


def spheroid_element_radii(a: float, b: float, facets: int,
                           grading: float = 1.0) -> list[float]:
    """Osculating meridional radius ``R1`` at each element midpoint."""
    th = spheroid_theta(a, b, facets, grading)
    return [meridian_curvature_radius(a, b, 0.5 * (th[i] + th[i + 1])) for i in range(facets)]


# --------------------------------------------------------------------------- #
# membrane reference
# --------------------------------------------------------------------------- #
def membrane_forces(a: float, b: float, pressure: float, theta: float) -> tuple[float, float]:
    """``(N_phi, N_theta)`` membrane forces at eccentric angle ``theta``."""
    w = _wrap(a, b, theta)
    n_phi = pressure * a * w / (2.0 * b)
    n_theta = pressure * (a * w / b) * (1.0 - a * a / (2.0 * w * w))
    return n_phi, n_theta


def membrane_shape(a: float, b: float, thickness: float, young: float, poisson: float,
                   pressure: float, *, order: int = 24, rule: int = 240) -> dict:
    """Membrane displacement field from an independent Rayleigh-Ritz solve.

    The spheroidal membrane problem has no elementary closed-form displacement
    solution, so the reference for ``w`` is obtained by minimising the thin-shell
    membrane potential

        Pi(u, w) = 1/2 int (eps_s N_s + eps_theta N_theta) 2 pi r ds
                   - int p w 2 pi r ds

    over polynomial-in-``theta`` trial spaces (``u`` vanishes at the pole and at
    the equator by symmetry; ``w`` is free).  Trial functions and quadrature are
    unrelated to the two-node ring element, and the same solve reproduces the
    closed-form membrane forces of :func:`membrane_forces`, which is used as the
    self-check of this reference (:func:`membrane_shape` returns
    ``N_selfcheck``).
    """
    # Gauss-Legendre rule on theta in (0, pi/2)
    x, wq = _gauss_legendre(rule)
    theta = [0.25 * math.pi * (xi + 1.0) for xi in x]
    wq = [0.25 * math.pi * w for w in wq]
    geo = []
    for th in theta:
        ds = _wrap(a, b, th)
        r = a * math.sin(th)
        cn = a * math.cos(th) / ds
        sn = b * math.sin(th) / ds
        r1 = meridian_curvature_radius(a, b, th)
        geo.append((th, ds, r, cn, sn, r1, 2.0 * math.pi * r * ds))

    n_u, n_w = order, order + 1

    def shape_u(k: int, th: float) -> tuple[float, float]:
        # u_k = sin(2 k th): zero at both ends
        return math.sin(2.0 * k * th), 2.0 * k * math.cos(2.0 * k * th)

    def shape_w(k: int, th: float) -> float:
        # w_0 = 1, w_k = cos(2 k th)
        return 1.0 if k == 0 else math.cos(2.0 * k * th)

    def dshape_w(k: int, th: float) -> float:
        return 0.0 if k == 0 else -2.0 * k * math.sin(2.0 * k * th)

    n = n_u + n_w
    kmat = [[0.0] * n for _ in range(n)]
    fvec = [0.0] * n
    cm = young * thickness / (1.0 - poisson * poisson)

    def basis(i: int, th: float, ds: float, r: float, cn: float, sn: float, r1: float):
        """Return (eps_s, eps_theta) of basis function i at the point."""
        if i < n_u:
            val, dval = shape_u(i + 1, th)
            # du/ds = dval * dtheta/ds
            return dval / ds, val * cn / r
        k = i - n_u
        val = shape_w(k, th)
        dval = dshape_w(k, th)
        return val / r1, val * sn / r

    for (th, ds, r, cn, sn, r1, dA), wqg in zip(geo, wq):
        # strain-displacement rows for all trial functions
        rows = [basis(i, th, ds, r, cn, sn, r1) for i in range(n)]
        const = cm * wqg * dA
        for i in range(n):
            esi, eti = rows[i]
            for j in range(i, n):
                esj, etj = rows[j]
                v = esi * esj + poisson * (esi * etj + eti * esj) + eti * etj
                kmat[i][j] += const * v
        for i in range(n):
            if i >= n_u:                      # only w carries the pressure
                fvec[i] += pressure * wqg * dA * shape_w(i - n_u, th)
    for i in range(n):
        for j in range(i):
            kmat[i][j] = kmat[j][i]

    coef = _solve_dense(kmat, fvec)

    def field_at(th: float) -> dict:
        ds = _wrap(a, b, th)
        r = a * math.sin(th)
        cn = a * math.cos(th) / ds
        sn = b * math.sin(th) / ds
        r1 = meridian_curvature_radius(a, b, th)
        u = sum(coef[i] * shape_u(i + 1, th)[0] for i in range(n_u))
        w = sum(coef[n_u + k] * shape_w(k, th) for k in range(n_w))
        eps_s = sum(coef[i] * basis(i, th, ds, r, cn, sn, r1)[0] for i in range(n))
        eps_t = sum(coef[i] * basis(i, th, ds, r, cn, sn, r1)[1] for i in range(n))
        return {"theta": th, "u": u, "w": w, "eps_s": eps_s, "eps_theta": eps_t,
                "N_phi": cm * (eps_s + poisson * eps_t),
                "N_theta": cm * (eps_t + poisson * eps_s)}

    prof = [field_at(th) for th in theta]
    n_phi_err = max(abs(p_["N_phi"] - membrane_forces(a, b, pressure, p_["theta"])[0])
                    / abs(membrane_forces(a, b, pressure, p_["theta"])[0]) for p_ in prof)
    n_th_err = max(abs(p_["N_theta"] - membrane_forces(a, b, pressure, p_["theta"])[1])
                   / abs(membrane_forces(a, b, pressure, p_["theta"])[1]) for p_ in prof)
    pole = field_at(1e-9)
    equator = field_at(0.5 * math.pi)
    return {
        "w_pole": pole["w"],
        "w_equator": equator["w"],
        "u_equator": equator["u"],
        "theta": theta,
        "u": [p_["u"] for p_ in prof],
        "w": [p_["w"] for p_ in prof],
        "N_selfcheck_phi": n_phi_err,
        "N_selfcheck_theta": n_th_err,
        "order": order,
    }


def _gauss_legendre(n: int) -> tuple[list[float], list[float]]:
    """Gauss-Legendre nodes/weights on [-1, 1] (Newton on Legendre polynomials)."""
    nodes, weights = [], []
    for i in range(1, n + 1):
        x = math.cos(math.pi * (i - 0.25) / (n + 0.5))
        for _ in range(100):
            p0, p1 = 1.0, x
            for k in range(2, n + 1):
                p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
            dp = n * (x * p1 - p0) / (x * x - 1.0)
            dx = -p1 / dp
            x += dx
            if abs(dx) < 1e-16:
                break
        p0, p1 = 1.0, x
        for k in range(2, n + 1):
            p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
        dp = n * (x * p1 - p0) / (x * x - 1.0)
        nodes.append(x)
        weights.append(2.0 / ((1.0 - x * x) * dp * dp))
    return nodes, weights


def _solve_dense(k: list[list[float]], f: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting on a small dense system."""
    n = len(f)
    m = [row[:] + [f[i]] for i, row in enumerate(k)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-300:
            raise ValueError("singular membrane Rayleigh-Ritz system")
        m[col], m[piv] = m[piv], m[col]
        pv = m[col][col]
        for r in range(col + 1, n):
            fac = m[r][col] / pv
            if fac:
                for c in range(col, n + 1):
                    m[r][c] -= fac * m[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        s = m[r][n] - sum(m[r][c] * x[c] for c in range(r + 1, n))
        x[r] = s / m[r][r]
    return x


# --------------------------------------------------------------------------- #
# finite element case
# --------------------------------------------------------------------------- #
def ellipsoid_pressure(*, a: float = 1.0, b: float = 0.5, thickness: float = 0.01,
                       young: float = 1.0, poisson: float = 0.3, pressure: float = 1.0,
                       facets: int = 16, shear_integration: str = "reduced",
                       grading: float = 1.0) -> dict:
    """Half-meridian (pole -> equator) model of a pressurised spheroid."""
    nodes = spheroid_nodes(a, b, facets, grading)
    radii = spheroid_element_radii(a, b, facets, grading)
    model = AxisymShell(nodes, young=young, poisson=poisson, thickness=thickness,
                        pressure=pressure, meridian_radius=radii,
                        shear_integration=shear_integration)
    u = model.solve({
        model.dof(0, "u"): 0.0, model.dof(0, "beta"): 0.0,
        model.dof(facets, "u"): 0.0, model.dof(facets, "beta"): 0.0,
    })
    field = model.field(u)

    rows = []
    for g in range(len(field["r"])):
        r = field["r"][g]
        sin_theta = min(1.0, max(0.0, r / a))
        theta = math.asin(sin_theta)
        n_phi, n_theta = membrane_forces(a, b, pressure, theta)
        ds = _wrap(a, b, theta)
        sn = b * math.sin(theta) / ds
        rows.append({
            "r": r, "theta": theta, "phi": field["phi"][g],
            "N_s_fe": field["N_s"][g], "N_theta_fe": field["N_theta"][g],
            "N_phi_ref": n_phi, "N_theta_ref": n_theta,
            "M_s_fe": field["M_s"][g], "Q_fe": field["Q"][g],
            "err_phi": abs(field["N_s"][g] - n_phi) / abs(n_phi),
            "err_theta": abs(field["N_theta"][g] - n_theta) / abs(n_theta),
            # axial equilibrium of the cap cut at r: 2 pi r N_phi sin(phi) = p pi r^2
            "axial_residual": (abs(2.0 * math.pi * r * field["N_s"][g] * sn
                                   - pressure * math.pi * r * r)
                               / (pressure * math.pi * r * r)) if r > 0 else 0.0,
        })

    shape = membrane_shape(a, b, thickness, young, poisson, pressure)
    w_fe = [float(u[model.dof(i, "w")]) for i in range(facets + 1)]
    thetas = shape["theta"]
    w_ref = []
    for i in range(facets + 1):
        th = 0.5 * math.pi * i / facets
        j = 0
        while j + 1 < len(thetas) and thetas[j + 1] < th:
            j += 1
        if j + 1 >= len(thetas):
            w_ref.append(shape["w"][-1])
            continue
        t0, t1 = thetas[j], thetas[j + 1]
        f0, f1 = shape["w"][j], shape["w"][j + 1]
        lam = 0.0 if t1 == t0 else (th - t0) / (t1 - t0)
        w_ref.append(f0 + lam * (f1 - f0))
    w_equator_fe = w_fe[-1]
    w_equator_ref = shape["w_equator"]

    pole = max(rows[:2], key=lambda d: d["r"])
    equator = rows[-1]
    return {
        "case": "spheroid_internal_pressure",
        "a": a, "b": b, "thickness": thickness, "young": young, "poisson": poisson,
        "pressure": pressure, "facets": facets, "shear_integration": shear_integration,
        "grading": grading,
        "slenderness_a_over_t": a / thickness,
        "dof": model.ndof,
        "gc": rows,
        "nodes": nodes,
        "w_fe": w_fe,
        "w_ref": w_ref,
        "N_pole_fe": pole["N_s_fe"], "N_pole_ref": pole["N_phi_ref"],
        "N_pole_err": pole["err_phi"],
        "N_theta_pole_fe": pole["N_theta_fe"], "N_theta_pole_ref": pole["N_theta_ref"],
        "N_theta_pole_err": pole["err_theta"],
        "N_phi_equator_fe": equator["N_s_fe"], "N_phi_equator_ref": equator["N_phi_ref"],
        "N_phi_equator_err": equator["err_phi"],
        "N_theta_equator_fe": equator["N_theta_fe"],
        "N_theta_equator_ref": equator["N_theta_ref"],
        "N_theta_equator_err": equator["err_theta"],
        "err_phi_max": max(d["err_phi"] for d in rows),
        "N_scale": max(abs(d["N_phi_ref"]) for d in rows),
        "err_phi_max_scaled": max(abs(d["N_s_fe"] - d["N_phi_ref"]) for d in rows)
                              / max(abs(d["N_phi_ref"]) for d in rows),
        "err_theta_max_scaled": max(abs(d["N_theta_fe"] - d["N_theta_ref"]) for d in rows)
                                / max(abs(d["N_phi_ref"]) for d in rows),
        "err_theta_max": max(d["err_theta"] for d in rows),
        "axial_residual_max": max(d["axial_residual"] for d in rows),
        "M_s_max": max(abs(d["M_s_fe"]) for d in rows),
        "bending_to_membrane": max(abs(d["M_s_fe"]) for d in rows)
                               / max(1e-30, abs(pressure * thickness * a)),
        "w_pole_fe": w_fe[0], "w_pole_ref": shape["w_pole"],
        "w_equator_fe": w_equator_fe, "w_equator_ref": w_equator_ref,
        "w_equator_err": abs(w_equator_fe - w_equator_ref) / abs(w_equator_ref),
        "w_max_abs_err": max(abs(x - y) for x, y in zip(w_fe, w_ref)),
        "membrane_shape": shape,
    }