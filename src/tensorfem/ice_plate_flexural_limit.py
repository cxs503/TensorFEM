"""Flexural (first-crack) limit load of a floating ice sheet on a Winkler bed.

Model
-----
An infinite ice sheet of thickness ``h``, Young's modulus ``E`` and Poisson
ratio ``nu`` floats on water modelled as a Winkler foundation ``k = rho_w g``
(the sheet is at rest, so only the static buoyancy stiffness enters).  A
vertical load is transmitted through a circular contact area of radius ``a``
with a uniform pressure ``p`` -- the standard idealisation of a quasi-static
ice load (Westergaard 1926; Nevel 1969; Kerr 1979; ISO 19906:2010).

Two ingredients are kept strictly apart:

* **Foundation kernel** -- the Q4 Mindlin-Reissner plate + Winkler element,
  the batched assembly and the sparse solve of
  :mod:`tensorfem.foundation_plate_bending` are used *unchanged*
  (:func:`graded_square_mesh`, :func:`assemble_plate_foundation_system`,
  :func:`solve_assembled_system`).  This module does not re-derive them.
* **Failure criterion** -- implemented here, independently of the analytic
  first-crack load of :mod:`tensorfem.levelset.ice_plate_fracture`, which
  multiplies a *closed-form* patch moment by the strength: here the moment
  field comes from the finite-element solution, is turned into surface
  stresses at the Gauss points and the first-crack load follows from the
  Rankine (maximum principal tensile stress) criterion.

Failure criterion
-----------------
For a plate moment state ``(Mx, My, Mxy)`` (per unit width, sagging positive)
the bending stress at the bottom fibre ``z = -h/2`` is ``6 M / h**2`` and at
the top fibre ``-6 M / h**2``; the principal moments are

    M_1,2 = (Mx + My)/2 +- sqrt(((Mx - My)/2)**2 + Mxy**2),

so the largest tensile stress anywhere through the thickness is

    sigma_max = (6/h**2) * max(M_1, -M_2).

Crack initiation is placed at ``sigma_max = sigma_f`` (flexural strength), hence

    P_f = sigma_f h**2 / (6 * m_hat),   m_hat = max_field M_1 / P_total.

``m_hat`` is linear in the load, so one unit-pressure solve per patch radius
gives the whole limit-load curve.

Anchors
-------
* Exact (Kelvin function) centre moment of the circular patch,
  ``M_c = (1+nu)/2 p l**2 X kei'(X)`` with ``X = a/l`` and ``l = (D/k)**(1/4)``
  (:func:`tensorfem.levelset.plate_crack.patch_max_moment`).  This is the
  reference the FE moment field must reproduce.
* The folklore Westergaard/Nevel closed form
  ``M_c = (1+nu)(P/4 pi)(ln(l/a) + 0.6159)``
  (:func:`tensorfem.levelset.plate_crack.patch_max_moment_nevel`).  It is the
  small-``a/l`` limit of the Kelvin solution, so it is *not* usable for
  ``a/l >~ 1`` (see :func:`nevel_moment_applicability`); the acceptance
  comparison therefore reports both references and the validity window.

Discretisation
--------------
Quarter model on ``[0, L]**2`` with ``L = half_width_in_lengths * l``: the
symmetry conditions ``theta_x = 0`` on ``x = 0`` and ``theta_y = 0`` on
``y = 0`` and the outer edges free (the natural truncation of the Winkler
far field).  The mesh is sinh-graded towards the load.  The circular patch is
loaded with consistent nodal forces from sub-cell quadrature of the uniform
pressure over the quarter disc, which keeps the applied load exact to the
sub-cell quadrature error (checked in ``tests/test_ice_plate_flexural_limit.py``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from .foundation_plate_bending import (
    assemble_plate_foundation_system,
    graded_square_mesh,
    solve_assembled_system,
    winkler_characteristic_length,
)
from .levelset.plate_crack import (
    patch_max_moment,
    patch_max_moment_nevel,
    plate_rigidity,
)

__all__ = [
    "IcePlateSolution",
    "solve_circular_patch",
    "principal_moments",
    "surface_tensile_stress",
    "characteristic_moment_per_load",
    "first_crack_load",
    "failure_mode",
    "nevel_moment_applicability",
    "quarter_disc_loads",
    "q4_shape_kinematics",
    "foundation_reaction",
]

_GP = 1.0 / math.sqrt(3.0)
_GAUSS_POINTS = ((-_GP, -_GP), (-_GP, _GP), (_GP, _GP), (_GP, -_GP))


# ---------------------------------------------------------------------------
# Q4 kinematics (natural coordinates in [-1, 1]^2, so d x / d xi = side/2)
# ---------------------------------------------------------------------------

def q4_shape_kinematics(
    xy: np.ndarray, xi: float, eta: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """``(N, dN/dx, det J)`` of a 4-node bilinear element at ``(xi, eta)``.

    ``xy`` is ``(4, 2)`` in counter-clockwise order; row ``i`` of ``dN/dx`` is
    ``(dN_i/dx, dN_i/dy)``.  The Jacobian is ``J[a, c] = d x_c / d xi_a`` —
    with the standard ``[-1, 1]`` natural square ``d x / d xi = side/2``, which
    is the factor that a naive Jacobian built from the *side length* gets wrong
    by two.
    """
    xy = np.asarray(xy, dtype=np.float64)
    if xy.shape != (4, 2):
        raise ValueError("xy must have shape (4, 2)")
    n = 0.25 * np.array(
        [
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ]
    )
    dn_dnat = 0.25 * np.array(
        [
            [-(1.0 - eta), -(1.0 - xi)],
            [(1.0 - eta), -(1.0 + xi)],
            [(1.0 + eta), (1.0 + xi)],
            [-(1.0 + eta), (1.0 - xi)],
        ]
    )
    jac = dn_dnat.T @ xy                       # J[a, c] = d x_c / d xi_a
    det_j = float(jac[0, 0] * jac[1, 1] - jac[0, 1] * jac[1, 0])
    if det_j <= 0.0:
        raise ValueError("element has non-positive Jacobian")
    # chain rule with the *transposed* inverse: dN_i/dx_d = sum_a dN_i/dxi_a J^-1[d, a]
    return n, dn_dnat @ np.linalg.inv(jac).T, det_j


def _batch_kinematics(xy: np.ndarray, xi: float, eta: float):
    """``(N, dN/dx, det J)`` for ``(ne, 4, 2)`` element coordinates."""
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 3 or xy.shape[1:] != (4, 2):
        raise ValueError("xy must have shape (ne, 4, 2)")
    dn_nat = 0.25 * np.array(
        [
            [-(1.0 - eta), -(1.0 - xi)],
            [(1.0 - eta), -(1.0 + xi)],
            [(1.0 + eta), (1.0 + xi)],
            [-(1.0 + eta), (1.0 - xi)],
        ]
    )
    n = 0.25 * np.array(
        [
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ]
    )
    jac = np.einsum("ia,eic->eac", dn_nat, xy)
    det_j = jac[:, 0, 0] * jac[:, 1, 1] - jac[:, 0, 1] * jac[:, 1, 0]
    if np.any(det_j <= 0.0):
        raise ValueError("element has non-positive Jacobian")
    inv = np.stack(
        (
            np.stack((jac[:, 1, 1], -jac[:, 0, 1]), axis=-1),
            np.stack((-jac[:, 1, 0], jac[:, 0, 0]), axis=-1),
        ),
        axis=-2,
    ) / det_j[:, None, None]
    # dN_i/dx_d = sum_a (dN_i/dxi_a) J^-1[d, a] -- the inverse is transposed
    dn_dx = np.einsum("ia,eca->eic", dn_nat, inv)
    return n, dn_dx, det_j


# ---------------------------------------------------------------------------
# Circular patch load
# ---------------------------------------------------------------------------

def quarter_disc_loads(
    nodes: np.ndarray,
    elements: np.ndarray,
    radius: float,
    pressure: float,
    *,
    subcells: int = 8,
) -> np.ndarray:
    """Consistent nodal forces of a uniform pressure over the quarter disc.

    The quarter disc ``{(x, y) >= 0, r <= radius}`` is the image of a full
    circular patch in the quarter model.  Candidate elements are subdivided
    into ``subcells**2`` equal sub-cells whose centroids are kept when
    ``r <= radius``; a kept sub-cell contributes ``pressure * area * N(c)`` to
    its four nodes.  The total is ``pressure * pi * radius**2 / 4`` to within
    the sub-cell quadrature error (O(1/subcells**2), measured < 3e-4).
    """
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    if subcells < 1:
        raise ValueError("subcells must be positive")
    nodes = np.asarray(nodes, dtype=np.float64)
    elements = np.asarray(elements, dtype=np.int64)
    xy = nodes[elements]
    centres = xy.mean(axis=1)
    extent = np.ptp(xy, axis=1)
    keep = np.nonzero(
        np.hypot(centres[:, 0], centres[:, 1])
        <= radius + 0.5 * np.hypot(extent[:, 0], extent[:, 1])
    )[0]
    if keep.size == 0:
        raise ValueError("no element lies inside the load patch")

    t = (np.arange(subcells) + 0.5) / subcells * 2.0 - 1.0
    xi, eta = np.meshgrid(t, t, indexing="ij")
    xi, eta = xi.ravel(), eta.ravel()
    nq = 0.25 * np.stack(
        [
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ],
        axis=1,
    )
    xy_keep = xy[keep]
    points = np.einsum("qk,ekc->eqc", nq, xy_keep)           # (nk, nq, 2)
    inside = np.hypot(points[..., 0], points[..., 1]) <= radius
    _, _, det_j = _batch_kinematics(xy_keep, 0.0, 0.0)
    sub_area = det_j * (2.0 / subcells) ** 2                  # dA = detJ dxi deta
    contrib = pressure * sub_area[:, None, None] * inside[:, :, None] * nq[None, :, :]
    nodal = contrib.sum(axis=1)                               # (nk, 4)
    force = np.zeros(3 * nodes.shape[0])
    np.add.at(force, (3 * elements[keep]).reshape(-1), nodal.reshape(-1))
    return force


# ---------------------------------------------------------------------------
# Solution container
# ---------------------------------------------------------------------------

@dataclass
class IcePlateSolution:
    """FEM solution of a circular patch load on an infinite Winkler plate."""

    nodes: np.ndarray            # (nn, 2)
    elements: np.ndarray         # (ne, 4)
    u: np.ndarray                # (3 nn,)
    total_load: float            # applied vertical force of the quarter model
    pressure: float
    patch_radius: float
    young: float
    poisson: float
    thickness: float
    foundation: float
    rigidity: float
    length_scale: float
    half_width: float
    centres: np.ndarray          # (ne, 2) element centres
    gauss_points: np.ndarray     # (4 ne, 2) physical Gauss-point coordinates
    moments: np.ndarray          # (4 ne, 3) (Mx, My, Mxy) at the Gauss points
    metadata: dict[str, float] = field(default_factory=dict)

    @property
    def full_load(self) -> float:
        """Load of the full circular patch, ``4 x`` the quarter-model total."""
        return 4.0 * self.total_load

    @property
    def surface_stress_per_moment(self) -> float:
        """``sigma = 6 M / h**2`` -- surface bending stress per unit moment."""
        return 6.0 / self.thickness ** 2

    def principal(self) -> tuple[np.ndarray, np.ndarray]:
        """Principal moments ``(M1, M2)`` at every Gauss point, ``M1 >= M2``."""
        return principal_moments(self.moments[:, 0], self.moments[:, 1], self.moments[:, 2])

    def bottom_top_stress(self) -> tuple[np.ndarray, np.ndarray]:
        """Maximum tensile stress at the bottom and the top fibre (Gauss points)."""
        m1, m2 = self.principal()
        return surface_tensile_stress(m1, m2, self.thickness)

    def polar_fields(self) -> dict[str, np.ndarray]:
        """Polar moment components ``M_r``, ``M_theta``, ``M_rtheta`` and ``r``.

        The rotation by the Gauss-point angle is exact for an axisymmetric
        state, so the scatter of ``M_r`` within a ring measures the
        axisymmetry (mesh-quality) error directly.
        """
        x, y = self.gauss_points[:, 0], self.gauss_points[:, 1]
        r = np.hypot(x, y)
        safe = np.where(r > 0.0, r, 1.0)
        c, s = x / safe, y / safe
        m = self.moments
        m_r = m[:, 0] * c ** 2 + m[:, 1] * s ** 2 + 2.0 * m[:, 2] * s * c
        m_t = m[:, 0] * s ** 2 + m[:, 1] * c ** 2 - 2.0 * m[:, 2] * s * c
        m_rt = (m[:, 1] - m[:, 0]) * s * c + m[:, 2] * (c ** 2 - s ** 2)
        return {"r": r, "M_r": m_r, "M_theta": m_t, "M_rtheta": m_rt}


# ---------------------------------------------------------------------------
# Failure criterion (independent of the analytic first-crack route)
# ---------------------------------------------------------------------------

def principal_moments(mx, my, mxy) -> tuple[np.ndarray, np.ndarray]:
    """Principal plate moments ``M_1 >= M_2`` of ``(Mx, My, Mxy)``."""
    mx = np.asarray(mx, dtype=np.float64)
    my = np.asarray(my, dtype=np.float64)
    mxy = np.asarray(mxy, dtype=np.float64)
    mean = 0.5 * (mx + my)
    radius = np.sqrt((0.5 * (mx - my)) ** 2 + mxy ** 2)
    return mean + radius, mean - radius


def surface_tensile_stress(m1, m2, thickness: float) -> tuple[np.ndarray, np.ndarray]:
    """Largest tensile stress at the bottom and top fibre for principal moments.

    Bottom fibre: ``sigma = +6 M / h**2`` (sagging puts the bottom in tension);
    top fibre: ``sigma = -6 M / h**2``.  Only tensile values are returned
    (``max(., 0)``), because a compressive fibre cannot initiate a crack.
    """
    if thickness <= 0.0:
        raise ValueError("thickness must be positive")
    scale = 6.0 / thickness ** 2
    m1 = np.asarray(m1, dtype=np.float64)
    m2 = np.asarray(m2, dtype=np.float64)
    return np.maximum(scale * m1, 0.0), np.maximum(-scale * m2, 0.0)


def _corner_moment(solution: IcePlateSolution) -> float:
    """Sagging moment at the symmetry corner (load centre), superconvergent.

    The 2x2 Gauss points of a Q4 element are its superconvergent points; the
    bilinear extrapolation of the Gauss-point field to a *corner node* is the
    value at the nearest Gauss point, which ``argmin |x_gp|`` selects.  The
    state at the centre is axisymmetric (``Mx = My``), so ``0.5 (Mx + My)`` is
    the centre moment.
    """
    idx = int(np.argmin(np.hypot(solution.gauss_points[:, 0], solution.gauss_points[:, 1])))
    m = solution.moments[idx]
    return float(0.5 * (m[0] + m[1]))


def foundation_reaction(solution: IcePlateSolution) -> float:
    """``k * integral(w dA)`` over the quarter model -- an equilibrium audit.

    For a free outer edge the foundation reaction must equal the applied
    vertical load; a mismatch exposes an assembly, load-vector or boundary
    condition error rather than a modelling choice.
    """
    xy = solution.nodes[solution.elements]
    w = solution.u[3 * solution.elements]
    integral = 0.0
    for xi, eta in _GAUSS_POINTS:
        n, _, det_j = _batch_kinematics(xy, xi, eta)
        integral += float(np.sum(np.einsum("k,ek->e", n, w) * det_j))
    return solution.foundation * integral


def characteristic_moment_per_load(solution: IcePlateSolution) -> dict[str, float]:
    """``m_hat`` (critical principal moment per unit total load) and its location.

    The maximum is taken over *both* fibres, so ``m_hat`` is exactly what the
    Rankine first-crack criterion needs; ``fibre`` says which surface reaches
    it and ``radius_over_patch``/``radius_over_length`` where.
    """
    bottom, top = solution.bottom_top_stress()
    scale = solution.surface_stress_per_moment
    m_bottom = bottom / scale
    m_top = top / scale
    i_bottom = int(np.argmax(m_bottom))
    i_top = int(np.argmax(m_top))
    use_bottom = m_bottom[i_bottom] >= m_top[i_top]
    idx = i_bottom if use_bottom else i_top
    m_crit = float((m_bottom if use_bottom else m_top)[idx])
    r = float(np.hypot(*solution.gauss_points[idx]))
    centre = _corner_moment(solution)
    return {
        "m_hat": m_crit / solution.full_load,
        "moment": m_crit,
        "radius": r,
        "radius_over_patch": r / solution.patch_radius,
        "radius_over_length": r / solution.length_scale,
        "bottom_max_moment": float(m_bottom[i_bottom]),
        "top_max_moment": float(m_top[i_top]),
        "top_over_bottom": float(m_top[i_top] / m_bottom[i_bottom]),
        "bottom_at_centre": bool(i_bottom == int(np.argmin(np.hypot(
            solution.gauss_points[:, 0], solution.gauss_points[:, 1])))),
        "centre_moment": centre,
        "centre_moment_per_load": centre / solution.full_load,
    }


# ---------------------------------------------------------------------------
# FE driver
# ---------------------------------------------------------------------------

def solve_circular_patch(
    *,
    patch_radius: float,
    young: float = 7.0e9,
    poisson: float = 0.30,
    thickness: float = 0.30,
    foundation: float = 1025.0 * 9.81,
    pressure: float = 1.0,
    cells: int = 80,
    beta: float = 5.0,
    half_width_in_lengths: float = 20.0,
    subcells: int = 8,
    shear_treatment: str = "mitc",
) -> IcePlateSolution:
    """Solve one circular patch load on the infinite Winkler plate (quarter model).

    ``patch_radius`` is in metres.  ``pressure`` only scales the solution, so
    ``pressure = 1`` gives the influence field used by :func:`first_crack_load`.
    """
    rigidity = plate_rigidity(young, poisson, thickness)
    ell = winkler_characteristic_length(rigidity, foundation)
    half_width = half_width_in_lengths * ell
    nodes_t, elements_t = graded_square_mesh(cells, half_width, beta)
    nodes, elements = nodes_t.numpy(), elements_t.numpy()

    rows, cols, vals, _, ndof = assemble_plate_foundation_system(
        nodes_t,
        elements_t,
        young=young,
        poisson=poisson,
        thickness=thickness,
        winkler=foundation,
        shear_treatment=shear_treatment,
    )
    force = quarter_disc_loads(nodes, elements, patch_radius, pressure, subcells=subcells)
    system = (rows, cols, vals, torch.as_tensor(force), ndof)

    tol = 1e-9 * half_width
    fixed = np.unique(
        np.concatenate(
            (
                3 * np.nonzero(np.abs(nodes[:, 0]) <= tol)[0] + 1,
                3 * np.nonzero(np.abs(nodes[:, 1]) <= tol)[0] + 2,
            )
        )
    )
    u = solve_assembled_system(system, torch.as_tensor(fixed), dense_limit=4000).numpy()
    if not np.all(np.isfinite(u)):
        raise RuntimeError("plate solve produced a non-finite solution")

    return IcePlateSolution(
        nodes=nodes,
        elements=elements,
        u=u,
        total_load=float(force[0::3].sum()),
        pressure=pressure,
        patch_radius=patch_radius,
        young=young,
        poisson=poisson,
        thickness=thickness,
        foundation=foundation,
        rigidity=rigidity,
        length_scale=ell,
        half_width=half_width,
        centres=nodes[elements].mean(axis=1),
        gauss_points=_gauss_point_coordinates(nodes, elements),
        moments=_gauss_point_moments(nodes, elements, u, young, thickness, poisson),
        metadata={
            "cells": float(cells),
            "beta": float(beta),
            "half_width_over_length": float(half_width_in_lengths),
            "element_size_at_centre": float(nodes[elements[0]][1, 0] - nodes[elements[0]][0, 0]),
        },
    )


def _gauss_point_coordinates(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    xy = nodes[elements]
    pts = []
    for xi, eta in _GAUSS_POINTS:
        n = 0.25 * np.array(
            [
                (1.0 - xi) * (1.0 - eta),
                (1.0 + xi) * (1.0 - eta),
                (1.0 + xi) * (1.0 + eta),
                (1.0 - xi) * (1.0 + eta),
            ]
        )
        pts.append(np.einsum("k,ekc->ec", n, xy))
    return np.concatenate(pts, axis=0)


def _gauss_point_moments(
    nodes: np.ndarray,
    elements: np.ndarray,
    u: np.ndarray,
    young: float,
    thickness: float,
    poisson: float,
) -> np.ndarray:
    """Bending/twisting moments at the four Gauss points of every element.

    Curvatures from the Mindlin rotations, ``kappa = sym(grad theta)`` with
    ``u = [w, theta_x, theta_y]`` per node, and ``M = D_b (kappa + nu ...)``
    with ``D_b = E h**3 / (12 (1 - nu**2))`` -- the same convention as
    :meth:`tensorfem.levelset.plate_crack.PlateCrackResult.moment_field`,
    validated against the torch element kernel in the tests.
    """
    db = young * thickness ** 3 / (12.0 * (1.0 - poisson ** 2))
    xy = nodes[elements]
    theta_x = u[3 * elements + 1]
    theta_y = u[3 * elements + 2]
    out = []
    for xi, eta in _GAUSS_POINTS:
        _, dn, _ = _batch_kinematics(xy, xi, eta)
        kx = np.einsum("ek,ek->e", dn[:, :, 0], theta_x)
        ky = np.einsum("ek,ek->e", dn[:, :, 1], theta_y)
        kxy = np.einsum("ek,ek->e", dn[:, :, 1], theta_x) + np.einsum(
            "ek,ek->e", dn[:, :, 0], theta_y
        )
        out.append(
            np.stack(
                [
                    db * (kx + poisson * ky),
                    db * (ky + poisson * kx),
                    db * (1.0 - poisson) * 0.5 * kxy,
                ],
                axis=1,
            )
        )
    return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
# Limit load and failure mode
# ---------------------------------------------------------------------------

def first_crack_load(
    *,
    patch_radius: float,
    flexural_strength: float,
    young: float = 7.0e9,
    poisson: float = 0.30,
    thickness: float = 0.30,
    foundation: float = 1025.0 * 9.81,
    cells: int = 80,
    beta: float = 5.0,
    half_width_in_lengths: float = 20.0,
    subcells: int = 8,
) -> dict[str, object]:
    """First-crack (flexural limit) load from the FE moment field + anchors.

    Returns the FE result, the Kelvin and Westergaard/Nevel references for the
    same input and the relative errors -- the acceptance comparison of the
    H3-ICE-PLATE-FEM anchor.
    """
    if flexural_strength <= 0.0:
        raise ValueError("flexural_strength must be positive")
    sol = solve_circular_patch(
        patch_radius=patch_radius,
        young=young,
        poisson=poisson,
        thickness=thickness,
        foundation=foundation,
        pressure=1.0,
        cells=cells,
        beta=beta,
        half_width_in_lengths=half_width_in_lengths,
        subcells=subcells,
    )
    specs = characteristic_moment_per_load(sol)
    m_crit = flexural_strength * thickness ** 2 / 6.0
    p_fem = m_crit / specs["m_hat"]

    anchor = patch_max_moment(1.0, patch_radius, sol.rigidity, foundation, poisson)
    nevel = patch_max_moment_nevel(
        math.pi * patch_radius ** 2, patch_radius, sol.rigidity, foundation, poisson
    )
    m_hat_kelvin = anchor / (math.pi * patch_radius ** 2)
    m_hat_nevel = nevel / (math.pi * patch_radius ** 2)
    p_kelvin = m_crit / m_hat_kelvin
    p_nevel = m_crit / m_hat_nevel
    return {
        "patch_radius": patch_radius,
        "radius_over_length": patch_radius / sol.length_scale,
        "length_scale": sol.length_scale,
        "rigidity": sol.rigidity,
        "critical_moment": m_crit,
        "m_hat_fem": specs["m_hat"],
        "m_hat_kelvin": m_hat_kelvin,
        "m_hat_nevel": m_hat_nevel,
        "first_crack_load_fem": p_fem,
        "first_crack_load_kelvin": p_kelvin,
        "first_crack_load_nevel": p_nevel,
        "error_vs_kelvin": p_fem / p_kelvin - 1.0,
        "error_vs_nevel": p_fem / p_nevel - 1.0,
        "centre_moment_over_kelvin": specs["centre_moment"] / anchor - 1.0,
        "applied_quarter_load": sol.total_load,
        "expected_quarter_load": math.pi * patch_radius ** 2 / 4.0,
        "load_error": sol.total_load / (math.pi * patch_radius ** 2 / 4.0) - 1.0,
        "reaction_over_load": foundation_reaction(sol) / sol.total_load,
        "critical_fibre": "bottom" if specs["moment"] == specs["bottom_max_moment"] else "top",
        "critical_radius": specs["radius"],
        "critical_radius_over_patch": specs["radius_over_patch"],
        "top_over_bottom": specs["top_over_bottom"],
    }


def failure_mode(
    *,
    patch_radius: float,
    young: float = 7.0e9,
    poisson: float = 0.30,
    thickness: float = 0.30,
    foundation: float = 1025.0 * 9.81,
    cells: int = 80,
    beta: float = 5.0,
    half_width_in_lengths: float = 20.0,
    subcells: int = 8,
) -> dict[str, object]:
    """Failure location, active fibre and crack orientation for one patch size.

    The crack plane is normal to the largest tensile stress.  With the critical
    fibre known, the polar components at the critical point decide the outcome:
    hoop tension larger than radial -> the opening stress is tangential, so the
    crack runs *radially* (``radial_bottom`` / ``radial_top``); radial tension
    larger -> a *circumferential* crack.  When the two are within 5% of each
    other the state does not select an orientation and the result is reported as
    ``transitional_<fibre>`` (the load-centre case is the exact equibiaxial
    degeneracy, flagged by ``critical_at_centre``).
    """
    sol = solve_circular_patch(
        patch_radius=patch_radius,
        young=young,
        poisson=poisson,
        thickness=thickness,
        foundation=foundation,
        pressure=1.0,
        cells=cells,
        beta=beta,
        half_width_in_lengths=half_width_in_lengths,
        subcells=subcells,
    )
    specs = characteristic_moment_per_load(sol)
    polar = sol.polar_fields()
    bottom, top = sol.bottom_top_stress()
    use_bottom = specs["moment"] == specs["bottom_max_moment"]
    field = bottom if use_bottom else top
    idx = int(np.argmax(field))
    sign = 1.0 if use_bottom else -1.0
    hoop = sign * float(polar["M_theta"][idx])
    radial = sign * float(polar["M_r"][idx])
    # the quarter mesh resolves the centre to r ~ 0.005 l; anything inside 2%
    # of the characteristic length is the load centre to within mesh resolution
    at_centre = specs["radius"] <= 0.02 * sol.length_scale
    fibre = "bottom" if use_bottom else "top"
    ratio = hoop / radial if radial != 0.0 else float("inf")
    equibiaxial = at_centre or abs(ratio - 1.0) <= 0.05
    if at_centre:
        # exact equibiaxial state at the load centre: every radial plane is
        # critical, so the axisymmetric solution of the crack problem is a star
        # of radial cracks (the classical first-crack pattern)
        mode = f"radial_{fibre}"
    elif equibiaxial:
        # both in-plane directions within 5%: the moment state does not select
        # an orientation, so this point cannot be assigned a single crack mode
        mode = f"transitional_{fibre}"
    elif hoop > radial:
        mode = f"radial_{fibre}"
    else:
        mode = f"circumferential_{fibre}"
    return {
        "radius_over_length": patch_radius / sol.length_scale,
        "critical_fibre": "bottom" if use_bottom else "top",
        "critical_stress": float(field[idx]),
        "critical_radius": specs["radius"],
        "critical_radius_over_patch": specs["radius_over_patch"],
        "critical_at_centre": bool(at_centre),
        "equibiaxial": bool(equibiaxial),
        "M_r": radial,
        "M_theta": hoop,
        "hoop_over_radial": ratio,
        "top_over_bottom": specs["top_over_bottom"],
        "mode": mode,
        "m_hat": specs["m_hat"],
        "length_scale": sol.length_scale,
        "solution": sol,
    }


def nevel_moment_applicability(
    *,
    young: float = 7.0e9,
    poisson: float = 0.30,
    thickness: float = 0.30,
    foundation: float = 1025.0 * 9.81,
    radii_over_length: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5, 1.0, 1.5, 2.0),
) -> list[dict[str, float]]:
    """Relative error of the Westergaard/Nevel closed form against the exact value.

    The log form ``(1+nu)(P/4 pi)(ln(l/a) + 0.6159)`` is the small-``a/l``
    expansion of the Kelvin solution; this table states where it may be quoted
    and where it must not be (it changes sign at ``a/l ~ 1.6``).
    """
    rigidity = plate_rigidity(young, poisson, thickness)
    ell = winkler_characteristic_length(rigidity, foundation)
    rows = []
    for frac in radii_over_length:
        a = frac * ell
        exact = patch_max_moment(1.0, a, rigidity, foundation, poisson)
        nevel = patch_max_moment_nevel(math.pi * a ** 2, a, rigidity, foundation, poisson)
        rows.append(
            {
                "radius_over_length": frac,
                "kelvin": exact,
                "nevel": nevel,
                "relative_error": nevel / exact - 1.0,
            }
        )
    return rows