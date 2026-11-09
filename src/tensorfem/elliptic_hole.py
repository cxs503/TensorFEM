"""Inglis elliptical-hole stress-concentration reference and quarter-ring Q4 case.

Why this case exists
--------------------
``docs/kirsch-hole-stress-concentration.md`` (circular hole, ``K_t = 3``) and
``docs/finite-width-hole-howland.md`` (finite-width band) closed two of the hole
debts listed in ``docs/benchmark-qualification-status.md``.  Both still assume a
*circular* opening, so an oval / elongated ship opening (hatch corners, scallops,
sea-chest lips, manholes) has to fall back on ``K_t = 3``, which is wrong in both
directions: an opening elongated *across* the load amplifies the peak (``K_t``
grows without bound in the crack limit), one elongated *along* the load
attenuates it (``K_t < 3``).  This module supplies the exact reference for the
elliptical opening (Inglis 1913) so the same production plane-stress path can be
verified off-circular.

Exact solution
--------------
Map the exterior of the unit disk onto the exterior of the ellipse with the
Joukowski-type map (``R > 0``, ``|m| < 1``)::

    z = omega(zeta) = R (zeta + m / zeta)
    omega'(zeta)   = R (1 - m / zeta^2)      (zero only at |zeta| = sqrt|m| < 1)

so ``|zeta| = 1`` is the ellipse with semi-axes ``R (1 + m)`` along ``x`` and
``R (1 - m)`` along ``y``::

    semi_x = R (1 + m),  semi_y = R (1 - m),  R = (semi_x + semi_y) / 2
    m      = (semi_x - semi_y) / (semi_x + semi_y),   -1 < m < 1

``m > 0`` is an opening elongated *along* the load (blunt transverse tips),
``m < 0`` an opening elongated *across* the load (sharp tips on the load axis),
``m = 0`` degenerates to Kirsch.  One formula pair covers both orientations.

With remote uniaxial tension ``T`` along ``x`` the Kolosov-Muskhelishvili
potentials are, in the ``zeta`` plane::

    phi(zeta) = (T R / 4) [ zeta + (2 - m) / zeta ]
    psi(zeta) = (T R / 4) [ 2/zeta - 2 (1 - m)^2 zeta - 2 zeta^3 ] / (zeta^2 - m)

and the stresses follow from the standard formulas (Muskhelishvili 1953 §32;
``Phi = phi' / omega'``)::

    sigma_xx + sigma_yy                = 4 Re Phi
    sigma_yy - sigma_xx + 2 i sigma_xy = 2 [ conj(omega) dPhi/dzeta / omega' + Psi ]

**How the coefficients were obtained (derived, not fitted).**  The traction-free
hole condition ``phi + omega conj(phi')/conj(omega') + conj(psi) = 0`` on
``|zeta| = 1`` is multiplied by ``(1 - m sigma^2)``; the resulting Laurent
polynomial has to vanish identically.  Requiring ``psi(1/sigma)`` to carry
exactly the far-field ``1/sigma`` singularity (and no ``sigma^{-2}`` or deeper)
fixes ``a_1 = 2 - m``, after which ``psi`` follows in closed form.  Two
identities then pin the result without any fitting, and both are asserted in the
test suite:

* ``sigma_yy`` at the load-axis tips ``(+/-semi_x, 0)`` is exactly ``-T``
  (``m``-independent — the same value Kirsch gives for the circle),
* ``sigma_xx`` at the transverse tips ``(0, +/-semi_y)`` is exactly
  ``T (1 + 2 semi_y / semi_x)``, the classical Inglis factor,
* ``m -> 0`` reproduces :class:`tensorfem.kirsch_hole.KirschHole` (stress and
  displacement) to ``1e-9``, so the ``K_t = 3`` debt stays closed inside it.

Convention warning (the Howland report burned a run on this): in
``1 + 2 semi_y / semi_x`` the numerator is the semi-axis **perpendicular** to the
load and the denominator the one **along** it.  Reading it the other way round
inverts the factor (``7`` vs ``1.667`` at ``semi_x / semi_y = 3``).

Verification use
----------------
:func:`solve_case` drives the *production* plane-stress path
(:class:`tensorfem.continuum.ContinuumModel` + :func:`solve_continuum`) on a
quarter ring between the hole and a **confocal** outer ellipse ``|zeta| = rho0``
that carries the exact tractions.  The outer data and the symmetry conditions are
the exact field's own, so the exact field is the unique solution of the discrete
boundary-value problem: this is a verification case, not FE-vs-FE, and nothing is
truncated (no "``b = 10 a``" far-field approximation) — the only error is the FE
discretisation itself.

``scripts/audit_elliptic_hole.py`` writes the JSON/figures; the write-up is
``docs/elliptic-hole-inglis-validation.md``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .continuum import ContinuumModel, solve_continuum

__all__ = [
    "InglisHole",
    "identical_hole",
    "quarter_elliptic_mesh",
    "consistent_arc_loads",
    "symmetry_dofs",
    "annulus_model",
    "solve_case",
    "convergence_study",
    "q4_stress_at_point",
    "element_centre_cross_check",
    "tip_nodal_stress",
    "boundary_hoop_profile",
]

# 4-point Gauss-Legendre on [-1, 1] for the consistent boundary loads.
_BOUNDARY_GAUSS = (
    (-0.8611363115940526, 0.3478548451374538),
    (-0.3399810435848563, 0.6521451548625461),
    (0.3399810435848563, 0.6521451548625461),
    (0.8611363115940526, 0.3478548451374538),
)

# 2x2 Gauss points of the independent element stress recovery.
_GAUSS = 1.0 / math.sqrt(3.0)
_ELEMENT_GAUSS = (
    (-_GAUSS, -_GAUSS),
    (_GAUSS, -_GAUSS),
    (_GAUSS, _GAUSS),
    (-_GAUSS, _GAUSS),
)
_GAUSS_BASIS = np.array([[1.0, xi, eta, xi * eta] for xi, eta in _ELEMENT_GAUSS])


# ---------------------------------------------------------------------------
# Exact Inglis solution
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InglisHole:
    """Elliptical opening under remote uniaxial tension along ``x``.

    ``semi_x`` is the semi-axis *along* the load, ``semi_y`` the semi-axis
    *transverse* to it; ``semi_x == semi_y`` is Kirsch.  ``plane='strain'``
    changes only the displacement field — the stress field of a traction-driven
    plane problem is material independent — exactly as in
    :class:`tensorfem.kirsch_hole.KirschHole`.
    """

    semi_x: float = 1.0
    semi_y: float = 1.0
    far_field: float = 1.0
    young: float = 1.0
    poisson: float = 0.3
    plane: str = "stress"
    thickness: float = 1.0

    def __post_init__(self) -> None:
        if not (self.semi_x > 0.0 and self.semi_y > 0.0):
            raise ValueError("both semi-axes must be positive")
        if self.plane not in ("stress", "strain"):
            raise ValueError("plane must be 'stress' or 'strain'")
        if not -0.9999 < self.poisson < 0.5:
            raise ValueError("poisson ratio out of range for a stable solid")
        if self.young <= 0.0 or self.thickness <= 0.0:
            raise ValueError("young modulus and thickness must be positive")
        if abs(self.mapping_m) >= 1.0:
            raise ValueError("semi-axis ratio out of range for the map |m| < 1")

    # -- geometry ----------------------------------------------------------
    @property
    def mapping_radius(self) -> float:
        """``R`` of ``z = R (zeta + m / zeta)``."""
        return 0.5 * (self.semi_x + self.semi_y)

    @property
    def mapping_m(self) -> float:
        """``m`` of the map; ``m > 0`` = elongated along the load."""
        return (self.semi_x - self.semi_y) / (self.semi_x + self.semi_y)

    @property
    def tip_ratio(self) -> float:
        """``semi_y / semi_x`` — the shape argument of the Inglis factor."""
        return self.semi_y / self.semi_x

    @property
    def stress_concentration(self) -> float:
        """Classical Inglis factor ``K_t = 1 + 2 semi_y / semi_x``."""
        return 1.0 + 2.0 * self.tip_ratio

    @property
    def transverse_tip(self) -> tuple[float, float]:
        """Point of the peak hoop stress, ``(0, semi_y)``."""
        return (0.0, self.semi_y)

    @property
    def load_axis_tip(self) -> tuple[float, float]:
        """Point where ``sigma_yy = -T``, ``(semi_x, 0)``."""
        return (self.semi_x, 0.0)

    @property
    def tip_curvature_radius(self) -> float:
        """Radius of curvature at the transverse tip, ``semi_x^2 / semi_y``."""
        return self.semi_x * self.semi_x / self.semi_y

    @property
    def effective_constants(self) -> tuple[float, float]:
        """``(E, nu)`` of the equivalent plane-stress problem."""
        if self.plane == "stress":
            return self.young, self.poisson
        nu = self.poisson
        return self.young / (1.0 - nu * nu), nu / (1.0 - nu)

    # -- map ---------------------------------------------------------------
    def physical(self, zeta):
        """``z = omega(zeta)``."""
        zeta = np.asarray(zeta, dtype=np.complex128)
        return self.mapping_radius * (zeta + self.mapping_m / zeta)

    def preimage(self, x, y):
        """``zeta = omega^{-1}(z)`` on the branch ``|zeta| >= 1``."""
        z = np.asarray(x, dtype=np.float64) + 1j * np.asarray(y, dtype=np.float64)
        r, m = self.mapping_radius, self.mapping_m
        root = np.sqrt(z * z - 4.0 * m * r * r + 0j)
        # the two roots multiply to m, so |zeta_-| = |m| / |zeta_+| < |zeta_+|:
        # taking the larger modulus picks the physically relevant branch
        # (``|zeta| >= 1``) without an epsilon test that misfires on the boundary
        plus = (z + root) / (2.0 * r)
        minus = (z - root) / (2.0 * r)
        return np.where(np.abs(plus) >= np.abs(minus), plus, minus)

    # -- potentials --------------------------------------------------------
    def _potentials(self, zeta):
        """``phi``, ``phi'``, ``phi''``, ``psi``, ``psi'`` in the ``zeta`` plane."""
        zeta = np.asarray(zeta, dtype=np.complex128)
        s4 = 0.25 * self.far_field * self.mapping_radius
        m = self.mapping_m
        c = 2.0 - m
        phi = s4 * (zeta + c / zeta)
        phi_p = s4 * (1.0 - c / zeta**2)
        phi_pp = s4 * (2.0 * c / zeta**3)
        d = zeta**2 - m
        n = 2.0 / zeta - 2.0 * (1.0 - m) ** 2 * zeta - 2.0 * zeta**3
        dn = -2.0 / zeta**2 - 2.0 * (1.0 - m) ** 2 - 6.0 * zeta**2
        psi = s4 * n / d
        psi_p = s4 * (dn * d - n * 2.0 * zeta) / (d * d)
        return phi, phi_p, phi_pp, psi, psi_p

    def _map_derivatives(self, zeta):
        zeta = np.asarray(zeta, dtype=np.complex128)
        r, m = self.mapping_radius, self.mapping_m
        return (
            r * (zeta + m / zeta),
            r * (1.0 - m / zeta**2),
            r * (2.0 * m / zeta**3),
        )

    # -- stress ------------------------------------------------------------
    def stress_at_zeta(self, zeta):
        """Exact ``(sigma_xx, sigma_yy, sigma_xy)`` at ``zeta``."""
        w, w_p, w_pp = self._map_derivatives(zeta)
        _, phi_p, phi_pp, _, psi_p = self._potentials(zeta)
        capital_phi = phi_p / w_p
        capital_psi = psi_p / w_p
        d_capital_phi = (phi_pp * w_p - phi_p * w_pp) / w_p**3
        trace = 4.0 * np.real(capital_phi)
        difference = 2.0 * (np.conj(w) * d_capital_phi + capital_psi)
        return (
            (trace - np.real(difference)) / 2.0,
            (trace + np.real(difference)) / 2.0,
            np.imag(difference) / 2.0,
        )

    def stress_cartesian(self, x, y):
        """Exact ``(sigma_xx, sigma_yy, sigma_xy)`` at Cartesian points."""
        return self.stress_at_zeta(self.preimage(x, y))

    def von_mises_cartesian(self, x, y):
        sxx, syy, sxy = self.stress_cartesian(x, y)
        return np.sqrt(sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy)

    def boundary_hoop_stress(self, theta):
        """Exact hoop stress at the hole boundary point ``omega(exp(i theta))``.

        ``sigma_tt = t . sigma . t`` with ``t`` the unit tangent of the ellipse.
        """
        sigma = np.exp(1j * np.asarray(theta, dtype=np.float64))
        sxx, syy, sxy = self.stress_at_zeta(sigma)
        dz_dtheta = 1j * (self.mapping_radius * (sigma - self.mapping_m / sigma))
        norm = np.abs(dz_dtheta)
        tx, ty = np.real(dz_dtheta) / norm, np.imag(dz_dtheta) / norm
        return sxx * tx * tx + syy * ty * ty + 2.0 * sxy * tx * ty

    def traction_at_zeta(self, zeta):
        """Exact traction ``sigma . n`` on the confocal ellipse through ``zeta``."""
        zeta = np.asarray(zeta, dtype=np.complex128)
        sxx, syy, sxy = self.stress_at_zeta(zeta)
        dz_dtheta = 1j * (self.mapping_radius * (zeta - self.mapping_m / zeta))
        norm = np.abs(dz_dtheta)
        tx, ty = np.real(dz_dtheta) / norm, np.imag(dz_dtheta) / norm
        nx, ny = ty, -tx
        return sxx * nx + sxy * ny, sxy * nx + syy * ny

    # -- displacement ------------------------------------------------------
    def displacement_cartesian(self, x, y):
        """Exact ``(u_x, u_y)`` at Cartesian points."""
        zeta = self.preimage(x, y)
        w, w_p, _ = self._map_derivatives(zeta)
        phi, phi_p, _, psi, _ = self._potentials(zeta)
        e, nu = self.effective_constants
        kappa = (3.0 - nu) / (1.0 + nu) if self.plane == "stress" else 3.0 - 4.0 * nu
        mu = e / (2.0 * (1.0 + nu))
        value = kappa * phi - w * np.conj(phi_p / w_p) - np.conj(psi)
        return np.real(value) / (2.0 * mu), np.imag(value) / (2.0 * mu)

    def strain_cartesian(self, x, y, step=None):
        """Numerically differentiated exact strain ``(eps_xx, eps_yy, gam_xy)``.

        Lets the constitutive law be checked independently of the potentials.
        """
        h = 1e-6 * max(self.mapping_radius, 1e-12) if step is None else step
        ux_xp, uy_xp = self.displacement_cartesian(np.asarray(x) + h, y)
        ux_xm, uy_xm = self.displacement_cartesian(np.asarray(x) - h, y)
        ux_yp, uy_yp = self.displacement_cartesian(x, np.asarray(y) + h)
        ux_ym, uy_ym = self.displacement_cartesian(x, np.asarray(y) - h)
        return (
            (ux_xp - ux_xm) / (2.0 * h),
            (uy_yp - uy_ym) / (2.0 * h),
            (ux_yp - ux_ym) / (2.0 * h) + (uy_xp - uy_xm) / (2.0 * h),
        )


def identical_hole(radius: float = 1.0, **kwargs) -> InglisHole:
    """Circular limit ``semi_x = semi_y`` (``m = 0``), i.e. the Kirsch case."""
    return InglisHole(semi_x=radius, semi_y=radius, **kwargs)


# ---------------------------------------------------------------------------
# Quarter-ring mesh and consistent boundary loads
# ---------------------------------------------------------------------------

def _chebyshev_theta(n_theta: int) -> np.ndarray:
    """``theta`` in ``[0, pi/2]`` clustered at both tips (``theta = 0, pi/2``)."""
    s = np.linspace(0.0, 1.0, n_theta + 1)
    return 0.25 * math.pi * (1.0 - np.cos(math.pi * s))


def _graded_rho(n_radial: int, rho_outer: float, grading: float) -> np.ndarray:
    """Radial coordinate ``rho`` from 1 to ``rho_outer``, refined at the hole."""
    if grading < 0.0:
        raise ValueError("grading must be non-negative")
    s = np.linspace(0.0, 1.0, n_radial + 1)
    if grading < 1e-12:
        return 1.0 + (rho_outer - 1.0) * s
    return 1.0 + (rho_outer - 1.0) * (np.exp(grading * s) - 1.0) / (math.exp(grading) - 1.0)


def quarter_elliptic_mesh(
    hole: InglisHole,
    rho_outer: float = 5.0,
    n_theta: int = 24,
    n_radial: int | None = None,
    grading: float = 3.0,
    *,
    dtype=torch.float64,
):
    """Structured Q4 quarter ring between ``|zeta| = 1`` and ``|zeta| = rho_outer``.

    Node ``(i, j)`` sits at ``zeta = rho_j exp(i theta_i)``, ``i = 0..n_theta``
    (``theta = 0`` on the load axis, ``pi/2`` on the transverse axis) and
    ``j = 0..n_radial`` (``j = 0`` the hole).  Element ``(i, j)`` is ordered
    ``(i, j), (i, j+1), (i+1, j+1), (i+1, j)``, which keeps the parent Jacobian
    positive because the conformal map preserves orientation.
    """
    if rho_outer <= 1.0:
        raise ValueError("rho_outer must exceed 1")
    n_radial = n_theta if n_radial is None else int(n_radial)
    if n_theta < 2 or n_radial < 2:
        raise ValueError("need at least two elements per direction")
    thetas = _chebyshev_theta(n_theta)
    rhos = _graded_rho(n_radial, rho_outer, grading)
    theta_grid, rho_grid = np.meshgrid(thetas, rhos, indexing="ij")
    zeta = rho_grid * np.exp(1j * theta_grid)
    z = hole.physical(zeta)
    nodes = np.stack([np.real(z).ravel(), np.imag(z).ravel()], axis=1)
    stride = n_radial + 1
    elements = []
    for i in range(n_theta):
        for j in range(n_radial):
            elements.append(
                [
                    i * stride + j,
                    i * stride + (j + 1),
                    (i + 1) * stride + (j + 1),
                    (i + 1) * stride + j,
                ]
            )
    return (
        torch.as_tensor(nodes, dtype=dtype),
        torch.as_tensor(np.asarray(elements, dtype=np.int64), dtype=torch.long),
        thetas,
        rhos,
    )


def consistent_arc_loads(
    hole: InglisHole,
    nodes: torch.Tensor,
    rho_outer: float,
    n_theta: int,
    n_radial: int,
    *,
    dtype=torch.float64,
) -> torch.Tensor:
    """Consistent nodal loads of the exact traction on ``|zeta| = rho_outer``.

    ``integral N_a t ds`` per boundary edge with 4-point Gauss in the parent
    coordinate and the exact edge metric ``|dz/dtheta|`` from the map.
    """
    stride = n_radial + 1
    forces = np.zeros(2 * int(nodes.shape[0]), dtype=np.float64)
    thetas = _chebyshev_theta(n_theta)
    rho = float(rho_outer)
    for i in range(n_theta):
        theta_a, theta_b = thetas[i], thetas[i + 1]
        half = 0.5 * (theta_b - theta_a)
        mid = 0.5 * (theta_b + theta_a)
        for xi, weight in _BOUNDARY_GAUSS:
            theta = mid + half * xi
            zeta = rho * complex(math.cos(theta), math.sin(theta))
            tx, ty = hole.traction_at_zeta(zeta)
            dz_dtheta = 1j * (hole.mapping_radius * (zeta - hole.mapping_m / zeta))
            factor = weight * half * abs(dz_dtheta)
            shape = (0.5 * (1.0 - xi), 0.5 * (1.0 + xi))
            for local, node in enumerate((i * stride + n_radial, (i + 1) * stride + n_radial)):
                forces[2 * node] += shape[local] * factor * tx
                forces[2 * node + 1] += shape[local] * factor * ty
    return torch.as_tensor(forces, dtype=dtype)


def symmetry_dofs(n_theta: int, n_radial: int) -> list[int]:
    """Zero-displacement dofs: ``u_y = 0`` on ``theta = 0``, ``u_x = 0`` on ``pi/2``."""
    stride = n_radial + 1
    fixed: set[int] = set()
    for j in range(n_radial + 1):
        fixed.add(2 * j + 1)  # u_y at theta = 0
        fixed.add(2 * (n_theta * stride + j))  # u_x at theta = pi/2
    return sorted(fixed)


def annulus_model(
    hole: InglisHole,
    rho_outer: float = 5.0,
    n_theta: int = 24,
    n_radial: int | None = None,
    grading: float = 3.0,
    *,
    dtype=torch.float64,
) -> tuple[ContinuumModel, dict]:
    """Build the production :class:`ContinuumModel` of the quarter-ring case."""
    n_radial = n_theta if n_radial is None else int(n_radial)
    nodes, elements, thetas, rhos = quarter_elliptic_mesh(
        hole, rho_outer, n_theta, n_radial, grading, dtype=dtype
    )
    forces = consistent_arc_loads(hole, nodes, rho_outer, n_theta, n_radial, dtype=dtype)
    model = ContinuumModel(
        nodes=nodes,
        elements=elements,
        young_modulus=torch.as_tensor(hole.young, dtype=dtype),
        poisson_ratio=torch.as_tensor(hole.poisson, dtype=dtype),
        thickness=torch.as_tensor(hole.thickness, dtype=dtype),
        forces=forces,
        fixed_dofs=torch.as_tensor(symmetry_dofs(n_theta, n_radial), dtype=torch.long),
        plane=hole.plane,
        element_type="q4",
    )
    meta = {
        "hole": hole,
        "nodes": nodes,
        "elements": elements,
        "thetas": thetas,
        "rhos": rhos,
        "forces": forces,
        "n_theta": n_theta,
        "n_radial": n_radial,
        "grading": grading,
        "rho_outer": rho_outer,
    }
    return model, meta


# ---------------------------------------------------------------------------
# Independent Q4 stress recovery
# ---------------------------------------------------------------------------

def _constitutive(hole: InglisHole) -> np.ndarray:
    e, nu = hole.effective_constants
    return e / (1.0 - nu * nu) * np.array(
        [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, 0.5 * (1.0 - nu)]]
    )


def _q4_b(xi: float, eta: float, coordinates: np.ndarray) -> tuple[np.ndarray, float]:
    """Bilinear strain-displacement matrix, written independently of the solver."""
    natural = np.array(
        [
            [-(1.0 - eta), -(1.0 - xi)],
            [(1.0 - eta), -(1.0 + xi)],
            [(1.0 + eta), (1.0 + xi)],
            [-(1.0 + eta), (1.0 - xi)],
        ]
    ) / 4.0
    jacobian = coordinates.T @ natural
    determinant = float(np.linalg.det(jacobian))
    cartesian = natural @ np.linalg.inv(jacobian)
    b = np.zeros((3, 8))
    b[0, 0::2] = cartesian[:, 0]
    b[1, 1::2] = cartesian[:, 1]
    b[2, 0::2] = cartesian[:, 1]
    b[2, 1::2] = cartesian[:, 0]
    return b, determinant


def _element_local_displacement(meta: dict, displacement, element: int) -> tuple[np.ndarray, np.ndarray]:
    connectivity = np.asarray(meta["elements"][element])
    local = np.empty(8)
    u = np.asarray(displacement, dtype=np.float64).reshape(-1, 2)
    local[0::2] = u[connectivity, 0]
    local[1::2] = u[connectivity, 1]
    coordinates = np.asarray(meta["nodes"], dtype=np.float64)[connectivity]
    return local, coordinates


def q4_stress_at_point(meta: dict, displacement, point) -> dict:
    """Independent pointwise stress at a physical point, plus its element id.

    The element is located through the exact inverse map (``zeta`` gives the
    ``(theta, rho)`` indices directly), then the parent coordinates follow from
    the structured grid.
    """
    hole: InglisHole = meta["hole"]
    thetas, rhos = meta["thetas"], meta["rhos"]
    n_theta, n_radial = meta["n_theta"], meta["n_radial"]
    zeta = hole.preimage(*np.asarray(point[:2], dtype=np.float64))
    # clamp instead of wrapping with a modulo: a point on the symmetry axis can
    # come back as theta = pi/2 + 1e-17, and ``% (pi/2)`` would then send it to
    # the *other* end of the quarter (the load axis) and sample the wrong element
    theta = math.atan2(zeta.imag, zeta.real)
    theta = min(max(theta, 0.0), 0.5 * math.pi)
    rho = max(abs(zeta), 1.0)
    i = min(max(int(np.searchsorted(thetas, theta)) - 1, 0), n_theta - 1)
    j = min(max(int(np.searchsorted(rhos, rho)) - 1, 0), n_radial - 1)
    element = i * n_radial + j
    local, coordinates = _element_local_displacement(meta, displacement, element)
    xi = 2.0 * (theta - thetas[i]) / (thetas[i + 1] - thetas[i]) - 1.0
    eta = 2.0 * (rho - rhos[j]) / (rhos[j + 1] - rhos[j]) - 1.0
    xi = float(min(max(xi, -1.0), 1.0))
    eta = float(min(max(eta, -1.0), 1.0))
    b, determinant = _q4_b(xi, eta, coordinates)
    stress = _constitutive(hole) @ b @ local
    return {
        "point": [float(point[0]), float(point[1])],
        "element": element,
        "parent": [xi, eta],
        "jacobian": determinant,
        "stress": [float(v) for v in stress],
    }


def element_centre_cross_check(meta: dict, displacement, stored_stress) -> dict:
    """Reproduce the production element-centre stress with the independent B matrix."""
    hole: InglisHole = meta["hole"]
    constitutive = _constitutive(hole)
    stored = np.asarray(stored_stress, dtype=np.float64)
    scale = max(float(np.abs(stored).max()), 1e-30)
    worst = 0.0
    for element in range(stored.shape[0]):
        local, coordinates = _element_local_displacement(meta, displacement, element)
        b, _ = _q4_b(0.0, 0.0, coordinates)
        mine = constitutive @ b @ local
        worst = max(worst, float(np.abs(mine - stored[element]).max()) / scale)
    return {"max_relative_difference": worst}


def tip_nodal_stress(meta: dict, displacement, i: int, j: int) -> dict:
    """Extrapolated nodal stress at grid node ``(i, j)`` (2x2 Gauss -> node).

    Bilinear extrapolation of the four Gauss values of every element touching the
    node, averaged.  Used for the peak, because a raw element-centre value
    under-reads a stress-concentration tip by ~10% on coarse meshes.
    """
    hole: InglisHole = meta["hole"]
    n_theta, n_radial = meta["n_theta"], meta["n_radial"]
    stride = n_radial + 1
    constitutive = _constitutive(hole)
    accum = np.zeros(3)
    hosts = 0
    for ei in (i - 1, i):
        for ej in (j - 1, j):
            if not (0 <= ei < n_theta and 0 <= ej < n_radial):
                continue
            local, coordinates = _element_local_displacement(meta, displacement, ei * n_radial + ej)
            gauss_values = np.empty((4, 3))
            for index, (xi, eta) in enumerate(_ELEMENT_GAUSS):
                b, _ = _q4_b(xi, eta, coordinates)
                gauss_values[index] = constitutive @ b @ local
            coefficients = np.linalg.solve(_GAUSS_BASIS, gauss_values)
            xi_node = -1.0 if ei == i else 1.0
            eta_node = -1.0 if ej == j else 1.0
            basis = np.array([1.0, xi_node, eta_node, xi_node * eta_node])
            accum += basis @ coefficients
            hosts += 1
    return {
        "node": int(i * stride + j),
        "hosts": hosts,
        "stress": [float(v) for v in accum / max(hosts, 1)],
    }


def nodal_stress_field(meta: dict, displacement, i: int, j: int) -> list[float]:
    """Extrapolated nodal stress at grid node ``(i, j)`` of the structured grid."""
    return tip_nodal_stress(meta, displacement, i, j)["stress"]


def boundary_hoop_profile(hole: InglisHole, count: int = 181) -> list[dict]:
    """Exact hoop stress around the hole boundary (tables and figures)."""
    rows = []
    for theta in np.linspace(0.0, 0.5 * math.pi, count):
        sigma = complex(math.cos(theta), math.sin(theta))
        point = hole.physical(sigma)
        value = float(hole.boundary_hoop_stress(theta))
        rows.append(
            {
                "theta_degrees": math.degrees(theta),
                "x": float(point.real),
                "y": float(point.imag),
                "hoop_over_far_field": value / hole.far_field,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Production solve and error measurement
# ---------------------------------------------------------------------------

def _nodal_displacement_error(hole: InglisHole, nodes: torch.Tensor, displacement) -> dict:
    coordinates = np.asarray(nodes, dtype=np.float64)
    exact_x, exact_y = hole.displacement_cartesian(coordinates[:, 0], coordinates[:, 1])
    u = np.asarray(displacement, dtype=np.float64).reshape(-1, 2)
    error = np.hypot(u[:, 0] - exact_x, u[:, 1] - exact_y)
    reference = np.hypot(exact_x, exact_y)
    scale = max(float(reference.max()), 1e-30)
    return {
        "max_absolute": float(error.max()),
        "max_relative": float(error.max() / scale),
        "rms_absolute": float(np.sqrt(np.mean(error * error))),
        "reference_max": float(scale),
    }


def _symmetry_edge_traction(hole: InglisHole, meta: dict) -> dict:
    """Exact x-resultant of the traction on the ``theta = pi/2`` edge (reaction check)."""
    rhos = meta["rhos"]
    zeta = 1j * rhos
    sxx, _, _ = hole.stress_at_zeta(zeta)
    # on theta = pi/2 the outward normal of the quarter domain is -e_x, so the
    # traction the fixed edge must supply is sigma_xx * e_x
    dz_drho = hole.mapping_radius * (1.0 + hole.mapping_m / (rhos * rhos))
    integrand = sxx * np.abs(dz_drho)
    integral = np.trapezoid(integrand, rhos) if hasattr(np, "trapezoid") else np.trapz(integrand, rhos)
    return {"exact_x_resultant": float(integral)}


def solve_case(
    hole: InglisHole | None = None,
    rho_outer: float = 5.0,
    n_theta: int = 24,
    n_radial: int | None = None,
    grading: float = 3.0,
    sample_scales=(1.05, 1.2, 1.6, 2.4),
) -> dict:
    """Solve one quarter mesh with the production path and measure against exact."""
    hole = hole or InglisHole()
    model, meta = annulus_model(hole, rho_outer, n_theta, n_radial, grading)
    result = solve_continuum(model)
    displacement = result.displacement.detach().cpu().numpy()
    displacement_error = _nodal_displacement_error(hole, meta["nodes"], displacement)
    cross = element_centre_cross_check(
        meta, displacement, result.stress.detach().cpu().numpy()
    )

    # peak hoop stress at both tips of the quarter domain (theta = pi/2 and 0)
    transverse = tip_nodal_stress(meta, displacement, meta["n_theta"], 0)
    load_axis = tip_nodal_stress(meta, displacement, 0, 0)
    transverse_hoop = transverse["stress"][0]
    load_axis_hoop = load_axis["stress"][1]
    exact_peak = hole.stress_concentration * hole.far_field
    exact_load_tip = -hole.far_field

    # stress profile along the transverse symmetry axis, node by node: the nodal
    # extrapolated field is the post-processing convention used for the peak, so
    # the profile and the peak are consistent with each other
    rows = []
    for j in range(meta["n_radial"] + 1):
        node_stress = nodal_stress_field(meta, displacement, meta["n_theta"], j)
        point = hole.physical(complex(0.0, meta["rhos"][j]))
        exact = hole.stress_cartesian(0.0, float(point.imag))
        rows.append(
            {
                "y": float(point.imag),
                "y_over_semi_y": float(point.imag / hole.semi_y),
                "sigma_xx_fe": float(node_stress[0]),
                "sigma_xx_exact": float(exact[0]),
                "sigma_yy_fe": float(node_stress[1]),
                "sigma_yy_exact": float(exact[1]),
                "relative_error": float(
                    abs(node_stress[0] - exact[0]) / max(abs(hole.stress_concentration * hole.far_field), 1e-30)
                ),
            }
        )

    # equilibrium: applied loads + reactions must cancel; and the x-reaction on
    # the transverse symmetry edge must equal the exact traction resultant there
    applied = meta["forces"].detach().cpu().numpy().reshape(-1)
    reaction = result.reaction.detach().cpu().numpy().reshape(-1)
    force_balance = np.array(
        [
            float(applied[0::2].sum() + reaction[0::2].sum()),
            float(applied[1::2].sum() + reaction[1::2].sum()),
        ]
    )
    edge = _symmetry_edge_traction(hole, meta)
    fixed = np.asarray(symmetry_dofs(meta["n_theta"], meta["n_radial"]))
    x_only = fixed[(fixed % 2) == 0]
    # ``reaction = K u - f`` is the residual, i.e. minus the force the support
    # supplies; the exact traction resultant acts in +x on the theta=pi/2 edge.
    reaction_x = -float(reaction[x_only].sum())
    return {
        "n_theta": meta["n_theta"],
        "n_radial": meta["n_radial"],
        "grading": grading,
        "rho_outer": rho_outer,
        "elements": int(meta["elements"].shape[0]),
        "dofs": int(2 * meta["nodes"].shape[0]),
        "semi_x": hole.semi_x,
        "semi_y": hole.semi_y,
        "mapping_m": hole.mapping_m,
        "displacement_error": displacement_error,
        "recovery_cross_check": cross,
        "transverse_tip": {
            "hoop_fe": float(transverse_hoop),
            "exact": float(exact_peak),
            "relative_error": float(abs(transverse_hoop - exact_peak) / abs(exact_peak)),
            "hosts": transverse["hosts"],
        },
        "load_axis_tip": {
            "hoop_fe": float(load_axis_hoop),
            "exact": float(exact_load_tip),
            "relative_error": float(abs(load_axis_hoop - exact_load_tip) / abs(exact_load_tip)),
            "hosts": load_axis["hosts"],
        },
        "sample_rows": rows,
        "force_balance": [float(v) for v in force_balance],
        "force_balance_relative": float(
            np.abs(force_balance).max()
            / max(float(np.abs(applied).sum()), 1e-30)
        ),
        "symmetry_edge": {
            "reaction_x": reaction_x,
            "exact_x_resultant": edge["exact_x_resultant"],
            "relative_error": float(
                abs(reaction_x - edge["exact_x_resultant"]) / abs(edge["exact_x_resultant"])
            ),
        },
    }


def convergence_study(
    hole: InglisHole | None = None,
    rho_outer: float = 5.0,
    levels=(16, 24, 32, 40, 48),
    grading: float = 3.0,
) -> dict:
    """Run the mesh ladder and report the observed rate of the peak stress."""
    hole = hole or InglisHole()
    cases = [
        solve_case(hole, rho_outer=rho_outer, n_theta=n, grading=grading) for n in levels
    ]
    peaks = [case["transverse_tip"]["hoop_fe"] for case in cases]
    exact = hole.stress_concentration * hole.far_field
    rates = []
    for k in range(1, len(cases)):
        ratio = abs(peaks[k] - exact) / max(abs(peaks[k - 1] - exact), 1e-300)
        size_ratio = levels[k] / levels[k - 1]
        rates.append(float(-math.log(ratio) / math.log(size_ratio)) if ratio > 0 else float("inf"))
    # Richardson extrapolation from the last three levels with the observed rate
    if len(peaks) >= 3:
        p = rates[-1] if rates else 2.0
        f2, f3 = peaks[-2], peaks[-1]
        n2, n3 = levels[-2], levels[-1]
        ratio = (n3 / n2) ** p
        extrapolated = f3 + (f3 - f2) / max(ratio - 1.0, 1e-12)
    else:
        extrapolated = peaks[-1]
    return {
        "levels": list(levels),
        "peaks": [float(v) for v in peaks],
        "exact": float(exact),
        "observed_rates": rates,
        "richardson": float(extrapolated),
        "richardson_relative_error": float(abs(extrapolated - exact) / abs(exact)),
        "cases": cases,
    }