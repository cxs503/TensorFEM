"""Kirsch circular-hole stress-concentration reference and quarter-annulus Q4 case.

Reference
---------
G. Kirsch (1898), re-derived in S. Timoshenko & J.N. Goodier, *Theory of
Elasticity*, 3rd ed., §39, and in A.P. Boresi & R.J. Schmidt, *Advanced
Mechanics of Materials*, §15.  For an infinite plate under uniaxial tension
``T`` along ``x`` with a traction-free circular hole of radius ``a``::

    sigma_rr = T/2 (1 - a^2/r^2) + T/2 (1 - 4 a^2/r^2 + 3 a^4/r^4) cos 2 theta
    sigma_tt = T/2 (1 + a^2/r^2) - T/2 (1 + 3 a^4/r^4) cos 2 theta
    sigma_rt = -T/2 (1 + 2 a^2/r^2 - 3 a^4/r^4) sin 2 theta

with the closed-form displacement field (plane stress, ``kappa`` form derived
here for ``E``/``nu`` directly)::

    u_r = T/(2E) { [(1-nu) r + (1+nu) a^2/r]
                  + [(1+nu) r + 4 a^2/r - (1+nu) a^4/r^3] cos 2 theta }
    u_t = T/(2E) [-(1+nu) r + 2 (nu-1) a^2/r - (1+nu) a^4/r^3] sin 2 theta

The hoop stress on the hole edge is ``sigma_tt(a, pi/2) = 3 T`` (the classical
stress-concentration factor ``K_t = 3``) and ``sigma_tt(a, 0) = -T``; both are
reproduced here as hard identities so the reference cannot drift silently.

Both the traction-free hole and the far field are exact for every ``r >= a``,
so the *annulus* ``a <= r <= b`` loaded on ``r = b`` by the Kirsch tractions has
the Kirsch field as its exact, unique solution.  That makes the quarter model
``0 <= theta <= pi/2`` with symmetry conditions (``u_y = 0`` on ``theta = 0``,
``u_x = 0`` on ``theta = pi/2``) a genuine verification case rather than a
FE-vs-FE comparison: the imposed boundary data and the exact solution are the
same mathematical object.

Verification use
----------------
:func:`solve_case` drives the *production* plane-stress path
(:class:`tensorfem.continuum.ContinuumModel` + :func:`solve_continuum`) on a
graded polar Q4 mesh and measures

* nodal displacement error against the closed form,
* element-centre / pointwise stress error against the closed form at fixed
  physical sample points (recovered by an independent Q4 derivative
  implementation in this module, cross-checked against the stored production
  stress so the sampling path itself is validated),
* the stress-concentration factor at the hole edge vs ``K_t = 3``,
* force balance between the consistent arc loads and the reactions.

``scripts/audit_kirsch_hole.py`` writes the JSON/figures and
``docs/kirsch-hole-stress-concentration.md`` is the write-up.

Everything here is double-precision: the case is a verification benchmark, not
a differentiable training target, and no autograd graph is retained.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .continuum import ContinuumModel, solve_continuum

__all__ = [
    "KirschHole",
    "quarter_annulus_mesh",
    "consistent_arc_loads",
    "symmetry_dofs",
    "annulus_model",
    "solve_case",
    "convergence_study",
    "q4_stress_at_point",
    "sample_rows",
    "element_centre_cross_check",
    "arc_resultant",
]

DEFAULT_SAMPLE_POINTS = (
    (1.02, math.pi / 2),
    (1.05, math.pi / 2),
    (1.10, math.pi / 2),
    (1.25, math.pi / 2),
    (1.50, math.pi / 2),
    (2.00, math.pi / 2),
)


# ---------------------------------------------------------------------------
# Exact Kirsch solution
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KirschHole:
    """Material/geometry parameters of the uniaxial-tension Kirsch problem.

    ``plane='strain'`` only changes the *displacement* field: the stress field of
    a traction-driven plane problem is material independent, and the plane-strain
    compliance equals the plane-stress one under ``E -> E/(1-nu^2)`` and
    ``nu -> nu/(1-nu)``.  That substitution is applied by
    :attr:`effective_constants` and is verified in the test suite.
    """

    radius: float = 1.0
    far_field: float = 1.0
    young: float = 1.0
    poisson: float = 0.3
    plane: str = "stress"
    thickness: float = 1.0

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError("hole radius must be positive")
        if self.plane not in ("stress", "strain"):
            raise ValueError("plane must be 'stress' or 'strain'")
        if not -0.9999 < self.poisson < 0.5:
            raise ValueError("poisson ratio out of range for a stable solid")
        if self.young <= 0.0 or self.thickness <= 0.0:
            raise ValueError("young modulus and thickness must be positive")

    # -- material -----------------------------------------------------------
    @property
    def effective_constants(self) -> tuple[float, float]:
        """``(E, nu)`` of the equivalent plane-stress problem."""
        nu = self.poisson
        if self.plane == "stress":
            return self.young, nu
        return self.young / (1.0 - nu * nu), nu / (1.0 - nu)

    @property
    def stress_concentration(self) -> float:
        """Exact ``sigma_tt(a, pi/2) / T`` = 3."""
        return 3.0

    # -- stress -------------------------------------------------------------
    def stress_polar(self, r, theta):
        """Exact ``(sigma_rr, sigma_tt, sigma_rt)``; ``r``/``theta`` broadcast."""
        r = np.asarray(r, dtype=np.float64)
        theta = np.asarray(theta, dtype=np.float64)
        if np.any(r <= 0.0):
            raise ValueError("r must be positive")
        a = self.radius
        t = self.far_field
        c2, s2 = np.cos(2.0 * theta), np.sin(2.0 * theta)
        a2r2 = (a / r) ** 2
        a4r4 = a2r2 * a2r2
        srr = t / 2.0 * (1.0 - a2r2) + t / 2.0 * (1.0 - 4.0 * a2r2 + 3.0 * a4r4) * c2
        stt = t / 2.0 * (1.0 + a2r2) - t / 2.0 * (1.0 + 3.0 * a4r4) * c2
        srt = -t / 2.0 * (1.0 + 2.0 * a2r2 - 3.0 * a4r4) * s2
        return srr, stt, srt

    def stress_cartesian(self, x, y):
        """Exact ``(sigma_xx, sigma_yy, sigma_xy)`` at Cartesian points."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        r = np.hypot(x, y)
        theta = np.arctan2(y, x)
        srr, stt, srt = self.stress_polar(r, theta)
        c, s = np.cos(theta), np.sin(theta)
        sxx = srr * c * c + stt * s * s - 2.0 * srt * c * s
        syy = srr * s * s + stt * c * c + 2.0 * srt * c * s
        sxy = (srr - stt) * c * s + srt * (c * c - s * s)
        return sxx, syy, sxy

    def von_mises_polar(self, r, theta):
        srr, stt, srt = self.stress_polar(r, theta)
        return np.sqrt(srr * srr - srr * stt + stt * stt + 3.0 * srt * srt)

    # -- displacement -------------------------------------------------------
    def displacement_polar(self, r, theta):
        """Exact ``(u_r, u_t)`` of the equivalent plane problem."""
        r = np.asarray(r, dtype=np.float64)
        theta = np.asarray(theta, dtype=np.float64)
        e, nu = self.effective_constants
        a = self.radius
        t = self.far_field
        f = t / (2.0 * e)
        c2, s2 = np.cos(2.0 * theta), np.sin(2.0 * theta)
        a2 = a * a
        a4 = a2 * a2
        ur = f * ((1.0 - nu) * r + (1.0 + nu) * a2 / r) + f * (
            (1.0 + nu) * r + 4.0 * a2 / r - (1.0 + nu) * a4 / r**3
        ) * c2
        ut = f * (-(1.0 + nu) * r + 2.0 * (nu - 1.0) * a2 / r - (1.0 + nu) * a4 / r**3) * s2
        return ur, ut

    def displacement_cartesian(self, x, y):
        """Exact ``(u_x, u_y)`` at Cartesian points."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        r = np.hypot(x, y)
        theta = np.arctan2(y, x)
        ur, ut = self.displacement_polar(r, theta)
        c, s = np.cos(theta), np.sin(theta)
        return ur * c - ut * s, ur * s + ut * c

    def traction_on_arc(self, r, theta):
        """Traction vector ``sigma . e_r`` of the exact field on the arc ``r``."""
        srr, _, srt = self.stress_polar(r, theta)
        c, s = np.cos(theta), np.sin(theta)
        return srr * c - srt * s, srr * s + srt * c


# ---------------------------------------------------------------------------
# Graded polar quarter-annulus mesh
# ---------------------------------------------------------------------------

def _graded_radii(radius: float, outer_radius: float, n_radial: int, grading: float) -> np.ndarray:
    if outer_radius <= radius:
        raise ValueError("outer radius must exceed the hole radius")
    if n_radial < 1:
        raise ValueError("n_radial must be positive")
    s = np.linspace(0.0, 1.0, n_radial + 1)
    if grading > 0.0:
        s = (np.exp(grading * s) - 1.0) / (math.exp(grading) - 1.0)
    return radius + (outer_radius - radius) * s


def quarter_annulus_mesh(
    hole: KirschHole,
    outer_radius: float,
    n_theta: int,
    n_radial: int,
    grading: float = 5.0,
    *,
    dtype=torch.float64,
):
    """Structured, counter-clockwise Q4 quarter annulus ``0 <= theta <= pi/2``.

    Radial spacing is geometric towards the hole (``grading`` controls the
    concentration; ``grading=0`` is uniform) because the stress gradient is
    strongest at ``r = a``.  The hole edge is a polygon with nodes on ``r = a``;
    the chord sagitta is ``O((pi/2/n_theta)^2)`` and is refined with the mesh.

    Returns ``(nodes, elements, radii, thetas)`` where ``nodes`` has shape
    ``[n_nodes, 2]`` ordered as ``i_theta * (n_radial + 1) + j_radius``.
    """
    if n_theta < 1:
        raise ValueError("n_theta must be positive")
    radii = _graded_radii(hole.radius, outer_radius, n_radial, grading)
    thetas = np.linspace(0.0, math.pi / 2.0, n_theta + 1)
    cos_t, sin_t = np.cos(thetas)[:, None], np.sin(thetas)[:, None]
    nodes = np.stack(
        [radii[None, :] * cos_t, radii[None, :] * sin_t], axis=-1
    ).reshape(-1, 2)

    def index(i: int, j: int) -> int:
        return i * (n_radial + 1) + j

    elements = np.empty((n_theta * n_radial, 4), dtype=np.int64)
    for i in range(n_theta):
        for j in range(n_radial):
            elements[i * n_radial + j] = (
                index(i, j), index(i, j + 1), index(i + 1, j + 1), index(i + 1, j)
            )
    return (
        torch.as_tensor(nodes, dtype=dtype),
        torch.as_tensor(elements, dtype=torch.long),
        radii,
        thetas,
    )


def consistent_arc_loads(
    hole: KirschHole,
    nodes: torch.Tensor,
    outer_radius: float,
    n_theta: int,
    n_radial: int,
    *,
    dtype=torch.float64,
) -> torch.Tensor:
    """Nodal forces equivalent to the exact Kirsch traction on the outer arc.

    Three-point Gauss along each boundary edge, with the traction evaluated at the
    *straight* chord point (the mesh edge), so the loads are consistent with the
    discretised boundary and their resultant can be compared with an independent
    high-accuracy quadrature of the exact traction.
    """
    coordinates = np.asarray(nodes, dtype=np.float64)
    forces = np.zeros(2 * coordinates.shape[0], dtype=np.float64)
    gauss = ((-math.sqrt(3.0 / 5.0), 5.0 / 9.0), (0.0, 8.0 / 9.0), (math.sqrt(3.0 / 5.0), 5.0 / 9.0))
    for i in range(n_theta):
        first = i * (n_radial + 1) + n_radial
        second = (i + 1) * (n_radial + 1) + n_radial
        p1, p2 = coordinates[first], coordinates[second]
        length = float(np.linalg.norm(p2 - p1))
        for xi, weight in gauss:
            n1, n2 = (1.0 - xi) / 2.0, (1.0 + xi) / 2.0
            point = n1 * p1 + n2 * p2
            r = float(np.hypot(point[0], point[1]))
            theta = float(np.arctan2(point[1], point[0]))
            tx, ty = hole.traction_on_arc(r, theta)
            jacobian = 0.5 * length * weight
            forces[2 * first] += n1 * tx * jacobian
            forces[2 * first + 1] += n1 * ty * jacobian
            forces[2 * second] += n2 * tx * jacobian
            forces[2 * second + 1] += n2 * ty * jacobian
    return torch.as_tensor(forces, dtype=dtype)


def symmetry_dofs(n_theta: int, n_radial: int) -> list[int]:
    """Zero-displacement dofs of the quarter model (``u_y`` on ``theta=0``, ``u_x`` on ``theta=pi/2``)."""
    fixed: set[int] = set()
    for i in range(n_theta + 1):
        for j in range(n_radial + 1):
            node = i * (n_radial + 1) + j
            if i == 0:
                fixed.add(2 * node + 1)
            if i == n_theta:
                fixed.add(2 * node)
    return sorted(fixed)


# ---------------------------------------------------------------------------
# Production solve
# ---------------------------------------------------------------------------

def annulus_model(
    hole: KirschHole,
    outer_radius: float,
    n_theta: int,
    n_radial: int,
    grading: float = 5.0,
    *,
    dtype=torch.float64,
) -> tuple[ContinuumModel, dict]:
    """Build the production :class:`ContinuumModel` of the quarter-annulus case."""
    nodes, elements, radii, thetas = quarter_annulus_mesh(
        hole, outer_radius, n_theta, n_radial, grading, dtype=dtype
    )
    forces = consistent_arc_loads(hole, nodes, outer_radius, n_theta, n_radial, dtype=dtype)
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
        "nodes": nodes,
        "elements": elements,
        "radii": radii,
        "thetas": thetas,
        "forces": forces,
        "n_theta": n_theta,
        "n_radial": n_radial,
        "grading": grading,
        "outer_radius": outer_radius,
    }
    return model, meta


# ---------------------------------------------------------------------------
# Independent Q4 stress recovery at arbitrary points
# ---------------------------------------------------------------------------

def _constitutive(hole: KirschHole) -> np.ndarray:
    e, nu = hole.effective_constants
    return e / (1.0 - nu * nu) * np.array(
        [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, 0.5 * (1.0 - nu)]]
    )


def _q4_b(xi: float, eta: float, coordinates: np.ndarray) -> tuple[np.ndarray, float]:
    """Bilinear strain-displacement matrix and Jacobian determinant.

    Written independently of :mod:`tensorfem.continuum` so that the pointwise
    recovery is a second implementation; ``element_centre_cross_check`` proves the
    two agree at element centres before any pointwise comparison is published.
    """
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


def _locate(meta: dict, point) -> tuple[int, float, float]:
    """Element index and parent coordinates of a physical point in the polar grid."""
    radii, thetas = meta["radii"], meta["thetas"]
    n_theta, n_radial = meta["n_theta"], meta["n_radial"]
    x, y = float(point[0]), float(point[1])
    r = math.hypot(x, y)
    theta = math.atan2(y, x)
    i = min(max(int(np.searchsorted(thetas, theta)) - 1, 0), n_theta - 1)
    j = min(max(int(np.searchsorted(radii, r)) - 1, 0), n_radial - 1)
    connectivity = meta["elements"][i * n_radial + j]
    coordinates = np.asarray(meta["nodes"], dtype=np.float64)[connectivity]
    target = np.array([x, y])
    xi = eta = 0.0
    for _ in range(80):
        shape = np.array(
            [
                (1.0 - xi) * (1.0 - eta),
                (1.0 + xi) * (1.0 - eta),
                (1.0 + xi) * (1.0 + eta),
                (1.0 - xi) * (1.0 + eta),
            ]
        ) / 4.0
        natural = np.array(
            [
                [-(1.0 - eta), -(1.0 - xi)],
                [(1.0 - eta), -(1.0 + xi)],
                [(1.0 + eta), (1.0 + xi)],
                [-(1.0 + eta), (1.0 - xi)],
            ]
        ) / 4.0
        residual = shape @ coordinates - target
        jacobian = coordinates.T @ natural
        step = np.linalg.solve(jacobian, -residual)
        norm = float(np.linalg.norm(step))
        if norm > 1.0:
            step = step / norm
        xi += step[0]
        eta += step[1]
        if norm < 1e-14:
            break
    return int(i * n_radial + j), float(xi), float(eta)


def q4_stress_at_point(meta: dict, displacement, point) -> dict:
    """Independent pointwise stress + exact reference at a physical point."""
    element, xi, eta = _locate(meta, point)
    connectivity = meta["elements"][element]
    coordinates = np.asarray(meta["nodes"], dtype=np.float64)[connectivity]
    u = np.asarray(displacement, dtype=np.float64).reshape(-1, 2)
    local = np.empty(8)
    local[0::2] = u[connectivity, 0]
    local[1::2] = u[connectivity, 1]
    b, determinant = _q4_b(xi, eta, coordinates)
    cartesian = _constitutive(meta["hole"]) @ b @ local
    return {
        "point": [float(point[0]), float(point[1])],
        "element": element,
        "parent": [xi, eta],
        "inside": bool(-1.0 - 1e-9 <= xi <= 1.0 + 1e-9 and -1.0 - 1e-9 <= eta <= 1.0 + 1e-9),
        "jacobian": determinant,
        "stress": [float(v) for v in cartesian],
    }


def element_centre_cross_check(meta: dict, displacement, stored_stress) -> dict:
    """Reproduce the production element-centre stress with the independent B matrix."""
    nodes = np.asarray(meta["nodes"], dtype=np.float64)
    elements = np.asarray(meta["elements"])
    u = np.asarray(displacement, dtype=np.float64).reshape(-1, 2)
    stored = np.asarray(stored_stress, dtype=np.float64)
    worst = 0.0
    for element in range(elements.shape[0]):
        connectivity = elements[element]
        local = np.empty(8)
        local[0::2] = u[connectivity, 0]
        local[1::2] = u[connectivity, 1]
        b, _ = _q4_b(0.0, 0.0, nodes[connectivity])
        mine = _constitutive(meta["hole"]) @ b @ local
        scale = max(float(np.linalg.norm(stored[element])), 1e-14)
        worst = max(worst, float(np.linalg.norm(mine - stored[element]) / scale))
    return {
        "elements": int(elements.shape[0]),
        "max_relative_difference": worst,
        "passed": worst <= 1e-10,
    }


# ---------------------------------------------------------------------------
# Case driver
# ---------------------------------------------------------------------------

def _nodal_displacement_error(hole: KirschHole, nodes: torch.Tensor, displacement) -> dict:
    coordinates = np.asarray(nodes, dtype=np.float64)
    exact_x, exact_y = hole.displacement_cartesian(coordinates[:, 0], coordinates[:, 1])
    exact = np.stack([exact_x, exact_y], axis=1)
    u = np.asarray(displacement, dtype=np.float64).reshape(-1, 2)
    difference = u - exact
    return {
        "l2_relative": float(np.linalg.norm(difference) / np.linalg.norm(exact)),
        "max_relative": float(
            np.max(np.linalg.norm(difference, axis=1)) / np.max(np.linalg.norm(exact, axis=1))
        ),
    }


def arc_resultant(hole: KirschHole, outer_radius: float, samples: int = 200) -> list[float]:
    """Exact resultant of the Kirsch traction over the quarter arc ``r = outer_radius``.

    200-point Gauss-Legendre on ``[0, pi/2]``; the integrand is smooth, so this is
    an independent high-accuracy oracle for the consistent-load generator.
    """
    nodes, weights = np.polynomial.legendre.leggauss(samples)
    theta = math.pi / 4.0 * (nodes + 1.0)
    weight = math.pi / 4.0 * weights
    tx, ty = hole.traction_on_arc(outer_radius, theta)
    return [float((weight * tx * outer_radius).sum()), float((weight * ty * outer_radius).sum())]


def _hole_edge_peak(hole: KirschHole, meta: dict, displacement) -> dict:
    """Peak hoop stress on the hole edge by nodal averaging, vs ``3 T``."""
    nodes = np.asarray(meta["nodes"], dtype=np.float64)
    elements = np.asarray(meta["elements"])
    u = np.asarray(displacement, dtype=np.float64).reshape(-1, 2)
    n_radial = meta["n_radial"]
    hole_nodes = [i * (n_radial + 1) for i in range(meta["n_theta"] + 1)]
    accumulate = {node: np.zeros(3) for node in hole_nodes}
    count = {node: 0 for node in hole_nodes}
    corner_parent = {0: (-1.0, -1.0), 1: (1.0, -1.0), 2: (1.0, 1.0), 3: (-1.0, 1.0)}
    for element in range(elements.shape[0]):
        connectivity = elements[element]
        hits = [local for local, node in enumerate(connectivity) if node in accumulate]
        if not hits:
            continue
        local_u = np.empty(8)
        local_u[0::2] = u[connectivity, 0]
        local_u[1::2] = u[connectivity, 1]
        coordinates = nodes[connectivity]
        for local in hits:
            xi, eta = corner_parent[local]
            b, _ = _q4_b(xi, eta, coordinates)
            accumulate[connectivity[local]] += _constitutive(meta["hole"]) @ b @ local_u
            count[connectivity[local]] += 1
    hoop = {}
    thetas = meta["thetas"]
    for index, node in enumerate(hole_nodes):
        stress = accumulate[node] / max(count[node], 1)
        theta = thetas[index]
        c, s = math.cos(theta), math.sin(theta)
        stt = stress[0] * s * s + stress[1] * c * c - 2.0 * stress[2] * c * s
        hoop[round(theta, 12)] = float(stt)
    peak_theta = max(hoop, key=lambda key: hoop[key])
    return {
        "peak_hoop": hoop[peak_theta],
        "peak_theta_degrees": math.degrees(peak_theta),
        "hole_edge_at_90_degrees": hoop[round(math.pi / 2.0, 12)],
        "hole_edge_at_0_degrees": hoop[round(0.0, 12)],
        "exact_at_90_degrees": hole.stress_concentration * hole.far_field,
        "exact_at_0_degrees": -hole.far_field,
        "profile": [[math.degrees(theta), value] for theta, value in sorted(hoop.items())],
    }


def solve_case(
    hole: KirschHole | None = None,
    outer_radius: float = 10.0,
    n_theta: int = 24,
    n_radial: int | None = None,
    grading: float = 5.0,
    sample_points=DEFAULT_SAMPLE_POINTS,
) -> dict:
    """Solve one quarter-annulus mesh with the production path and measure errors."""
    hole = hole or KirschHole()
    n_radial = n_theta if n_radial is None else n_radial
    model, meta = annulus_model(hole, outer_radius, n_theta, n_radial, grading)
    meta["hole"] = hole
    result = solve_continuum(model)
    displacement = result.displacement.detach().cpu().numpy()
    displacement_error = _nodal_displacement_error(hole, meta["nodes"], displacement)

    rows = []
    for scale, theta in sample_points:
        point = (scale * hole.radius * math.cos(theta), scale * hole.radius * math.sin(theta))
        sampled = q4_stress_at_point(meta, displacement, point)
        sxx, syy, sxy = sampled["stress"]
        c, s = math.cos(theta), math.sin(theta)
        stt = sxx * s * s + syy * c * c - 2.0 * sxy * c * s
        srr = sxx * c * c + syy * s * s + 2.0 * sxy * c * s
        srt = (srr - stt) * c * s + sxy * (c * c - s * s)
        exact = hole.stress_polar(scale * hole.radius, theta)
        rows.append(
            {
                "r_over_a": scale,
                "theta_degrees": math.degrees(theta),
                "hoop_fe": float(stt),
                "hoop_exact": float(exact[1]),
                "radial_fe": float(srr),
                "radial_exact": float(exact[0]),
                "shear_fe": float(srt),
                "shear_exact": float(exact[2]),
                "hoop_relative_error": float(
                    abs(stt - exact[1])
                    / max(abs(exact[1]), 1e-12 * max(abs(hole.far_field), 1e-30))
                ),
                "inside": sampled["inside"],
                "element": sampled["element"],
            }
        )

    applied = meta["forces"].detach().cpu().numpy()
    reaction = result.reaction.detach().cpu().numpy()
    cross = element_centre_cross_check(
        meta, displacement, result.stress.detach().cpu().numpy()
    )
    return {
        "n_theta": n_theta,
        "n_radial": n_radial,
        "grading": grading,
        "elements": int(meta["elements"].shape[0]),
        "dofs": int(model.n_dofs),
        "displacement_error": displacement_error,
        "stress_rows": rows,
        "max_stress_relative_error": max(row["hoop_relative_error"] for row in rows),
        "hole_edge": _hole_edge_peak(hole, meta, displacement),
        "load_resultant": {
            "applied": [float(applied[0::2].sum()), float(applied[1::2].sum())],
            "exact_arc": arc_resultant(hole, outer_radius),
            "reaction": [float(reaction[0::2].sum()), float(reaction[1::2].sum())],
            "relative_error": float(
                abs(applied[0::2].sum() - arc_resultant(hole, outer_radius)[0])
                / max(abs(arc_resultant(hole, outer_radius)[0]), 1e-12)
            ),
        },
        "recovery_cross_check": cross,
        "meta": meta,
        "displacement": displacement,
        "stored_stress": result.stress.detach().cpu().numpy(),
    }


def _observed_order(coarse: float, fine: float, ratio: float) -> float:
    if coarse <= 0.0 or fine <= 0.0 or fine >= coarse:
        return float("nan")
    return math.log(coarse / fine) / math.log(ratio)


def convergence_study(
    hole: KirschHole | None = None,
    levels: tuple[int, ...] = (16, 24, 32, 48),
    outer_radius: float = 10.0,
    grading: float = 5.0,
    sample_points=DEFAULT_SAMPLE_POINTS,
) -> dict:
    """Refine the quarter-annulus mesh and tabulate the measured errors."""
    hole = hole or KirschHole()
    cases = [
        solve_case(hole, outer_radius, n_theta=n, n_radial=n, grading=grading, sample_points=sample_points)
        for n in levels
    ]
    orders = []
    for coarse, fine in zip(cases, cases[1:]):
        ratio = fine["n_theta"] / coarse["n_theta"]
        orders.append(
            {
                "levels": [coarse["n_theta"], fine["n_theta"]],
                "displacement_order": _observed_order(
                    coarse["displacement_error"]["l2_relative"],
                    fine["displacement_error"]["l2_relative"],
                    ratio,
                ),
                "stress_order": _observed_order(
                    coarse["max_stress_relative_error"],
                    fine["max_stress_relative_error"],
                    ratio,
                ),
            }
        )
    return {
        "hole": {
            "radius": hole.radius,
            "far_field": hole.far_field,
            "young": hole.young,
            "poisson": hole.poisson,
            "plane": hole.plane,
            "outer_radius": outer_radius,
            "grading": grading,
        },
        "levels": list(levels),
        "cases": cases,
        "observed_orders": orders,
        "max_stress_relative_error": max(case["max_stress_relative_error"] for case in cases),
        "max_displacement_l2": max(case["displacement_error"]["l2_relative"] for case in cases),
    }


def sample_rows(case: dict) -> list[dict]:
    """Compact numeric rows of one case for report tables."""
    return [
        {
            key: row[key]
            for key in (
                "r_over_a",
                "hoop_fe",
                "hoop_exact",
                "hoop_relative_error",
                "radial_fe",
                "radial_exact",
                "shear_fe",
                "shear_exact",
            )
        }
        for row in case["stress_rows"]
    ]