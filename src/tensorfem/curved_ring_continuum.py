"""Plane-stress continuum oracle for the quarter-ring curved cantilever.

Why this module exists
----------------------
:mod:`tensorfem.curved_beam_benchmark` grades the shell elements against a
*curved-beam theory* closed form (Castigliano on a thin curved bar with axial and
transverse-shear terms retained)::

    delta_t = (3 pi/4 - 2) P R^3 / (E I) + pi P R / (4 E A) + pi P R / (4 k G A)
    delta_r = P R^3 / (2 E I)                  (tangent load, quarter ring)

Both the shell under test and that reference share the *beam/shell kinematic
assumption*, so agreement between them does not by itself prove the coefficient
``3 pi/4 - 2``.  This module supplies a genuinely independent route: the same
quarter ring meshed as a **two-dimensional plane-stress continuum** with
eight-node serendipity (Q8) elements and 3x3 Gauss integration.  The continuum
carries full 2-D elasticity (no plane-sections assumption, no shear-correction
factor, no curved-beam kinematics) and therefore converges to the *true*
solution of the boundary-value problem; the closed form is then graded against
it, and its validity envelope in ``t/R`` can be measured instead of assumed.

The continuum model is also the natural oracle for the *stress* field that the
shell elements recover (which is what structural-ice and hull-strength work
actually consumes), so :func:`hoop_stress_section` extracts the hoop stress
through a section and two analytic profiles are provided for comparison:

* :func:`winkler_bach_stress` -- curved-beam (Winkler--Bach) hoop stress
  ``N/A + M (r - r_n) / (A e r)`` with the neutral-axis radius
  ``r_n = A / int dA/r`` and ``e = R_c - r_n``;
* :func:`straight_beam_stress` -- the flat-beam formula ``N/A + M (r - R_c)/I``
  that is routinely (and, for curved sections, wrongly) used for a first check.

Geometry and sign conventions
-----------------------------
Polar sweep ``0 <= phi <= pi/2``, mid-surface radius ``R``, in-plane (radial)
depth ``t``, out-of-plane width ``b`` (the plane-stress thickness).  The root
cross-section is the line ``phi = pi/2`` (the ``x = 0`` axis, clamped in both
components); the free end is the line ``phi = 0`` (the ``y = 0`` axis), where at
the point ``(R, 0)`` the radial direction is ``+x`` and the tangent is ``+y``.

* ``direction="tangential"`` applies the total load ``P`` along ``+y`` on the
  free end (the load case of the shell benchmark);
* ``direction="radial"`` applies it along ``+x`` (used for the reciprocity
  check and for the direct radial compliance ``pi P R^3 / (4 E I)``).

Section resultants at a cut of angle ``phi`` (statics, independent of the
finite-element solution)::

    N(phi) = -P cos(phi)            (hoop force, positive tensile)
    M(phi) = -P R (1 - cos(phi))    (moment about the section centroid)

Bending inertia of the strip for the closed forms is ``I = b t^3 / 12`` with
``A = b t``, matching how :func:`tensorfem.curved_beam_benchmark.quarter_ring_reference`
builds its reference, so ``t`` is the radial depth and ``b`` the width.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "RingMesh",
    "RingSolution",
    "Q8_GAUSS",
    "q8_shape",
    "quarter_ring_q8_mesh",
    "solve_quarter_ring",
    "hoop_stress_section",
    "section_forces",
    "winkler_bach_stress",
    "straight_beam_stress",
    "richardson_limit",
]

#: 3x3 Gauss-Legendre rule on the parent square: ``(xi, eta, weight)``.
Q8_GAUSS: tuple[tuple[float, float, float], ...] = tuple(
    (xi, eta, wx * wy)
    for xi, wx in (
        (-math.sqrt(3.0 / 5.0), 5.0 / 9.0),
        (0.0, 8.0 / 9.0),
        (math.sqrt(3.0 / 5.0), 5.0 / 9.0),
    )
    for eta, wy in (
        (-math.sqrt(3.0 / 5.0), 5.0 / 9.0),
        (0.0, 8.0 / 9.0),
        (math.sqrt(3.0 / 5.0), 5.0 / 9.0),
    )
)


def q8_shape(xi: float, eta: float) -> tuple[np.ndarray, np.ndarray]:
    """Serendipity Q8 shape functions and parent-coordinate derivatives.

    Node order is the standard one: four corners counter-clockwise from
    ``(-1, -1)`` followed by the four edge mid-points (1-2, 2-3, 3-4, 4-1).
    """
    shape = np.array(
        [
            -0.25 * (1 - xi) * (1 - eta) * (1 + xi + eta),
            -0.25 * (1 + xi) * (1 - eta) * (1 - xi + eta),
            -0.25 * (1 + xi) * (1 + eta) * (1 - xi - eta),
            -0.25 * (1 - xi) * (1 + eta) * (1 + xi - eta),
            0.5 * (1 - xi * xi) * (1 - eta),
            0.5 * (1 + xi) * (1 - eta * eta),
            0.5 * (1 - xi * xi) * (1 + eta),
            0.5 * (1 - xi) * (1 - eta * eta),
        ]
    )
    derivative = np.array(
        [
            [0.25 * (1 - eta) * (2 * xi + eta), 0.25 * (1 - xi) * (xi + 2 * eta)],
            [0.25 * (1 - eta) * (2 * xi - eta), 0.25 * (1 + xi) * (2 * eta - xi)],
            [0.25 * (1 + eta) * (2 * xi + eta), 0.25 * (1 + xi) * (2 * eta + xi)],
            [0.25 * (1 + eta) * (2 * xi - eta), 0.25 * (1 - xi) * (2 * eta - xi)],
            [-xi * (1 - eta), -0.5 * (1 - xi * xi)],
            [0.5 * (1 - eta * eta), -(1 + xi) * eta],
            [-xi * (1 + eta), 0.5 * (1 - xi * xi)],
            [-0.5 * (1 - eta * eta), -(1 - xi) * eta],
        ]
    )
    return shape, derivative


@dataclass(frozen=True)
class RingMesh:
    """Structured Q8 quarter-ring mesh.

    ``n_arc`` elements sweep the quarter circle, ``n_thick`` elements span the
    radial depth.  ``elements[e]`` lists the eight node ids in the Q8 order.
    """

    nodes: np.ndarray
    elements: np.ndarray
    n_arc: int
    n_thick: int
    radius: float
    thickness: float
    root_nodes: np.ndarray
    free_end_nodes: np.ndarray
    centroid_node: int

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_dofs(self) -> int:
        return 2 * self.n_nodes


def quarter_ring_q8_mesh(
    n_arc: int, n_thick: int, radius: float, thickness: float
) -> RingMesh:
    """Build the Q8 quarter-ring mesh.

    Corners follow the radial direction first (``1 -> 2`` outward) so that every
    element has a positive Jacobian under the Q8 node ordering above.
    """
    for name, value in (("n_arc", n_arc), ("n_thick", n_thick)):
        if int(value) != value or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not (radius > 0.0 and 0.0 < thickness < 2.0 * radius):
        raise ValueError("need radius > 0 and 0 < thickness < 2*radius")

    index: dict[tuple, int] = {}
    points: list[tuple[float, float]] = []

    def add(key: tuple, xy: tuple[float, float]) -> int:
        if key not in index:
            index[key] = len(points)
            points.append(xy)
        return index[key]

    inner = radius - 0.5 * thickness

    def r_of(j: float) -> float:
        return inner + thickness * j / n_thick

    def phi_of(i: float) -> float:
        return 0.5 * math.pi * i / n_arc

    def corner(i: int, j: int) -> int:
        r, phi = r_of(j), phi_of(i)
        return add(("c", i, j), (r * math.cos(phi), r * math.sin(phi)))

    def mid_radius(i: int, j: int) -> int:
        """Mid-side node between corners ``(i, j)`` and ``(i, j+1)``."""
        r, phi = 0.5 * (r_of(j) + r_of(j + 1)), phi_of(i)
        return add(("mh", i, j), (r * math.cos(phi), r * math.sin(phi)))

    def mid_angle(i: int, j: int) -> int:
        """Mid-side node between corners ``(i, j)`` and ``(i+1, j)``."""
        r, phi = r_of(j), 0.5 * (phi_of(i) + phi_of(i + 1))
        return add(("mv", i, j), (r * math.cos(phi), r * math.sin(phi)))

    elements = []
    for i in range(n_arc):
        for j in range(n_thick):
            elements.append(
                [
                    corner(i, j),
                    corner(i, j + 1),
                    corner(i + 1, j + 1),
                    corner(i + 1, j),
                    mid_radius(i, j),
                    mid_angle(i, j + 1),
                    mid_radius(i + 1, j),
                    mid_angle(i, j),
                ]
            )

    nodes = np.array(points, dtype=float)
    # Root (phi = pi/2) and free end (phi = 0) node sets, selected geometrically
    # so the selection cannot silently drift with the node numbering.
    tol = 1e-12 * max(1.0, radius)
    root_mask = np.abs(nodes[:, 0]) <= tol
    free_mask = np.abs(nodes[:, 1]) <= tol
    centroid = int(np.argmin(np.hypot(nodes[:, 0] - radius, nodes[:, 1])))
    return RingMesh(
        nodes=nodes,
        elements=np.array(elements, dtype=int),
        n_arc=int(n_arc),
        n_thick=int(n_thick),
        radius=float(radius),
        thickness=float(thickness),
        root_nodes=np.flatnonzero(root_mask),
        free_end_nodes=np.flatnonzero(free_mask),
        centroid_node=centroid,
    )


def _plane_stress_matrix(young: float, poisson: float) -> np.ndarray:
    if not (young > 0.0) or not (-1.0 < poisson < 0.5):
        raise ValueError("need young > 0 and -1 < poisson < 0.5")
    return (
        young
        / (1.0 - poisson * poisson)
        * np.array(
            [[1.0, poisson, 0.0], [poisson, 1.0, 0.0], [0.0, 0.0, 0.5 * (1.0 - poisson)]]
        )
    )


def _strain_matrix(xi: float, eta: float, coordinates: np.ndarray) -> tuple[np.ndarray, float]:
    _, derivative = q8_shape(xi, eta)
    jacobian = derivative.T @ coordinates
    determinant = float(np.linalg.det(jacobian))
    if determinant <= 0.0:
        raise ValueError("inverted Q8 element (non-positive Jacobian)")
    global_derivative = derivative @ np.linalg.inv(jacobian).T
    matrix = np.zeros((3, 16))
    matrix[0, 0::2] = global_derivative[:, 0]
    matrix[1, 1::2] = global_derivative[:, 1]
    matrix[2, 0::2] = global_derivative[:, 1]
    matrix[2, 1::2] = global_derivative[:, 0]
    return matrix, determinant


def _element_stiffness(
    coordinates: np.ndarray, constitutive: np.ndarray, width: float
) -> np.ndarray:
    stiffness = np.zeros((16, 16))
    for xi, eta, weight in Q8_GAUSS:
        matrix, determinant = _strain_matrix(xi, eta, coordinates)
        stiffness += matrix.T @ constitutive @ matrix * determinant * weight * width
    return stiffness


@dataclass(frozen=True)
class RingSolution:
    """Solved plane-stress quarter ring (load case and its resultants)."""

    mesh: RingMesh
    displacement: np.ndarray
    reaction: np.ndarray
    stiffness: np.ndarray
    dof_forces: np.ndarray
    width: float
    young: float
    poisson: float
    load: float
    direction: str

    @property
    def tip_tangent(self) -> float:
        """Free-end displacement along the loaded (tangent) direction, ``+y``."""
        node = self.mesh.centroid_node
        return float(self.displacement[2 * node + 1])

    @property
    def tip_radial(self) -> float:
        """Free-end radial displacement at the mid-surface node, ``+x``."""
        node = self.mesh.centroid_node
        return float(self.displacement[2 * node])

    @property
    def reaction_force(self) -> tuple[float, float]:
        forces = self.reaction.reshape(-1, 2)
        return float(forces[:, 0].sum()), float(forces[:, 1].sum())

    @property
    def reaction_moment(self) -> float:
        """Moment of the clamping reactions about the ring centre."""
        forces = self.reaction.reshape(-1, 2)
        x, y = self.mesh.nodes[:, 0], self.mesh.nodes[:, 1]
        return float((x * forces[:, 1] - y * forces[:, 0]).sum())

    def equilibrium_error(self) -> float:
        """Relative imbalance of the reaction against the applied load."""
        fx, fy = self.reaction_force
        applied = (0.0, self.load) if self.direction == "tangential" else (self.load, 0.0)
        return math.hypot(fx + applied[0], fy + applied[1]) / self.load

    def moment_error(self) -> float:
        """Relative error of the reaction moment against the static value ``P R``."""
        return abs(abs(self.reaction_moment) - self.load * self.mesh.radius) / (
            self.load * self.mesh.radius
        )


def solve_quarter_ring(
    n_arc: int,
    n_thick: int,
    radius: float,
    thickness: float,
    width: float,
    young: float,
    poisson: float,
    load: float,
    *,
    direction: str = "tangential",
) -> RingSolution:
    """Solve the clamped quarter ring as a 2-D plane-stress continuum.

    The root cross-section is fully clamped, the free end carries the total load
    ``load`` shared equally by its nodes (a statically equivalent nodal load --
    the resulting local stress redistribution is Saint-Venant and therefore
    decays before the mid-arc section used for the stress checks).
    """
    if direction not in ("tangential", "radial"):
        raise ValueError("direction must be 'tangential' or 'radial'")
    if not (width > 0.0) or not (load > 0.0):
        raise ValueError("width and load must be positive")
    mesh = quarter_ring_q8_mesh(n_arc, n_thick, radius, thickness)
    constitutive = _plane_stress_matrix(young, poisson)

    stiffness = np.zeros((mesh.n_dofs, mesh.n_dofs))
    for element in mesh.elements:
        dofs = np.stack((2 * element, 2 * element + 1), axis=1).reshape(-1)
        stiffness[np.ix_(dofs, dofs)] += _element_stiffness(
            mesh.nodes[element], constitutive, width
        )

    forces = np.zeros(mesh.n_dofs)
    component = 1 if direction == "tangential" else 0
    forces[2 * mesh.free_end_nodes + component] += load / len(mesh.free_end_nodes)

    fixed = np.stack((2 * mesh.root_nodes, 2 * mesh.root_nodes + 1), axis=1).reshape(-1)
    mask = np.ones(mesh.n_dofs, dtype=bool)
    mask[fixed] = False
    free = np.flatnonzero(mask)
    displacement = np.zeros(mesh.n_dofs)
    displacement[free] = np.linalg.solve(
        stiffness[np.ix_(free, free)], forces[free]
    )
    reaction = stiffness @ displacement - forces
    return RingSolution(
        mesh=mesh,
        displacement=displacement,
        reaction=reaction,
        stiffness=stiffness,
        dof_forces=forces,
        width=float(width),
        young=float(young),
        poisson=float(poisson),
        load=float(load),
        direction=direction,
    )


def section_forces(radius: float, load: float, phi: float) -> tuple[float, float]:
    """Static hoop force and moment at the cut of angle ``phi`` (tangent load).

    Positive ``axial`` is tensile, positive ``moment`` produces tension at the
    inner fibre.  Independent of any finite-element result.
    """
    return -load * math.cos(phi), -load * radius * (1.0 - math.cos(phi))


def hoop_stress_section(
    solution: RingSolution,
    row: int,
    *,
    sample: tuple = (0.0, 0.0),
    fibres: float | None = None,
) -> dict:
    """Hoop stress through a mesh section.

    ``row`` is the arc index of the element row to sample; every element through
    the depth of that row is evaluated at the parent point ``sample`` (default
    the element centre, where the Q8 stress is most accurate).

    ``fibres`` (parent coordinate in ``0 < |xi| < 1``) switches to a
    surface-oriented sampling: the innermost element is evaluated at
    ``(-fibres, 0)`` and the outermost at ``(+fibres, 0)``, the interior
    elements at the centre.  In the Q8 parent frame ``xi`` runs *radially*
    (``xi = -1`` on the inner surface, ``+1`` on the outer), so this reports the
    stress at a small distance inside the free surface instead of at the element
    centre -- the quantity a strength check actually needs, and the only way a
    coarse mesh resolves the surface fibre.

    Returns the sampled radii, the section angle, the hoop stress and the
    element indices so a caller can reproduce the extraction.
    """
    mesh = solution.mesh
    if not 0 <= row < mesh.n_arc:
        raise ValueError("row outside the mesh")
    constitutive = _plane_stress_matrix(solution.young, solution.poisson)
    radii, stresses, angles, elements = [], [], [], []
    for j in range(mesh.n_thick):
        element = mesh.elements[row * mesh.n_thick + j]
        coordinates = mesh.nodes[element]
        if fibres is None:
            point = sample
        elif j == 0:
            point = (-fibres, sample[1])
        elif j == mesh.n_thick - 1:
            point = (fibres, sample[1])
        else:
            point = sample
        shape, _ = q8_shape(*point)
        centre = shape @ coordinates
        matrix, _ = _strain_matrix(*point, coordinates)
        dofs = np.stack((2 * element, 2 * element + 1), axis=1).reshape(-1)
        stress = constitutive @ (matrix @ solution.displacement[dofs])
        angle = math.atan2(centre[1], centre[0])
        hoop = (
            stress[0] * math.sin(angle) ** 2
            + stress[1] * math.cos(angle) ** 2
            - 2.0 * stress[2] * math.sin(angle) * math.cos(angle)
        )
        radii.append(float(math.hypot(centre[0], centre[1])))
        stresses.append(float(hoop))
        angles.append(angle)
        elements.append(int(row * mesh.n_thick + j))
    order = np.argsort(radii)
    return {
        "radii": np.array(radii)[order],
        "hoop_stress": np.array(stresses)[order],
        "angle": float(np.mean(angles)),
        "elements": [elements[i] for i in order],
    }


def winkler_bach_stress(
    radii: np.ndarray,
    radius: float,
    thickness: float,
    width: float,
    axial: float,
    moment: float,
) -> np.ndarray:
    """Curved-beam (Winkler--Bach) hoop stress for a rectangular radial section.

    ``sigma_theta = N/A + M (r - r_n) / (A e r)`` with the neutral-axis radius
    ``r_n = A / int dA/r = t / ln(b/a)`` and ``e = R_c - r_n``.  The ``1/r``
    weighting is what a flat-beam formula cannot reproduce; it raises the inner
    fibre stress above the straight-beam value as ``t/R`` grows.
    """
    radii = np.asarray(radii, dtype=float)
    inner = radius - 0.5 * thickness
    outer = radius + 0.5 * thickness
    area = width * thickness
    neutral = thickness / math.log(outer / inner)
    eccentric = radius - neutral
    axial_stress = axial / area
    return axial_stress + moment * (radii - neutral) / (area * eccentric * radii)


def straight_beam_stress(
    radii: np.ndarray,
    radius: float,
    thickness: float,
    width: float,
    axial: float,
    moment: float,
) -> np.ndarray:
    """Flat-beam hoop stress ``N/A + M (r - R_c) / I`` used as the naive baseline."""
    radii = np.asarray(radii, dtype=float)
    area = width * thickness
    inertia = width * thickness**3 / 12.0
    return axial / area + moment * (radii - radius) / inertia


def richardson_limit(
    values: list[float] | tuple[float, ...], refinement_ratio: float = 2.0
) -> dict:
    """Observed order and Richardson limit from three successive approximations.

    Requires a monotone, same-sign error sequence; returns ``nan`` otherwise
    rather than inventing an order.
    """
    if len(values) != 3:
        raise ValueError("richardson_limit needs exactly three levels")
    coarse, medium, fine = (float(v) for v in values)
    d_21, d_32 = medium - coarse, fine - medium
    if d_21 == 0.0 or d_32 == 0.0 or d_21 / d_32 <= 0.0:
        return {"observed_order": float("nan"), "limit": float("nan")}
    order = math.log(abs(d_21 / d_32)) / math.log(refinement_ratio)
    if not math.isfinite(order) or order <= 0.0:
        return {"observed_order": float("nan"), "limit": float("nan")}
    limit = fine + (fine - medium) / (refinement_ratio**order - 1.0)
    return {"observed_order": order, "limit": limit}