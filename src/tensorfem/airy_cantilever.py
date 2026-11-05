"""Exact plane-stress elasticity solution for a shear-loaded cantilever.

Timoshenko & Goodier, *Theory of Elasticity*, Art. 24 (bending of a cantilever
by an end load) gives a polynomial Airy stress function for the rectangle
``0 <= x <= L``, ``-c <= y <= c`` carrying a parabolic end shear ``P`` at
``x = L``::

    sigma_x = -P (L - x) y / I
    sigma_y =  0
    tau_xy  =  P (c^2 - y^2) / (2 I)

with ``I = t H^3 / 12`` the section second moment (``c = H/2``, thickness ``t``).
The field satisfies equilibrium, compatibility and traction-free long edges;
the resultant of ``tau_xy`` over the end section is exactly ``P``.

**Why this module exists.**  The repository's Q4 bending benchmark
(``rectangular_benchmarks``) validates against *Euler--Bernoulli beam theory*,
which is itself an approximation: it drops transverse shear deformation and
assumes plane sections.  This module supplies a *pointwise exact* elasticity
reference for the same geometry, so the finite-element discretisation error can
be separated from the beam-theory model error.  Displacements follow from
integrating the strains::

    u_x = -P y (L x - x^2/2)/(E I) - P y^3 (1 + nu/2)/(3 E I) + a0
    u_y =  nu P (L - x) y^2/(2 E I) + P (L x^2/2 - x^3/6)/(E I)
          + (1 + nu) P c^2 x/(E I) + b0

The rigid-body constants are chosen as ``a0 = b0 = 0``, so the solution is
pinned (``u = 0``) at the origin.  Stresses are independent of ``E`` and ``nu``
and therefore provide a material-independent verification target.

Nothing here discretises anything: the module is pure closed form, and it
deliberately does not import the finite-element assembly code, so a bug shared
between the solver and its reference cannot hide.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

D = torch.float64

#: 3-point Gauss rule on ``[-1, 1]``; exact for polynomials up to degree 5,
#: which covers the quadratic shear and quartic energy integrands exactly.
_GAUSS = ((-(3.0 / 5.0) ** 0.5, 5.0 / 9.0), (0.0, 8.0 / 9.0), ((3.0 / 5.0) ** 0.5, 5.0 / 9.0))


@dataclass(frozen=True)
class AiryCantilever:
    """Rectangular cantilever loaded by an exact parabolic end shear."""

    length: float = 1.0
    height: float = 0.1
    thickness: float = 0.012
    young_modulus: float = 210e9
    poisson: float = 0.3
    end_shear: float = -100.0

    def __post_init__(self):
        if self.length <= 0 or self.height <= 0 or self.thickness <= 0:
            raise ValueError("cantilever dimensions must be positive")
        if self.young_modulus <= 0 or not -1.0 < self.poisson < 0.5:
            raise ValueError("invalid elastic constants")
        if self.end_shear == 0:
            raise ValueError("end shear must be non-zero")

    @property
    def half_height(self) -> float:
        return self.height / 2

    @property
    def second_moment(self) -> float:
        return self.thickness * self.height**3 / 12

    def _points(self, points) -> torch.Tensor:
        p = torch.as_tensor(points, dtype=D)
        if p.ndim == 1:
            p = p.reshape(-1, 2)
        if p.shape[-1] != 2:
            raise ValueError("points must have shape (n, 2)")
        return p

    def stress(self, points) -> torch.Tensor:
        """Return ``(sigma_x, sigma_y, tau_xy)`` at ``points``."""
        p = self._points(points)
        x, y = p[:, 0], p[:, 1]
        c, inertia, load = self.half_height, self.second_moment, self.end_shear
        sigma_x = -load * (self.length - x) * y / inertia
        sigma_y = torch.zeros_like(sigma_x)
        tau_xy = load * (c**2 - y**2) / (2 * inertia)
        return torch.stack((sigma_x, sigma_y, tau_xy), 1)

    def displacement(self, points) -> torch.Tensor:
        """Return ``(u_x, u_y)`` at ``points`` (plane stress)."""
        p = self._points(points)
        x, y = p[:, 0], p[:, 1]
        c, inertia, load = self.half_height, self.second_moment, self.end_shear
        e, nu = self.young_modulus, self.poisson
        ux = (-load * y * (self.length * x - x**2 / 2) / (e * inertia)
              - load * y**3 * (1 + nu / 2) / (3 * e * inertia))
        uy = (nu * load * (self.length - x) * y**2 / (2 * e * inertia)
              + load * (self.length * x**2 / 2 - x**3 / 6) / (e * inertia)
              + (1 + nu) * load * c**2 * x / (e * inertia))
        return torch.stack((ux, uy), 1)

    def strain(self, points) -> torch.Tensor:
        """Return engineering strains ``(eps_x, eps_y, gamma_xy)``."""
        s = self.stress(points)
        e, nu = self.young_modulus, self.poisson
        eps_x = (s[:, 0] - nu * s[:, 1]) / e
        eps_y = (s[:, 1] - nu * s[:, 0]) / e
        gamma = 2 * (1 + nu) * s[:, 2] / e
        return torch.stack((eps_x, eps_y, gamma), 1)

    def traction(self, points, normals) -> torch.Tensor:
        """Traction ``sigma . n`` (true stress, i.e. per unit area)."""
        s = self.stress(points)
        n = torch.as_tensor(normals, dtype=D)
        if n.ndim == 1:
            n = n.reshape(-1, 2)
        if n.shape[0] == 1 and s.shape[0] > 1:
            n = n.expand(s.shape[0], 2)
        tx = s[:, 0] * n[:, 0] + s[:, 2] * n[:, 1]
        ty = s[:, 2] * n[:, 0] + s[:, 1] * n[:, 1]
        return torch.stack((tx, ty), 1)

    def strain_energy(self) -> float:
        """Exact strain energy of the rectangle (3x3 Gauss, polynomial exact).

        The quadrature is built in ``float64`` throughout: constructing the
        sample points with the torch default dtype would round them to
        ``float32`` first and lose ~1e-7 of the energy.
        """
        c = self.half_height
        points, weights = [], []
        for gx, wx in _GAUSS:
            for gy, wy in _GAUSS:
                points.append((self.length * (1 + gx) / 2, c * gy))
                weights.append(wx * wy * (self.length / 2) * c)
        p = torch.tensor(points, dtype=D)
        w = torch.tensor(weights, dtype=D)
        s = self.stress(p)
        e = self.strain(p)
        density = 0.5 * (s[:, 0] * e[:, 0] + s[:, 1] * e[:, 1] + s[:, 2] * e[:, 2])
        return float((density * w).sum()) * self.thickness

    def tip_deflection(self) -> float:
        """Transverse deflection of the end section centroid (x = L, y = 0)."""
        return float(self.displacement(torch.tensor([[self.length, 0.0]]))[0, 1])


def boundary_nodal_forces(nodes: torch.Tensor, elements: torch.Tensor,
                          model: AiryCantilever) -> torch.Tensor:
    """Consistent nodal forces generated by the *exact* boundary tractions.

    The two long edges carry the parabolic shear traction ``tau_xy`` and both
    end sections carry their exact ``sigma_x``/``tau_xy`` distribution, so the
    resulting load vector is fully prescribed polynomial traction data.  Edges
    are integrated with a 3-point Gauss rule, which is exact for the quadratic
    traction on a straight edge.
    """
    nodes = torch.as_tensor(nodes, dtype=D)
    elements = torch.as_tensor(elements).long()
    edges: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for element in elements.tolist():
        for i in range(len(element)):
            a, b = element[i], element[(i + 1) % len(element)]
            edges.setdefault((a, b), []).append((a, b))
    unseen = dict(edges)
    boundary: list[tuple[int, int]] = []
    for key in list(edges):
        a, b = key
        if key in unseen and (b, a) in unseen:      # interior edge, shared
            unseen.pop((b, a), None)
            unseen.pop((a, b), None)
            continue
        if key in unseen:
            boundary.append(key)
            unseen.pop(key, None)
    forces = torch.zeros_like(nodes)
    thickness = model.thickness
    for a, b in boundary:
        pa, pb = nodes[a], nodes[b]
        direction = pb - pa
        length = float(torch.linalg.norm(direction))
        if length <= 0:
            raise ValueError("degenerate boundary edge")
        # Counter-clockwise elements have outward normal (dy, -dx)/|d|.
        normal = torch.tensor([direction[1], -direction[0]], dtype=D) / length
        for xi, weight in _GAUSS:
            point = 0.5 * (1 - xi) * pa + 0.5 * (1 + xi) * pb
            traction = model.traction(point.reshape(1, 2), normal.reshape(1, 2))[0]
            shape = (0.5 * (1 - xi), 0.5 * (1 + xi))
            scale = thickness * (length / 2) * weight
            forces[a] += shape[0] * traction * scale
            forces[b] += shape[1] * traction * scale
    return forces


def traction_resultant(model: AiryCantilever) -> tuple[torch.Tensor, torch.Tensor]:
    """Net force and moment of the exact traction field (self-equilibrium).

    Closed-form integrals along the four edges: the two end sections share the
    shear resultant, and the bending couples cancel.  Returns ``(force, moment)``
    in ``(Fx, Fy, Mz)`` about the origin.
    """
    c, inertia, load = model.half_height, model.second_moment, model.end_shear
    depth = 2 * c
    # integral of tau_xy over the section, per unit thickness
    shear = load * (c**2 * depth - depth**3 / 12) / (2 * inertia)
    force = torch.tensor([0.0, model.thickness * shear, 0.0], dtype=D)
    # integral of sigma_x times (-y) over the end sections gives the coupling
    bending = load * model.length * model.thickness * c**3 / (3 * inertia)
    moment = torch.tensor([0.0, 0.0, load * model.length - bending], dtype=D)
    return force, moment