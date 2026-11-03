"""Quarter-ring curved-cantilever benchmark with a closed-form reference.

A quarter of a circular ring of radius ``R`` and rectangular section
``width x thickness`` is clamped at one end and loaded at the free end.  The
tip tangent is perpendicular to the root tangent, so the problem couples
membrane (hoop) action with bending and exercises exactly the curved-shell
mechanism that flat-facet shells are known to distort.  Unlike the classical
Scordelis--Lo roof the response is bending dominated and has a closed form, so
an element can be *verified* rather than merely compared.

Reference (Castigliano on the thin, inextensible-section curved bar; axial and
transverse-shear terms retained):

``M(phi) = -P R (1 - cos phi)`` for a load tangent to the free end, with
``phi = 0`` at the free end and ``phi = pi/2`` at the clamp.  Hence

* tangent load -> tip displacement along the tangent
  ``delta_t = P R^3 (3 pi/4 - 2) / (E I) + pi P R /(4 E A) + pi P R /(4 k G A)``
* the *transverse* (radial) tip displacement under the same tangent load is
  ``delta_r = P R^3 / (2 E I)`` (Maxwell--Betti reciprocity with the radial
  load case, whose own direct response is ``pi P R^3 / (4 E I)``).

The three coefficients ``3 pi/4 - 2 = 0.356194``, ``1/2`` and ``pi/4`` were
cross-checked against an independent polygonal Euler--Bernoulli frame model,
against the flat-facet :func:`tensorfem.shell4.shell4_stiffness` and against the
metric-consistent
:func:`tensorfem.cylindrical_shell4.cylindrical_shell4_stiffness`; all three
numerical routes agree to better than 0.5%.

Two models are provided:

``element="flat"``
    :func:`tensorfem.shell4.shell4_stiffness` on a faceted ring whose axis is
    the global ``z`` axis (radial = ``x`` at the free end).
``element="cylindrical"``
    :func:`tensorfem.cylindrical_shell4.cylindrical_shell4_stiffness`, whose
    exact cylindrical metric requires the ring axis to be the global ``x`` axis
    (radial = ``z`` at the free end).

Both models are clamped over all six degrees of freedom on the root edge; the
free-end edge carries an equal share of the total tangent load.  The two
meshes are mirror images of one another, because the cylindrical element measures
``theta`` with the opposite sense to the flat mesh (its exact-metric
construction requires that), so the load acts along ``-tangent`` in the flat
mesh and along ``+tangent`` in the cylindrical mesh for the same physical
configuration.  Reported free-end components and reactions are therefore
magnitudes along the declared tangent/normal axes; every verification gate
compares magnitudes, and the reciprocity test builds its own radial load case
instead of relying on a cross component.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .cylindrical_shell4 import cylindrical_shell4_stiffness
from .shell4 import shell4_stiffness

__all__ = [
    "RingBenchmark",
    "quarter_ring_reference",
    "quarter_ring_shell",
    "ring_qualification_table",
]

#: ``delta_t * E I / (P R^3)`` for a tangent load.
TANGENT_COEFFICIENT = 3.0 * math.pi / 4.0 - 2.0
#: ``delta_r * E I / (P R^3)``, the radial tip displacement under a tangent load.
RADIAL_COEFFICIENT = 0.5
#: ``delta_x * E I / (P R^3)``, direct response to a radial end load.
RADIAL_LOAD_COEFFICIENT = math.pi / 4.0


@dataclass(frozen=True)
class RingBenchmark:
    """Solved quarter-ring model with its analytic reference."""

    element: str
    n_arc: int
    n_width: int
    tip_tangent: float
    tip_radial: float
    tip_tangent_exact: float
    tip_radial_exact: float
    reaction_tangent: float
    reaction_moment: float
    load: float
    radius: float
    nodes: torch.Tensor
    elements: torch.Tensor

    @property
    def tangent_error(self) -> float:
        return abs(self.tip_tangent / self.tip_tangent_exact - 1.0)

    @property
    def radial_error(self) -> float:
        return abs(self.tip_radial / self.tip_radial_exact - 1.0)

    def summary(self) -> dict[str, object]:
        return {
            "element": self.element,
            "n_arc": self.n_arc,
            "n_width": self.n_width,
            "tip_tangent": self.tip_tangent,
            "tip_tangent_exact": self.tip_tangent_exact,
            "tangent_error": self.tangent_error,
            "tip_radial": self.tip_radial,
            "tip_radial_exact": self.tip_radial_exact,
            "radial_error": self.radial_error,
            "reaction_tangent": self.reaction_tangent,
            "reaction_moment": self.reaction_moment,
            "load": self.load,
            "radius": self.radius,
        }


def quarter_ring_reference(
    radius: float,
    width: float,
    thickness: float,
    young: float,
    poisson: float,
    load: float,
    *,
    shear_factor: float = 5.0 / 6.0,
) -> dict[str, float]:
    """Closed-form free-end displacement of a clamped quarter ring.

    Returns ``delta_tangent`` (along the free-end tangent), ``delta_radial``
    (radial component under the same tangent load) and ``delta_radial_load``
    (direct response to a radial free-end load), all positive downward.
    """
    for name, value in (
        ("radius", radius),
        ("width", width),
        ("thickness", thickness),
        ("young", young),
        ("load", load),
        ("shear_factor", shear_factor),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    if not (-1.0 < poisson < 0.5):
        raise ValueError("poisson must lie in (-1, 0.5)")
    area = width * thickness
    inertia = width * thickness**3 / 12.0
    shear_modulus = young / (2.0 * (1.0 + poisson))
    bending = load * radius**3 / (young * inertia)
    axial = math.pi * load * radius / (4.0 * young * area)
    shear = math.pi * load * radius / (4.0 * shear_factor * shear_modulus * area)
    return {
        "delta_tangent": TANGENT_COEFFICIENT * bending + axial + shear,
        "delta_radial": RADIAL_COEFFICIENT * bending,
        "delta_radial_load": RADIAL_LOAD_COEFFICIENT * bending,
        "bending_scale": bending,
        "axial_term": axial,
        "shear_term": shear,
    }


def _flat_ring_mesh(n_arc: int, n_width: int, radius: float, width: float, dtype, device):
    nodes: list[tuple[float, float, float]] = []
    for i in range(n_arc + 1):
        phi = 0.5 * math.pi * i / n_arc
        for j in range(n_width + 1):
            nodes.append(
                (
                    radius * math.cos(phi),
                    radius * math.sin(phi),
                    -0.5 * width + width * j / n_width,
                )
            )
    elements = []
    for i in range(n_arc):
        for j in range(n_width):
            s = i * (n_width + 1) + j
            elements.append((s, s + n_width + 1, s + n_width + 2, s + 1))
    return (
        torch.tensor(nodes, dtype=dtype, device=device),
        torch.tensor(elements, dtype=torch.long, device=device),
    )


def _cylindrical_ring_mesh(n_arc: int, n_width: int, radius: float, width: float, dtype, device):
    nodes: list[tuple[float, float, float]] = []
    for i in range(n_arc + 1):
        theta = 0.5 * math.pi * i / n_arc
        for j in range(n_width + 1):
            nodes.append(
                (
                    -0.5 * width + width * j / n_width,
                    radius * math.sin(theta),
                    radius * math.cos(theta),
                )
            )
    elements = []
    for i in range(n_arc):
        for j in range(n_width):
            s = i * (n_width + 1) + j
            elements.append((s, s + n_width + 1, s + n_width + 2, s + 1))
    return (
        torch.tensor(nodes, dtype=dtype, device=device),
        torch.tensor(elements, dtype=torch.long, device=device),
    )


def quarter_ring_shell(
    n_arc: int,
    n_width: int,
    *,
    radius: float = 10.0,
    width: float = 0.2,
    thickness: float = 0.1,
    young: float = 2.1e5,
    poisson: float = 0.3,
    load: float = 1.0,
    element: str = "flat",
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> RingBenchmark:
    """Solve the clamped quarter ring with one of the two shell elements."""
    if element not in ("flat", "cylindrical"):
        raise ValueError("element must be 'flat' or 'cylindrical'")
    if n_arc < 1 or n_width < 1:
        raise ValueError("n_arc and n_width must be positive")
    dev = torch.device(device or "cpu")
    if element == "flat":
        nodes, elements = _flat_ring_mesh(n_arc, n_width, radius, width, dtype, dev)
        tangent = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=dev)
        radial = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=dev)
    else:
        nodes, elements = _cylindrical_ring_mesh(n_arc, n_width, radius, width, dtype, dev)
        tangent = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=dev)
        radial = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=dev)
    ndof = 6 * nodes.shape[0]
    stiffness = torch.zeros((ndof, ndof), dtype=dtype, device=dev)
    force = torch.zeros(ndof, dtype=dtype, device=dev)
    for conn in elements:
        xyz = nodes[conn]
        if element == "flat":
            ke = shell4_stiffness(xyz, young, poisson, thickness)
        else:
            theta = torch.atan2(xyz[:, 1], xyz[:, 2])
            theta = torch.where(theta < 0, theta + 2 * math.pi, theta)
            x_theta = torch.stack((xyz[:, 0], theta), dim=1)
            ke = cylindrical_shell4_stiffness(x_theta, radius, young, poisson, thickness)
        dofs = torch.stack(tuple(6 * conn + k for k in range(6)), 1).reshape(-1)
        stiffness[dofs[:, None], dofs] += ke
    tip = torch.arange(0, n_width + 1, device=dev)  # arc index 0 -> theta = 0
    root = n_arc * (n_width + 1) + torch.arange(0, n_width + 1, device=dev)
    share = load / (n_width + 1)
    applied = tangent * share
    for node in tip:
        base = 6 * int(node)
        force[base : base + 3] -= applied
    fixed = torch.tensor(
        sorted({6 * int(node) + k for node in root for k in range(6)}),
        dtype=torch.long,
        device=dev,
    )
    free_mask = torch.ones(ndof, dtype=torch.bool, device=dev)
    free_mask[fixed] = False
    free = torch.nonzero(free_mask).flatten()
    displacement = torch.zeros(ndof, dtype=dtype, device=dev)
    displacement[free] = torch.linalg.solve(
        stiffness[free[:, None], free], force[free]
    )
    reaction = stiffness @ displacement - force
    tip_mean = displacement.reshape(-1, 6)[tip].mean(dim=0)
    reference = quarter_ring_reference(radius, width, thickness, young, poisson, load)
    root_reaction = reaction.reshape(-1, 6)[root].sum(dim=0)
    lever = nodes[tip].mean(dim=0) - nodes[root].mean(dim=0)
    root_moment = torch.linalg.vector_norm(torch.linalg.cross(lever, tangent * load))
    return RingBenchmark(
        element=element,
        n_arc=n_arc,
        n_width=n_width,
        tip_tangent=float((tip_mean[:3] @ tangent).abs()),
        tip_radial=float((tip_mean[:3] @ radial).abs()),
        tip_tangent_exact=reference["delta_tangent"],
        tip_radial_exact=reference["delta_radial"],
        reaction_tangent=float((root_reaction[:3] @ tangent).abs()),
        reaction_moment=float(root_moment),
        load=load,
        radius=radius,
        nodes=nodes,
        elements=elements,
    )


def ring_qualification_table(
    meshes: tuple[int, ...] = (4, 8, 16, 32, 64),
    *,
    elements: tuple[str, ...] = ("flat", "cylindrical"),
    n_width: int = 2,
    **kwargs,
) -> dict[str, object]:
    """Convergence table for both shell elements against the closed form."""
    rows = []
    for element in elements:
        for n_arc in meshes:
            result = quarter_ring_shell(n_arc, n_width, element=element, **kwargs)
            rows.append(result.summary())
    return {
        "benchmark": "quarter_ring_curved_cantilever",
        "reference": "delta_t=(3pi/4-2)PR^3/EI, delta_r=PR^3/(2EI)",
        "coefficients": {
            "tangent": TANGENT_COEFFICIENT,
            "radial": RADIAL_COEFFICIENT,
            "radial_load": RADIAL_LOAD_COEFFICIENT,
        },
        "n_width": n_width,
        "rows": rows,
    }