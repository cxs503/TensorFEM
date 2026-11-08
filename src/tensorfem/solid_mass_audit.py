"""Quantitative audit of the solid mass matrices against the exact rod solution.

Purpose
-------
``solid_mass`` adds TET4/HEX8 mass matrices to the library.  A mass matrix is
easy to get subtly wrong (transposed Jacobian, missing density, wrong 3-DOF
scatter), so this module pins it with two *independent* kinds of evidence:

1. **Algebraic invariants** (``mass_invariant_audit``): total mass equals
   ``rho V`` and the quadratic form of the rigid rotation field equals
   ``rho * integral (x^2 + y^2) dV`` -- both to round-off for the consistent
   matrices and for a deliberately distorted element.
2. **Physical benchmark** (``audit_axial_rod``): a mesh of real 3-D solids with
   lateral displacements suppressed is exactly a one-dimensional rod, whose
   fixed--free axial frequencies are
   ``omega_n = (2n-1) pi / (2L) * sqrt(M_c / rho)``.  The audit reports the
   relative error per mode and mesh, the observed convergence order and the
   finest-mesh error, so ``<3%`` can be answered numerically rather than by
   assertion.

Row-sum lumping is measured too: it conserves total mass exactly while
inflating rotational inertia by a fixed amount, and it changes the frequency
error sign and order.  Those differences are reported, not smoothed away.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .modal import solve_modes
from .solid3d import SolidModel, hex_to_tet_mesh, solve_solid, structured_hex_mesh
from .solid_mass import (
    assemble_solid_mass,
    confined_modulus,
    mass_properties,
    rod_axial_angular_frequency,
)


@dataclass(frozen=True)
class RodCase:
    """One axial-rod eigen-solve on a solid mesh."""

    element_type: str
    mass_form: str
    elements: int
    mesh_size: float
    frequencies: tuple[float, ...]
    exact: tuple[float, ...]

    @property
    def relative_errors(self) -> tuple[float, ...]:
        return tuple(abs(w / e - 1.0) for w, e in zip(self.frequencies, self.exact))


@dataclass(frozen=True)
class RodAudit:
    """Convergence audit over a mesh sequence for one (element, mass) pair."""

    element_type: str
    mass_form: str
    cases: tuple[RodCase, ...]
    observed_order: tuple[float, ...]

    @property
    def finest(self) -> RodCase:
        return self.cases[-1]

    @property
    def finest_max_error(self) -> float:
        return max(self.finest.relative_errors)


def _solid_stiffness(
    nodes: torch.Tensor,
    elements: torch.Tensor,
    young: float,
    poisson: float,
    element_type: str = "hex8",
) -> torch.Tensor:
    """Assemble the elastic stiffness with the ``solid3d`` element kernels.

    The assembly mirrors ``solve_solid`` but skips its static solve: the audit
    only needs ``K`` and must not depend on a well-posed boundary-value problem.
    """
    from .solid3d import elasticity_matrix_3d, hex8_stiffness, tet4_stiffness
    from .solid_mass import _dof_map

    ne, nn = elements.shape
    x = nodes[elements.long()]
    e = torch.full((ne,), float(young), dtype=nodes.dtype, device=nodes.device)
    nu = torch.full((ne,), float(poisson), dtype=nodes.dtype, device=nodes.device)
    d = elasticity_matrix_3d(e, nu)
    kind = element_type.lower()
    if kind == "tet4":
        element = tet4_stiffness(x, d)[0]
    elif kind == "tet10":
        from .quadratic_solid import tet10_stiffness

        element = tet10_stiffness(x, d)[0]
    else:
        element = hex8_stiffness(x, d)[0]
    dofs = _dof_map(elements)
    rows = dofs[:, :, None].expand(-1, -1, dofs.shape[1]).reshape(-1)
    cols = dofs[:, None, :].expand(-1, dofs.shape[1], -1).reshape(-1)
    ndofs = 3 * len(nodes)
    stiffness = torch.zeros((ndofs, ndofs), dtype=nodes.dtype, device=nodes.device)
    return stiffness.index_put((rows, cols), element.reshape(-1), accumulate=True)


def axial_rod_case(
    elements: int,
    *,
    element_type: str = "hex8",
    mass_form: str = "consistent",
    modes: int = 3,
    length: float = 10.0,
    width: float = 0.2,
    height: float = 0.2,
    young: float = 210.0e9,
    poisson: float = 0.3,
    density: float = 7850.0,
) -> RodCase:
    """Solve the laterally constrained rod with real 3-D solids.

    The bar occupies ``0 <= x <= length`` with ``ux`` free, ``uy = uz = 0``
    everywhere and ``ux = 0`` at ``x = 0``.  That constrained 3-D problem is the
    uniaxial-strain rod, whose wave speed is ``sqrt(M_c/rho)`` with the confined
    modulus ``M_c`` (``= E`` when ``poisson = 0``).
    """
    if int(elements) < 1:
        raise ValueError("elements must be positive")
    if mass_form not in {"consistent", "lumped"}:
        raise ValueError("mass_form must be 'consistent' or 'lumped'")
    kind = element_type.lower()
    nodes, hexes = structured_hex_mesh(length, width, height, int(elements), 1, 1)
    if kind == "hex8":
        connectivity = hexes
    elif kind == "tet4":
        connectivity = hex_to_tet_mesh(nodes, hexes)[1]
    elif kind == "tet10":
        from .quadratic_solid import tet4_to_tet10

        nodes, connectivity = tet4_to_tet10(nodes, hex_to_tet_mesh(nodes, hexes)[1])
    else:
        raise ValueError("element_type must be 'hex8', 'tet4' or 'tet10'")
    stiffness = _solid_stiffness(nodes, connectivity, young, poisson, kind)
    mass = assemble_solid_mass(
        nodes, connectivity, density, element_type=kind, lumped=mass_form == "lumped"
    )
    axial_free = [3 * i for i in range(len(nodes)) if float(nodes[i, 0]) > 0.0]
    if len(axial_free) < modes:
        raise ValueError("mesh is too coarse for the requested number of modes")
    keep = set(axial_free)
    constrained = [dof for dof in range(3 * len(nodes)) if dof not in keep]
    result = solve_modes(stiffness, mass, constrained, modes=modes)
    frequencies = tuple(float(w) for w in result.angular_frequencies)
    modulus = confined_modulus(young, poisson)
    exact = tuple(
        rod_axial_angular_frequency(n + 1, length, modulus, density) for n in range(modes)
    )
    return RodCase(kind, mass_form, int(elements), length / int(elements), frequencies, exact)


def _observed_order(cases: tuple[RodCase, ...], mode: int) -> float:
    """Least-squares slope of ``log(error)`` against ``log(h)``, last 3 meshes."""
    tail = cases[-3:]
    xs = [math.log(case.mesh_size) for case in tail]
    ys = [math.log(max(case.relative_errors[mode], 1e-16)) for case in tail]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    denominator = sum((x - mx) ** 2 for x in xs)
    if denominator == 0.0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denominator


def audit_axial_rod(
    *,
    element_type: str = "hex8",
    mass_form: str = "consistent",
    meshes: tuple[int, ...] = (4, 8, 16, 32),
    modes: int = 3,
    **kwargs,
) -> RodAudit:
    """Run the axial-rod convergence audit over a mesh sequence."""
    cases = tuple(
        axial_rod_case(n, element_type=element_type, mass_form=mass_form, modes=modes, **kwargs)
        for n in meshes
    )
    order = tuple(_observed_order(cases, mode) for mode in range(modes))
    return RodAudit(element_type.lower(), mass_form, cases, order)


@dataclass(frozen=True)
class InvariantCase:
    element_type: str
    mass_form: str
    total_mass: float
    exact_mass: float
    polar_inertia: float
    exact_polar_inertia: float

    @property
    def mass_error(self) -> float:
        return abs(self.total_mass / self.exact_mass - 1.0)

    @property
    def polar_inertia_error(self) -> float:
        return abs(self.polar_inertia / self.exact_polar_inertia - 1.0)


def mass_invariant_audit(
    *,
    length: float = 2.0,
    width: float = 1.0,
    height: float = 0.8,
    density: float = 7850.0,
    distortion: float = 0.0,
) -> tuple[InvariantCase, ...]:
    """Total-mass and polar-inertia invariants for both elements and mass forms.

    ``distortion`` displaces one interior node so the elements are no longer
    rectangular bricks (the quadrature must still reproduce the exact integrals).
    """
    nodes, hexes = structured_hex_mesh(length, width, height, 2, 2, 2)
    if distortion:
        interior = 1 + 3 * (1 + 3 * 1)  # node (i,j,k) = (1,1,1) of a 3x3x3 grid
        nodes[interior] = nodes[interior] + torch.tensor(
            [distortion, -0.5 * distortion, 0.7 * distortion], dtype=nodes.dtype
        )
    tets = hex_to_tet_mesh(nodes, hexes)[1]
    out = []
    for kind, connectivity in (("hex8", hexes), ("tet4", tets)):
        for form in ("consistent", "lumped"):
            props = mass_properties(
                nodes, connectivity, density, element_type=kind, lumped=form == "lumped"
            )
            out.append(
                InvariantCase(
                    kind,
                    form,
                    props.total_mass,
                    props.exact_mass,
                    props.polar_inertia,
                    props.exact_polar_inertia,
                )
            )
    return tuple(out)


def summary() -> dict:
    """Compact JSON-serialisable audit summary (used by the report script)."""
    audits = [
        audit_axial_rod(element_type=kind, mass_form=form)
        for kind in ("hex8", "tet4")
        for form in ("consistent", "lumped")
    ]
    return {
        "rod": [
            {
                "element_type": audit.element_type,
                "mass_form": audit.mass_form,
                "observed_order": list(audit.observed_order),
                "finest_max_error": audit.finest_max_error,
                "cases": [
                    {
                        "elements": case.elements,
                        "mesh_size": case.mesh_size,
                        "frequencies": list(case.frequencies),
                        "exact": list(case.exact),
                        "relative_errors": list(case.relative_errors),
                    }
                    for case in audit.cases
                ],
            }
            for audit in audits
        ],
        "invariants": [
            {
                "element_type": case.element_type,
                "mass_form": case.mass_form,
                "mass_error": case.mass_error,
                "polar_inertia_error": case.polar_inertia_error,
            }
            for case in mass_invariant_audit()
        ],
    }