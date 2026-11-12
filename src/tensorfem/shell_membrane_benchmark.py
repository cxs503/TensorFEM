"""Membrane-dominated closed-form benchmark for the flat Shell4 facet family.

Every curved-shell benchmark in :mod:`tensorfem.shell_benchmarks` is a
*bending*-dominated problem, so a formulation can win them by getting its
bending/transverse-shear block right and still be wrong in membrane action.
This module supplies the complementary side of the matrix: a thin cylindrical
shell whose exact solution is a pure membrane state.

Half-cylinder cut on two symmetry planes
---------------------------------------
``theta in [0, pi]`` (the ``z = 0`` plane) and ``x in [0, L/2]`` (the ``x = 0``
plane) with ``u_z = 0`` on both generators and ``u_x = 0`` on the ``x = 0``
ring.  A single ``u_y`` anchor removes the remaining rigid translation.  All
supports are symmetry planes or a single tangential anchor, so **no support
restrains the radial expansion that is being measured** -- the usual reason a
pressurised-cylinder check disagrees with ``w = p R^2 / (E t)``.

Exact solution (thin shell, plane stress, free ends, ``sigma_z = 0``)::

    sigma_theta = p R / t
    eps_theta   = p R / (E t)      ->  u_r  = p R^2 / (E t)
    eps_x       = -nu p R / (E t)  ->  u_x(L/2) = -nu p R L / (2 E t)

The state is uniform along ``x``, so there is no edge boundary layer and the
only discretisation error is the faceting of the ring plus whatever the
transverse-shear/plate block of the facet contributes spuriously.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .shell4 import shell4_stiffness
from .shell4_mitc import shell4_mitc_stiffness

__all__ = [
    "MembraneCylinderResult",
    "pressurised_cylinder",
    "pressurised_cylinder_reference",
]


@dataclass(frozen=True)
class MembraneCylinderResult:
    """Auditable outcome of one pressurised-cylinder solve."""

    displacement: torch.Tensor
    nodes: torch.Tensor
    elements: torch.Tensor
    radius: float
    length: float
    thickness: float
    young: float
    poisson: float
    pressure: float
    mean_radial_displacement: float
    perimeter_radial_displacement: float
    radial_ripple: float
    rigid_translation: tuple[float, float]
    end_axial_displacement: float
    reaction_residual: float
    anchor_node: int

    @property
    def exact_radial_displacement(self) -> float:
        return pressurised_cylinder_reference(
            self.radius, self.thickness, self.young, self.pressure
        )["radial_displacement"]

    @property
    def radial_error(self) -> float:
        exact = self.exact_radial_displacement
        return self.mean_radial_displacement / exact - 1.0

    @property
    def exact_end_axial_displacement(self) -> float:
        exact = pressurised_cylinder_reference(
            self.radius,
            self.thickness,
            self.young,
            self.pressure,
            self.poisson,
            self.length,
        )
        return exact["end_axial_displacement"]

    @property
    def axial_error(self) -> float:
        return self.end_axial_displacement / self.exact_end_axial_displacement - 1.0

    @property
    def relative_ripple(self) -> float:
        """Facet-scale radial scatter divided by the exact membrane answer."""
        return self.radial_ripple / abs(self.exact_radial_displacement)

    @property
    def perimeter_error(self) -> float:
        return self.perimeter_radial_displacement / self.exact_radial_displacement - 1.0


def pressurised_cylinder_reference(
    radius: float,
    thickness: float,
    young: float,
    pressure: float,
    poisson: float | None = None,
    length: float | None = None,
) -> dict[str, float]:
    """Return the closed-form thin-shell membrane state."""
    for name, value in (
        ("radius", radius), ("thickness", thickness), ("young", young), ("pressure", pressure)
    ):
        if not value > 0:
            raise ValueError(f"{name} must be positive")
    eps_theta = pressure * radius / (young * thickness)
    out = {
        "hoop_stress": pressure * radius / thickness,
        "hoop_strain": eps_theta,
        "radial_displacement": radius * eps_theta,
    }
    if poisson is not None:
        out["axial_strain"] = -poisson * eps_theta
        if length is not None:
            out["end_axial_displacement"] = -poisson * eps_theta * length / 2.0
    return out


def _half_ring_mesh(
    nt: int, nx: int, radius: float, length: float, dtype: torch.dtype, device
) -> tuple[torch.Tensor, torch.Tensor]:
    theta = torch.linspace(0.0, math.pi, nt + 1, dtype=dtype, device=device)
    xs = torch.linspace(0.0, length / 2.0, nx + 1, dtype=dtype, device=device)
    theta_grid, x_grid = torch.meshgrid(theta, xs, indexing="ij")
    nodes = torch.stack(
        (
            x_grid.reshape(-1),
            radius * torch.cos(theta_grid).reshape(-1),
            radius * torch.sin(theta_grid).reshape(-1),
        ),
        -1,
    )
    cells = []
    for i in range(nt):
        for j in range(nx):
            a = i * (nx + 1) + j
            b = (i + 1) * (nx + 1) + j
            cells.append((a, a + 1, b + 1, b))
    return nodes, torch.tensor(cells, dtype=torch.long, device=device)


def pressurised_cylinder(
    nt: int = 16,
    nx: int = 2,
    *,
    radius: float = 100.0,
    length: float = 100.0,
    thickness: float = 1.0,
    young: float = 1000.0,
    poisson: float = 0.3,
    pressure: float = 1.0,
    formulation: str = "shell4",
    plate_shear: str = "sri",
    anchor: str = "theta0",
    dtype: torch.dtype = torch.float64,
    device=None,
) -> MembraneCylinderResult:
    """Solve the half-cylinder membrane state with flat Shell4 facets.

    ``anchor`` selects the single node used to remove the last rigid
    translation (``"theta0"`` = ``theta = 0``, ``x = 0``; ``"thetapi"`` =
    ``theta = pi``, ``x = 0``).  The two choices must agree: the membrane
    answer is uniform, so only a support artefact can make them differ.
    """
    if nt < 2 or nx < 1:
        raise ValueError("nt >= 2 and nx >= 1 are required")
    builder = {"shell4": shell4_stiffness, "shell4_mitc": shell4_mitc_stiffness}.get(
        formulation.lower()
    )
    if builder is None:
        raise ValueError(f"unknown formulation {formulation!r}")
    dev = torch.device(device or "cpu")
    nodes, elements = _half_ring_mesh(nt, nx, radius, length, dtype, dev)
    ndof = 6 * nodes.shape[0]
    K = torch.zeros((ndof, ndof), dtype=dtype, device=dev)
    f = torch.zeros(ndof, dtype=dtype, device=dev)
    for element in elements:
        xyz = nodes[element]
        ke = builder(xyz, young, poisson, thickness, plate_shear=plate_shear)
        dofs = torch.stack(tuple(6 * element + k for k in range(6)), 1).reshape(-1)
        K[dofs[:, None], dofs] += ke
        normal = torch.linalg.cross(xyz[1] - xyz[0], xyz[3] - xyz[0], dim=0)
        normal = normal / torch.linalg.vector_norm(normal)
        centroid = xyz.mean(0)
        radial = torch.stack(
            (centroid.new_tensor(0.0), centroid[1], centroid[2]), 0
        )
        radial = radial / torch.linalg.vector_norm(radial)
        if float(normal @ radial) < 0.0:
            normal = -normal
        area = 0.5 * torch.linalg.vector_norm(torch.linalg.cross(xyz[1] - xyz[0], xyz[2] - xyz[0]))
        area += 0.5 * torch.linalg.vector_norm(torch.linalg.cross(xyz[2] - xyz[0], xyz[3] - xyz[0]))
        for k in range(4):
            f[6 * element[k]: 6 * element[k] + 3] += pressure * area / 4.0 * normal
    anchor_i = {"theta0": 0, "thetapi": nt}[anchor]
    anchor_node = anchor_i * (nx + 1)
    fixed = [6 * anchor_node + 1]
    theta_edges = (0, nt)
    for i in theta_edges:
        row = torch.arange(i * (nx + 1), (i + 1) * (nx + 1), device=dev)
        fixed.extend((6 * row + 2).tolist())
    x0_row = torch.arange(0, (nt + 1) * (nx + 1), nx + 1, device=dev)
    fixed.extend((6 * x0_row).tolist())
    fixed = torch.tensor(sorted(set(fixed)), dtype=torch.long, device=dev)
    free_mask = torch.ones(ndof, dtype=torch.bool, device=dev)
    free_mask[fixed] = False
    free = torch.nonzero(free_mask).flatten()
    u = torch.zeros(ndof, dtype=dtype, device=dev)
    u[free] = torch.linalg.solve(K[free[:, None], free], f[free])
    residual = float((K @ u - f)[free].abs().max()) / float(f.abs().max())

    radial_disp, ripple, translation, perimeter_disp = _ring_metrics(
        nodes, u, elements, nt, nx, radius
    )
    end_row = torch.arange(nx, (nt + 1) * (nx + 1), nx + 1, device=dev)
    end_axial = float(u[6 * end_row].mean())
    return MembraneCylinderResult(
        displacement=u,
        nodes=nodes,
        elements=elements,
        radius=radius,
        length=length,
        thickness=thickness,
        young=young,
        poisson=poisson,
        pressure=pressure,
        mean_radial_displacement=radial_disp,
        perimeter_radial_displacement=perimeter_disp,
        radial_ripple=ripple,
        rigid_translation=translation,
        end_axial_displacement=end_axial,
        reaction_residual=residual,
        anchor_node=anchor_node,
    )


def _ring_metrics(nodes, u, elements, nt, nx, radius):
    """Gauge-invariant membrane metrics for the half-ring mesh.

    A single anchored node removes the last rigid translation by letting the
    whole ring slide, so a raw ``u . e_r`` reading is *not* the expansion.  Two
    independent gauge-invariant measures are returned instead:

    * a per-ring least-squares fit ``u = t + w e_r`` -- ``w`` is the uniform
      expansion amplitude, ``|t|`` must be the support-induced slide, and the
      RMS of what the fit leaves behind is the facet-scale ripple;
    * the deformed arc length of each ring, ``eps = L' / L0 - 1`` with
      ``w = R eps`` -- a purely geometric measure that uses no fit at all.
    """
    deformed = (nodes + u.reshape(-1, 6)[:, :3]).detach()
    per_ring_w, residual_sq, length_ratio, translation = [], [], [], []
    rows = torch.arange((nt + 1) * (nx + 1), device=nodes.device).reshape(nt + 1, nx + 1)
    for ring in rows.T:
        x0 = nodes[ring]
        xd = deformed[ring]
        er = x0[:, 1:] / radius
        design = torch.zeros((2 * len(ring), 3), dtype=x0.dtype, device=x0.device)
        rhs = (xd - x0)[:, 1:].reshape(-1)
        for k in range(len(ring)):
            design[2 * k, 0] = 1.0
            design[2 * k, 2] = er[k, 0]
            design[2 * k + 1, 1] = 1.0
            design[2 * k + 1, 2] = er[k, 1]
        solution = torch.linalg.lstsq(design, rhs).solution
        per_ring_w.append(float(solution[2]))
        translation.append((float(solution[0]), float(solution[1])))
        residual_sq.append(float(((design @ solution - rhs) ** 2).sum()))
        seg0 = torch.linalg.vector_norm((x0[1:] - x0[:-1]), dim=1).sum()
        segd = torch.linalg.vector_norm((xd[1:] - xd[:-1]), dim=1).sum()
        length_ratio.append(float(segd / seg0))
    n_nodes = 2 * len(residual_sq) * (nt + 1)
    ripple = math.sqrt(sum(residual_sq) / max(n_nodes, 1))
    return (
        sum(per_ring_w) / len(per_ring_w),
        ripple,
        (sum(t[0] for t in translation) / len(translation),
         sum(t[1] for t in translation) / len(translation)),
        radius * (sum(length_ratio) / len(length_ratio) - 1.0),
    )
