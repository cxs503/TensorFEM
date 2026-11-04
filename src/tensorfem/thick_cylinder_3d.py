"""Three-dimensional Lamé thick cylinder: closed-form oracle and HEX8 sector model.

The repository already verifies the *plane-strain* Lamé cylinder with Q4 elements
(``docs/lame-cylinder-plane-strain.md``) but explicitly records that the
three-dimensional thick cylinder is not done.  This module closes that gap in a
way that is decidable rather than approximate:

* The **closed-end** Lamé solution is an *exact* solution of three-dimensional
  isotropic elasticity in the interior of a hollow cylinder: ``sigma_z`` is
  uniform, ``eps_z`` is uniform, and ``(sigma_r, sigma_theta)`` carry the usual
  ``A -/+ B/r**2`` form.  End effects are exponentially localised, so a
  cylindrical sector with a half-length several times the decay length and a
  mid-plane symmetry condition reproduces the fields to discretisation error.
* The **plane-strain** Lamé solution (``eps_z = 0``) is kept as a discriminating
  negative control: a correct three-dimensional model must *not* match it.

Both references are single closed forms, so every gate below compares the
computed fields with a value that was never used to build the model.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .solid3d import SolidModel, solve_solid


# --------------------------------------------------------------------------- #
# closed-form reference
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LameCylinderReference:
    """Closed-form Lamé hollow cylinder (closed-end or plane-strain)."""

    inner_radius: float = .5
    outer_radius: float = .8
    young: float = 2.e11
    poisson: float = .3
    inner_pressure: float = 1.e7
    outer_pressure: float = 0.
    closed_ends: bool = True

    def __post_init__(self):
        if not 0 < self.inner_radius < self.outer_radius:
            raise ValueError("inner_radius must be positive and below outer_radius")
        if self.young <= 0 or not -1 < self.poisson < .5:
            raise ValueError("invalid elastic constants")

    # A = axial stress of the closed-end solution, B = Lamé constant.
    @property
    def axial_stress(self) -> float:
        a, b = self.inner_radius, self.outer_radius
        return ((self.inner_pressure * a**2 - self.outer_pressure * b**2)
                / (b**2 - a**2))

    @property
    def lame_constant(self) -> float:
        a, b = self.inner_radius, self.outer_radius
        return a**2 * b**2 * (self.inner_pressure - self.outer_pressure) / (b**2 - a**2)

    def sigma_r(self, r):
        return self.axial_stress - self.lame_constant / _as_tensor(r) ** 2

    def sigma_theta(self, r):
        return self.axial_stress + self.lame_constant / _as_tensor(r) ** 2

    def sigma_z(self, r):
        r = _as_tensor(r)
        if self.closed_ends:
            return torch.full_like(r, self.axial_stress)
        # eps_z = 0 plane strain
        return torch.full_like(r, self.poisson * 2 * self.axial_stress)

    def radial_displacement(self, r):
        r = _as_tensor(r)
        a = self.axial_stress * r
        b = self.lame_constant / r
        if self.closed_ends:
            # u_r/r = (sigma_theta - nu*(sigma_r + sigma_z))/E
            return ((1 - 2 * self.poisson) * a + (1 + self.poisson) * b) / self.young
        return (1 + self.poisson) * ((1 - 2 * self.poisson) * a + b) / self.young

    def fields(self, xyz: torch.Tensor) -> dict[str, torch.Tensor]:
        """All compared quantities at the given point coordinates."""
        r = torch.linalg.vector_norm(xyz[..., :2], dim=-1)
        return {
            "u_r": self.radial_displacement(r),
            "sigma_r": self.sigma_r(r),
            "sigma_theta": self.sigma_theta(r),
            "sigma_z": self.sigma_z(r),
        }


def _as_tensor(value):
    return value if torch.is_tensor(value) else torch.as_tensor(value, dtype=torch.float64)


# --------------------------------------------------------------------------- #
# quarter-sector HEX8 mesh (axis-aligned symmetry planes)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CylinderMesh:
    nodes: torch.Tensor
    elements: torch.Tensor
    inner_faces: torch.Tensor
    outer_faces: torch.Tensor
    end_faces: torch.Tensor
    mid_nodes: torch.Tensor
    symmetry: dict[str, torch.Tensor]
    radial_stations: torch.Tensor


def quarter_cylinder_mesh(nr: int, ntheta: int, nz: int, *, inner_radius: float = .5,
                          outer_radius: float = .8, half_length: float = 1.2,
                          radial_grading: float = 1.6, dtype=torch.float64) -> CylinderMesh:
    """A 90-degree sector of a hollow cylinder, z in [0, half_length].

    ``theta`` runs from the +x axis to the +y axis so that both cut faces are
    axis-aligned planes (``y = 0`` and ``x = 0``) and the symmetry conditions
    are plain single-component constraints.  Radial stations are graded with
    ``r = ri + (ro - ri) * s**grading`` (``grading > 1`` clusters elements at the
    inner bore, where ``1/r**2`` gradients are steepest).  ``z = half_length`` is
    the mid-plane of the full cylinder.
    """
    for name, value in (("nr", nr), ("ntheta", ntheta), ("nz", nz)):
        if value < 1:
            raise ValueError(f"{name} must be at least one")
    s = torch.linspace(0., 1., nr + 1, dtype=dtype)
    radial = inner_radius + (outer_radius - inner_radius) * s ** radial_grading
    theta = torch.linspace(0., math.pi / 2, ntheta + 1, dtype=dtype)
    zeta = torch.linspace(0., half_length, nz + 1, dtype=dtype)
    theta[0] = 0.
    theta[-1] = math.pi / 2
    cos, sin = torch.cos(theta), torch.sin(theta)
    # exact zero on the cut planes (torch keeps ~6e-17 residues otherwise)
    cos[0] = 1.; sin[0] = 0.; cos[-1] = 0.; sin[-1] = 1.
    rr = radial[:, None, None].expand(nr + 1, ntheta + 1, nz + 1)
    th = theta[None, :, None].expand(nr + 1, ntheta + 1, nz + 1)
    zz = zeta[None, None, :].expand(nr + 1, ntheta + 1, nz + 1)
    # exact sines/cosines on the two cut planes so the symmetry nodes are exactly on them
    sin_plane = sin[None, :, None].expand(nr + 1, ntheta + 1, nz + 1)
    cos_plane = cos[None, :, None].expand(nr + 1, ntheta + 1, nz + 1)
    nodes = torch.stack((rr * cos_plane, rr * sin_plane, zz), -1).reshape(-1, 3)

    def n(i, j, k):
        return (i * (ntheta + 1) + j) * (nz + 1) + k

    cells = []
    for i in range(nr):
        for j in range(ntheta):
            for k in range(nz):
                cells.append((n(i, j, k), n(i + 1, j, k), n(i + 1, j + 1, k),
                              n(i, j + 1, k), n(i, j, k + 1), n(i + 1, j, k + 1),
                              n(i + 1, j + 1, k + 1), n(i, j + 1, k + 1)))
    elements = torch.tensor(cells, dtype=torch.long)

    inner, outer, ends = [], [], []
    for j in range(ntheta):
        for k in range(nz):
            inner.append((n(0, j, k), n(0, j, k + 1), n(0, j + 1, k + 1), n(0, j + 1, k)))
            outer.append((n(nr, j, k), n(nr, j + 1, k), n(nr, j + 1, k + 1), n(nr, j, k + 1)))
    for j in range(ntheta):
        for i in range(nr):
            ends.append((n(i, j, 0), n(i + 1, j, 0), n(i + 1, j + 1, 0), n(i, j + 1, 0)))
    face = lambda rows: torch.tensor(rows, dtype=torch.long)
    cut_a = torch.arange(nodes.shape[0])[torch.isclose(nodes[:, 1], nodes.new_tensor(0.))]
    cut_b = torch.arange(nodes.shape[0])[torch.isclose(nodes[:, 0], nodes.new_tensor(0.))]
    mid = torch.arange(nodes.shape[0])[torch.isclose(nodes[:, 2], nodes.new_tensor(half_length))]
    return CylinderMesh(nodes, elements, face(inner), face(outer), face(ends),
                        mid, {"y_zero": cut_a, "x_zero": cut_b, "z_mid": mid},
                        radial)


# --------------------------------------------------------------------------- #
# consistent surface loads on a curved bilinear facet
# --------------------------------------------------------------------------- #
_GP = 1 / math.sqrt(3)


def _quad_shape(s: float, t: float) -> tuple[float, float, float, float]:
    return (.25 * (1 - s) * (1 - t), .25 * (1 + s) * (1 - t),
            .25 * (1 + s) * (1 + t), .25 * (1 - s) * (1 + t))


def face_load(xyz: torch.Tensor, traction_fn) -> torch.Tensor:
    """Consistent nodal forces ``int N_i t dA`` on a bilinear facet.

    ``traction_fn`` receives Gauss-point coordinates ``[g,3]`` and returns
    ``[g,3]`` tractions.  The facet Jacobian is 2x2 Gauss integrated, and the
    traction direction comes from the *analytic* surface normal supplied by
    ``traction_fn`` rather than from the facet, so a faceted mesh is loaded with
    the exact radial pressure.
    """
    forces = xyz.new_zeros((4, 3))
    for s in (-_GP, _GP):
        for t in (-_GP, _GP):
            shape = _quad_shape(s, t)
            point = sum(w * x for w, x in zip(shape, xyz))
            d_s = .25 * (-(1 - t) * xyz[0] + (1 - t) * xyz[1] + (1 + t) * xyz[2] - (1 + t) * xyz[3])
            d_t = .25 * (-(1 - s) * xyz[0] - (1 + s) * xyz[1] + (1 + s) * xyz[2] + (1 - s) * xyz[3])
            jac = torch.linalg.vector_norm(torch.linalg.cross(d_s, d_t))
            traction = traction_fn(point.reshape(1, 3)).reshape(3)
            forces = forces + torch.stack([w * traction * jac for w in shape])
    return forces


def _radial_pressure_magnitude(pressure: float, sign: float):
    def fn(points: torch.Tensor) -> torch.Tensor:
        radial = points.clone()
        radial[:, 2] = 0.
        return sign * pressure * radial / torch.linalg.vector_norm(radial, dim=-1, keepdim=True)
    return fn


def _axial_traction(sigma: float):
    def fn(points: torch.Tensor) -> torch.Tensor:
        out = points.new_zeros(points.shape)
        out[:, 2] = sigma
        return out
    return fn


# --------------------------------------------------------------------------- #
# assembled sector solve
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CylinderSolve:
    displacement: torch.Tensor
    residual: torch.Tensor
    stress: torch.Tensor
    nodes: torch.Tensor
    elements: torch.Tensor
    mesh: CylinderMesh
    reference: LameCylinderReference
    applied_forces: torch.Tensor
    axial_reaction: float
    element_centres: torch.Tensor


def solve_thick_cylinder(reference: LameCylinderReference, *, nr: int, ntheta: int, nz: int,
                         half_length: float = 1.2, radial_grading: float = 1.6) -> CylinderSolve:
    """Solve one quarter-sector closed-end (or open-end) Lamé cylinder."""
    mesh = quarter_cylinder_mesh(nr, ntheta, nz, inner_radius=reference.inner_radius,
                                 outer_radius=reference.outer_radius, half_length=half_length,
                                 radial_grading=radial_grading)
    nodes = mesh.nodes
    forces = torch.zeros(3 * nodes.shape[0], dtype=nodes.dtype)
    lin = lambda rows: torch.stack(tuple(3 * rows + i for i in range(3)), 1).reshape(-1)
    # Traction is t = -p * n on the body's *outward* normal: n = -e_r at the bore
    # (so the bore is pushed outward) and n = +e_r at the outer surface.
    if reference.inner_pressure:
        for quad in mesh.inner_faces:
            forces[lin(quad)] += face_load(
                nodes[quad], _radial_pressure_magnitude(reference.inner_pressure, +1.)).reshape(-1)
    if reference.outer_pressure:
        for quad in mesh.outer_faces:
            forces[lin(quad)] += face_load(
                nodes[quad], _radial_pressure_magnitude(reference.outer_pressure, -1.)).reshape(-1)
    if reference.closed_ends:
        # the material end kept by the mid-plane symmetry sits at z = 0, outward
        # normal -e_z, so a tensile axial stress A gives traction -A e_z.
        for quad in mesh.end_faces:
            forces[lin(quad)] += face_load(
                nodes[quad], _axial_traction(-reference.axial_stress)).reshape(-1)
    fixed = torch.cat((3 * mesh.symmetry["y_zero"] + 1, 3 * mesh.symmetry["x_zero"],
                       3 * mesh.symmetry["z_mid"] + 2)).unique()
    model = SolidModel(nodes, mesh.elements, torch.tensor([reference.young]),
                       torch.tensor([reference.poisson]), forces, fixed, element_type="hex8")
    result = solve_solid(model)
    centres = nodes[mesh.elements].mean(1)
    return CylinderSolve(result.displacement, result.reaction, result.stress, nodes,
                         mesh.elements, mesh, reference, forces,
                         float(result.reaction[3 * mesh.symmetry["z_mid"] + 2].sum()),
                         centres)


# --------------------------------------------------------------------------- #
# error metrics and gates
# 
def stress_in_cylindrical_frame(stress: torch.Tensor, xyz: torch.Tensor) -> dict[str, torch.Tensor]:
    """Rotate Cartesian stress components into (sigma_r, sigma_theta, sigma_z).

    ``solid3d`` returns ``[s_xx, s_yy, s_zz, s_xy, s_yz, s_xz]`` in the global
    frame; the Lamé comparison needs the radial and hoop components, so the
    tensor is rotated by the local polar basis at each point.
    """
    x, y = xyz[:, 0], xyz[:, 1]
    radius = torch.sqrt(x * x + y * y)
    er = torch.stack((x / radius, y / radius, torch.zeros_like(x)), -1)
    et = torch.stack((-y / radius, x / radius, torch.zeros_like(x)), -1)
    full = torch.zeros(stress.shape[0], 3, 3, dtype=stress.dtype)
    (full[:, 0, 0], full[:, 1, 1], full[:, 2, 2],
     full[:, 0, 1], full[:, 1, 2], full[:, 0, 2]) = (stress[:, 0], stress[:, 1],
                                                    stress[:, 2], stress[:, 3],
                                                    stress[:, 4], stress[:, 5])
    full[:, 1, 0], full[:, 2, 1], full[:, 2, 0] = (stress[:, 3], stress[:, 4], stress[:, 5])
    return {"sigma_r": torch.einsum("ni,nij,nj->n", er, full, er),
            "sigma_theta": torch.einsum("ni,nij,nj->n", et, full, et),
            "sigma_z": stress[:, 2]}

# --------------------------------------------------------------------------- #
def _mid_band_centres(solve: CylinderSolve, *, band: float = .35) -> torch.Tensor:
    """Element centres in the mid-plane band, i.e. far from the loaded end."""
    zmax = float(solve.mesh.nodes[:, 2].max())
    return (solve.element_centres[:, 2] >= band * zmax)


def field_errors(solve: CylinderSolve, reference: LameCylinderReference | None = None,
                 *, band: float = .35) -> dict[str, float]:
    """Element-centre stress errors and bore displacement error vs a reference."""
    reference = reference or solve.reference
    mask = _mid_band_centres(solve, band=band)
    centres = solve.element_centres[mask]
    xyz = centres
    numeric = stress_in_cylindrical_frame(solve.stress[mask], xyz)
    exact = reference.fields(xyz)
    scale = float(torch.abs(exact["sigma_theta"]).max())
    errors = {}
    for key, value in numeric.items():
        diff = value - exact[key]
        errors[f"{key}_l2"] = float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(exact[key]))
        errors[f"{key}_max"] = float(torch.abs(diff).max() / scale)
    # bore radial displacement at mid-plane nodes (theta-free by symmetry)
    bore = torch.isclose(torch.linalg.vector_norm(solve.nodes[:, :2], dim=-1),
                         solve.nodes.new_tensor(reference.inner_radius))
    mid = torch.isclose(solve.nodes[:, 2], solve.nodes.new_tensor(float(solve.nodes[:, 2].max())))
    picked = bore & mid
    u = solve.displacement.reshape(-1, 3)[picked]
    radial = u[:, :2]
    r_true = torch.linalg.vector_norm(solve.nodes[picked][:, :2], dim=-1, keepdim=True)
    u_r = (radial * solve.nodes[picked][:, :2] / r_true).sum(-1)
    exact_u = reference.radial_displacement(r_true.reshape(-1))
    errors["u_r_max"] = float(torch.abs(u_r - exact_u).max() / torch.abs(exact_u).max())
    errors["u_r_mean"] = float(((u_r - exact_u) / exact_u).abs().mean())
    return errors


@dataclass(frozen=True)
class CylinderQualification:
    """Fail-closed verdict for the three-dimensional Lamé cylinder."""

    status: str
    tolerance: float
    errors: dict[str, float]
    convergence: dict[str, float]
    length_sensitivity: float
    negative_control: dict[str, float]
    equilibrium: dict[str, float]
    missing: tuple[str, ...]

    @property
    def qualified(self) -> bool:
        return self.status == "qualified"

    def as_dict(self) -> dict[str, object]:
        return {"status": self.status, "tolerance": self.tolerance,
                "errors": self.errors, "convergence": self.convergence,
                "length_sensitivity": self.length_sensitivity,
                "negative_control": self.negative_control,
                "equilibrium": self.equilibrium, "missing": list(self.missing)}


GATE_KEYS = ("sigma_r_l2", "sigma_theta_l2", "sigma_z_l2", "u_r_max")
MONOTONE_KEYS = ("sigma_r_l2", "sigma_theta_l2", "u_r_max")


def _plane_strain_counterpart(reference: LameCylinderReference) -> LameCylinderReference:
    return LameCylinderReference(
        inner_radius=reference.inner_radius, outer_radius=reference.outer_radius,
        young=reference.young, poisson=reference.poisson,
        inner_pressure=reference.inner_pressure,
        outer_pressure=reference.outer_pressure, closed_ends=False)


def _observed_order(errors: list[float], sizes: list[float]) -> float:
    """Least-squares slope of log(error) against log(element size)."""
    if len(errors) < 2:
        return float("nan")
    xs = [math.log(size) for size in sizes]
    ys = [math.log(max(value, 1e-16)) for value in errors]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    return float(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom)


def qualify_thick_cylinder(reference: LameCylinderReference,
                           solves: list[CylinderSolve], *,
                           length_run: CylinderSolve | None = None,
                           tolerance: float = .03) -> CylinderQualification:
    """Strict gate: three refined meshes, length truncation, negative control.

    Every reported quantity must beat ``tolerance`` (capped at 3 %) on the finest
    mesh, the error sequences must decrease monotonically, a shorter cylinder
    must reproduce the same error level (Saint-Venant check), the plane-strain
    Lamé solution must *not* be reproduced, and the load path across the
    symmetry cuts must close.
    """
    if not 0 < tolerance <= .03:
        raise ValueError("tolerance must lie in (0, 0.03]")
    missing: list[str] = []
    solves = list(solves)
    if len(solves) < 3:
        missing.append("at least three mesh densities")
    counts = [s.mesh.elements.shape[0] for s in solves]
    if len(solves) >= 3 and not all(b > a for a, b in zip(counts, counts[1:])):
        missing.append("strictly refined meshes")
    per_mesh = [field_errors(s, reference) for s in solves]
    errors = per_mesh[-1]
    if any(errors[key] >= tolerance for key in GATE_KEYS):
        missing.append("fine-mesh closed-form agreement below 3 %")

    sizes = [float(s.mesh.elements.shape[0]) ** (-1 / 3) for s in solves]
    convergence = {}
    for key in MONOTONE_KEYS + ("sigma_z_l2",):
        sequence = [entry[key] for entry in per_mesh]
        convergence[f"{key}_order"] = _observed_order(sequence, sizes)
        if key in MONOTONE_KEYS and any(b >= a for a, b in zip(sequence, sequence[1:])):
            missing.append(f"monotone decrease of {key}")

    length_sensitivity = float("nan")
    if length_run is None:
        missing.append("cylinder-length truncation study")
    else:
        other = field_errors(length_run, reference)
        # absolute difference of the two error levels: a truncation study must not
        # move the fields by more than the tolerance itself.
        length_sensitivity = max(abs(other[key] - errors[key]) for key in GATE_KEYS)
        if not length_sensitivity < tolerance:
            missing.append("Saint-Venant length sensitivity below 3 %")

    negative_control = field_errors(solves[-1], _plane_strain_counterpart(reference))
    if negative_control["sigma_z_l2"] < .1:
        missing.append("plane-strain negative control must be rejected")

    fine = solves[-1]
    a, b = reference.inner_radius, reference.outer_radius
    sector_area = (math.pi / 2) / 2 * (b**2 - a**2)
    half_length = float(fine.mesh.nodes[:, 2].max())
    # the end traction is pushed at z = 0 in -z, so the mid-plane cut carries +A*area
    expected_axial = reference.axial_stress * sector_area
    expected_radial = -reference.inner_pressure * a * half_length
    reaction = fine.residual.reshape(-1, 3)
    reaction_flat = fine.residual
    equilibrium = {
        "axial_cut_reaction": float(reaction[fine.mesh.symmetry["z_mid"], 2].sum()),
        "axial_expected": expected_axial,
        "axial_balance": abs(float(reaction[fine.mesh.symmetry["z_mid"], 2].sum())
                             / expected_axial - 1),
        "radial_cut_reaction": float(reaction[fine.mesh.symmetry["y_zero"], 1].sum()),
        "radial_expected": expected_radial,
        "radial_balance": abs(float(reaction[fine.mesh.symmetry["y_zero"], 1].sum())
                              / expected_radial - 1),
    }
    free = torch.ones(3 * fine.nodes.shape[0], dtype=torch.bool)
    free[torch.cat((3 * fine.mesh.symmetry["y_zero"] + 1, 3 * fine.mesh.symmetry["x_zero"],
                    3 * fine.mesh.symmetry["z_mid"] + 2)).unique()] = False
    peak_load = max(float(torch.abs(fine.applied_forces.reshape(-1, 3)).max()), 1e-30)
    equilibrium["free_residual"] = float(torch.abs(reaction_flat[free]).max()) / peak_load
    if equilibrium["axial_balance"] >= tolerance or equilibrium["radial_balance"] >= tolerance:
        missing.append("cut-plane load path closure below 3 %")
    if equilibrium["free_residual"] >= 1e-8:
        missing.append("solved free-dof equilibrium residual below 1e-8")
    return CylinderQualification(
        "qualified" if not missing else "blocked", tolerance, errors, convergence,
        length_sensitivity, negative_control, equilibrium,
        tuple(dict.fromkeys(missing)))
