"""Independent three-dimensional solid benchmarks and HEX8 mass assembly.

The routines in this module deliberately use analytical continuum/bar solutions
that are independent of the finite-element stiffness and mass implementations.
"""
from dataclasses import dataclass
import math

import torch

from .solid3d import SolidModel, _SIGNS, _hex_gradient, solve_solid, structured_hex_mesh


@dataclass(frozen=True)
class SolidModalResult:
    angular_frequencies: torch.Tensor
    modes: torch.Tensor
    mass: torch.Tensor
    stiffness: torch.Tensor
    free_dofs: torch.Tensor


def _hex_shape(x: torch.Tensor, xi: float, eta: float, zeta: float) -> torch.Tensor:
    signs = x.new_tensor(_SIGNS)
    return ((1 + signs[:, 0] * xi) * (1 + signs[:, 1] * eta)
            * (1 + signs[:, 2] * zeta)) / 8


def assemble_hex8_consistent_mass(
    nodes: torch.Tensor, elements: torch.Tensor, density: torch.Tensor | float
) -> torch.Tensor:
    """Assemble the translational consistent mass using 2x2x2 integration."""
    ne = len(elements)
    rho = torch.as_tensor(density, dtype=nodes.dtype, device=nodes.device).reshape(-1)
    rho = rho.expand(ne)
    x = nodes[elements.long()]
    me_scalar = torch.zeros((ne, 8, 8), dtype=nodes.dtype, device=nodes.device)
    q = 1 / math.sqrt(3)
    for xi in (-q, q):
        for eta in (-q, q):
            for zeta in (-q, q):
                shape = _hex_shape(x, xi, eta, zeta)
                _, det = _hex_gradient(x, xi, eta, zeta)
                me_scalar += rho[:, None, None] * det[:, None, None] * (
                    shape[None, :, None] * shape[None, None, :]
                )
    me = torch.zeros((ne, 24, 24), dtype=nodes.dtype, device=nodes.device)
    for component in range(3):
        me[:, component::3, component::3] = me_scalar
    edofs = torch.stack(tuple(3 * elements + i for i in range(3)), dim=2).reshape(ne, 24).long()
    rows = edofs[:, :, None].expand(-1, -1, 24).reshape(-1)
    cols = edofs[:, None, :].expand(-1, 24, -1).reshape(-1)
    mass = torch.zeros((3 * len(nodes), 3 * len(nodes)), dtype=nodes.dtype, device=nodes.device)
    return mass.index_put((rows, cols), me.reshape(-1), accumulate=True)


def solve_hex8_modes(
    nodes: torch.Tensor,
    elements: torch.Tensor,
    young_modulus: float,
    poisson_ratio: float,
    density: float,
    fixed_dofs: torch.Tensor,
    n_modes: int = 1,
) -> SolidModalResult:
    """Solve the symmetric generalized eigenproblem after essential constraints."""
    zeros = torch.zeros(3 * len(nodes), dtype=nodes.dtype, device=nodes.device)
    model = SolidModel(
        nodes, elements,
        nodes.new_tensor(young_modulus), nodes.new_tensor(poisson_ratio),
        zeros, fixed_dofs, "hex8",
    )
    stiffness = solve_solid(model).stiffness
    mass = assemble_hex8_consistent_mass(nodes, elements, density)
    mask = torch.ones(3 * len(nodes), dtype=torch.bool, device=nodes.device)
    mask[fixed_dofs.long()] = False
    free = torch.arange(3 * len(nodes), device=nodes.device)[mask]
    kff, mff = stiffness[free][:, free], mass[free][:, free]
    # Cholesky reduction preserves symmetry: A=L^-1 K L^-T.
    chol = torch.linalg.cholesky(mff)
    reduced = torch.linalg.solve_triangular(chol, kff, upper=False)
    reduced = torch.linalg.solve_triangular(chol, reduced.T, upper=False).T
    eigenvalues, vectors = torch.linalg.eigh((reduced + reduced.T) / 2)
    positive = eigenvalues > torch.finfo(nodes.dtype).eps * eigenvalues[-1]
    eigenvalues, vectors = eigenvalues[positive][:n_modes], vectors[:, positive][:, :n_modes]
    free_modes = torch.linalg.solve_triangular(chol.T, vectors, upper=True)
    modes = torch.zeros((3 * len(nodes), len(eigenvalues)), dtype=nodes.dtype, device=nodes.device)
    modes[free] = free_modes
    return SolidModalResult(torch.sqrt(eigenvalues), modes, mass, stiffness, free)


def longitudinal_bar_frequency_benchmark(nx: int = 6) -> tuple[float, float, float]:
    """Fixed-free bar first axial frequency; transverse DOFs are constrained."""
    length, width, height = 4.0, 0.2, 0.2
    young, density = 70e9, 2700.0
    nodes, elements = structured_hex_mesh(length, width, height, nx, 1, 1)
    left_x = torch.where(torch.isclose(nodes[:, 0], nodes.new_tensor(0.0)))[0]
    transverse = torch.stack((3 * torch.arange(len(nodes)) + 1,
                              3 * torch.arange(len(nodes)) + 2), dim=1).reshape(-1)
    fixed = torch.unique(torch.cat((3 * left_x, transverse))).long()
    result = solve_hex8_modes(nodes, elements, young, 0.0, density, fixed)
    computed = float(result.angular_frequencies[0])
    reference = math.pi / (2 * length) * math.sqrt(young / density)
    return computed, reference, abs(computed - reference) / reference


def distorted_body_force_bar_benchmark(nx: int = 6) -> tuple[float, float, float]:
    """Non-affine HEX8 mesh under constant axial body force.

    For nu=0, constrained transverse displacement and a uniform body force b,
    the exact tip displacement is b L^2/(2 E). Interior cross-sections are
    warped without changing the boundary, so the test exercises Jacobians at
    all quadrature points rather than reproducing an affine patch.
    """
    length, width, height = 3.0, 0.6, 0.4
    young, body = 200e9, 8e6
    nodes, elements = structured_hex_mesh(length, width, height, nx, 2, 2)
    # Deterministic smooth distortion that vanishes on every exterior face.
    x, y, z = nodes[:, 0], nodes[:, 1], nodes[:, 2]
    envelope = torch.sin(math.pi * x / length) * torch.sin(math.pi * y / width) * torch.sin(math.pi * z / height)
    nodes[:, 0] += 0.10 * (length / nx) * envelope
    nodes[:, 1] += 0.08 * (width / 2) * envelope
    nodes[:, 2] -= 0.06 * (height / 2) * envelope

    mass_unit = assemble_hex8_consistent_mass(nodes, elements, 1.0)
    acceleration = torch.zeros(3 * len(nodes), dtype=nodes.dtype)
    acceleration[0::3] = body
    forces = mass_unit @ acceleration
    left = torch.where(torch.isclose(nodes[:, 0], nodes.new_tensor(0.0)))[0]
    transverse = torch.stack((3 * torch.arange(len(nodes)) + 1,
                              3 * torch.arange(len(nodes)) + 2), dim=1).reshape(-1)
    fixed = torch.unique(torch.cat((3 * left, transverse))).long()
    result = solve_solid(SolidModel(
        nodes, elements, nodes.new_tensor(young), nodes.new_tensor(0.0), forces, fixed, "hex8"
    ))
    right = torch.where(torch.isclose(nodes[:, 0], nodes.new_tensor(length)))[0]
    computed = float(result.displacement[3 * right].mean())
    reference = body * length**2 / (2 * young)
    return computed, reference, abs(computed - reference) / reference
