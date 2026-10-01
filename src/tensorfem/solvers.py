"""Linear structural solvers."""
from dataclasses import dataclass

import torch

from .assembly import assemble_truss_stiffness
from .elements import truss2d_dofs
from .model import TrussModel


@dataclass(frozen=True)
class StaticResult:
    displacement: torch.Tensor
    reaction: torch.Tensor
    axial_strain: torch.Tensor
    axial_stress: torch.Tensor
    axial_force: torch.Tensor
    strain_energy: torch.Tensor


def solve_linear_static(model: TrussModel) -> StaticResult:
    stiffness, length, direction = assemble_truss_stiffness(model)
    all_dofs = torch.arange(model.n_dofs, device=model.device)
    free_mask = torch.ones(model.n_dofs, dtype=torch.bool, device=model.device)
    free_mask[model.fixed_dofs.long()] = False
    free = all_dofs[free_mask]
    if free.numel() == 0:
        raise ValueError("model has no free degrees of freedom")
    reduced = stiffness[free][:, free]
    displacement_free = torch.linalg.solve(reduced, model.forces[free])
    displacement = torch.zeros(model.n_dofs, dtype=model.dtype, device=model.device)
    displacement = displacement.index_put((free,), displacement_free)
    reaction = stiffness @ displacement - model.forces
    element_u = displacement[truss2d_dofs(model.elements)].reshape(-1, 2, 2)
    extension = torch.sum((element_u[:, 1] - element_u[:, 0]) * direction, dim=1)
    strain = extension / length
    young, area = model.material_vectors()
    stress = young * strain
    axial_force = stress * area
    energy = 0.5 * displacement @ stiffness @ displacement
    return StaticResult(displacement, reaction, strain, stress, axial_force, energy)
