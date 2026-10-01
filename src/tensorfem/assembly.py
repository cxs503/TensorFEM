"""Differentiable global matrix assembly."""
import torch

from .elements import truss2d_dofs, truss2d_stiffness
from .model import TrussModel


def assemble_truss_stiffness(model: TrussModel) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    young, area = model.material_vectors()
    coordinates = model.nodes[model.elements]
    local, length, direction = truss2d_stiffness(coordinates, young, area)
    dofs = truss2d_dofs(model.elements)
    rows = dofs[:, :, None].expand(-1, 4, 4).reshape(-1)
    cols = dofs[:, None, :].expand(-1, 4, 4).reshape(-1)
    global_matrix = torch.zeros(
        (model.n_dofs, model.n_dofs), dtype=model.dtype, device=model.device
    )
    global_matrix = global_matrix.index_put((rows, cols), local.reshape(-1), accumulate=True)
    return global_matrix, length, direction
