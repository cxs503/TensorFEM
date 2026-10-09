"""Validated finite-element model containers."""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TrussModel:
    """A batch-free 2D truss model with two translational DOFs per node."""

    nodes: torch.Tensor
    elements: torch.Tensor
    young_modulus: torch.Tensor
    area: torch.Tensor
    forces: torch.Tensor
    fixed_dofs: torch.Tensor

    def __post_init__(self) -> None:
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 2:
            raise ValueError("nodes must have shape [n_nodes, 2]")
        if self.elements.ndim != 2 or self.elements.shape[1] != 2:
            raise ValueError("elements must have shape [n_elements, 2]")
        if self.forces.shape != (self.nodes.shape[0] * 2,):
            raise ValueError("forces must have shape [2 * n_nodes]")
        n_elements = self.elements.shape[0]
        if self.young_modulus.numel() not in (1, n_elements):
            raise ValueError("young_modulus must be scalar or per-element")
        if self.area.numel() not in (1, n_elements):
            raise ValueError("area must be scalar or per-element")
        if self.elements.numel() and (
            self.elements.min() < 0 or self.elements.max() >= self.nodes.shape[0]
        ):
            raise ValueError("element connectivity references an invalid node")
        if self.fixed_dofs.numel() and (
            self.fixed_dofs.min() < 0 or self.fixed_dofs.max() >= self.forces.numel()
        ):
            raise ValueError("fixed_dofs contains an invalid degree of freedom")

    @property
    def dtype(self) -> torch.dtype:
        return self.nodes.dtype

    @property
    def device(self) -> torch.device:
        return self.nodes.device

    @property
    def n_dofs(self) -> int:
        return self.nodes.shape[0] * 2

    def material_vectors(self) -> tuple[torch.Tensor, torch.Tensor]:
        n = self.elements.shape[0]
        young = self.young_modulus.reshape(-1).expand(n)
        area = self.area.reshape(-1).expand(n)
        return young, area
