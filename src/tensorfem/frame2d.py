"""Differentiable linear 2-D Euler--Bernoulli frame analysis.

Each node owns ``(u_x, u_y, theta_z)``.  Distributed loads are specified in
the element local transverse direction; positive values act along local +y.
"""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FrameModel:
    nodes: torch.Tensor
    elements: torch.Tensor
    young_modulus: torch.Tensor
    area: torch.Tensor
    inertia: torch.Tensor
    forces: torch.Tensor
    fixed_dofs: torch.Tensor
    distributed_load_y: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 2:
            raise ValueError("nodes must have shape [n_nodes, 2]")
        if self.elements.ndim != 2 or self.elements.shape[1] != 2:
            raise ValueError("elements must have shape [n_elements, 2]")
        if self.forces.shape != (3 * self.nodes.shape[0],):
            raise ValueError("forces must have shape [3 * n_nodes]")
        n = self.elements.shape[0]
        for name in ("young_modulus", "area", "inertia"):
            if getattr(self, name).numel() not in (1, n):
                raise ValueError(f"{name} must be scalar or per-element")
        if self.distributed_load_y is not None and self.distributed_load_y.numel() not in (1, n):
            raise ValueError("distributed_load_y must be scalar or per-element")
        if self.elements.numel() and (self.elements.min() < 0 or self.elements.max() >= len(self.nodes)):
            raise ValueError("element connectivity references an invalid node")
        if self.fixed_dofs.numel() and (self.fixed_dofs.min() < 0 or self.fixed_dofs.max() >= self.forces.numel()):
            raise ValueError("fixed_dofs contains an invalid degree of freedom")

    @property
    def n_dofs(self) -> int:
        return 3 * self.nodes.shape[0]


@dataclass(frozen=True)
class FrameResult:
    displacement: torch.Tensor
    reaction: torch.Tensor
    element_end_forces_local: torch.Tensor
    strain_energy: torch.Tensor


def frame2d_dofs(elements: torch.Tensor) -> torch.Tensor:
    i, j = elements[:, 0], elements[:, 1]
    return torch.stack((3*i, 3*i+1, 3*i+2, 3*j, 3*j+1, 3*j+2), dim=1)


def _element_matrices(xy: torch.Tensor, e: torch.Tensor, a: torch.Tensor, inertia: torch.Tensor):
    delta = xy[:, 1] - xy[:, 0]
    length = torch.linalg.vector_norm(delta, dim=1)
    if torch.any(length <= torch.finfo(xy.dtype).eps):
        raise ValueError("zero-length frame element")
    c, s = delta[:, 0] / length, delta[:, 1] / length
    ea_l = e*a/length
    ei = e*inertia
    k = torch.zeros((len(xy), 6, 6), dtype=xy.dtype, device=xy.device)
    k[:, 0, 0] = k[:, 3, 3] = ea_l
    k[:, 0, 3] = k[:, 3, 0] = -ea_l
    k[:, 1, 1] = k[:, 4, 4] = 12*ei/length**3
    k[:, 1, 4] = k[:, 4, 1] = -12*ei/length**3
    k[:, 1, 2] = k[:, 2, 1] = 6*ei/length**2
    k[:, 1, 5] = k[:, 5, 1] = 6*ei/length**2
    k[:, 2, 4] = k[:, 4, 2] = -6*ei/length**2
    k[:, 4, 5] = k[:, 5, 4] = -6*ei/length**2
    k[:, 2, 2] = k[:, 5, 5] = 4*ei/length
    k[:, 2, 5] = k[:, 5, 2] = 2*ei/length
    transform = torch.zeros_like(k)
    transform[:, 0, 0], transform[:, 0, 1] = c, s
    transform[:, 1, 0], transform[:, 1, 1] = -s, c
    transform[:, 2, 2] = 1
    transform[:, 3, 3], transform[:, 3, 4] = c, s
    transform[:, 4, 3], transform[:, 4, 4] = -s, c
    transform[:, 5, 5] = 1
    return k, transform, length


def solve_frame_static(model: FrameModel) -> FrameResult:
    n = model.elements.shape[0]
    e = model.young_modulus.reshape(-1).expand(n)
    a = model.area.reshape(-1).expand(n)
    inertia = model.inertia.reshape(-1).expand(n)
    xy = model.nodes[model.elements.long()]
    local_k, transform, length = _element_matrices(xy, e, a, inertia)
    global_k_e = transform.transpose(1, 2) @ local_k @ transform
    dofs = frame2d_dofs(model.elements.long())
    K = torch.zeros((model.n_dofs, model.n_dofs), dtype=model.nodes.dtype, device=model.nodes.device)
    flat = dofs[:, :, None] * model.n_dofs + dofs[:, None, :]
    K = K.reshape(-1).scatter_add(0, flat.reshape(-1), global_k_e.reshape(-1)).reshape(model.n_dofs, model.n_dofs)
    total_force = model.forces.clone()
    equivalent_local = torch.zeros((n, 6), dtype=model.nodes.dtype, device=model.nodes.device)
    if model.distributed_load_y is not None:
        q = model.distributed_load_y.reshape(-1).expand(n)
        equivalent_local[:, 1] = q*length/2
        equivalent_local[:, 2] = q*length**2/12
        equivalent_local[:, 4] = q*length/2
        equivalent_local[:, 5] = -q*length**2/12
        equivalent_global = (transform.transpose(1, 2) @ equivalent_local[:, :, None]).squeeze(-1)
        total_force = total_force.reshape(-1).scatter_add(0, dofs.reshape(-1), equivalent_global.reshape(-1))
    free_mask = torch.ones(model.n_dofs, dtype=torch.bool, device=model.nodes.device)
    free_mask[model.fixed_dofs.long()] = False
    free = torch.arange(model.n_dofs, device=model.nodes.device)[free_mask]
    if not free.numel():
        raise ValueError("model has no free degrees of freedom")
    u_free = torch.linalg.solve(K[free][:, free], total_force[free])
    u = torch.zeros(model.n_dofs, dtype=model.nodes.dtype, device=model.nodes.device).index_put((free,), u_free)
    reaction = K @ u - total_force
    u_local = (transform @ u[dofs][:, :, None]).squeeze(-1)
    end_forces = (local_k @ u_local[:, :, None]).squeeze(-1) - equivalent_local
    return FrameResult(u, reaction, end_forces, 0.5*u @ K @ u)
