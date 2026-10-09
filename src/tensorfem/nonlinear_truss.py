"""Small, explicit total-Lagrangian 2-D truss solver.

This module deliberately covers only pin-jointed bars with a St. Venant--Kirchhoff
axial law.  It is a verification kernel, not a general nonlinear FE solver.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class NonlinearTrussModel:
    nodes: torch.Tensor
    elements: torch.Tensor
    young: torch.Tensor
    area: torch.Tensor
    fixed_dofs: torch.Tensor

    def __post_init__(self) -> None:
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 2:
            raise ValueError("nodes must have shape (n, 2)")
        if self.elements.ndim != 2 or self.elements.shape[1] != 2:
            raise ValueError("elements must have shape (m, 2)")
        if len(self.young) != len(self.elements) or len(self.area) != len(self.elements):
            raise ValueError("young and area must have one value per element")


@dataclass(frozen=True)
class NonlinearStep:
    load_factor: float
    displacement: torch.Tensor
    iterations: int
    residual_norm: float


def internal_force_and_tangent(model: NonlinearTrussModel, u: torch.Tensor):
    """Return internal force, consistent tangent and element Green strains."""
    ndof = 2 * len(model.nodes)
    fint = torch.zeros(ndof, dtype=model.nodes.dtype, device=model.nodes.device)
    kt = torch.zeros((ndof, ndof), dtype=model.nodes.dtype, device=model.nodes.device)
    strains = []
    eye = torch.eye(2, dtype=model.nodes.dtype, device=model.nodes.device)
    for e, (i0, j0) in enumerate(model.elements.tolist()):
        X = model.nodes[j0] - model.nodes[i0]
        x = X + u[2*j0:2*j0+2] - u[2*i0:2*i0+2]
        L = torch.linalg.vector_norm(X)
        strain = 0.5 * (torch.dot(x, x) / L**2 - 1.0)
        ea = model.young[e] * model.area[e]
        q = ea * strain * x / L
        ke = ea / L * (torch.outer(x, x) / L**2 + strain * eye)
        dofs = [2*i0, 2*i0+1, 2*j0, 2*j0+1]
        fe = torch.cat((-q, q))
        for a, da in enumerate(dofs):
            fint[da] += fe[a]
            for b, db in enumerate(dofs):
                sa = -1.0 if a < 2 else 1.0
                sb = -1.0 if b < 2 else 1.0
                kt[da, db] += sa * sb * ke[a % 2, b % 2]
        strains.append(strain)
    return fint, kt, torch.stack(strains)


def solve_load_control(
    model: NonlinearTrussModel,
    reference_load: torch.Tensor,
    load_factors: torch.Tensor,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 30,
) -> list[NonlinearStep]:
    """Incremental Newton solve; raises instead of returning unconverged states."""
    ndof = 2 * len(model.nodes)
    if reference_load.shape != (ndof,):
        raise ValueError("reference_load has incompatible shape")
    fixed = set(model.fixed_dofs.tolist())
    free = torch.tensor([i for i in range(ndof) if i not in fixed], dtype=torch.long,
                        device=model.nodes.device)
    u = torch.zeros(ndof, dtype=model.nodes.dtype, device=model.nodes.device)
    result = []
    scale = max(float(torch.linalg.vector_norm(reference_load[free])), 1.0)
    for factor_t in load_factors:
        factor = float(factor_t)
        target = factor * reference_load
        for iteration in range(1, max_iterations + 1):
            fint, kt, _ = internal_force_and_tangent(model, u)
            residual = target - fint
            norm = float(torch.linalg.vector_norm(residual[free]))
            if norm <= tolerance * scale:
                result.append(NonlinearStep(factor, u.clone(), iteration - 1, norm))
                break
            du = torch.linalg.solve(kt[free][:, free], residual[free])
            u = u.index_add(0, free, du)
        else:
            raise RuntimeError(f"Newton iteration did not converge at load factor {factor:g}")
    return result


def two_bar_shallow_arch_reaction(a: float, h: float, young: float, area: float,
                                  downward_displacement: torch.Tensor) -> torch.Tensor:
    """Exact load-displacement curve for a symmetric two-bar shallow arch.

    Positive return value is the downward external force required for equilibrium.
    Displacement control naturally passes limit and snap-through points.
    """
    y = h - downward_displacement
    L = (a*a + h*h) ** 0.5
    return young * area * (h*h - y*y) * y / L**3
