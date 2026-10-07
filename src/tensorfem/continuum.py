"""Small-strain two-dimensional continuum finite elements.

This module is deliberately self-contained so the continuum solver can evolve
without coupling its public data structures to the truss implementation.
"""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ContinuumModel:
    nodes: torch.Tensor
    elements: torch.Tensor
    young_modulus: torch.Tensor
    poisson_ratio: torch.Tensor
    thickness: torch.Tensor
    forces: torch.Tensor
    fixed_dofs: torch.Tensor
    element_type: str = "q4"
    plane: str = "stress"
    prescribed_values: torch.Tensor | None = None

    def __post_init__(self) -> None:
        nen = {"q4": 4, "cst": 3}.get(self.element_type.lower())
        if nen is None:
            raise ValueError("element_type must be 'q4' or 'cst'")
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 2:
            raise ValueError("nodes must have shape [n_nodes, 2]")
        if self.elements.ndim != 2 or self.elements.shape[1] != nen:
            raise ValueError(f"{self.element_type} connectivity must have {nen} nodes")
        if self.forces.shape != (2 * self.nodes.shape[0],):
            raise ValueError("forces must have shape [2 * n_nodes]")
        if self.plane not in ("stress", "strain"):
            raise ValueError("plane must be 'stress' or 'strain'")
        if self.prescribed_values is not None and self.prescribed_values.numel() != self.fixed_dofs.numel():
            raise ValueError("prescribed_values must match fixed_dofs")
        if self.elements.numel() and (self.elements.min() < 0 or self.elements.max() >= len(self.nodes)):
            raise ValueError("invalid node in element connectivity")

    @property
    def n_dofs(self) -> int:
        return 2 * self.nodes.shape[0]


@dataclass(frozen=True)
class ContinuumResult:
    displacement: torch.Tensor
    reaction: torch.Tensor
    strain: torch.Tensor
    stress: torch.Tensor
    strain_energy: torch.Tensor
    stiffness: torch.Tensor


def elasticity_matrix(young: torch.Tensor, poisson: torch.Tensor, plane: str) -> torch.Tensor:
    """Isotropic constitutive matrix using engineering shear strain."""
    z = torch.zeros_like(young)
    o = torch.ones_like(young)
    if plane == "stress":
        base = torch.stack((o, poisson, z, poisson, o, z, z, z, (o-poisson)/2), dim=-1).reshape(-1, 3, 3)
        return (young / (1-poisson**2))[:, None, None] * base
    c = young / ((1+poisson)*(1-2*poisson))
    base = torch.stack((1-poisson, poisson, z, poisson, 1-poisson, z, z, z, (1-2*poisson)/2), dim=-1).reshape(-1, 3, 3)
    return c[:, None, None] * base


def _b_matrix(derivatives: torch.Tensor) -> torch.Tensor:
    # derivatives: [element, node, xy]
    ne, nn, _ = derivatives.shape
    b = torch.zeros((ne, 3, 2*nn), dtype=derivatives.dtype, device=derivatives.device)
    b[:, 0, 0::2] = derivatives[:, :, 0]
    b[:, 1, 1::2] = derivatives[:, :, 1]
    b[:, 2, 0::2] = derivatives[:, :, 1]
    b[:, 2, 1::2] = derivatives[:, :, 0]
    return b


def q4_stiffness(coordinates: torch.Tensor, dmat: torch.Tensor, thickness: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fully integrated bilinear quadrilateral stiffness and centre B matrix."""
    g = 1.0 / 3.0**0.5
    points = ((-g,-g), (g,-g), (g,g), (-g,g))
    ke = torch.zeros((len(coordinates), 8, 8), dtype=coordinates.dtype, device=coordinates.device)
    centre_b = None
    for xi, eta in points + ((0.0, 0.0),):
        natural = coordinates.new_tensor([
            [-(1-eta), -(1-xi)], [(1-eta), -(1+xi)],
            [(1+eta), (1+xi)], [-(1+eta), (1-xi)]
        ]) / 4
        jac = torch.einsum("eia,ib->eab", coordinates, natural)
        det = torch.linalg.det(jac)
        if torch.any(det <= 0):
            raise ValueError("Q4 element has non-positive Jacobian")
        deriv = torch.einsum("ib,ebc->eic", natural, torch.linalg.inv(jac))
        b = _b_matrix(deriv)
        if xi == 0.0:
            centre_b = b
        else:
            ke = ke + (b.transpose(1,2) @ dmat @ b) * (det*thickness)[:,None,None]
    return ke, centre_b


def cst_stiffness(coordinates: torch.Tensor, dmat: torch.Tensor, thickness: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = coordinates[:, :, 0], coordinates[:, :, 1]
    twice_area = (x[:,1]-x[:,0])*(y[:,2]-y[:,0])-(x[:,2]-x[:,0])*(y[:,1]-y[:,0])
    if torch.any(twice_area <= 0):
        raise ValueError("CST element has non-positive area")
    bcoef = torch.stack((y[:,1]-y[:,2], y[:,2]-y[:,0], y[:,0]-y[:,1]), 1)
    ccoef = torch.stack((x[:,2]-x[:,1], x[:,0]-x[:,2], x[:,1]-x[:,0]), 1)
    deriv = torch.stack((bcoef, ccoef), 2) / twice_area[:,None,None]
    b = _b_matrix(deriv)
    ke = (b.transpose(1,2) @ dmat @ b) * (0.5*twice_area*thickness)[:,None,None]
    return ke, b


def solve_continuum(model: ContinuumModel) -> ContinuumResult:
    ne, nn = model.elements.shape
    expand = lambda x: x.to(dtype=model.nodes.dtype, device=model.nodes.device).reshape(-1).expand(ne)
    young, poisson, thickness = map(expand, (model.young_modulus, model.poisson_ratio, model.thickness))
    dmat = elasticity_matrix(young, poisson, model.plane)
    coords = model.nodes[model.elements.long()]
    ke, b = (q4_stiffness if model.element_type.lower() == "q4" else cst_stiffness)(coords, dmat, thickness)
    edofs = torch.stack(tuple(2*model.elements + i for i in (0,1)), 2).reshape(ne, 2*nn).long()
    rows = edofs[:,:,None].expand(-1,-1,2*nn).reshape(-1)
    cols = edofs[:,None,:].expand(-1,2*nn,-1).reshape(-1)
    stiffness = torch.zeros((model.n_dofs, model.n_dofs), dtype=model.nodes.dtype, device=model.nodes.device)
    stiffness = stiffness.index_put((rows, cols), ke.reshape(-1), accumulate=True)
    fixed = model.fixed_dofs.to(device=model.nodes.device, dtype=torch.long)
    values = (torch.zeros_like(fixed, dtype=model.nodes.dtype) if model.prescribed_values is None else model.prescribed_values)
    mask = torch.ones(model.n_dofs, dtype=torch.bool, device=model.nodes.device)
    mask[fixed] = False
    free = torch.arange(model.n_dofs, device=model.nodes.device)[mask]
    forces = model.forces.to(dtype=model.nodes.dtype, device=model.nodes.device)
    values = values.to(dtype=model.nodes.dtype, device=model.nodes.device)
    rhs = forces[free] - stiffness[free][:,fixed] @ values
    u = torch.zeros(model.n_dofs, dtype=model.nodes.dtype, device=model.nodes.device)
    u = u.index_put((fixed,), values).index_put((free,), torch.linalg.solve(stiffness[free][:,free], rhs))
    reaction = stiffness @ u - forces
    strain = torch.einsum("eij,ej->ei", b, u[edofs])
    stress = torch.einsum("eij,ej->ei", dmat, strain)
    return ContinuumResult(u, reaction, strain, stress, 0.5*u@stiffness@u, stiffness)


def rectangular_q4_mesh(length: float, height: float, nx: int, ny: int, *, dtype=torch.float64):
    """Structured, counter-clockwise Q4 mesh."""
    x = torch.linspace(0, length, nx+1, dtype=dtype)
    y = torch.linspace(-height/2, height/2, ny+1, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    nodes = torch.stack((xx.reshape(-1), yy.reshape(-1)), 1)
    cells = []
    for j in range(ny):
        for i in range(nx):
            n0 = j*(nx+1)+i
            cells.append((n0,n0+1,n0+nx+2,n0+nx+1))
    return nodes, torch.tensor(cells, dtype=torch.long)
