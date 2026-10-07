"""Locking-resistant, fully integrated B-bar HEX8 linear elasticity."""
import torch
from tensorfem.solid3d import SolidModel, SolidResult, elasticity_matrix_3d, _b, _hex_gradient


def hex8_bbar_stiffness(x: torch.Tensor, d: torch.Tensor):
    """Project volumetric strain to its element mean without hourglass modes."""
    q = 1 / 3**.5
    points = []
    volume = torch.zeros(len(x), dtype=x.dtype, device=x.device)
    div_integral = torch.zeros((len(x), 24), dtype=x.dtype, device=x.device)
    selector = x.new_tensor([1., 1., 1., 0., 0., 0.])
    for xi in (-q, q):
        for eta in (-q, q):
            for zeta in (-q, q):
                gradient, det = _hex_gradient(x, xi, eta, zeta)
                b = _b(gradient)
                divergence = b[:, 0] + b[:, 1] + b[:, 2]
                points.append((b, divergence, det))
                volume += det
                div_integral += divergence * det[:, None]
    div_average = div_integral / volume[:, None]
    average_vol = selector[None, :, None] * div_average[:, None, :] / 3
    stiffness = torch.zeros((len(x), 24, 24), dtype=x.dtype, device=x.device)
    for b, divergence, det in points:
        point_vol = selector[None, :, None] * divergence[:, None, :] / 3
        b_bar = b - point_vol + average_vol
        stiffness += (b_bar.transpose(1, 2) @ d @ b_bar) * det[:, None, None]
    gradient, _ = _hex_gradient(x, 0., 0., 0.)
    b = _b(gradient)
    divergence = b[:, 0] + b[:, 1] + b[:, 2]
    b_centroid = b - selector[None, :, None] * divergence[:, None, :] / 3 + average_vol
    return stiffness, b_centroid


def solve_solid_bbar(model: SolidModel):
    """Solve an existing :class:`SolidModel` using B-bar HEX8 elements."""
    if model.element_type.lower() != "hex8":
        raise ValueError("B-bar formulation is implemented for HEX8 only")
    ne, nn = model.elements.shape
    expand = lambda a: a.to(dtype=model.nodes.dtype, device=model.nodes.device).reshape(-1).expand(ne)
    d = elasticity_matrix_3d(expand(model.young_modulus), expand(model.poisson_ratio))
    ke, b = hex8_bbar_stiffness(model.nodes[model.elements.long()], d)
    ed = torch.stack(tuple(3*model.elements+i for i in range(3)), 2).reshape(ne, 3*nn).long()
    rows = ed[:, :, None].expand(-1, -1, 3*nn).reshape(-1)
    cols = ed[:, None, :].expand(-1, 3*nn, -1).reshape(-1)
    k = torch.zeros((model.n_dofs, model.n_dofs), dtype=model.nodes.dtype, device=model.nodes.device)
    k = k.index_put((rows, cols), ke.reshape(-1), accumulate=True)
    fixed = model.fixed_dofs.to(device=model.nodes.device, dtype=torch.long)
    vals = (torch.zeros_like(fixed, dtype=model.nodes.dtype) if model.prescribed_values is None
            else model.prescribed_values.to(model.nodes))
    mask = torch.ones(model.n_dofs, dtype=torch.bool, device=model.nodes.device)
    mask[fixed] = False
    free = torch.arange(model.n_dofs, device=model.nodes.device)[mask]
    force = model.forces.to(model.nodes)
    u = torch.zeros(model.n_dofs, dtype=model.nodes.dtype, device=model.nodes.device).index_put((fixed,), vals)
    if free.numel():
        rhs = force[free] - k[free][:, fixed] @ vals
        u = u.index_put((free,), torch.linalg.solve(k[free][:, free], rhs))
    strain = torch.einsum("eij,ej->ei", b, u[ed])
    stress = torch.einsum("eij,ej->ei", d, strain)
    return SolidResult(u, k@u-force, strain, stress, .5*u@k@u, k)


def bbar_free_element_eigenvalues(nodes, young=1., poisson=.4999):
    d = elasticity_matrix_3d(nodes.new_tensor([young]), nodes.new_tensor([poisson]))
    k, _ = hex8_bbar_stiffness(nodes.reshape(1, 8, 3), d)
    return torch.linalg.eigvalsh((k[0]+k[0].T)/2)
