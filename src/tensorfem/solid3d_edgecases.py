"""Mass, free-body and transient verification helpers for 3-D solids."""
from dataclasses import dataclass
import math
import torch

from .solid3d import (SolidModel, solve_solid, structured_hex_mesh,
                      hex_to_tet_mesh)
from .solid3d_benchmarks import assemble_hex8_consistent_mass
from .dynamics import newmark_linear


def assemble_tet4_consistent_mass(nodes, elements, density):
    """Assemble the exact consistent translational TET4 mass matrix."""
    ne = len(elements)
    x = nodes[elements.long()]
    jac = torch.stack((x[:, 1]-x[:, 0], x[:, 2]-x[:, 0], x[:, 3]-x[:, 0]), 2)
    volume = torch.abs(torch.linalg.det(jac))/6
    rho = torch.as_tensor(density, dtype=nodes.dtype, device=nodes.device).reshape(-1).expand(ne)
    scalar = (torch.ones((4, 4), dtype=nodes.dtype, device=nodes.device)
              + torch.eye(4, dtype=nodes.dtype, device=nodes.device))[None]
    scalar = scalar * (rho*volume/20)[:, None, None]
    me = torch.zeros((ne, 12, 12), dtype=nodes.dtype, device=nodes.device)
    for c in range(3):
        me[:, c::3, c::3] = scalar
    ed = torch.stack(tuple(3*elements+i for i in range(3)), 2).reshape(ne, 12).long()
    rows = ed[:, :, None].expand(-1, -1, 12).reshape(-1)
    cols = ed[:, None, :].expand(-1, 12, -1).reshape(-1)
    mass = torch.zeros((3*len(nodes), 3*len(nodes)), dtype=nodes.dtype, device=nodes.device)
    return mass.index_put((rows, cols), me.reshape(-1), accumulate=True)


def solid_matrices(nodes, elements, young, poisson, density, element_type):
    zeros = torch.zeros(3*len(nodes), dtype=nodes.dtype, device=nodes.device)
    k = solve_solid(SolidModel(nodes, elements, nodes.new_tensor(young),
                    nodes.new_tensor(poisson), zeros, torch.empty(0, dtype=torch.long),
                    element_type)).stiffness
    if element_type.lower() == "tet4":
        m = assemble_tet4_consistent_mass(nodes, elements, density)
    else:
        m = assemble_hex8_consistent_mass(nodes, elements, density)
    return k, m


def generalized_eigenvalues(k, m):
    """Symmetric generalized eigenvalues, including rigid-body values."""
    l = torch.linalg.cholesky(m)
    a = torch.linalg.solve_triangular(l, k, upper=False)
    a = torch.linalg.solve_triangular(l, a.T, upper=False).T
    return torch.linalg.eigvalsh((a+a.T)/2)


def free_body_rigid_modes(element_type="tet4"):
    nodes, hexes = structured_hex_mesh(1., .8, .6, 1, 1, 1)
    elements = hexes
    if element_type == "tet4":
        nodes, elements = hex_to_tet_mesh(nodes, hexes)
    k, m = solid_matrices(nodes, elements, 2.1e11, .27, 7800., element_type)
    eigenvalues = generalized_eigenvalues(k, m)
    scale = eigenvalues[-1]
    return eigenvalues, torch.max(torch.abs(eigenvalues[:6]))/scale


def tet4_axial_frequency_benchmark(nx=12):
    """Fixed-free 3-D TET4 rod compared with the exact axial frequency."""
    length, side, young, density = 4., .2, 70e9, 2700.
    nodes, hexes = structured_hex_mesh(length, side, side, nx, 1, 1)
    nodes, tets = hex_to_tet_mesh(nodes, hexes)
    k, m = solid_matrices(nodes, tets, young, 0., density, "tet4")
    all_nodes = torch.arange(len(nodes))
    left = torch.where(torch.isclose(nodes[:, 0], nodes.new_tensor(0.)))[0]
    fixed = torch.unique(torch.cat((3*left, 3*all_nodes+1, 3*all_nodes+2))).long()
    mask = torch.ones(3*len(nodes), dtype=torch.bool); mask[fixed] = False
    free = torch.arange(3*len(nodes))[mask]
    ev = generalized_eigenvalues(k[free][:, free], m[free][:, free])
    computed = float(torch.sqrt(ev[0]))
    reference = math.pi/(2*length)*math.sqrt(young/density)
    return computed, reference, abs(computed-reference)/reference


def tet4_transient_axial_benchmark(nx=12, periods=3, steps_per_period=160):
    """Integrate the first FE axial mode and compare to its exact cosine response."""
    length, side, young, density = 4., .2, 70e9, 2700.
    nodes, hexes = structured_hex_mesh(length, side, side, nx, 1, 1)
    nodes, tets = hex_to_tet_mesh(nodes, hexes)
    k, m = solid_matrices(nodes, tets, young, 0., density, "tet4")
    all_nodes = torch.arange(len(nodes)); left = torch.where(nodes[:, 0] == 0)[0]
    fixed = torch.unique(torch.cat((3*left, 3*all_nodes+1, 3*all_nodes+2))).long()
    mask = torch.ones(3*len(nodes), dtype=torch.bool); mask[fixed] = False
    free = torch.arange(3*len(nodes))[mask]; kf, mf = k[free][:, free], m[free][:, free]
    l = torch.linalg.cholesky(mf)
    a = torch.linalg.solve_triangular(l, kf, upper=False)
    a = torch.linalg.solve_triangular(l, a.T, upper=False).T
    ev, q = torch.linalg.eigh((a+a.T)/2)
    phi = torch.linalg.solve_triangular(l.T, q[:, :1], upper=True)[:, 0]
    omega = torch.sqrt(ev[0]); period = 2*math.pi/float(omega)
    time = torch.linspace(0., periods*period, periods*steps_per_period+1, dtype=nodes.dtype)
    result = newmark_linear(mf, kf, torch.zeros((len(time), len(free)), dtype=nodes.dtype),
                            time, phi, torch.zeros_like(phi))
    numerical = result.displacement @ (mf @ phi)
    exact = torch.cos(omega*time)
    return float(torch.max(torch.abs(numerical-exact))), time, numerical, exact


def hex8_near_incompressible_locking_diagnostic(nx=4, poisson=.4999):
    """Return stiffness inflation under bending; this is a diagnostic, not a pass benchmark."""
    length, side, young, load = 10., 1., 1e6, -1.
    nodes, elements = structured_hex_mesh(length, side, side, nx, 1, 1)
    forces = torch.zeros(3*len(nodes), dtype=nodes.dtype)
    right = torch.where(nodes[:, 0] == length)[0]
    forces[3*right+2] = load/len(right)
    left = torch.where(nodes[:, 0] == 0)[0]
    fixed = torch.stack((3*left, 3*left+1, 3*left+2), 1).reshape(-1)
    result = solve_solid(SolidModel(nodes, elements, nodes.new_tensor(young),
                         nodes.new_tensor(poisson), forces, fixed, "hex8"))
    computed = abs(float(result.displacement[3*right+2].mean()))
    reference = abs(load)*length**3/(3*young*(side**4/12))
    return computed, reference, abs(computed-reference)/reference
