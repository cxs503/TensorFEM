"""Published/analytical verification cases for the B-bar HEX8."""
import torch
from tensorfem.solid3d import SolidModel, solve_solid, structured_hex_mesh
from tensorfem.solid3d_bbar import solve_solid_bbar


def near_incompressible_cantilever(nx=48, ny=4, poisson=.4999):
    """Slender cantilever: full integration locking vs B-bar on one mesh."""
    length, side, young, load = 10., 1., 1e6, -1.
    nodes, elements = structured_hex_mesh(length, side, side, nx, ny, ny)
    force = torch.zeros(3*len(nodes), dtype=nodes.dtype)
    right = torch.where(nodes[:, 0] == length)[0]
    force[3*right+2] = load/len(right)
    left = torch.where(nodes[:, 0] == 0)[0]
    fixed = torch.stack((3*left, 3*left+1, 3*left+2), 1).reshape(-1)
    model = SolidModel(nodes, elements, nodes.new_tensor(young),
                       nodes.new_tensor(poisson), force, fixed, "hex8")
    standard = abs(float(solve_solid(model).displacement[3*right+2].mean()))
    bbar = abs(float(solve_solid_bbar(model).displacement[3*right+2].mean()))
    reference = abs(load)*length**3/(3*young*(side**4/12))
    return standard, bbar, reference


def near_incompressible_uniaxial_patch(poisson=.4999):
    """Uniform traction cube with exact free-lateral contraction solution."""
    young, traction = 2.0e6, 1000.
    nodes, elements = structured_hex_mesh(1., 1., 1., 2, 2, 2)
    force = torch.zeros(3*len(nodes), dtype=nodes.dtype)
    right = torch.where(nodes[:, 0] == 1.)[0]
    # Consistent nodal resultants for a 2x2 surface grid.
    y, z = nodes[right, 1], nodes[right, 2]
    weights = torch.where(((y == 0.) | (y == 1.)) & ((z == 0.) | (z == 1.)), .0625,
              torch.where((y == .5) & (z == .5), .25, .125)).to(nodes)
    force[3*right] = traction*weights
    # Symmetry planes remove rigid modes while preserving exact Poisson contraction.
    x0 = torch.where(nodes[:, 0] == 0.)[0]
    y0 = torch.where(nodes[:, 1] == 0.)[0]
    z0 = torch.where(nodes[:, 2] == 0.)[0]
    fixed = torch.unique(torch.cat((3*x0, 3*y0+1, 3*z0+2))).long()
    model = SolidModel(nodes, elements, nodes.new_tensor(young), nodes.new_tensor(poisson),
                       force, fixed, "hex8")
    result = solve_solid_bbar(model)
    ux = result.displacement[3*right].mean()
    reference = nodes.new_tensor(traction/young)
    return float(ux), float(reference), abs(float(ux-reference))/float(reference)
