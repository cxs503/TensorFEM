import torch

from tensorfem.continuum import ContinuumModel, rectangular_q4_mesh, solve_continuum


DTYPE = torch.float64


def test_q4_constant_strain_patch_is_exact():
    nodes, elements = rectangular_q4_mesh(2.0, 1.0, 2, 2, dtype=DTYPE)
    # Affine field: exact constant engineering strain [0.02, -0.01, 0.03].
    ux = 0.02*nodes[:,0] + 0.01*nodes[:,1]
    uy = 0.02*nodes[:,0] - 0.01*nodes[:,1]
    exact_u = torch.stack((ux, uy), 1).reshape(-1)
    boundary = ((nodes[:,0] == 0) | (nodes[:,0] == 2) | (nodes[:,1] == -0.5) | (nodes[:,1] == 0.5))
    fixed = torch.stack((2*torch.where(boundary)[0], 2*torch.where(boundary)[0]+1), 1).reshape(-1)
    model = ContinuumModel(nodes, elements, torch.tensor(210e9), torch.tensor(0.3), torch.tensor(0.1),
                           torch.zeros(len(nodes)*2, dtype=DTYPE), fixed, prescribed_values=exact_u[fixed])
    result = solve_continuum(model)
    error = torch.linalg.vector_norm(result.displacement-exact_u)/torch.linalg.vector_norm(exact_u)
    assert error.item() < 1e-11
    assert torch.allclose(result.strain, torch.tensor([0.02,-0.01,0.03], dtype=DTYPE).expand(len(elements),3), atol=1e-11)


def test_q4_cantilever_tip_deflection_within_three_percent():
    # Slender cantilever, end shear. Euler-Bernoulli is the accepted limiting
    # reference; the 40x8 Q4 mesh makes discretisation + shear error < 3%.
    length, height, thickness = 10.0, 1.0, 0.2
    young, poisson, load = 200e9, 0.3, -1000.0
    nx, ny = 40, 8
    nodes, elements = rectangular_q4_mesh(length, height, nx, ny, dtype=DTYPE)
    forces = torch.zeros(2*len(nodes), dtype=DTYPE)
    right = torch.where(torch.isclose(nodes[:,0], torch.tensor(length, dtype=DTYPE)))[0]
    # Consistent nodal loads for uniform edge traction (trapezoidal rule, exact here).
    forces[2*right+1] = load/ny
    forces[2*right[[0,-1]]+1] *= 0.5
    left = torch.where(nodes[:,0] == 0)[0]
    fixed = torch.stack((2*left, 2*left+1), 1).reshape(-1)
    model = ContinuumModel(nodes, elements, torch.tensor(young), torch.tensor(poisson), torch.tensor(thickness), forces, fixed)
    result = solve_continuum(model)
    tip = result.displacement[2*right[len(right)//2]+1].item()
    inertia = thickness*height**3/12
    reference = load*length**3/(3*young*inertia)
    relative_error = abs(tip-reference)/abs(reference)
    assert relative_error < 0.03, (tip, reference, relative_error)


def test_cst_uniaxial_tension_exact():
    nodes = torch.tensor([[0.,0.],[2.,0.],[2.,1.],[0.,1.]], dtype=DTYPE)
    elements = torch.tensor([[0,1,2],[0,2,3]])
    young, poisson, stress = 70e9, 0.25, 2e6
    exact = torch.stack((stress/young*nodes[:,0], -poisson*stress/young*nodes[:,1]), 1).reshape(-1)
    forces = torch.zeros(8, dtype=DTYPE)
    # Right-edge constant traction, shared equally by its two nodes.
    forces[2*torch.tensor([1,2])] = stress*0.1/2
    # Physical supports only: roller in x along the left edge and one y anchor.
    # All right-edge and Poisson-contraction DOFs remain solved unknowns.
    fixed = torch.tensor([0, 1, 6])
    result = solve_continuum(ContinuumModel(nodes,elements,torch.tensor(young,dtype=DTYPE),torch.tensor(poisson,dtype=DTYPE),torch.tensor(0.1,dtype=DTYPE),forces,fixed,
                                            element_type="cst"))
    displacement_error = torch.linalg.vector_norm(result.displacement-exact)/torch.linalg.vector_norm(exact)
    assert displacement_error.item() < 1e-10
    assert torch.allclose(result.stress, torch.tensor([stress,0.,0.], dtype=DTYPE).expand(2,3), rtol=1e-11, atol=1e-5)
