import torch

from tensorfem.shell4 import shell4_stiffness

D = torch.float64


def _facet():
    return torch.tensor([[0.,0.,0.],[2.,0.,0.],[2.,1.,0.],[0.,1.,0.]], dtype=D)


def test_six_rigid_body_modes_have_zero_energy():
    x = _facet(); k = shell4_stiffness(x, 210e9, .3, .02)
    modes = []
    for j in range(3):
        q = torch.zeros(24, dtype=D); q[j::6] = 1.; modes.append(q)
    for j in range(3):
        omega = torch.zeros(3, dtype=D); omega[j] = 1.
        q = torch.zeros(24, dtype=D)
        q.reshape(4,6)[:,:3] = torch.linalg.cross(omega.expand_as(x), x, dim=1)
        q.reshape(4,6)[:,3:] = omega
        modes.append(q)
    scale = torch.linalg.matrix_norm(k)
    for q in modes:
        residual = torch.linalg.vector_norm(k@q)/(scale*torch.linalg.vector_norm(q))
        assert float(residual) < 1e-12


def test_membrane_constant_strain_patch_energy():
    x = _facet(); E, nu, t = 70e9, .25, .03
    k = shell4_stiffness(x, E, nu, t)
    ex, ey, gxy = 1.2e-4, -0.4e-4, .7e-4
    q = torch.zeros(24, dtype=D)
    q.reshape(4,6)[:,0] = ex*x[:,0] + .5*gxy*x[:,1]
    q.reshape(4,6)[:,1] = ey*x[:,1] + .5*gxy*x[:,0]
    numeric = .5*q@k@q
    strain = torch.tensor([ex,ey,gxy], dtype=D)
    C = E/(1-nu**2)*torch.tensor([[1.,nu,0.],[nu,1.,0.],[0.,0.,(1-nu)/2]], dtype=D)
    exact = .5*(strain@C@strain)*t*2.
    assert abs(float(numeric/exact)-1.) < 1e-12


def test_flat_shell_pure_bending_degenerates_to_plate():
    x = _facet(); E, nu, t, curvature = 30e9, .2, .04, 2e-3
    k = shell4_stiffness(x, E, nu, t)
    q = torch.zeros(24, dtype=D)
    # w=.5*kappa*x^2, ry=-dw/dx: exact cylindrical constant curvature.
    q.reshape(4,6)[:,2] = .5*curvature*x[:,0]**2
    q.reshape(4,6)[:,4] = -curvature*x[:,0]
    numeric = .5*q@k@q
    Db = E*t**3/(12*(1-nu**2))
    exact = .5*Db*curvature**2*2.
    assert abs(float(numeric/exact)-1.) < 1e-12


def test_coordinate_rotation_invariance():
    x = _facet(); k0 = shell4_stiffness(x, 10e9, .29, .01)
    axis = torch.tensor([1.,2.,3.], dtype=D); axis /= torch.linalg.vector_norm(axis)
    a = torch.tensor(.63, dtype=D); A = torch.zeros((3,3), dtype=D)
    A[0,1],A[0,2],A[1,0],A[1,2],A[2,0],A[2,1] = -axis[2],axis[1],axis[2],-axis[0],-axis[1],axis[0]
    R = torch.eye(3,dtype=D)+torch.sin(a)*A+(1-torch.cos(a))*(A@A)
    xr=x@R.T; kr=shell4_stiffness(xr,10e9,.29,.01)
    T=torch.zeros((24,24),dtype=D)
    for i in range(4):
        T[6*i:6*i+3,6*i:6*i+3]=R; T[6*i+3:6*i+6,6*i+3:6*i+6]=R
    assert torch.linalg.matrix_norm(kr-T@k0@T.T)/torch.linalg.matrix_norm(k0) < 1e-12

