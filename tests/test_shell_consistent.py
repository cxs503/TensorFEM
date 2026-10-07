import torch

from tensorfem.shell_consistent import (consistent_internal_force_tangent,
                                         shell_energy)
from tensorfem.cylindrical_shell_benchmarks import scordelis_lo_cylindrical

D=torch.float64


def _patch(): return torch.tensor([[0.,-.2],[2.,-.2],[2.,.2],[0.,.2]],dtype=D)


def _state():
    q=torch.tensor([[0.,0.,0.,.002,-.001,.0003],
                    [.001,-.0004,.0007,.003,.0005,-.0002],
                    [.0012,.0002,.0011,.0025,-.0007,.0004],
                    [-.0001,.0001,.0002,.0015,.0003,-.0001]],dtype=D)
    return q.reshape(-1)


def test_consistent_force_and_tangent_against_central_difference():
    p=_patch(); q=_state()
    result=consistent_internal_force_tangent(p,q,5.,70e9,.25,.03)
    direction=torch.linspace(-1.,1.,24,dtype=D); direction/=torch.linalg.vector_norm(direction)
    h=2e-7
    qp=(q+h*direction).requires_grad_(); qm=(q-h*direction).requires_grad_()
    fp=torch.autograd.grad(shell_energy(p,qp,5.,70e9,.25,.03),qp)[0]
    fm=torch.autograd.grad(shell_energy(p,qm,5.,70e9,.25,.03),qm)[0]
    fd=(fp-fm)/(2*h); exact=result.tangent@direction
    assert torch.linalg.vector_norm(fd-exact)/torch.linalg.vector_norm(exact) < 2e-6
    assert torch.linalg.matrix_norm(result.tangent-result.tangent.T)/torch.linalg.matrix_norm(result.tangent) < 2e-10


def test_rigid_state_has_zero_force_and_objective_null_tangent_modes():
    p=_patch(); q=torch.zeros(24,dtype=D)
    result=consistent_internal_force_tangent(p,q,5.,70e9,.25,.03)
    assert torch.linalg.vector_norm(result.internal_force) < 1e-5
    # The exact corotational Hessian has the three translational null modes.
    scale=torch.linalg.matrix_norm(result.tangent)
    for axis in range(3):
        mode=torch.zeros(24,dtype=D); mode.reshape(4,6)[:,axis]=1.
        assert torch.linalg.vector_norm(result.tangent@mode)/(scale*torch.linalg.vector_norm(mode)) < 2e-12


def test_scordelis_lo_recognized_roof_convergence_remains_below_gate():
    values=[]
    for n in (6,8,12):
        r=scordelis_lo_cylindrical(n,n); values.append(r.probe_displacement)
    errors=[abs(v/-0.3024-1.) for v in values]
    assert errors[0] > errors[1] > errors[2]
    assert errors[1] < .03 and errors[2] < .01
