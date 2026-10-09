"""Qualification tests for the metric-consistent cylindrical shell."""
import math
import torch

from tensorfem.cylindrical_shell4 import cylindrical_shell4_stiffness
from tensorfem.cylindrical_shell_benchmarks import scordelis_lo_cylindrical

D = torch.float64


def _patch():
    return torch.tensor([[0.,-.2],[2.,-.2],[2.,.2],[0.,.2]], dtype=D)


def _global_nodes(param, radius):
    return torch.stack((param[:,0], radius*torch.sin(param[:,1]),
                        radius*torch.cos(param[:,1])), dim=1)


def test_rigid_translation_defect_converges_but_blocks_qualification():
    defects=[]
    for span in (.2,.1,.05):
        param=torch.tensor([[0.,-span/2],[2.,-span/2],[2.,span/2],[0.,span/2]],dtype=D)
        K=cylindrical_shell4_stiffness(param,5.,70e9,.25,.03)
        scale=float(torch.linalg.matrix_norm(K)); worst=0.
        for i in range(3):
            q=torch.zeros(24,dtype=D); q.reshape(4,6)[:,i]=1.
            worst=max(worst,abs(float(q@K@q))/(scale*float(q@q)))
        defects.append(worst)
    assert defects[0] > defects[1] > defects[2]
    # It becomes small rapidly, but is deliberately asserted nonzero: nodal
    # local-vector interpolation is not exactly objective on a finite patch.
    assert defects[-1] < 3e-11 and defects[-1] > 1e-14


def test_constant_biaxial_membrane_energy_patch():
    param=_patch(); R=5.; E=70e9; nu=.25; t=.03; ex=1.2e-4; et=-.4e-4
    xyz=_global_nodes(param,R); K=cylindrical_shell4_stiffness(param,R,E,nu,t)
    q=torch.zeros((4,6),dtype=D)
    q[:,0]=ex*param[:,0]
    # A constant radial displacement generates exact circumferential strain.
    q[:,:3] += et*R*torch.stack((torch.zeros(4,dtype=D),
                                 torch.sin(param[:,1]),torch.cos(param[:,1])),1)
    numeric=.5*q.reshape(-1)@K@q.reshape(-1)
    strain=torch.tensor([ex,et,0.],dtype=D)
    C=E/(1-nu**2)*torch.tensor([[1.,nu,0.],[nu,1.,0.],[0.,0.,(1-nu)/2]],dtype=D)
    area=2.*R*.4
    exact=.5*(strain@C@strain)*t*area
    assert abs(float(numeric/exact)-1.) < 1e-12


def test_scordelis_lo_converges_and_qualifies_below_three_percent():
    errors=[]
    for n in (6,8,12):
        r=scordelis_lo_cylindrical(n,n)
        errors.append(abs(r.probe_displacement/-0.3024-1.))
    assert errors[0] > errors[1] > errors[2]
    assert errors[1] < .03
    assert errors[2] < .01


def test_scordelis_lo_vertical_load_balance():
    r=scordelis_lo_cylindrical(8,8)
    reaction=float(r.reaction.reshape(-1,6)[:,2].sum())
    exact=90.*50.*25.*math.radians(80.)
    assert abs(reaction/exact-1.) < .003
