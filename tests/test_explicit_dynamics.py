import math
import pytest
import torch
from tensorfem.explicit_dynamics import central_difference, critical_time_step

D=torch.float64


def test_sdof_frequency_displacement_and_energy_below_three_percent():
    omega=20.; mass=torch.tensor([[2.]],dtype=D); stiffness=mass*omega**2
    period=2*math.pi/omega; dt=period/100
    time=torch.arange(0.,5*period+dt/2,dt,dtype=D)
    result=central_difference(mass,stiffness,torch.zeros((len(time),1),dtype=D),time,
                              torch.tensor([.01],dtype=D),torch.tensor([0.],dtype=D))
    exact=.01*torch.cos(omega*time)
    assert float(torch.max(torch.abs(result.displacement[:,0]-exact))/.01) < .03
    total=result.kinetic_energy+result.strain_energy
    assert float(torch.max(torch.abs(total-total[0]))/total[0]) < .03
    assert abs(result.critical_time_step-2/omega)/(2/omega) < 1e-12


def test_two_element_axial_bar_first_mode():
    # Uniform fixed-free axial bar, two linear elements and lumped mass.
    E,A,rho,L=70e9,.002,2700.,2.; le=L/2
    ke=E*A/le*torch.tensor([[1.,-1.],[-1.,1.]],dtype=D)
    K=torch.zeros((3,3),dtype=D)
    for e in range(2): K[e:e+2,e:e+2]+=ke
    lump=rho*A*le*torch.tensor([.5,1.,.5],dtype=D)
    M=torch.diag(lump[1:]); Kr=K[1:,1:]
    eig=torch.linalg.eigvalsh(Kr/torch.sqrt(lump[1:,None]*lump[None,1:]))[0]
    numerical=torch.sqrt(eig); exact=math.pi/(2*L)*math.sqrt(E/rho)
    assert abs(float(numerical)-exact)/exact < .03
    assert critical_time_step(M,Kr)>0


def test_explicit_stability_and_lumped_mass_fail_closed():
    M=torch.eye(1,dtype=D); K=torch.tensor([[100.]],dtype=D)
    time=torch.tensor([0.,.21],dtype=D)
    with pytest.raises(ValueError,match="exceeds stable limit"):
        central_difference(M,K,torch.zeros((2,1),dtype=D),time,torch.zeros(1,dtype=D),torch.zeros(1,dtype=D))
    with pytest.raises(ValueError,match="diagonal lumped mass"):
        critical_time_step(torch.tensor([[1.,.1],[.1,1.]],dtype=D),torch.eye(2,dtype=D))
