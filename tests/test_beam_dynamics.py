import math
import torch
from tensorfem.advanced_beam import exact_tip_deflection, solve_timoshenko_cantilever
from tensorfem.advanced_buckling import column_buckling, exact_critical_load
from tensorfem.dynamics import mechanical_energy, newmark_linear

def rel(a,b): return abs(a-b)/abs(b)

def test_timoshenko_slender_and_deep_beams_are_locking_free():
    E, nu, b, P = 210e9, .3, .1, -10_000.
    G=E/(2*(1+nu))
    for L,h in ((10.,.1),(.4,.2)):
        A=b*h; I=b*h**3/12
        exact=exact_tip_deflection(L,E,I,G,A,P)
        errors=[]
        for n in (1,2,4):
            result=solve_timoshenko_cantilever(L,n,E,I,G,A,P)
            errors.append(rel(result.displacement[-2].item(),exact))
            assert rel(result.reaction[0].item(),-P) < 1e-8
        assert max(errors) < .03
        assert errors[-1] <= errors[0]+1e-9

def test_all_classical_euler_boundaries_and_mesh_convergence():
    L, EI=3., 1.2e6
    for boundary in ("pinned-pinned","fixed-free","fixed-pinned","fixed-fixed"):
        exact=exact_critical_load(L,EI,boundary)
        coarse=column_buckling(L,4,EI,boundary).load_factors[0].item()
        fine=column_buckling(L,16,EI,boundary).load_factors[0].item()
        assert rel(fine,exact) < .03
        assert rel(fine,exact) <= rel(coarse,exact)+1e-10

def test_newmark_sdof_free_vibration_and_energy_conservation():
    omega=7.; period=2*math.pi/omega; dt=period/100
    time=torch.arange(0.,5*period+dt/2,dt,dtype=torch.float64)
    M=torch.tensor([[2.]],dtype=torch.float64); K=M*omega**2
    force=torch.zeros((len(time),1),dtype=torch.float64)
    result=newmark_linear(M,K,force,time,torch.tensor([.02]),torch.tensor([0.]))
    exact=.02*torch.cos(omega*time)
    assert torch.max(torch.abs(result.displacement[:,0]-exact))/.02 < .03
    energy=mechanical_energy(M,K,result.displacement,result.velocity)
    assert torch.max(torch.abs(energy-energy[0]))/energy[0] < 1e-10
