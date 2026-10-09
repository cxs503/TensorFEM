import math

import pytest
import torch

from tensorfem.thermal import (
    ThermalModel, assemble_thermal, solve_steady_thermal,
    solve_thermoelastic_bar, solve_transient_thermal,
)

D = torch.float64


def relative(a, b):
    return abs(float(a)-float(b))/abs(float(b))


def line_model(*, fixed=(0,), values=(100.,), convection=False, source=0.):
    nodes=torch.linspace(0.,2.,9,dtype=D)[:,None]
    elements=torch.stack((torch.arange(8),torch.arange(1,9)),1)
    kw=dict(nodes=nodes,elements=elements,conductivity=torch.tensor(5.,dtype=D),
            density=torch.tensor(3.,dtype=D),specific_heat=torch.tensor(7.,dtype=D),
            thickness=torch.tensor(2.,dtype=D),fixed_nodes=torch.tensor(fixed),
            fixed_temperatures=torch.tensor(values,dtype=D),heat_source=torch.tensor(source,dtype=D))
    if convection:
        kw.update(convection_edges=torch.tensor([[8,8]]),
                  convection_coefficient=torch.tensor(4.,dtype=D),
                  ambient_temperature=torch.tensor(20.,dtype=D))
    return ThermalModel(**kw)


def test_steady_rod_prescribed_temperature_and_flux_reactions():
    model=line_model(fixed=(0,8),values=(100.,40.))
    t=solve_steady_thermal(model)
    exact=100.-30.*model.nodes[:,0]
    assert torch.max(torch.abs(t-exact)) < 1e-11
    m=assemble_thermal(model)
    reactions=m.conductivity@t-m.heat
    assert relative(reactions[0],300.) < 1e-12
    assert relative(reactions[-1],-300.) < 1e-12


def test_steady_rod_convection_matches_series_resistance():
    model=line_model(convection=True)
    t=solve_steady_thermal(model)
    # q = (T0-Tinf)/(L/(kA)+1/(hA)); A=2
    q=(100.-20.)/(2./(5.*2.)+1./(4.*2.))
    exact_right=100.-q*2./(5.*2.)
    assert relative(t[-1],exact_right) < 1e-12


def test_q4_linear_temperature_patch_and_energy_balance():
    nodes=torch.tensor([[0.,0.],[2.,0.],[2.,1.],[0.,1.]],dtype=D)
    exact=50.+12.*nodes[:,0]-7.*nodes[:,1]
    model=ThermalModel(nodes,torch.tensor([[0,1,2,3]]),torch.tensor(9.,dtype=D),
        torch.tensor(2.,dtype=D),torch.tensor(4.,dtype=D),torch.arange(4),exact,
        thickness=torch.tensor(.3,dtype=D),element_type="q4")
    t=solve_steady_thermal(model)
    assert torch.max(torch.abs(t-exact)) < 1e-12
    residual=assemble_thermal(model).conductivity@t
    assert abs(float(residual.sum())) < 1e-12


def test_transient_rod_semidiscrete_analytical_decay_below_three_percent():
    nodes=torch.tensor([[0.],[1.]],dtype=D)
    model=ThermalModel(nodes,torch.tensor([[0,1]]),torch.tensor(2.,dtype=D),
        torch.tensor(3.,dtype=D),torch.tensor(5.,dtype=D),torch.tensor([0]),
        torch.tensor([0.],dtype=D))
    tau=3.*5./(3.*2.)  # C_ff/K_ff = rho cp L^2/(3 k)
    dt=tau/100.; time=torch.arange(0.,tau+dt/2,dt,dtype=D)
    result=solve_transient_thermal(model,time,torch.tensor([0.,100.],dtype=D))
    exact=100.*torch.exp(-time/tau)
    error=torch.max(torch.abs(result.temperature[:,1]-exact))/100.
    assert float(error) < .03
    assert torch.all(result.temperature[1:,1] <= result.temperature[:-1,1])


def test_transient_stability_gate_fails_closed():
    model=line_model(fixed=(0,8),values=(100.,40.))
    with pytest.raises(ValueError,match="unconditional stability"):
        solve_transient_thermal(model,torch.tensor([0.,1.]),torch.ones(9),theta=.49)


def test_sequential_thermal_expansion_free_and_fully_restrained():
    nodes=torch.tensor([[0.],[1.],[2.]],dtype=D); elements=torch.tensor([[0,1],[1,2]])
    temp=torch.full((3,),80.,dtype=D); E=200e9; A=.01; alpha=12e-6; dT=60.
    free=solve_thermoelastic_bar(nodes,elements,E,A,alpha,temp,20.,torch.tensor([0]))
    assert relative(free.displacement[-1],alpha*dT*2.) < 1e-12
    assert torch.max(torch.abs(free.axial_force)) < 1e-5
    held=solve_thermoelastic_bar(nodes,elements,E,A,alpha,temp,20.,torch.tensor([0,2]))
    exact=-E*A*alpha*dT
    assert relative(held.axial_force[0],exact) < 1e-12
    assert relative(held.axial_force[1],exact) < 1e-12
