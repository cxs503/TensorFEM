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



def test_q4_anisotropic_conductivity_integrates_piecewise_energy():
    nodes = torch.tensor([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]], dtype=D)
    material = torch.tensor([[[3, 1], [1, 2]], [[2, -0.5], [-0.5, 4]]], dtype=D)
    model = ThermalModel(nodes, torch.tensor([[0, 1, 4, 3], [1, 2, 5, 4]]),
                         material, 1., 1., torch.tensor([], dtype=torch.long),
                         nodes.new_empty(0), element_type="q4")
    gradient = nodes.new_tensor([2., -1.])
    temperature = nodes @ gradient
    matrix = assemble_thermal(model).conductivity
    # Each element has unit area, so the integral of grad(T).K.grad(T)
    # is the sum of the two constant energy densities, 10 and 14.
    assert (temperature @ matrix @ temperature).item() == pytest.approx(24.)
    assert torch.allclose(matrix.sum(1), torch.zeros(6, dtype=D), atol=1e-12)


@pytest.mark.parametrize("hot,cold", [(310., 290.), (290., 310.)])
def test_q4_two_layer_contact_jump_and_heat_reactions(hot, cold):
    # Coincident interface coordinates, separate temperature DOFs. Area 1.2
    # is split equally between the two endpoint contacts on the Q4 edge.
    nodes = torch.tensor([[0, 0], [.4, 0], [.4, 1.2], [0, 1.2],
                          [.4, 0], [1, 0], [1, 1.2], [.4, 1.2]], dtype=D)
    model = ThermalModel(nodes, torch.arange(8).reshape(2, 4), nodes.new_tensor([4., 1.]),
                         1., 1., torch.tensor([0, 3, 5, 6]), nodes.new_tensor([hot, hot, cold, cold]),
                         element_type="q4", contact_pairs=torch.tensor([[1, 4], [2, 7]]),
                         contact_conductance=nodes.new_tensor(0.6 / 0.3))
    t = solve_steady_thermal(model)
    q = (hot - cold) / (.4 / 4 + .3 + .6)
    npairs = t[[1, 2]] - t[[4, 7]]
    torch.testing.assert_close(npairs, nodes.new_full((2,), q * .3), atol=1e-11, rtol=0)
    reaction = assemble_thermal(model).conductivity @ t
    assert reaction[[0, 3]].sum().item() == pytest.approx(1.2*q, abs=1e-10)
    assert reaction[[5, 6]].sum().item() == pytest.approx(-1.2*q, abs=1e-10)


def test_contact_exchange_block_and_conductance_gradient():
    from dataclasses import replace
    model = line_model()
    g = torch.tensor([0., 2., 5.], dtype=D, requires_grad=True)
    pairs = torch.tensor([[1, 2], [2, 3], [2, 3]])
    coupled = replace(model, contact_pairs=pairs, contact_conductance=g)
    delta = assemble_thermal(coupled).conductivity - assemble_thermal(model).conductivity
    t = torch.arange(len(model.nodes), dtype=D)**2
    energy = t @ delta @ t
    expected = (t[pairs[:, 0]] - t[pairs[:, 1]])**2
    torch.testing.assert_close(energy, g @ expected)
    torch.testing.assert_close(torch.autograd.grad(energy, g)[0], expected)
    torch.testing.assert_close(delta.sum(1), torch.zeros_like(t))


@pytest.mark.parametrize("theta", [.5, 1.])
def test_sparse_thermal_history_matches_dense_with_contact_and_film(theta):
    from dataclasses import replace

    pytest.importorskip("scipy")
    base = line_model(convection=True, source=1.2)
    model = replace(base, contact_pairs=torch.tensor([[1, 6], [2, 7]]),
                    contact_conductance=torch.tensor([.7, 1.1], dtype=D))
    times = torch.tensor([0., .125, .25, .4, .65], dtype=D)
    initial = torch.arange(9, dtype=D) + 32.
    reference = solve_transient_thermal(model, times, initial, theta)
    result = solve_transient_thermal(model, times, initial, theta, sparse=True)
    for name in ("conductivity", "capacity"):
        actual = getattr(result.matrices, name)
        assert actual.is_sparse
        torch.testing.assert_close(actual.to_dense(), getattr(reference.matrices, name))
    torch.testing.assert_close(result.temperature, reference.temperature, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("theta", [.5, .75, 1.])
def test_integrated_nodal_reactions_include_storage_and_applied_loads(sparse, theta):
    model = line_model(fixed=(0, 8), values=(5., 1.), source=2.)
    time = torch.tensor([0., .01, .035, .1], dtype=D)
    solution = solve_transient_thermal(model, time, torch.zeros(9, dtype=D), theta, sparse=sparse)
    heat = solution.integrated_nodal_reactions()
    increments = solution.nodal_reaction_heat_steps()
    torch.testing.assert_close(increments.sum(dim=0), heat, atol=1e-12, rtol=0.)
    assert increments[:,1:-1].abs().max() < 1e-12
    assert heat[1:-1].abs().max() < 1e-12
    delta = solution.temperature[-1] - solution.temperature[0]
    # Capacity per length = rho*c*A = 42, source per length = 4.
    stored = 42.*torch.trapezoid(delta, model.nodes[:, 0])
    torch.testing.assert_close(heat[[0,-1]].sum(), stored - 8.*time[-1], rtol=0., atol=1e-12)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("theta", [.5, 1.])
def test_ramping_dirichlet_rod_with_exact_storage(sparse, theta):
    # T=1+x+2t, source=rho*c*2. A LINE2 element represents this field exactly.
    model = line_model(fixed=(8,0), values=(3.,1.), source=42.)
    time = torch.tensor([0., .001, .017, .1], dtype=D)
    exact = 1. + model.nodes[:,0] + 2*time[:,None]
    result = solve_transient_thermal(model, time, exact[0], theta, sparse=sparse,
                                     fixed_temperature_history=exact[:,model.fixed_nodes])
    torch.testing.assert_close(result.temperature, exact, rtol=0., atol=2e-14)
    heat = result.integrated_nodal_reactions()
    torch.testing.assert_close(heat[model.fixed_nodes], torch.tensor([1.,-1.], dtype=D),
                               rtol=0., atol=1e-13)


def test_harmonic_rod_matches_complex_affine_solution_and_wall_powers():
    from dataclasses import replace
    from tensorfem import solve_harmonic_thermal

    model = line_model(fixed=(8,0), values=(0.,0.))
    omega = 3.
    x = model.nodes[:,0]
    field = (2.+1j) + (.5-2j)*x
    # Integrate the linear field against each hat exactly, including endpoints.
    heat = torch.zeros_like(field)
    heat[:-1] += 1j*omega*42.*.25*(2*field[:-1]+field[1:])/6
    heat[1:] += 1j*omega*42.*.25*(field[:-1]+2*field[1:])/6
    model = replace(model, fixed_temperatures=field[model.fixed_nodes], nodal_heat=heat)
    result = solve_harmonic_thermal(model, omega)
    torch.testing.assert_close(result.temperature, field, atol=1e-13, rtol=0.)
    expected = torch.zeros_like(field)
    expected[-1], expected[0] = 5.-20j, -5.+20j
    torch.testing.assert_close(result.reaction_power, expected, atol=1e-12, rtol=0.)


def test_harmonic_uniform_driven_rod_complex_source_and_ambient():
    from dataclasses import replace
    from tensorfem import solve_harmonic_thermal

    model = line_model(fixed=(), values=(), convection=True)
    value = 1.5-.7j
    model = replace(model, heat_source=21.*2j*value, ambient_temperature=value)
    result = solve_harmonic_thermal(model, 2.)
    torch.testing.assert_close(result.temperature, torch.full_like(result.temperature, value))
    assert result.reaction_power.abs().max() < 1e-12
    with pytest.raises(ValueError, match="angular_frequency"):
        solve_harmonic_thermal(model, 0.)
