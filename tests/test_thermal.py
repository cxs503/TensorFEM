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


@pytest.mark.parametrize("sparse", [False, True])
def test_harmonic_rod_matches_complex_affine_solution_and_wall_powers(sparse):
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
    result = solve_harmonic_thermal(model, omega, sparse=sparse)
    torch.testing.assert_close(result.temperature, field, atol=1e-13, rtol=0.)
    expected = torch.zeros_like(field)
    expected[-1], expected[0] = 5.-20j, -5.+20j
    torch.testing.assert_close(result.reaction_power, expected, atol=1e-12, rtol=0.)


@pytest.mark.parametrize("sparse", [False, True])
def test_harmonic_uniform_driven_rod_complex_source_and_ambient(sparse):
    from dataclasses import replace
    from tensorfem import solve_harmonic_thermal

    model = line_model(fixed=(), values=(), convection=True)
    value = 1.5-.7j
    model = replace(model, heat_source=21.*2j*value, ambient_temperature=value)
    result = solve_harmonic_thermal(model, 2., sparse=sparse)
    torch.testing.assert_close(result.temperature, torch.full_like(result.temperature, value))
    assert result.reaction_power.abs().max() < 1e-12
    with pytest.raises(ValueError, match="angular_frequency"):
        solve_harmonic_thermal(model, 0.)


def test_element_flux_on_sheared_cells_uses_global_tensor_and_complex_gradient():
    from tensorfem import thermal_element_flux
    from tensorfem.thermal_analytic import structured_quad_mesh

    mesh = structured_quad_mesh(2, 1, shear=.4)
    xy = mesh.nodes
    k = xy.new_tensor([[[2., .6], [.6, 1.]], [[1., -.3], [-.3, 3.]]])
    model = ThermalModel(xy, mesh.elements, k, 1., 1., torch.arange(0), xy.new_empty(0),
                         element_type="q4")
    gradient = torch.tensor([2.+1j, -.5j], dtype=torch.complex128)
    field = xy.to(gradient) @ gradient
    expected = -(k.to(gradient) @ gradient)
    for xi, eta in ((0., 0.), (-1., .5), (1., -1.)):
        torch.testing.assert_close(thermal_element_flux(model, field, xi=xi, eta=eta), expected)
        history = torch.stack((field,2*field,3*field))
        torch.testing.assert_close(thermal_element_flux(model, history, xi=xi, eta=eta),
                                   torch.stack((expected,2*expected,3*expected)))


def test_line_element_flux_recovers_variable_conductivity():
    from dataclasses import replace
    from tensorfem import thermal_element_flux

    model = line_model()
    k = torch.arange(1,9, dtype=D)
    flux = thermal_element_flux(replace(model, conductivity=k), 5.-3.*model.nodes[:,0])
    torch.testing.assert_close(flux[:,0], 3*k)
    history = torch.stack((model.nodes[:,0], -model.nodes[:,0]))
    result = thermal_element_flux(replace(model, conductivity=k), history)
    torch.testing.assert_close(result[...,0], torch.stack((-k,k)))


@pytest.mark.parametrize("all_fixed", [False, True])
def test_sparse_steady_matches_dense_with_source_and_convection(all_fixed):
    model = (line_model(fixed=tuple(range(9)), values=tuple(range(9)), convection=True, source=3.)
             if all_fixed else line_model(convection=True, source=3.))
    torch.testing.assert_close(solve_steady_thermal(model, sparse=True), solve_steady_thermal(model))
    model.fixed_temperatures.requires_grad_()
    with pytest.raises(ValueError, match="autograd"):
        solve_steady_thermal(model, sparse=True)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("theta", [.5, 1.])
def test_nodal_power_history_drives_free_nodes_and_enters_reaction_heat(sparse, theta):
    model = line_model(fixed=(0, 8), values=(0., 0.), source=2.)
    time = torch.tensor([0., .02, .1, .16], dtype=D)
    x = model.nodes[:, 0]
    exact = time[:, None]**2*(1+x)
    m = assemble_thermal(model)
    # Manufacture loads for the time-discrete equation on nonuniform steps.
    stages = (1-theta)*exact[:-1]+theta*exact[1:]
    storage = (exact.diff(dim=0)/time.diff()[:, None]) @ m.capacity.T
    required = storage+stages @ m.conductivity.T-m.heat
    powers = torch.zeros_like(exact)
    for i in range(len(time)-1):
        powers[i+1] = (required[i]-(1-theta)*powers[i])/theta
    powers[:, 0] += 2.
    result = solve_transient_thermal(model, time, exact[0], theta, sparse=sparse,
                fixed_temperature_history=exact[:, model.fixed_nodes], nodal_heat_history=powers)
    torch.testing.assert_close(result.temperature, exact)
    reactions = torch.zeros_like(required)
    reactions[:, 0] = -2*time.diff()
    torch.testing.assert_close(result.nodal_reaction_heat_steps(), reactions)
    torch.testing.assert_close(result.integrated_nodal_reactions(), reactions.sum(0))


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("interval_fraction",[0.,.4,1.])
def test_nonuniform_rod_element_end_heat_with_local_load_history(sparse,interval_fraction):
    from tensorfem import thermal_element_nodal_heat_steps

    x = torch.tensor([0., .13, .46, .7, 1.], dtype=D)
    ids = torch.arange(4)
    conn = torch.stack((ids, ids+1), 1)
    k = torch.tensor([1.2, 2.1, .8, 3.], dtype=D)
    capacity = torch.tensor([3., 2., 4., 1.], dtype=D)
    fixed = torch.tensor([0, 4])
    time = torch.tensor([0., .02, .11, .15], dtype=D)
    # Unit axial conduction power and a uniform-in-space temperature ramp t².
    initial = torch.cat((x.new_zeros(1), (-x.diff()/k).cumsum(0)))
    exact = initial + time[:, None].square()
    model = ThermalModel(x[:, None], conn, k, 1., capacity, fixed, initial[fixed])
    end_load = time[:, None]*capacity*x.diff()
    local = end_load[:, :, None].expand(-1, -1, 2)
    nodal = torch.zeros_like(exact)
    nodal[:, :-1] += end_load
    nodal[:, 1:] += end_load
    dt = time.diff()[:, None]
    step_heat = interval_fraction*.5*dt*(nodal[:-1]+nodal[1:])
    result = solve_transient_thermal(model, time, initial, .5, sparse=sparse,
                 fixed_temperature_history=exact[:, fixed], nodal_heat_history=(1-interval_fraction)*nodal,
                 nodal_step_heat=step_heat)
    heat = thermal_element_nodal_heat_steps(model, result, element_nodal_heat_history=(1-interval_fraction)*local,
                 element_nodal_step_heat=interval_fraction*.5*dt[:,:,None]*(local[:-1]+local[1:]))
    reference = time.diff()[:, None, None]*x.new_tensor([1., -1.])
    torch.testing.assert_close(result.temperature, exact)
    torch.testing.assert_close(heat, reference.expand_as(heat), rtol=0, atol=2e-15)
    torch.testing.assert_close(heat[:, :-1, 1], -heat[:, 1:, 0], rtol=0, atol=2e-15)


def test_q4_nodal_heat_for_linear_temperature_and_uniform_storage():
    from tensorfem import thermal_element_nodal_heat_steps

    xy = torch.tensor([[0., 0.], [2., 0.], [2.25, 1.], [.25, 1.]], dtype=D)
    ids = torch.arange(4)
    gradient = xy.new_tensor([.8, -.3])
    k = xy.new_tensor([[2., .4], [.4, 1.]])
    time = xy.new_tensor([0., .02, .07])
    exact = (xy @ gradient) + time[:, None]
    model = ThermalModel(xy, ids[None], k[None], 1., 3., ids, exact[0], heat_source=3., element_type="q4")
    result = solve_transient_thermal(model, time, exact[0], .5, fixed_temperature_history=exact)
    # Integrate n·K grad(T) against both linear endpoint weights on each edge.
    edges = xy.roll(-1, 0)-xy
    normals = torch.stack((edges[:, 1], -edges[:, 0]), 1)
    side = normals @ (k @ gradient)
    inward = .5*(side+side.roll(1))
    torch.testing.assert_close(thermal_element_nodal_heat_steps(model, result)[:, 0],
                               time.diff()[:, None]*inward, rtol=0, atol=2e-15)


def test_harmonic_element_power_with_complex_volume_and_nodal_heat():
    from tensorfem import thermal_element_nodal_power

    x = torch.tensor([0., .2, 1.], dtype=D)
    conn = torch.tensor([[0, 1], [1, 2]])
    temp = (1+.7j)*x + .3j
    k = x.new_tensor([2., 3.])
    cp = x.new_tensor([4., 5.])
    volume_source = torch.tensor([1j, 2.-.3j], dtype=temp.dtype)
    extra = torch.tensor([[.1+.3j, -.1j], [2., -.7j]], dtype=temp.dtype)
    fixed = torch.tensor([0, 2])
    model = ThermalModel(x[:, None], conn, k, 1., cp, fixed, temp[fixed], heat_source=volume_source)
    power = thermal_element_nodal_power(model, temp, 7., element_nodal_heat=extra)
    # Exact linear-field storage integrated against each endpoint's hat.
    left, right = temp[:-1], temp[1:]
    stored = cp[:, None]*x.diff()[:, None]*torch.stack((2*left+right, left+2*right), 1)/6
    flux = k*(1+.7j)
    expected = (7j*stored + torch.stack((-flux, flux), 1)
                - (x.diff()*volume_source/2)[:, None] - extra)
    torch.testing.assert_close(power, expected, atol=2e-14, rtol=0)
    with pytest.raises(ValueError, match="element_nodal_heat"):
        thermal_element_nodal_power(model, temp, 7., element_nodal_heat=extra[0])


@pytest.mark.parametrize("sparse", [False, True])
def test_concentrated_capacity_on_contact_only_store(sparse):
    from tensorfem import solve_harmonic_thermal

    xy = torch.tensor([[0.], [1.], [.5]], dtype=D)
    model = ThermalModel(xy, torch.tensor([[0,1]]), 1., 1., 1., torch.tensor([0,1]),
                        xy.new_tensor([1., 0.]), contact_pairs=torch.tensor([[0,2],[2,1]]),
                        contact_conductance=xy.new_tensor([2., 4.]), nodal_heat_capacity=xy.new_tensor([0.,0.,3.]))
    harmonic = solve_harmonic_thermal(model, 5., sparse=sparse)
    assert abs(complex(harmonic.temperature[2])-2/(6+15j)) < 1e-14
    time = torch.linspace(0., 1., 129, dtype=D)
    transient = solve_transient_thermal(model, time, xy.new_zeros(3), .5, sparse=sparse)
    exact = (1-torch.exp(-2*time))/3
    torch.testing.assert_close(transient.temperature[:, 2], exact, rtol=0, atol=3e-6)


@pytest.mark.parametrize("sparse", [False, True])
def test_interpolated_volumetric_heat_integrates_distinct_local_traces(sparse):
    from tensorfem import thermal_element_nodal_power

    x = torch.tensor([[0.], [.2], [1.]], dtype=D)
    conn = torch.tensor([[0, 1], [1, 2]])
    s = torch.tensor([[1.+.3j, -2j], [.8j, 2.]], dtype=torch.complex128)
    model = ThermalModel(x, conn, 1., 1., 1., torch.tensor([0,2]), x.new_zeros(2),
                         heat_source=s, thickness=x.new_tensor([2., 3.]))
    local = torch.stack((2*s[:,0]+s[:,1], s[:,0]+2*s[:,1]), 1)
    local = local*x.new_tensor([.4, 2.4])[:, None]/6
    assembled = assemble_thermal(model, sparse=sparse, complex_loads=True)
    expected = torch.stack((local[0,0], local[0,1]+local[1,0], local[1,1]))
    torch.testing.assert_close(assembled.heat, expected)
    torch.testing.assert_close(thermal_element_nodal_power(model, x[:,0]*0., 2.), -local)


def test_q4_bilinear_corner_heat_density_integrates_exact_hat_weights():
    x = torch.tensor([[0.,0.], [2.,0.], [2.25,1.], [.25,1.]], dtype=D)
    model = ThermalModel(x, torch.arange(4)[None], 1., 1., 1., torch.tensor([0]),
                         x.new_zeros(1), element_type="q4", heat_source=x.new_tensor([[0.,0.,1.,0.]]))
    torch.testing.assert_close(assemble_thermal(model).heat, x.new_tensor([1.,2.,4.,2.])/18)


def test_q4_trapezoid_edge_power_has_outward_sign_thickness_and_phase():
    from tensorfem import thermal_element_edge_power

    xy = torch.tensor([[0.,0.], [2.,0.], [1.6,1.], [.2,.8]], dtype=D)
    material = xy.new_tensor([[2., .3], [.3, 1.]])
    field = (1+.5j)*xy[:,0]+(.2-.7j)*xy[:,1]
    model = ThermalModel(xy, torch.arange(4)[None], material[None], 1., 1., torch.tensor([0]),
                         field[:1], element_type="q4", thickness=.4)
    flux = -material.to(field) @ field.new_tensor([1+.5j, .2-.7j])
    # Length-weighted outward normals, in local edge order.
    expected = .4*(field.new_tensor([[0.,-2.],[1.,.4],[-.2,1.4],[-.8,.2]]) @ flux)
    torch.testing.assert_close(thermal_element_edge_power(model,field)[0], expected)
    amplitudes = field.new_tensor([[1.,2j],[-.5j,.2],[3.,-2.]])
    history = amplitudes[...,None]*field
    torch.testing.assert_close(thermal_element_edge_power(model,history),
                               amplitudes[...,None,None]*expected[None,:])


@pytest.mark.parametrize("sparse", [False, True])
def test_consistent_wall_capacity_includes_tangential_variation_and_area(sparse):
    from dataclasses import replace
    model = ThermalModel(torch.tensor([[0.,0.],[2.,0.],[2.,1.],[0.,1.]],dtype=D),
                         torch.tensor([[0,1,2,3]]),1.,1.,1.,torch.empty(0,dtype=torch.long),
                         torch.empty(0,dtype=D),thickness=3.,element_type="q4",
                         convection_edges=torch.tensor([[0,1],[1,2]]),
                         convection_coefficient=0.,ambient_temperature=0.)
    base = assemble_thermal(model,sparse=sparse).capacity.to_dense()
    storage = assemble_thermal(replace(model,wall_areal_heat_capacity=torch.tensor([2.,4.])),
                               sparse=sparse).capacity.to_dense()-base
    expected = torch.tensor([[4.,2.,0.,0.],[2.,8.,2.,0.],[0.,2.,4.,0.],[0.,0.,0.,0.]],dtype=D)
    torch.testing.assert_close(storage,expected)
    line = line_model(convection=True)
    increment = (assemble_thermal(replace(line,wall_areal_heat_capacity=2.), sparse=sparse).capacity.to_dense()
                 - assemble_thermal(line,sparse=sparse).capacity.to_dense())
    assert increment[-1,-1] == pytest.approx(4.)
    with pytest.raises(ValueError,match="nonnegative"):
        assemble_thermal(replace(line,wall_areal_heat_capacity=-1.), sparse=sparse)


def test_endpoint_ambient_load_preserves_edge_orientation_and_complex_phase():
    from dataclasses import replace
    nodes = torch.tensor([[0.,0.],[1.,0.],[1.,1.],[0.,1.]],dtype=D)
    ambient = torch.tensor([[1j, 3.+2j]],dtype=torch.complex128)
    model = ThermalModel(nodes,torch.tensor([[0,1,2,3]]),1.,1.,1.,torch.tensor([],dtype=torch.long),
                         torch.zeros(0,dtype=D),element_type="q4",convection_edges=torch.tensor([[0,1]]),
                         convection_coefficient=6.,ambient_temperature=ambient)
    heat = assemble_thermal(model,complex_loads=True).heat
    torch.testing.assert_close(heat[:2],torch.tensor([3.+4j,6.+5j],dtype=ambient.dtype))
    reverse = replace(model,convection_edges=model.convection_edges.flip((1,)),ambient_temperature=ambient.flip((1,)))
    torch.testing.assert_close(assemble_thermal(reverse,sparse=True,complex_loads=True).heat,heat)


@pytest.mark.parametrize("sparse", [False, True])
def test_surface_measure_uses_element_owner_with_reordered_edges(sparse):
    from dataclasses import replace
    nodes = torch.tensor([[0.,0.],[1.,0.],[3.,0.],[0.,1.],[1.,1.],[3.,1.]],dtype=D)
    model = ThermalModel(nodes,torch.tensor([[1,2,5,4],[0,1,4,3]]),1.,1.,1.,
                         torch.empty(0,dtype=torch.long),torch.empty(0,dtype=D),
                         element_type="q4",thickness=torch.tensor([5.,2.],dtype=D))
    edges = torch.tensor([[3,0],[5,4],[0,1]])
    bc = replace(model,convection_edges=edges,convection_coefficient=3.,ambient_temperature=4.,
                 wall_areal_heat_capacity=2.)
    original = assemble_thermal(model,sparse=sparse)
    with_edges = assemble_thermal(bc,sparse=sparse)
    # Left, top right and lower left have areas 2, 10 and 2, respectively.
    area_matrix = nodes.new_zeros((6,6))
    for (a,b),area in zip(edges.tolist(), [2.,10.,2.]):
        area_matrix[a,a] += area/3
        area_matrix[b,b] += area/3
        area_matrix[a,b] += area/6
        area_matrix[b,a] += area/6
    torch.testing.assert_close(with_edges.capacity.to_dense()-original.capacity.to_dense(),2*area_matrix)
    torch.testing.assert_close(with_edges.conductivity.to_dense()-original.conductivity.to_dense(),3*area_matrix)
    torch.testing.assert_close(with_edges.heat,12*area_matrix.sum(dim=1))
    with pytest.raises(ValueError,match="unique owning element"):
        assemble_thermal(replace(bc,convection_edges=torch.tensor([[1,4]])),sparse=sparse)


def test_line_endpoint_area_follows_connectivity_instead_of_edge_index():
    from dataclasses import replace
    model = replace(line_model(convection=True),thickness=torch.arange(1,9,dtype=D),
                    convection_edges=torch.tensor([[8,8],[0,0]]),wall_areal_heat_capacity=2.)
    base = replace(model,convection_edges=None,wall_areal_heat_capacity=None)
    delta = assemble_thermal(model).capacity-assemble_thermal(base).capacity
    torch.testing.assert_close(delta.diag()[[8,0]],torch.tensor([16.,2.],dtype=D))


def test_internal_shoulder_assembles_only_exposed_area():
    from dataclasses import replace
    line = replace(line_model(convection=True),thickness=torch.arange(1,9,dtype=D),
                   convection_edges=torch.tensor([[4,4]]),convection_area=.3,
                   wall_areal_heat_capacity=6.,convection_coefficient=3.,ambient_temperature=10.)
    base = replace(line,convection_edges=None,convection_area=None,wall_areal_heat_capacity=None)
    a, b = assemble_thermal(line), assemble_thermal(base)
    assert float(a.heat[4]) == pytest.approx(9.)
    assert float((a.capacity-b.capacity).sum()) == pytest.approx(1.8)
    assert float((a.conductivity-b.conductivity).sum()) == pytest.approx(.9)
    with pytest.raises(ValueError,match="nonnegative"):
        assemble_thermal(replace(line,convection_area=-1.))


@pytest.mark.parametrize("sparse",[False,True])
def test_dirichlet_jump_conserves_free_heat_and_resolves_wall_impulse(sparse):
    from tensorfem import apply_thermal_temperature_jump
    x = torch.tensor([[0.],[.5],[1.]],dtype=torch.float64)
    fixed = torch.tensor([2,0])
    model = ThermalModel(x,torch.tensor([[0,1],[1,2]]),1.,1.,1.,fixed,x.new_zeros(2),
                         nodal_heat_capacity=x.new_tensor([.1,0.,.2]))
    before = x.new_tensor([.3,.8,.6])
    target = x.new_tensor([.4,1.3])
    after,impulse,local = apply_thermal_temperature_jump(model,before,target,sparse=sparse,
                                                       return_element_impulse=True)
    torch.testing.assert_close(local,x.new_tensor([[1.8,.6],[-.6,-.6]])/12)
    # C_ff=1/3 and each end couples to the middle with C_fc=1/12.
    expected_delta = x.new_tensor([1.,-.2,-.2])
    torch.testing.assert_close(after,before+expected_delta)
    torch.testing.assert_close(impulse,x.new_tensor([1/6+.1-.2/12,0.,-.2*(1/6+.2)-.2/12]))
    restored,reverse = apply_thermal_temperature_jump(model,after,before[fixed],sparse=sparse)
    torch.testing.assert_close(restored,before)
    torch.testing.assert_close(reverse,-impulse)


@pytest.mark.parametrize("element",["line2","q4"])
def test_bulk_heat_loss_matches_unit_capacity_mass(element):
    from dataclasses import replace
    if element=="line2":
        model = line_model()
    else:
        nodes = torch.tensor([[0.,0.],[2.,.1],[1.8,1.5],[0.,1.]],dtype=D)
        model = ThermalModel(nodes,torch.tensor([[0,1,2,3]]),2.,3.,1.,torch.tensor([0]),
                             nodes.new_zeros(1),thickness=.8,element_type="q4")
    h = torch.linspace(.5,2.,len(model.elements),dtype=D)
    reference = assemble_thermal(replace(model,density=h,specific_heat=1.))
    cooled = assemble_thermal(replace(model,heat_loss_coefficient=h))
    torch.testing.assert_close(cooled.conductivity-reference.conductivity,reference.capacity)


@pytest.mark.parametrize("invalid",[-1.,float("nan"),1j,[1.,2.,3.]])
def test_invalid_bulk_heat_loss(invalid):
    from dataclasses import replace
    with pytest.raises(ValueError,match="heat_loss_coefficient"):
        assemble_thermal(replace(line_model(),heat_loss_coefficient=invalid))


def test_distorted_q4_nodal_bulk_loss_against_physical_integral():
    from dataclasses import replace
    import numpy as np
    coords = np.array([[0.,0.],[1.8,.2],[1.3,1.5],[-.2,.8]])
    nodes = torch.tensor(coords,dtype=D)
    h = np.array([.4,1.3,2.8,.1])
    model = ThermalModel(nodes,torch.tensor([[0,1,2,3]]),1.,1.,1.,torch.tensor([0]),nodes.new_zeros(1),
                         element_type="q4",thickness=.7)
    k = assemble_thermal(replace(model,heat_loss_coefficient=[h.tolist()])).conductivity
    loss = (k-assemble_thermal(model).conductivity).numpy()
    corners = np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
    rule,weights = np.polynomial.legendre.leggauss(6)
    expected = np.zeros((4,4))
    for xi,wx in zip(rule,weights):
        for eta,wy in zip(rule,weights):
            fx,fy = (1+corners*[xi,eta]).T
            shape = fx*fy/4
            grad = np.column_stack((corners[:,0]*fy,corners[:,1]*fx))/4
            expected += .7*wx*wy*np.linalg.det(coords.T @ grad)*(h @ shape)*np.outer(shape,shape)
    np.testing.assert_allclose(loss,expected,atol=1e-14,rtol=0.)


@pytest.mark.parametrize("sparse",[False,True])
def test_low_frequency_rod_matches_integrated_heat_capacity(sparse):
    from dataclasses import replace
    from tensorfem.thermal import solve_low_frequency_thermal
    drive = 1.+.4j
    model = replace(line_model(),fixed_temperatures=torch.tensor([drive],dtype=torch.complex128))
    result = solve_low_frequency_thermal(model,sparse=sparse)
    x = model.nodes[:,0]
    expected = (3*7/5)*drive*(x*x/2-2*x)
    torch.testing.assert_close(result.steady.temperature,torch.full_like(expected,drive))
    torch.testing.assert_close(result.first_order.temperature,expected)
    power = torch.zeros_like(expected)
    power[0] = 3*7*2*2*drive
    torch.testing.assert_close(result.first_order.reaction_power,power)


def test_dense_low_frequency_capacity_gradient():
    from dataclasses import replace
    from tensorfem.thermal import solve_low_frequency_thermal
    capacity = torch.tensor(1.5,dtype=D,requires_grad=True)
    model = replace(line_model(),specific_heat=capacity)
    lag = solve_low_frequency_thermal(model).first_order.temperature[-1].real
    derivative, = torch.autograd.grad(lag,capacity)
    torch.testing.assert_close(derivative,lag.detach()/capacity.detach())


@pytest.mark.parametrize("sparse",[False,True])
def test_element_low_frequency_power_with_complex_volume_heat(sparse):
    from dataclasses import replace
    from tensorfem.thermal import solve_low_frequency_thermal, thermal_element_low_frequency_power
    model = replace(line_model(fixed=(0,8),values=(1.,.2)),heat_source=torch.tensor(.6+.8j,dtype=torch.complex128),
                    heat_loss_coefficient=1.4)
    result = solve_low_frequency_thermal(model,sparse=sparse)
    steady,lag = thermal_element_low_frequency_power(model,result)
    for local,coeff in ((steady,result.steady),(lag,result.first_order)):
        assembled = torch.zeros_like(coeff.temperature).scatter_add(0,model.elements.flatten(),local.flatten())
        torch.testing.assert_close(assembled,coeff.reaction_power)


@pytest.mark.parametrize("sparse",[False,True])
def test_sdirk2_stiff_rod_mode_and_interval_reactions(sparse):
    model = line_model(fixed=(0,8),values=(0.,0.))
    m = assemble_thermal(model)
    # Uniform LINE2 rod: sine vectors diagonalize both consistent matrices.
    initial = torch.sin(7*math.pi*model.nodes[:,0]/2)
    rate = (initial@m.conductivity@initial)/(initial@m.capacity@initial)
    time = torch.tensor([0.,1.,601.],dtype=D)
    result = solve_transient_thermal(model,time,initial,method="sdirk2",sparse=sparse)
    gamma = 1-1/math.sqrt(2)
    state = initial
    for i,z in enumerate(time.diff()*rate):
        first = state/(1+gamma*z)
        state = state*(1-(1-2*gamma)*z)/(1+gamma*z)**2
        torch.testing.assert_close(result.temperature[i+1],state,atol=1e-13,rtol=0.)
        torch.testing.assert_close(result.interval_temperature[i],(1-gamma)*first+gamma*state,atol=1e-13,rtol=0.)
    assert abs(result.nodal_reaction_heat_steps()[:,1:-1]).max()<1e-12


@pytest.mark.parametrize("sparse",[False,True])
def test_sdirk2_moving_temperatures_and_independent_heat_inputs(sparse):
    model = line_model(fixed=(0,8),values=(0.,0.))
    m = assemble_thermal(model)
    time = torch.tensor([0.,.04,.15,.23],dtype=D)
    slope = 1+model.nodes[:,0]
    exact = time[:,None]*slope
    # Supply the storage load as integrated heat and conduction as endpoint power.
    heat = time.diff()[:,None]*(m.capacity@slope)
    powers = exact@m.conductivity.T
    heat[:,0] += .25
    result = solve_transient_thermal(model,time,exact[0],method="sdirk2",sparse=sparse,
        fixed_temperature_history=exact[:,model.fixed_nodes],nodal_heat_history=powers,nodal_step_heat=heat)
    torch.testing.assert_close(result.temperature,exact,atol=1e-13,rtol=0.)
    reactions = torch.zeros_like(heat)
    reactions[:,0] = -.25
    torch.testing.assert_close(result.nodal_reaction_heat_steps(),reactions,rtol=0.,atol=1e-13)
    torch.testing.assert_close(result.integrated_nodal_reactions(),reactions.sum(0),rtol=0.,atol=1e-13)


@pytest.mark.parametrize("sparse",[False,True])
@pytest.mark.parametrize("method",["theta","sdirk2"])
def test_callable_heating_matches_time_quadrature_and_energy(sparse,method):
    model = line_model(fixed=tuple(),values=tuple())
    ones = torch.ones(len(model.nodes),dtype=D)
    mass_load = assemble_thermal(model).capacity@ones
    times = torch.tensor([0.,.05,.16,.3],dtype=D)
    result = solve_transient_thermal(model,times,0*ones,theta=.5,method=method,sparse=sparse,
                                     nodal_heat_history=lambda t:3*t*t*mass_load)
    gamma = 1-1/math.sqrt(2)
    first = times[:-1]+gamma*times.diff()
    power = ((1-gamma)*3*first**2+gamma*3*times[1:]**2 if method=="sdirk2"
             else 1.5*(times[:-1]**2+times[1:]**2))
    exact = (times.diff()*power).cumsum(0)
    torch.testing.assert_close(result.temperature[1:],exact[:,None]*ones,atol=1e-14,rtol=0.)
    torch.testing.assert_close(result.additional_stage_power,power[:,None]*mass_load)
    assert abs(result.nodal_reaction_heat_steps()).max()<1e-13


@pytest.mark.parametrize("sparse",[False,True])
@pytest.mark.parametrize("fully_fixed",[False,True])
def test_callable_dirichlet_stages_follow_fixed_node_order(sparse,fully_fixed):
    ids = tuple(reversed(range(9))) if fully_fixed else (8,0)
    model = line_model(fixed=ids,values=(0.,)*len(ids))
    times = torch.tensor([0.,.07,.21],dtype=D)
    shape = 1+model.nodes[:,0]
    def boundary(t):
        return t.square()*shape[model.fixed_nodes]
    result = solve_transient_thermal(model,times,0*shape,method="sdirk2",sparse=sparse,
                                     fixed_temperature_history=boundary)
    gamma = 1-1/math.sqrt(2)
    first = times[:-1]+gamma*times.diff()
    avg = (1-gamma)*first.square()+gamma*times[1:].square()
    torch.testing.assert_close(result.interval_temperature[:,model.fixed_nodes],avg[:,None]*shape[model.fixed_nodes])
    torch.testing.assert_close(result.temperature[:,model.fixed_nodes],times[:,None].square()*shape[model.fixed_nodes])
    heat = result.nodal_reaction_heat_steps()
    torch.testing.assert_close(heat.sum(0),result.integrated_nodal_reactions())
    if not fully_fixed:
        assert heat[:,1:-1].abs().max()<1e-13


@pytest.mark.parametrize("element_type", ["line2", "q4"])
def test_transient_section_power_from_quadratic_slab_with_storage(element_type):
    from tensorfem import thermal_element_nodal_heat_steps, thermal_element_section_power

    x = torch.tensor([0., .13, .46, .7, 1.], dtype=D)
    i = torch.arange(len(x)-1)
    area, k, capacity, rate = 1.3, 2.4, 1.7, .6
    if element_type=="line2":
        nodes,conn = x[:,None],torch.stack((i,i+1),1)
        fixed = torch.tensor([0,len(x)-1])
    else:
        nodes = torch.stack((x.repeat(2),torch.cat((0*x,0*x+1))),1)
        conn = torch.stack((i,i+1,i+1+len(x),i+len(x)),1)
        fixed = torch.tensor([0,len(x)-1,len(x),len(nodes)-1])
    initial = 1.+nodes[:,0]*(.4-.3*nodes[:,0])
    model = ThermalModel(nodes,conn,k,1.,capacity,fixed,initial[fixed],thickness=area,
                         heat_source=capacity*rate+.6*k,element_type=element_type)
    times = x.new_tensor([0.,.03,.12,.2])
    expected = initial+rate*times[:,None]
    result = solve_transient_thermal(model,times,initial,theta=.5,fixed_temperature_history=expected[:,fixed])
    f = x.new_tensor([0.,.19,.57,1.])
    local = thermal_element_nodal_heat_steps(model,result)/times.diff()[:,None,None]
    power = thermal_element_section_power(model,local,f)
    analytic = -k*area*(.4-.6*(x[:-1,None]+x.diff()[:,None]*f))
    torch.testing.assert_close(power,analytic.expand_as(power),atol=2e-13,rtol=0.)
    torch.testing.assert_close(result.temperature,expected,atol=2e-13,rtol=0.)
    phase = .2+.7j
    torch.testing.assert_close(thermal_element_section_power(model,phase*local,f),phase*power)
    with pytest.raises(ValueError,match="fractions"):
        thermal_element_section_power(model,local,[1.1])


@pytest.mark.parametrize("element_type", ["line2","q4"])
def test_section_power_with_spatially_quadratic_net_source(element_type):
    from tensorfem import thermal_element_nodal_power, thermal_element_section_power

    length, area, k = 1.4, 1.7, 2.1
    xy = torch.tensor([[0.,0.],[length,0.],[length,1.],[0.,1.]],dtype=D)
    nodes = xy[:2,:1] if element_type=="line2" else xy
    fixed = torch.arange(len(nodes))
    x = nodes[:,0]
    # P(x)=-k*A*T'(x)=.7+.4*x²-.2*x³. The prescribed steady volume
    # heating P'(x)/A contains linear and quadratic terms.
    temperature = 1.-(.7*x+.4*x**3/3-.05*x**4)/(k*area)
    model = ThermalModel(nodes,fixed[None],k,1.,1.,fixed,temperature,thickness=area,
                         heat_source=(.8*x/area)[None],element_type=element_type,
                         heat_source_quadratic=-.6/area if element_type=="line2" else [-.6/area,0.,0.])
    local = thermal_element_nodal_power(model,temperature)
    f = x.new_tensor([0.,.12,.5,.81,1.])
    power = thermal_element_section_power(model,local,f,net_line_source_gradient=.8-.6*length,
                                           net_line_source_curvature=-1.2)
    ref = .7+.4*(length*f)**2-.2*(length*f)**3
    torch.testing.assert_close(power[0],ref.to(power),atol=1e-13,rtol=0.)
    # Integrate the two shape-weighted sources independently at Gauss points.
    gauss = x.new_tensor([.5-(3/5)**.5/2,.5,.5+(3/5)**.5/2])
    weighted = length*x.new_tensor([5.,8.,5.])/18*(.8*length*gauss-.6*(length*gauss)**2)
    west,east = weighted@(1-gauss),weighted@gauss
    load = torch.stack([west,east] if element_type=="line2" else [west/2,east/2,east/2,west/2])[None]
    recovered = thermal_element_section_power(model,local,f,net_element_nodal_input=load,
                                               net_line_source_curvature=-1.2)
    torch.testing.assert_close(recovered,power,atol=1e-13,rtol=0.)
    with pytest.raises(ValueError,match="broadcast"):
        thermal_element_section_power(model,local,f,net_line_source_gradient=[1.,2.])


@pytest.mark.parametrize("element_type",["line2","q4"])
def test_element_net_input_separates_storage_and_graded_bulk_loss(element_type):
    from tensorfem import thermal_element_nodal_heat_steps, thermal_element_nodal_power

    xy = torch.tensor([[0.,0.],[1.,0.],[1.,1.],[0.,1.]],dtype=D)
    nodes = xy[:2,:1] if element_type=="line2" else xy
    x = nodes[:,0]
    ids = torch.arange(len(nodes))
    model = ThermalModel(nodes,ids[None],1.,1.,2.,ids,1+x,element_type=element_type,
                         heat_loss_coefficient=(.3+.8*x)[None],heat_source=(2-x)[None])
    gauss = x.new_tensor([.5-(3/5)**.5/2,.5,.5+(3/5)**.5/2])
    weights = x.new_tensor([5.,8.,5.])/18
    def project(density):
        left,right = (weights*(1-gauss))@density.real,(weights*gauss)@density.real
        if density.is_complex():
            left = left+1j*((weights*(1-gauss))@density.imag)
            right = right+1j*((weights*gauss)@density.imag)
        return torch.stack([left,right] if element_type=="line2" else [left/2,right/2,right/2,left/2])[None]
    omega,phase = 2.4,.4+.7j
    extra = (.3-.1j)*x[None]
    _,net = thermal_element_nodal_power(model,phase*(1+x),omega,element_nodal_heat=extra,return_net_input=True)
    target = project(2-gauss-(.3+.8*gauss+2j*omega)*phase*(1+gauss))+extra
    torch.testing.assert_close(net,target,rtol=0.,atol=3e-15)

    times = x.new_tensor([0.,.09,.25])
    history = 1+x+times[:,None]*(.5-x)
    result = solve_transient_thermal(model,times,history[0],theta=.5,fixed_temperature_history=history)
    supplemental = (times[:,None,None]*x[None,None,:])
    added_heat = torch.full_like(supplemental[:-1],.013)
    _,net_heat = thermal_element_nodal_heat_steps(model,result,element_nodal_heat_history=supplemental,
            element_nodal_step_heat=added_heat,return_net_input=True)
    for i,dt in enumerate(times.diff()):
        mid = (times[i]+times[i+1])/2
        target = project(2-gauss-2*(.5-gauss)-(.3+.8*gauss)*(1+gauss+mid*(.5-gauss)))
        torch.testing.assert_close(net_heat[i],dt*(target+mid*x)+added_heat[i],rtol=0.,atol=3e-15)


@pytest.mark.parametrize("sparse",[False,True])
def test_insulated_components_have_independent_energy_gauges(sparse):
    from dataclasses import replace
    from tensorfem.thermal import solve_low_frequency_thermal

    load = torch.tensor([1.,-1.,2.,-2.],dtype=torch.complex128)*(1+.3j)
    model = ThermalModel(torch.tensor([[0.],[1.],[1.],[2.]],dtype=D),torch.tensor([[0,1],[2,3]]),
        1.,1.,1.,torch.empty(0,dtype=torch.long),torch.empty(0,dtype=D),
        nodal_heat=load,nodal_heat_capacity=torch.tensor([1.,0.,0.,2.],dtype=D),
        contact_pairs=torch.tensor([[1,2]]),contact_conductance=0.)
    for g in (0.,3.):
        case = replace(model,contact_conductance=g)
        result = solve_low_frequency_thermal(case,sparse=sparse)
        for coeff in (result.steady,result.first_order):
            energy = torch.tensor([1.5,.5,.5,2.5],dtype=D)*coeff.temperature
            groups = energy.reshape(2,2).sum(1) if g == 0. else energy.sum()
            torch.testing.assert_close(groups,torch.zeros_like(groups),atol=1e-13,rtol=0.)
            torch.testing.assert_close(coeff.reaction_power,torch.zeros_like(load),atol=1e-13,rtol=0.)
        steady = result.steady.temperature
        torch.testing.assert_close(steady[[0,2]]-steady[[1,3]],load[[0,2]])
    with pytest.raises(ValueError,match="each insulated component"):
        solve_low_frequency_thermal(replace(model,nodal_heat=load[[0,2,1,3]]),sparse=sparse)


def test_insulated_dense_capacity_gradient():
    from dataclasses import replace
    from tensorfem.thermal import solve_low_frequency_thermal

    model = line_model(fixed=(),values=())
    c = torch.tensor(2.,dtype=D,requires_grad=True)
    load = torch.zeros(9,dtype=D)
    load[0],load[-1] = 1.,-1.
    lag = solve_low_frequency_thermal(replace(model,specific_heat=c,nodal_heat=load)).first_order.temperature[0].real
    gradient, = torch.autograd.grad(lag,c)
    torch.testing.assert_close(gradient,lag.detach()/c.detach())


def test_recovered_corner_edges_integrate_quadratic_on_graded_sheared_grid():
    from tensorfem.thermal import recover_q4_boundary_power
    from tensorfem.thermal_analytic import structured_quad_mesh

    mesh = structured_quad_mesh(3,2,dtype=D)
    x,y = mesh.nodes.T
    nodes = torch.stack((x**2+.25*y,y),dim=1)
    k = nodes.new_tensor([[2.,.6],[.6,1.]])
    model = ThermalModel(nodes,mesh.elements,k[None],1.,1.,torch.tensor([],dtype=torch.long),
                         nodes.new_empty(0),element_type="q4",thickness=1.7)
    x,y = nodes.T
    u = (1+.3j)*(x*x+2*x*y-.4*y*y)
    for edge,outer,inner in ((3,[0,3],[1,4]),(1,[2,5],[1,4]),(0,[0,1,2],[3,4,5]),
                             (2,[3,4,5],[0,1,2])):
        coords = nodes[mesh.elements[outer]]
        start,end = coords[:,edge],coords[:,(edge+1)%4]
        a,b = ((start+end)/2).T
        gradient = torch.stack((2*a+2*b,2*a-.8*b),dim=1)
        dx,dy = (end-start).T
        expected = -(1+.3j)*1.7*((gradient @ k)*torch.stack((dy,-dx),dim=1)).sum(1)
        power = recover_q4_boundary_power(model,torch.stack((u,2*u)),outer,inner,edge)
        torch.testing.assert_close(power,torch.stack((expected,2*expected)),rtol=1e-12,atol=1e-13)
    with pytest.raises(ValueError,match="share"):
        recover_q4_boundary_power(model,u,[0],[5],3)


def test_quadratic_boundary_patch_on_distorted_mesh():
    from tensorfem import recover_q4_patch_boundary_power
    from tensorfem.thermal_analytic import structured_quad_mesh
    mesh = structured_quad_mesh(2,2,dtype=D)
    xy = mesh.nodes.clone()
    xy[4] += xy.new_tensor([.12,-.09])
    k = xy.new_tensor([[2.,.25],[.25,.8]])
    model = ThermalModel(xy,mesh.elements,k[None],1.,1.,torch.tensor([],dtype=torch.long),
                         xy.new_empty(0),element_type="q4")
    x,y = xy.T
    field = (1.+.4j)*(x*x+2*x*y-.5*y*y)+y
    for edge,first in enumerate((0,1,3,2)):
        patch = [[first,*[v for v in range(4) if v != first]]]
        coords = xy[mesh.elements[first]]
        a,b = coords[edge],coords[(edge+1)%4]
        u,v = (a+b)/2
        grad = (1.+.4j)*torch.stack((2*u+2*v,2*u-v))+xy.new_tensor([0.,1.])
        tangent = b-a
        normal = torch.stack((tangent[1],-tangent[0]))
        target = -(grad*(k@normal)).sum()
        result = recover_q4_patch_boundary_power(model,torch.stack((field,2*field)),patch,edge)
        torch.testing.assert_close(result[:,0],torch.stack((target,2*target)),atol=1e-12,rtol=0.)
    with pytest.raises(ValueError,match="edge-connected"):
        recover_q4_patch_boundary_power(model,field,[[0,3]],0)
    with pytest.raises(ValueError,match="complete quadratic"):
        recover_q4_patch_boundary_power(model,field,[[0]],0)


def test_cubic_patch_integrates_varying_flux_on_sheared_edges():
    from tensorfem import recover_q4_patch_boundary_power
    from tensorfem.thermal_analytic import structured_quad_mesh

    mesh = structured_quad_mesh(3,3,shear=.3,dtype=D)
    xy = mesh.nodes.clone()
    xy[5] += xy.new_tensor([.06,-.04])
    k = xy.new_tensor([[1.5,-.4],[-.4,2.]])
    model = ThermalModel(xy,mesh.elements,k[None],1.,1.,torch.tensor([],dtype=torch.long),
                         xy.new_empty(0),element_type="q4",thickness=.7)
    x,y = xy.T
    t = (1.+.4j)*(2*x**3+x*x*y-3*x*y*y+y**3)
    for edge,outer in enumerate((0,2,8,6)):
        patch = [[outer,*[i for i in range(9) if i!=outer]]]
        endpoints = xy[mesh.elements[outer,[edge,(edge+1)%4]]]
        tangent = endpoints[1]-endpoints[0]
        # Two-point Gauss integration of the analytic quadratic flux.
        a,b = (endpoints.mean(0)+xy.new_tensor([[-1.],[1.]])*tangent/(2*3**.5)).T
        grad = torch.stack((6*a*a+2*a*b-3*b*b,a*a-6*a*b+3*b*b),dim=1)
        normal = torch.stack((tangent[1],-tangent[0]))
        exact = -.7*(1.+.4j)*(grad @ k @ normal).mean()
        result = recover_q4_patch_boundary_power(model,t,patch,edge,degree=3)
        torch.testing.assert_close(result[0],exact,atol=1e-12,rtol=0.)
    with pytest.raises(ValueError,match="complete cubic"):
        recover_q4_patch_boundary_power(model,t,[[0,1,3,4]],0,degree=3)


@pytest.mark.parametrize("sparse",[False,True])
def test_consistent_edge_contact_linear_jump(sparse):
    from dataclasses import replace
    x = torch.tensor([[0,0],[1,0],[1,1],[0,1],[1,0],[2,0],[2,1],[1,1]],dtype=D)
    base = ThermalModel(x,torch.arange(8).reshape(2,4),1.,1.,1.,torch.tensor([],dtype=torch.long),
                        x.new_empty(0),element_type="q4")
    edges = torch.tensor([[[1,2],[4,7]]])
    attached = replace(base,contact_edge_pairs=edges,contact_edge_conductance=3.)
    def conductivity(model):
        return assemble_thermal(model,sparse=sparse).conductivity.to_dense()
    jump_operator = conductivity(attached)-conductivity(base)
    temp = x.new_tensor([0,2,6,0,1,0,0,2])
    # Jump varies from 1 to 4 K; integrate it against each endpoint hat.
    npower = x.new_tensor([0,3,4.5,0,-3,0,0,-4.5])
    torch.testing.assert_close(jump_operator@temp,npower)
    torch.testing.assert_close(jump_operator@torch.ones_like(temp),torch.zeros_like(temp))
    # For endpoint g=(0,6) and jump=(1,4), the density is 6s(1+3s).
    # Integrating against the two endpoint hats gives 2.5 and 6.5 W.
    from tensorfem import thermal_contact_edge_power
    from tensorfem import thermal_contact_edge_line_power_density
    graded = replace(attached,contact_edge_conductance=x.new_tensor([[0.,6.]]))
    expected = x.new_tensor([0,2.5,6.5,0,-2.5,0,0,-6.5])
    torch.testing.assert_close((conductivity(graded)-conductivity(base))@temp,expected)
    histories = torch.stack((temp,2*temp))*(1+.4j)
    length = (x[2]-x[1]).norm()
    for s in (0., .31, 1.):
        local = thermal_contact_edge_line_power_density(graded,histories,fraction=s)
        expected_local = x.new_tensor([1.,2.])*(1+.4j)*6*s*(1+3*s)/length
        torch.testing.assert_close(local[:,0],expected_local)
    torch.testing.assert_close(thermal_contact_edge_power(graded,histories)[:,0],
                               x.new_tensor([9.,18.])*(1+.4j))
    # For the linear jump and graded conductance the primitive is 3s²+6s³.
    half = thermal_contact_edge_power(graded,histories,end=.5)
    torch.testing.assert_close(half[:,0], x.new_tensor([1.5,3.])*(1+.4j))
    torch.testing.assert_close(half+thermal_contact_edge_power(graded,histories,start=.5),
                               thermal_contact_edge_power(graded,histories))
    with pytest.raises(ValueError,match="fractions"):
        thermal_contact_edge_power(graded,temp,start=.9,end=.2)
    with pytest.raises(ValueError,match="matching order"):
        replace(attached,contact_edge_pairs=torch.tensor([[[1,2],[7,4]]]))


@pytest.mark.parametrize("sparse", [False, True])
def test_graded_two_material_contact_subedge_power(sparse, record_property):
    from tensorfem import thermal_contact_edge_power

    # Unit height with three unequal interface segments. Each Q4 represents
    # the bilinear solution T=A-(4+6y)*(x-1)/k exactly. Its interface jump
    # is 2 K and the local conductance per area is 2+3y.
    ys = torch.tensor([0., .2, .65, 1.], dtype=D)
    coords, elems, pairs = [], [], []
    for y in ys.tolist():
        coords.extend([(0.,y), (1.,y), (1.,y), (2.,y)])
    for row in range(len(ys)-1):
        i = 4*row
        elems.extend([[i,i+1,i+5,i+4], [i+2,i+3,i+7,i+6]])
        pairs.append([[i+1,i+5], [i+2,i+6]])
    nodes = torch.tensor(coords, dtype=D)
    x,y = nodes.T
    left = torch.arange(len(x))%4 < 2
    exact = torch.where(left, 12., 10.) - (4+6*y)*(x-1)/torch.where(left, 2., 5.)
    fixed = torch.where((x==0)|(x==2)|(y==0)|(y==1))[0]
    dy = ys.diff()
    conductance = dy[:,None]*(2+3*torch.stack((ys[:-1],ys[1:]),dim=1))
    model = ThermalModel(nodes, torch.tensor(elems), nodes.new_tensor([2.,5.]).repeat(len(dy)),
                         1., 1., fixed, exact[fixed], element_type="q4",
                         contact_edge_pairs=torch.tensor(pairs), contact_edge_conductance=conductance)
    field = solve_steady_thermal(model,sparse=sparse)
    terr = float((field-exact).abs().max())
    record_property("max_temperature_error_K", terr)
    assert terr < 2e-13
    from tensorfem import thermal_contact_edge_line_power_density
    point_errors = []
    for fraction in (0., .37, 1.):
        local = thermal_contact_edge_line_power_density(model,field,fraction=fraction)
        expected_local = 4 + 6*(ys[:-1] + fraction*dy)
        point_errors.append(float((local-expected_local).abs().max()))
    record_property("max_line_power_density_error_W_per_m", max(point_errors))
    assert max(point_errors) < 2e-13
    a,b = .15,.8
    lo,hi = ys[:-1]+a*dy, ys[:-1]+b*dy
    expected = 4*(hi-lo)+3*(hi**2-lo**2)
    power = thermal_contact_edge_power(model,field,start=a,end=b)
    err = float((power-expected).abs().max())
    record_property("max_partial_edge_power_error_W",err)
    assert err < 2e-13
    assert abs(float(thermal_contact_edge_power(model,field).sum())-7.) < 2e-13


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("method,theta,order", [("theta",1.,1), ("theta",.5,2), ("sdirk2",.5,2)])
def test_transient_contact_heat_against_two_layer_relaxation(sparse, method, theta, order, record_property):
    from tensorfem import thermal_contact_edge_heat_steps

    # Unit square per slab, insulated exterior. Uniformity in y reduces Q4 to
    # two LINE2 elements of lengths one with four independent temperatures.
    x = torch.tensor([[0,0],[1,0],[1,1],[0,1],[1,0],[2,0],[2,1],[1,1]],dtype=D)
    model = ThermalModel(x,torch.arange(8).reshape(2,4),x.new_tensor([2.,5.]),1.,
                         x.new_tensor([3.,4.]),torch.empty(0,dtype=torch.long),x.new_empty(0),
                         element_type="q4",contact_edge_pairs=torch.tensor([[[1,2],[4,7]]]),
                         contact_edge_conductance=1.5)
    initial = x.new_tensor([10.,10.,10.,10.,2.,2.,2.,2.])
    # Independent 1-D capacity and stiffness. The contact is a 1.5 W/K link.
    c = torch.block_diag(3*x.new_tensor([[2,1],[1,2]])/6,4*x.new_tensor([[2,1],[1,2]])/6)
    k = x.new_tensor([[2,-2,0,0],[-2,3.5,-1.5,0],[0,-1.5,6.5,-5],[0,0,-5,5]])
    end = .2
    reference = torch.matrix_exp(-end*torch.linalg.solve(c,k)) @ x.new_tensor([10,10,2,2])
    reference_heat = 3*(10-reference[:2].mean())
    errors = []
    worst_budget = 0.
    for n in (8,16,32):
        time = end*torch.linspace(0,1,n+1,dtype=D)**1.25
        result = solve_transient_thermal(model,time,initial,theta=theta,method=method,sparse=sparse)
        heat = thermal_contact_edge_heat_steps(model,result)[:,0]
        # Unit volume, so each layer's stored heat is capacity times its
        # four-corner average; this is independent of the discrete reactions.
        left = 3*result.temperature[:,:4].mean(1).diff()
        right = 4*result.temperature[:,4:].mean(1).diff()
        worst_budget = max(worst_budget,float((left+heat).abs().max()),float((right-heat).abs().max()))
        part = thermal_contact_edge_heat_steps(model,result,start=.1,end=.7)[:,0]
        torch.testing.assert_close(part,.6*heat,rtol=1e-12,atol=1e-14)
        errors.append(abs(float(heat.sum()-reference_heat)))
        record_property(f"cumulative_heat_error_{n}_J", errors[-1])
    record_property("max_layer_step_balance_error_J", worst_budget)
    assert worst_budget < 1e-13
    assert all(2**(order-.2) < a/b < 2**(order+.2) for a,b in zip(errors,errors[1:]))


@pytest.mark.parametrize("sparse", [False, True])
def test_low_frequency_contact_power_against_layer_capacity_moments(sparse, record_property):
    from tensorfem import (thermal_contact_edge_low_frequency_power, thermal_contact_edge_power,
                           solve_harmonic_thermal, solve_low_frequency_thermal)

    # Two unit slabs, unit contact area, fixed complex wall drives.
    x = torch.tensor([[0,0],[1,0],[1,1],[0,1],[1,0],[2,0],[2,1],[1,1]],dtype=D)
    hot, cold = 2.+.5j, -.3+.7j
    kl, kr, cl, cr, g = 2., 5., 3., 4., 1.5
    model = ThermalModel(x,torch.arange(8).reshape(2,4),x.new_tensor([kl,kr]),1.,
                         x.new_tensor([cl,cr]),torch.tensor([0,3,5,6]),
                         torch.tensor([hot,hot,cold,cold],dtype=torch.complex128),
                         element_type="q4",contact_edge_pairs=torch.tensor([[[1,2],[4,7]]]),
                         contact_edge_conductance=g)
    resistance = 1/kl+1/g+1/kr
    dc = (hot-cold)/resistance
    # Integrate dq1/dx=-C*T0 and dT1/dx=-q1/k on each layer. T1 is
    # zero at both outer walls and jumps by q1/g at the interface.
    left_moment = cl/kl*(hot/2-dc/(3*kl))
    right_moment = cr/kr*(cold/2+dc/(3*kr))
    lag = (right_moment-left_moment)/resistance
    interval = dict(start=.2,end=.8)
    steady, first = thermal_contact_edge_low_frequency_power(
        model, solve_low_frequency_thermal(model,sparse=sparse), **interval)
    for name, value, reference in (("dc_W",steady,dc),("lag_W_s",first,lag)):
        error = abs(value.item()-.6*reference)
        record_property(f"absolute_error_{name}", error)
        assert error < 2e-14
    errors = []
    for omega in (.05,.025,.0125):
        harmonic = solve_harmonic_thermal(model,omega,sparse=sparse)
        actual = thermal_contact_edge_power(model,harmonic.temperature,**interval)
        errors.append(abs((actual-steady-1j*omega*first).item()))
        record_property(f"complex_power_remainder_{omega}_W", errors[-1])
    assert all(3.9 < a/b < 4.1 for a,b in zip(errors,errors[1:]))


@pytest.mark.parametrize("sparse", [False, True])
def test_harmonic_contact_slab_against_layer_transfer_solution(sparse, record_property):
    from tensorfem import solve_harmonic_thermal, thermal_contact_edge_power

    # Unit cross section, insulated lateral edges, interface at x=1.
    omega = 2.3
    conductance = 2.5
    interface_power = 0.8 - 0.4j
    cold_trace = 0.7 + 0.2j
    traces = [cold_trace + interface_power/conductance, cold_trace]
    materials = [(2., 3.), (4., 1.5)]
    errors = []
    for n in (8, 16, 32):
        coords, elements, exact, wall_power = [], [], [], []
        for side, (k, capacity) in enumerate(materials):
            wave = (1j*omega*capacity/k)**0.5
            x = torch.linspace(side, side+1, n+1, dtype=D)
            arg = wave*(x-1)
            profile = traces[side]*arg.cosh() - interface_power*arg.sinh()/(k*wave)
            flux = interface_power*arg.cosh() - k*wave*traces[side]*arg.sinh()
            wall_power.append(flux[0] if side == 0 else -flux[-1])
            base = len(coords)
            for xi in x.tolist():
                coords.extend(((xi, 0.), (xi, 1.)))
            for j in range(base, base+2*n, 2):
                elements.append([j, j+2, j+3, j+1])
            exact.append(profile.repeat_interleave(2))
        exact = torch.cat(exact)
        nodes = torch.tensor(coords, dtype=D)
        fixed = torch.tensor([0, 1, len(coords)-2, len(coords)-1])
        interface = torch.tensor([[[2*n, 2*n+1], [2*n+2, 2*n+3]]])
        model = ThermalModel(
            nodes, torch.tensor(elements), nodes.new_tensor([2., 4.]).repeat_interleave(n),
            1., nodes.new_tensor([3., 1.5]).repeat_interleave(n), fixed, exact[fixed],
            element_type="q4", contact_edge_pairs=interface, contact_edge_conductance=conductance)
        result = solve_harmonic_thermal(model, omega, sparse=sparse)
        transferred = thermal_contact_edge_power(model, result.temperature)
        inward = result.reaction_power[fixed].reshape(2, 2).sum(dim=1)
        row = [float((result.temperature-exact).abs().max()),
               float((transferred-interface_power).abs().max()),
               float((inward-torch.stack(wall_power)).abs().max())]
        for name, error in zip(("temperature_K", "contact_power_W", "wall_power_W"), row):
            record_property(f"max_error_{n}_{name}", error)
        errors.append(row)
    for coarse, fine in zip(errors, errors[1:]):
        assert all(a/b > 3.8 for a, b in zip(coarse, fine))


def test_periodic_contact_entropy_production_against_cycle_integral(record_property):
    import math
    from tensorfem import thermal_contact_edge_entropy_production_density

    nodes = torch.tensor([[0,0],[1,0],[1,2],[0,2],[1,0],[2,0],[2,2],[1,2]], dtype=D)
    model = ThermalModel(nodes, torch.arange(8).reshape(2,4), 1., 1., 1.,
                         torch.empty(0, dtype=torch.long), nodes.new_empty(0), element_type="q4",
                         contact_edge_pairs=torch.tensor([[[1,2],[4,7]]]),
                         contact_edge_conductance=nodes.new_tensor([[2., 5.]]))
    phase = torch.arange(128, dtype=D) * (2*math.pi/128)
    phi = 0.7
    history = nodes.new_full((128, 8), 300.)
    history[:, [1,2]] = 310 + phase.cos()[:, None]*nodes.new_tensor([18., 25.])
    history[:, [4,7]] = 290 + (phase-phi).cos()[:, None]*nodes.new_tensor([31., 12.])

    # For denominator b+B*cos(t) the means of 1/den and cos(t)/den
    # are 1/root and -B/(root*(b+root)), root=sqrt(b²-B²).
    def mean_ratio(a, amp, b, other):
        root = math.sqrt(b*b-other*other)
        return (a - amp*math.cos(phi)*other/(b+root))/root

    worst = 0.
    for s in (0., 0.37, 1.):
        rate = thermal_contact_edge_entropy_production_density(model, history, fraction=s)
        a, b = 18+7*s, 31-19*s
        expected = (2+3*s)/2 * (mean_ratio(310,a,290,b) + mean_ratio(290,b,310,a) - 2)
        worst = max(worst, abs(rate.mean().item()-expected))
        assert rate.shape == (128, 1) and (rate >= 0).all()
    record_property("max_cycle_mean_error_W_per_m_K", worst)
    assert worst < 2e-15
    with pytest.raises(ValueError, match="positive"):
        thermal_contact_edge_entropy_production_density(model, 0*history[0], fraction=0.5)
    with pytest.raises(ValueError, match="real"):
        thermal_contact_edge_entropy_production_density(model, history[0].to(torch.complex128), fraction=0.5)


def test_integrated_contact_entropy_on_graded_subedge(record_property):
    import math
    from tensorfem import thermal_contact_edge_entropy_production

    nodes = torch.tensor([[0,0],[1,0],[1,2],[0,2],[1,0],[2,0],[2,2],[1,2]], dtype=D)
    model = ThermalModel(nodes, torch.arange(8).reshape(2,4), 1., 1., 1.,
                         torch.empty(0, dtype=torch.long), nodes.new_empty(0), element_type="q4",
                         contact_edge_pairs=torch.tensor([[[1,2],[4,7]]]),
                         contact_edge_conductance=nodes.new_tensor([[2., 6.]]))
    t = nodes.new_tensor([300, 100, 900, 300, 500, 300, 300, 200])
    histories = torch.stack((t, t*1.3, torch.full_like(t, 350.)))
    for start, end in ((0., 1.), (.12, .86)):
        # Partial fractions of (2+4s)*(1100s-400)^2/((100+800s)*(500-300s)).
        reference = (-121/12*(end**2-start**2) - 3817/144*(end-start)
                     + 129/128*math.log1p(8*(end-start)/(1+8*start))
                     - 1118/27*math.log1p(-3*(end-start)/(5-3*start)))
        errors = []
        for order in (4, 8, 16, 32):
            result = thermal_contact_edge_entropy_production(
                model, histories, start=start, end=end, quadrature_order=order)
            torch.testing.assert_close(result[0], result[1])
            assert result.shape == (3, 1) and result[2, 0] == 0
            error = abs(float(result[0, 0])-reference)
            record_property(f"error_{start}_{end}_order_{order}_W_per_K", error)
            errors.append(error)
        assert errors[1] < errors[0]/100
        assert errors[2] < errors[1]/100
        assert errors[-1] < 2e-14


def test_contact_entropy_history_against_isolated_reservoirs(record_property):
    from tensorfem import thermal_contact_edge_entropy_steps

    nodes = torch.tensor([[0,0],[1,0],[1,2],[0,2],[1,0],[2,0],[2,2],[1,2]], dtype=D)
    g, c1, c2 = 1.5, 3., 5.
    model = ThermalModel(nodes, torch.arange(8).reshape(2,4), 1., 1., 1.,
                         torch.empty(0, dtype=torch.long), nodes.new_empty(0), element_type="q4",
                         contact_edge_pairs=torch.tensor([[[1,2],[4,7]]]), contact_edge_conductance=g)
    equilibrium = (400*c1+280*c2)/(c1+c2)
    errors = []
    for n in (12, 24, 48):
        time = 3*torch.linspace(0, 1, n+1, dtype=D)**1.25
        difference = 120*torch.exp(-g*(1/c1+1/c2)*time)
        hot = equilibrium + c2/(c1+c2)*difference
        cold = equilibrium - c1/(c1+c2)*difference
        histories = torch.cat((hot[:, None].expand(-1, 4), cold[:, None].expand(-1, 4)), dim=1)
        entropy = thermal_contact_edge_entropy_steps(model, time, histories)[:, 0]
        # No external exchange. Total entropy change equals contact production.
        exact = c1*torch.log(hot/400) + c2*torch.log(cold/280)
        error = (entropy.cumsum(dim=0)-exact[1:]).abs().max().item()
        assert (entropy > 0).all()
        part = thermal_contact_edge_entropy_steps(model, time, histories, start=.2, end=.7)
        torch.testing.assert_close(part[:, 0], .5*entropy)
        errors.append(error)
        record_property(f"max_cumulative_entropy_error_{n}_J_per_K", error)
    assert all(3.8 < a/b < 4.2 for a, b in zip(errors, errors[1:]))


@pytest.mark.parametrize("sparse", [False, True])
def test_contact_and_bulk_entropy_of_insulated_slab_mode(sparse, record_property):
    from tensorfem import thermal_contact_edge_entropy_steps

    # Unit k, heat capacity and cross section. The insulated outer walls admit
    # opposite cosine modes with a jump at x=1. Fourier flux fixes G.
    beta, mean, amplitude = 1.1, 300., 70.
    g = beta*math.tan(beta)/2
    errors = []
    for n in (8, 16, 32):
        local_x = torch.linspace(0., 1., n+1, dtype=D)
        x = torch.cat((local_x, local_x+1)).repeat_interleave(2)
        y = x.new_tensor([0., 1.]).repeat(2*(n+1))
        nodes = torch.stack((x, y), dim=1)
        first = torch.cat((2*torch.arange(n), 2*(n+1)+2*torch.arange(n)))
        elements = first[:, None] + torch.tensor([0, 2, 3, 1])
        interface = torch.tensor([[[2*n, 2*n+1], [2*n+2, 2*n+3]]])
        model = ThermalModel(nodes, elements, 1., 1., 1., torch.empty(0, dtype=torch.long),
                             x.new_empty(0), element_type="q4", contact_edge_pairs=interface,
                             contact_edge_conductance=g)
        mode = torch.cos(beta*local_x)
        initial = mean + amplitude*torch.cat((mode, -mode.flip((0,)))).repeat_interleave(2)
        time = torch.linspace(0., .4, n+1, dtype=D)
        field = solve_transient_thermal(model, time, initial, theta=.5, sparse=sparse).temperature
        contact = thermal_contact_edge_entropy_steps(model, time, field)[:, 0].cumsum(0)
        delta = amplitude*math.cos(beta)
        exact_contact = (2*g/beta**2)*torch.log1p(
            delta**2 * (-torch.expm1(-2*beta**2*time[1:])) / (mean**2-delta**2))

        # The solution is constant across the unit width. For a linear
        # element temperature, the spatial mean of 1/T² is 1/(T_left*T_right).
        left, right = field[:, first], field[:, first+2]
        bulk_rate = (n*(right-left).square()/(left*right)).sum(dim=1)
        bulk = (time.diff()*(bulk_rate[1:]+bulk_rate[:-1])/2).cumsum(0)
        # Integrate C log(T/T_initial) independently with two Gauss points.
        storage = torch.zeros_like(time)
        for fraction in (0.5-0.5/math.sqrt(3), 0.5+0.5/math.sqrt(3)):
            sample = torch.lerp(left, right, fraction)
            storage += torch.log(sample/sample[0]).sum(dim=1)/(2*n)
        row = ((contact-exact_contact).abs().max().item(),
               (storage[1:]-bulk-contact).abs().max().item())
        record_property(f"max_contact_entropy_error_{n}_J_per_K", row[0])
        record_property(f"max_total_entropy_balance_error_{n}_J_per_K", row[1])
        assert (contact.diff() > 0).all() and (bulk.diff() > 0).all()
        errors.append(row)
    for coarse, fine in zip(errors, errors[1:]):
        assert all(a/b > 3.7 for a, b in zip(coarse, fine))


@pytest.mark.parametrize("sparse",[False,True])
def test_harmonic_uniform_heating_on_disconnected_components(sparse):
    from tensorfem.thermal import solve_harmonic_thermal, thermal_element_nodal_power
    rates = torch.tensor([.27+.34j, -.6+.13j], dtype=torch.complex128)
    c = torch.tensor([.9, 2.], dtype=D)
    model = ThermalModel(torch.tensor([[0.],[1.],[1.],[2.]],dtype=D),torch.tensor([[0,1],[2,3]]),
        1.,1.,c,torch.empty(0,dtype=torch.long),torch.empty(0,dtype=D),heat_source=c*rates)
    for omega in (1., 1e-12):
        state = solve_harmonic_thermal(model, omega, sparse=sparse, separate_mean=True)
        power, net = thermal_element_nodal_power(model, state.temperature, omega,
            mean_temperature=state.mean_temperature, return_net_input=True)
        torch.testing.assert_close(1j*omega*state.mean_temperature, rates.repeat_interleave(2), atol=1e-14, rtol=0)
        for v in (power, net, state.temperature, state.reaction_power):
            assert v.abs().max() < 1e-14
    for omega in (0., 1e-12, 1.):
        state = solve_harmonic_thermal(model, omega, sparse=sparse, return_mean_rate=True)
        torch.testing.assert_close(state.mean_heating_rate, rates.repeat_interleave(2), atol=1e-14, rtol=0)
        power, net = thermal_element_nodal_power(model, state.temperature, omega,
            mean_heating_rate=state.mean_heating_rate, return_net_input=True)
        for v in (power, net, state.temperature, state.reaction_power):
            assert v.abs().max() < 1e-14


def test_linear_edge_flux_recovers_complex_traces_on_unequal_edges():
    from tensorfem import recover_linear_edge_flux
    areas = torch.tensor([.4, 1.2, .7], dtype=D)
    traces = torch.tensor([[1+.5j, -.7j, .2, 1j], [.3, 1., -.8j, .5]], dtype=torch.complex128)
    power = torch.zeros_like(traces)
    power[:,:-1] += areas*(2*traces[:,:-1]+traces[:,1:])/6
    power[:,1:] += areas*(traces[:,:-1]+2*traces[:,1:])/6
    result = recover_linear_edge_flux(power, areas)
    torch.testing.assert_close(result, traces)
    torch.testing.assert_close((areas*(result[:,:-1]+result[:,1:])/2).sum(-1), power.sum(-1))
    with pytest.raises(ValueError, match="area"):
        recover_linear_edge_flux(power, -areas)


@pytest.mark.parametrize("n", [3, 7])
def test_closed_linear_edge_flux_is_invariant_to_seam(n):
    from tensorfem import recover_linear_edge_flux
    area = torch.linspace(.2, .9, n, dtype=D)
    x = torch.arange(n, dtype=D)
    flux = torch.stack((torch.exp(1j*x), x+1j*x.square()))
    neighbor = flux.roll(-1, dims=-1)
    power = (area*(2*flux+neighbor)+(area*(flux+2*neighbor)).roll(1, dims=-1))/6
    for shift in (0, 1, n-1):
        recovered = recover_linear_edge_flux(power.roll(shift, dims=-1), area.roll(shift), closed=True)
        torch.testing.assert_close(recovered, flux.roll(shift, dims=-1), atol=1e-13, rtol=0)
    with pytest.raises(ValueError, match="three"):
        recover_linear_edge_flux(power[:,:2], area[:2], closed=True)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("omega", [.125, 2., 16.])
def test_driven_two_material_rod_contact_pair_power(sparse,omega,record_property):
    from tensorfem import solve_harmonic_thermal, thermal_contact_pair_power, thermal_element_nodal_power

    # Axial power q fixes the slope -q/(k*A) in each segment and the
    # temperature jump q/G. Local volumetric heating i*omega*C*T supplies
    # storage, keeping these piecewise-linear fields exact at every frequency.
    x = torch.tensor([0.,.375,1.,1.,1.25,2.],dtype=D)
    conn = torch.tensor([[0,1],[1,2],[3,4],[4,5]])
    q, cold, g = .75-.25j, .5+.375j, 2.5
    area = torch.tensor([1.,1.,.5,.5],dtype=D)
    k = torch.tensor([2.,2.,3.,3.],dtype=D)
    capacity = torch.tensor([1.5,1.5,4.,4.],dtype=D)
    exact = torch.cat((cold+q/g-q*(x[:3]-1)/2, cold-q*(x[3:]-1)/1.5))
    model = ThermalModel(x[:,None],conn,k,1.,capacity,torch.tensor([0,5]),exact[[0,5]],
                         thickness=area,heat_source=1j*omega*capacity[:,None]*exact[conn],
                         contact_pairs=torch.tensor([[2,3]]),contact_conductance=g)
    result = solve_harmonic_thermal(model,omega,sparse=sparse)
    contact = thermal_contact_pair_power(model,result.temperature).item()
    element = thermal_element_nodal_power(model,result.temperature,omega)
    errors = dict(temperature_K=float((result.temperature-exact).abs().max()),
                  contact_W=abs(contact-q),left_balance_W=abs(element[1,1].item()+contact),
                  right_balance_W=abs(element[2,0].item()-contact))
    for name,error in errors.items():
        record_property(name,error)
        assert error < 3e-14


def test_batched_contact_pair_power_reverses_with_node_order():
    from dataclasses import replace
    from tensorfem import thermal_contact_pair_power

    model = replace(line_model(),contact_pairs=torch.tensor([[1,2],[4,3],[6,7]]),
                    contact_conductance=torch.tensor([2.,.5,0.],dtype=D))
    field = model.nodes[:,0]
    batch = torch.stack((field,-3*field))
    expected = batch.new_tensor([[-.5,.125,0.],[1.5,-.375,0.]])
    torch.testing.assert_close(thermal_contact_pair_power(model,batch),expected,rtol=0,atol=0)


@pytest.mark.parametrize("sparse", [False, True])
def test_curved_harmonic_rod_contact_power_refinement(sparse, record_property):
    from tensorfem import solve_harmonic_thermal, thermal_contact_pair_power

    # In each constant layer k*T''=i*omega*C*T-S. Set both interface traces
    # and their slopes independently of the mesh, then prescribe the resulting
    # exact temperatures at the outer ends. Rightward power is -A*k*T'.
    omega, conductance = 2.3, 2.5
    q, cold = .8-.4j, .7+.2j
    layers = [(2., 3., 1., .75+.25j, cold+q/conductance),
              (4., 1.5, .5, -.5+.75j, cold)]
    previous = None
    for n in (8, 16, 32, 64):
        coords, exact, ends = [], [], []
        for side,(k,c,area,source,trace) in enumerate(layers):
            x = torch.linspace(side,side+1,n+1,dtype=D)
            wave = (1j*omega*c/k)**.5
            background = source/(1j*omega*c)
            arg = wave*(x-1)
            profile = background+(trace-background)*arg.cosh()-q*arg.sinh()/(area*k*wave)
            flux = q*arg.cosh()-area*k*wave*(trace-background)*arg.sinh()
            coords.append(x)
            exact.append(profile)
            ends.append(flux[0] if side==0 else -flux[-1])
        nodes = torch.cat(coords)[:,None]
        expected = torch.cat(exact)
        left = torch.cat((torch.arange(n),torch.arange(n)+n+1))
        fixed = torch.tensor([0,2*n+1])
        model = ThermalModel(nodes,torch.stack((left,left+1),dim=1),
            torch.tensor([row[0] for row in layers],dtype=D).repeat_interleave(n),1.,
            torch.tensor([row[1] for row in layers],dtype=D).repeat_interleave(n),fixed,expected[fixed],
            thickness=torch.tensor([row[2] for row in layers],dtype=D).repeat_interleave(n),
            heat_source=torch.tensor([row[3] for row in layers],dtype=torch.complex128).repeat_interleave(n),
            contact_pairs=torch.tensor([[n,n+1]]),contact_conductance=conductance)
        result = solve_harmonic_thermal(model,omega,sparse=sparse)
        contact = thermal_contact_pair_power(model,result.temperature).item()
        errors = dict(nodal_temperature_K=float((result.temperature-expected).abs().max()),
                      contact_W=abs(contact-q),west_W=abs((result.reaction_power[0]-ends[0]).item()),
                      east_W=abs((result.reaction_power[-1]-ends[1]).item()))
        for name,value in errors.items():
            record_property(f"n{n}_{name}",value)
            if previous is not None:
                order = math.log2(previous[name]/value)
                record_property(f"n{n}_{name}_order",order)
                assert 1.9 < order < 2.1
        previous = errors


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("method,theta,order", [("theta",1.,1),("theta",.5,2),("sdirk2",.5,2)])
def test_reversing_pair_heat_against_two_driven_stores(sparse,method,theta,order,record_property):
    from tensorfem import thermal_contact_pair_heat_steps

    # Two parallel links in opposite node order. A 3.75 W heater in store 2
    # reverses their heat flow: D=T1-T2 satisfies D'=-1.25*(D+1), D(0)=2.
    x = torch.tensor([[0.],[1.]],dtype=D)
    initial = x.new_tensor([2.,0.])
    model = ThermalModel(x,torch.empty((0,2),dtype=torch.long),1.,1.,1.,
                         torch.empty(0,dtype=torch.long),x.new_empty(0),
                         contact_pairs=torch.tensor([[0,1],[1,0]]),contact_conductance=x.new_tensor([1.,.5]),
                         nodal_heat_capacity=x.new_tensor([2.,3.]),nodal_heat=x.new_tensor([0.,3.75]))
    previous = None
    for n in (16,32,64):
        time = 1.5*torch.linspace(0.,1.,n+1,dtype=D)**1.25
        result = solve_transient_thermal(model,time,initial,theta=theta,method=method,sparse=sparse)
        heat = thermal_contact_pair_heat_steps(model,result)
        integral = -3*torch.expm1(-1.25*time)/1.25-time
        expected = integral[:,None]*x.new_tensor([1.,-.5])
        error = float((heat.cumsum(0)-expected[1:]).abs().max())
        record_property(f"n{n}_max_cumulative_heat_error_J",error)
        if previous is not None:
            measured = math.log2(previous/error)
            record_property(f"n{n}_order",measured)
            assert order-.15 < measured < order+.15
        previous = error
        transfer = heat[:,0]-heat[:,1]
        change = result.temperature.diff(dim=0)
        balance = max(float((2*change[:,0]+transfer).abs().max()),
                      float((3*change[:,1]-transfer-3.75*time.diff()).abs().max()))
        record_property(f"n{n}_max_step_balance_J",balance)
        assert balance < 2e-13
        assert heat[0,0] > 0 and heat[-1,0] < 0


@pytest.mark.parametrize("sparse",[False,True])
def test_directional_heat_across_two_store_reversal(sparse,record_property):
    from tensorfem import thermal_contact_pair_directional_heat, thermal_contact_pair_heat_steps

    x = torch.tensor([[0.],[1.]],dtype=D)
    model = ThermalModel(x,torch.empty((0,2),dtype=torch.long),1.,1.,1.,
                         torch.empty(0,dtype=torch.long),x.new_empty(0),contact_pairs=torch.tensor([[0,1]]),
                         contact_conductance=1.5,nodal_heat_capacity=x.new_tensor([2.,3.]),
                         nodal_heat=x.new_tensor([0.,3.75]))
    # q(t)=1.5*(3*exp(-1.25*t)-1); its primitive peaks at log(3)/1.25.
    crossing = math.log(3)/1.25
    def primitive(t):
        return 1.5*(-3*math.expm1(-1.25*t)/1.25-t)
    expected = [primitive(crossing),primitive(crossing)-primitive(1.5)]
    previous = None
    for n in (16,32,64,128):
        time = 1.5*torch.linspace(0.,1.,n+1,dtype=D)
        result = solve_transient_thermal(model,time,x.new_tensor([2.,0.]),theta=.5,sparse=sparse)
        forward,reverse = thermal_contact_pair_directional_heat(model,time,result.temperature)
        assert ((forward>0)&(reverse>0)).sum()==1
        residual = float((forward-reverse-thermal_contact_pair_heat_steps(model,result)).abs().max())
        record_property(f"n{n}_net_residual_J",residual)
        assert residual < 1e-14
        errors = [abs(float(q.sum())-ref) for q,ref in zip((forward,reverse),expected)]
        for i,name in enumerate(("forward","reverse")):
            record_property(f"n{n}_{name}_error_J",errors[i])
            if previous is not None:
                order = math.log2(previous[i]/errors[i])
                record_property(f"n{n}_{name}_order",order)
                assert 1.85 < order < 2.15
        previous = errors


def test_linear_directional_contact_heat_zero_endpoints_and_weak_asymmetric_crossing():
    from dataclasses import replace
    from tensorfem import thermal_contact_pair_directional_heat

    model = replace(line_model(),contact_pairs=torch.tensor([[0,1],[1,0]]),contact_conductance=1.)
    time = torch.tensor([0.,1.,3.,4.,6.],dtype=D)
    field = torch.zeros((5,len(model.nodes)),dtype=D)
    amplitude = 2.**-600
    field[:,0] = field.new_tensor([0.,3.,-1.,0.,0.])*amplitude
    forward,reverse = thermal_contact_pair_directional_heat(model,time,field)
    expected = field.new_tensor([[1.5,0.],[2.25,.25],[0.,.5],[0.,0.]])
    torch.testing.assert_close(forward/amplitude,expected,atol=0,rtol=0)
    torch.testing.assert_close(reverse/amplitude,expected.flip((1,)),atol=0,rtol=0)


@pytest.mark.parametrize("graded",[False,True])
def test_directional_contact_edge_power_cancelling_traces(graded,record_property):
    from tensorfem import thermal_contact_edge_directional_power, thermal_contact_edge_power

    nodes = torch.tensor([[0,0],[1,0],[1,1],[0,1],[1,0],[2,0],[2,1],[1,1]],dtype=D)
    # With g=1+s the zero at 5/9 makes integral(g*(s-5/9)) vanish.
    root = 5/9 if graded else .5
    model = ThermalModel(nodes,torch.arange(8).reshape(2,4),1.,1.,1.,torch.arange(8),
                         nodes.new_zeros(8),element_type="q4",contact_edge_pairs=torch.tensor([[[1,2],[4,7]]]),
                         contact_edge_conductance=nodes.new_tensor([[1.,1.+graded]]))
    field = nodes.new_zeros(8)
    field[[1,2]] = nodes.new_tensor([-root,1-root])
    batch = torch.stack((field,-field,field*0))
    def primitive(s):
        return .5*s*s-root*s + graded*(s**3/3-.5*root*s*s)
    worst,net_error = 0.,0.
    for a,b in ((0.,1.),(.125,.875),(0.,root),(root,1.),(.25,.25)):
        forward,reverse = thermal_contact_edge_directional_power(model,batch,start=a,end=b)
        split = max(a,min(b,root))
        pos,neg = primitive(b)-primitive(split),primitive(a)-primitive(split)
        expected = nodes.new_tensor([pos,neg,0.])
        worst = max(worst,float((forward[:,0]-expected).abs().max()),
                    float((reverse[:,0]-expected[[1,0,2]]).abs().max()))
        signed = thermal_contact_edge_power(model,batch,start=a,end=b)
        net_error = max(net_error,float((forward-reverse-signed).abs().max()))
        if (a,b)==(0.,1.):
            assert min(forward[0].item(),reverse[0].item())>.12 and abs(signed[0].item())<1e-15
    record_property("max_directional_error_W",worst)
    record_property("max_signed_residual_W",net_error)
    assert max(worst,net_error)<2e-15


@pytest.mark.parametrize("imag", [0., .25, 1.])
def test_harmonic_contact_edge_cycle_heat_against_local_magnitude(imag,record_property):
    from tensorfem import thermal_contact_edge_harmonic_cycle_heat

    x = torch.tensor([[0,0],[1,0],[1,1],[0,1],[1,0],[2,0],[2,1],[1,1]],dtype=D)
    model = ThermalModel(x,torch.arange(8).reshape(2,4),1.,1.,1.,torch.arange(8),x.new_zeros(8),
                         element_type="q4",contact_edge_pairs=torch.tensor([[[1,2],[4,7]]]),
                         contact_edge_conductance=x.new_tensor([[1.,3.]]))
    # With y=s-1/2, g=2+2y and the local jump amplitude is |y+i*b|.
    def primitive(y):
        r = math.hypot(y,imag)
        return y*r + (imag**2*math.asinh(y/imag) if imag else 0.) + 2*r**3/3
    field = torch.zeros(8,dtype=torch.complex128)
    field[1],field[2] = -.5+1j*imag,.5+1j*imag
    batch = torch.stack((field,1j*field,field*0))
    omega = 2.5
    worst = 0.
    for start,end in ((0.,1.),(.15,.85),(.5,.5),(.65,.9)):
        result = thermal_contact_edge_harmonic_cycle_heat(model,batch,omega,start=start,end=end)
        expected = 2*(primitive(end-.5)-primitive(start-.5))/omega
        worst = max(worst,float((result[:,0]-x.new_tensor([expected,expected,0.])).abs().max()))
    record_property("max_per_direction_cycle_heat_error_J",worst)
    assert worst < 1e-13


@pytest.mark.parametrize("bias", [-2.,-1.,-.375,0.,.375,1.,2.])
def test_biased_contact_cycle_heat_against_time_quadrature(bias,record_property):
    from dataclasses import replace
    from scipy.integrate import quad
    from tensorfem import thermal_contact_pair_harmonic_cycle_heat

    model = replace(line_model(),contact_pairs=torch.tensor([[0,1],[1,0]]),contact_conductance=1.)
    dc = torch.zeros(len(model.nodes),dtype=D)
    dc[0] = bias
    ac = dc.to(torch.complex128)
    ac[0] = .6+.8j
    omega = 1.75
    forward,reverse = thermal_contact_pair_harmonic_cycle_heat(model,torch.stack([ac,1j*ac]),omega,
                                                              steady_temperature=dc)
    phase = math.atan2(.8,.6)
    roots = [(s*math.acos(-bias)-phase)%(2*math.pi) for s in (-1,1)] if abs(bias)<1 else []
    def q(angle):
        return bias+.6*math.cos(angle)-.8*math.sin(angle)
    refs = dc.new_tensor([quad(lambda t:max(sign*q(t),0.),0,2*math.pi,points=roots)[0]/omega
                          for sign in (1,-1)])
    error = max(float((forward-refs).abs().max()),float((reverse-refs.flip(0)).abs().max()))
    residual = float((forward[:,0]-reverse[:,0]-2*math.pi*bias/omega).abs().max())
    record_property("directional_error_J",error)
    record_property("net_residual_J",residual)
    assert max(error,residual)<5e-14


def test_contact_cycle_heat_resolves_tiny_reverse_lobe_and_static_load():
    from dataclasses import replace
    from tensorfem import thermal_contact_pair_harmonic_cycle_heat

    model = replace(line_model(),contact_pairs=torch.tensor([[0,1]]),contact_conductance=1.)
    ac = torch.zeros(len(model.nodes),dtype=D)
    ac[0] = 1.
    gap = 2.**-50
    dc = ac*(1-gap)
    _,reverse = thermal_contact_pair_harmonic_cycle_heat(model,ac,1.,steady_temperature=dc)
    # Expanding cos near its minimum gives (4 sqrt(2)/3)*gap^(3/2)
    # with a relative correction O(gap).
    expected = 4*math.sqrt(2)/3*gap**1.5
    assert abs(reverse.item()/expected-1)<3e-15
    forward,reverse = thermal_contact_pair_harmonic_cycle_heat(model,0*ac,2.,steady_temperature=dc)
    torch.testing.assert_close(forward,dc[:1]*math.pi)
    assert reverse.item()==0.


def test_biased_edge_cycle_heat_against_space_time_quadrature(record_property):
    from scipy.integrate import quad
    from tensorfem import thermal_contact_edge_biased_cycle_heat, thermal_contact_edge_power

    x = torch.tensor([[0,0],[1,0],[1,1],[0,1],[1,0],[2,0],[2,1],[1,1]], dtype=D)
    model = ThermalModel(x, torch.arange(8).reshape(2,4), 1., 1., 1., torch.arange(8), x.new_zeros(8),
                         element_type="q4", contact_edge_pairs=torch.tensor([[[1,2],[4,7]]]),
                         contact_edge_conductance=x.new_tensor([[1.,3.]]))
    dc = x.new_zeros(8)
    ac = dc.to(torch.complex128)
    dc[1], dc[2] = -1.3, .9
    ac[1], ac[2] = .8+.3j, -.2+.9j
    omega, start, end = 2.5, .12, .93
    forward, reverse = thermal_contact_edge_biased_cycle_heat(
        model, torch.stack((ac, 1j*ac)), omega, steady_temperature=dc, start=start, end=end)
    def local(s, sign):
        bias = -1.3+2.2*s
        phasor = (1-s)*(.8+.3j)+s*(-.2+.9j)
        roots = []
        if abs(bias) < abs(phasor):
            angle = math.acos(-bias/abs(phasor))
            roots = [(v*angle-math.atan2(phasor.imag, phasor.real))%(2*math.pi) for v in (-1,1)]
        def power(t):
            return max(sign*(bias+phasor.real*math.cos(t)-phasor.imag*math.sin(t)), 0.)
        return (1+2*s)*quad(power, 0., 2*math.pi, points=roots)[0]/omega
    exact = [quad(local, start, end, args=(sign,), epsabs=1e-12, epsrel=1e-12)[0] for sign in (1,-1)]
    error = max((forward-exact[0]).abs().max().item(), (reverse-exact[1]).abs().max().item())
    net = thermal_contact_edge_power(model, dc, start=start, end=end)*2*math.pi/omega
    balance = (forward-reverse-net).abs().max().item()
    record_property("max_directional_error_J", error)
    record_property("max_balance_error_J", balance)
    assert error < 1e-9 and balance < 1e-12
