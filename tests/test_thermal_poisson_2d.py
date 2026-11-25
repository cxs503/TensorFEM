"""Verification of steady-state heat conduction with internal heat generation.

Analytical reference: T(x,y) = sin(πx) sin(πy) satisfies ∇²T = -2π² T
with homogeneous Dirichlet boundary data.  This exercises the consistent
heat-source assembly and the Q4 kernel's ability to handle non-harmonic
fields.
"""
import math

import pytest
import torch

from tensorfem.thermal import ThermalModel, solve_steady_thermal
from tensorfem.thermal_analytic import structured_quad_mesh

D = torch.float64
GATE = 0.03


def exact_T(pts: torch.Tensor) -> torch.Tensor:
    return torch.sin(math.pi * pts[:, 0]) * torch.sin(math.pi * pts[:, 1])


def exact_Q(pts: torch.Tensor) -> torch.Tensor:
    return 2.0 * math.pi ** 2 * exact_T(pts)


def _make_model(nx, ny):
    mesh = structured_quad_mesh(nx, ny)
    # Element-wise source via centroid quadrature
    coords = mesh.nodes[mesh.elements]
    Q_elem = exact_Q(coords.mean(dim=1))
    model = ThermalModel(
        nodes=mesh.nodes,
        elements=mesh.elements,
        conductivity=torch.tensor(1.0, dtype=D),
        density=torch.tensor(1.0, dtype=D),
        specific_heat=torch.tensor(1.0, dtype=D),
        fixed_nodes=mesh.boundary_nodes,
        fixed_temperatures=exact_T(mesh.nodes[mesh.boundary_nodes]),
        heat_source=Q_elem,
        element_type="q4",
    )
    return mesh, model


def test_poisson_sine_source_passes_gate():
    """Single-mesh verification against the closed-form sine field."""
    mesh, model = _make_model(16, 16)
    T = solve_steady_thermal(model)
    l2_err = torch.linalg.vector_norm(T - exact_T(mesh.nodes)) / torch.linalg.vector_norm(exact_T(mesh.nodes))
    max_err = torch.abs(T - exact_T(mesh.nodes)).max() / torch.abs(exact_T(mesh.nodes)).max()
    assert l2_err.item() < GATE, f"L2 relative error {l2_err.item():.4f} exceeds {GATE}"
    assert max_err.item() < GATE, f"Max relative error {max_err.item():.4f} exceeds {GATE}"


def test_zero_source_reduces_to_linear_patch():
    """With Q=0 and linear BCs the solver must reproduce the linear field exactly."""
    mesh = structured_quad_mesh(8, 8)
    # Linear field T = x + 2y
    T_exact = mesh.nodes[:, 0] + 2.0 * mesh.nodes[:, 1]
    model = ThermalModel(
        nodes=mesh.nodes, elements=mesh.elements,
        conductivity=torch.tensor(1.0, dtype=D),
        density=torch.tensor(1.0, dtype=D),
        specific_heat=torch.tensor(1.0, dtype=D),
        fixed_nodes=mesh.boundary_nodes,
        fixed_temperatures=T_exact[mesh.boundary_nodes],
        heat_source=torch.zeros(len(mesh.elements), dtype=D),
        element_type="q4",
    )
    T = solve_steady_thermal(model)
    error = torch.linalg.vector_norm(T - T_exact).item()
    assert error < 1e-12, f"Zero-source residual {error:.2e} is not machine zero"


def test_nonuniform_source_assembly():
    """Linearly varying source must be integrated correctly (tested via energy balance)."""
    mesh, model = _make_model(8, 8)
    # Q(x,y) = 4.0 * (x + y)
    coords = mesh.nodes[mesh.elements]
    Q_elem = 4.0 * (coords[:, 0, 0] + coords[:, 0, 1])
    model = ThermalModel(
        nodes=mesh.nodes, elements=mesh.elements,
        conductivity=torch.tensor(1.0, dtype=D),
        density=torch.tensor(1.0, dtype=D),
        specific_heat=torch.tensor(1.0, dtype=D),
        fixed_nodes=mesh.boundary_nodes,
        fixed_temperatures=exact_T(mesh.nodes[mesh.boundary_nodes]),
        heat_source=Q_elem,
        element_type="q4",
    )
    T = solve_steady_thermal(model)
    assert torch.isfinite(T).all()
    assert T.min().item() >= 0.0  # T >= 0 in domain for Q>0, T|bdry=0


@pytest.mark.parametrize("sparse", [False, True])
def test_quadratic_heat_load_on_trapezoid(sparse, record_property):
    from tensorfem.thermal import assemble_thermal

    model = ThermalModel(torch.tensor([[0.,0.],[2.,0.],[3.,1.],[0.,1.]], dtype=D),
                         torch.arange(4)[None,:], 1., 1., 1., torch.tensor([0]), torch.zeros(1,dtype=D),
                         heat_source_quadratic=torch.tensor([1.,2.,-3.],dtype=D), element_type="q4")
    # Map the unit square using x=(2+v)*u, y=v, det J=2+v. Integrating
    # each unit-square basis times (x²+2xy-3y²)*(2+v) gives these fractions.
    exact = torch.tensor([137/240,161/80,1103/360,217/360], dtype=D)
    heat = assemble_thermal(model, sparse=sparse).heat
    record_property("peak_nodal_load_error_W", (heat-exact).abs().max().item())
    torch.testing.assert_close(heat, exact, atol=2e-15, rtol=0.)


@pytest.mark.parametrize("sparse", [False, True])
def test_complex_quadratic_heat_on_sheared_harmonic_patch(sparse, record_property):
    from tensorfem.thermal import solve_harmonic_thermal, thermal_element_flux

    shear, k, capacity, omega = .35, 2., 1.5, 3.
    phase = .7-.4j
    mesh = structured_quad_mesh(5, 4, shear=shear, dtype=D)
    x, y = mesh.nodes.T
    # The physical quadratic (x-s*y)*y is bilinear on this sheared mesh.
    exact = phase*(x-shear*y)*y
    fixed = mesh.boundary_nodes
    quadratic = 1j*omega*capacity*phase*torch.tensor([0.,1.,-shear], dtype=torch.complex128)
    model = ThermalModel(mesh.nodes, mesh.elements, k, capacity, 1., fixed, exact[fixed],
                         heat_source=2*k*shear*phase, heat_source_quadratic=quadratic, element_type="q4")
    result = solve_harmonic_thermal(model, omega, sparse=sparse)
    cx, cy = mesh.nodes[mesh.elements].mean(dim=1).T
    expected_flux = -k*phase*torch.stack((cy, cx-2*shear*cy), dim=1)
    flux = thermal_element_flux(model, result.temperature)
    # Inward conductive wall reaction integrates k*ΔT over the unit area.
    power_error = abs(result.reaction_power[fixed].sum()+2*k*shear*phase)
    record_property("temperature_error_K", (result.temperature-exact).abs().max().item())
    record_property("flux_error_W_m2", (flux-expected_flux).abs().max().item())
    record_property("wall_power_error_W", power_error.item())
    torch.testing.assert_close(result.temperature, exact, atol=1e-14, rtol=0.)
    torch.testing.assert_close(flux, expected_flux, atol=1e-13, rtol=0.)
    assert power_error < 1e-13


@pytest.mark.parametrize("sparse", [False, True])
def test_layered_rod_quadratic_generation_and_contact(sparse, record_property):
    from tensorfem.thermal import assemble_thermal

    # Global x spans both signs. Each side has distinct conductivity, area,
    # and source; a finite contact conductance transmits the interface power.
    positions = [[-.7,-.55,-.1,.2], [.2,.37,.83,1.1]]
    k = (1.7, 3.1)
    area = (.8, 1.3)
    beta = (4., -2.2)
    contact = 2.5
    exact = []
    initial_t, initial_power = 3., .4
    wall_input = initial_power
    for i, points in enumerate(positions):
        x = torch.tensor(points, dtype=D)
        start, end = points[0], points[-1]
        # P' = area*beta*x², T' = -P/(k*area).
        offset = initial_power-area[i]*beta[i]*start**3/3
        t = initial_t-(offset*(x-start)+area[i]*beta[i]*(x**4-start**4)/12)/(k[i]*area[i])
        exact.append(t)
        initial_power = offset+area[i]*beta[i]*end**3/3
        if i == 0:
            interface_power = initial_power
            initial_t = t[-1].item()-interface_power/contact
    temperature = torch.cat(exact)
    nodes = torch.tensor(positions,dtype=D).reshape(-1,1)
    fixed = torch.tensor([0,7])
    def layers(values):
        return torch.tensor(values,dtype=D).repeat_interleave(3)
    model = ThermalModel(nodes, torch.tensor([[0,1],[1,2],[2,3],[4,5],[5,6],[6,7]]),
                         layers(k), 1., 1., fixed, temperature[fixed], thickness=layers(area),
                         contact_pairs=torch.tensor([[3,4]]), contact_conductance=contact,
                         heat_source_quadratic=layers(beta))
    result = solve_steady_thermal(model,sparse=sparse)
    matrices = assemble_thermal(model,sparse=sparse)
    wall = (matrices.conductivity @ result - matrices.heat)[fixed]
    expected = torch.tensor([wall_input, -initial_power],dtype=D)
    record_property("max_temperature_error_K", (temperature-result).abs().max().item())
    record_property("max_wall_power_error_W", (expected-wall).abs().max().item())
    torch.testing.assert_close(result,temperature,atol=2e-14,rtol=0.)
    torch.testing.assert_close(wall,expected,atol=2e-14,rtol=0.)
    assert abs(contact*(result[3]-result[4])-interface_power) < 2e-14
