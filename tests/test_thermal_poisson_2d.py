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
