"""Tests for cylindrical shell modal analysis."""
import math

import pytest
import torch

from tensorfem.cylindrical_shell_modal import (
    structured_cylindrical_mesh,
    cylindrical_shell4_mass_matrix,
    assemble_cylindrical_shell,
    apply_simply_supported_bc,
    solve_shell_modes,
    dmv_angular_frequency,
    dmv_frequency_hz,
    cylindrical_shell_modal_analysis,
    NASA_TN_D3939_FREQS,
)


def test_structured_cylindrical_mesh():
    """Mesh generation produces correct topology."""
    n_ax, n_circ = 4, 8
    nodes, elements, theta = structured_cylindrical_mesh(n_ax, n_circ, 0.5, 1.0)
    expected_nodes = (n_ax + 1) * n_circ
    assert nodes.shape == (expected_nodes, 3)
    assert elements.shape == (n_ax * n_circ, 4)
    assert theta.shape == (expected_nodes,)
    assert theta.min() >= 0
    assert theta.max() < 2 * math.pi


def test_mass_matrix_positive_definite():
    """Element mass matrix is symmetric positive definite."""
    nodes, elements, theta = structured_cylindrical_mesh(4, 8, 0.5, 1.0)
    x_theta = torch.stack((nodes[elements[0], 0], theta[elements[0]]), dim=1)
    M = cylindrical_shell4_mass_matrix(x_theta, 0.5, 7800.0, 0.005)
    assert M.shape == (24, 24)
    assert torch.allclose(M, M.T, atol=1e-12)
    eigenvalues = torch.linalg.eigvalsh(M)
    assert (eigenvalues > 0).all()


def test_mass_matrix_trace():
    """Element mass matrix trace matches expected value for consistent mass."""
    R, t, rho = 0.5, 0.005, 7800.0
    nodes, elements, theta = structured_cylindrical_mesh(4, 8, R, 1.0)
    x_theta = torch.stack((nodes[elements[0], 0], theta[elements[0]]), dim=1)
    M = cylindrical_shell4_mass_matrix(x_theta, R, rho, t)

    # For consistent mass with bilinear shape functions:
    # trace = 4 nodes * 6 DOFs * (rho*t*area/9 for translation + rho*t^3/108 for rotation)
    # Simplified: trace ≈ 1.333 * rho * t * area (translational) + small rotary term
    elem_area = (1.0/4) * (2*math.pi*R/8)
    expected_trans_trace = 4/3 * rho * t * elem_area  # 1.333 * mass
    trans_trace = sum(M[6*a+d, 6*a+d].item() for a in range(4) for d in range(3))

    # Within 5% tolerance
    assert abs(trans_trace - expected_trans_trace) / expected_trans_trace < 0.05


def test_dmv_frequency_positive():
    """DMV frequencies are positive for valid inputs."""
    f = dmv_frequency_hz(radius=0.5, length=1.0, thickness=0.005,
                         young=210e9, poisson=0.3, density=7800.0,
                         n_axial=1, n_circ=2)
    assert f > 0


def test_dmv_frequency_increases_with_mode():
    """Higher modes have higher frequencies."""
    f1 = dmv_frequency_hz(radius=0.5, length=1.0, thickness=0.005,
                          young=210e9, poisson=0.3, density=7800.0,
                          n_axial=1, n_circ=2)
    f2 = dmv_frequency_hz(radius=0.5, length=1.0, thickness=0.005,
                          young=210e9, poisson=0.3, density=7800.0,
                          n_axial=2, n_circ=2)
    assert f2 > f1


def test_shell_modal_analysis_runs():
    """Full pipeline completes without error."""
    result = cylindrical_shell_modal_analysis(
        radius=0.5, length=1.0, thickness=0.005,
        young=210e9, poisson=0.3, density=7800.0,
        n_axial_mesh=4, n_circ_mesh=8, n_modes=5,
    )
    assert len(result.frequencies_hz) == 5
    assert (result.frequencies_hz > 0).all()
    assert result.n_nodes > 0
    assert result.n_elements > 0
    assert result.n_dof > 0
    assert result.n_free_dof > 0


def test_shell_frequencies_in_reasonable_range():
    """FEM frequencies are in the expected range for the shell geometry."""
    result = cylindrical_shell_modal_analysis(
        radius=0.5, length=1.0, thickness=0.005,
        young=210e9, poisson=0.3, density=7800.0,
        n_axial_mesh=4, n_circ_mesh=8, n_modes=10,
    )
    for f in result.frequencies_hz[:5]:
        assert 10 < f < 500, f"Frequency {f} Hz out of expected range"


def test_nasa_reference_data():
    """NASA TN D-3939 reference data is present and reasonable."""
    assert len(NASA_TN_D3939_FREQS) == 3
    assert all(f > 0 for f in NASA_TN_D3939_FREQS)
    assert NASA_TN_D3939_FREQS[0] < NASA_TN_D3939_FREQS[1] < NASA_TN_D3939_FREQS[2]
