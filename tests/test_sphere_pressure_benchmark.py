"""Verification tests for the HEX8 spherical pressure benchmark."""
import torch
import pytest
from tensorfem.sphere_pressure_benchmark import solve_sphere_pressure, verify_sphere_pressure, PROBLEM

D = torch.float64


def test_verify_sphere_pressure_fails_on_modified_displacement():
    """verify_sphere_pressure should fail if displacement is modified."""
    case = solve_sphere_pressure(n=4, radial_layers=2)
    # Modify displacement slightly
    case["displacement"] = [[u[0] + 1e-3, u[1], u[2]] for u in case["displacement"]]
    with pytest.raises(ValueError):
        verify_sphere_pressure(case)


def test_verify_sphere_pressure_fails_on_modified_mesh():
    """verify_sphere_pressure should fail if mesh is modified."""
    case = solve_sphere_pressure(n=4, radial_layers=2)
    # Modify nodes slightly
    case["nodes"] = [[n[0] + 1e-3, n[1], n[2]] for n in case["nodes"]]
    with pytest.raises(ValueError, match="sphere mesh mismatch"):
        verify_sphere_pressure(case)


def test_verify_sphere_pressure_fails_on_modified_stress():
    """verify_sphere_pressure should fail if stress is modified."""
    case = solve_sphere_pressure(n=4, radial_layers=2)
    # Modify stress slightly
    case["element_stress"] = [[s[0] + 1e3, s[1], s[2], s[3], s[4], s[5]] for s in case["element_stress"]]
    with pytest.raises(ValueError):
        verify_sphere_pressure(case)
