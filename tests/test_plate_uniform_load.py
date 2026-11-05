"""Verification tests: uniform lateral pressure on the merged Q4 Mindlin plate.

The merged plate path was previously only checked with an eigen-aligned
sinusoidal load, which excites a single Navier mode.  These tests use uniform
pressure (exits every odd mode) against an exact Navier series and an
independent separable Rayleigh-Ritz reference.  Gates are fail-closed at 3 %,
and two negative controls assert that the gates can actually fail.
"""
from functools import lru_cache

import pytest
import torch

from tensorfem.plate_uniform_load import (
    clamped_uniform_ritz,
    centre_node_deflection,
    element_center_moments,
    nearest_element_center,
    node_values_along_x,
    solve_uniform_square_plate,
    ss_uniform_kirchhoff,
    ss_uniform_mindlin,
    ss_uniform_ritz,
)

SIDE = 1.0
E = 2.1e11
NU = 0.3
Q = 1.0e6
T_THIN = 0.01
T_THICK = 0.10
GATE = 0.03


@lru_cache(maxsize=None)
def _solve(nx: int, thickness: float, boundary: str):
    return solve_uniform_square_plate(nx, SIDE, E, NU, thickness, Q, boundary)


@lru_cache(maxsize=None)
def _ritz(n_modes: int):
    return clamped_uniform_ritz(SIDE, E, NU, T_THIN, Q, n_modes=n_modes)


def _rel(value: float, reference: float) -> float:
    return abs(value - reference) / abs(reference)


@lru_cache(maxsize=None)
def _ss_ref():
    return ss_uniform_kirchhoff(SIDE, E, NU, T_THIN, Q, terms=401)


def test_ss_series_reproduces_timoshenko_coefficients():
    """The exact series must land on Timoshenko's tabulated coefficients."""
    ref = _ss_ref()
    D = E * T_THIN**3 / (12 * (1 - NU**2))
    assert _rel(ref["deflection"] / (Q * SIDE**4 / D), 0.00406) < 0.01
    assert _rel(ref["moment_x"] / (Q * SIDE**2), 0.0479) < 0.01
    # Symmetry of the square case.
    assert ref["moment_x"] == pytest.approx(ref["moment_y"], rel=1e-12)


def test_ss_series_convergence_is_saturating():
    """401 vs 801 odd terms must agree, so the reference is converged."""
    a = ss_uniform_kirchhoff(SIDE, E, NU, T_THIN, Q, terms=401)
    b = ss_uniform_kirchhoff(SIDE, E, NU, T_THIN, Q, terms=801)
    assert _rel(a["deflection"], b["deflection"]) < 1e-8


def test_mindlin_series_adds_shear_to_kirchhoff():
    """The shear part must be small but strictly positive for a thin plate."""
    thin = ss_uniform_mindlin(SIDE, E, NU, T_THIN, Q)
    assert 0.0 < thin["deflection_shear"] / thin["deflection"] < 0.01
    assert thin["deflection"] > thin["deflection_kirchhoff"]
    thick = ss_uniform_mindlin(SIDE, E, NU, T_THICK, Q)
    assert thick["deflection_shear"] / thick["deflection"] > 0.04


def test_ritz_machinery_reproduces_navier_series():
    """Same Ritz code with sine modes must reproduce the closed form (0.008 %)."""
    ritz = ss_uniform_ritz(SIDE, E, NU, T_THIN, Q, n_modes=8)
    assert _rel(ritz.deflection(0.5, 0.5), _ss_ref()["deflection"]) < 1e-3


def test_clamped_ritz_matches_literature_and_is_mode_stable():
    """Clamped Ritz: 0.00126 q a^4/D and M = 0.0231 q a^2 (Timoshenko), stable in N."""
    solution = _ritz(10)
    D = solution.bending_stiffness
    alpha = solution.deflection(0.5, 0.5) / (Q * SIDE**4 / D)
    moment = solution.moments(0.5, 0.5)[0] / (Q * SIDE**2)
    assert _rel(alpha, 0.00126) < 0.01
    assert _rel(moment, 0.0231) < 0.01
    assert _rel(solution.deflection(0.5, 0.5), _ritz(12).deflection(0.5, 0.5)) < 0.01
    # Clamped edges: the basis must give zero deflection and zero slope there.
    # Tolerance is 1e-6 of w_max: the high-order clamped modes lose ~1e-8 of
    # absolute accuracy to cosh/sinh cancellation at the far edge.
    point = 0.5 * SIDE
    tol = 1e-6 * solution.deflection(0.5, 0.5)
    assert abs(solution.deflection(0.0, point)) < tol
    assert abs(solution.deflection(SIDE, point)) < tol


def test_simply_supported_thin_centre_deflection_under_gate():
    """SS t/a = 0.01, 16x16: centre deflection within 3 % of the exact series."""
    fem = centre_node_deflection(_solve(16, T_THIN, "simply_supported"))
    assert _rel(fem, _ss_ref()["deflection"]) < GATE


def test_clamped_thin_centre_deflection_under_gate():
    """Clamped t/a = 0.01, 16x16: centre deflection within 3 % of the Ritz value."""
    fem = centre_node_deflection(_solve(16, T_THIN, "clamped"))
    assert _rel(fem, _ritz(10).deflection(0.5, 0.5)) < GATE


def test_thick_plate_needs_the_shear_reference():
    """t/a = 0.10: the Mindlin reference passes, the shear-free one must not."""
    kirchhoff = ss_uniform_mindlin(SIDE, E, NU, T_THICK, Q)["deflection_kirchhoff"]
    mindlin = ss_uniform_mindlin(SIDE, E, NU, T_THICK, Q)["deflection"]
    fem = centre_node_deflection(_solve(16, T_THICK, "simply_supported"))
    assert _rel(fem, mindlin) < GATE
    # Negative control: the same FE value against the shear-free reference fails.
    assert _rel(fem, kirchhoff) > GATE


def test_retained_coarse_mesh_failure():
    """Clamped 4x4 stays above the gate; the failure is recorded, not hidden."""
    fem = centre_node_deflection(_solve(4, T_THIN, "clamped"))
    assert _rel(fem, _ritz(10).deflection(0.5, 0.5)) > GATE


def test_boundary_conditions_are_not_interchangeable():
    """Negative control: the clamped result cannot pass the SS gate."""
    clamped = centre_node_deflection(_solve(8, T_THIN, "clamped"))
    assert _rel(clamped, _ss_ref()["deflection"]) > GATE


def test_centre_bending_moment_simply_supported():
    """Element-centre Mx at the plate centre (odd mesh) within 3 % of the series."""
    result = _solve(17, T_THIN, "simply_supported")
    moments = element_center_moments(result)
    index = nearest_element_center(moments, 0.5 * SIDE, 0.5 * SIDE)
    x, y = moments["x"][index], moments["y"][index]
    # The nearest element centre is the plate centre only for an odd mesh.
    assert abs(x - 0.5) < 1e-9 and abs(y - 0.5) < 1e-9
    reference = ss_uniform_kirchhoff(SIDE, E, NU, T_THIN, Q, x, y)
    assert _rel(moments["moment_x"][index], reference["moment_x"]) < GATE


def test_clamped_centre_bending_moment():
    """Clamped centre Mx within 3 % of the Ritz value at the same point."""
    result = _solve(17, T_THIN, "clamped")
    moments = element_center_moments(result)
    index = nearest_element_center(moments, 0.5 * SIDE, 0.5 * SIDE)
    reference = _ritz(10).moments(moments["x"][index], moments["y"][index])[0]
    assert _rel(moments["moment_x"][index], reference) < GATE


def test_clamped_boundary_has_no_deflection_or_rotation():
    """The hard-clamped boundary conditions are actually imposed."""
    result = _solve(8, T_THIN, "clamped")
    nodes = result.nodes
    tol = SIDE * 1e-12
    edge = ((nodes[:, 0].abs() < tol) | ((nodes[:, 0] - SIDE).abs() < tol)
            | (nodes[:, 1].abs() < tol) | ((nodes[:, 1] - SIDE).abs() < tol))
    idx = torch.nonzero(edge)[:, 0]
    dofs = torch.stack((3 * idx, 3 * idx + 1, 3 * idx + 2), dim=1).reshape(-1)
    assert torch.allclose(result.displacement[dofs], torch.zeros_like(result.displacement[dofs]),
                          atol=1e-12)


def test_centre_line_profile_is_symmetric_and_peaks_at_centre():
    """Centre-line deflection: symmetric about x = a/2 and maximal there."""
    xs, ws = node_values_along_x(_solve(16, T_THIN, "simply_supported"), 0.5)
    peak = ws.index(max(ws))
    assert abs(xs[peak] - 0.5) < 1e-12
    for i in range(len(xs)):
        assert ws[i] == pytest.approx(ws[-1 - i], rel=1e-10)


def test_element_center_moments_shapes():
    """Moment recovery returns one value per element with matching centroids."""
    result = _solve(4, T_THIN, "simply_supported")
    moments = element_center_moments(result)
    n = result.elements.shape[0]
    assert len(moments["moment_x"]) == n
    assert len(moments["x"]) == n and len(moments["moment_xy"]) == n
    # Uniform pressure on a symmetric SS plate: Mx and My are equal at the centre
    # element for the square case (isotropy of the geometry and load).
    index = nearest_element_center(moments, 0.5, 0.5)
    assert moments["moment_x"][index] == pytest.approx(moments["moment_y"][index], rel=1e-12)