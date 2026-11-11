"""Regression gates for the Robinson patch test A/B/C (``tensorfem.patch_test``).

Each test encodes one clause of the criterion, including the *negative* controls:
a gate that cannot fail is not a gate, so the Q4/bending case and the distorted-Q8
case are asserted to be non-exact, with the measured magnitude pinned.
"""

import pytest
import torch

from tensorfem.patch_test import (
    EXACT_TOL,
    LENGTH,
    YOUNG,
    KAPPA,
    boundary_nodes_2d,
    exact_field,
    numbering_study,
    q4_patch_mesh,
    q8_patch_mesh,
    run_q4_patch,
    run_q8_patch,
    run_tet10_patch,
    tet10_patch_mesh,
)


@pytest.fixture(autouse=True)
def _double_precision():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def test_mesh_is_irregular_and_boundary_is_exact():
    nodes, elements = q4_patch_mesh(3, 3, 1.0)
    assert nodes.shape == (16, 2) and elements.shape == (9, 4)
    interior = [5, 6, 9, 10]
    grid_x = torch.tensor([0.0, 1 / 3, 2 / 3, 1.0])
    moved = sum(1 for n in interior
                for k in (0, 1)
                if abs(nodes[n, k] - grid_x[(n % 4) if k == 0 else (n // 4)]) > 1e-6)
    assert moved >= 4                      # interior nodes are genuinely off-grid
    bnd = boundary_nodes_2d(nodes)
    assert len(bnd) == 12                  # 16 nodes - 4 interior


def test_case_A_uniaxial_patch_is_exact():
    r = run_q4_patch("A")
    assert r.disp_error <= EXACT_TOL
    assert r.stress_error <= EXACT_TOL
    assert r.passed


def test_case_B_pure_shear_patch_is_exact():
    r = run_q4_patch("B")
    assert r.disp_error <= EXACT_TOL
    assert r.stress_error <= EXACT_TOL


def test_case_C_bending_is_NOT_reproduced_by_bilinear_Q4():
    """Negative control: the test must be able to fail."""
    r = run_q4_patch("C")
    assert r.stress_error > 1e-2            # measured ~1e-1 relative to E*kappa*L
    assert not r.passed


def test_case_C_is_exact_for_Q8_on_an_affine_mesh():
    r = run_q8_patch("C", 3, 0.0)
    assert r.disp_error <= 1e-11
    assert r.stress_error <= 1e-10
    assert r.passed


def test_case_C_Q8_distorted_mesh_degrades_but_is_bounded():
    """Documented serendipity limitation: not exact, but small and mesh-driven."""
    r = run_q8_patch("C", 3, 1.0)
    assert 1e-4 < r.stress_error < 5e-2
    assert not r.passed                    # honest: the patch test is NOT passed
    # the same distorted mesh still reproduces the constant-strain cases exactly
    for case in ("A", "B"):
        a = run_q8_patch(case, 3, 1.0)
        assert a.stress_error <= 1e-10


def test_case_C3_tet10_linear_stress_is_exact_on_distorted_tets():
    r = run_tet10_patch(2, 1.0)
    assert r.disp_error <= 1e-12
    assert r.stress_error <= 1e-10
    assert r.passed


def test_patch_answer_is_numbering_independent():
    out = numbering_study("A")
    assert out["identical"]


def test_exact_targets_are_analytical():
    """The reference values are the closed forms, not tabulated numbers."""
    nodes, _ = q4_patch_mesh(2, 2, 0.0)
    u, sigma_fn, scale, _, _, _ = exact_field("C", nodes)
    # sigma_xx = -E*kappa*y exactly, and the displacement is quadratic
    for p in (torch.tensor([0.3, 0.7]), torch.tensor([1.0, 1.0])):
        assert abs(float(sigma_fn(p)[0]) + YOUNG * KAPPA * float(p[1])) < 1e-10
    assert abs(float(u[0, 0])) < 1e-15
    assert abs(scale - YOUNG * KAPPA * LENGTH) < 1e-9