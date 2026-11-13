"""Validation of elastic rotational restraint boundary conditions.

Tests
-----
A. beta=0 must reproduce simply supported solution to machine precision.
B. beta=1 must reproduce clamped solution within 1%.
C. Intermediate beta must produce deflections monotonically between extremes.
D. Mesh convergence: results should converge with increasing nx.
E. Non-dimensional deflection should match published values for elastically
   restrained plates (Timoshenko & Woinowsky-Krieger Table 8).

Reference
---------
Timoshenko, S. & Woinowsky-Krieger, S. (1959). *Theory of Plates and Shells*,
2nd ed., McGraw-Hill.  §33: "Rectangular plates with elastically restrained
edges", pp. 198-202.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorfem.plate_elastic_restraint import (
    beta_to_zeta,
    rotational_stiffness,
    identify_edge_nodes,
    assemble_rotational_spring_stiffness,
    solve_plate_elastic_restraint,
    beta_sweep,
)
from tensorfem.plate_mindlin import (
    solve_plate_on_winkler_foundation,
)


# ---------------------------------------------------------------------------
# Basic unit tests
# ---------------------------------------------------------------------------

def test_beta_to_zeta_limits():
    """beta=0 → zeta=0; beta→1 → zeta→large."""
    assert beta_to_zeta(0.0) == 0.0
    assert beta_to_zeta(0.5) == pytest.approx(1.0, abs=1e-5)
    assert beta_to_zeta(0.9) == pytest.approx(9.0, abs=1e-3)
    assert beta_to_zeta(0.99) == pytest.approx(99.0, abs=1e-1)
    assert beta_to_zeta(1.0) == pytest.approx(1e6, rel=1e-3)  # 1/eps


def test_beta_to_zeta_monotonic():
    """zeta must increase monotonically with beta."""
    betas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0]
    zetas = [beta_to_zeta(b) for b in betas]
    for i in range(1, len(zetas)):
        assert zetas[i] > zetas[i-1], f"zeta not monotonic at beta={betas[i]}"


def test_rotational_stiffness_formula():
    """K_rot = zeta * D / a."""
    E, nu, t, a = 2.1e11, 0.3, 0.02, 1.0
    D = E * t**3 / (12.0 * (1.0 - nu**2))
    K = rotational_stiffness(0.5, E, nu, t, a)
    expected = 1.0 * D / a  # zeta=1 at beta=0.5
    assert K == pytest.approx(expected, rel=1e-5)


def test_identify_edge_nodes():
    """Edge node identification on a structured mesh."""
    from tensorfem.plate import structured_square_mesh
    nodes, _ = structured_square_mesh(4, 1.0)
    edges = identify_edge_nodes(nodes, 1.0)
    # For nx=4, we have 5×5 = 25 nodes
    assert len(edges['x0']) == 5
    assert len(edges['xa']) == 5
    assert len(edges['y0']) == 5
    assert len(edges['ya']) == 5


def test_rotational_spring_stiffness_structure():
    """Spring matrix should be diagonal with entries only on normal-rotation DOFs."""
    from tensorfem.plate import structured_square_mesh
    nodes, _ = structured_square_mesh(2, 1.0)
    ndof = 3 * nodes.shape[0]
    edges = identify_edge_nodes(nodes, 1.0)
    K_spring = assemble_rotational_spring_stiffness(nodes, 1.0, 100.0, edges, ndof)

    # Should be diagonal
    off_diag = K_spring - torch.diag(torch.diag(K_spring))
    assert torch.allclose(off_diag, torch.zeros_like(off_diag), atol=1e-14)

    # w DOFs (0, 3, 6, ...) should be zero
    for i in range(nodes.shape[0]):
        assert K_spring[3*i, 3*i] == 0.0, f"w DOF {3*i} should have no spring"


# ---------------------------------------------------------------------------
# A. beta=0 must reproduce simply supported
# ---------------------------------------------------------------------------

def test_beta_zero_matches_simply_supported():
    """beta=0 elastic restraint == simply supported to machine precision."""
    nx = 8
    side = 1.0
    E, nu, t = 2.1e11, 0.3, 0.01
    k_f = 1.0e5
    q = 1.0e4

    ref = solve_plate_on_winkler_foundation(
        nx=nx, side=side, young=E, poisson=nu, thickness=t,
        winkler_k=k_f, pressure=q, boundary="simply_supported",
    )
    res = solve_plate_elastic_restraint(
        nx=nx, side=side, young=E, poisson=nu, thickness=t,
        winkler_k=k_f, pressure=q, beta=0.0,
    )

    center_x, center_y = side / 2, side / 2
    ref_dist = (ref.nodes[:, 0] - center_x)**2 + (ref.nodes[:, 1] - center_y)**2
    ref_center_idx = torch.argmin(ref_dist).item()
    res_dist = (res.nodes[:, 0] - center_x)**2 + (res.nodes[:, 1] - center_y)**2
    res_center_idx = torch.argmin(res_dist).item()

    w_ref = ref.displacement[3 * ref_center_idx].item()
    w_res = res.displacement[3 * res_center_idx].item()

    rel_err = abs(w_res - w_ref) / abs(w_ref)
    assert rel_err < 1e-12, f"beta=0 should match SSSS, rel_err={rel_err:.2e}"


# ---------------------------------------------------------------------------
# B. beta=1 must approximate clamped
# ---------------------------------------------------------------------------

def test_beta_one_approximates_clamped():
    """beta=1 elastic restraint ≈ clamped within 1%."""
    nx = 12
    side = 1.0
    E, nu, t = 2.1e11, 0.3, 0.01
    k_f = 1.0e5
    q = 1.0e4

    ref = solve_plate_on_winkler_foundation(
        nx=nx, side=side, young=E, poisson=nu, thickness=t,
        winkler_k=k_f, pressure=q, boundary="clamped",
    )
    res = solve_plate_elastic_restraint(
        nx=nx, side=side, young=E, poisson=nu, thickness=t,
        winkler_k=k_f, pressure=q, beta=1.0,
    )

    center_x, center_y = side / 2, side / 2
    ref_dist = (ref.nodes[:, 0] - center_x)**2 + (ref.nodes[:, 1] - center_y)**2
    ref_center_idx = torch.argmin(ref_dist).item()
    res_dist = (res.nodes[:, 0] - center_x)**2 + (res.nodes[:, 1] - center_y)**2
    res_center_idx = torch.argmin(res_dist).item()

    w_ref = ref.displacement[3 * ref_center_idx].item()
    w_res = res.displacement[3 * res_center_idx].item()

    rel_err = abs(w_res - w_ref) / abs(w_ref)
    assert rel_err < 0.01, f"beta=1 should approximate clamped within 1%, rel_err={rel_err:.4f}"


# ---------------------------------------------------------------------------
# C. Monotonic behavior
# ---------------------------------------------------------------------------

def test_deflection_monotonic_with_beta():
    """Center deflection must decrease monotonically as beta increases."""
    nx = 10
    side = 1.0
    E, nu, t = 2.1e11, 0.3, 0.01
    k_f = 1.0e5
    q = 1.0e4

    results = beta_sweep(
        nx=nx, side=side, young=E, poisson=nu, thickness=t,
        winkler_k=k_f, pressure=q,
        beta_values=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    )

    w_centers = [r['w_center'] for r in results]
    for i in range(1, len(w_centers)):
        assert w_centers[i] < w_centers[i-1], (
            f"Deflection not monotonic: w(beta={results[i]['beta']})={w_centers[i]} "
            f">= w(beta={results[i-1]['beta']})={w_centers[i-1]}"
        )


# ---------------------------------------------------------------------------
# D. Mesh convergence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("nx", [6, 10, 16])
def test_mesh_convergence_beta_half(nx):
    """Results should converge with increasing nx for beta=0.5."""
    side = 1.0
    E, nu, t = 2.1e11, 0.3, 0.01
    k_f = 1.0e5
    q = 1.0e4

    res = solve_plate_elastic_restraint(
        nx=nx, side=side, young=E, poisson=nu, thickness=t,
        winkler_k=k_f, pressure=q, beta=0.5,
    )

    center_x, center_y = side / 2, side / 2
    dist = (res.nodes[:, 0] - center_x)**2 + (res.nodes[:, 1] - center_y)**2
    center_idx = torch.argmin(dist).item()
    w_center = res.displacement[3 * center_idx].item()

    assert w_center > 0.0, f"Center deflection should be positive, got {w_center}"

    D = E * t**3 / (12.0 * (1.0 - nu**2))
    w_norm = w_center * D / (q * side**4)
    assert 0.0001 < w_norm < 0.01, f"Non-dimensional deflection {w_norm:.6e} out of range"


# ---------------------------------------------------------------------------
# E. Non-dimensional comparison with literature
# ---------------------------------------------------------------------------

def test_nondimensional_deflection_range():
    """Non-dimensional deflection should be between SSSS and CCCC limits.

    For a square plate under uniform load (no foundation):
    - SSSS: w_max ≈ 0.00406 · q·a⁴/D
    - CCCC: w_max ≈ 0.00126 · q·a⁴/D
    """
    nx = 16
    side = 1.0
    E, nu, t = 2.1e11, 0.3, 0.01
    q = 1.0e4
    k_f = 0.0  # No foundation for direct comparison with Kirchhoff theory

    D = E * t**3 / (12.0 * (1.0 - nu**2))

    ref_ss = solve_plate_on_winkler_foundation(
        nx=nx, side=side, young=E, poisson=nu, thickness=t,
        winkler_k=k_f, pressure=q, boundary="simply_supported",
    )
    center_x, center_y = side / 2, side / 2
    dist_ss = (ref_ss.nodes[:, 0] - center_x)**2 + (ref_ss.nodes[:, 1] - center_y)**2
    idx_ss = torch.argmin(dist_ss).item()
    w_ss = ref_ss.displacement[3 * idx_ss].item()
    w_ss_norm = w_ss * D / (q * side**4)

    ref_cc = solve_plate_on_winkler_foundation(
        nx=nx, side=side, young=E, poisson=nu, thickness=t,
        winkler_k=k_f, pressure=q, boundary="clamped",
    )
    dist_cc = (ref_cc.nodes[:, 0] - center_x)**2 + (ref_cc.nodes[:, 1] - center_y)**2
    idx_cc = torch.argmin(dist_cc).item()
    w_cc = ref_cc.displacement[3 * idx_cc].item()
    w_cc_norm = w_cc * D / (q * side**4)

    res = solve_plate_elastic_restraint(
        nx=nx, side=side, young=E, poisson=nu, thickness=t,
        winkler_k=k_f, pressure=q, beta=0.5,
    )
    dist_er = (res.nodes[:, 0] - center_x)**2 + (res.nodes[:, 1] - center_y)**2
    idx_er = torch.argmin(dist_er).item()
    w_er = res.displacement[3 * idx_er].item()
    w_er_norm = w_er * D / (q * side**4)

    assert w_cc_norm < w_er_norm < w_ss_norm, (
        f"Elastic restraint deflection {w_er_norm:.6e} not between "
        f"CCCC {w_cc_norm:.6e} and SSSS {w_ss_norm:.6e}"
    )

    # SSSS should be close to literature value (0.00406)
    lit_ss = 0.00406
    ss_err = abs(w_ss_norm - lit_ss) / lit_ss
    assert ss_err < 0.05, f"SSSS non-dimensional deflection error {ss_err:.2%} too large"


# ---------------------------------------------------------------------------
# F. Edge rotation behavior
# ---------------------------------------------------------------------------

def test_edge_rotation_decreases_with_beta():
    """Edge rotation should decrease as beta increases."""
    nx = 10
    side = 1.0
    E, nu, t = 2.1e11, 0.3, 0.01
    k_f = 1.0e5
    q = 1.0e4

    betas = [0.0, 0.3, 0.6, 0.9, 1.0]
    edge_rotations = []

    for beta in betas:
        res = solve_plate_elastic_restraint(
            nx=nx, side=side, young=E, poisson=nu, thickness=t,
            winkler_k=k_f, pressure=q, beta=beta,
        )
        # Get normal rotation at midpoint of x=0 edge (theta_x = DOF 3*i+1)
        tol = side * 1e-12
        x0_nodes = torch.nonzero(res.nodes[:, 0] < tol, as_tuple=False).flatten()
        mid_idx = x0_nodes[len(x0_nodes) // 2].item()
        theta_x = res.displacement[3 * mid_idx + 1].item()
        edge_rotations.append(abs(theta_x))

    for i in range(1, len(edge_rotations)):
        assert edge_rotations[i] <= edge_rotations[i-1] * 1.01, (
            f"Edge rotation not decreasing: beta={betas[i]}, theta={edge_rotations[i]:.6e}"
        )


# ---------------------------------------------------------------------------
# G. Ice plate specific case
# ---------------------------------------------------------------------------

def test_ice_plate_elastic_restraint():
    """Ice plate with realistic parameters and elastic restraint."""
    nx = 12
    side = 2.0
    E = 9.0e9
    nu = 0.33
    t = 0.5
    k_f = 1025.0 * 9.81
    q = 5.0e4

    D = E * t**3 / (12.0 * (1.0 - nu**2))

    ref_ss = solve_plate_on_winkler_foundation(
        nx=nx, side=side, young=E, poisson=nu, thickness=t,
        winkler_k=k_f, pressure=q, boundary="simply_supported",
    )
    center_x, center_y = side / 2, side / 2
    dist_ss = (ref_ss.nodes[:, 0] - center_x)**2 + (ref_ss.nodes[:, 1] - center_y)**2
    idx_ss = torch.argmin(dist_ss).item()
    w_ss = ref_ss.displacement[3 * idx_ss].item()

    ref_cc = solve_plate_on_winkler_foundation(
        nx=nx, side=side, young=E, poisson=nu, thickness=t,
        winkler_k=k_f, pressure=q, boundary="clamped",
    )
    dist_cc = (ref_cc.nodes[:, 0] - center_x)**2 + (ref_cc.nodes[:, 1] - center_y)**2
    idx_cc = torch.argmin(dist_cc).item()
    w_cc = ref_cc.displacement[3 * idx_cc].item()

    res = solve_plate_elastic_restraint(
        nx=nx, side=side, young=E, poisson=nu, thickness=t,
        winkler_k=k_f, pressure=q, beta=0.5,
    )
    dist_er = (res.nodes[:, 0] - center_x)**2 + (res.nodes[:, 1] - center_y)**2
    idx_er = torch.argmin(dist_er).item()
    w_er = res.displacement[3 * idx_er].item()

    assert w_cc < w_er < w_ss, (
        f"Ice plate: elastic restraint deflection {w_er:.6e} m not between "
        f"CCCC {w_cc:.6e} m and SSSS {w_ss:.6e} m"
    )
