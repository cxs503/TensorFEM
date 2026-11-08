"""Verification ladder: Winkler → Kerr → Buoyancy coupling for plate foundations.

This test suite validates the TensorFEM plate-on-elastic-foundation solvers
against analytical reference solutions, establishing a three-tier verification
ladder:

    Tier 1: Winkler foundation (independent springs)
    Tier 2: Kerr two-parameter foundation (springs + shear layer)
    Tier 3: Hydrostatic buoyancy coupling (floating plate)

Plus cross-cutting verification:
    - Mesh convergence (GCI < 2%)
    - Kirchhoff vs Mindlin thin-plate limit (< 1%)

Acceptance criteria (from campaign P27W2-N05):
    (a) Winkler center deflection vs Navier/Hetényi < 2%
    (b) Kerr center deflection vs analytical < 3%
    (c) Buoyancy coupling center deflection vs analytical < 3%
    (d) GCI < 2%
    (e) Kirchhoff vs Mindlin thin-plate limit < 1%
    (f) ≥ 15 tests all green

References
----------
[1] Hetényi, M., "Beams on Elastic Foundation", 1946.
[2] Timoshenko & Woinowsky-Krieger, "Theory of Plates and Shells", 2nd ed.
[3] Kerr, A.D., "Elastic and Viscoelastic Foundation Models", JAM 1964.
[4] Roache, P.J., "Perspective: A Method for Uniform Reporting of Grid
    Refinement Studies", JFE 1994.
"""
from __future__ import annotations

import math
import pytest
import torch

from src.tensorfem.plate import (
    q4_mindlin_stiffness,
    structured_square_mesh,
    _shape,
)
from src.tensorfem.plate_mindlin import (
    solve_plate_on_winkler_foundation,
    q4_winkler_foundation_stiffness,
    consistent_load_vector,
)
from src.tensorfem.kerr_foundation import (
    q4_kerr_foundation_stiffness,
    assemble_kerr_foundation,
)
from src.tensorfem.buoyancy_coupling import (
    buoyancy_coefficient,
    characteristic_length,
    solve_floating_plate_point_load,
    analytical_floating_plate_center_deflection,
)
from src.tensorfem.plate_foundation_analytical import (
    navier_ss_plate_uniform,
    navier_ss_plate_point_load,
    kerr_ss_plate_uniform_approx,
    kirchhoff_ss_center,
    mindlin_ss_center_correction,
    gci_order_of_convergence,
)

DTYPE = torch.float64


# ============================================================================
# Tier 1: Winkler Foundation Verification
# ============================================================================

class TestWinklerFoundation:
    """Tier 1: Winkler elastic foundation – uniform load on SS plate."""

    def test_winkler_center_deflection_vs_navier_2percent(self):
        """(a) Winkler center deflection vs Navier analytical < 2%.

        Simply supported square plate, uniform load q, Winkler foundation k.
        Navier double-sine series is the exact analytical solution.
        """
        E, nu, h = 1.0e7, 0.3, 0.01
        a = 1.0
        q = 1000.0
        k = 1.0e5
        D = E * h ** 3 / (12 * (1 - nu ** 2))

        # Analytical (Navier series, high resolution)
        w_analytical = navier_ss_plate_uniform(q, a, a, D, k, M=200, N=200)

        # FEM (Mindlin Q4, 12×12 mesh)
        result = solve_plate_on_winkler_foundation(
            nx=12, side=a, young=E, poisson=nu, thickness=h,
            winkler_k=k, pressure=q, boundary="simply_supported",
        )
        w_fem = _get_center_deflection(result.displacement, result.nodes, a)

        rel_err = abs(w_fem - w_analytical) / abs(w_analytical) * 100
        assert rel_err < 2.0, (
            f"Winkler center deflection error {rel_err:.2f}% >= 2% "
            f"(FEM={w_fem:.6e}, analytical={w_analytical:.6e})"
        )

    def test_winkler_converges_to_free_plate_as_k_zero(self):
        """As k → 0, Winkler plate → free plate (no foundation)."""
        E, nu, h = 1.0e7, 0.3, 0.01
        a = 1.0
        q = 1000.0
        D = E * h ** 3 / (12 * (1 - nu ** 2))

        w_free_analytical = navier_ss_plate_uniform(q, a, a, D, 0.0, M=200, N=200)

        result = solve_plate_on_winkler_foundation(
            nx=12, side=a, young=E, poisson=nu, thickness=h,
            winkler_k=1e-6, pressure=q, boundary="simply_supported",
        )
        w_fem = _get_center_deflection(result.displacement, result.nodes, a)

        rel_err = abs(w_fem - w_free_analytical) / abs(w_free_analytical) * 100
        assert rel_err < 2.0, (
            f"k→0 limit error {rel_err:.2f}% (FEM={w_fem:.6e}, "
            f"free plate={w_free_analytical:.6e})"
        )

    def test_winkler_stiffness_symmetry(self):
        """Winkler foundation stiffness matrix must be symmetric."""
        xy = torch.tensor([[0., 0.], [1., 0.], [1., 1.], [0., 1.]], dtype=DTYPE)
        kw = q4_winkler_foundation_stiffness(xy, 1.0e6)
        assert torch.allclose(kw, kw.T, atol=1e-12)

    def test_winkler_stiffness_positive_semidefinite(self):
        """Winkler foundation stiffness must be positive semi-definite for k > 0."""
        xy = torch.tensor([[0., 0.], [1., 0.], [1., 1.], [0., 1.]], dtype=DTYPE)
        kw = q4_winkler_foundation_stiffness(xy, 1.0e6)
        eigvals = torch.linalg.eigvalsh(kw)
        assert bool((eigvals >= -1e-10).all())

    def test_winkler_higher_k_reduces_deflection(self):
        """Increasing Winkler k must monotonically reduce center deflection."""
        E, nu, h = 1.0e7, 0.3, 0.01
        a = 1.0
        q = 1000.0
        w_prev = float('inf')
        for k in [1e3, 1e4, 1e5]:
            result = solve_plate_on_winkler_foundation(
                nx=8, side=a, young=E, poisson=nu, thickness=h,
                winkler_k=k, pressure=q, boundary="simply_supported",
            )
            w = _get_center_deflection(result.displacement, result.nodes, a)
            assert w < w_prev, f"w(k={k})={w} >= w(k_prev)={w_prev}"
            w_prev = w


# ============================================================================
# Tier 2: Kerr Two-Parameter Foundation Verification
# ============================================================================

class TestKerrFoundation:
    """Tier 2: Kerr (three-parameter) foundation – springs + shear layer."""

    def test_kerr_reduces_to_winkler_when_gp_zero(self):
        """Kerr with Gp=0 must reproduce Winkler stiffness exactly."""
        xy = torch.tensor([[0., 0.], [1., 0.], [1., 1.], [0., 1.]], dtype=DTYPE)
        k = 1.0e5
        kw_winkler = q4_winkler_foundation_stiffness(xy, k)
        ke_kerr = q4_kerr_foundation_stiffness(xy, k, 0.0)
        assert torch.allclose(kw_winkler, ke_kerr, atol=1e-10), (
            "Kerr(Gp=0) ≠ Winkler"
        )

    def test_kerr_center_deflection_vs_analytical_3percent(self):
        """(b) Kerr center deflection vs Vlasov-type analytical < 3%.

        Simply supported plate, uniform load, Kerr foundation with Gp > 0.
        """
        E, nu, h = 1.0e7, 0.3, 0.01
        a = 1.0
        q = 1000.0
        k = 1.0e5
        Gp = 5.0e3  # shear parameter
        D = E * h ** 3 / (12 * (1 - nu ** 2))

        # Analytical (Vlasov-type Navier series for Kerr)
        w_analytical = kerr_ss_plate_uniform_approx(q, a, a, D, k, Gp, M=200, N=200)

        # FEM: assemble plate + Kerr foundation manually
        nx = 12
        nodes, elements = structured_square_mesh(nx, a, dtype=DTYPE)
        n_nodes = nodes.shape[0]
        ndof = 3 * n_nodes
        K = torch.zeros((ndof, ndof), dtype=DTYPE)
        f = torch.zeros(ndof, dtype=DTYPE)

        for conn in elements:
            xy = nodes[conn, :2]
            ke = q4_mindlin_stiffness(xy, E, nu, h)
            ke += q4_kerr_foundation_stiffness(xy, k, Gp)
            dofs = torch.stack((3*conn, 3*conn+1, 3*conn+2), dim=1).reshape(-1)
            K[dofs[:, None], dofs] += ke

            # Uniform load via consistent load vector
            fe = consistent_load_vector(xy, q)
            K_dofs = torch.stack((3*conn, 3*conn+1, 3*conn+2), dim=1).reshape(-1)
            f[K_dofs] += fe

        # Simply-supported BCs
        tol = a * 1e-12
        xedge = (nodes[:, 0].abs() < tol) | ((nodes[:, 0] - a).abs() < tol)
        yedge = (nodes[:, 1].abs() < tol) | ((nodes[:, 1] - a).abs() < tol)
        edge = xedge | yedge
        fixed = torch.cat((
            3 * torch.nonzero(edge, as_tuple=False).flatten(),
            3 * torch.nonzero(yedge, as_tuple=False).flatten() + 1,
            3 * torch.nonzero(xedge, as_tuple=False).flatten() + 2,
        )).unique()
        all_dofs = torch.arange(ndof)
        free = all_dofs[~torch.isin(all_dofs, fixed)]
        u_f = torch.linalg.solve(K[free][:, free], f[free])
        u = torch.zeros(ndof, dtype=DTYPE)
        u[free] = u_f

        w_fem = _get_center_deflection(u, nodes, a)

        rel_err = abs(w_fem - w_analytical) / abs(w_analytical) * 100
        assert rel_err < 3.0, (
            f"Kerr center deflection error {rel_err:.2f}% >= 3% "
            f"(FEM={w_fem:.6e}, analytical={w_analytical:.6e})"
        )

    def test_kerr_shear_increases_stiffness(self):
        """Adding shear parameter Gp > 0 must increase foundation stiffness."""
        xy = torch.tensor([[0., 0.], [1., 0.], [1., 1.], [0., 1.]], dtype=DTYPE)
        k = 1.0e5
        ke_winkler = q4_kerr_foundation_stiffness(xy, k, 0.0)
        ke_kerr = q4_kerr_foundation_stiffness(xy, k, 5.0e3)
        # Diagonal (w DOFs) should be larger with shear coupling
        w_dofs = [0, 3, 6, 9]
        for d in w_dofs:
            assert ke_kerr[d, d] >= ke_winkler[d, d] - 1e-10

    def test_kerr_stiffness_symmetry(self):
        """Kerr foundation stiffness must be symmetric."""
        xy = torch.tensor([[0., 0.], [1., 0.], [1., 1.], [0., 1.]], dtype=DTYPE)
        ke = q4_kerr_foundation_stiffness(xy, 1.0e5, 5.0e3)
        assert torch.allclose(ke, ke.T, atol=1e-10)


# ============================================================================
# Tier 3: Hydrostatic Buoyancy Coupling Verification
# ============================================================================

class TestBuoyancyCoupling:
    """Tier 3: Floating plate – hydrostatic buoyancy as Winkler foundation."""

    def test_buoyancy_coefficient_physical(self):
        """Buoyancy coefficient k_b = ρ_w·g must be physically reasonable."""
        k_b = buoyancy_coefficient(1025.0, 9.81)
        assert abs(k_b - 10055.25) < 1.0  # ≈ 10 kN/m³

    def test_characteristic_length_scaling(self):
        """Characteristic length L_c = (4D/k_b)^{1/4} must scale correctly."""
        D = 1.0
        k_b = 1.0
        Lc = characteristic_length(D, k_b)
        assert abs(Lc - 4.0**0.25) < 1e-12

    def test_buoyancy_center_deflection_vs_analytical_3percent(self):
        """(c) Floating plate center deflection under uniform load vs analytical < 3%.

        Simply supported floating plate, uniform load q.
        Buoyancy acts as Winkler foundation with k = ρ_w·g.
        """
        E, nu, h = 1.0e7, 0.3, 0.05
        a = 2.0
        q = 10000.0  # uniform load
        rho_w = 1025.0
        g = 9.81
        D = E * h ** 3 / (12 * (1 - nu ** 2))
        k_b = buoyancy_coefficient(rho_w, g)

        # Analytical (Navier series for uniform load on SS plate with foundation)
        w_analytical = navier_ss_plate_uniform(q, a, a, D, k_b, M=200, N=200)

        # FEM: assemble plate + buoyancy foundation manually
        nx = 12
        nodes, elements = structured_square_mesh(nx, a, dtype=DTYPE)
        n_nodes = nodes.shape[0]
        ndof = 3 * n_nodes
        K = torch.zeros((ndof, ndof), dtype=DTYPE)
        f = torch.zeros(ndof, dtype=DTYPE)

        for conn in elements:
            xy = nodes[conn, :2]
            ke = q4_mindlin_stiffness(xy, E, nu, h)
            ke += q4_winkler_foundation_stiffness(xy, k_b)
            fe = consistent_load_vector(xy, q)
            dofs = torch.stack((3*conn, 3*conn+1, 3*conn+2), dim=1).reshape(-1)
            K[dofs[:, None], dofs] += ke
            f[dofs] += fe

        # Simply-supported BCs
        tol = a * 1e-12
        xedge = (nodes[:, 0].abs() < tol) | ((nodes[:, 0] - a).abs() < tol)
        yedge = (nodes[:, 1].abs() < tol) | ((nodes[:, 1] - a).abs() < tol)
        edge = xedge | yedge
        fixed = torch.cat((
            3 * torch.nonzero(edge, as_tuple=False).flatten(),
            3 * torch.nonzero(yedge, as_tuple=False).flatten() + 1,
            3 * torch.nonzero(xedge, as_tuple=False).flatten() + 2,
        )).unique()
        all_dofs = torch.arange(ndof)
        free = all_dofs[~torch.isin(all_dofs, fixed)]
        u_f = torch.linalg.solve(K[free][:, free], f[free])
        u = torch.zeros(ndof, dtype=DTYPE)
        u[free] = u_f

        w_fem = _get_center_deflection(u, nodes, a)

        rel_err = abs(w_fem - w_analytical) / abs(w_analytical) * 100
        assert rel_err < 3.0, (
            f"Buoyancy center deflection error {rel_err:.2f}% >= 3% "
            f"(FEM={w_fem:.6e}, analytical={w_analytical:.6e})"
        )

    def test_buoyancy_deflection_decreases_with_density(self):
        """Higher water density → stiffer buoyancy → less deflection."""
        E, nu, h = 1.0e7, 0.3, 0.05
        a = 2.0
        P = 10000.0
        w_prev = float('inf')
        for rho in [500.0, 1000.0, 1025.0]:
            result = solve_floating_plate_point_load(
                nx=8, side=a, young=E, poisson=nu, thickness=h,
                water_density=rho, gravity=9.81,
                point_load=P, load_position=(a/2, a/2),
                boundary="simply_supported",
            )
            w = _get_center_deflection(result.displacement, result.nodes, a)
            assert w < w_prev, f"w(ρ={rho})={w} >= w_prev={w_prev}"
            w_prev = w


# ============================================================================
# Cross-cutting: Mesh Convergence (GCI)
# ============================================================================

class TestMeshConvergence:
    """Grid Convergence Index verification."""

    def test_gci_winkler_under_2percent(self):
        """(d) GCI for Winkler plate h-refinement < 2%.

        Three systematically refined meshes: 6×6, 8×8, 12×12.
        """
        E, nu, h = 1.0e7, 0.3, 0.01
        a = 1.0
        q = 1000.0
        k = 1.0e5

        w = []
        for nx in [6, 8, 12]:
            result = solve_plate_on_winkler_foundation(
                nx=nx, side=a, young=E, poisson=nu, thickness=h,
                winkler_k=k, pressure=q, boundary="simply_supported",
            )
            w.append(_get_center_deflection(result.displacement, result.nodes, a))

        # Refinement ratios
        r12 = 8.0 / 6.0  # coarse/medium
        r23 = 12.0 / 8.0  # medium/fine
        p, gci = gci_order_of_convergence(w[2], w[1], w[0], r12, r23)

        assert gci < 2.0, (
            f"GCI = {gci:.2f}% >= 2% (order p={p:.2f}, "
            f"w=[{w[0]:.6e}, {w[1]:.6e}, {w[2]:.6e}])"
        )

    def test_fine_mesh_accuracy_winkler(self):
        """Fine mesh (12x12) must achieve < 1% error for Winkler plate."""
        E, nu, h = 1.0e7, 0.3, 0.01
        a = 1.0
        q = 1000.0
        k = 1.0e5
        D = E * h ** 3 / (12 * (1 - nu ** 2))
        w_exact = navier_ss_plate_uniform(q, a, a, D, k, M=200, N=200)

        result = solve_plate_on_winkler_foundation(
            nx=12, side=a, young=E, poisson=nu, thickness=h,
            winkler_k=k, pressure=q, boundary="simply_supported",
        )
        w = _get_center_deflection(result.displacement, result.nodes, a)
        err = abs(w - w_exact) / abs(w_exact) * 100
        assert err < 1.0, f"Fine mesh error {err:.2f}% >= 1%"


# ============================================================================
# Cross-cutting: Kirchhoff vs Mindlin Thin-Plate Limit
# ============================================================================

class TestKirchhoffVsMindlin:
    """Thin-plate limit: Kirchhoff vs Mindlin agreement."""

    def test_thin_plate_limit_under_1percent(self):
        """(e) For h/L < 0.1, Mindlin vs Kirchhoff center deflection < 1%.

        Square SS plate, uniform load, no foundation.
        h/a = 0.01/1.0 = 0.01 << 0.1
        """
        E, nu = 1.0e7, 0.3
        a = 1.0
        q = 1000.0
        h = 0.01  # h/a = 0.01
        D = E * h ** 3 / (12 * (1 - nu ** 2))

        # Kirchhoff analytical
        w_kirchhoff = kirchhoff_ss_center(q, a, a, D, M=200, N=200)

        # Mindlin FEM (no foundation)
        result = solve_plate_on_winkler_foundation(
            nx=12, side=a, young=E, poisson=nu, thickness=h,
            winkler_k=0.0, pressure=q, boundary="simply_supported",
        )
        w_mindlin = _get_center_deflection(result.displacement, result.nodes, a)

        rel_err = abs(w_mindlin - w_kirchhoff) / abs(w_kirchhoff) * 100
        assert rel_err < 1.0, (
            f"Thin-plate limit error {rel_err:.2f}% >= 1% "
            f"(Mindlin={w_mindlin:.6e}, Kirchhoff={w_kirchhoff:.6e})"
        )

    def test_thick_plate_shows_shear_correction(self):
        """For h/a > 0.1, Mindlin should show transverse shear softening."""
        E, nu = 1.0e7, 0.3
        a = 1.0
        q = 1000.0
        h_thick = 0.15  # h/a = 0.15 > 0.1
        D_thick = E * h_thick ** 3 / (12 * (1 - nu ** 2))

        w_kirchhoff = kirchhoff_ss_center(q, a, a, D_thick, M=200, N=200)

        result = solve_plate_on_winkler_foundation(
            nx=10, side=a, young=E, poisson=nu, thickness=h_thick,
            winkler_k=0.0, pressure=q, boundary="simply_supported",
        )
        w_mindlin = _get_center_deflection(result.displacement, result.nodes, a)

        # Mindlin should deflect MORE than Kirchhoff (shear softening)
        assert w_mindlin > w_kirchhoff, (
            f"Thick plate: Mindlin ({w_mindlin:.6e}) should be > "
            f"Kirchhoff ({w_kirchhoff:.6e}) due to shear deformation"
        )


# ============================================================================
# Helper functions
# ============================================================================

def _get_center_deflection(displacement, nodes, side):
    """Extract center-node transverse deflection from FEM result."""
    n_nodes = nodes.shape[0]
    center_idx = 0
    min_dist = float('inf')
    for i in range(n_nodes):
        d = (nodes[i, 0] - side / 2) ** 2 + (nodes[i, 1] - side / 2) ** 2
        if d < min_dist:
            min_dist = d
            center_idx = i
    return displacement[3 * center_idx].item()
