"""Standalone verification of the Kerr (three-parameter) elastic foundation.

The Kerr model [1] extends the Winkler foundation by introducing an intermediate
shear layer coupled to a lower spring bed, yielding three parameters that reduce
to an equivalent Pasternak form (k_eq, G_eq).

Verification anchors
--------------------
* (a) Kerr(G_p=0) → Winkler(k_eq) element stiffness to machine precision
* (b) Kerr reduction algebra: k_eq, G_eq formulas verified
* (c) Kerr element stiffness is symmetric positive semi-definite
* (d) Kerr shear layer increases stiffness monotonically with G_p
* (e) Negative controls: invalid inputs fail closed

References
----------
[1] Kerr, A.D., "Elastic and Viscoelastic Foundation Models", J. Appl. Mech.,
    31(2), 1964, pp. 263-268.
"""
from __future__ import annotations

import pytest
import torch

from tensorfem.kerr_foundation import q4_kerr_foundation_stiffness
from tensorfem.winkler_foundation import q4_winkler_foundation_stiffness
from tensorfem.foundation_plate_bending import kerr_equivalent_pasternak
from tensorfem.plate import structured_square_mesh

DTYPE = torch.float64

# Kerr parameters (representative soil-structure values)
K_UPPER = 4.0e7   # upper spring stiffness, N/m³
K_LOWER = 2.0e7   # lower spring stiffness, N/m³
G_SHEAR = 3.0e7   # shear layer parameter, N/m

# Equivalent Pasternak parameters
K_EQ, G_EQ = kerr_equivalent_pasternak(K_UPPER, K_LOWER, G_SHEAR)


# ---------------------------------------------------------------------------
# (a) Kerr(G_p=0) → Winkler(k_eq) element stiffness
# ---------------------------------------------------------------------------

class TestKerrToWinklerReduction:
    """Kerr with zero shear layer must reproduce Winkler with series spring."""

    def test_element_stiffness_matches_winkler(self):
        """Kerr(G_p=0) element stiffness == Winkler(k_eq) to machine precision."""
        xy = torch.tensor(
            [[0.0, 0.0], [0.4, 0.0], [0.4, 0.4], [0.0, 0.4]], dtype=DTYPE
        )
        ke_kerr = q4_kerr_foundation_stiffness(xy, K_EQ, 0.0)
        ke_winkler = q4_winkler_foundation_stiffness(xy, K_EQ)
        assert torch.allclose(ke_kerr, ke_winkler, atol=1e-12), (
            "Kerr(G_p=0) ≠ Winkler(k_eq)"
        )


# ---------------------------------------------------------------------------
# (b) Kerr reduction algebra
# ---------------------------------------------------------------------------

class TestKerrReductionAlgebra:
    """Kerr → Pasternak equivalence algebra.

    Limits (from docstring of kerr_equivalent_pasternak):
    - G_p = 0 → Winkler k_eq
    - k_u → ∞ (plate tied to shear layer) → Pasternak (k_l, G_p)
    - k_l → ∞ (rigid base pins shear layer) → Winkler k_u with G_eq → 0
    """

    def test_equivalent_parameters(self):
        """k_eq = k_u*k_l/(k_u+k_l), G_eq = k_u*G_p/(k_u+k_l)."""
        k_eq, g_eq = kerr_equivalent_pasternak(K_UPPER, K_LOWER, G_SHEAR)
        assert k_eq == pytest.approx(K_UPPER * K_LOWER / (K_UPPER + K_LOWER), rel=1e-14)
        assert g_eq == pytest.approx(K_UPPER * G_SHEAR / (K_UPPER + K_LOWER), rel=1e-14)

    def test_k_u_large_gives_pasternak(self):
        """k_u >> k_l: plate tied to shear layer → Pasternak (k_l, G_p)."""
        k_eq, g_eq = kerr_equivalent_pasternak(1000 * K_LOWER, K_LOWER, G_SHEAR)
        assert k_eq == pytest.approx(K_LOWER, rel=1e-3)
        assert g_eq == pytest.approx(G_SHEAR, rel=1e-3)

    def test_k_l_large_gives_winkler(self):
        """k_l >> k_u: rigid base pins shear layer → Winkler k_u, G_eq → 0.

        With k_l = 1000 * k_u:
            G_eq = k_u * G_p / (k_u + 1000*k_u) = G_p / 1001 ≈ 0.001 * G_p
        This is much smaller than G_p but not exactly zero.
        """
        k_eq, g_eq = kerr_equivalent_pasternak(K_UPPER, 1000 * K_UPPER, G_SHEAR)
        assert k_eq == pytest.approx(K_UPPER, rel=1e-3)
        # G_eq = G_SHEAR / 1001 ≈ 29970, which is ~0.1% of G_SHEAR
        expected_g_eq = G_SHEAR / 1001.0
        assert g_eq == pytest.approx(expected_g_eq, rel=1e-10)
        # Verify it's much smaller than G_SHEAR
        assert g_eq < 0.01 * G_SHEAR, f"G_eq={g_eq} should be < 1% of G_SHEAR={G_SHEAR}"

    def test_g_p_zero_gives_winkler(self):
        """G_p=0: G_eq=0, k_eq = series spring."""
        k_eq, g_eq = kerr_equivalent_pasternak(K_UPPER, K_LOWER, 0.0)
        assert g_eq == pytest.approx(0.0, abs=1e-14)
        assert k_eq == pytest.approx(K_UPPER * K_LOWER / (K_UPPER + K_LOWER), rel=1e-14)

    def test_springs_in_series_soften(self):
        """k_eq < min(k_u, k_l): springs in series always soften."""
        k_eq, _ = kerr_equivalent_pasternak(K_UPPER, K_LOWER, G_SHEAR)
        assert k_eq < min(K_UPPER, K_LOWER)


# ---------------------------------------------------------------------------
# (c) Kerr element stiffness is symmetric positive semi-definite
# ---------------------------------------------------------------------------

class TestKerrElementProperties:
    """Kerr element stiffness matrix properties."""

    def _element_coords(self):
        return torch.tensor(
            [[0.0, 0.0], [0.4, 0.0], [0.4, 0.4], [0.0, 0.4]], dtype=DTYPE
        )

    def test_symmetry(self):
        """Kerr element stiffness must be symmetric."""
        xy = self._element_coords()
        ke = q4_kerr_foundation_stiffness(xy, K_EQ, G_EQ)
        assert torch.allclose(ke, ke.T, atol=1e-12), "Kerr stiffness not symmetric"

    def test_positive_semidefinite(self):
        """Kerr element stiffness must be positive semi-definite (within float64 noise)."""
        xy = self._element_coords()
        ke = q4_kerr_foundation_stiffness(xy, K_EQ, G_EQ)
        eigenvalues = torch.linalg.eigvalsh(ke)
        # Allow float64 round-off: eigenvalues should be >= -1e-7 * max_eigenvalue
        max_ev = float(eigenvalues.max())
        assert float(eigenvalues.min()) >= -1e-7 * max_ev, (
            f"Kerr stiffness has significantly negative eigenvalue: {eigenvalues.min():.6e}"
        )

    def test_zero_energy_for_constant_w_shear_part(self):
        """Constant transverse displacement carries no shear-layer energy."""
        xy = self._element_coords()
        ke_shear = q4_kerr_foundation_stiffness(xy, 0.0, G_EQ)
        u = torch.zeros(12, dtype=DTYPE)
        u[0::3] = 1.0  # w=1, rotations=0
        energy_shear = 0.5 * (u @ ke_shear @ u).item()
        # Allow float64 round-off
        max_ke = float(ke_shear.abs().max())
        assert abs(energy_shear) < 1e-8 * max_ke, (
            f"Rigid body shear energy {energy_shear:.6e} too large"
        )


# ---------------------------------------------------------------------------
# (d) Kerr shear layer increases stiffness monotonically
# ---------------------------------------------------------------------------

class TestKerrShearStiffening:
    """Increasing G_p must monotonically increase diagonal stiffness."""

    def test_diagonal_increases_with_gp(self):
        """Higher shear parameter → larger diagonal entries."""
        xy = torch.tensor(
            [[0.0, 0.0], [0.4, 0.0], [0.4, 0.4], [0.0, 0.4]], dtype=DTYPE
        )
        diagonals = []
        for gp in [0.0, 1.0e7, G_EQ, 5.0e7]:
            ke = q4_kerr_foundation_stiffness(xy, K_EQ, gp)
            diagonals.append(float(ke.diag().sum()))

        for i in range(len(diagonals) - 1):
            assert diagonals[i] < diagonals[i + 1], (
                f"Diagonal sum not monotone: {diagonals[i]:.6e} → {diagonals[i+1]:.6e}"
            )


# ---------------------------------------------------------------------------
# (e) Negative controls
# ---------------------------------------------------------------------------

class TestKerrNegativeControls:
    """Invalid inputs must fail closed."""

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError):
            q4_kerr_foundation_stiffness(
                torch.tensor([[0, 0], [1, 0]], dtype=DTYPE),
                K_EQ, G_EQ
            )

    def test_negative_winkler_raises(self):
        xy = torch.tensor(
            [[0.0, 0.0], [0.4, 0.0], [0.4, 0.4], [0.0, 0.4]], dtype=DTYPE
        )
        with pytest.raises(ValueError):
            q4_kerr_foundation_stiffness(xy, -1.0, G_EQ)

    def test_negative_shear_gp_raises(self):
        xy = torch.tensor(
            [[0.0, 0.0], [0.4, 0.0], [0.4, 0.4], [0.0, 0.4]], dtype=DTYPE
        )
        with pytest.raises(ValueError):
            q4_kerr_foundation_stiffness(xy, K_EQ, -1.0)


# ---------------------------------------------------------------------------
# (f) Integration with plate element: full system check
# ---------------------------------------------------------------------------

class TestKerrPlateIntegration:
    """Kerr foundation integrated with Mindlin plate element."""

    def test_assembled_system_is_symmetric(self):
        """Plate + Kerr foundation stiffness must be symmetric."""
        from tensorfem.plate import q4_mindlin_stiffness

        nodes, elements = structured_square_mesh(4, 1.0, dtype=DTYPE)
        ndof = 3 * nodes.shape[0]
        K = torch.zeros((ndof, ndof), dtype=DTYPE)

        for conn in elements:
            xy = nodes[conn, :2]
            ke = q4_mindlin_stiffness(xy, 2.1e11, 0.3, 0.05, shear_treatment="sri")
            ke += q4_kerr_foundation_stiffness(xy, K_EQ, G_EQ)
            dofs = torch.stack((3 * conn, 3 * conn + 1, 3 * conn + 2), dim=1).reshape(-1)
            K[dofs[:, None], dofs] += ke

        assert torch.allclose(K, K.T, atol=1e-8), "Assembled K not symmetric"

    def test_kerr_deflection_less_than_winkler(self):
        """Kerr (with shear) must give smaller deflection than pure Winkler."""
        from tensorfem.plate import q4_mindlin_stiffness

        nodes, elements = structured_square_mesh(8, 1.0, dtype=DTYPE)
        ndof = 3 * nodes.shape[0]

        def solve(winkler_k, shear_gp):
            K = torch.zeros((ndof, ndof), dtype=DTYPE)
            F = torch.zeros(ndof, dtype=DTYPE)
            for conn in elements:
                xy = nodes[conn, :2]
                ke = q4_mindlin_stiffness(xy, 2.1e11, 0.3, 0.05, shear_treatment="sri")
                ke += q4_kerr_foundation_stiffness(xy, winkler_k, shear_gp)
                dofs = torch.stack((3 * conn, 3 * conn + 1, 3 * conn + 2), dim=1).reshape(-1)
                K[dofs[:, None], dofs] += ke
            elem_area = (1.0 / 8) ** 2
            for conn in elements:
                F[3 * conn] += 1.0e4 * elem_area / 4.0
            # Simply supported: w=0 on edges
            fixed = []
            tol = 1.0 / 8 * 0.01
            for i, (x, y) in enumerate(nodes.tolist()):
                if abs(x) < tol or abs(x - 1.0) < tol or abs(y) < tol or abs(y - 1.0) < tol:
                    fixed.append(3 * i)
            fixed = sorted(set(fixed))
            free = [d for d in range(ndof) if d not in fixed]
            u_f = torch.linalg.solve(K[free][:, free], F[free])
            u = torch.zeros(ndof, dtype=DTYPE)
            u[free] = u_f
            centre = torch.argmin((nodes[:, 0] - 0.5).abs() + (nodes[:, 1] - 0.5).abs())
            return u[3 * centre].item()

        w_winkler = solve(K_EQ, 0.0)
        w_kerr = solve(K_EQ, G_EQ)
        assert abs(w_kerr) < abs(w_winkler), (
            f"Kerr deflection |{w_kerr:.6e}| >= Winkler |{w_winkler:.6e}|"
        )
