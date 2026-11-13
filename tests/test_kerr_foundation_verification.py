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
* (g) Exact two-field Kerr model (independent reference) vs closed form: O(h^2)
      convergence, <1% centre deflection error, lower-field ratio ``w_1/w``
* (h) Reduction bias ``p_eq/p_exact = 1 + 2 lambda G_p/(k_u+k_l)`` to machine
      precision, and the measured validity range of ``kerr_equivalent_pasternak``
      (<3% inside it, >15% once ``G_p`` leaves it)
* (i) ``k_l -> inf`` limit: ``(k_eq, G_eq) -> (k_u, G_p)`` and the bias vanishes
* (j) Closed-form quadrature identities of the element blocks (constant and
      linear fields)

References
----------
[1] Kerr, A.D., "Elastic and Viscoelastic Foundation Models", J. Appl. Mech.,
    31(2), 1964, pp. 263-268.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorfem.kerr_foundation import q4_kerr_foundation_stiffness
from tensorfem.winkler_foundation import q4_winkler_foundation_stiffness
from tensorfem.foundation_plate_bending import kerr_equivalent_pasternak
from tensorfem.plate import _shape, q4_mindlin_stiffness, structured_square_mesh

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


# ---------------------------------------------------------------------------
# (g)-(j) Excess verification: the exact two-field Kerr model vs the
#          three-parameter -> Pasternak reduction
#
# ``tensorfem.kerr_foundation`` implements the single-field *Pasternak* operator
# ``p = k w - G_p lap w``, and ``kerr_equivalent_pasternak`` reduces Kerr's
# three-parameter idealisation ``(k_u, k_l, G_p)`` to it "to first order in
# G_p".  Neither the element anchors (a)-(e) nor the full-plate anchors of
# ``tests/test_kerr_plate_verification.py`` can see that reduction error: they
# only ever exercise one field.  The anchors below add the *exact* Kerr model as
# an independent reference and quantify the reduction, so the accuracy claim of
# the production reduction becomes a measured number.
#
# Exact Kerr model (Kerr 1964; k_u at the plate, k_l at the rigid base, shear
# layer G_p carrying the lower field w_1), total potential energy
#
#     Pi = 1/2 int k_u (w - w_1)^2 + 1/2 int k_l w_1^2
#          + 1/2 int G_p |grad w_1|^2 - int q w
#
# so the three stiffness blocks are, element-wise, exactly the production
# foundation element evaluated with three different parameter triples:
#
#     K_ww   = k_u int N^T N                        = _kerr(xy, k_u, 0)
#     K_w1w1 = (k_u+k_l) int N^T N + G_p int gradN^T gradN
#                                                   = _kerr(xy, k_u+k_l, G_p)
#     K_ww1  = -k_u int N^T N                       = -_kerr(xy, k_u, 0)
#
# i.e. the reference model reuses the production element instead of
# re-implementing it.  For a simply supported plate under the single sinusoidal
# load ``q0 sin(pi x/L) sin(pi y/L)`` (``lap -> -2 lambda``, ``lambda =
# (pi/L)^2``) the exact model has the closed form
#
#     w_1 = k_u w / (k_u + k_l + 2 G_p lambda),
#     p   = k_u (k_l + 2 G_p lambda) / (k_u + k_l + 2 G_p lambda) * w,
#     w   = q0 / (4 D lambda^2 + p/w),
#
# while the equivalent Pasternak model of ``kerr_equivalent_pasternak`` gives
#
#     p_eq = k_eq + 2 G_eq lambda = (k_u/(k_u+k_l)) (k_l + 2 G_p lambda).
#
# The ratio is an exact algebraic factor,
#
#     p_eq / p_exact = 1 + 2 lambda G_p / (k_u + k_l)   (=: reduction bias),
#
# which is 1 for G_p = 0 and -> 1 for k_l -> inf, and which grows linearly in
# G_p: the reduction is first-order accurate in G_p, exactly as documented.
# ---------------------------------------------------------------------------

KERR_L = 1.0
KERR_E = 2.1e11
KERR_NU = 0.3
KERR_T = 0.01
KERR_Q0 = 1.0e4
KERR_D = KERR_E * KERR_T**3 / (12.0 * (1.0 - KERR_NU**2))
KERR_LAMBDA = (math.pi / KERR_L) ** 2
KERR_W_LOCAL = (0, 3, 6, 9)


def _kerr_sinusoidal_load(nodes, elements):
    """Consistent load vector for ``q0 sin(pi x/L) sin(pi y/L)`` (2x2 Gauss)."""
    f = torch.zeros(3 * nodes.shape[0], dtype=DTYPE)
    gq = 1.0 / 3.0**0.5
    for conn in elements:
        xy = nodes[conn]
        fe = torch.zeros(12, dtype=DTYPE)
        for xi_f in (-gq, gq):
            for eta_f in (-gq, gq):
                N, dN_nat = _shape(nodes.new_tensor(xi_f), nodes.new_tensor(eta_f))
                J = dN_nat.T @ xy
                p = N @ xy
                q = KERR_Q0 * math.sin(math.pi * p[0].item() / KERR_L) * math.sin(
                    math.pi * p[1].item() / KERR_L
                )
                fe[0::3] += N * q * torch.linalg.det(J)
        dofs = torch.stack((3 * conn, 3 * conn + 1, 3 * conn + 2), dim=1).reshape(-1)
        f[dofs] += fe
    return f


def _simply_supported_dofs(nodes):
    """Simply supported dofs (w on all edges, tangential rotations) + edge nodes."""
    tol = KERR_L * 1e-12
    xedge = (nodes[:, 0].abs() < tol) | ((nodes[:, 0] - KERR_L).abs() < tol)
    yedge = (nodes[:, 1].abs() < tol) | ((nodes[:, 1] - KERR_L).abs() < tol)
    ix = torch.nonzero(xedge, as_tuple=False).flatten()
    iy = torch.nonzero(yedge, as_tuple=False).flatten()
    ie = torch.nonzero(xedge | yedge, as_tuple=False).flatten()
    fixed = torch.cat((3 * ie, 3 * iy + 1, 3 * ix + 2)).unique()
    return fixed, ie


def _kerr_two_field_solve(n, k_upper, k_lower, shear_gp):
    """Exact two-field Kerr plate; returns ``(w_centre, w1_centre)``.

    The lower field ``w_1`` is pinned at the supported edge, which is the
    boundary condition of the Navier-type closed form it is compared against.
    """
    nodes, elements = structured_square_mesh(n, KERR_L, dtype=DTYPE)
    nnode = nodes.shape[0]
    ndof = 4 * nnode
    K = torch.zeros((ndof, ndof), dtype=DTYPE)
    f = torch.zeros(ndof, dtype=DTYPE)
    f[: 3 * nnode] = _kerr_sinusoidal_load(nodes, elements)
    wl = list(KERR_W_LOCAL)
    for conn in elements:
        xy = nodes[conn]
        k_upper_block = q4_kerr_foundation_stiffness(xy, k_upper, 0.0)
        k_lower_block = q4_kerr_foundation_stiffness(xy, k_upper + k_lower, shear_gp)
        dw = torch.stack((3 * conn, 3 * conn + 1, 3 * conn + 2), dim=1).reshape(-1)
        K[dw[:, None], dw] += q4_mindlin_stiffness(xy, KERR_E, KERR_NU, KERR_T)
        K[dw[:, None], dw] += k_upper_block
        d1 = 3 * nnode + conn
        K[d1[:, None], d1] += k_lower_block[wl][:, wl]
        K[dw[wl][:, None], d1] += -k_upper_block[wl][:, wl]
        K[d1[:, None], dw[wl]] += -k_upper_block[wl][:, wl]
    fixed, iedge = _simply_supported_dofs(nodes)
    fixed = torch.cat((fixed, 3 * nnode + iedge)).unique()
    free = torch.arange(ndof)
    free = free[~torch.isin(free, fixed)]
    u = torch.zeros(ndof, dtype=DTYPE)
    u[free] = torch.linalg.solve(K[free[:, None], free], f[free])
    c = (n // 2) * (n + 1) + n // 2
    return float(u[3 * c]), float(u[3 * nnode + c])


def _kerr_two_field_exact(k_upper, k_lower, shear_gp):
    """Closed-form centre deflection of the exact two-field Kerr plate."""
    lam = KERR_LAMBDA
    pressure_coef = (
        k_upper
        * (k_lower + 2.0 * shear_gp * lam)
        / (k_upper + k_lower + 2.0 * shear_gp * lam)
    )
    return KERR_Q0 / (4.0 * KERR_D * lam**2 + pressure_coef)


def _kerr_two_field_w1_ratio(k_upper, k_lower, shear_gp):
    """Closed-form ``w_1 / w`` of the exact two-field model (homogeneous ratio)."""
    return k_upper / (k_upper + k_lower + 2.0 * shear_gp * KERR_LAMBDA)


def _equivalent_pasternak_deflection(k_upper, k_lower, shear_gp):
    """Centre deflection of the production ``kerr_equivalent_pasternak`` model."""
    k_eq, g_eq = kerr_equivalent_pasternak(k_upper, k_lower, shear_gp)
    return KERR_Q0 / (4.0 * KERR_D * KERR_LAMBDA**2 + k_eq + 2.0 * g_eq * KERR_LAMBDA)


def _reduction_bias(k_upper, k_lower, shear_gp):
    """Exact single-mode pressure bias of the reduction, ``p_eq/p_exact``."""
    return 1.0 + 2.0 * KERR_LAMBDA * shear_gp / (k_upper + k_lower)


class TestExactTwoFieldKerrModel:
    """The exact three-parameter model and its reduction to Pasternak."""

    def test_two_field_fe_matches_closed_form(self):
        """Exact Kerr FE converges to the two-field closed form (<1% at n=16)."""
        k_u, k_l, g_p = 4.0e7, 2.0e7, 1.0e6
        w_fem, _ = _kerr_two_field_solve(16, k_u, k_l, g_p)
        w_exact = _kerr_two_field_exact(k_u, k_l, g_p)
        assert abs(w_fem / w_exact - 1.0) < 1.0e-2, (
            f"two-field Kerr FE {w_fem:.6e} vs closed form {w_exact:.6e}"
        )

    def test_two_field_fe_convergence_order(self):
        """h-refinement must show the O(h^2) rate of the Q4 Mindlin plate."""
        k_u, k_l, g_p = 4.0e7, 2.0e7, 1.0e6
        w_exact = _kerr_two_field_exact(k_u, k_l, g_p)
        e_coarse = abs(_kerr_two_field_solve(8, k_u, k_l, g_p)[0] / w_exact - 1.0)
        e_fine = abs(_kerr_two_field_solve(16, k_u, k_l, g_p)[0] / w_exact - 1.0)
        order = math.log(e_coarse / e_fine) / math.log(2.0)
        assert 1.6 < order < 2.6, f"observed convergence order {order:.2f}"

    def test_shear_layer_deflection_ratio(self):
        """The lower field follows ``w_1 = k_u w / (k_u+k_l+2 G_p lambda)``."""
        k_u, k_l, g_p = 4.0e7, 2.0e7, 1.0e6
        w_fem, w1_fem = _kerr_two_field_solve(8, k_u, k_l, g_p)
        ratio_fem = w1_fem / w_fem
        ratio_exact = _kerr_two_field_w1_ratio(k_u, k_l, g_p)
        assert 0.0 < ratio_fem < 1.0, "lower field must deflect less than the plate"
        assert abs(ratio_fem / ratio_exact - 1.0) < 5.0e-3, (
            f"w1/w FE {ratio_fem:.5f} vs closed form {ratio_exact:.5f}"
        )

    def test_gp_zero_collapses_to_series_spring(self):
        """G_p = 0: exact model, reduction and Winkler(k_eq) all coincide."""
        k_u, k_l, g_p = 4.0e7, 2.0e7, 0.0
        k_eq, g_eq = kerr_equivalent_pasternak(k_u, k_l, g_p)
        w_exact = _kerr_two_field_exact(k_u, k_l, g_p)
        w_reduced = _equivalent_pasternak_deflection(k_u, k_l, g_p)
        w_winkler = KERR_Q0 / (4.0 * KERR_D * KERR_LAMBDA**2 + k_eq)
        assert g_eq == 0.0
        assert abs(w_reduced / w_exact - 1.0) < 1e-14
        assert abs(w_winkler / w_exact - 1.0) < 1e-14
        w_fem, _ = _kerr_two_field_solve(8, k_u, k_l, g_p)
        assert abs(w_fem / w_exact - 1.0) < 2.0e-2


class TestKerrReductionBias:
    """Quantitative accuracy of ``kerr_equivalent_pasternak`` (the reduction)."""

    # representative soil/structure triple used by the element anchors above
    K_U, K_L = 4.0e7, 2.0e7

    def test_bias_identity_is_exact(self):
        """``p_eq/p_exact = 1 + 2 lambda G_p/(k_u+k_l)`` to machine precision.

        This identity is what makes the reduction *first order* in G_p; it is
        built from the production ``kerr_equivalent_pasternak`` only.
        """
        lam = KERR_LAMBDA
        for g_p in (0.0, 1.0e5, 1.0e6, 1.0e7):
            k_eq, g_eq = kerr_equivalent_pasternak(self.K_U, self.K_L, g_p)
            p_eq = k_eq + 2.0 * g_eq * lam
            p_exact = (
                self.K_U
                * (self.K_L + 2.0 * g_p * lam)
                / (self.K_U + self.K_L + 2.0 * g_p * lam)
            )
            bias = _reduction_bias(self.K_U, self.K_L, g_p)
            assert abs(p_eq / p_exact - bias) < 1e-13, (
                f"G_p={g_p:.1e}: p_eq/p_exact={p_eq / p_exact:.12f} vs bias {bias:.12f}"
            )

    def test_reduction_validity_range(self):
        """Measured reduction error: <3% in its validity range, >15% outside it.

        The deflection error is ``(bias-1) p_exact / (4 D lambda^2 + p_exact)``.
        With the representative triple it stays under the platform's 3%
        acceptance line at ``G_p = 1e5`` and already exceeds the 15% rejection
        line at ``G_p = 1e6``, i.e. the reduction is only usable for
        ``G_p << (k_u + k_l) / (2 lambda)``.
        """
        g_p_small, g_p_large = 1.0e5, 1.0e6
        err_small = abs(
            _equivalent_pasternak_deflection(self.K_U, self.K_L, g_p_small)
            / _kerr_two_field_exact(self.K_U, self.K_L, g_p_small)
            - 1.0
        )
        err_large = abs(
            _equivalent_pasternak_deflection(self.K_U, self.K_L, g_p_large)
            / _kerr_two_field_exact(self.K_U, self.K_L, g_p_large)
            - 1.0
        )
        assert _reduction_bias(self.K_U, self.K_L, g_p_small) < 1.05
        assert err_small < 3.0e-2, f"validity-range error {err_small:.3%} exceeds 3%"
        assert err_large > 0.15, (
            f"expected the reduction to leave its validity range at G_p={g_p_large:.0e}, "
            f"measured error {err_large:.3%}"
        )
        # the FE two-field model must reproduce the same loss of accuracy
        w_fe_exact, _ = _kerr_two_field_solve(8, self.K_U, self.K_L, g_p_large)
        fe_ratio = w_fe_exact / _equivalent_pasternak_deflection(self.K_U, self.K_L, g_p_large)
        analytic_ratio = _kerr_two_field_exact(self.K_U, self.K_L, g_p_large) / (
            _equivalent_pasternak_deflection(self.K_U, self.K_L, g_p_large)
        )
        assert abs(fe_ratio / analytic_ratio - 1.0) < 0.02, (
            f"FE deflection ratio {fe_ratio:.4f} vs closed form {analytic_ratio:.4f}"
        )

    def test_reduction_is_exact_as_lower_spring_stiffens(self):
        """``k_l -> inf``: ``(k_eq, G_eq) -> (k_u, 0)`` and the bias vanishes.

        A rigid base pins the shear layer, so the spreading vanishes and the
        reduction becomes exact (Winkler ``k_u``) -- the documented limit.
        """
        g_p = 1.0e6
        previous = None
        for ratio in (1.0, 1.0e2, 1.0e4, 1.0e6):
            k_l = self.K_U * ratio
            k_eq, g_eq = kerr_equivalent_pasternak(self.K_U, k_l, g_p)
            err = abs(
                _equivalent_pasternak_deflection(self.K_U, k_l, g_p)
                / _kerr_two_field_exact(self.K_U, k_l, g_p)
                - 1.0
            )
            # k_eq -> k_u and G_eq -> 0 (in series with the stiffening base)
            assert abs(k_eq / self.K_U - 1.0) < 2.0 / ratio
            g_eq_exact = g_p * self.K_U / (self.K_U + k_l)
            assert abs(g_eq / g_eq_exact - 1.0) < 1e-12
            assert abs(_reduction_bias(self.K_U, k_l, g_p) - 1.0) < 2.0 / ratio
            if previous is not None:
                assert err < previous, f"error not decreasing at k_l/k_u={ratio:.0e}"
            previous = err
        assert previous < 1.0e-3, f"limit error {previous:.3%} not below 0.1%"


class TestKerrFoundationElementExactness:
    """Closed-form quadrature identities of the production element blocks."""

    def _element(self):
        return torch.tensor(
            [[0.0, 0.0], [0.3, 0.05], [0.35, 0.4], [0.05, 0.3]], dtype=DTYPE
        )

    @staticmethod
    def _area(xy):
        """Area of the bilinear quad as the sum of its two node triangles."""
        t1 = torch.linalg.det(torch.stack((xy[1] - xy[0], xy[2] - xy[0])))
        t2 = torch.linalg.det(torch.stack((xy[2] - xy[0], xy[3] - xy[0])))
        return 0.5 * abs(float(t1)) + 0.5 * abs(float(t2))

    def test_constant_field_winkler_energy_exact(self):
        """Constant ``w``: ``1/2 u^T K u = 1/2 k A w^2`` exactly (2x2 Gauss)."""
        xy = self._element()
        k, w = 3.0e7, 1.0e-3
        ke = q4_kerr_foundation_stiffness(xy, k, 0.0)
        u = torch.zeros(12, dtype=DTYPE)
        u[0::3] = w
        energy = 0.5 * float(u @ ke @ u)
        expected = 0.5 * k * self._area(xy) * w * w
        assert abs(energy / expected - 1.0) < 1e-12, (
            f"Winkler energy {energy:.12e} vs 1/2 k A w^2 {expected:.12e}"
        )

    def test_linear_field_shear_energy_exact(self):
        """Linear ``w``: shear energy equals ``1/2 G_p A |grad w|^2`` exactly."""
        xy = self._element()
        g_p = 5.0e6
        gx, gy = 2.0e-3, -3.0e-3
        ke = q4_kerr_foundation_stiffness(xy, 0.0, g_p)
        u = torch.zeros(12, dtype=DTYPE)
        u[0::3] = gx * xy[:, 0] + gy * xy[:, 1]
        energy = 0.5 * float(u @ ke @ u)
        expected = 0.5 * g_p * self._area(xy) * (gx * gx + gy * gy)
        assert abs(energy / expected - 1.0) < 1e-12, (
            f"shear energy {energy:.12e} vs 1/2 G_p A |grad w|^2 {expected:.12e}"
        )

    def test_rotational_dofs_stay_uncoupled(self):
        """The foundation blocks act on transverse DOFs only."""
        xy = self._element()
        ke = q4_kerr_foundation_stiffness(xy, 1.0e7, 2.0e6)
        rot = [1, 2, 4, 5, 7, 8, 10, 11]
        assert float(ke[rot].abs().sum()) == 0.0
        assert float(ke[:, rot].abs().sum()) == 0.0