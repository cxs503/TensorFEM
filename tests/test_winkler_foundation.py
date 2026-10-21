"""Unit tests for Winkler foundation, added mass, and Hetényi reference."""
import math
import pytest
import torch

from tensorfem.winkler_foundation import (
    q4_winkler_foundation_stiffness,
    assemble_winkler_foundation,
)
from tensorfem.added_mass import (
    q4_added_mass_consistent,
    assemble_added_mass_consistent,
    assemble_added_mass_lumped,
)
from tensorfem.hetenyi_reference import (
    hetenyi_A, hetenyi_B, hetenyi_C, hetenyi_D,
    infinite_beam_point_load,
    finite_beam_series,
    plate_strip_parameters,
)
from tensorfem.plate import q4_mindlin_stiffness, structured_square_mesh

D = torch.float64


# ---- Winkler foundation ----

class TestWinklerFoundation:
    def test_zero_foundation_returns_zeros(self):
        """k=0 must produce an identically zero 12x12 matrix."""
        xy = torch.tensor([[0., 0.], [1., 0.], [1., 1.], [0., 1.]], dtype=D)
        kw = q4_winkler_foundation_stiffness(xy, 0.0)
        assert kw.abs().max().item() == 0.0

    def test_k_to_zero_degenerate_to_free_plate(self):
        """As k -> 0, the plate solution should approach the free-plate solution."""
        nodes, elems = structured_square_mesh(4, 1.0, dtype=D)
        young, poisson, thickness = 1.0e7, 0.3, 0.01
        ndof = 3 * nodes.shape[0]

        K_nof = torch.zeros((ndof, ndof), dtype=D)
        for conn in elems:
            ke = q4_mindlin_stiffness(nodes[conn], young, poisson, thickness)
            dofs = torch.stack((3 * conn, 3 * conn + 1, 3 * conn + 2), dim=1).reshape(-1)
            K_nof[dofs[:, None], dofs] += ke

        K_with = torch.zeros((ndof, ndof), dtype=D)
        for conn in elems:
            ke = q4_mindlin_stiffness(nodes[conn], young, poisson, thickness)
            kw = q4_winkler_foundation_stiffness(nodes[conn], 1e-6)
            dofs = torch.stack((3 * conn, 3 * conn + 1, 3 * conn + 2), dim=1).reshape(-1)
            K_with[dofs[:, None], dofs] += ke + kw

        diff = (K_with - K_nof).abs().max()
        assert diff < 1.0

    def test_positive_definite_for_positive_k(self):
        """Winkler contribution should be positive semi-definite for k > 0."""
        xy = torch.tensor([[0., 0.], [1., 0.], [1., 1.], [0., 1.]], dtype=D)
        kw = q4_winkler_foundation_stiffness(xy, 1.0e6)
        eigvals = torch.linalg.eigvalsh(kw)
        assert bool((eigvals >= -1e-10).all())

    def test_assembly_shape(self):
        Kw = assemble_winkler_foundation(
            torch.tensor([[0., 0.], [1., 0.], [1., 1.], [0., 1.]], dtype=D),
            torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
            1.0e6,
        )
        assert Kw.shape == (12, 12)

    def test_config_switch_off_bit_aligned(self):
        """Foundation off (k=0) gives bit-identical stiffness to pure plate."""
        nodes, elems = structured_square_mesh(3, 1.0, dtype=D)
        young, poisson, thickness = 1.0e7, 0.3, 0.01
        ndof = 3 * nodes.shape[0]

        K_dry = torch.zeros((ndof, ndof), dtype=D)
        for conn in elems:
            ke = q4_mindlin_stiffness(nodes[conn], young, poisson, thickness)
            dofs = torch.stack((3 * conn, 3 * conn + 1, 3 * conn + 2), dim=1).reshape(-1)
            K_dry[dofs[:, None], dofs] += ke

        K_winkler_off = torch.zeros((ndof, ndof), dtype=D)
        for conn in elems:
            ke = q4_mindlin_stiffness(nodes[conn], young, poisson, thickness)
            kw = q4_winkler_foundation_stiffness(nodes[conn], 0.0)
            dofs = torch.stack((3 * conn, 3 * conn + 1, 3 * conn + 2), dim=1).reshape(-1)
            K_winkler_off[dofs[:, None], dofs] += ke + kw

        assert (K_dry == K_winkler_off).all().item()


# ---- Added mass ----

class TestAddedMass:
    def test_zero_added_mass_returns_zeros(self):
        """ma=0 must produce an identically zero 12x12 matrix."""
        xy = torch.tensor([[0., 0.], [1., 0.], [1., 1.], [0., 1.]], dtype=D)
        ma = q4_added_mass_consistent(xy, 0.0)
        assert ma.abs().max().item() == 0.0

    def test_switch_off_bit_aligned(self):
        """With added_mass=0, the returned matrix is identically zero.

        This guarantees that when the switch is off the dry mass matrix is
        reproduced bit-for-bit (the added-mass contribution is a true zero).
        """
        nodes, elems = structured_square_mesh(3, 1.0, dtype=D)
        ndof = 3 * nodes.shape[0]
        M_added = assemble_added_mass_consistent(nodes, elems, 0.0)
        zero = torch.zeros((ndof, ndof), dtype=D)
        assert (M_added == zero).all().item()

    def test_lumped_mass_diagonal_only(self):
        """Lumped added mass should be diagonal."""
        nodes = torch.tensor([[0., 0.], [1., 0.], [1., 1.], [0., 1.]], dtype=D)
        elems = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        diag = assemble_added_mass_lumped(nodes, elems, 100.0)
        assert diag.shape == (12,)
        for i in range(12):
            if i % 3 == 0:
                assert diag[i].item() > 0
            else:
                assert diag[i].item() == 0.0


# ---- Hetényi reference ----

class TestHetenyiReference:
    def test_shape_functions_at_zero(self):
        """A(0)=B(0)=C(0)=1, D(0)=0."""
        zero = torch.tensor(0.0, dtype=D)
        assert hetenyi_A(zero).item() == pytest.approx(1.0, abs=1e-14)
        assert hetenyi_B(zero).item() == pytest.approx(1.0, abs=1e-14)
        assert hetenyi_C(zero).item() == pytest.approx(1.0, abs=1e-14)
        assert hetenyi_D(zero).item() == pytest.approx(0.0, abs=1e-14)

    def test_shape_functions_decay(self):
        """All shape functions should decay for large xi."""
        xi = torch.linspace(0, 10, 100, dtype=D)
        for fn in (hetenyi_A, hetenyi_B, hetenyi_C, hetenyi_D):
            vals = fn(xi)
            assert bool(vals[-1].abs() < 1e-3)

    def test_point_load_deflection_symmetry(self):
        """w(x) should be symmetric about the load point."""
        x_left = torch.linspace(-2, -0.01, 20, dtype=D)
        x_right = torch.linspace(0.01, 2, 20, dtype=D)
        w_l = infinite_beam_point_load(x_left, 0.0, 1.0, 1.0, 1.0)
        w_r = infinite_beam_point_load(x_right, 0.0, 1.0, 1.0, 1.0)
        torch.testing.assert_close(w_l, w_r.flip(0), atol=1e-12, rtol=1e-12)

    def test_finite_beam_series_convergence(self):
        """Increasing n_terms should reduce change in deflection."""
        x_eval = torch.tensor([0.5], dtype=D)
        w50 = finite_beam_series(x_eval, 10.0, 1.0, 1.0, 1.0, [(5.0, 1.0)], n_terms=50)
        w100 = finite_beam_series(x_eval, 10.0, 1.0, 1.0, 1.0, [(5.0, 1.0)], n_terms=100)
        w200 = finite_beam_series(x_eval, 10.0, 1.0, 1.0, 1.0, [(5.0, 1.0)], n_terms=200)
        err50_100 = (w100 - w50).abs().item()
        err100_200 = (w200 - w100).abs().item()
        assert err100_200 < err50_100

    def test_plate_strip_parameters(self):
        EI, k_b, lam = plate_strip_parameters(9e9, 0.33, 0.15, 1e6, 1.0)
        assert EI > 0
        assert k_b > 0
        assert lam > 0
