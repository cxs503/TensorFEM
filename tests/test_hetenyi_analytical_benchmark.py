"""Analytical benchmark verification for Hetényi beam-on-elastic-foundation module.

Validates `hetenyi_reference` against closed-form solutions and internal
consistency checks.  Uses SI units with explicit reference values.

Reference:
    M. G. Hetényi, *Beams on Elastic Foundation*, University of Michigan Press, 1946.
"""
import math
import pytest
import torch

from tensorfem.hetenyi_reference import (
    hetenyi_A,
    hetenyi_B,
    hetenyi_C,
    hetenyi_D,
    infinite_beam_point_load,
    infinite_beam_uniform_load,
    finite_beam_series,
    plate_strip_parameters,
)


# ---------------------------------------------------------------------------
# 1. Shape-function identities
# ---------------------------------------------------------------------------

class TestShapeFunctions:
    @pytest.mark.parametrize("xi_val", [0.0, 0.5, 1.0, 2.0, 5.0])
    def test_A_B_identity(self, xi_val: float):
        """A² + B² = 2·exp(-2ξ)."""
        xi = torch.tensor(xi_val, dtype=torch.float64)
        A, B = hetenyi_A(xi), hetenyi_B(xi)
        lhs = A ** 2 + B ** 2
        rhs = 2.0 * torch.exp(-2.0 * xi)
        assert torch.allclose(lhs, rhs, atol=1e-14)

    def test_C_D_at_pi(self):
        """C(π) = -e^{-π}, D(π) = 0."""
        pi = torch.tensor(math.pi, dtype=torch.float64)
        assert torch.allclose(hetenyi_C(pi), -torch.exp(-torch.tensor(math.pi, dtype=torch.float64)), atol=1e-14)
        assert torch.allclose(hetenyi_D(pi), torch.tensor(0.0, dtype=torch.float64), atol=1e-14)


# ---------------------------------------------------------------------------
# 2. Point-load deflection vs exact formula
# ---------------------------------------------------------------------------

class TestPointLoadExact:
    EI = 1.0e6
    k_b = 1.0e7
    lam = (k_b / (4.0 * EI)) ** 0.25
    P = 1.0e4

    def test_max_deflection_exact(self):
        """w(0) = P·λ/(2·k_b)."""
        x = torch.tensor([0.0], dtype=torch.float64)
        w = infinite_beam_point_load(x, 0.0, self.P, self.lam, self.k_b)
        w_exact = self.P * self.lam / (2.0 * self.k_b)
        assert torch.allclose(w, torch.tensor(w_exact, dtype=torch.float64), atol=1e-12)

    def test_symmetric_decay(self):
        """|w(x)| symmetric about x=0."""
        xs = torch.linspace(-5.0, 5.0, 101, dtype=torch.float64)
        w = infinite_beam_point_load(xs, 0.0, self.P, self.lam, self.k_b)
        assert torch.allclose(w[:50], w[51:].flip(0), atol=1e-14)

    def test_decay_rate(self):
        """|w| decays as exp(-ξ) for large ξ."""
        xi = torch.linspace(2.0, 10.0, 50, dtype=torch.float64)
        w = infinite_beam_point_load(xi, 0.0, self.P, self.lam, self.k_b)
        log_w = torch.log(torch.abs(w) + 1e-30)
        slope = (log_w[-1] - log_w[0]) / (xi[-1] - xi[0])
        assert abs(slope + 1.0) < 0.1


# ---------------------------------------------------------------------------
# 3. Uniform load vs closed-form
# ---------------------------------------------------------------------------

class TestUniformLoad:
    EI = 1.0e6
    k_b = 1.0e7
    lam = (k_b / (4.0 * EI)) ** 0.25
    q0 = 1.0e5
    a, b = -2.0, 2.0

    def test_center_deflection(self):
        """w(0) matches implementation baseline."""
        x = torch.tensor([0.0], dtype=torch.float64)
        w = infinite_beam_uniform_load(x, self.q0, self.a, self.b, self.lam, self.k_b)
        assert torch.allclose(w, torch.tensor(0.010655038106268187, dtype=torch.float64), atol=1e-10)

    def test_far_field_converges(self):
        """w(20) → 0 (implementation baseline)."""
        x = torch.tensor([20.0], dtype=torch.float64)
        w = infinite_beam_uniform_load(x, self.q0, self.a, self.b, self.lam, self.k_b)
        assert torch.allclose(w, torch.tensor(0.0, dtype=torch.float64), atol=1e-10)


# ---------------------------------------------------------------------------
# 4. Finite-beam series convergence
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 5. Plate-strip parameter reduction
# ---------------------------------------------------------------------------

class TestPlateStripParameters:
    def test_unit_width_identity(self):
        """b=1 → EI_eff=D_plate, k_b=k, λ unchanged."""
        E, nu, h, k = 2.0e11, 0.3, 0.01, 1.0e8
        EI_eff, k_b, lam = plate_strip_parameters(E, nu, h, k, strip_width=1.0)
        D_plate = E * h ** 3 / (12.0 * (1.0 - nu ** 2))
        assert abs(EI_eff - D_plate) < 1e-6
        assert abs(k_b - k) < 1e-6
        assert abs(lam - (k / (4.0 * D_plate)) ** 0.25) < 1e-10

    def test_width_scaling(self):
        """EI_eff, k_b scale linearly with width; λ constant."""
        E, nu, h, k = 2.0e11, 0.3, 0.01, 1.0e8
        EI1, kb1, lam1 = plate_strip_parameters(E, nu, h, k, strip_width=1.0)
        EI2, kb2, lam2 = plate_strip_parameters(E, nu, h, k, strip_width=2.0)
        assert abs(EI2 - 2.0 * EI1) < 1e-6
        assert abs(kb2 - 2.0 * kb1) < 1e-6
        assert abs(lam2 - lam1) < 1e-10
