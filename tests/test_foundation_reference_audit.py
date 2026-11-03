"""Independent verification of the Winkler-foundation reference solutions.

This module audits the analytical references in ``tensorfem.hetenyi_reference``
and ``tensorfem.plate_mindlin.hetenyi_reference`` against oracles that are
**independent of the closed forms under test**:

* a Green's-function quadrature (superposition of the point-load solution) for
  the infinite beam under a distributed load,
* a five-point finite-difference solve of ``EI w'''' + k_b w = q`` for the simply
  supported finite beam,
* a Hankel quadrature ``w(r) = P/(2 pi) ∫ xi J0(xi r)/(D xi^4 + k) d xi`` for the
  infinite plate on a Winkler foundation (the Bessel function is a self-contained
  A&S 9.4 polynomial approximation validated against tabulated values below),
* force equilibrium ``∫ k w dA = P`` and the exact centre deflection
  ``w(0) = P/(8 sqrt(D k))`` for the plate,
* tabulated Kelvin-function values and branch continuity.

The audit that motivated these tests is written up in
``docs/hetanyi-reference-audit.md``.
"""
import math

import numpy as np
import pytest
import torch

from tensorfem.hetenyi_reference import (
    finite_beam_series,
    infinite_beam_point_load,
    infinite_beam_uniform_load,
    infinite_plate_point_load,
    kelvin_kei,
    plate_strip_parameters,
)

D = torch.float64


# ---------------------------------------------------------------------------
# Oracles (independent of the production code)
# ---------------------------------------------------------------------------

def bessel_j0(x):
    """J0(x) from the Abramowitz & Stegun 9.4 polynomial approximations.

    Absolute error < 5e-8 (|x| <= 3) and < 1e-8 (x > 3); validated in
    ``test_j0_oracle_matches_tabulated_values``.
    """
    x = np.asarray(x, dtype=np.float64)
    ax = np.abs(x)
    small = ax <= 3.0
    out = np.empty_like(ax)

    t = (ax[small] / 3.0) ** 2
    p = (
        1.0
        - 2.2499997 * t
        + 1.2656208 * t**2
        - 0.3163866 * t**3
        + 0.0444479 * t**4
        - 0.0039444 * t**5
        + 0.00021 * t**6
    )
    out[small] = p

    big = ~small
    xb = ax[big]
    z = 3.0 / xb
    amp = 0.79788456 - 0.00000077 * z - 0.00552740 * z**2 - 0.00009512 * z**3 \
        + 0.00137237 * z**4 - 0.00072805 * z**5 + 0.00014476 * z**6
    phase = (
        xb
        - 0.78539816
        - 0.04166397 * z
        - 0.00003954 * z**2
        + 0.00262573 * z**3
        - 0.00054125 * z**4
        - 0.00029333 * z**5
        + 0.00013558 * z**6
    )
    out[big] = amp * np.cos(phase) / np.sqrt(xb)
    return out


def gauss_legendre(a, b, nodes):
    u, w = np.polynomial.legendre.leggauss(nodes)
    return 0.5 * (b - a) * u + 0.5 * (a + b), 0.5 * (b - a) * w


def beam_uniform_greens_oracle(x_eval, q0, x_start, x_end, lam, k_b, nodes=4001):
    """∫_a^b P*lam/(2k_b)*A(lam|x-s|) ds by midpoint quadrature of the point load."""
    s = np.linspace(x_start, x_end, nodes + 1)
    s = 0.5 * (s[1:] + s[:-1])
    ds = (x_end - x_start) / nodes
    d = np.abs(np.asarray(x_eval)[:, None] - s[None, :])
    kernel = np.exp(-lam * d) * (np.cos(lam * d) + np.sin(lam * d))
    return q0 * lam / (2.0 * k_b) * kernel.sum(axis=1) * ds


def plate_hankel_oracle(r_eval, P, D_plate, k, nodes=1600):
    """Axisymmetric Hankel quadrature of the infinite plate on a Winkler foundation.

        w(r) = (P/(2 pi k l^2)) * ∫_0^∞ u J0(u r/l)/(u^4 + 1) du
    """
    l = (D_plate / k) ** 0.25
    out = np.empty(len(r_eval), dtype=np.float64)
    for idx, r in enumerate(r_eval):
        rho = abs(r) / l
        upper = 60.0 + 20.0 * rho
        u, w = gauss_legendre(0.0, upper, nodes)
        integrand = u * bessel_j0(u * rho) / (u**4 + 1.0)
        out[idx] = (P / (2.0 * math.pi * k * l * l)) * np.sum(integrand * w)
    return out


def beam_simply_supported_fd(q_of_x, L, EI, k_b, n=800):
    """Dense finite-difference solve of EI w'''' + k_b w = q with w = w'' = 0 at ends."""
    h = L / (n + 1)
    x = h * np.arange(1, n + 1)
    diag = 6.0 * np.ones(n)
    off1 = -4.0 * np.ones(n - 1)
    off2 = np.ones(n - 2)
    A = np.diag(diag) + np.diag(off1, 1) + np.diag(off1, -1) + np.diag(off2, 2) + np.diag(off2, -2)
    # w_0 = w_{n+1} = 0 and w'' = 0 at both ends.  w'' = 0 gives the anti-symmetric
    # ghost value w_{-1} = -w_1 (and w_{n+2} = -w_n), i.e. a -1 contribution to the
    # first and last diagonal entries of the 5-point stencil.
    A[0, 0] -= 1.0
    A[n - 1, n - 1] -= 1.0
    A *= EI / h**4
    A += k_b * np.eye(n)
    rhs = np.asarray(q_of_x(x), dtype=np.float64)
    return x, np.linalg.solve(A, rhs)


# ---------------------------------------------------------------------------
# Kelvin function / plate solution
# ---------------------------------------------------------------------------

class TestKelvinKei:
    def test_j0_oracle_matches_tabulated_values(self):
        """The Bessel oracle used below must itself be trustworthy."""
        assert bessel_j0(0.0) == pytest.approx(1.0, abs=5e-8)
        assert bessel_j0(1.0) == pytest.approx(0.7651976866, abs=1e-7)
        assert bessel_j0(3.0) == pytest.approx(-0.2600519549, abs=1e-7)
        assert bessel_j0(5.0) == pytest.approx(-0.1775967713, abs=1e-7)
        assert bessel_j0(10.0) == pytest.approx(-0.2459357645, abs=1e-7)
        assert bessel_j0(2.4048255577) == pytest.approx(0.0, abs=1e-7)

    @pytest.mark.parametrize(
        "x,expected",
        [
            (0.0, -0.7853981633974483),
            (1.0, -0.49499463651872),
            (2.0, -0.20240006776470),
            (5.0, 0.01118758650987),
            (10.0, -3.0752456909e-4),
        ],
    )
    def test_tabulated_values(self, x, expected):
        """kei against DLMF/standard tabulated values (rel tol 1e-8)."""
        got = float(kelvin_kei(torch.tensor([x], dtype=D)))
        assert got == pytest.approx(expected, rel=1e-8, abs=1e-12)

    def test_branch_continuity(self):
        """Series and asymptotic branches must agree at the switch point."""
        lo = float(kelvin_kei(torch.tensor([13.999999], dtype=D)))
        hi = float(kelvin_kei(torch.tensor([14.000001], dtype=D)))
        assert abs(hi - lo) / abs(lo) < 1e-6

    def test_asymptotic_decay(self):
        """kei must decay like exp(-x/sqrt(2)) at large argument."""
        x = torch.tensor([60.0], dtype=D)
        val = float(kelvin_kei(x))
        amp = math.sqrt(math.pi / (2 * 60.0)) * math.exp(-60.0 / math.sqrt(2.0))
        assert val == pytest.approx(-amp * math.sin(60.0 / math.sqrt(2.0) + math.pi / 8.0), rel=1e-2)


class TestInfinitePlate:
    """Exact infinite Kirchhoff plate on a Winkler foundation."""

    E, nu, t, k, P = 2.1e11, 0.3, 0.01, 1.0e6, 1.0

    @property
    def plate_D(self):
        return self.E * self.t**3 / (12.0 * (1.0 - self.nu**2))

    def test_centre_deflection_closed_form(self):
        """w(0) = P/(8 sqrt(D k)) to machine precision."""
        w0 = float(infinite_plate_point_load(torch.tensor([0.0], dtype=D), self.P, self.plate_D, self.k))
        assert w0 == pytest.approx(self.P / (8.0 * math.sqrt(self.plate_D * self.k)), rel=1e-12)

    def test_matches_hankel_quadrature(self):
        """Pointwise agreement with an independent Hankel quadrature (<1e-4)."""
        l = (self.plate_D / self.k) ** 0.25
        r = np.array([0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0]) * l
        closed = infinite_plate_point_load(torch.tensor(r, dtype=D), self.P, self.plate_D, self.k).numpy()
        oracle = plate_hankel_oracle(r, self.P, self.plate_D, self.k)
        rel = np.abs(closed - oracle) / np.abs(oracle)
        assert rel.max() < 1e-4, f"max relative error {rel.max():.2e}"

    def test_force_equilibrium(self):
        """Total foundation reaction must recover the applied load: ∫ k w 2 pi r dr = P."""
        l = (self.plate_D / self.k) ** 0.25
        r = np.linspace(0.0, 25.0 * l, 200001)
        w = infinite_plate_point_load(torch.tensor(r, dtype=D), self.P, self.plate_D, self.k).numpy()
        reaction = np.trapezoid(self.k * w * 2.0 * math.pi * r, r)
        assert reaction == pytest.approx(self.P, rel=2e-3)

    def test_uplift_ring_exists(self):
        """The exact solution changes sign near r/l ≈ 2 (the old exponential fit did not)."""
        l = (self.plate_D / self.k) ** 0.25
        r = torch.linspace(0.5 * l, 4.0 * l, 701, dtype=D)
        w = infinite_plate_point_load(r, self.P, self.plate_D, self.k)
        assert bool((w < 0).any()), "expected a shallow uplift ring outside the loaded area"
        assert float(w[0]) > 0.0

    def test_governing_equation_residual(self):
        """D ∇⁴w + k w = 0 for r > 0 (finite differences on the closed form)."""
        l = (self.plate_D / self.k) ** 0.25
        h = 1e-3 * l
        r0 = torch.tensor([0.4, 0.8, 1.5, 3.0], dtype=D) * l

        def w(rr):
            return infinite_plate_point_load(rr, self.P, self.plate_D, self.k)

        wm2, wm1, w0_, wp1, wp2 = (w(r0 + k_ * h) for k_ in (-2, -1, 0, 1, 2))
        w1 = (wp1 - wm1) / (2 * h)
        w2 = (wp1 - 2 * w0_ + wm1) / h**2
        w3 = (wp2 - 2 * wp1 + 2 * wm1 - wm2) / (2 * h**3)
        w4 = (wp2 - 4 * wp1 + 6 * w0_ - 4 * wm1 + wm2) / h**4
        laplacian_u = w4 + 2.0 * w3 / r0 - w2 / r0**2 + w1 / r0**3
        residual = self.plate_D * laplacian_u + self.k * w0_
        scale = self.k * w0_.abs()
        # The floor of this check is the fourth-derivative round-off,
        # ~eps*|w|/h^4 / (k|w|) ≈ 1e-3; the discarded exponential fit fails it by
        # O(1), so the gate still separates exact from fabricated references.
        assert bool((residual.abs() / scale < 2e-2).all()), f"residual/scale={residual.abs().div(scale).tolist()}"

    def test_old_exponential_fit_was_wrong(self):
        """Regression guard: the discarded exp(-beta r) fit differs by >50% at r/l ≈ 1.2."""
        l = (self.plate_D / self.k) ** 0.25
        w0 = self.P / (8.0 * math.sqrt(self.plate_D * self.k))
        beta = (self.k / (4.0 * self.plate_D)) ** 0.25
        r = torch.linspace(0.5 * l, 6.0 * l, 221, dtype=D)
        old = w0 * torch.exp(-beta * r)
        new = infinite_plate_point_load(r, self.P, self.plate_D, self.k)
        rel = (old - new).abs() / new.abs()
        assert float(rel.max()) > 0.5, f"max deviation {float(rel.max()):.2f}"""

    def test_plate_mindlin_delegates_to_exact_solution(self):
        """plate_mindlin.hetenyi_reference must return the exact Kelvin solution."""
        from tensorfem.plate_mindlin import hetenyi_reference

        exact = float(
            infinite_plate_point_load(torch.tensor([0.3], dtype=D), self.P, self.plate_D, self.k)
        )
        got = hetenyi_reference(0.3, 0.0, self.P, self.E, self.nu, self.t, self.k)
        assert got == pytest.approx(exact, rel=1e-12)


# ---------------------------------------------------------------------------
# Beam reference solutions
# ---------------------------------------------------------------------------

class TestInfiniteBeamUniformLoad:
    lam, k_b, q0, a, b = 0.35, 2.0e6, 1.0e4, 0.5, 1.5

    def _w(self, x):
        x = torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=D)
        return infinite_beam_uniform_load(x, self.q0, self.a, self.b, self.lam, self.k_b)

    def test_matches_greens_function_quadrature(self):
        """Closed form vs midpoint quadrature of the point-load Green's function."""
        x = np.linspace(-8.0, 10.0, 9)
        closed = self._w(x).numpy()
        oracle = beam_uniform_greens_oracle(x, self.q0, self.a, self.b, self.lam, self.k_b)
        rel = np.abs(closed - oracle) / np.abs(oracle)
        assert rel.max() < 1e-5, f"max relative error {rel.max():.2e}"

    def test_governing_ode_residual(self):
        """EI w'''' + k_b w must equal q0 inside the loaded interval and 0 outside."""
        EI = self.k_b / (4.0 * self.lam**4)
        h = 1e-3
        x = torch.as_tensor(np.linspace(-3.0, 5.0, 9), dtype=D)
        w4 = (
            self._w(x - 2 * h) - 4 * self._w(x - h) + 6 * self._w(x)
            - 4 * self._w(x + h) + self._w(x + 2 * h)
        ) / h**4
        q = torch.where(
            (x >= self.a) & (x <= self.b),
            torch.tensor(self.q0, dtype=D),
            torch.tensor(0.0, dtype=D),
        )
        residual = (EI * w4 + self.k_b * self._w(x) - q) / self.q0
        assert bool((residual.abs() < 2e-2).all()), f"residual/q0={residual.tolist()}"

    def test_far_field_decays_to_zero(self):
        """Net finite load ⇒ w → 0 in both far fields (the pre-audit formula returned q0/k_b)."""
        far = self._w([-40.0, 40.0]).abs().max().item()
        near = self._w([1.0]).abs().item()
        assert far < 1e-5 * near  # exp(-lambda*38) = 1.6e-6
        # and the discarded expression is what it was: ~q0/k_b far away
        assert self.q0 / self.k_b == pytest.approx(5e-3)

    def test_superposition_consistency(self):
        """Two half-interval uniform loads must add up to the full-interval result."""
        x = torch.as_tensor(np.linspace(-2.0, 3.0, 7), dtype=D)
        full = self._w(x)
        left = infinite_beam_uniform_load(x, self.q0, self.a, self.b - 0.5, self.lam, self.k_b)
        right = infinite_beam_uniform_load(x, self.q0, self.b - 0.5, self.b, self.lam, self.k_b)
        assert torch.allclose(full, left + right, rtol=1e-12, atol=1e-18)


class TestFiniteBeamSeries:
    def test_rejects_free_boundary(self):
        """Free ends are not implemented and must fail closed instead of mislabelling."""
        x = torch.tensor([0.5], dtype=D)
        with pytest.raises(ValueError, match="simply supported"):
            finite_beam_series(x, 10.0, 1.0, 1.0, 1.0, [(5.0, 1.0)], boundary='free')

    def test_simply_supported_matches_finite_difference(self):
        """Eigenfunction series and FD solve vs the exact single-mode response.

        For ``q(x) = q0 sin(pi x/L)`` the simply supported beam on a Winkler
        foundation has the closed-form ``w = q0/K sin(pi x/L)`` with
        ``K = EI (pi/L)^4 + k_b``.  The FD operator and the truncated series must
        both reproduce it; that validates the two oracles against each other and
        against an exact solution.
        """
        L, EI, k_b, q0 = 10.0, 1.0e5, 1.0e5, 2.0e3
        lam = (k_b / (4.0 * EI)) ** 0.25
        K = EI * (math.pi / L) ** 4 + k_b

        x_fd, w_fd = beam_simply_supported_fd(
            lambda xx: q0 * np.sin(math.pi * xx / L), L, EI, k_b, n=800
        )
        w_exact = q0 / K * np.sin(math.pi * x_fd / L)
        assert np.abs(w_fd - w_exact).max() / np.abs(w_exact).max() < 1e-3

        # the series reproduces the same mode: sample the sine load as a dense
        # set of point loads and compare against the exact modal response
        s = (np.arange(400) + 0.5) * L / 400.0
        loads = [(float(si), q0 * math.sin(math.pi * float(si) / L) * (L / 400.0)) for si in s]
        x_eval = torch.as_tensor(x_fd, dtype=D)
        w_series = finite_beam_series(x_eval, L, lam, k_b, EI, loads, n_terms=600).numpy()
        assert np.abs(w_series - w_exact).max() / np.abs(w_exact).max() < 5e-3

    def test_plate_strip_parameters_consistency(self):
        E, nu, h, k = 9e9, 0.33, 0.15, 1e6
        EI, k_b, lam = plate_strip_parameters(E, nu, h, k, 1.0)
        D_plate = E * h**3 / (12.0 * (1.0 - nu**2))
        assert EI == pytest.approx(D_plate, rel=1e-14)
        assert k_b == pytest.approx(k, rel=1e-14)
        assert lam == pytest.approx((k / (4.0 * D_plate)) ** 0.25, rel=1e-14)
        # the characteristic length must reproduce the infinite-beam decay of the point load
        w = infinite_beam_point_load(torch.tensor([1.0 / lam], dtype=D), 0.0, 1.0, lam, k_b)
        w0 = infinite_beam_point_load(torch.tensor([0.0], dtype=D), 0.0, 1.0, lam, k_b)
        assert float(w[0] / w0[0]) == pytest.approx(math.exp(-1.0) * (math.cos(1.0) + math.sin(1.0)), rel=1e-12)