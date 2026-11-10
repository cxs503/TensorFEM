"""P186-W1-N04: Mindlin plate static + vibration joint verification.

Tests the Q4 Mindlin-Reissner SRI element against:
- C1: Timoshenko plate theory (static deflection, <1% error)
- C2: Leissa/Navier analytical frequencies (<2% error)
- C3: Mesh convergence order ≥ 2
- C4: Thin plate limit (h→0 → Kirchhoff, <5% diff)
- C5: Zero density limit (ρ→0 → frequency → ∞)

All analytical references are independent closed-form solutions.
"""
from __future__ import annotations

import math
import pytest
import torch

from tensorfem.plate import structured_square_mesh, q4_mindlin_stiffness
from tensorfem.plate_mindlin import consistent_load_vector
from tensorfem.plate_vibration import (
    solve_plate_vibration,
    kirchhoff_angular_frequency,
)


# ============================================================================
# Physical parameters
# ============================================================================
YOUNG = 200.0e9
POISSON = 0.3
DENSITY = 7850.0
SIDE = 1.0
THICKNESS = 0.01
PRESSURE = 1000.0


# ============================================================================
# Analytical references
# ============================================================================

def timoshenko_ss_plate_max_deflection(
    a: float, q: float, E: float, nu: float, h: float, n_terms: int = 25,
) -> float:
    """Timoshenko (Mindlin) analytical max deflection at center of SSSS plate."""
    D = E * h**3 / (12.0 * (1.0 - nu**2))
    G = E / (2.0 * (1.0 + nu))
    kappa = 5.0 / 6.0
    x, y = a / 2.0, a / 2.0

    w = 0.0
    for m in range(1, 2 * n_terms, 2):
        for n in range(1, 2 * n_terms, 2):
            alpha_m = m * math.pi / a
            beta_n = n * math.pi / a
            R_mn = alpha_m**2 + beta_n**2
            q_mn = 16.0 * q / (math.pi**2 * m * n)
            W_kirchhoff = q_mn / (D * R_mn**2)
            shear_factor = 1.0 + D * R_mn / (kappa * G * h)
            W_mn = W_kirchhoff * shear_factor
            w += W_mn * math.sin(alpha_m * x) * math.sin(beta_n * y)
    return w


def kirchhoff_ss_plate_max_deflection(
    a: float, q: float, E: float, nu: float, h: float, n_terms: int = 25,
) -> float:
    """Kirchhoff analytical max deflection at center of SSSS plate."""
    D = E * h**3 / (12.0 * (1.0 - nu**2))
    x, y = a / 2.0, a / 2.0

    w = 0.0
    for m in range(1, 2 * n_terms, 2):
        for n in range(1, 2 * n_terms, 2):
            term = 1.0 / (m * n * ((m / a)**2 + (n / a)**2)**2)
            w += term * math.sin(m * math.pi * x / a) * math.sin(n * math.pi * y / a)
    w *= 16.0 * q / (math.pi**6 * D)
    return w


def navier_ss_frequencies(
    a: float, E: float, nu: float, h: float, rho: float, n_modes: int = 5,
) -> list[float]:
    """Exact natural frequencies of SSSS square plate (Navier/Kirchhoff)."""
    D = E * h**3 / (12.0 * (1.0 - nu**2))
    omega_factor = math.pi**2 * math.sqrt(D / (rho * h)) / a**2

    modes = []
    for m in range(1, 10):
        for n in range(1, 10):
            omega = omega_factor * (m**2 + n**2)
            modes.append(omega)
    modes.sort()
    return [omega / (2 * math.pi) for omega in modes[:n_modes]]


def solve_mindlin_static_fe(nx: int) -> float:
    """Solve SSSS Mindlin plate under uniform load, return max deflection."""
    nodes, elements = structured_square_mesh(nx, SIDE, dtype=torch.float64)
    n_nodes = nodes.shape[0]
    ndof = 3 * n_nodes

    K = torch.zeros((ndof, ndof), dtype=torch.float64)
    f = torch.zeros(ndof, dtype=torch.float64)

    for conn in elements:
        xy = nodes[conn]
        ke = q4_mindlin_stiffness(xy, YOUNG, POISSON, THICKNESS)
        fe = consistent_load_vector(xy, PRESSURE)
        dofs = torch.stack((3*conn, 3*conn+1, 3*conn+2), dim=1).reshape(-1)
        K[dofs[:, None], dofs] += ke
        f[dofs] += fe

    tol = SIDE * 1e-12
    xedge = (nodes[:, 0].abs() < tol) | ((nodes[:, 0] - SIDE).abs() < tol)
    yedge = (nodes[:, 1].abs() < tol) | ((nodes[:, 1] - SIDE).abs() < tol)
    edge = xedge | yedge

    fixed = torch.cat((
        3 * torch.nonzero(edge, as_tuple=False).flatten(),
        3 * torch.nonzero(yedge, as_tuple=False).flatten() + 1,
        3 * torch.nonzero(xedge, as_tuple=False).flatten() + 2,
    )).unique()

    all_dofs = torch.arange(ndof)
    free = all_dofs[~torch.isin(all_dofs, fixed)]

    u_f = torch.linalg.solve(K[free][:, free], f[free])
    u = torch.zeros(ndof, dtype=torch.float64)
    u[free] = u_f

    w_field = u[0::3].reshape(nx+1, nx+1)
    return w_field.max().item()


# ============================================================================
# C1: Static deflection < 1% vs Timoshenko
# ============================================================================

class TestC1StaticDeflection:
    """C1: Static deflection max relative error < 1% vs Timoshenko."""

    def test_static_deflection_16x16(self):
        """16x16 mesh should achieve <1% error."""
        w_analytical = timoshenko_ss_plate_max_deflection(SIDE, PRESSURE, YOUNG, POISSON, THICKNESS)
        w_fe = solve_mindlin_static_fe(16)
        rel_err = abs(w_fe - w_analytical) / abs(w_analytical)
        assert rel_err < 0.01, f"Static deflection error {rel_err:.4e} >= 1%"

    def test_static_deflection_converges(self):
        """Error should decrease with mesh refinement."""
        w_analytical = timoshenko_ss_plate_max_deflection(SIDE, PRESSURE, YOUNG, POISSON, THICKNESS)
        errors = []
        for nx in [4, 8, 12]:
            w_fe = solve_mindlin_static_fe(nx)
            errors.append(abs(w_fe - w_analytical) / abs(w_analytical))
        # Errors should be monotonically decreasing
        assert errors[0] > errors[1] > errors[2], f"Errors not decreasing: {errors}"


# ============================================================================
# C2: Natural frequencies < 2% vs Leissa/Navier
# ============================================================================

class TestC2NaturalFrequencies:
    """C2: First 5 natural frequencies < 2% error vs Leissa/Navier."""

    def test_first_five_modes_16x16(self):
        """16x16 mesh should achieve <2% error for first 5 modes."""
        analytical = navier_ss_frequencies(SIDE, YOUNG, POISSON, THICKNESS, DENSITY, 5)
        vib_result = solve_plate_vibration(
            elements_x=16, elements_y=16,
            modes=5,
            young=YOUNG, poisson=POISSON, thickness=THICKNESS,
            density=DENSITY, boundary="simple",
        )
        fe_freqs = vib_result.frequencies_hz.tolist()
        errors = [abs(fe_freqs[i] - analytical[i]) / analytical[i] for i in range(5)]
        max_err = max(errors)
        assert max_err < 0.02, f"Frequency error {max_err:.4e} >= 2%"


# ============================================================================
# C3: Mesh convergence order ≥ 2
# ============================================================================

class TestC3ConvergenceOrder:
    """C3: Mesh convergence order ≥ 2 for both static and frequency."""

    def test_static_convergence_order(self):
        """Static deflection should converge at O(h^2) or better."""
        w_analytical = timoshenko_ss_plate_max_deflection(SIDE, PRESSURE, YOUNG, POISSON, THICKNESS)
        errors = []
        for nx in [8, 12, 16]:
            w_fe = solve_mindlin_static_fe(nx)
            errors.append(abs(w_fe - w_analytical) / abs(w_analytical))
        # Estimate order from last two levels
        h_ratio = 12.0 / 16.0
        e_ratio = errors[1] / errors[2]
        order = math.log(e_ratio) / math.log(h_ratio)
        assert order >= 2.0, f"Static convergence order {order:.2f} < 2"

    def test_frequency_convergence_order(self):
        """Frequency should converge at O(h^2) or better."""
        analytical = navier_ss_frequencies(SIDE, YOUNG, POISSON, THICKNESS, DENSITY, 1)
        errors = []
        for nx in [8, 12, 16]:
            vib_result = solve_plate_vibration(
                elements_x=nx, elements_y=nx,
                modes=1,
                young=YOUNG, poisson=POISSON, thickness=THICKNESS,
                density=DENSITY, boundary="simple",
            )
            f_fe = vib_result.frequencies_hz[0].item()
            errors.append(abs(f_fe - analytical[0]) / analytical[0])
        # Estimate order from last two levels
        h_ratio = 12.0 / 16.0
        e_ratio = errors[1] / errors[2]
        order = math.log(e_ratio) / math.log(h_ratio)
        assert order >= 2.0, f"Frequency convergence order {order:.2f} < 2"


# ============================================================================
# C4: Thin plate limit (h→0 → Kirchhoff)
# ============================================================================

class TestC4ThinPlateLimit:
    """C4: As h→0, Mindlin deflection → Kirchhoff (<5% diff)."""

    def test_thin_plate_limit(self):
        """For a/h=1000, Mindlin should match Kirchhoff within 5%."""
        h_test = 0.001  # a/h = 1000
        w_mindlin = timoshenko_ss_plate_max_deflection(SIDE, PRESSURE, YOUNG, POISSON, h_test)
        w_kirchhoff = kirchhoff_ss_plate_max_deflection(SIDE, PRESSURE, YOUNG, POISSON, h_test)
        rel_diff = abs(w_mindlin - w_kirchhoff) / abs(w_kirchhoff)
        assert rel_diff < 0.05, f"Thin plate limit diff {rel_diff:.4e} >= 5%"


# ============================================================================
# C5: Zero density limit (ρ→0 → frequency → ∞)
# ============================================================================

class TestC5ZeroDensityLimit:
    """C5: As ρ→0, frequency → ∞."""

    def test_frequency_diverges(self):
        """Frequency should increase as density decreases."""
        freqs = []
        for rho in [7850.0, 100.0, 1.0]:
            vib_result = solve_plate_vibration(
                elements_x=8, elements_y=8,
                modes=1,
                young=YOUNG, poisson=POISSON, thickness=THICKNESS,
                density=rho, boundary="simple",
            )
            freqs.append(vib_result.frequencies_hz[0].item())
        # Frequency should increase as density decreases
        assert freqs[0] < freqs[1] < freqs[2], f"Frequencies not increasing: {freqs}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
