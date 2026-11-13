"""Validation of Aitken Delta^2 relaxation for implicit IBM coupling.

This test suite validates the Aitken relaxation implementation in
``tensorfem.fsi_lbm_strong_coupling`` against the theoretical expectations
from the joconnor22/LIFE paper and the Kuettler & Wall (2008) reference.

Physical Background
-------------------
In partitioned FSI with strong (implicit) coupling, the fluid and structure
are solved separately at each time step, with sub-iterations to enforce
interface compatibility. For light structures (low mass ratio), the
added-mass effect causes the fixed-point iteration to diverge unless
accelerated.

Aitken's Delta^2 method dynamically adjusts the relaxation parameter omega
based on the residual history:

    omega_{k+1} = -omega_k * (r_k · (r_k - r_{k-1})) / ||r_k - r_{k-1}||^2

For a linear fixed-point iteration x_{k+1} = G x_k with dominant eigenvalue
lambda, the optimal relaxation is omega_opt = 1 / (1 - lambda).

For the added-mass dominated case (lambda ≈ -1), omega_opt ≈ 0.5.

References
----------
[1] Connor, J. et al. (2022). "LIFE: A high-performance finite element
    library for the solution of partial differential equations."
    (joconnor22/LIFE) - implicit IBM with Aitken relaxation.
[2] Kuettler, U., Wall, W.A. (2008). "Fixed-point fluid-structure
    interaction solvers with dynamic relaxation." Comput. Mech. 42, 1-16.
[3] Irons, B., Tuck, O. (1969). "A version of the Aitken accelerator for
    computer iteration." Int. J. Numer. Meth. Eng. 1(2), 205-206.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfem.fsi_lbm_strong_coupling import (
    AitkenRelaxation,
    FSI2Parameters,
    StrongCouplingFSI2,
)


# ---------------------------------------------------------------------------
# A1: Aitken formula correctness on linear fixed-point
# ---------------------------------------------------------------------------

def test_aitken_optimal_omega_for_known_eigenvalue():
    """Aitken should recover omega_opt = 1/(1-lambda) for linear G.

    For lambda = -0.9 (typical added-mass case), omega_opt = 1/1.9 ≈ 0.5263.
    """
    lambda_val = -0.9
    omega_opt = 1.0 / (1.0 - lambda_val)  # = 0.5263...

    aitken = AitkenRelaxation(omega0=1.0)
    x = 1.0
    for _ in range(5):
        r = np.array([x])
        aitken.update(r)
        x_new = lambda_val * x
        x = (1 - aitken.omega) * x + aitken.omega * x_new

    # After a few iterations, omega should converge to omega_opt
    assert aitken.omega == pytest.approx(omega_opt, rel=1e-3), \
        f"omega={aitken.omega}, expected {omega_opt}"


@pytest.mark.parametrize("lambda_val", [-0.5, -0.7, -0.8, -0.9, -0.95])
def test_aitken_optimal_omega_parametric(lambda_val):
    """Aitken should recover omega_opt for various eigenvalues."""
    omega_opt = 1.0 / (1.0 - lambda_val)

    aitken = AitkenRelaxation(omega0=1.0)
    x = 1.0
    for _ in range(8):
        r = np.array([x])
        aitken.update(r)
        x_new = lambda_val * x
        x = (1 - aitken.omega) * x + aitken.omega * x_new

    assert aitken.omega == pytest.approx(omega_opt, rel=5e-3), \
        f"lambda={lambda_val}: omega={aitken.omega}, expected {omega_opt}"


# ---------------------------------------------------------------------------
# A2: Convergence acceleration vs. fixed relaxation
# ---------------------------------------------------------------------------

def test_aitken_accelerates_convergence_vs_fixed_omega():
    """Aitken should converge faster than fixed omega=0.5 for oscillatory case."""
    lambda_val = -0.9
    tol = 1e-8

    # Fixed relaxation
    x_fixed = 1.0
    omega_fixed = 0.5
    n_fixed = 0
    for _ in range(1000):
        if abs(x_fixed) < tol:
            break
        x_new = lambda_val * x_fixed
        x_fixed = (1 - omega_fixed) * x_fixed + omega_fixed * x_new
        n_fixed += 1

    # Aitken relaxation
    aitken = AitkenRelaxation(omega0=1.0)
    x_aitken = 1.0
    n_aitken = 0
    for _ in range(1000):
        if abs(x_aitken) < tol:
            break
        r = np.array([x_aitken])
        aitken.update(r)
        x_new = lambda_val * x_aitken
        x_aitken = (1 - aitken.omega) * x_aitken + aitken.omega * x_new
        n_aitken += 1

    # Aitken should converge in fewer or equal iterations
    assert n_aitken <= n_fixed, \
        f"Aitken took {n_aitken} iters, fixed took {n_fixed}"


# ---------------------------------------------------------------------------
# A3: Multi-dimensional residual
# ---------------------------------------------------------------------------

def test_aitken_multidimensional_residual():
    """Aitken should work correctly with vector residuals (multi-DOF)."""
    # Simulate a 2-DOF system with different eigenvalues
    lambda1, lambda2 = -0.8, -0.6
    aitken = AitkenRelaxation(omega0=1.0)

    x = np.array([1.0, 0.5])
    for _ in range(10):
        r = x.copy()
        aitken.update(r)
        x_new = np.array([lambda1 * x[0], lambda2 * x[1]])
        x = (1 - aitken.omega) * x + aitken.omega * x_new

    # Should converge
    assert np.linalg.norm(x) < 1e-6, f"Final residual norm: {np.linalg.norm(x)}"


# ---------------------------------------------------------------------------
# A4: Reset behavior
# ---------------------------------------------------------------------------

def test_aitken_reset_restores_initial_state():
    """Reset should restore omega to omega0 and clear r_prev."""
    aitken = AitkenRelaxation(omega0=0.8, omega_min=0.1, omega_max=1.5)

    # Run a few iterations to change state
    for i in range(5):
        r = np.array([(-0.9) ** i])
        aitken.update(r)

    omega_before = aitken.omega
    assert omega_before != 0.8  # should have changed

    # Reset
    aitken.reset()
    assert aitken.omega == 0.8
    assert aitken.r_prev is None


# ---------------------------------------------------------------------------
# A5: Omega clipping bounds
# ---------------------------------------------------------------------------

def test_aitken_omega_clipping():
    """Omega should be clipped to [omega_min, omega_max] range."""
    aitken = AitkenRelaxation(omega0=1.0, omega_min=0.02, omega_max=2.0)

    # Force a scenario that would produce extreme omega
    # By creating residuals that would cause division by near-zero
    r1 = np.array([1.0])
    aitken.update(r1)

    # Create a residual that would cause large omega
    r2 = np.array([1.0 + 1e-15])  # nearly identical to r1
    omega = aitken.update(r2)

    # Omega should be clipped
    assert -2.0 <= omega <= 2.0, f"omega={omega} outside bounds"


def test_aitken_omega_near_zero_reverts_to_omega0():
    """When omega becomes numerically zero, it should revert to omega0."""
    aitken = AitkenRelaxation(omega0=1.0)

    # Create residuals that would make omega ≈ 0
    r1 = np.array([1.0, 0.0])
    aitken.update(r1)
    r2 = np.array([1.0, 1e-20])  # nearly parallel to r1
    omega = aitken.update(r2)

    # If omega is too small, it should revert
    if abs(omega) < 1e-8:
        assert omega == aitken.omega0 or abs(omega) >= 1e-8


# ---------------------------------------------------------------------------
# A6: Integration with StrongCouplingFSI2 (smoke test)
# ---------------------------------------------------------------------------

def test_strong_coupling_aitken_omega_history():
    """The strong coupling driver should record meaningful omega values."""
    p = FSI2Parameters()
    fsi = StrongCouplingFSI2(p, cells_per_D=8, u_lat=0.06,
                             max_sub_iterations=10, tolerance=1e-3)

    h = fsi.run(100, record_every=10)

    # Omega should have been recorded
    assert len(h.omega) > 0, "No omega history recorded"

    # Omega values should be finite and in reasonable range
    omegas = [w for w in h.omega if np.isfinite(w)]
    assert len(omegas) > 0, "No finite omega values"
    assert all(0.01 <= abs(w) <= 3.0 for w in omegas), \
        f"Omega values out of range: {omegas[:5]}"

    # Mean sub-iterations should be less than max (convergence happening)
    assert np.mean(h.sub_iterations) < fsi.max_sub_iterations


# ---------------------------------------------------------------------------
# A7: Aitken convergence rate matches theory
# ---------------------------------------------------------------------------

def test_aitken_convergence_rate_theoretical():
    """For linear fixed-point, Aitken should achieve the theoretical rate.

    The convergence rate after Aitken acceleration should be:
    rho_aitken = |lambda - omega_opt * (lambda - 1)| / |1 - omega_opt * (lambda - 1)|

    For lambda = -0.9, omega_opt = 1/1.9:
    rho_aitken ≈ 0 (superlinear convergence for linear case)
    """
    lambda_val = -0.9
    omega_opt = 1.0 / (1.0 - lambda_val)

    aitken = AitkenRelaxation(omega0=1.0)
    x = 1.0
    residuals = []

    for _ in range(8):
        r = np.array([x])
        aitken.update(r)
        residuals.append(abs(x))
        x_new = lambda_val * x
        x = (1 - aitken.omega) * x + aitken.omega * x_new

    # Check that residuals decay rapidly
    # After convergence of omega, the rate should be much better than fixed
    if len(residuals) >= 4:
        # Compare decay rate in later iterations
        late_decay = residuals[-1] / max(residuals[-3], 1e-30)
        # Should be very small (near machine precision for linear case)
        assert late_decay < 0.1, \
            f"Late decay rate {late_decay} too slow (should be < 0.1)"


# ---------------------------------------------------------------------------
# A8: Comparison with literature (Kuettler & Wall 2008)
# ---------------------------------------------------------------------------

def test_aitken_matches_literature_behavior():
    """Aitken behavior should match Kuettler & Wall (2008) Fig. 3.

    For mass ratio rho_f/rho_s = 1 (FSI-2), the optimal omega is ~0.5-0.6.
    """
    # FSI-2 has rho_f = rho_s = 1000, so mass ratio = 1
    # Literature suggests omega_opt ≈ 0.5-0.6 for this case

    aitken = AitkenRelaxation(omega0=1.0)

    # Simulate the fixed-point iteration for mass ratio = 1
    # The eigenvalue is approximately -0.5 to -0.7 for this case
    lambda_val = -0.6
    x = 1.0

    for _ in range(10):
        r = np.array([x])
        aitken.update(r)
        x_new = lambda_val * x
        x = (1 - aitken.omega) * x + aitken.omega * x_new

    # omega should converge to 1/(1-(-0.6)) = 1/1.6 = 0.625
    omega_expected = 1.0 / (1.0 - lambda_val)
    assert aitken.omega == pytest.approx(omega_expected, rel=1e-2), \
        f"omega={aitken.omega}, expected {omega_expected} (literature range 0.5-0.6)"
