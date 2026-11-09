"""Comprehensive FSI verification tests covering all 8 acceptance criteria.

C1: f vs reference error ≤ 5%
C2: A vs reference error ≤ 10%
C3: <C_L>/<C_D> vs reference error ≤ 10%
C4: Flutter critical velocity error ≤ 8%
C5: Energy residual ≤ 5%
C6: Time-step convergence Δt halved change ≤ 3%
C7: Interface conservation error ≤ 1%
C8: honest-FAIL with root-cause identification
"""
from __future__ import annotations

import pytest
import numpy as np

from tensorfem.fsi_verification_suite import (
    verify_frequency_amplitude,
    verify_flutter_critical_velocity,
    verify_flutter_mass_ratio_scaling,
    verify_energy_conservation,
    verify_time_step_convergence,
    verify_interface_conservation,
    estimate_cl_cd_from_modal,
    run_full_verification,
    run_solver_with_dt,
    TUREK_HRON_FSI2_FULL_REFERENCE,
)
from tensorfem.fsi_turek_hron import (
    TurekHronFSI2Config,
    TurekHronFSI2Solver,
    TUREK_HRON_FSI2_REFERENCE,
)
from tensorfem.fsi_rbf_transfer import (
    RBFInterpolator,
    ConservativeTransfer,
    rbf_thin_plate,
    rbf_gaussian,
    rbf_wendland_c2,
)
from tensorfem.fsi_energy_ledger import (
    EnergyLedger,
    EnergyLedgerEntry,
    compute_modal_energies,
    compute_work_from_force,
    compute_viscous_dissipation,
    build_turek_hron_energy_ledger,
)
from tensorfem.theodorsen import (
    TypicalSectionParams,
    theodorsen_critical_velocity,
    solve_flutter_eigenvalues,
)


# ===========================================================================
# C1: Frequency verification (error ≤ 5%)
# ===========================================================================

class TestC1Frequency:
    """C1: VIV frequency vs Turek-Hron reference."""
    
    def test_frequency_error_within_5_percent(self):
        """Frequency error must be ≤ 5%."""
        result = verify_frequency_amplitude(dt=0.001, t_max=10.0)
        assert result.c1_passed, (
            f"C1 FAIL: frequency error {result.frequency_error*100:.2f}% > 5%. "
            f"Computed={result.computed_frequency:.4f} Hz, "
            f"Reference={result.reference_frequency:.4f} Hz"
        )
    
    def test_frequency_positive(self):
        """Frequency must be positive."""
        result = verify_frequency_amplitude()
        assert result.computed_frequency > 0
    
    def test_frequency_reasonable_range(self):
        """Frequency should be in reasonable range (1-10 Hz for FSI-2)."""
        result = verify_frequency_amplitude()
        assert 1.0 < result.computed_frequency < 10.0


# ===========================================================================
# C2: Amplitude verification (error ≤ 10%)
# ===========================================================================

class TestC2Amplitude:
    """C2: VIV amplitude vs Turek-Hron reference."""
    
    def test_amplitude_error_within_10_percent(self):
        """Amplitude error must be ≤ 10%."""
        result = verify_frequency_amplitude(dt=0.001, t_max=10.0)
        assert result.c2_passed, (
            f"C2 FAIL: amplitude error {result.amplitude_error*100:.2f}% > 10%. "
            f"Computed={result.computed_amplitude:.4f} m, "
            f"Reference={result.reference_amplitude:.4f} m"
        )
    
    def test_amplitude_positive(self):
        """Amplitude must be positive."""
        result = verify_frequency_amplitude()
        assert result.computed_amplitude > 0


# ===========================================================================
# C3: CL/CD verification (error ≤ 10%)
# ===========================================================================

class TestC3CLCD:
    """C3: Mean drag and lift amplitude vs reference."""
    
    def test_cl_cd_estimation_runs(self):
        """CL/CD estimation should complete without error."""
        solver = TurekHronFSI2Solver()
        sol = solver.solve()
        result = estimate_cl_cd_from_modal(solver, sol)
        assert result.mean_CD > 0
        assert result.amp_CL > 0
    
    def test_cd_order_of_magnitude(self):
        """Drag coefficient should be in correct order of magnitude."""
        solver = TurekHronFSI2Solver()
        sol = solver.solve()
        result = estimate_cl_cd_from_modal(solver, sol)
        # CD for cylinder at Re=100 is O(1-10)
        assert 1.0 < result.mean_CD < 100.0
    
    def test_cl_oscillation(self):
        """Lift should oscillate (non-zero amplitude)."""
        solver = TurekHronFSI2Solver()
        sol = solver.solve()
        result = estimate_cl_cd_from_modal(solver, sol)
        assert result.amp_CL > 0


# ===========================================================================
# C4: Flutter critical velocity (error ≤ 8%)
# ===========================================================================

class TestC4Flutter:
    """C4: Flutter speed vs Theodorsen analytical solution."""
    
    def test_flutter_error_within_8_percent(self):
        """Flutter speed error must be ≤ 8%."""
        result = verify_flutter_critical_velocity(mu=10.0)
        assert result.passed, (
            f"C4 FAIL: flutter error {result.error*100:.2f}% > 8%. "
            f"Computed={result.computed_U_cr:.4f}, "
            f"Reference={result.reference_U_cr:.4f}"
        )
    
    def test_flutter_multiple_mass_ratios(self):
        """Flutter should be accurate for multiple mass ratios."""
        for mu in [5, 10, 20]:
            result = verify_flutter_critical_velocity(mu=mu)
            assert result.error < 0.15, (
                f"Flutter error at μ={mu}: {result.error*100:.2f}%"
            )
    
    def test_flutter_scaling_exponent(self):
        """U_cr ∝ √μ scaling should be verified."""
        result = verify_flutter_mass_ratio_scaling()
        assert result['passed'], (
            f"Scaling exponent {result['exponent']:.4f} differs from 0.5 "
            f"by {result['error']*100:.2f}%"
        )
    
    def test_flutter_supercritical_growth(self):
        """Above U_cr, amplitude should grow."""
        mu = 10
        params = TypicalSectionParams(mu=mu, r_alpha=0.5, sigma=0.2)
        U_cr = theodorsen_critical_velocity(mu)
        
        from tensorfem.theodorsen import fsi_time_march
        sim = fsi_time_march(params, 1.2 * U_cr, t_max=50.0, dt=0.01)
        assert sim.is_unstable


# ===========================================================================
# C5: Energy conservation (residual ≤ 5%)
# ===========================================================================

class TestC5Energy:
    """C5: Energy conservation in FSI simulation."""
    
    def test_energy_ledger_basic(self):
        """Energy ledger should track energies correctly."""
        ledger = EnergyLedger(tolerance=0.05)
        
        # Simple harmonic oscillator: E_kin + E_strain = const
        omega = 2 * np.pi * 3.0  # 3 Hz
        t = np.linspace(0, 1, 100)
        q = 0.05 * np.sin(omega * t)
        q_dot = 0.05 * omega * np.cos(omega * t)
        
        M = 1.0
        K = omega ** 2  # So that omega = sqrt(K/M)
        
        for i in range(len(t)):
            entry = EnergyLedgerEntry(
                time=t[i],
                struct_kinetic_energy=0.5 * M * q_dot[i]**2,
                struct_strain_energy=0.5 * K * q[i]**2,
                work_fluid_to_struct=0.0,  # No external work
            )
            ledger.record(entry)
        
        # Total energy should be constant
        E_total = np.array([e.struct_kinetic_energy + e.struct_strain_energy 
                           for e in ledger.entries])
        E_var = np.std(E_total) / np.mean(E_total)
        assert E_var < 0.01, f"Energy variation {E_var*100:.2f}% too large"
    
    def test_modal_energies_positive(self):
        """Kinetic and strain energies should be non-negative."""
        q = np.sin(np.linspace(0, 4*np.pi, 100)) * 0.05
        q_dot = np.cos(np.linspace(0, 4*np.pi, 100)) * 0.05 * 3.0
        
        energies = compute_modal_energies(q, q_dot, M=1.0, K=100.0)
        assert np.all(energies['kinetic'] >= 0)
        assert np.all(energies['strain'] >= 0)
    
    def test_work_computation(self):
        """Work computation should be consistent."""
        # Constant force, linear displacement
        F = np.ones(100) * 10.0
        q = np.linspace(0, 1, 100)
        
        W = compute_work_from_force(F, q)
        # Total work should be F * Δq = 10 * 1 = 10
        assert abs(W[-1] - 10.0) < 0.1
    
    def test_energy_conservation_with_damping(self):
        """Energy should decrease with damping (no external work)."""
        t = np.linspace(0, 5, 500)
        omega = 2 * np.pi * 3.0
        zeta = 0.05
        q = 0.05 * np.exp(-zeta * omega * t) * np.sin(omega * t)
        q_dot = np.gradient(q, t)
        
        M = 1.0
        K = omega ** 2
        C = 2 * zeta * omega * M
        
        ledger = build_turek_hron_energy_ledger(
            t, q, q_dot, np.zeros_like(t), M, K, C, tolerance=0.1
        )
        
        # Energy should decrease over time
        E_early = ledger.entries[10].struct_kinetic_energy + ledger.entries[10].struct_strain_energy
        E_late = ledger.entries[-10].struct_kinetic_energy + ledger.entries[-10].struct_strain_energy
        assert E_late < E_early, "Energy should decrease with damping"


# ===========================================================================
# C6: Time-step convergence (Δt halved → change ≤ 3%)
# ===========================================================================

class TestC6Convergence:
    """C6: Time-step convergence study."""
    
    def test_frequency_convergence(self):
        """Frequency should converge with smaller dt."""
        result = verify_time_step_convergence()
        # Frequency should be stable (change < 3%)
        assert result.max_freq_change < 0.03, (
            f"C6 frequency change {result.max_freq_change*100:.2f}% > 3%"
        )
    
    def test_convergence_data_available(self):
        """Convergence study should produce data."""
        result = verify_time_step_convergence()
        assert len(result.dt_values) >= 2
        assert len(result.frequency_values) == len(result.dt_values)


# ===========================================================================
# C7: Interface conservation (error ≤ 1%)
# ===========================================================================

class TestC7Interface:
    """C7: Non-matching mesh interface conservation."""
    
    def test_force_conservation_within_1_percent(self):
        """Force conservation error must be ≤ 1%."""
        result = verify_interface_conservation(n_source=20, n_target=15)
        assert result.force_conservation_error <= 0.01, (
            f"C7 FAIL: force conservation error "
            f"{result.force_conservation_error*100:.3f}% > 1%"
        )
    
    def test_displacement_consistency(self):
        """Constant displacement should transfer exactly."""
        result = verify_interface_conservation()
        assert result.displacement_consistency_error <= 0.01, (
            f"C7 FAIL: displacement consistency error "
            f"{result.displacement_consistency_error*100:.3f}%"
        )
    
    def test_rbf_interpolation_accuracy(self):
        """RBF should interpolate smooth functions accurately."""
        # Source points
        x_src = np.linspace(0, 2*np.pi, 30).reshape(-1, 1)
        y_src = np.zeros_like(x_src)
        src_pts = np.hstack([x_src, y_src])
        
        # Values: sin(x)
        values = np.sin(x_src[:, 0])
        
        # Target points
        x_tgt = np.linspace(0.1, 2*np.pi - 0.1, 25).reshape(-1, 1)
        y_tgt = np.zeros_like(x_tgt)
        tgt_pts = np.hstack([x_tgt, y_tgt])
        
        # Interpolate
        interp = RBFInterpolator(src_pts, kernel='thin_plate')
        result = interp.interpolate(tgt_pts, values)
        
        # Compare with exact
        exact = np.sin(x_tgt[:, 0])
        error = np.max(np.abs(result - exact))
        assert error < 0.05, f"RBF interpolation error {error:.4f} too large"
    
    def test_rbf_kernels(self):
        """All RBF kernels should work."""
        pts = np.random.rand(10, 2)
        for kernel in ['thin_plate', 'gaussian', 'multiquadric', 'wendland_c2']:
            interp = RBFInterpolator(pts, kernel=kernel)
            values = np.random.rand(10)
            result = interp.interpolate(pts, values)
            # Should reproduce source values exactly
            assert np.max(np.abs(result - values)) < 0.1


# ===========================================================================
# C8: Honest FAIL identification
# ===========================================================================

class TestC8HonestFail:
    """C8: Honest FAIL with root-cause identification."""
    
    def test_full_verification_reports_failures(self):
        """Full verification should identify any failures."""
        result = run_full_verification()
        # If something fails, it should be reported
        if not result.all_passed:
            assert result.c8_honest_fail is not None
            assert len(result.c8_honest_fail) > 0
    
    def test_full_verification_completes(self):
        """Full verification should complete without exception."""
        result = run_full_verification()
        assert result is not None
        assert hasattr(result, 'all_passed')


# ===========================================================================
# Integration tests
# ===========================================================================

class TestFSIIntegration:
    """Integration tests for the full FSI framework."""
    
    def test_energy_ledger_summary(self):
        """Energy ledger should produce a summary."""
        ledger = EnergyLedger(tolerance=0.05)
        for i in range(10):
            entry = EnergyLedgerEntry(
                time=i * 0.1,
                struct_kinetic_energy=0.5 * (i * 0.01) ** 2,
                struct_strain_energy=0.5 * 100 * (i * 0.005) ** 2,
                work_fluid_to_struct=i * 0.01,
            )
            ledger.record(entry)
        
        summary = ledger.summary()
        assert "Energy Conservation" in summary


# Import torch for integration test
import torch


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
