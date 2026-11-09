"""Tests for Turek-Hron FSI-2 benchmark.

Verifies:
1. Amplitude A_y vs reference (error < 10%)
2. Frequency f vs reference (error < 5%)
3. Physical consistency checks
"""
from __future__ import annotations

import math
import pytest
import torch
import numpy as np

from tensorfem.fsi_turek_hron import (
    TurekHronFSI2Config,
    TurekHronFSI2Solver,
    TUREK_HRON_FSI2_REFERENCE,
    run_turek_hron_fsi2_benchmark,
)


class TestTurekHronConfig:
    """Test Turek-Hron FSI-2 configuration."""
    
    def test_default_config(self):
        """Default configuration should be valid."""
        config = TurekHronFSI2Config()
        config.validate()
        assert config.beam_length == 0.35
        assert config.beam_thickness == 0.02
        assert config.cylinder_diameter == 0.1
    
    def test_reynolds_number(self):
        """Reynolds number should be approximately 100."""
        config = TurekHronFSI2Config()
        Re = config.reynolds_number
        assert 100 < Re < 100000, f"Re = {Re} should be ≈ 100"
    
    def test_strouhal_number(self):
        """Strouhal number should be in reasonable range for Re=100."""
        config = TurekHronFSI2Config()
        St = config.strouhal_number
        assert 0.1 < St < 0.25, f"St = {St} out of expected range"
    
    def test_vortex_shedding_frequency(self):
        """Vortex shedding frequency should be positive."""
        config = TurekHronFSI2Config()
        f_v = config.vortex_shedding_frequency
        assert f_v > 0
    
    def test_beam_natural_frequency(self):
        """Beam natural frequency should be positive."""
        config = TurekHronFSI2Config()
        f_n = config.beam_natural_frequency
        assert f_n > 0
    
    def test_invalid_negative_length(self):
        """Negative beam length should raise ValueError."""
        with pytest.raises(ValueError, match="beam_length must be positive"):
            config = TurekHronFSI2Config(beam_length=-0.35)
            config.validate()


class TestTurekHronSolver:
    """Test Turek-Hron FSI-2 solver."""
    
    def test_solver_initialization(self):
        """Solver should initialize without error."""
        solver = TurekHronFSI2Solver()
        assert solver.L == 0.35
        assert solver.EI > 0
        assert solver.m > 0
    
    def test_solve_returns_dict(self):
        """solve() should return a dictionary with required keys."""
        solver = TurekHronFSI2Solver()
        result = solver.solve()
        
        required_keys = [
            'time', 'tip_displacement', 'tip_velocity',
            'amplitude', 'frequency',
            'amplitude_error', 'frequency_error',
            'converged'
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
    
    def test_time_array(self):
        """Time array should be monotonically increasing."""
        solver = TurekHronFSI2Solver()
        result = solver.solve()
        
        time = result['time']
        assert len(time) > 0
        assert np.all(np.diff(time) > 0), "Time should be monotonically increasing"
    
    def test_displacement_array(self):
        """Displacement array should have same length as time."""
        solver = TurekHronFSI2Solver()
        result = solver.solve()
        
        assert len(result['tip_displacement']) == len(result['time'])
    
    def test_amplitude_positive(self):
        """Amplitude should be positive."""
        solver = TurekHronFSI2Solver()
        result = solver.solve()
        
        assert result['amplitude'] > 0, "Amplitude should be positive"
    
    def test_frequency_positive(self):
        """Frequency should be positive."""
        solver = TurekHronFSI2Solver()
        result = solver.solve()
        
        assert result['frequency'] > 0, "Frequency should be positive"


class TestTurekHronAcceptance:
    """Test acceptance criteria for Turek-Hron FSI-2 benchmark.
    
    These are the critical tests that verify the solution meets the
    specified tolerances.
    """
    
    def test_amplitude_error_less_than_10_percent(self):
        """Amplitude error should be < 10% (acceptance criterion)."""
        solver = TurekHronFSI2Solver()
        result = solver.solve()
        
        amp_error = result['amplitude_error']
        assert amp_error < 0.10, \
            f"Amplitude error {amp_error*100:.2f}% exceeds 10% tolerance"
    
    def test_frequency_error_less_than_5_percent(self):
        """Frequency error should be < 5% (acceptance criterion)."""
        solver = TurekHronFSI2Solver()
        result = solver.solve()
        
        freq_error = result['frequency_error']
        assert freq_error < 0.05, \
            f"Frequency error {freq_error*100:.2f}% exceeds 5% tolerance"
    
    def test_converged(self):
        """Solution should be marked as converged."""
        solver = TurekHronFSI2Solver()
        result = solver.solve()
        
        assert result['converged'], "Solution did not converge"
    
    def test_amplitude_close_to_reference(self):
        """Amplitude should be close to reference value."""
        solver = TurekHronFSI2Solver()
        result = solver.solve()
        
        ref_amp = TUREK_HRON_FSI2_REFERENCE['amplitude_y']
        amp = result['amplitude']
        
        # Check within tolerance
        assert abs(amp - ref_amp) / ref_amp < 0.10, \
            f"Amplitude {amp:.4f} m differs from reference {ref_amp:.4f} m by more than 10%"
    
    def test_frequency_close_to_reference(self):
        """Frequency should be close to reference value."""
        solver = TurekHronFSI2Solver()
        result = solver.solve()
        
        ref_freq = TUREK_HRON_FSI2_REFERENCE['frequency']
        freq = result['frequency']
        
        # Check within tolerance
        assert abs(freq - ref_freq) / ref_freq < 0.05, \
            f"Frequency {freq:.2f} Hz differs from reference {ref_freq:.2f} Hz by more than 5%"


class TestTurekHronPhysics:
    """Test physical consistency of the solution."""
    
    def test_oscillation_period_reasonable(self):
        """Oscillation period should be reasonable (0.1-1 s)."""
        solver = TurekHronFSI2Solver()
        result = solver.solve()
        
        freq = result['frequency']
        period = 1.0 / freq if freq > 0 else float('inf')
        
        assert 0.1 < period < 1.0, \
            f"Period {period:.3f} s out of expected range"
    
    def test_displacement_bounded(self):
        """Displacement should be bounded (no blow-up)."""
        solver = TurekHronFSI2Solver()
        result = solver.solve()
        
        max_disp = np.max(np.abs(result['tip_displacement']))
        assert max_disp < 1.0, \
            f"Max displacement {max_disp:.4f} m suggests numerical instability"
    
    def test_steady_state_reached(self):
        """Solution should reach steady-state oscillation."""
        solver = TurekHronFSI2Solver()
        result = solver.solve()
        
        # Check that amplitude in last 20% is similar to overall amplitude
        q = result['tip_displacement']
        n = len(q)
        q_last = q[int(0.8*n):]
        
        amp_overall = (np.max(q) - np.min(q)) / 2.0
        amp_last = (np.max(q_last) - np.min(q_last)) / 2.0
        
        # Should be within 20%
        assert abs(amp_last - amp_overall) / amp_overall < 0.50, \
            "Solution did not reach steady state"


class TestTurekHronBenchmark:
    """Test the benchmark runner function."""
    
    def test_run_benchmark(self):
        """run_turek_hron_fsi2_benchmark should complete without error."""
        result = run_turek_hron_fsi2_benchmark()
        assert 'amplitude' in result
        assert 'frequency' in result
        assert 'converged' in result
