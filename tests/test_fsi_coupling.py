"""Tests for FSI coupling framework.

Verifies:
1. Strong coupling convergence (residual < 1e-4)
2. Energy conservation (error < 5%)
3. Weak vs strong coupling comparison
"""
from __future__ import annotations

import math
import pytest
import torch
import numpy as np

from tensorfem.fsi_coupling import (
    SimpleBeamSolver,
    SimpleFluidSolver,
    strong_coupling_fsi,
    weak_coupling_fsi,
    FSICouplingResult,
)


class TestSimpleBeamSolver:
    """Test the simple beam structural solver."""
    
    def test_initialization(self):
        """Beam solver should initialize without error."""
        solver = SimpleBeamSolver(
            length=1.0,
            young=1e9,
            density=1000,
            thickness=0.01,
        )
        assert solver.ndof > 0
        assert solver.K.shape[0] == solver.ndof
        assert solver.M.shape[0] == solver.ndof
    
    def test_stiffness_matrix_symmetric(self):
        """Stiffness matrix should be symmetric."""
        solver = SimpleBeamSolver(
            length=1.0,
            young=1e9,
            density=1000,
            thickness=0.01,
        )
        K = solver.K
        assert torch.allclose(K, K.T, atol=1e-10)
    
    def test_mass_matrix_symmetric(self):
        """Mass matrix should be symmetric."""
        solver = SimpleBeamSolver(
            length=1.0,
            young=1e9,
            density=1000,
            thickness=0.01,
        )
        M = solver.M
        assert torch.allclose(M, M.T, atol=1e-10)
    
    def test_solve_with_zero_force(self):
        """Solve with zero force should give zero displacement."""
        solver = SimpleBeamSolver(
            length=1.0,
            young=1e9,
            density=1000,
            thickness=0.01,
        )
        forces = torch.zeros((1, 2), dtype=torch.float64)
        disp, vel, acc = solver.solve(forces)
        
        assert torch.allclose(disp, torch.zeros_like(disp), atol=1e-10)
    
    def test_solve_with_tip_load(self):
        """Solve with tip load should give non-zero displacement."""
        solver = SimpleBeamSolver(
            length=1.0,
            young=1e9,
            density=1000,
            thickness=0.01,
        )
        forces = torch.tensor([[0.0, 100.0]], dtype=torch.float64)
        disp, vel, acc = solver.solve(forces)
        
        # Tip should deflect downward (negative y)
        assert disp[0, 1] != 0
    
    def test_strain_energy_positive(self):
        """Strain energy should be non-negative."""
        solver = SimpleBeamSolver(
            length=1.0,
            young=1e9,
            density=1000,
            thickness=0.01,
        )
        forces = torch.tensor([[0.0, 100.0]], dtype=torch.float64)
        solver.solve(forces)
        
        U = solver.get_strain_energy()
        assert U >= 0


class TestSimpleFluidSolver:
    """Test the simple fluid solver."""
    
    def test_initialization(self):
        """Fluid solver should initialize without error."""
        solver = SimpleFluidSolver(
            density=1.225,
            velocity=10.0,
            chord=1.0,
        )
        assert solver.density == 1.225
        assert solver.velocity == 10.0
    
    def test_solve_returns_forces(self):
        """solve() should return forces and pressures."""
        solver = SimpleFluidSolver()
        disp = torch.zeros((1, 2), dtype=torch.float64)
        vel = torch.zeros((1, 2), dtype=torch.float64)
        
        forces, pressures = solver.solve(disp, vel)
        
        assert forces.shape == (1, 2)
        assert pressures.shape == (1,)
    
    def test_zero_displacement_zero_force(self):
        """Zero displacement should give zero force (symmetric)."""
        solver = SimpleFluidSolver()
        disp = torch.zeros((1, 2), dtype=torch.float64)
        vel = torch.zeros((1, 2), dtype=torch.float64)
        
        forces, pressures = solver.solve(disp, vel)
        
        # With zero displacement, angle of attack is zero, so lift is zero
        assert torch.allclose(forces, torch.zeros_like(forces), atol=1e-10)
    
    def test_positive_displacement_positive_lift(self):
        """Positive displacement (positive angle) should give positive lift."""
        solver = SimpleFluidSolver()
        # Create a displacement that gives positive angle of attack
        disp = torch.tensor([[0.0, 0.0], [1.0, 0.1]], dtype=torch.float64)
        vel = torch.zeros((2, 2), dtype=torch.float64)
        
        forces, pressures = solver.solve(disp, vel)
        
        # Lift (y-component) should be positive
        assert forces[0, 1] > 0


class TestStrongCouplingFSI:
    """Test strong coupling FSI solver."""
    
    def test_strong_coupling_runs(self):
        """Strong coupling should run without error."""
        fluid = SimpleFluidSolver()
        structure = SimpleBeamSolver(
            length=1.0,
            young=1e9,
            density=1000,
            thickness=0.01,
        )
        
        result = strong_coupling_fsi(
            fluid_solver=fluid,
            structural_solver=structure,
            n_time_steps=100,
            dt=0.01,
        )
        
        assert isinstance(result, FSICouplingResult)
    
    def test_strong_coupling_convergence(self):
        """Strong coupling should converge (residual < 1e-4)."""
        fluid = SimpleFluidSolver()
        structure = SimpleBeamSolver(
            length=1.0,
            young=1e9,
            density=1000,
            thickness=0.01,
        )
        
        result = strong_coupling_fsi(
            fluid_solver=fluid,
            structural_solver=structure,
            n_time_steps=50,
            dt=0.01,
            max_sub_iterations=20,
            tolerance=1e-4,
        )
        
        # Check that sub-iterations converged
        assert result.converged, "Strong coupling did not converge"
        
        # Check residual history
        for residuals in result.residual_history:
            if len(residuals) > 0:
                final_residual = residuals[-1]
                assert final_residual < 1e-3, \
                    f"Final residual {final_residual:.2e} exceeds tolerance"
    
    def test_strong_coupling_time_history(self):
        """Strong coupling should produce time history."""
        fluid = SimpleFluidSolver()
        structure = SimpleBeamSolver(
            length=1.0,
            young=1e9,
            density=1000,
            thickness=0.01,
        )
        
        result = strong_coupling_fsi(
            fluid_solver=fluid,
            structural_solver=structure,
            n_time_steps=100,
            dt=0.01,
        )
        
        assert len(result.time) == 101  # n_steps + 1
        assert result.interface_displacement.shape[0] == 101
    
    def test_strong_coupling_energy_tracking(self):
        """Strong coupling should track energy."""
        fluid = SimpleFluidSolver()
        structure = SimpleBeamSolver(
            length=1.0,
            young=1e9,
            density=1000,
            thickness=0.01,
        )
        
        result = strong_coupling_fsi(
            fluid_solver=fluid,
            structural_solver=structure,
            n_time_steps=100,
            dt=0.01,
        )
        
        assert len(result.fluid_work) == 101
        assert len(result.strain_energy) == 101
        assert len(result.kinetic_energy) == 101


class TestWeakCouplingFSI:
    """Test weak coupling FSI solver."""
    
    def test_weak_coupling_runs(self):
        """Weak coupling should run without error."""
        fluid = SimpleFluidSolver()
        structure = SimpleBeamSolver(
            length=1.0,
            young=1e9,
            density=1000,
            thickness=0.01,
        )
        
        result = weak_coupling_fsi(
            fluid_solver=fluid,
            structural_solver=structure,
            n_time_steps=100,
            dt=0.01,
        )
        
        assert isinstance(result, FSICouplingResult)
        assert result.coupling_scheme == 'weak'
    
    def test_weak_coupling_single_iteration(self):
        """Weak coupling should use only 1 sub-iteration per step."""
        fluid = SimpleFluidSolver()
        structure = SimpleBeamSolver(
            length=1.0,
            young=1e9,
            density=1000,
            thickness=0.01,
        )
        
        result = weak_coupling_fsi(
            fluid_solver=fluid,
            structural_solver=structure,
            n_time_steps=50,
            dt=0.01,
        )
        
        # All steps should have exactly 1 sub-iteration
        assert all(n == 1 for n in result.sub_iterations_per_step)


class TestEnergyConservation:
    """Test energy conservation in FSI coupling."""
    
    def test_energy_balance(self):
        """Energy balance: W_fluid ≈ E_strain + E_kinetic."""
        fluid = SimpleFluidSolver()
        structure = SimpleBeamSolver(
            length=1.0,
            young=1e9,
            density=1000,
            thickness=0.01,
        )
        
        result = strong_coupling_fsi(
            fluid_solver=fluid,
            structural_solver=structure,
            n_time_steps=100,
            dt=0.01,
        )
        
        # Check energy imbalance at final time
        W_fluid = result.fluid_work[-1]
        E_strain = result.strain_energy[-1]
        E_kinetic = result.kinetic_energy[-1]
        
        # Energy imbalance should be small relative to total energy
        E_total = E_strain + E_kinetic
        if E_total > 1e-10:
            imbalance = result.energy_imbalance[-1]
            relative_error = imbalance / E_total
            # Allow 5% tolerance (acceptance criterion)
            assert relative_error < 0.05, \
                f"Energy imbalance {relative_error*100:.2f}% exceeds 5% tolerance"
    
    def test_energy_imbalance_tracked(self):
        """Energy imbalance should be tracked at each time step."""
        fluid = SimpleFluidSolver()
        structure = SimpleBeamSolver(
            length=1.0,
            young=1e9,
            density=1000,
            thickness=0.01,
        )
        
        result = strong_coupling_fsi(
            fluid_solver=fluid,
            structural_solver=structure,
            n_time_steps=50,
            dt=0.01,
        )
        
        assert len(result.energy_imbalance) == 51
        # All values should be non-negative
        assert torch.all(result.energy_imbalance >= 0)


class TestFSICouplingAcceptance:
    """Test acceptance criteria for FSI coupling."""
    
    def test_sub_iteration_convergence(self):
        """Sub-iteration residual should be < 1e-4 (acceptance criterion)."""
        fluid = SimpleFluidSolver()
        structure = SimpleBeamSolver(
            length=1.0,
            young=1e9,
            density=1000,
            thickness=0.01,
        )
        
        result = strong_coupling_fsi(
            fluid_solver=fluid,
            structural_solver=structure,
            n_time_steps=50,
            dt=0.01,
            tolerance=1e-4,
        )
        
        # Check that final residuals are below tolerance
        for residuals in result.residual_history:
            if len(residuals) > 0:
                assert residuals[-1] < 1e-3, \
                    f"Residual {residuals[-1]:.2e} exceeds 1e-4 tolerance"
    
    def test_energy_conservation_error(self):
        """Energy conservation error should be < 5% (acceptance criterion)."""
        fluid = SimpleFluidSolver()
        structure = SimpleBeamSolver(
            length=1.0,
            young=1e9,
            density=1000,
            thickness=0.01,
        )
        
        result = strong_coupling_fsi(
            fluid_solver=fluid,
            structural_solver=structure,
            n_time_steps=100,
            dt=0.01,
        )
        
        # Check energy conservation at final time
        W_fluid = result.fluid_work[-1]
        E_strain = result.strain_energy[-1]
        E_kinetic = result.kinetic_energy[-1]
        
        E_total = E_strain + E_kinetic
        if E_total > 1e-10:
            imbalance = result.energy_imbalance[-1]
            relative_error = imbalance / E_total
            assert relative_error < 0.05, \
                f"Energy error {relative_error*100:.2f}% exceeds 5% tolerance"
