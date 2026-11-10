"""Pytest tests for elastoplastic plate buckling module.

Tests cover:
1. Analytical formulas (classical plate buckling, CSR ultimate strength)
2. Mesh generation and boundary conditions
3. Elastic buckling eigenvalue verification
4. Imperfection computation
5. Module integration
"""
from __future__ import annotations

import math

import pytest
import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)

from tensorfem.elastoplastic_plate_buckling import (
    classical_plate_buckling_stress,
    compute_buckling_imperfection,
    csr_plate_ultimate_stress,
    simply_supported_plate_mesh,
    verify_elastic_plate_buckling,
)


# ---------------------------------------------------------------------------
# Analytical formulas
# ---------------------------------------------------------------------------

class TestAnalyticalFormulas:
    """Test closed-form reference solutions."""

    def test_plate_buckling_square(self):
        """Square plate: k = 4 for m=1."""
        a = b = 1.0
        t = 0.01
        E = 206e9
        nu = 0.3
        sigma_cr = classical_plate_buckling_stress(a, b, t, E, nu)
        # k=4 for square plate, sigma_cr = 4*pi^2*D/(b^2*t)
        D = E * t**3 / (12 * (1 - nu**2))
        expected = 4 * math.pi**2 * D / (b**2 * t)
        assert abs(sigma_cr - expected) / expected < 1e-10

    def test_plate_buckling_long_plate(self):
        """Long plate (a >> b): k -> 4."""
        a = 10.0
        b = 1.0
        t = 0.01
        E = 206e9
        nu = 0.3
        sigma_cr = classical_plate_buckling_stress(a, b, t, E, nu)
        # For a/b > 2, k ≈ 4
        D = E * t**3 / (12 * (1 - nu**2))
        expected = 4 * math.pi**2 * D / (b**2 * t)
        assert abs(sigma_cr - expected) / expected < 0.01

    def test_plate_buckling_positive(self):
        """Buckling stress must be positive."""
        sigma_cr = classical_plate_buckling_stress(1.0, 0.5, 0.01, 206e9, 0.3)
        assert sigma_cr > 0

    def test_plate_buckling_invalid_params(self):
        """Invalid parameters should raise."""
        with pytest.raises(ValueError):
            classical_plate_buckling_stress(-1.0, 0.5, 0.01, 206e9, 0.3)
        with pytest.raises(ValueError):
            classical_plate_buckling_stress(1.0, 0.5, 0.01, 206e9, 0.6)

    def test_csr_stocky_plate(self):
        """Stocky plate (beta < 1): sigma_u = sigma_y."""
        # Very thick plate: sigma_cr >> sigma_y
        a, b, t = 0.2, 0.1, 0.05
        E, nu, sigma_y = 206e9, 0.3, 355e6
        sigma_u = csr_plate_ultimate_stress(a, b, t, E, nu, sigma_y)
        assert abs(sigma_u - sigma_y) / sigma_y < 1e-10

    def test_csr_slender_plate(self):
        """Slender plate: sigma_u < sigma_y."""
        a, b, t = 1.0, 0.5, 0.006
        E, nu, sigma_y = 206e9, 0.3, 355e6
        sigma_u = csr_plate_ultimate_stress(a, b, t, E, nu, sigma_y)
        assert sigma_u < sigma_y
        assert sigma_u > 0

    def test_csr_reduces_to_yield_for_stocky(self):
        """CSR formula should not exceed yield."""
        sigma_u = csr_plate_ultimate_stress(1.0, 0.5, 0.01, 206e9, 0.3, 355e6)
        assert sigma_u <= 355e6 * 1.001  # small tolerance


# ---------------------------------------------------------------------------
# Mesh generation
# ---------------------------------------------------------------------------

class TestMeshGeneration:
    """Test plate mesh generation and boundary conditions."""

    def test_mesh_dimensions(self):
        """Mesh should have correct number of nodes and elements."""
        nx, ny = 4, 3
        model, info = simply_supported_plate_mesh(
            1.0, 0.5, 0.01, nx, ny, 206e9, 0.3, 355e6, 1000e6)
        assert len(model.nodes) == (nx + 1) * (ny + 1)
        assert len(model.elements) == nx * ny

    def test_mesh_node_coordinates(self):
        """Nodes should span the plate domain."""
        a, b = 1.0, 0.5
        model, _ = simply_supported_plate_mesh(
            a, b, 0.01, 4, 2, 206e9, 0.3, 355e6, 1000e6)
        assert float(model.nodes[:, 0].min()) == 0.0
        assert abs(float(model.nodes[:, 0].max()) - a) < 1e-12
        assert float(model.nodes[:, 1].min()) == 0.0
        assert abs(float(model.nodes[:, 1].max()) - b) < 1e-12
        assert float(model.nodes[:, 2].abs().max()) < 1e-12  # z=0

    def test_boundary_conditions_rigid_body(self):
        """BCs should prevent rigid body motion but allow deformation."""
        model, info = simply_supported_plate_mesh(
            1.0, 0.5, 0.01, 4, 2, 206e9, 0.3, 355e6, 1000e6)
        fixed = info['fixed_dofs']
        ndof = model.n_dofs
        n_nodes = len(model.nodes)
        
        # Check that some DOFs are free (interior w DOFs)
        free_mask = torch.ones(ndof, dtype=torch.bool)
        free_mask[fixed] = False
        n_free = int(free_mask.sum())
        assert n_free > 0, "All DOFs are fixed!"
        
        # Interior nodes should have free w DOFs
        nx, ny = 4, 2
        for i in range(1, nx):
            for j in range(1, ny):
                node = i * (ny + 1) + j
                w_dof = 6 * node + 2
                assert w_dof not in fixed.tolist(), \
                    f"Interior node {node} w-DOF should be free"


# ---------------------------------------------------------------------------
# Elastic buckling verification
# ---------------------------------------------------------------------------

class TestElasticBuckling:
    """Test elastic buckling eigenvalue computation."""

    def test_buckling_verification_passes(self):
        """Best mesh should pass 2% tolerance."""
        result = verify_elastic_plate_buckling(
            a=1.0, b=0.5, t=0.01,
            young=206e9, poisson=0.3,
            yield_stress=355e6,
            nx=8, ny=4,
        )
        assert result.passed, f"Error {result.relative_error*100:.2f}% exceeds 2%"

    def test_buckling_positive_eigenvalue(self):
        """Buckling load should be positive."""
        result = verify_elastic_plate_buckling(
            a=1.0, b=0.5, t=0.01,
            young=206e9, poisson=0.3,
            yield_stress=355e6,
            nx=8, ny=4,
        )
        assert result.fe_sigma_cr > 0

    def test_buckling_reasonable_magnitude(self):
        """FE buckling should be within 10% of analytical."""
        result = verify_elastic_plate_buckling(
            a=1.0, b=0.5, t=0.01,
            young=206e9, poisson=0.3,
            yield_stress=355e6,
            nx=10, ny=5,
        )
        assert result.relative_error < 0.10


# ---------------------------------------------------------------------------
# Imperfection
# ---------------------------------------------------------------------------

class TestImperfection:
    """Test buckling imperfection computation."""

    def test_imperfection_amplitude(self):
        """Imperfection should have the requested amplitude."""
        model, info = simply_supported_plate_mesh(
            1.0, 0.5, 0.01, 8, 4, 206e9, 0.3, 355e6, 1000e6)
        unit_resultant = torch.tensor([0.01, 0.0, 0.0], dtype=torch.float64)
        amplitude = 0.01 / 200  # t/200
        imperfection, eigenvalues = compute_buckling_imperfection(
            model, info['fixed_dofs'], unit_resultant, amplitude=amplitude)
        max_w = float(imperfection.abs().max())
        assert abs(max_w - amplitude) / amplitude < 0.01

    def test_imperfection_shape(self):
        """Imperfection should have correct shape."""
        model, info = simply_supported_plate_mesh(
            1.0, 0.5, 0.01, 4, 2, 206e9, 0.3, 355e6, 1000e6)
        unit_resultant = torch.tensor([0.01, 0.0, 0.0], dtype=torch.float64)
        imperfection, _ = compute_buckling_imperfection(
            model, info['fixed_dofs'], unit_resultant)
        assert imperfection.shape == (model.n_dofs,)

    def test_imperfection_zero_at_fixed(self):
        """Imperfection should be zero at fixed DOFs."""
        model, info = simply_supported_plate_mesh(
            1.0, 0.5, 0.01, 4, 2, 206e9, 0.3, 355e6, 1000e6)
        unit_resultant = torch.tensor([0.01, 0.0, 0.0], dtype=torch.float64)
        imperfection, _ = compute_buckling_imperfection(
            model, info['fixed_dofs'], unit_resultant)
        fixed_vals = imperfection[info['fixed_dofs']]
        assert float(fixed_vals.abs().max()) < 1e-12


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

class TestIntegration:
    """Integration tests for the full module."""

    def test_module_imports(self):
        """Module should import without errors."""
        from tensorfem import elastoplastic_plate_buckling
        assert hasattr(elastoplastic_plate_buckling, 'classical_plate_buckling_stress')
        assert hasattr(elastoplastic_plate_buckling, 'verify_elastic_plate_buckling')

    def test_layered_shell_model_creation(self):
        """Should create a valid layered shell model."""
        model, info = simply_supported_plate_mesh(
            1.0, 0.5, 0.01, 4, 2, 206e9, 0.3, 355e6, 1000e6)
        model.validate()  # Should not raise

    def test_reference_load_shape(self):
        """Reference load should have correct shape."""
        model, info = simply_supported_plate_mesh(
            1.0, 0.5, 0.01, 4, 2, 206e9, 0.3, 355e6, 1000e6)
        assert info['reference_load'].shape == (model.n_dofs,)
