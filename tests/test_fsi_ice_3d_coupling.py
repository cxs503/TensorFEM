"""Local helper checks and fail-closed admission for the quarantined FSI3D path.

These tests do not establish fluid coupling, ice fracture, mass conservation,
energy conservation, or experimental agreement.
"""
import pytest
import torch
from tensorfem.fsi_ice_3d_coupling import (
    IceMaterialConfig, GeometryConfig, SimulationConfig,
    generate_ice_mesh, hex8_shape_functions, hex8_stiffness_matrix,
    DruckerPragerPlasticity, FSI3DIceCouplingSolver, compute_mass_conservation_residual, check_ice_failure,
    FSI3DUnavailableError, fsi3d_qualification_status
)

D = torch.float64


class TestIceMaterialConfig:
    """Test ice material configuration."""
    
    def test_default_values(self):
        """Test default material properties."""
        config = IceMaterialConfig()
        assert config.young == 5.0e9
        assert config.poisson == 0.33
        assert config.density == 917.0
        assert config.cohesion == 1.0e6
    
    def test_validation_positive_young(self):
        """Test validation rejects non-positive Young's modulus."""
        with pytest.raises(ValueError, match="Young's modulus must be positive"):
            IceMaterialConfig(young=-1.0).validate()
    
    def test_validation_poisson_range(self):
        """Test validation rejects Poisson's ratio out of range."""
        with pytest.raises(ValueError, match="Poisson's ratio out of range"):
            IceMaterialConfig(poisson=0.6).validate()
    
    def test_shear_modulus(self):
        """Test shear modulus calculation."""
        config = IceMaterialConfig(young=5.0e9, poisson=0.33)
        G = config.shear_modulus
        expected = 5.0e9 / (2.0 * 1.33)
        assert abs(G - expected) < 1e-6
    
    def test_bulk_modulus(self):
        """Test bulk modulus calculation."""
        config = IceMaterialConfig(young=5.0e9, poisson=0.33)
        K = config.bulk_modulus
        expected = 5.0e9 / (3.0 * 0.34)
        assert abs(K - expected) < 1e-6
    
    def test_drucker_prager_parameters(self):
        """Test Drucker-Prager parameter calculation."""
        config = IceMaterialConfig(cohesion=1.0e6, friction_angle=30.0)
        alpha = config.drucker_prager_alpha
        k = config.drucker_prager_k
        assert alpha > 0
        assert k > 0


class TestGeometryConfig:
    """Test geometry configuration."""
    
    def test_default_dimensions(self):
        """Test default geometry dimensions."""
        config = GeometryConfig()
        assert config.ice_length == 10.0
        assert config.ice_width == 10.0
        assert config.ice_thickness == 0.5
    
    def test_validation_positive_dimensions(self):
        """Test validation rejects non-positive dimensions."""
        with pytest.raises(ValueError, match="ice_length must be positive"):
            GeometryConfig(ice_length=-1.0).validate()
    
    def test_validation_mesh_resolution(self):
        """Test validation rejects coarse mesh."""
        with pytest.raises(ValueError, match="Ice mesh resolution too coarse"):
            GeometryConfig(ice_nx=1).validate()


class TestMeshGeneration:
    """Test mesh generation."""
    
    def test_ice_mesh_node_count(self):
        """Test ice mesh has correct number of nodes."""
        config = GeometryConfig(ice_nx=5, ice_ny=5, ice_nz=2)
        mesh = generate_ice_mesh(config)
        expected_nodes = (5 + 1) * (5 + 1) * (2 + 1)
        assert mesh['nodes'].shape[0] == expected_nodes
    
    def test_ice_mesh_element_count(self):
        """Test ice mesh has correct number of elements."""
        config = GeometryConfig(ice_nx=5, ice_ny=5, ice_nz=2)
        mesh = generate_ice_mesh(config)
        expected_elements = 5 * 5 * 2
        assert mesh['elements'].shape[0] == expected_elements
    
    def test_ice_mesh_node_coordinates(self):
        """Test ice mesh node coordinates are within bounds."""
        config = GeometryConfig(ice_length=10.0, ice_width=10.0, ice_thickness=0.5)
        mesh = generate_ice_mesh(config)
        
        assert mesh['nodes'][:, 0].min() >= 0.0
        assert mesh['nodes'][:, 0].max() <= 10.0
        assert mesh['nodes'][:, 1].min() >= 0.0
        assert mesh['nodes'][:, 1].max() <= 10.0
        assert mesh['nodes'][:, 2].min() >= 0.0
        assert mesh['nodes'][:, 2].max() <= 0.5


class TestShapeFunctions:
    """Test shape functions."""
    
    def test_partition_of_unity(self):
        """Test shape functions sum to 1."""
        N, _ = hex8_shape_functions(0.0, 0.0, 0.0)
        assert abs(N.sum().item() - 1.0) < 1e-10
    
    def test_corner_values(self):
        """Test shape function values at corners."""
        # At corner (1, 1, 1), N[6] should be 1, others 0
        N, _ = hex8_shape_functions(1.0, 1.0, 1.0)
        assert abs(N[6].item() - 1.0) < 1e-10
        for i in [0, 1, 2, 3, 4, 5, 7]:
            assert abs(N[i].item()) < 1e-10


class TestStiffnessMatrix:
    """Test element stiffness matrix."""
    
    def test_symmetry(self):
        """Test stiffness matrix is symmetric."""
        coords = torch.tensor([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ], dtype=D)
        ke = hex8_stiffness_matrix(coords, 5.0e9, 0.33)
        
        assert torch.allclose(ke, ke.T, atol=1e-6)
    
    def test_positive_definite(self):
        """Test stiffness matrix is positive definite."""
        coords = torch.tensor([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ], dtype=D)
        ke = hex8_stiffness_matrix(coords, 5.0e9, 0.33)
        
        # Check eigenvalues are positive
        eigenvalues = torch.linalg.eigvalsh(ke)
        assert (eigenvalues > -1e-6).all()


class TestDruckerPragerPlasticity:
    """Test Drucker-Prager plasticity model."""
    
    def test_yield_function_elastic(self):
        """Test yield function returns negative for elastic state."""
        config = IceMaterialConfig()
        plasticity = DruckerPragerPlasticity(config)
        
        # Hydrostatic compression (should be elastic)
        stress = torch.tensor([-1e6, -1e6, -1e6, 0, 0, 0], dtype=D)
        f = plasticity.yield_function(stress)
        assert f < 0
    
    def test_yield_function_yielded(self):
        """Test yield function returns positive for yielded state."""
        config = IceMaterialConfig(cohesion=1.0e6, friction_angle=30.0)
        plasticity = DruckerPragerPlasticity(config)
        
        # High deviatoric stress (should yield)
        stress = torch.tensor([2e6, -1e6, 0, 0, 0, 0], dtype=D)
        f = plasticity.yield_function(stress)
        assert f > 0
    
    def test_return_mapping_elastic(self):
        """Test return mapping for elastic trial stress."""
        config = IceMaterialConfig()
        plasticity = DruckerPragerPlasticity(config)
        
        stress_trial = torch.tensor([-1e6, -1e6, -1e6, 0, 0, 0], dtype=D)
        stress, plastic_strain, yielded = plasticity.return_mapping(stress_trial, 0.0)
        
        assert not yielded
        assert torch.allclose(stress, stress_trial)
        assert plastic_strain == 0.0
    
    def test_return_mapping_plastic(self):
        """Test return mapping for plastic trial stress."""
        config = IceMaterialConfig(cohesion=1.0e6, friction_angle=30.0)
        plasticity = DruckerPragerPlasticity(config)
        
        stress_trial = torch.tensor([2e6, -1e6, 0, 0, 0, 0], dtype=D)
        with pytest.raises(FSI3DUnavailableError, match="return mapping"):
            plasticity.return_mapping(stress_trial, 0.0)


class TestFSI3DAdmission:
    def test_constructor_rejects_before_assembly(self, monkeypatch):
        def unexpected_allocation(*args):
            pytest.fail("quarantined solver attempted mesh assembly")
        monkeypatch.setattr("tensorfem.fsi_ice_3d_coupling.generate_ice_mesh", unexpected_allocation)
        with pytest.raises(FSI3DUnavailableError, match="blocked"):
            FSI3DIceCouplingSolver(SimulationConfig())

    @pytest.mark.parametrize("method,args", [
        ("step", (None,)), ("run", (None,)), ("solve_ice_step", (None, .01)),
        ("compute_contact_forces", (None,)), ("compute_energy_balance", ()),
    ])
    def test_preexisting_instance_cannot_emit_results(self, method, args):
        instance = object.__new__(FSI3DIceCouplingSolver)
        with pytest.raises(FSI3DUnavailableError, match="blocked"):
            getattr(instance, method)(*args)

    @pytest.mark.parametrize("velocity", [torch.zeros((3, 3)), torch.ones((3, 3)),
                                         torch.full((3, 3), float("nan"))])
    def test_missing_mass_diagnostic_never_becomes_zero(self, velocity):
        with pytest.raises(FSI3DUnavailableError, match="fluxes"):
            compute_mass_conservation_residual(velocity, .01)

    def test_status_and_validation_cli_reject(self, tmp_path):
        import json
        import os
        from pathlib import Path
        import subprocess
        import sys
        status = fsi3d_qualification_status()
        assert status["status"] == "blocked" and status["passed"] is False
        root = Path(__file__).resolve().parents[1]
        output = tmp_path / "status.json"
        result = subprocess.run(
            [sys.executable, str(root / "validate_fsi_ice_3d.py"), "--output", str(output)],
            cwd=tmp_path, env=dict(os.environ, PYTHONPATH=str(root / "src")),
            capture_output=True, text=True,
        )
        assert result.returncode == 2
        assert json.loads(result.stdout) == json.loads(output.read_text()) == status
        assert set(tmp_path.iterdir()) == {output}


def test_generated_hexes_have_positive_jacobians_and_correct_volume():
    config = GeometryConfig(ice_length=3., ice_width=2., ice_thickness=.5,
                            ice_nx=3, ice_ny=2, ice_nz=1)
    mesh = generate_ice_mesh(config)
    _, derivatives = hex8_shape_functions(0., 0., 0.)
    determinants = torch.linalg.det(derivatives.T @ mesh["nodes"][mesh["elements"]])
    assert torch.all(determinants > 0)
    assert float(8 * determinants.sum()) == pytest.approx(3.)


def test_inverted_hex_is_rejected():
    config = GeometryConfig(ice_nx=2, ice_ny=1, ice_nz=1)
    mesh = generate_ice_mesh(config)
    coords = mesh["nodes"][mesh["elements"][0]].clone()
    coords[:, 0] *= -1
    with pytest.raises(ValueError, match="positive finite Jacobian"):
        hex8_stiffness_matrix(coords, 5e9, .33)


class TestC2_IceFailure:
    """Test C2: Ice bending failure criterion."""
    
    def test_failure_detection(self):
        """Test ice failure is detected when stress exceeds strength."""
        config = IceMaterialConfig(cohesion=1.0e6, friction_angle=30.0)
        
        # Stress state that should cause failure
        stress = torch.tensor([2e6, -1e6, 0, 0, 0, 0], dtype=D)
        failed = check_ice_failure(stress, config)
        
        assert failed
    
    def test_no_failure_elastic(self):
        """Test no failure for elastic stress state."""
        config = IceMaterialConfig(cohesion=1.0e6, friction_angle=30.0)
        
        # Stress state that should not cause failure
        stress = torch.tensor([-1e6, -1e6, -1e6, 0, 0, 0], dtype=D)
        failed = check_ice_failure(stress, config)
        
        assert not failed


class TestSimulationConfig:
    """Test simulation configuration."""
    
    def test_default_config(self):
        """Test default configuration is valid."""
        config = SimulationConfig(geometry=GeometryConfig(ice_nx=4, ice_ny=4, ice_nz=1))
        config.validate()  # Should not raise
    
    def test_invalid_time_step(self):
        """Test validation rejects invalid time step."""
        with pytest.raises(ValueError, match="Time step must be positive"):
            SimulationConfig(dt=-0.01).validate()
    
    def test_invalid_total_time(self):
        """Test validation rejects invalid total time."""
        with pytest.raises(ValueError, match="Total time must be positive"):
            SimulationConfig(time_total=-1.0).validate()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
