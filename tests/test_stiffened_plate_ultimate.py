"""Pytest suite for P127W2-N06: stiffened plate ultimate strength verification."""
from __future__ import annotations

import math
import json
from pathlib import Path

import pytest
import numpy as np

from tensorfem.stiffened_plate_ultimate import (
    StiffenedPlateGeometry,
    MaterialProperties,
    classical_plate_buckling_stress,
    classical_stiffened_panel_buckling_load,
    build_stiffened_plate_mesh,
    apply_imperfection,
    fd_plate_buckling_eigenvalue,
    run_arc_length_postbuckling,
    smith_stiffened_plate_mk_curve,
    csr_section_modulus_check,
    mesh_convergence_gci,
    paik2007_reference_panels,
    compare_with_paik2007,
    run_full_verification,
    PROVENANCE,
    CRITERIA,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_geometry():
    return StiffenedPlateGeometry()


@pytest.fixture
def default_material():
    return MaterialProperties()


# ---------------------------------------------------------------------------
# Geometry tests
# ---------------------------------------------------------------------------

class TestGeometry:
    def test_default_geometry_values(self, default_geometry):
        geo = default_geometry
        assert geo.length == pytest.approx(0.800)
        assert geo.width == pytest.approx(0.400)
        assert geo.plate_thickness == pytest.approx(0.006)
        assert geo.web_height == pytest.approx(0.150)
        assert geo.web_thickness == pytest.approx(0.010)

    def test_plate_slenderness(self, default_geometry):
        geo = default_geometry
        beta = geo.plate_slenderness
        expected = (0.4 / 0.006) * math.sqrt(355e6 / 2.06e11)
        assert beta == pytest.approx(expected, rel=1e-10)
        assert beta > 1.0  # slender plate

    def test_stiffener_area(self, default_geometry):
        geo = default_geometry
        expected = 0.150 * 0.010 + 0.090 * 0.010
        assert geo.stiffener_area == pytest.approx(expected)

    def test_stiffener_second_moment_positive(self, default_geometry):
        geo = default_geometry
        I = geo._stiffener_second_moment()
        assert I > 0

    def test_geometry_validation_rejects_negative(self):
        with pytest.raises(ValueError):
            StiffenedPlateGeometry(plate_thickness=-0.006).validate()


class TestMaterial:
    def test_default_material(self, default_material):
        mat = default_material
        assert mat.young == pytest.approx(2.06e11)
        assert mat.yield_stress == pytest.approx(355e6)
        assert mat.tangent_modulus == pytest.approx(1.0e9)

    def test_material_validation(self):
        mat = MaterialProperties()
        mat.validate()

    def test_material_rejects_invalid_poisson(self):
        with pytest.raises(ValueError):
            MaterialProperties(poisson=0.6).validate()


# ---------------------------------------------------------------------------
# Classical buckling tests
# ---------------------------------------------------------------------------

class TestClassicalBuckling:
    def test_plate_buckling_stress_square(self):
        """Square plate: k=4, sigma_cr = 4*pi^2*D/(b^2*t)."""
        L, W, t = 1.0, 1.0, 0.01
        E, nu = 2.06e11, 0.3
        sigma = classical_plate_buckling_stress(
            length=L, width=W, thickness=t, young=E, poisson=nu)
        D = E * t**3 / (12.0 * (1.0 - nu**2))
        expected = 4.0 * math.pi**2 * D / (W**2 * t)
        assert sigma == pytest.approx(expected, rel=1e-10)

    def test_plate_buckling_stress_long_plate(self):
        L, W, t = 2.0, 1.0, 0.01
        E, nu = 2.06e11, 0.3
        sigma = classical_plate_buckling_stress(
            length=L, width=W, thickness=t, young=E, poisson=nu)
        assert sigma > 0

    def test_stiffened_panel_buckling_load(self, default_geometry, default_material):
        result = classical_stiffened_panel_buckling_load(default_geometry, default_material)
        assert result["sigma_cr"] > 0
        assert result["sigma_cr_plate"] > 0
        assert result["P_cr"] > 0
        assert result["P_squash"] > 0
        # For slender plate, buckling occurs before squash
        assert result["P_cr"] < result["P_squash"]
        assert result["buckling_coefficient"] >= 4.0
        assert result["buckling_mode_m"] >= 1

    def test_buckling_coefficient_for_2x1_plate(self):
        geo = StiffenedPlateGeometry(length=0.8, width=0.4)
        mat = MaterialProperties()
        result = classical_stiffened_panel_buckling_load(geo, mat)
        assert result["buckling_coefficient"] >= 4.0


# ---------------------------------------------------------------------------
# Mesh tests
# ---------------------------------------------------------------------------

class TestMesh:
    def test_mesh_node_count(self, default_geometry):
        nx, ny = 20, 10
        mesh = build_stiffened_plate_mesh(default_geometry, nx=nx, ny=ny)
        expected_nodes = (nx + 1) * (ny + 1)
        assert mesh["n_nodes"] == expected_nodes

    def test_mesh_element_count(self, default_geometry):
        nx, ny = 20, 10
        mesh = build_stiffened_plate_mesh(default_geometry, nx=nx, ny=ny)
        expected_elements = nx * ny
        assert mesh["n_shell_elements"] == expected_elements

    def test_mesh_beam_elements(self, default_geometry):
        nx, ny = 20, 10
        mesh = build_stiffened_plate_mesh(default_geometry, nx=nx, ny=ny)
        assert mesh["n_beam_elements"] == nx

    def test_imperfection_amplitude(self, default_geometry):
        mesh = build_stiffened_plate_mesh(default_geometry, nx=20, ny=10)
        amp = 0.1 * default_geometry.plate_thickness
        imperfect = apply_imperfection(
            mesh["nodes"], length=default_geometry.length,
            width=default_geometry.width, amplitude=amp)
        max_w = float(imperfect[:, 2].abs().max())
        assert max_w == pytest.approx(amp, rel=1e-10)


# ---------------------------------------------------------------------------
# FD buckling tests
# ---------------------------------------------------------------------------

class TestFDBuckling:
    def test_fd_buckling_converges(self, default_geometry, default_material):
        geo = default_geometry
        mat = default_material
        result = fd_plate_buckling_eigenvalue(
            length=geo.length, width=geo.width, thickness=geo.plate_thickness,
            young=mat.young, poisson=mat.poisson,
            grid_x=20, grid_y=10,
        )
        assert result["critical_stress"] > 0
        assert result["relative_error_vs_navier"] < 0.03

    def test_fd_buckling_vs_classical_plate(self, default_geometry, default_material):
        """FD buckling should match classical plate buckling (not stiffened panel)."""
        geo = default_geometry
        mat = default_material
        classical = classical_stiffened_panel_buckling_load(geo, mat)
        fd = fd_plate_buckling_eigenvalue(
            length=geo.length, width=geo.width, thickness=geo.plate_thickness,
            young=mat.young, poisson=mat.poisson,
            grid_x=20, grid_y=10,
        )
        # Compare FD with classical PLATE buckling (not stiffened panel)
        error = abs(fd["critical_stress"] - classical["sigma_cr_plate"]) / classical["sigma_cr_plate"]
        assert error < 0.05


# ---------------------------------------------------------------------------
# Arc-length post-buckling tests
# ---------------------------------------------------------------------------

class TestArcLengthPostbuckling:
    def test_postbuckling_returns_points(self, default_geometry, default_material):
        result = run_arc_length_postbuckling(
            default_geometry, default_material, nx=20, ny=10, steps=20)
        assert len(result["points"]) > 0
        assert result["peak_load_Pu"] > 0
        assert result["buckling_load_Pcr"] > 0

    def test_peak_load_less_than_squash(self, default_geometry, default_material):
        result = run_arc_length_postbuckling(
            default_geometry, default_material, nx=20, ny=10, steps=20)
        geo = default_geometry
        mat = default_material
        P_squash = mat.yield_stress * (geo.plate_thickness * geo.width + geo.stiffener_area)
        assert result["peak_load_Pu"] <= P_squash * 1.01

    def test_energy_positive(self, default_geometry, default_material):
        result = run_arc_length_postbuckling(
            default_geometry, default_material, nx=20, ny=10, steps=20)
        assert result["energy_absorption"] > 0

    def test_load_starts_at_zero(self, default_geometry, default_material):
        result = run_arc_length_postbuckling(
            default_geometry, default_material, nx=20, ny=10, steps=20)
        assert result["points"][0]["load"] == 0.0


# ---------------------------------------------------------------------------
# Smith method tests
# ---------------------------------------------------------------------------

class TestSmithMethod:
    def test_smith_mk_curve(self, default_geometry, default_material):
        result = smith_stiffened_plate_mk_curve(default_geometry, default_material)
        assert result["ultimate_moment"] > 0
        assert len(result["mk_curve"]) > 0
        assert result["plate_ultimate_stress"] > 0
        assert result["stiffener_ultimate_stress"] > 0

    def test_smith_section_properties(self, default_geometry, default_material):
        result = smith_stiffened_plate_mk_curve(default_geometry, default_material)
        props = result["section_properties"]
        assert props["A_total"] > 0
        assert props["I_total"] > 0
        assert props["elastic_rigidity"] > 0

    def test_smith_mk_curve_has_peak(self, default_geometry, default_material):
        result = smith_stiffened_plate_mk_curve(default_geometry, default_material)
        mk = result["mk_curve"]
        moments = [abs(p["moment"]) for p in mk]
        peak_m = max(moments)
        peak_idx = moments.index(peak_m)
        # After peak, moment should not increase significantly
        if peak_idx < len(mk) - 1:
            assert abs(mk[peak_idx + 1]["moment"]) <= peak_m * 1.01


# ---------------------------------------------------------------------------
# CSR check tests
# ---------------------------------------------------------------------------

class TestCSRCheck:
    def test_csr_section_modulus_positive(self, default_geometry, default_material):
        result = csr_section_modulus_check(default_geometry, default_material)
        assert result["section_properties"]["Z_act"] > 0
        assert result["csr_check"]["Z_req"] > 0

    def test_csr_ratio(self, default_geometry, default_material):
        result = csr_section_modulus_check(default_geometry, default_material)
        ratio = result["csr_check"]["Z_act_over_Z_req"]
        assert ratio > 0

    def test_csr_with_small_moment_passes(self, default_geometry, default_material):
        result = csr_section_modulus_check(
            default_geometry, default_material, design_bending_moment=1e3)
        assert result["csr_check"]["passes"] is True
        assert result["csr_check"]["Z_act_over_Z_req"] > 1.0


# ---------------------------------------------------------------------------
# Mesh convergence tests
# ---------------------------------------------------------------------------

class TestMeshConvergence:
    def test_gci_computed(self, default_geometry, default_material):
        result = mesh_convergence_gci(
            default_geometry, default_material, mesh_levels=(10, 20, 40))
        assert "GCI_fine" in result
        assert "GCI_medium" in result
        assert "observed_order" in result

    def test_gci_passes_3pct(self, default_geometry, default_material):
        result = mesh_convergence_gci(
            default_geometry, default_material, mesh_levels=(10, 20, 40))
        assert result["passes_3pct"] is True


# ---------------------------------------------------------------------------
# Experimental comparison tests
# ---------------------------------------------------------------------------

class TestExperimentalComparison:
    def test_synthetic_panels_generated(self):
        panels = paik2007_reference_panels()
        assert len(panels) == 6
        for p in panels:
            assert p["primary_source_verified"] is False
            assert "SYNTHETIC" in p["provenance"]

    def test_compare_with_synthetic(self, default_geometry, default_material):
        geo = default_geometry
        mat = default_material
        panels = paik2007_reference_panels()
        avg_Pu = np.mean([p["measured_ultimate_load_N"] for p in panels])
        result = compare_with_paik2007(geo, mat, float(avg_Pu), use_synthetic=True)
        assert result["n_specimens"] == 6
        assert result["avg_relative_error"] < 0.10

    def test_compare_with_wrong_value_fails(self, default_geometry, default_material):
        geo = default_geometry
        mat = default_material
        result = compare_with_paik2007(geo, mat, 1e6, use_synthetic=True)
        assert result["passes_10pct"] is False


# ---------------------------------------------------------------------------
# Full verification runner tests
# ---------------------------------------------------------------------------

class TestFullVerification:
    def test_full_verification_runs(self, default_geometry, default_material):
        result = run_full_verification(
            geometry=default_geometry,
            material=default_material,
            mesh_levels=(10, 20, 40),
            arc_steps=30,
        )
        assert result["schema"] == "tensorfem.stiffened-plate-ultimate/1.0"
        assert result["campaign"] == "P127W2-N06"
        assert "classical_buckling" in result
        assert "postbuckling" in result
        assert "smith_method" in result
        assert "csr_check" in result
        assert "mesh_convergence" in result
        assert "experimental_comparison" in result
        assert "acceptance_criteria" in result

    def test_full_verification_acceptance_criteria(self, default_geometry, default_material):
        result = run_full_verification(
            geometry=default_geometry,
            material=default_material,
            mesh_levels=(10, 20, 40),
            arc_steps=30,
        )
        criteria = result["acceptance_criteria"]
        assert criteria["a_Pcr_vs_classical_5pct"]["passes"] is True
        assert criteria["d_smith_Mu_vs_plastic_8pct"]["passes"] is True
        assert criteria["e_CSR_Z_act_over_Z_req"]["passes"] is True
        assert criteria["f_GCI_3pct"]["passes"] is True

    def test_full_verification_saves_json(self, default_geometry, default_material, tmp_path):
        save_dir = tmp_path / "results"
        result = run_full_verification(
            geometry=default_geometry,
            material=default_material,
            mesh_levels=(10, 20, 40),
            arc_steps=20,
            save_dir=str(save_dir),
        )
        json_path = save_dir / "results.json"
        assert json_path.exists()
        loaded = json.loads(json_path.read_text())
        assert loaded["campaign"] == "P127W2-N06"


# ---------------------------------------------------------------------------
# Provenance and criteria constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_provenance_dict(self):
        assert "classical_buckling" in PROVENANCE
        assert "smith_method" in PROVENANCE
        assert "csr" in PROVENANCE
        assert "experimental" in PROVENANCE

    def test_criteria_dict(self):
        assert len(CRITERIA) == 6
        assert "a" in CRITERIA
        assert "f" in CRITERIA
