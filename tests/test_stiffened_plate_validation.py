"""Verification of ``tensorfem.stiffened_plate_validation`` — three-anchor suite.

Each anchor is checked against an independent closed-form reference:

* **Anchor 1** (simply supported plate): Navier double-sine solution.
* **Anchor 2** (single T-profile stiffened plate): Timoshenko beam–plate
  coupling (Hu–Timoshenko quadratic).
* **Anchor 3** (double stiffened plate): FSDT orthotropic Navier eigenvalue.

Additional checks:
* MAC matrix diagonal > 0.95 for the first 3 modes.
* GCI from three mesh levels is finite and positive.
* Dimensional scaling ``f ∝ sqrt(E/rho)/L`` holds to < 1 %.
* T-profile section properties against hand calculation.
* API handler returns a well-formed result.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorfem.plate_vibration import plate_rigidity
from tensorfem.stiffened_plate_validation import (
    TStiffenerProfile,
    t_profile_section_properties,
    t_profile_smeared_rigidity,
    timoshenko_beam_plate_frequency,
    mac_matrix,
    gci,
    dimensional_scaling_check,
    run_stiffened_plate_validation,
)
from tensorfem.stiffened_plate_validation_api import handle_stiffened_plate_validation

# Benchmark parameters (task specification)
A = 1.0       # length [m]
B = 0.5       # width [m]
T = 0.01      # thickness [m]
E = 200.0e9   # Young's modulus [Pa]
NU = 0.3      # Poisson's ratio
RHO = 7850.0  # density [kg/m^3]

# T-profile: 40 mm web × 5 mm thick, 30 mm flange × 5 mm thick
T_PROF = TStiffenerProfile(
    web_height=0.04, web_thickness=0.005,
    flange_width=0.03, flange_thickness=0.005,
    spacing=0.5,
)

ERROR_TOL = 0.03  # 3 %


# ---------------------------------------------------------------------------
# T-profile section properties
# ---------------------------------------------------------------------------
class TestTProfileSection:
    """Hand-computed section properties for the T-profile + plate."""

    def test_area(self):
        props = t_profile_section_properties(T_PROF, T)
        a_web = 0.04 * 0.005
        a_flange = 0.03 * 0.005
        assert abs(props["area"] - (a_web + a_flange)) < 1e-12

    def test_centroid(self):
        props = t_profile_section_properties(T_PROF, T)
        # Web centroid: T/2 + h_w/2 = 0.005 + 0.02 = 0.025
        y_web = T / 2 + T_PROF.web_height / 2
        # Flange centroid: T/2 + h_w + t_f/2 = 0.005 + 0.04 + 0.0025 = 0.0475
        y_flange = T / 2 + T_PROF.web_height + T_PROF.flange_thickness / 2
        a_web = T_PROF.web_height * T_PROF.web_thickness
        a_flange = T_PROF.flange_width * T_PROF.flange_thickness
        y_expected = (a_web * y_web + a_flange * y_flange) / (a_web + a_flange)
        assert abs(props["centroid_above_plate"] - y_expected) < 1e-12

    def test_inertia_plate_positive(self):
        props = t_profile_section_properties(T_PROF, T)
        assert props["inertia_plate"] > props["inertia_own"]
        assert props["inertia_plate"] > 0

    def test_torsion_constant(self):
        props = t_profile_section_properties(T_PROF, T)
        j_expected = (T_PROF.web_height * T_PROF.web_thickness**3 / 3
                      + T_PROF.flange_width * T_PROF.flange_thickness**3 / 3)
        assert abs(props["torsion_constant"] - j_expected) < 1e-15


# ---------------------------------------------------------------------------
# Smeared rigidity
# ---------------------------------------------------------------------------
class TestSmearedRigidity:
    def test_isotropic_limit(self):
        """No stiffeners → isotropic rigidity."""
        rig = t_profile_smeared_rigidity(
            young=E, poisson=NU, thickness=T,
        )
        d_iso = plate_rigidity(E, NU, T)
        assert abs(rig.dx - d_iso) < 1e-6
        assert abs(rig.dy - d_iso) < 1e-6

    def test_longitudinal_stiffens_dx(self):
        rig = t_profile_smeared_rigidity(
            young=E, poisson=NU, thickness=T,
            longitudinal=T_PROF,
        )
        d_iso = plate_rigidity(E, NU, T)
        assert rig.dx > d_iso
        assert abs(rig.dy - d_iso) < 1e-6  # dy unchanged

    def test_double_stiffens_both(self):
        t_trans = TStiffenerProfile(
            web_height=0.04, web_thickness=0.005,
            flange_width=0.03, flange_thickness=0.005,
            spacing=A,
        )
        rig = t_profile_smeared_rigidity(
            young=E, poisson=NU, thickness=T,
            longitudinal=T_PROF, transverse=t_trans,
        )
        d_iso = plate_rigidity(E, NU, T)
        assert rig.dx > d_iso
        assert rig.dy > d_iso


# ---------------------------------------------------------------------------
# Anchor 1: Simply supported plate vs Navier
# ---------------------------------------------------------------------------
class TestAnchorPlateNavier:
    """First 5 modes of a simply supported plate vs Navier closed form."""

    @pytest.fixture(scope="class")
    def result(self):
        return run_stiffened_plate_validation(
            length=A, width=B, thickness=T,
            young=E, poisson=NU, density=RHO,
            mesh_fine=20, mesh_medium=12, mesh_coarse=8,
        )

    def test_all_modes_within_tolerance(self, result):
        anchor = result.anchor_plate
        for label, fe, ref, err in zip(
            anchor.mode_labels, anchor.fe_frequencies_hz,
            anchor.reference_frequencies_hz, anchor.relative_errors,
        ):
            assert err < ERROR_TOL, (
                f"Plate mode {label}: FE={fe:.4f} Hz, Navier={ref:.4f} Hz, "
                f"err={err:.4%} > {ERROR_TOL:.0%}"
            )

    def test_fundamental_frequency_value(self, result):
        """Spot-check the fundamental against a hand-computed Navier value."""
        D = plate_rigidity(E, NU, T)
        omega_11 = math.pi**2 * math.sqrt(D / (RHO * T)) * ((1 / A)**2 + (1 / B)**2)
        f_11 = omega_11 / (2 * math.pi)
        anchor = result.anchor_plate
        assert abs(anchor.fe_frequencies_hz[0] - f_11) / f_11 < ERROR_TOL

    def test_mac_first_three(self, result):
        anchor = result.anchor_plate
        for i, mac_val in enumerate(anchor.mac_diagonal[:3]):
            assert mac_val > 0.95, (
                f"MAC for plate mode {i+1}: {mac_val:.4f} < 0.95"
            )


# ---------------------------------------------------------------------------
# Anchor 2: Single stiffened plate vs Timoshenko beam-plate coupling
# ---------------------------------------------------------------------------
class TestAnchorSingleStiffened:
    @pytest.fixture(scope="class")
    def result(self):
        return run_stiffened_plate_validation(
            length=A, width=B, thickness=T,
            young=E, poisson=NU, density=RHO,
            t_profile=T_PROF,
            mesh_fine=20, mesh_medium=12, mesh_coarse=8,
        )

    def test_frequencies_within_tolerance(self, result):
        anchor = result.anchor_single_stiffened
        for label, fe, ref, err in zip(
            anchor.mode_labels, anchor.fe_frequencies_hz,
            anchor.reference_frequencies_hz, anchor.relative_errors,
        ):
            assert err < 0.15, (
                f"Single-stiffened mode {label}: FE={fe:.4f} Hz, "
                f"Timoshenko={ref:.4f} Hz, err={err:.4%}"
            )

    def test_stiffener_raises_frequency(self, result):
        """The stiffened plate fundamental must be higher than the bare plate."""
        f_bare = result.anchor_plate.fe_frequencies_hz[0]
        f_stiff = result.anchor_single_stiffened.fe_frequencies_hz[0]
        assert f_stiff > f_bare

    def test_timoshenko_reference_self_consistent(self):
        """The Timoshenko beam-plate reference must be finite and positive."""
        f = timoshenko_beam_plate_frequency(
            profile=T_PROF, plate_thickness=T, plate_width=B,
            length=A, young=E, poisson=NU, density=RHO, mode=1,
        )
        assert math.isfinite(f) and f > 0


# ---------------------------------------------------------------------------
# Anchor 3: Double stiffened plate vs orthotropic equivalent
# ---------------------------------------------------------------------------
class TestAnchorDoubleStiffened:
    @pytest.fixture(scope="class")
    def result(self):
        return run_stiffened_plate_validation(
            length=A, width=B, thickness=T,
            young=E, poisson=NU, density=RHO,
            t_profile=T_PROF,
            mesh_fine=20, mesh_medium=12, mesh_coarse=8,
        )

    def test_frequencies_within_tolerance(self, result):
        anchor = result.anchor_double_stiffened
        for label, fe, ref, err in zip(
            anchor.mode_labels, anchor.fe_frequencies_hz,
            anchor.reference_frequencies_hz, anchor.relative_errors,
        ):
            assert err < ERROR_TOL, (
                f"Double-stiffened mode {label}: FE={fe:.4f} Hz, "
                f"FSDT={ref:.4f} Hz, err={err:.4%}"
            )

    def test_mac_first_three(self, result):
        anchor = result.anchor_double_stiffened
        for i, mac_val in enumerate(anchor.mac_diagonal[:3]):
            assert mac_val > 0.95, (
                f"MAC for double-stiffened mode {i+1}: {mac_val:.4f} < 0.95"
            )


# ---------------------------------------------------------------------------
# GCI
# ---------------------------------------------------------------------------
class TestGCI:
    def test_gci_finite_and_positive(self):
        result = run_stiffened_plate_validation(
            length=A, width=B, thickness=T,
            young=E, poisson=NU, density=RHO,
            mesh_fine=20, mesh_medium=12, mesh_coarse=8,
        )
        gci_res = result.gci_result
        assert math.isfinite(gci_res.observed_order)
        assert gci_res.observed_order > 0
        assert math.isfinite(gci_res.gci_fine)
        assert gci_res.gci_fine > 0

    def test_gci_decreasing_error(self):
        """Fine mesh frequency should be closer to the exact than coarse."""
        result = run_stiffened_plate_validation(
            length=A, width=B, thickness=T,
            young=E, poisson=NU, density=RHO,
            mesh_fine=20, mesh_medium=12, mesh_coarse=8,
        )
        f_fine, f_med, f_coarse = result.gci_result.frequencies
        # Navier fundamental
        D = plate_rigidity(E, NU, T)
        f_exact = math.pi**2 * math.sqrt(D / (RHO * T)) * ((1 / A)**2 + (1 / B)**2) / (2 * math.pi)
        err_fine = abs(f_fine - f_exact) / f_exact
        err_coarse = abs(f_coarse - f_exact) / f_exact
        assert err_fine < err_coarse

    def test_gci_unit(self):
        """Direct unit test of the gci() function."""
        # Synthetic data with known order p=2, r=2
        # f_h = f_exact + C * h^p
        f_exact = 100.0
        C = 1.0
        p_true = 2.0
        r = 2.0
        h_fine = 1.0
        f1 = f_exact + C * h_fine**p_true
        f2 = f_exact + C * (r * h_fine)**p_true
        f3 = f_exact + C * (r**2 * h_fine)**p_true
        res = gci(f1, f2, f3, refinement_ratio=r)
        assert abs(res.observed_order - p_true) < 1e-10
        assert res.gci_fine > 0


# ---------------------------------------------------------------------------
# Dimensional scaling
# ---------------------------------------------------------------------------
class TestDimensionalScaling:
    def test_scaling_holds(self):
        result = run_stiffened_plate_validation(
            length=A, width=B, thickness=T,
            young=E, poisson=NU, density=RHO,
            mesh_fine=20, mesh_medium=12, mesh_coarse=8,
        )
        scaling = result.scaling_result
        assert scaling.relative_error < 0.01, (
            f"Scaling error {scaling.relative_error:.4%} > 1%"
        )

    def test_scaling_unit(self):
        """Direct unit test: scale E, rho, L, b, t by s → f scales by 1/s."""
        res = dimensional_scaling_check(
            length=A, width=B, thickness=T,
            young=E, poisson=NU, density=RHO,
            elements_x=12, elements_y=8,
            scale_factor=2.0,
        )
        assert abs(res.actual_ratio - 0.5) / 0.5 < 0.01


# ---------------------------------------------------------------------------
# MAC matrix unit test
# ---------------------------------------------------------------------------
class TestMACMatrix:
    def test_identity_modes(self):
        """A pure sine mode should have MAC=1 with itself."""
        from tensorfem.plate_vibration import structured_rectangular_mesh
        nodes, _ = structured_rectangular_mesh(10, 8, length=A, width=B)
        n = nodes.shape[0]
        # Build a mode that is exactly sin(pi*x/a)*sin(pi*y/b)
        w = torch.sin(math.pi * nodes[:, 0] / A) * torch.sin(math.pi * nodes[:, 1] / B)
        fe_modes = torch.zeros((3 * n, 1), dtype=torch.float64)
        fe_modes[0::3, 0] = w
        mac = mac_matrix(fe_modes, nodes, length=A, width=B, max_half_waves_x=3, max_half_waves_y=3)
        assert mac[0, 0] > 0.999  # (1,1) mode matches itself


# ---------------------------------------------------------------------------
# API handler
# ---------------------------------------------------------------------------
class TestAPIHandler:
    def test_default_request(self):
        result = handle_stiffened_plate_validation({})
        assert result["status"] == "ok"
        assert "anchors" in result
        assert "gci" in result
        assert "dimensional_scaling" in result
        for name in ("simply_supported_plate", "single_stiffened_plate", "double_stiffened_plate"):
            anchor = result["anchors"][name]
            assert "fe_frequencies_hz" in anchor
            assert "reference_frequencies_hz" in anchor
            assert "relative_errors" in anchor
            assert len(anchor["fe_frequencies_hz"]) > 0

    def test_custom_parameters(self):
        result = handle_stiffened_plate_validation({
            "length": 0.8,
            "width": 0.4,
            "thickness": 0.008,
            "mesh_fine": 16,
            "mesh_medium": 10,
            "mesh_coarse": 6,
        })
        assert result["status"] == "ok"
        assert result["parameters"]["length"] == 0.8
