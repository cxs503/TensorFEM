"""
TensorFEM Verification Ladder - Pytest Suite
=============================================

This pytest suite runs all verification cases and checks acceptance criteria.

Usage:
    pytest test_verification_ladder.py -v

The suite verifies:
- L1: Beam static analysis (simply supported + cantilever)
- L2: Plate static analysis (clamped square plate)
- L3: Modal analysis (beam + plate natural frequencies)
"""

import json
import pytest
from pathlib import Path


VERIFIED_DIR = Path(__file__).parent


def load_result(case_dir: str) -> dict:
    """Load result.json from a verification case directory."""
    result_file = VERIFIED_DIR / case_dir / "result.json"
    if not result_file.exists():
        pytest.skip(f"Results not found for {case_dir}. Run verification first.")
    return json.loads(result_file.read_text())


def load_gates(case_dir: str) -> dict:
    """Load gates_verified.json from a verification case directory."""
    gates_file = VERIFIED_DIR / case_dir / "gates_verified.json"
    if not gates_file.exists():
        pytest.skip(f"Gates not found for {case_dir}.")
    return json.loads(gates_file.read_text())


class TestL1BeamStatic:
    """L1 Beam Static Verification Tests"""
    
    def test_simply_supported_error(self):
        """Simply supported beam: max section error < 0.5%"""
        result = load_result("fem_beam_static")
        finest = result["simply_supported_levels"][-1]
        max_error = finest["max_section_relative_error"]
        assert max_error < 0.005, f"Max section error {max_error:.6e} exceeds 0.5%"
    
    def test_cantilever_error(self):
        """Cantilever beam: tip deflection error < 0.5%"""
        result = load_result("fem_beam_static")
        finest = result["cantilever_levels"][-1]
        tip_error = finest["tip_deflection_relative_error"]
        assert tip_error < 0.005, f"Tip deflection error {tip_error:.6e} exceeds 0.5%"
    
    def test_beam_convergence_order(self):
        """Beam: convergence order >= 2"""
        result = load_result("fem_beam_static")
        order_l2 = result["convergence"]["field_l2_order"]
        order_sections = result["convergence"]["section_order"]
        min_order = min(order_l2, order_sections)
        assert min_order >= 2.0, f"Convergence order {min_order:.3f} < 2.0"
    
    def test_beam_gates_pass(self):
        """All beam gates pass"""
        gates = load_gates("fem_beam_static")
        assert gates["passed"] is True


class TestL2PlateStatic:
    """L2 Plate Static Verification Tests"""
    
    def test_ssss_centre_deflection_error(self):
        """SSSS plate: centre deflection error vs Navier < 1%"""
        result = load_result("fem_plate_static")
        finest = result["ssss_levels"][-1]
        error = finest["error_vs_navier_301"]
        assert error < 0.01, f"SSSS centre deflection error {error:.6e} exceeds 1%"
    
    def test_plate_convergence_order(self):
        """Plate: convergence order >= 1.5"""
        result = load_result("fem_plate_static")
        order = result["convergence"]["order_ssss_deflection"]
        assert order >= 1.5, f"Convergence order {order:.3f} < 1.5"
    
    def test_plate_gates_pass(self):
        """All plate gates pass"""
        gates = load_gates("fem_plate_static")
        assert gates["passed"] is True


class TestL3Modal:
    """L3 Modal Analysis Verification Tests"""
    
    def _load_modal_result(self):
        """Load the modal result file."""
        result_path = VERIFIED_DIR / "fem_modal" / "result_p19.json"
        if not result_path.exists():
            pytest.skip("result_p19.json not found. Run verification first.")
        return json.loads(result_path.read_text())
    
    def test_beam_modal_error(self):
        """Beam modal: max frequency error < 3%"""
        modal_result = self._load_modal_result()
        finest = modal_result["beam_levels"][-1]
        max_error = finest["max_relative_error"]
        assert max_error < 0.03, f"Max beam frequency error {max_error:.6e} exceeds 3%"
    
    def test_plate_modal_error(self):
        """Plate modal: max frequency error < 3%"""
        modal_result = self._load_modal_result()
        finest = modal_result["plate_levels"][-1]
        max_error = finest["max_relative_error"]
        assert max_error < 0.03, f"Max plate frequency error {max_error:.6e} exceeds 3%"
    
    def test_modal_convergence_order(self):
        """Modal: convergence order >= 1.5"""
        gates = load_gates("fem_modal")
        beam_order = None
        plate_order = None
        for gate in gates["gates"]:
            if gate["id"] == "L3.beam_order":
                beam_order = gate["value"]
            elif gate["id"] == "L3.plate_order":
                plate_order = gate["value"]
        if beam_order is not None and plate_order is not None:
            min_order = min(beam_order, plate_order)
            assert min_order >= 1.5, f"Convergence order {min_order:.3f} < 1.5"
    
    def test_modal_gates_pass(self):
        """All modal gates pass"""
        gates = load_gates("fem_modal")
        assert gates["passed"] is True


class TestOverallVerification:
    """Overall Verification Ladder Tests"""
    
    def test_all_cases_exist(self):
        """All verification case directories exist"""
        required_dirs = ["fem_beam_static", "fem_plate_static", "fem_modal"]
        for case_dir in required_dirs:
            assert (VERIFIED_DIR / case_dir).exists(), f"Missing directory: {case_dir}"
    
    def test_all_results_exist(self):
        """All result.json files exist"""
        required_dirs = ["fem_beam_static", "fem_plate_static", "fem_modal"]
        for case_dir in required_dirs:
            result_file = VERIFIED_DIR / case_dir / "result.json"
            assert result_file.exists(), f"Missing result.json in {case_dir}"
    
    def test_all_gates_exist(self):
        """All gates_verified.json files exist"""
        required_dirs = ["fem_beam_static", "fem_plate_static", "fem_modal"]
        for case_dir in required_dirs:
            gates_file = VERIFIED_DIR / case_dir / "gates_verified.json"
            assert gates_file.exists(), f"Missing gates_verified.json in {case_dir}"
    
    def test_all_verdicts_pass(self):
        """All verification cases pass"""
        required_dirs = ["fem_beam_static", "fem_plate_static", "fem_modal"]
        for case_dir in required_dirs:
            gates = load_gates(case_dir)
            assert gates["passed"] is True, f"{case_dir} verdict is {gates['passed']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
