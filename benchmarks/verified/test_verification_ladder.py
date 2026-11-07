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
        """Simply supported beam: max error < 1%"""
        result = load_result("fem_beam_static")
        ss_case = result["cases"]["simply_supported_uniform"]
        max_error = ss_case["max_error_at_finest"]
        assert max_error < 0.01, f"Max error {max_error:.6e} exceeds 1%"
    
    def test_cantilever_error(self):
        """Cantilever beam: tip error < 1%"""
        result = load_result("fem_beam_static")
        cant_case = result["cases"]["cantilever_point_load"]
        max_error = cant_case["max_error_at_finest"]
        assert max_error < 0.01, f"Max error {max_error:.6e} exceeds 1%"
    
    def test_beam_convergence_order(self):
        """Beam: convergence order ≥ 2"""
        result = load_result("fem_beam_static")
        ss_order = result["cases"]["simply_supported_uniform"]["observed_convergence_order"]
        cant_order = result["cases"]["cantilever_point_load"]["observed_convergence_order"]
        min_order = min(ss_order, cant_order)
        assert min_order >= 2.0, f"Convergence order {min_order:.3f} < 2.0"
    
    def test_beam_gates_pass(self):
        """All beam gates pass"""
        gates = load_gates("fem_beam_static")
        assert gates["overall_verdict"] == "PASS"


class TestL2PlateStatic:
    """L2 Plate Static Verification Tests"""
    
    def test_clamped_plate_error(self):
        """Clamped plate: center deflection error < 2%"""
        result = load_result("fem_plate_static")
        plate_case = result["case"]
        max_error = plate_case["max_error_at_finest"]
        assert max_error < 0.02, f"Max error {max_error:.6e} exceeds 2%"
    
    def test_plate_convergence_order(self):
        """Plate: convergence order ≥ 1.5"""
        result = load_result("fem_plate_static")
        plate_case = result["case"]
        order = plate_case["observed_convergence_order"]
        assert order >= 1.5, f"Convergence order {order:.3f} < 1.5"
    
    def test_plate_gates_pass(self):
        """All plate gates pass"""
        gates = load_gates("fem_plate_static")
        assert gates["overall_verdict"] == "PASS"


class TestL3Modal:
    """L3 Modal Analysis Verification Tests"""
    
    def test_beam_modal_error(self):
        """Beam modal: max frequency error < 3%"""
        result = load_result("fem_modal")
        beam_case = result["cases"]["simply_supported_beam_modal"]
        max_error = beam_case["max_error_at_finest"]
        assert max_error < 0.03, f"Max error {max_error:.6e} exceeds 3%"
    
    def test_plate_modal_error(self):
        """Plate modal: max frequency error < 3%"""
        result = load_result("fem_modal")
        plate_case = result["cases"]["rectangular_plate_modal"]
        max_error = plate_case["max_error_at_finest"]
        assert max_error < 0.03, f"Max error {max_error:.6e} exceeds 3%"
    
    def test_modal_convergence_order(self):
        """Modal: convergence order ≥ 1.5"""
        result = load_result("fem_modal")
        beam_order = result["cases"]["simply_supported_beam_modal"]["observed_convergence_order"]
        plate_order = result["cases"]["rectangular_plate_modal"]["observed_convergence_order"]
        min_order = min(beam_order, plate_order)
        assert min_order >= 1.5, f"Convergence order {min_order:.3f} < 1.5"
    
    def test_modal_gates_pass(self):
        """All modal gates pass"""
        gates = load_gates("fem_modal")
        assert gates["overall_verdict"] == "PASS"


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
            assert gates["overall_verdict"] == "PASS", f"{case_dir} verdict is {gates['overall_verdict']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
