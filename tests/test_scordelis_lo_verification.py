"""Test for Scordelis-Lo cylindrical roof benchmark verification.

This test ensures that the Scordelis-Lo benchmark verification is preserved
and that the cylindrical shell4 element continues to pass the accuracy gate.
"""
import json
import pytest
import torch
from pathlib import Path

from tensorfem.shell_benchmarks import scordelis_lo_roof
from tensorfem.cylindrical_shell_benchmarks import scordelis_lo_cylindrical


PUBLISHED_REFERENCE = -0.3024
ACCURACY_THRESHOLD = 0.03  # 3%


def test_flat_facet_shell4_fails_accuracy_gate():
    """Flat facet Shell4 should fail the 3% accuracy gate.
    
    This is a known limitation: flat facet approximation cannot capture
    membrane-bending coupling in curved shells.
    """
    result = scordelis_lo_roof(8, 8)
    error = abs(result.probe_displacement / PUBLISHED_REFERENCE - 1.0)
    
    # This test documents the known limitation
    assert error > ACCURACY_THRESHOLD, (
        f"Flat facet Shell4 error {error*100:.2f}% should exceed {ACCURACY_THRESHOLD*100}% "
        "due to flat facet approximation limitation"
    )


def test_cylindrical_shell4_passes_accuracy_gate():
    """Cylindrical shell4 should pass the 3% accuracy gate.
    
    The specialized cylindrical element properly captures membrane-bending
    coupling in curved shells.
    """
    result = scordelis_lo_cylindrical(8, 8)
    error = abs(result.probe_displacement / PUBLISHED_REFERENCE - 1.0)
    
    assert error <= ACCURACY_THRESHOLD, (
        f"Cylindrical shell4 error {error*100:.2f}% should be within "
        f"{ACCURACY_THRESHOLD*100}% threshold"
    )


def test_cylindrical_shell4_high_accuracy():
    """Cylindrical shell4 should achieve high accuracy (<1%) on fine mesh."""
    result = scordelis_lo_cylindrical(16, 16)
    error = abs(result.probe_displacement / PUBLISHED_REFERENCE - 1.0)
    
    assert error <= 0.01, (
        f"Cylindrical shell4 should achieve <1% error on 16x16 mesh, got {error*100:.2f}%"
    )


def test_verification_artifacts_exist():
    """Verify that the verification artifacts exist and are valid."""
    artifact_dir = Path(__file__).parent.parent / "benchmarks" / "verified" / "fem_shell_scordelis_lo"
    
    # Check that result.json exists and is valid
    result_file = artifact_dir / "result.json"
    assert result_file.exists(), "result.json should exist"
    
    with open(result_file) as f:
        result = json.load(f)
    
    assert "benchmark" in result
    assert "reference" in result
    assert "verdict" in result
    assert result["benchmark"] == "Scordelis-Lo cylindrical roof"
    assert result["reference"]["value"] == PUBLISHED_REFERENCE
    
    # Check that gates_verified.json exists and is valid
    gates_file = artifact_dir / "gates_verified.json"
    assert gates_file.exists(), "gates_verified.json should exist"
    
    with open(gates_file) as f:
        gates = json.load(f)
    
    assert "schema" in gates
    assert "gates" in gates
    assert "overall_verdict" in gates


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
