"""Test for Scordelis-Lo cylindrical roof benchmark verification.

This test ensures that the Scordelis-Lo benchmark verification is preserved
and documents the actual performance of both flat facet and cylindrical shell4 elements.

NOTE: The cylindrical shell4 element does NOT achieve <3% accuracy on the Scordelis-Lo
benchmark with the current implementation. The verification shows ~54-78% error.
This is an honest assessment - see benchmarks/verified/fem_shell_scordelis_lo/ for
the full verification with accuracy gates.
"""
import json
import pytest
import torch
from pathlib import Path

from tensorfem.shell_benchmarks import scordelis_lo_roof
from tensorfem.cylindrical_shell_benchmarks import scordelis_lo_cylindrical


PUBLISHED_REFERENCE = -0.3024


def test_flat_facet_shell4_fails_accuracy_gate():
    """Flat facet Shell4 should fail the 3% accuracy gate.
    
    This is a known limitation: flat facet approximation cannot capture
    membrane-bending coupling in curved shells.
    """
    result = scordelis_lo_roof(8, 8)
    error = abs(result.probe_displacement / PUBLISHED_REFERENCE - 1.0)
    
    # This test documents the known limitation
    assert error > 0.03, (
        f"Flat facet Shell4 error {error*100:.2f}% should exceed 3% "
        "due to flat facet approximation limitation"
    )


def test_cylindrical_shell4_documents_actual_performance():
    """Cylindrical shell4 performance on Scordelis-Lo is documented honestly.
    
    The cylindrical shell4 element shows monotone convergence but is pre-asymptotic
    at 32x32 (~54% error). The observed order is negative (p ≈ -1.18), indicating
    the ladder is not in the asymptotic range.
    
    Full verification with accuracy gates is in benchmarks/verified/fem_shell_scordelis_lo/.
    """
    result = scordelis_lo_cylindrical(8, 8)
    error = abs(result.probe_displacement / PUBLISHED_REFERENCE - 1.0)
    
    # Document actual performance: ~89% error at 8x8
    assert 0.80 < error < 0.95, (
        f"Cylindrical shell4 error at 8x8 should be ~89%, got {error*100:.2f}%"
    )


def test_cylindrical_shell4_convergence_is_monotone():
    """Cylindrical shell4 shows monotone convergence on Scordelis-Lo."""
    results = [scordelis_lo_cylindrical(n, n) for n in (4, 8, 16, 32)]
    displacements = [r.probe_displacement for r in results]
    
    # Monotone convergence (displacement magnitude increases with refinement)
    assert abs(displacements[0]) < abs(displacements[1]) < abs(displacements[2]) < abs(displacements[3])
    
    # But still far from reference (~54% error at 32x32)
    error_32 = abs(displacements[3] / PUBLISHED_REFERENCE - 1.0)
    assert 0.50 < error_32 < 0.60, f"Expected ~54% error at 32x32, got {error_32*100:.2f}%"


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
