"""Integration tests for Cook membrane cross-validation (XVAL-FEM-COOK)."""

import json
import os
import pytest
from pathlib import Path


@pytest.fixture
def metrics_path():
    """Path to the Cook xval metrics file."""
    here = Path(__file__).parent.parent
    return here / "artifacts" / "cook_xval" / "metrics.json"


@pytest.fixture
def evidence_path():
    """Path to the Cook xval evidence file."""
    here = Path(__file__).parent.parent
    return here / "benchmarks" / "cook_xval_evidence.json"


class TestCookXvalEntry:
    """Tests for the cook_xval entry point module."""

    def test_import(self):
        """Test that the entry point can be imported."""
        from tensorfem.cook_xval import (
            CookXvalResult,
            run_cook_xval,
            load_cook_xval_results,
            generate_convergence_plot,
            generate_error_plot,
        )

    def test_import_from_package(self):
        """Test that the entry point can be imported from the package."""
        from tensorfem import (
            CookXvalResult,
            run_cook_xval,
            load_cook_xval_results,
            generate_convergence_plot,
            generate_error_plot,
        )


class TestCookXvalResults:
    """Tests for Cook xval result loading and validation."""

    def test_load_metrics(self, metrics_path):
        """Test that metrics.json can be loaded."""
        from tensorfem.cook_xval import load_cook_xval_results

        assert metrics_path.exists(), f"Metrics file not found: {metrics_path}"
        result = load_cook_xval_results(str(metrics_path))

        assert result.spec is not None
        assert result.levels == [8, 16]
        assert "tensorfem_q4" in result.parties
        assert "skfem_quad2" in result.parties
        assert "skfem_tri2" in result.parties

    def test_uy_mid_errors(self, metrics_path):
        """Test that uy_mid errors are computed correctly."""
        from tensorfem.cook_xval import load_cook_xval_results

        result = load_cook_xval_results(str(metrics_path))
        errors = result.uy_mid_errors

        # TensorFEM Q4 n=16 should be < 3%
        assert errors["tensorfem_q4"]["16"] < 3.0, \
            f"TensorFEM Q4 n=16 error {errors['tensorfem_q4']['16']:.2f}% >= 3%"

        # scikit-fem should be < 1%
        assert errors["skfem_quad2"]["16"] < 1.0
        assert errors["skfem_tri2"]["16"] < 1.0

    def test_reference_value(self, metrics_path):
        """Test that the published reference is correct."""
        from tensorfem.cook_xval import load_cook_xval_results

        result = load_cook_xval_results(str(metrics_path))
        assert result.spec["published_reference_uy_mid"] == 23.96

    def test_tensorfem_q4_n16_passes(self, metrics_path):
        """Test that TensorFEM Q4 at n=16 passes the 3% gate."""
        from tensorfem.cook_xval import load_cook_xval_results

        result = load_cook_xval_results(str(metrics_path))
        uy_mid = result.results["tensorfem_q4"]["16"]["uy_mid"]
        ref = result.spec["published_reference_uy_mid"]
        error = abs(uy_mid - ref) / ref * 100

        assert error < 3.0, f"TensorFEM Q4 n=16 error {error:.2f}% >= 3%"


class TestCookXvalEvidence:
    """Tests for the Cook xval evidence file."""

    def test_evidence_file_exists(self, evidence_path):
        """Test that the evidence file exists."""
        assert evidence_path.exists(), f"Evidence file not found: {evidence_path}"

    def test_evidence_structure(self, evidence_path):
        """Test that the evidence file has the correct structure."""
        with open(evidence_path) as f:
            evidence = json.load(f)

        assert "capability" in evidence
        assert evidence["capability"] == "XVAL-FEM-COOK"
        assert "evidence" in evidence
        assert "artifacts" in evidence
        assert "repro_cmd" in evidence
        assert "maturity" in evidence

    def test_maturity_levels(self, evidence_path):
        """Test that maturity levels are documented."""
        with open(evidence_path) as f:
            evidence = json.load(f)

        maturity = evidence["maturity"]
        assert "L2" in maturity
        assert "L3" in maturity
        assert "L4" in maturity

        # Check L2 requirements
        assert "impl_files" in maturity["L2"]
        assert "entry" in maturity["L2"]
        assert "run_artifacts" in maturity["L2"]

        # Check L3 requirements
        assert "integration_callers" in maturity["L3"]

        # Check L4 requirements
        assert "benchmark" in maturity["L4"]
        assert "repro_cmd" in maturity["L4"]


class TestCookXvalArtifacts:
    """Tests for Cook xval artifacts."""

    def test_metrics_json_exists(self, metrics_path):
        """Test that metrics.json exists."""
        assert metrics_path.exists()

    def test_convergence_plot_exists(self, metrics_path):
        """Test that convergence plot exists."""
        plot_path = metrics_path.parent / "convergence.png"
        assert plot_path.exists(), f"Convergence plot not found: {plot_path}"

    def test_error_plot_exists(self, metrics_path):
        """Test that error plot exists."""
        plot_path = metrics_path.parent / "error.png"
        assert plot_path.exists(), f"Error plot not found: {plot_path}"

    def test_benchmark_script_exists(self):
        """Test that the benchmark script exists."""
        here = Path(__file__).parent.parent
        script = here / "benchmarks" / "cook_membrane_xval.py"
        assert script.exists(), f"Benchmark script not found: {script}"

    def test_entry_point_exists(self):
        """Test that the entry point module exists."""
        here = Path(__file__).parent.parent
        entry = here / "src" / "tensorfem" / "cook_xval.py"
        assert entry.exists(), f"Entry point not found: {entry}"

    def test_documentation_exists(self):
        """Test that the documentation exists."""
        here = Path(__file__).parent.parent
        doc = here / "docs" / "cook-xval-report.md"
        assert doc.exists(), f"Documentation not found: {doc}"
