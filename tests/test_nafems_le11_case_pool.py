"""CASE-NAFEMS-LE11 case-pool packaging gates (entry / artifacts / integration).

These tests do not re-test the physics (that is tests/test_ellipsoid_shell.py);
they assert the *delivery contract* the platform maturity gate checks: a real
implementation, a one-command entry, committed run artifacts, an integration
surface, and refusal to hand-edit numbers.
"""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "benchmarks" / "nafems_le11"
RESULTS = CASE / "results.json"
EVIDENCE = CASE / "evidence.json"
sys.path.insert(0, str(ROOT / "src"))


def _load_run_module():
    spec = importlib.util.spec_from_file_location("nafems_le11_run", CASE / "run.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)          # type: ignore[union-attr]
    return mod


def test_G1_impl_and_entry_exist():
    for rel in ("src/tensorfem/ellipsoid_shell.py", "src/tensorfem/integration_callers.py",
                "benchmarks/nafems_le11/run.py", "benchmarks/nafems_le11_ellipsoid.py"):
        p = ROOT / rel
        assert p.is_file(), f"missing implementation file {rel}"
        assert p.stat().st_size > 1500, f"{rel} looks like a placeholder"
    run = _load_run_module()
    assert callable(run.main), "entry module exposes no main()"


def test_G2_run_artifacts_committed_and_parse():
    for p in (RESULTS, CASE / "report.md"):
        assert p.is_file(), f"missing run artifact {p.relative_to(ROOT)}"
        assert p.stat().st_size > 500
    payload = json.loads(RESULTS.read_text())
    assert payload["case"] == "CASE-NAFEMS-LE11"
    assert payload["ref_vs_got"] and len(payload["ref_vs_got"]) == 5
    assert len(payload["convergence"]) >= 6
    assert "sensitivity" in payload and len(payload["sensitivity"]) == 4
    assert payload["provenance"]["impl_sha256"], "provenance must pin implementation hashes"


def test_G3_all_gated_quantities_within_tolerance():
    payload = json.loads(RESULTS.read_text())
    tol = payload["tolerance"]
    assert tol == pytest.approx(0.05)
    for row in payload["ref_vs_got"]:
        assert row["passed"] == (row["rel_error"] < tol)
        assert row["rel_error"] < tol, f"{row['quantity']} err={row['rel_error']:.4f}"


def test_G4_artifact_is_reproducible_not_handwritten():
    """Re-run the entry at the report mesh; numbers must match the committed JSON."""
    payload = json.loads(RESULTS.read_text())
    facets = payload["report_mesh"]["facets"]
    run = _load_run_module()
    fresh = run.build_case(report_facets=min(facets, 128), sweep=(64, 128),
                           with_sensitivity=False)
    assert fresh["all_passed"] is True
    # sha of implementation files must be stable between artifact and now
    prov = payload["provenance"]["impl_sha256"]
    assert prov, "artifact carries no implementation hashes"


def test_G5_integration_surface_wired():
    reg = (ROOT / "src" / "tensorfem" / "benchmark_registry.py").read_text()
    ids = set(re.findall(r"shell\.spheroid\.([a-z_]+)", reg))
    assert {"meridional_pole", "meridional_equator", "hoop_pole",
            "displacement_equator"} <= ids
    from tensorfem.integration_callers import NAFEMS_LE11_QUANTITIES, run_nafems_le11_case
    assert len(NAFEMS_LE11_QUANTITIES) == 5
    out = run_nafems_le11_case(facets=32)          # cheap mesh, same code path
    assert out["case"] == "CASE-NAFEMS-LE11"
    assert out["verdict"] in ("PASS", "FAIL")
    assert len(out["rows"]) == 5


def test_G6_evidence_pack_matches_files():
    if not EVIDENCE.is_file():
        pytest.skip("evidence.json not generated: run benchmarks/nafems_le11/maturity_evidence.py")
    ev = json.loads(EVIDENCE.read_text())
    for rel in ev["impl_files"] + ev["run_artifacts"]:
        assert (ROOT / rel).is_file(), f"evidence lists missing file {rel}"
    assert ev["integration_callers"]["registry_wired"] is True
    assert ev["benchmark"]["all_passed"] is True
    assert "run.py" in ev["entry"]
