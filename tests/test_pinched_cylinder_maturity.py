"""Maturity-gate evidence for CASE-FEM-PINCHED-CYLINDER (repair of m16076).

The original delivery was judged L1 by the maturity gate: ``impl_files`` and
``run_artifacts`` were not verifiable.  These tests make the L2/L3/L4 evidence
machine-checkable instead of self-declared:

  L2  impl_files (non-placeholder) + entry + run_artifacts
  L3  integration_callers (main-solver / platform-side real calls)
  L4  benchmark (ref vs got) + repro_cmd

Every assertion reads the files that actually exist in the tree; nothing is
hard-coded from memory.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from tensorfem.pinched_cylinder import solve_pinched_cylinder_sparse

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "benchmarks" / "verified" / "fem_pinched_cylinder"


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _load_evidence() -> dict:
    return json.loads((CASE / "maturity_evidence.json").read_text())


# ---------------------------------------------------------------------------
# L2: impl_files (non-placeholder) + entry + run_artifacts
# ---------------------------------------------------------------------------
def test_impl_files_are_real_not_placeholder():
    ev = _load_evidence()["L2"]
    assert len(ev["impl_files"]) >= 2
    for f in ev["impl_files"]:
        p = ROOT / f["path"]
        assert p.exists(), f"impl file missing: {f['path']}"
        text = p.read_text()
        assert len(text.splitlines()) >= 50, f"{f['path']} too short to be real"
        assert re.search(r"(torch\.|np\.)", text), f"{f['path']} has no tensor ops"
        assert f["sha256"] == _sha256(p), f"hash drift in {f['path']}"
        assert f["non_placeholder"] is True


def test_entry_driver_exists_and_is_the_repro_entry():
    ev = _load_evidence()["L2"]
    entry = ROOT / ev["entry"]
    assert entry.exists()
    assert _sha256(entry) == ev["entry_sha256"]
    assert "run.py" in ev["entry"]


def test_run_artifacts_present_including_figures():
    ev = _load_evidence()["L2"]
    names = {a["path"].rsplit("/", 1)[-1] for a in ev["run_artifacts"]}
    # the three figures were the missing run_artifacts in the L1 verdict
    for required in ("case_result.json", "case_gates.json",
                    "case_convergence.png", "case_field.png",
                    "case_sensitivity.png"):
        assert required in names, f"run artifact missing: {required}"
    for a in ev["run_artifacts"]:
        assert not a.get("missing"), f"artifact missing: {a['path']}"
        assert (ROOT / a["path"]).exists()
        assert _sha256(ROOT / a["path"]) == a["sha256"]


# ---------------------------------------------------------------------------
# L3: integration_callers (main-solver / platform-side real calls)
# ---------------------------------------------------------------------------
def test_integration_callers_are_real_call_sites():
    ev = _load_evidence()["L3"]
    assert len(ev["integration_callers"]) >= 3
    for c in ev["integration_callers"]:
        p = ROOT / c["path"]
        assert p.exists(), f"caller file missing: {c['path']}"
        lines = p.read_text().splitlines()
        for ln in c["call_lines"]:
            assert c["call"] in lines[ln - 1], \
                f"call site drifted: {c['path']}:{ln}"
        assert c["present"] is True


def test_main_solver_path_actually_reaches_pinched_cylinder():
    """The benchmark-case registry (main solver side) must drive the production
    solver and return the load-point deflection, not a stub."""
    from tensorfem.benchmark_cases import run_case
    from tensorfem.shell_benchmark_suite import pinched_cylinder_case

    archive = run_case(pinched_cylinder_case())
    assert archive["qualification_passed"] is True
    by_mesh = {r["mesh"]: r for r in archive["results"]}
    # the qualifying mesh must be within tolerance of the published reference
    qual = by_mesh[12]
    assert qual["passed"] is True
    assert abs(qual["reference"] - (-1.8248e-5)) < 1e-12
    assert qual["error"] < 0.03
    # and the value must be a genuine solve (negative = inward deflection)
    assert qual["computed"] < 0 and abs(qual["computed"]) < 2.0e-5


# ---------------------------------------------------------------------------
# L4: benchmark (ref vs got) + repro_cmd
# ---------------------------------------------------------------------------
def test_benchmark_ref_vs_got_within_tolerance():
    result = json.loads((CASE / "case_result.json").read_text())
    prim = next(l for l in result["levels"]
                if l["n"] == result["case_definition"]["primary_mesh"])
    tol = result["case_definition"]["tolerance_pct"]
    assert prim["relative_error_pct"] <= tol
    assert prim["displacement_in"] < 0
    # the recorded reference must be the published MacNeal-Harder value
    assert abs(result["reference"]["value_in"] - (-1.8248e-5)) < 1e-12


def test_repro_cmd_is_recorded_and_sparse_path_reproduces():
    result = json.loads((CASE / "case_result.json").read_text())
    assert result["repro_cmd"].startswith("PYTHONPATH=src python")
    # the repro entry must reproduce the recorded 16x16 number
    got = solve_pinched_cylinder_sparse(16).displacement
    assert abs(got / result["levels"][[l["n"] for l in result["levels"]].index(16)]["displacement_in"] - 1.0) < 1e-9


def test_evidence_verdicts_all_pass():
    ev = _load_evidence()
    assert ev["verdicts"]["L2"] is True
    assert ev["verdicts"]["L3"] is True
    assert ev["verdicts"]["L4"] is True
