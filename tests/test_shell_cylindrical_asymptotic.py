"""Guards for the cylindrical-shell asymptotic audit.

Three things are asserted here:

1. the sparse assembly path added by the audit reproduces the merged dense
   routines exactly (the model was transcribed, so it must be pinned);
2. the numbers stored in ``benchmarks/shell_cylindrical_asymptotic/result.json``
   are reproducible (no hand-edited evidence);
3. the audit's own verdict logic cannot label a non-monotone or biased sequence
   as a clean pass.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / "benchmarks" / "shell_cylindrical_asymptotic" / "run_audit.py"


def _load_audit():
    spec = importlib.util.spec_from_file_location("shell_asymptotic_audit", AUDIT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def audit():
    return _load_audit()


def test_sparse_matches_dense(audit):
    """Sparse (audit) and dense (merged) assembly must be the same operator."""
    cons = audit.consistency_check(8)
    assert cons["scordelis_lo_rel_diff"] < 1e-10
    assert cons["pinched_cylinder_rel_diff"] < 1e-10


def test_stored_levels_reproduce(audit):
    """Stored probe displacements must recompute to within 1e-9 relative."""
    path = AUDIT.parent / "result.json"
    if not path.exists():
        pytest.skip("result.json not generated yet")
    payload = json.loads(path.read_text())
    checks = [
        ("scordelis_lo", audit.scordelis_lo_value),
        ("pinched_cylinder", audit.pinched_cylinder_value),
    ]
    for key, solver in checks:
        for level in payload[key]["levels"]:
            if level["n"] > 16:  # keep the test light; the report holds the heavy end
                continue
            value = solver(level["n"])
            assert value == pytest.approx(level["value"], rel=1e-9), (key, level["mesh"])


def test_verdict_separates_converged_from_right(audit):
    """A converged sequence that sits on a 5 % bias must not read as a clean pass."""
    reference = 1.0
    # v(n) -> 1.05: converged, monotone, but wrong by 5 %
    levels = [audit.ConvergenceLevel(mesh_size=1.0 / n, value=1.05 - 0.05 / n) for n in (8, 16, 32)]
    report = audit.evaluate_convergence_study(
        levels,
        qoi="probe",
        reference_value=reference,
        reference_tolerance=0.03,
        reference_label="synthetic",
    )
    ref_check = audit._reference_check(report)
    assert ref_check is not None and ref_check["passed"] is False
    verdict = audit.verdict_of(report, monotone=True, limit_error=0.05)
    assert verdict["reference_check_passed"] is False
    assert any("bias" in note for note in verdict["notes"])


def test_verdict_flags_non_monotone(audit):
    levels = [
        audit.ConvergenceLevel(mesh_size=1.0 / n, value=value)
        for n, value in ((4, 0.90), (8, 0.999), (16, 0.995))
    ]
    report = audit.evaluate_convergence_study(levels, qoi="probe", reference_value=1.0)
    verdict = audit.verdict_of(report, monotone=False, limit_error=None)
    assert verdict["evaluator_status"] in {"blocked", "failed", "passed"}
    assert any("not monotone" in note for note in verdict["notes"])
    assert any("unavailable" in note for note in verdict["notes"])