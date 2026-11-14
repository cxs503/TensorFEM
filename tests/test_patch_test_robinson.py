"""Regression gates for the Robinson patch test A/B/C (``tensorfem.patch_test``).

Each test encodes one clause of the criterion, including the *negative* controls:
a gate that cannot fail is not a gate, so the Q4/bending case and the distorted-Q8
case are asserted to be non-exact, with the measured magnitude pinned.
"""

import json
import math
from pathlib import Path

import pytest
import torch

from tensorfem.patch_test import (
    EXACT_TOL,
    LENGTH,
    YOUNG,
    KAPPA,
    boundary_nodes_2d,
    exact_field,
    numbering_study,
    q4_patch_mesh,
    run_q4_patch,
    run_q8_patch,
    run_tet10_patch,
)


@pytest.fixture(autouse=True)
def _double_precision():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def test_mesh_is_irregular_and_boundary_is_exact():
    nodes, elements = q4_patch_mesh(3, 3, 1.0)
    assert nodes.shape == (16, 2) and elements.shape == (9, 4)
    interior = [5, 6, 9, 10]
    grid_x = torch.tensor([0.0, 1 / 3, 2 / 3, 1.0])
    moved = sum(
        1
        for n in interior
        for k in (0, 1)
        if abs(nodes[n, k] - grid_x[(n % 4) if k == 0 else (n // 4)]) > 1e-6
    )
    assert moved >= 4  # interior nodes are genuinely off-grid
    bnd = boundary_nodes_2d(nodes)
    assert len(bnd) == 12  # 16 nodes - 4 interior


def test_case_A_uniaxial_patch_is_exact():
    r = run_q4_patch("A")
    assert r.disp_error <= EXACT_TOL
    assert r.stress_error <= EXACT_TOL
    assert r.passed


def test_case_B_pure_shear_patch_is_exact():
    r = run_q4_patch("B")
    assert r.disp_error <= EXACT_TOL
    assert r.stress_error <= EXACT_TOL


def test_case_C_bending_is_NOT_reproduced_by_bilinear_Q4():
    """Negative control: the test must be able to fail."""
    r = run_q4_patch("C")
    assert r.stress_error > 1e-2  # measured ~1e-1 relative to E*kappa*L
    assert not r.passed


def test_case_C_is_exact_for_Q8_on_an_affine_mesh():
    r = run_q8_patch("C", 3, 0.0)
    assert r.disp_error <= 1e-11
    assert r.stress_error <= 1e-10
    assert r.passed


def test_case_C_Q8_distorted_mesh_degrades_but_is_bounded():
    """Documented serendipity limitation: not exact, but small and mesh-driven."""
    r = run_q8_patch("C", 3, 1.0)
    assert 1e-4 < r.stress_error < 5e-2
    assert not r.passed  # honest: the patch test is NOT passed
    # the same distorted mesh still reproduces the constant-strain cases exactly
    for case in ("A", "B"):
        a = run_q8_patch(case, 3, 1.0)
        assert a.stress_error <= 1e-10


def test_case_C3_tet10_linear_stress_is_exact_on_distorted_tets():
    r = run_tet10_patch(2, 1.0)
    assert r.disp_error <= 1e-12
    assert r.stress_error <= 1e-10
    assert r.passed


def test_patch_answer_is_numbering_independent():
    out = numbering_study("A")
    assert out["identical"]


def test_exact_targets_are_analytical():
    """The reference values are the closed forms, not tabulated numbers."""
    nodes, _ = q4_patch_mesh(2, 2, 0.0)
    u, sigma_fn, scale, _, _, _ = exact_field("C", nodes)
    # sigma_xx = -E*kappa*y exactly, and the displacement is quadratic
    for p in (torch.tensor([0.3, 0.7]), torch.tensor([1.0, 1.0])):
        assert abs(float(sigma_fn(p)[0]) + YOUNG * KAPPA * float(p[1])) < 1e-10
    assert abs(float(u[0, 0])) < 1e-15
    assert abs(scale - YOUNG * KAPPA * LENGTH) < 1e-9


# --------------------------------------------------------------------------- #
# entry points and integration (maturity gate L2/L3/L4)
# --------------------------------------------------------------------------- #
def test_cli_patch_test_entry_writes_report_and_exits_zero(tmp_path, monkeypatch):
    """``tensorfem patch-test`` is a real entry point, not a module that only tests use."""
    import sys
    from tensorfem import cli

    out = tmp_path / "qualification.json"
    monkeypatch.setattr(sys, "argv", ["tensorfem", "patch-test", "--output", str(out)])
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    assert excinfo.value.code == 0
    payload = json.loads(out.read_text())
    assert payload["schema"] == "tensorfem.patch_test.qualification/1"
    assert payload["summary"] == {"total": 8, "passed": 8}
    assert {c["id"] for c in payload["cases"]} >= {"patch.A.Q4", "patch.C.Q8_distorted"}
    # every must-fail row is reported as *not* exact, i.e. the gate can fail
    must_fail = [c for c in payload["cases"] if c["expectation"] == "must_fail"]
    assert len(must_fail) == 2 and not any(c["exact"] for c in must_fail)


def test_registry_registers_the_patch_cases():
    """The default ``tensorfem verify`` gate carries the three Robinson cases."""
    from tensorfem.benchmark_registry import patch_benchmark_evidence

    rows = {e.id: e for e in patch_benchmark_evidence()}
    assert set(rows) == {"patch.robinson_A", "patch.robinson_B", "patch.robinson_C"}
    for e in rows.values():
        assert e.passed and e.error <= 1e-10
    # A/B references are the imposed amplitudes, C's is the local exact value
    assert rows["patch.robinson_A"].reference == pytest.approx(100.0)
    assert rows["patch.robinson_B"].reference == pytest.approx(50.0)
    assert rows["patch.robinson_C"].reference == pytest.approx(
        rows["patch.robinson_C"].computed, rel=1e-10
    )


def test_registry_gate_is_reachable_from_the_documented_entry():
    """grep-able integration: the registry calls the patch evidence function."""
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[1] / "src" / "tensorfem" / "benchmark_registry.py"
    ).read_text()
    assert "patch_benchmark_evidence()" in src


RUNS = Path(__file__).resolve().parents[1] / "runs" / "patch_test_robinson"


def test_committed_run_artifacts_are_current():
    """The committed run artefacts must be reproducible, not hand-written."""
    root = Path(__file__).resolve().parents[1]
    result = json.loads((root / "runs/patch_test_robinson/result.json").read_text())
    evidence = json.loads((root / "benchmarks/patch_test_robinson_evidence.json").read_text())
    cli_rep = json.loads((root / "runs/patch_test_robinson/cli_qualification.json").read_text())

    live_a = run_q4_patch("A")
    row = next(c for c in result["cases"] if c["case"] == "A" and c["family"] == "Q4")
    assert row["stress_error"] == pytest.approx(live_a.stress_error, abs=1e-15)
    assert row["nodes"] == live_a.nodes and row["cells"] == live_a.cells

    assert evidence["repro_cmd"] == "PYTHONPATH=src python3 benchmarks/patch_test_robinson.py"
    assert evidence["all_expected_behaviour"] is True
    for r in evidence["rows"]:
        assert r["reference_stress"] is not None and math.isfinite(r["reference_stress"])
        assert r["computed_stress"] is not None and math.isfinite(r["computed_stress"])

    assert cli_rep["passed"] is True and cli_rep["summary"]["total"] == 8
    # result.json and the CLI solve the same cases: identical tolerances
    assert result["exact_tolerance"] == cli_rep["tolerance"] == EXACT_TOL


def test_committed_artifacts_cover_every_documented_entry_point():
    """Every artefact the CHANGELOG/doc promise is actually in the tree.

    The predecessor of this case shipped ``runs/patch_test_robinson/*`` in its docs
    and CHANGELOG but not in git, so the reproducibility gate could not run on a
    fresh checkout.  Pinning the file set here makes that gap a test failure.
    """
    promised = {
        "result.json",  # benchmark driver
        "cli_qualification.json",  # tensorfem patch-test
        "cli_qualification.log",  # ... its stdout
        "verify_gate.log",  # tensorfem verify (the registered gate)
    }
    present = {p.name for p in RUNS.glob("*") if p.is_file()}
    assert promised <= present, sorted(promised - present)


def test_committed_verify_gate_log_carries_the_patch_rows():
    """``tensorfem verify`` output is committed, and it agrees with the registry."""
    gate = json.loads((RUNS / "verify_gate.log").read_text())
    assert gate["passed"] is True and gate["summary"]["passed"] == gate["summary"]["total"]
    rows = {r["id"]: r for r in gate["results"] if r["id"].startswith("patch.robinson")}
    assert set(rows) == {"patch.robinson_A", "patch.robinson_B", "patch.robinson_C"}
    for r in rows.values():
        assert r["passed"] and r["error"] <= r["tolerance"] == EXACT_TOL


def test_committed_cli_qualification_reproduces_case_by_case():
    """The committed CLI report is the live report, not a stale or hand-edited file.

    Case-C/Q8-distorted is round-off-sensitive to the reduction order, so only the
    columns that carry a decision (mesh size, free DOFs, expectation, exactness,
    verdict, tolerance) are compared exactly; the error magnitudes are compared to
    the tolerance band the report itself declares.
    """
    from tensorfem.patch_test import qualification_report

    committed = json.loads((RUNS / "cli_qualification.json").read_text())
    live = qualification_report()
    assert committed["schema"] == live["schema"]
    assert committed["tolerance"] == live["tolerance"] == EXACT_TOL
    assert committed["summary"] == live["summary"] == {"total": 8, "passed": 8}
    keys = ("id", "case", "family", "mesh", "cells", "nodes", "free_dofs", "expectation")
    assert [{k: c[k] for k in keys} for c in committed["cases"]] == [
        {k: c[k] for k in keys} for c in live["cases"]
    ]
    for c_live, c_old in zip(live["cases"], committed["cases"]):
        assert c_old["exact"] == c_live["exact"]
        assert c_old["passed"] == c_live["passed"]
        for col in ("disp_error", "stress_error", "energy_error"):
            # same verdict band (round-off vs discretisation error), no decision flip
            assert (c_old[col] <= EXACT_TOL) == (c_live[col] <= EXACT_TOL)
            assert c_old[col] == pytest.approx(c_live[col], rel=1e-6, abs=1e-15)


def test_evidence_provenance_is_checkout_portable():
    """Evidence is committed, so it must not leak the absolute path of one machine."""
    evidence = json.loads(
        (Path(__file__).resolve().parents[1] / "benchmarks/patch_test_robinson_evidence.json")
        .read_text()
    )
    prov = evidence["provenance"]
    assert prov["tensorfem_module_file"] == "src/tensorfem/__init__.py"
    assert not Path(prov["tensorfem_module_file"]).is_absolute()
    assert prov["imported_from_this_checkout"] is True
    assert prov["default_dtype"] == "torch.float64"


def test_free_dof_count_is_reported_positive_for_q8():
    """Regression: the Q8 runner reported -16 free dofs (reference-versus-total mix-up)."""
    for case in ("A", "B", "C"):
        r = run_q8_patch(case, 3, 1.0)
        boundary = 4 * (2 * 3 + 1) - 4  # boundary nodes of the 3x3 Q8 patch
        assert r.free_dofs == 2 * r.nodes - 2 * boundary > 0
