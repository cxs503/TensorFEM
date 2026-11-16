"""Entry, gate and artifact binding for the body-force inversion case.

What this file guards (and what it deliberately does not)
---------------------------------------------------------
``tests/test_body_force_inversion.py`` proves the *numerics*.  This file proves
the parts that a maturity review can only see from outside the module:

1. **entry** -- the case is reachable from the package namespace and from the
   command line (``tensorfem.cli body-force-inversion``), not only from a script
   under ``scripts/``.
2. **gate** -- the CLI exit status is the acceptance decision: 0 only when the
   <=10 %-node / <=5 %-field / <=50-iteration triple holds, non-zero (with the
   JSON still written) when it does not.  A gate that always returns 0 is not a
   gate.
3. **artifacts** -- every number the case doc quotes is read back from
   ``artifacts/body_force_inversion/*.json``; if the artifacts are regenerated
   and a doc number goes stale, this file fails instead of the reader being
   misled.
4. **benchmark rows** -- ``benchmark_rows.json`` (ref vs got, produced by
   ``benchmarks/body_force_inversion_benchmark.py``) reproduces the summary
   scalars exactly and keeps its honest-FAIL / flagged rows.

The tests do *not* re-derive the FEM solution: that is the module's own test
file's job, and duplicating it here would double the runtime without adding
evidence.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "body_force_inversion"
DOC = ROOT / "docs" / "CASE-FEM-BODY-FORCE-INVERSION.md"


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env.setdefault("OMP_NUM_THREADS", "2")
    return subprocess.run(
        [sys.executable, "-m", "tensorfem.cli", "body-force-inversion", *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


# ---------------------------------------------------------------- entry ----- #
def test_package_namespace_exposes_the_case():
    import tensorfem
    from tensorfem import body_force_inversion as module

    assert tensorfem.run_body_force_inversion is module.run_body_force_inversion
    for name in (
        "InversionOptions",
        "ModalBasis",
        "NodalBasis",
        "ObservationSet",
        "PlaneStressMesh",
        "PlaneStressModel",
    ):
        assert getattr(tensorfem, name) is getattr(module, name), name


def test_cli_advertises_the_subcommand():
    proc = subprocess.run(
        [sys.executable, "-m", "tensorfem.cli", "--help"],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0
    assert "body-force-inversion" in proc.stdout


def test_cli_rejects_an_order_it_has_no_truth_vector_for():
    proc = _run_cli("--order", "3")
    assert proc.returncode != 0
    assert "fixed at 2" in (proc.stdout + proc.stderr)


# ----------------------------------------------------------------- gate ----- #
def test_cli_gate_passes_on_the_acceptance_case(tmp_path: Path):
    target = tmp_path / "acceptance.json"
    proc = _run_cli("--nx", "12", "--ny", "6", "--output", str(target))
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(target.read_text())
    assert payload["schema"] == "tensorfem.body-force-inversion/1"
    assert payload["entry"] == "tensorfem.cli body-force-inversion"
    assert payload["passed"] is True
    assert all(payload["acceptance"].values())
    assert payload["observations"]["fraction"] <= 0.10 + 1e-12
    assert payload["field_rel_l2"] < 0.05
    assert payload["iterations"] <= payload["adjoint_iteration_budget"]
    # the convergence curve is part of the anchor acceptance, not a nicety
    assert len(payload["convergence"]) == payload["iterations"]
    assert payload["convergence"][-1]["schema"] == "tensorfem.body-force-inversion-iter/1"


def test_cli_gate_fails_and_still_writes_the_evidence_on_noisy_data(tmp_path: Path):
    target = tmp_path / "noisy.json"
    proc = _run_cli("--nx", "12", "--ny", "6", "--noise", "0.02", "--output", str(target))
    assert proc.returncode != 0
    payload = json.loads(target.read_text())
    assert payload["passed"] is False
    assert payload["acceptance"]["field_within_5pct"] is False
    assert payload["field_rel_l2"] > 0.05


def test_cli_gate_fails_when_the_adjoint_budget_is_too_small(tmp_path: Path):
    target = tmp_path / "starved.json"
    proc = _run_cli(
        "--nx", "12", "--ny", "6", "--max-adjoint", "2", "--output", str(target)
    )
    assert proc.returncode != 0
    payload = json.loads(target.read_text())
    assert payload["passed"] is False
    assert payload["iterations"] <= 2
    assert payload["acceptance"]["iterations_within_budget"] is True
    assert payload["acceptance"]["field_within_5pct"] is False


def test_cli_output_paths_resolve_for_file_dir_and_default_runs(tmp_path: Path):
    # explicit file
    as_file = tmp_path / "explicit.json"
    assert _run_cli("--nx", "8", "--ny", "4", "--output", str(as_file)).returncode == 0
    assert json.loads(as_file.read_text())["passed"] is True
    # explicit directory (trailing separator): canonical result.json inside it
    as_dir = tmp_path / "explicit_dir"
    assert _run_cli("--nx", "8", "--ny", "4", "--output", f"{as_dir}{os.sep}").returncode == 0
    assert (as_dir / "result.json").is_file()
    # no --output at all: runs/<command>/<tag>/result.json under --runs
    runs = tmp_path / "runs"
    assert _run_cli("--nx", "8", "--ny", "4", "--runs", str(runs)).returncode == 0
    default = runs / "body-force-inversion" / "order2-frac0.1-noise0" / "result.json"
    assert default.is_file(), "default output directory was not created"


# ------------------------------------------------------------ artifacts ----- #
def _summary() -> dict:
    path = ART / "summary.json"
    if not path.exists():  # pragma: no cover - guards a checkout accident
        pytest.fail(
            f"{path} is missing; regenerate with "
            "`PYTHONPATH=src python3 scripts/run_body_force_inversion.py`"
        )
    return json.loads(path.read_text())


def test_run_artifacts_exist_and_all_acceptance_flags_hold():
    for name in (
        "summary.json",
        "case_acceptance.json",
        "observability.json",
        "controls.json",
        "noise_sweep.json",
        "forward_verification.json",
        "benchmark_rows.json",
    ):
        assert (ART / name).is_file(), name
    summary = _summary()
    assert all(summary["acceptance"].values())
    assert summary["adjoint_solves"] == summary["iterations"]
    assert summary["iterations"] <= summary["adjoint_budget"]
    for fig in ("fig_convergence.png", "fig_fields.png", "fig_noise.png"):
        path = ART / "figures" / fig
        assert path.is_file() and path.stat().st_size > 10_000, fig


def test_every_doc_number_is_the_artifact_number():
    summary = _summary()
    doc = DOC.read_text()
    quoted = [
        f"{summary['field_rel_l2']:.2e}",  # field error
        f"{summary['observed_fraction'] * 100:.2f}%",  # observed node fraction
        f"{summary['patch_max_relative_error']:.2e}",  # patch error
    ]
    missing = [token for token in quoted if token not in doc]
    assert not missing, f"doc does not quote the artifact values {missing}"
    assert str(summary["iterations"]) in doc
    # the honest-FAIL numbers must survive a regeneration too
    assert f"{summary['noise_1pct_field_rel_l2_undamped']:.2f}" in doc


def test_benchmark_rows_reproduce_the_summary_and_keep_the_honest_rows():
    rows = json.loads((ART / "benchmark_rows.json").read_text())
    assert rows["anchor_slug"] == "JOUR-DIFFLBM-L3-INVERSION"
    assert rows["repo"] == "TensorFEM"
    assert rows["repro_cmd"].endswith("body_force_inversion_benchmark.py")
    by_status = {}
    for row in rows["rows"]:
        by_status.setdefault(row["status"], []).append(row)
        assert {"case_name", "ref_source", "metric", "tolerance", "ref", "got"} <= set(row)
    summary = _summary()
    recovery = by_status["passed"][0]
    assert recovery["got"]["field_rel_l2"] == summary["field_rel_l2"]
    patch = [r for r in rows["rows"] if "patch" in r["case_name"]][0]
    assert patch["got"]["max_relative_error"] == summary["patch_max_relative_error"]
    manufactured = [r for r in rows["rows"] if "manufactured" in r["case_name"]][0]
    assert [round(r, 3) for r in manufactured["got"]["observed_rates"]] == summary[
        "manufactured_rates"
    ]
    # 1 % noise is an honest FAIL, both controls are flagged, never 'passed'
    assert len(by_status["failed"]) == 1 and "noise" in by_status["failed"][0]["case_name"]
    assert len(by_status["flagged"]) == 2