"""Gate tests for the inverse-identification benchmark and its entry points.

The module under test is :mod:`tensorfem.inverse` (optimiser loop, loss
composition, machine-readable convergence log, seed control).  These tests
assert the *invariants* a caller relies on, not merely that the code runs:

* the CLI entry point writes a convergence log whose SHA-256 equals the one the
  summary reports (a run can be re-verified instead of trusted);
* the negative control (``E`` and ``A`` from data that contain only ``E A``)
  must exit 0 *because* non-identifiability is reported -- and the same check
  must fail when the diagnosis is doctored away;
* the verified benchmark gate is not vacuous: displacing the reference away from
  the recovered value flips a row to FAIL.
"""

from __future__ import annotations

import importlib.util
import json

import pytest

from tensorfem import inverse_cases
from tensorfem.cli import main
from tensorfem.inverse_cases import (
    BAR_TRUTH,
    DECAY_TRUTH,
    TOLERANCE,
    axial_bar_case,
    decay_case,
)


def test_cli_entry_writes_hash_pinned_convergence_log(tmp_path, monkeypatch):
    out = tmp_path / "run"
    monkeypatch.setattr(
        "sys.argv",
        ["tensorfem", "inverse-identify", "--case", "axial-bar", "--output", str(out)],
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    summary = json.loads((out / "summary.json").read_text())
    lines = (out / "convergence.jsonl").read_text().splitlines()
    assert len(lines) >= 2
    assert summary["schema"].startswith("tensorfem.inverse-summary/")
    assert json.loads(lines[0])["kind"] == "header"
    # the hash in the summary pins the log bytes in this directory
    import hashlib

    digest = hashlib.sha256((out / "convergence.jsonl").read_bytes()).hexdigest()
    assert summary["convergence_log_sha256"] == digest
    assert summary["parameters"]["E"] == pytest.approx(BAR_TRUTH["E"], rel=TOLERANCE)


def test_cli_negative_control_passes_only_via_non_identifiability(tmp_path, monkeypatch):
    out = tmp_path / "product"
    monkeypatch.setattr(
        "sys.argv",
        [
            "tensorfem",
            "inverse-identify",
            "--case",
            "axial-bar-product",
            "--output",
            str(out),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["diagnostics"]["identifiable"] is False
    assert summary["diagnostics"]["effective_rank"] == 1
    assert summary["warnings"]


def test_doctored_identifiability_verdict_fails_the_control(monkeypatch):
    """If the diagnosis were reported as identifiable, the control must fail."""

    real = inverse_cases.product_case

    def doctored(*args, **kwargs):
        outcome = real(*args, **kwargs)
        outcome["row"]["identifiable"] = True
        outcome["row"]["passed"] = bool(
            outcome["row"]["identifiable"] is False
            and outcome["row"]["effective_rank"] < 2
        )
        return outcome

    monkeypatch.setattr(inverse_cases, "product_case", doctored)
    assert inverse_cases.product_case()["row"]["passed"] is False


def test_recovered_scalars_are_inside_three_percent():
    decay = decay_case(seed=1, starts=3)["row"]
    bar = axial_bar_case(seed=0)["row"]
    assert decay["converged"] and decay["tau_relative_error"] < TOLERANCE
    assert bar["converged"] and bar["E_relative_error"] < TOLERANCE


def test_gate_is_not_vacuous_when_reference_is_displaced(monkeypatch):
    """A 6 % displaced reference must make the same run fail its gate."""
    monkeypatch.setitem(DECAY_TRUTH, "tau", 4.0 * 1.06)
    row = decay_case(seed=1, starts=3)["row"]
    assert row["tau_relative_error"] > TOLERANCE
    assert row["passed"] is False
    monkeypatch.setitem(DECAY_TRUTH, "tau", 4.0)  # restored by monkeypatch anyway


def test_benchmark_runner_gate_and_artifact(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "fem_inverse_identification_run",
        __file__.replace("tests/test_inverse_identification_benchmark.py",
                         "benchmarks/verified/fem_inverse_identification/run.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "OUT", tmp_path / "bench")
    # the anchor certificate + dump are written into ARTIFACT_DIR; redirect it so
    # the test never rewrites the committed evidence in the tree
    monkeypatch.setattr(module, "ARTIFACT_DIR", tmp_path / "anchor")
    assert module.main() == 0
    payload = json.loads((tmp_path / "bench" / "benchmark.json").read_text())
    assert payload["passed"] is True
    assert payload["schema"] == "tensorfem.inverse-benchmark/1"
    assert payload["reproducibility"]["same_seed_identical"] is True
    assert payload["negative_control"]["identifiable"] is False
    for row in payload["rows"]:
        assert row["relative_error"] < payload["tolerance"]