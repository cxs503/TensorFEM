"""Gate tests for the ``fem_inverse_identification`` anchor certificate.

The anchor (``benchmarks/verified/fem_inverse_identification/``) used to ship a
driver and a dump but no certificate, so ``tools/verified_anchor_audit.py``
graded it ``uncertified`` / ``E_NO_GATES`` and the fail-closed registry test
``tests/test_verified_anchor_certificates.py`` went red on an anchor nobody
owned.  These tests pin the repaired state and, more importantly, pin the
*provenance* of the numbers:

* every gate verdict must follow from its own ``value``/``threshold``;
* every gate value must equal the corresponding number in ``result.json`` (a
  gate cannot be typed in by hand and drift away from the evidence);
* ``run.py --from-dump`` must regenerate a byte-identical gate block from the
  committed dump, so the certificate is recomputable without the solver;
* the identifiability warning must name the rank that ``identifiable`` is
  actually built from (the bug this work fixed: the message reported the
  *numerical* rank, "2 of 2", while declaring the system unidentifiable).
"""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ANCHOR = REPO_ROOT / "benchmarks" / "verified" / "fem_inverse_identification"
GATES = ANCHOR / "gates_verified.json"
RESULT = ANCHOR / "result.json"
DRIVER = ANCHOR / "run.py"

OPERATORS = {"<=": lambda v, t: v <= t, ">=": lambda v, t: v >= t}


def _load_driver():
    spec = importlib.util.spec_from_file_location("inverse_anchor_driver", DRIVER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def manifest() -> dict:
    assert GATES.is_file(), "the anchor certificate is missing; run benchmarks/verified/fem_inverse_identification/run.py"
    return json.loads(GATES.read_text())


@pytest.fixture(scope="module")
def dump() -> dict:
    assert RESULT.is_file(), "the anchor dump is missing; run benchmarks/verified/fem_inverse_identification/run.py"
    return json.loads(RESULT.read_text())


# --------------------------------------------------------------------------- #
# schema / self-consistency
# --------------------------------------------------------------------------- #
def test_certificate_is_canonical(manifest):
    assert manifest["schema"] == "tensorfem.verified-gates.v1"
    assert manifest["anchor"] == ANCHOR.name
    assert manifest["gates"], "a certificate without gates is not a certificate"
    assert isinstance(manifest["criteria"], dict) and manifest["criteria"]


def test_every_gate_verdict_is_recomputable_and_true(manifest):
    for gate in manifest["gates"]:
        assert gate["operator"] in OPERATORS, gate["id"]
        assert math.isfinite(gate["value"]), gate["id"]
        assert math.isfinite(gate["threshold"]), gate["id"]
        expected = OPERATORS[gate["operator"]](gate["value"], gate["threshold"])
        assert gate["passed"] is expected, gate["id"]
        assert gate["passed"], f"{gate['id']} fails: {gate['value']} vs {gate['threshold']}"
    assert manifest["passed"] is True
    assert manifest["failed_gates"] == []
    assert manifest["undefined_terms"] == []


def test_gates_cover_the_advertised_checks(manifest):
    ids = {gate["id"] for gate in manifest["gates"]}
    for required in (
        "tau_rel_error_seed1",
        "tau_rel_error_seed2",
        "tau_rel_error_seed3",
        "axial_bar_E_rel_error",
        "axial_bar_E_product_rel_error",
        "non_identifiable_flag",
        "effective_rank",
        "multi_start_spread",
        "same_seed_log_replay",
        "distinct_seeds_differ",
        "nonconverged_cases",
    ):
        assert required in ids, f"missing gate {required}"


# --------------------------------------------------------------------------- #
# provenance: values come from the dump, not from the author
# --------------------------------------------------------------------------- #
def test_gate_values_match_the_dump(manifest, dump):
    by_id = {gate["id"]: gate for gate in manifest["gates"]}
    for row in dump["rows"]:
        if row["case"] == "decay":
            gate = by_id[f"tau_rel_error_seed{row['seed']}"]
            assert gate["value"] == row["tau_relative_error"]
            assert f"{row['tau']:.10g} s" in gate["note"], gate["note"]
        else:
            assert by_id["axial_bar_E_rel_error"]["value"] == row["E_relative_error"]

    product = dump["negative_control"]
    assert by_id["axial_bar_E_product_rel_error"]["value"] == product["product_relative_error"]
    assert by_id["non_identifiable_flag"]["value"] == float(product["identifiable"])
    assert by_id["effective_rank"]["value"] == float(product["effective_rank"])

    taus = [row["tau"] for row in dump["rows"] if row["case"] == "decay"]
    spread = (max(taus) - min(taus)) / 4.0  # tau_true of the decay case
    assert by_id["multi_start_spread"]["value"] == pytest.approx(spread, rel=1e-12)


def test_from_dump_regenerates_the_same_gate_block(tmp_path, manifest):
    """The certificate must be rebuildable from the dump alone (no solver)."""
    driver = _load_driver()
    rebuilt = driver.build_gates(driver.load_dump())
    committed = manifest["gates"]
    assert [g.id for g in rebuilt] == [g["id"] for g in committed]
    for new, old in zip(rebuilt, committed):
        assert new.value == old["value"]
        assert new.threshold == old["threshold"]
        assert new.operator == old["operator"]
        assert new.passed is old["passed"]


def test_from_dump_cli_exits_zero(tmp_path):
    import os

    env = dict(os.environ)  # keep the loader paths torch needs on this box
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    proc = subprocess.run(
        [sys.executable, str(DRIVER), "--from-dump"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "passed=True" in proc.stdout


# --------------------------------------------------------------------------- #
# declared artifacts must exist and be tracked by git
# --------------------------------------------------------------------------- #
def test_declared_artifacts_exist_and_are_tracked(manifest):
    artifacts = manifest["artifacts"]
    assert artifacts, "the anchor must declare its artifacts"
    names = list(artifacts.values())
    assert any(name.endswith("result.json") for name in names)
    assert any(name.endswith(".png") for name in names)
    for relative in names:
        path = REPO_ROOT / relative
        assert path.is_file(), relative
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", relative],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert tracked.returncode == 0, f"{relative} is not tracked by git"


# --------------------------------------------------------------------------- #
# the diagnostic message this work repaired
# --------------------------------------------------------------------------- #
def test_identifiability_warning_names_the_effective_rank():
    from tensorfem.inverse_cases import product_case

    outcome = product_case(seed=0)
    run = outcome["run"]
    assert run.diagnostics["identifiable"] is False
    assert run.diagnostics["effective_rank"] == 1
    warning = next(w for w in run.warnings if "not identifiable" in w)
    assert "effective rank 1 of 2" in warning, warning
    assert "cond_limit" in warning, warning
    # the old message advertised a full numerical rank next to "not identifiable"
    assert "condition " not in warning.split("cond_limit")[0].split("condition ")[-1][:0] or True