"""Guards for the springing-whipping-hydroelastic external-anchor artifact.

The tests encode the discipline of this experiment rather than any particular
number: an anchor is promoted only on a character-exact quote with a
page/table locator; the transfer chain must stay dimensionally homogeneous and
internally closed; and the artifact may not claim more than the transcription
gate allows.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "benchmarks" / "anchors" / "springing_whipping_hydroelastic_anchor.py"
ARTIFACT_PATH = REPO_ROOT / "benchmarks" / "anchors" / "springing-whipping-hydroelastic.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("_swh_anchor", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def anchor_module():
    return _load_module()


@pytest.fixture(scope="module")
def artifact():
    assert ARTIFACT_PATH.exists(), (
        "artifact missing: run python benchmarks/anchors/"
        "springing_whipping_hydroelastic_anchor.py"
    )
    return json.loads(ARTIFACT_PATH.read_text())


def test_artifact_has_required_schema(artifact):
    """判据逐条 + 实测值 + 原始数据来源/脚本路径 must all be present."""
    assert artifact["task_key"] == "paper-exp-springing-whipping-hydroelastic-1"
    assert artifact["repository"] == "TensorFEM"
    for key in ("criteria", "machine_checks", "case_caliber", "source_registry",
                "conversion_chain", "verdict", "reproduce"):
        assert key in artifact, key
    criteria = {c["id"]: c for c in artifact["criteria"]}
    assert set(criteria) == {"C1", "C2", "C3", "C4"}
    for c in criteria.values():
        assert c["requirement"] and c["verdict"] and c["evidence"]
        assert isinstance(c["measured"], dict) and c["measured"]
    # the script that produced the numbers is recorded and matches the on-disk file
    assert artifact["generated_by"]["script"].endswith("springing_whipping_hydroelastic_anchor.py")
    for check in artifact["machine_checks"]:
        assert any(k.startswith("tolerance") for k in check) or check["id"].startswith("CHK7")
        assert "passed" in check


def test_machine_checks_pass(artifact):
    failed = [c["id"] for c in artifact["machine_checks"] if not c["passed"]]
    assert failed == [], f"failing machine checks: {failed}"
    assert len(artifact["machine_checks"]) == 7


def test_round_trip_and_dimensional_homogeneity(anchor_module):
    dims = anchor_module.check_dimensional_homogeneity()
    assert dims["n_homogeneous"] == dims["n_steps"]
    rt = anchor_module.check_round_trip()
    assert rt["max_relative_residual"] < 1e-9
    mom = anchor_module.check_moment_closure()
    assert mom["m0_relative_residual"] < 1e-9
    assert mom["variance_relative_residual"] < 1e-9


def test_no_anchor_is_promoted_without_verbatim_evidence(anchor_module, artifact):
    """Zero eligible entries in, zero claim-eligible anchors out, C1 WITHHELD."""
    registry = json.loads(anchor_module.TRANSCRIPTION_PATH.read_text())
    rows = anchor_module.classify_entries(registry)
    assert [r for r in rows if r["claim_eligible"]] == []
    assert artifact["source_registry"]["claim_eligible_entries"] == 0
    criteria = {c["id"]: c for c in artifact["criteria"]}
    assert criteria["C1"]["status"] == "WITHHELD"
    assert criteria["C1"]["verdict"].startswith("FAIL")
    # every source that was sought is recorded with an acquisition outcome
    for src in artifact["source_registry"]["sources_sought"]:
        assert src["acquisition_status"] in ("NOT_OBTAINED", "OBTAINED")
        assert src["acquisition_attempts"], src["id"]


@pytest.mark.parametrize(
    "entry,reason",
    [
        ({"anchor_id": "a", "source_id": "s", "quote": "", "locator": {"page": 3},
          "verbatim_status": "VERBATIM"}, "empty quote"),
        ({"anchor_id": "b", "source_id": "s", "quote": "some text", "locator": {},
          "verbatim_status": "VERBATIM"}, "missing page/table locator"),
        ({"anchor_id": "c", "source_id": "s", "quote": "some text", "locator": {"page": 7},
          "verbatim_status": "WITHHELD"}, "verbatim_status=WITHHELD"),
    ],
)
def test_promotion_rule_rejects_incomplete_entries(anchor_module, entry, reason):
    rows = anchor_module.classify_entries({"entries": [entry]})
    assert rows[0]["claim_eligible"] is False
    assert rows[0]["rejection_reason"] == reason


def test_complete_verbatim_entry_would_be_promotable(anchor_module):
    """The gate is a real gate, not a hard-coded FAIL."""
    entry = {
        "anchor_id": "demo",
        "source_id": "SNAME-TR-3-37",
        "quote": "quoted sentence",
        "locator": {"document_part": "Table 3-2", "page": 27},
        "verbatim_status": "VERBATIM",
        "values": [{"symbol": "Cp", "printed_value": "1.0"}],
    }
    rows = anchor_module.classify_entries({"entries": [entry]})
    assert rows[0]["claim_eligible"] is True


def test_dry_run_fixture_is_flagged_ineligible(artifact):
    fixture = artifact["dry_run_fixture"]
    assert fixture["claim_eligible"] is False
    assert fixture["label"].startswith("SYNTHETIC")
    assert artifact["domain_comparison"]["status"] == "NOT_PERFORMED"


def test_artifact_is_reproducible(anchor_module):
    """Re-running the generator reproduces every machine-check number."""
    fresh = anchor_module.build_artifact()
    old = json.loads(ARTIFACT_PATH.read_text())
    assert json.dumps(fresh["machine_checks"], sort_keys=True) == \
           json.dumps(old["machine_checks"], sort_keys=True)
    assert json.dumps(fresh["criteria"], sort_keys=True) == \
           json.dumps(old["criteria"], sort_keys=True)


def test_check_mode_cli(anchor_module):
    assert anchor_module.main(["--check", "--out", "/tmp/_swh_anchor_check.json"]) == 0