"""Fail-closed contract for the reviewed public-API manifest (mission m15038 / #760).

The reviewed snapshot has two copies that must never diverge:

* ``api/public-api-v1.json`` -- reviewed source of truth,
* ``src/tensorfem/public_api_manifest.json`` -- the twin that ships inside the
  wheel (``pyproject.toml`` package-data + ``MANIFEST.in``).

Before this gate the only packaged-copy check lived in a pytest, so the release
evidence path (``scripts/run_release_quality_evidence.py`` -> ``api_audit``) could
report PASS while a built wheel advertised a stale API.  It also silently accepted
malformed entries: an unknown key, a missing ``introduced``, or a stability class
without a ``policy`` sentence all passed because only ``name``/``module``/
``stability`` were read.

These tests are metadata-only (sub-second).  They do not re-run any solver.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import tensorfem

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "api" / "public-api-v1.json"
PACKAGED = ROOT / "src" / "tensorfem" / "public_api_manifest.json"
SCHEMA_DOC = ROOT / "api" / "public-api-manifest.schema.json"

sys.path.insert(0, str(ROOT / "scripts"))
import audit_public_api as gate  # noqa: E402


@pytest.fixture()
def payload() -> dict:
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


def _write(tmp_path: Path, name: str, data: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _audit(tmp_path: Path, data: dict, packaged: dict | None = None) -> dict:
    manifest = _write(tmp_path, "snapshot.json", data)
    twin = PACKAGED if packaged is None else _write(tmp_path, "packaged.json", packaged)
    return gate.audit(manifest, twin, True)


def test_checked_in_snapshot_and_packaged_twin_are_in_sync():
    result = gate.audit()
    assert result["ok"], result["errors"]
    assert result["packaged"] == "in_sync"
    assert result["schema"] == gate.SCHEMA_ID
    assert result["count"] == len(tensorfem.__all__)


def test_stale_packaged_copy_fails_the_gate(tmp_path, payload):
    stale = copy.deepcopy(payload)
    stale["entries"][0]["introduced"] = "0.0.0"
    result = _audit(tmp_path, payload, stale)
    assert not result["ok"]
    assert result["packaged"] == "drift"
    assert any("packaged copy" in error for error in result["errors"])


def test_missing_packaged_copy_fails_the_gate(tmp_path, payload):
    manifest = _write(tmp_path, "snapshot.json", payload)
    result = gate.audit(manifest, tmp_path / "absent.json", True)
    assert not result["ok"]
    assert result["packaged"] == "missing"
    assert any("missing manifest" in error for error in result["errors"])


def test_packaged_check_can_be_skipped_for_out_of_tree_manifests(tmp_path, payload):
    manifest = _write(tmp_path, "snapshot.json", payload)
    assert gate.audit(manifest, tmp_path / "absent.json", False)["ok"]


def test_unknown_or_missing_entry_keys_fail_closed(tmp_path, payload):
    typo = copy.deepcopy(payload)
    typo["entries"][3]["stabilty"] = typo["entries"][3].pop("stability")
    result = _audit(tmp_path, typo)
    assert not result["ok"]
    assert any("unknown keys" in error for error in result["errors"])

    dropped = copy.deepcopy(payload)
    del dropped["entries"][3]["introduced"]
    result = _audit(tmp_path, dropped)
    assert not result["ok"]
    assert any("missing keys" in error for error in result["errors"])


def test_wrong_schema_id_and_unknown_top_level_key_fail(tmp_path, payload):
    wrong = copy.deepcopy(payload)
    wrong["schema"] = "tensorfem.public-api-manifest/0.9"
    result = _audit(tmp_path, wrong)
    assert any("schema id" in error for error in result["errors"])

    extra = copy.deepcopy(payload)
    extra["generated_by"] = "hand edit"
    result = _audit(tmp_path, extra)
    assert any("unknown top-level keys" in error for error in result["errors"])


def test_stability_without_policy_sentence_fails(tmp_path, payload):
    unpoliced = copy.deepcopy(payload)
    del unpoliced["policy"]["internal_candidate"]
    result = _audit(tmp_path, unpoliced)
    assert not result["ok"]
    assert any("has no policy sentence" in error for error in result["errors"])


def test_bad_values_in_entries_fail(tmp_path, payload):
    for mutate, fragment in (
        (lambda e: e.update(stability="stable-ish"), "invalid stability"),
        (lambda e: e.update(introduced="v0.1"), "invalid introduced"),
        (lambda e: e.update(module="tensorfem/plate"), "invalid module"),
        (lambda e: e.update(name="not a name"), "invalid name"),
        (lambda e: e.update(object=""), "must be a non-empty string"),
    ):
        broken = copy.deepcopy(payload)
        mutate(broken["entries"][5])
        result = _audit(tmp_path, broken)
        assert not result["ok"], fragment
        assert any(fragment in error for error in result["errors"]), result["errors"]


def test_schema_document_matches_the_manifest_contract(payload):
    schema = json.loads(SCHEMA_DOC.read_text(encoding="utf-8"))
    assert schema["$schema"].endswith("2020-12/schema")
    assert set(schema["required"]) == set(payload) == gate.TOP_LEVEL_KEYS
    assert schema["properties"]["schema"]["const"] == gate.SCHEMA_ID
    entry_schema = schema["properties"]["entries"]["items"]
    assert set(entry_schema["required"]) == gate.ENTRY_KEYS == set(payload["entries"][0])
    assert set(entry_schema["properties"]["stability"]["enum"]) == gate.VALID_STABILITY
    assert entry_schema["additionalProperties"] is False
    assert schema["additionalProperties"] is False


def test_jsonschema_cross_check_agrees_on_the_shipped_manifest():
    pytest.importorskip("jsonschema")
    result = gate.audit()
    assert result["jsonschema"] == "passed", result["errors"]


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    return env


def test_cli_exit_code_and_message_are_gate_contract(tmp_path, payload):
    good = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "audit_public_api.py"), "--json"],
        capture_output=True, text=True, cwd=ROOT, env=_env(),
    )
    assert good.returncode == 0, good.stdout + good.stderr
    assert json.loads(good.stdout)["packaged"] == "in_sync"

    broken = copy.deepcopy(payload)
    broken["entries"][0]["stability"] = "unknown"
    manifest = _write(tmp_path, "broken.json", broken)
    bad = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "audit_public_api.py"),
         "--manifest", str(manifest), "--no-package-check"],
        capture_output=True, text=True, cwd=ROOT, env=_env(),
    )
    assert bad.returncode == 1, bad.stdout
    assert "API audit: FAIL" in bad.stdout
    assert any(line.startswith("ERROR:") for line in bad.stdout.splitlines())

def test_hardening_contrast_without_the_gate_is_measured_not_claimed():
    """The pre-change gate let 6/7 in-manifest faults through; the new one 0/8."""
    contrast = ROOT / "artifacts" / "m17947_public_api_gate_baseline.json"
    payload = json.loads(contrast.read_text(encoding="utf-8"))
    summary = payload["summary"]
    assert summary["hardened_escaped"] == 0
    assert summary["baseline_escaped"] >= 6
    assert summary["clean_manifest_still_passes"]
    assert summary["faults_injected"] == len(payload["fault_cases"])
    for row in payload["fault_cases"]:
        assert row["hardened"]["ok"] is False, row["case"]
        assert row["hardened"]["error_count"] >= 1, row["case"]

    fresh = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "m17947_gate_baseline_compare.py"), "--check"],
        capture_output=True, text=True, cwd=ROOT, env=_env(),
    )
    if "unknown revision" in (fresh.stderr + fresh.stdout):  # shallow/exported checkout
        pytest.skip("git history unavailable")
    assert fresh.returncode == 0, fresh.stdout + fresh.stderr
