"""Certificate integrity of the ``fem_shell_modal`` anchor.

The anchor is an honest **FAIL** (two shell gates are red), and this test does not
argue with that.  What it locks down is the *evidence chain*: the certificate used
to name ``shell_modal_audit.png`` while no commit contained that file, so
``tests/test_verified_anchor_certificates.py`` rejected the anchor as unregistered
debt and the fail-closed gate went red on ``main``.  The repair renders the
figures from the committed ``result.json`` (``--from-dump``) and stamps
``figure_provenance.json`` with the sha256 of that JSON.  These tests make the
repair non-reversible-by-accident:

* every artifact the certificate names must exist **and** be git-tracked;
* the provenance stamp must match the ``result.json`` on disk (a re-run that
  rewrites the numbers without re-rendering the figures is caught here);
* re-rendering from the dump must reproduce the committed PNG pixels;
* the anchor must audit clean, i.e. carry no hard certificate defect.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ANCHOR = ROOT / "benchmarks" / "verified" / "fem_shell_modal"
DRIVER = ANCHOR / "run_verification.py"


def _load_driver():
    spec = importlib.util.spec_from_file_location("fem_shell_modal_driver", DRIVER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _tracked(rel: str) -> bool:
    done = subprocess.run(
        ["git", "ls-files", "--error-unmatch", rel],
        cwd=ROOT, capture_output=True, text=True,
    )
    return done.returncode == 0


def _named_artifacts() -> list[str]:
    payload = json.loads((ANCHOR / "gates_verified.json").read_text())
    names: list[str] = []
    for value in (payload.get("artifacts") or {}).values():
        if isinstance(value, str):
            names.append(value)
        else:
            names.extend(str(item) for item in value)
    return names


def test_every_named_artifact_exists_and_is_tracked():
    names = _named_artifacts()
    assert names, "certificate names no artifacts at all"
    missing = [name for name in names if not (ANCHOR / name).exists()]
    assert missing == [], f"certificate names artifacts that are not on disk: {missing}"
    untracked = [
        name for name in names
        if not _tracked(str((ANCHOR / name).relative_to(ROOT)))
    ]
    assert untracked == [], f"certificate names artifacts that no commit contains: {untracked}"


def test_provenance_stamp_matches_the_result_json_on_disk():
    stamp = json.loads((ANCHOR / "figure_provenance.json").read_text())
    live = hashlib.sha256((ANCHOR / "result.json").read_bytes()).hexdigest()
    assert stamp["schema"] == "tensorfem.figure-provenance/1"
    assert stamp["source"] == "result.json"
    assert stamp["source_sha256"] == live, (
        "result.json changed after the figures were rendered: re-run "
        "`run_verification.py --from-dump` and commit the figures + stamp together"
    )
    assert stamp["solver_executed"] is False


def test_provenance_lists_every_rendered_figure_with_its_hash():
    stamp = json.loads((ANCHOR / "figure_provenance.json").read_text())
    for name, digest in stamp["figures"].items():
        path = ANCHOR / name
        assert path.exists(), f"provenance names a figure that is missing: {name}"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, (
            f"{name} was edited or re-rendered without refreshing figure_provenance.json"
        )


def test_rerender_from_dump_reproduces_the_committed_pixels(tmp_path):
    driver = _load_driver()
    payload = json.loads((ANCHOR / "result.json").read_text())["anchors"]
    written = driver.render_figures(
        payload["beam"], payload["plate"], payload["shell"], tmp_path
    )
    assert {p.name for p in written} == set(driver.FIGURE_CAPTIONS)
    for path in written:
        committed = mpimg.imread(ANCHOR / path.name)
        rebuilt = mpimg.imread(path)
        assert committed.shape == rebuilt.shape, f"{path.name}: shape changed"
        assert np.array_equal(committed, rebuilt), (
            f"{path.name}: the committed figure is not the render of result.json"
        )


def test_certificate_figure_is_among_the_rendered_figures():
    driver = _load_driver()
    assert driver.CERTIFICATE_FIGURE == "shell_modal_audit.png"
    assert driver.CERTIFICATE_FIGURE in driver.FIGURE_CAPTIONS


def test_anchor_audits_clean():
    sys.path.insert(0, str(ROOT / "tools"))
    import verified_anchor_audit as audit

    tracked = audit.tracked_files()
    record = audit.audit_anchor(ANCHOR, tracked)
    assert record.defects == [], f"hard certificate defects: {record.defects}"
    assert record.grade == "certified"


def test_verdict_stays_honest_fail():
    """The artifact repair must not have quietly turned the anchor green."""
    envelope = json.loads((ANCHOR / "gates_verified.json").read_text())
    failed = [gate["id"] for gate in envelope["gates"] if not gate["passed"]]
    assert envelope["passed"] is False
    assert failed == ["G3.shell_lowest_physical_vs_donnell", "G4.shell_mechanism_free_low_spectrum"]
    assert sorted(envelope["failed_gates"]) == sorted(failed)
    for gate in envelope["gates"]:
        assert gate["threshold"] is not None, f"{gate['id']} is unfalsifiable"


def test_triage_tool_classifies_the_repaired_anchor_as_clean():
    """The triage must drop the anchor once it is certificate-clean."""
    sys.path.insert(0, str(ROOT / "tools"))
    import anchor_artifact_triage

    payload = anchor_artifact_triage.build()
    anchors = {row["anchor"] for row in payload["rows"]}
    assert "fem_shell_modal" not in anchors, (
        "fem_shell_modal is certificate-clean again; if this reappears, read "
        "docs/fem-anchor-artifact-debt-triage.md before touching the debt registry"
    )
    assert payload["named_artifacts_missing"] >= 0