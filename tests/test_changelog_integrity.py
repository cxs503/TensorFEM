"""Structural gate for ``CHANGELOG.md`` after merge-conflict resolutions.

The changelog is the one file that two long-lived branches always touch, so it
is also the file where a resolution can pass a textual merge and still corrupt
the document: PR #77 (``agent/m3369`` onto ``main``) had exactly that, a
``## 2026-11-03 Cook`` heading left with no bullets while its four bullets ap-
peared one section further down, inside the plane-strain B-bar entry, and a
duplicated ``## Unreleased`` heading from an earlier mechanical overlay.

``scripts/audit_changelog_merge.py`` encodes those checks; this module pins
them in the suite so the corruption cannot be reintroduced silently.  The test
is pure text parsing - no solver, no numerics - and runs in milliseconds.
"""
from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_changelog_merge.py"


def _auditor():
    spec = importlib.util.spec_from_file_location("audit_changelog_merge", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def audit():
    return _auditor()


@pytest.fixture(scope="module")
def changelog_lines():
    return (ROOT / "CHANGELOG.md").read_text(encoding="utf-8").splitlines()


def test_auditor_script_is_present():
    assert SCRIPT.is_file(), "the merge audit must ship with the repair it documents"


def test_changelog_structure_passes(audit, changelog_lines):
    report = audit.structural_audit(changelog_lines)
    assert report["passed"], report["findings"]
    assert report["unreleased_headings"] == 1


def test_cook_membrane_section_keeps_its_bullets(audit, changelog_lines):
    report = audit.structural_audit(changelog_lines)
    assert report["cook_membrane_bullets"] >= 4
    assert report["cook_bullets_in_bbar_section"] == 0


def test_no_orphaned_level2_heading(audit, changelog_lines):
    assert audit.structural_audit(changelog_lines)["empty_level2_sections"] == []


def test_loss_audit_detects_a_dropped_duplicate(audit):
    """A duplicate bullet must not be able to hide behind its twin."""
    reference = ["## Unreleased", "- same text", "- same text", "- other"]
    result = ["## Unreleased", "- same text", "- other"]
    report = audit.loss_audit(result, reference)
    assert not report["passed"]
    assert report["unexpected_lost_lines"] == ["- same text"]

    ok = audit.loss_audit(result, reference, allow_dropped=("- same text",))
    assert ok["passed"] and ok["lost_lines"] == 1


def test_structural_audit_flags_the_pr77_corruption(audit):
    """Negative control: the pre-resolution file must be rejected."""
    corrupted = [
        "# Changelog",
        "",
        "## Unreleased",
        "- entry",
        "",
        "## 2026-11-03 Cook membrane audit",
        "",
        "## 2026-11-04 - plane-strain B-bar",
        "- bbar bullet",
        "",
        "- `benchmarks/cook_membrane_stress_reference.json`: regenerated",
        "- added `tests/test_cook_stress_audit_closure.py`",
        "",
    ]
    report = audit.structural_audit(corrupted)
    assert not report["passed"]
    assert report["cook_membrane_bullets"] == 0
    assert report["cook_bullets_in_bbar_section"] == 2
    assert any("lost its bullets" in f for f in report["findings"])


def test_merge_audit_reproduces_zero_unexpected_loss(audit):
    """On a merge commit, both parents' lines must survive the resolution."""
    parents = subprocess.run(
        ["git", "rev-list", "--parents", "-n", "1", "HEAD"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if parents.returncode != 0 or len(parents.stdout.split()) < 3:
        pytest.skip("HEAD is not a merge commit in this checkout")
    report = audit.merge_audit("HEAD", allow_dropped=("## Unreleased",))
    assert report["passed"], report
    assert len(report["sides"]) == 2