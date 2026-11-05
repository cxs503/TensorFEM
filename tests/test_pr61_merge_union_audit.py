"""Gate for the PR #61 reconciliation audit (``artifacts/pr61_union_audit.py``).

The merge that heals PR #61 is a three-way merge whose failure mode is silent:
a mechanical overlay can take one side and leave no conflict marker, so a green
test suite would not notice.  These tests pin the audit itself:

* the delivered merge satisfies the union obligation;
* the checker has teeth — it rejects both one-sided overlays and a single-line
  deletion, so "zero loss" cannot be an artefact of a permissive checker;
* the evidence JSON shipped in ``artifacts/`` equals a fresh run, so the record
  cannot drift away from the tree it describes.

The frozen PR coordinates are commits of this repository's own history; on a
history that does not contain them (e.g. an export without branch history) the
tests skip instead of failing, because there is nothing to audit there.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
AUDIT = ROOT / "artifacts" / "pr61_union_audit.py"
EVIDENCE = ROOT / "artifacts" / "pr61_union_audit_evidence.json"
TABLES = ROOT / "artifacts" / "pr61_report_tables.py"
DOC = ROOT / "docs" / "pr61-merge-reconciliation.md"


def _load_audit():
    spec = importlib.util.spec_from_file_location("pr61_union_audit", AUDIT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _have_history(*revs: str) -> bool:
    for rev in revs:
        proc = subprocess.run(
            ("git", "cat-file", "-e", f"{rev}^{{commit}}"), cwd=ROOT, capture_output=True
        )
        if proc.returncode != 0:
            return False
    return True


audit = _load_audit()
needs_history = pytest.mark.skipif(
    not _have_history(audit.BASE, audit.MAIN, audit.BRANCH, audit.MERGE, audit.TIP),
    reason="PR #61 history (merge-base/main/branch/merge/tip) not present in this clone",
)


@needs_history
def test_delivered_merge_satisfies_union_obligation():
    evidence = audit.check(audit.MERGE, audit.TIP)
    assert evidence["failures"] == []
    assert evidence["verdict"] == "UNION_OK"
    assert evidence["main_is_ancestor"] is True, "PR would stay dirty against main"
    assert evidence["merge_tree_oracle"]["conflicted_files"] == sorted(audit.HAND_RESOLVED)


@needs_history
def test_every_single_side_change_is_carried_verbatim():
    evidence = audit.check(audit.MERGE, audit.TIP)
    mismatched = [rec for rec in evidence["carried_files"] if rec["verdict"].startswith("FAIL")]
    assert mismatched == []
    # 234 main-only + 7 branch-only + 2 hand-resolved == the whole PR surface
    assert evidence["counts"]["files_carried_verbatim_or_hand"] == (
        evidence["counts"]["files_main_only"]
        + evidence["counts"]["files_branch_only"]
        + evidence["counts"]["files_both_sides"]
    )


@needs_history
def test_checker_rejects_one_sided_overlays():
    """A checker that accepts an overlay is worthless; both must be rejected."""
    evidence = audit.check(audit.MERGE, audit.TIP)
    controls = {c["control"]: c for c in evidence["negative_controls"]}
    for label in ("main-only overlay", "branch-only overlay"):
        assert controls[label]["verdict"] == "control-fires", f"{label} not rejected"
        assert controls[label]["rejected_files"], f"{label} rejected no file"


@needs_history
def test_checker_rejects_a_single_dropped_line():
    changelog = audit.blob_text(audit.MERGE, "CHANGELOG.md")
    line = next(ln for ln in changelog.splitlines() if "1.33e-15" in ln)
    mutated = changelog.replace(line + "\n", "", 1)
    rec = audit.union_verdict(
        "CHANGELOG.md",
        audit.blob_text(audit.BASE, "CHANGELOG.md"),
        audit.blob_text(audit.MAIN, "CHANGELOG.md"),
        audit.blob_text(audit.BRANCH, "CHANGELOG.md"),
        mutated,
    )
    assert rec["verdict"] == "FAIL-union", "dropping a measured number must be detected"


@needs_history
def test_unrelated_edit_is_detected():
    """Editing text that neither side contributed must not look like a union."""
    changelog = audit.blob_text(audit.MERGE, "CHANGELOG.md")
    mutated = changelog.replace("Residual 1.33e-15", "Residual 1.33e-14", 1)
    if mutated == changelog:  # string not present: pick another number
        mutated = changelog.replace("1.33e-15", "1.33e-14", 1)
    rec = audit.union_verdict(
        "CHANGELOG.md",
        audit.blob_text(audit.BASE, "CHANGELOG.md"),
        audit.blob_text(audit.MAIN, "CHANGELOG.md"),
        audit.blob_text(audit.BRANCH, "CHANGELOG.md"),
        mutated,
    )
    assert rec["verdict"] == "FAIL-union", "a corrupted measurement must be detected"


def _stable(evidence: dict) -> dict:
    """Evidence fields that do not move with the tip (the tip advances per commit)."""
    stable = {k: v for k, v in evidence.items() if k not in {"shas", "counts", "added_after_merge"}}
    stable["shas"] = {k: v for k, v in evidence["shas"].items() if k not in {"tip", "tip_tree"}}
    stable["counts"] = {
        k: v for k, v in evidence["counts"].items() if k != "files_added_after_merge"
    }
    return stable


@needs_history
def test_shipped_evidence_matches_a_fresh_run():
    """The committed JSON must be reproducible from the tree, not hand-edited."""
    shipped = _stable(json.loads(EVIDENCE.read_text()))
    fresh = _stable(audit.check(audit.MERGE, audit.TIP))
    assert shipped == fresh
    assert shipped["verdict"] == "UNION_OK"
    assert shipped["failures"] == []


def test_generated_tables_match_evidence():
    """The report tables are a pure function of the shipped evidence JSON.

    This one needs no branch history, so it is the part of the gate that still
    fires on a shallow CI checkout: it pins the numbers in the document to the
    evidence file instead of to whatever a human last typed (the delivered
    block had in fact lost its table header/separator rows).
    """
    spec = importlib.util.spec_from_file_location("pr61_report_tables", TABLES)
    tables = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tables)
    doc = DOC.read_text()
    assert tables.BEGIN in doc and tables.END in doc, "generated block markers missing"
    delivered = doc.split(tables.BEGIN)[1].split(tables.END)[0]
    assert delivered.strip() == tables.render().strip(), (
        "docs block drifted from artifacts/pr61_union_audit_evidence.json; "
        "run python3 artifacts/pr61_report_tables.py"
    )
