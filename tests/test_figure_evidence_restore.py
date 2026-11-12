"""Invariants of the figure-evidence restore report.

``benchmarks/verified/figure_evidence_restore.json`` says, per anchor, whether the
figures its ``gates_verified.json`` names are now on disk.  The report is only
worth something if it is *bound to the tree*, so these tests re-derive the facts
it claims instead of trusting the file:

* a restored/complete anchor must have no declared figure missing;
* every figure it reports present must be a real PNG, big enough to carry
  content, and **git-tracked** (a build product left untracked reproduces the
  exact defect this report exists to record);
* an anchor the report marks failed/timeout must still be in the shrink-only
  certificate-debt registry -- honest failure is registered, never tolerated
  silently.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT = REPO_ROOT / "benchmarks" / "verified" / "figure_evidence_restore.json"
DEBT = REPO_ROOT / "benchmarks" / "verified" / "anchor_certificate_debt.json"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
MIN_PNG_BYTES = 2048


def load_report() -> dict:
    return json.loads(REPORT.read_text())


def tracked_files() -> set[str]:
    out = subprocess.check_output(["git", "ls-files"], cwd=REPO_ROOT, text=True)
    return set(out.split())


def test_report_schema_and_entries():
    payload = load_report()
    assert payload["schema"] == "tensorfem.figure-evidence-restore/1"
    assert payload["entries"], "report carries no anchors"
    names = [entry["anchor"] for entry in payload["entries"]]
    assert len(names) == len(set(names)), "duplicate anchor in report"


def test_complete_anchors_have_no_missing_figure():
    payload = load_report()
    for entry in payload["entries"]:
        if entry["status"] in ("restored", "restored_with_drift", "already_complete"):
            missing = entry.get("missing_after") or [
                item["path"] for item in entry.get("declared", []) if not item.get("present")
            ]
            assert missing == [], f"{entry['anchor']} claims complete but misses {missing}"


def test_reported_figures_exist_and_are_tracked():
    payload = load_report()
    tracked = tracked_files()
    checked = 0
    for entry in payload["entries"]:
        for item in entry.get("declared", []):
            if not item.get("present"):
                continue
            path = REPO_ROOT / item["path"]
            assert path.exists(), f"{entry['anchor']}: {item['path']} reported present but absent"
            if path.suffix.lower() != ".png":
                continue
            assert path.is_file()
            blob = path.read_bytes()
            assert blob.startswith(PNG_MAGIC), f"{item['path']} is not a PNG"
            assert len(blob) >= MIN_PNG_BYTES, f"{item['path']} is a placeholder ({len(blob)} B)"
            assert item["path"] in tracked, (
                f"{item['path']} exists but is untracked -- the certificate would 404 "
                "for anyone reading the merged tree"
            )
            checked += 1
    assert checked > 0, "no figure artifact was actually verified"


def test_failed_anchors_remain_registered_as_debt():
    payload = load_report()
    debt = json.loads(DEBT.read_text())["anchors"]
    for entry in payload["entries"]:
        if entry["status"] in ("timeout", "runner_failed", "figure_missing"):
            assert entry["anchor"] in debt, (
                f"{entry['anchor']} failed to restore its evidence but is not in the debt "
                "registry, so the failure would be invisible"
            )


def test_summary_matches_entries():
    payload = load_report()
    summary = payload["summary"]
    assert summary["anchors"] == len(payload["entries"])
    counted = sum(summary["status_counts"].values())
    assert counted == len(payload["entries"])
    assert summary["missing_after"] == sum(
        len(entry.get("missing_after") or []) for entry in payload["entries"]
    )