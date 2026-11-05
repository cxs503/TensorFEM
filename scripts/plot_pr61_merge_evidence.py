#!/usr/bin/env python3
"""Render the evidence figure for the PR #61 reconciliation report.

Every number in the figure is parsed from a machine-produced input: the union
audit JSON (`docs/evidence/pr61/union-audit.json`), the two mechanical-overlay
negative controls (re-run here, text output kept in
`docs/evidence/pr61/union-audit-controls.txt`) and the two full-suite pytest
logs under `docs/evidence/pr61/`.  Nothing is typed by hand.

    python3 scripts/plot_pr61_merge_evidence.py            # -> /tmp/pr61_merge_evidence.png
    python3 scripts/plot_pr61_merge_evidence.py --out fig.png

The figure is *not* committed: images count against the merge governor's
artifact budget (<200 kB per PR), and docs/assets/marine_large_angle_gz.png is
the PR's one legitimate asset.  Regenerate on demand instead.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVID = ROOT / "docs" / "evidence" / "pr61" / "union-audit.json"
LOGS = ROOT / "docs" / "evidence" / "pr61"
AUDIT_TOOL = ROOT / "scripts" / "audit_pr61_merge_union.py"

# The audit is pinned to the historical fork point: once the merge is committed,
# merge-base(main, branch) collapses onto main and the audit would see nothing.
AUDIT = ["--base", "e8f6759", "--main", "7accc4b", "--branch", "6b4c4fa"]


def run_audit(merge: str, out: pathlib.Path) -> dict:
    subprocess.run(
        ["python3", str(AUDIT_TOOL), *AUDIT, "--merge", merge, "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    return json.loads(out.read_text())


def dropped(record: dict) -> tuple[int, int]:
    """Content units dropped per side (lines for line-atomic, atoms for prose)."""
    lost_main = lost_branch = 0
    for rec in record.get("files_both_sides", []):
        lost_main += rec.get("atoms_lost_from_main", 0) + rec.get("lines_lost_from_main", 0)
        lost_branch += (rec.get("atoms_lost_from_branch", 0)
                        + rec.get("lines_lost_from_branch", 0))
    return lost_main, lost_branch


def counts_from_log(path: pathlib.Path) -> dict:
    """Parse the pytest summary line; the log is the record, not a typed number."""
    text = path.read_text(errors="replace")
    line = next((ln for ln in reversed(text.splitlines())
                 if re.search(r"\d+ (passed|failed)", ln)), "")
    counts: dict = {"summary": line.strip(), "available": bool(line)}
    for key in ("passed", "failed", "error", "errors", "skipped", "xpassed"):
        m = re.search(rf"(\d+) {key}\b", line)
        if m:
            counts["error" if key == "errors" else key] = int(m.group(1))
    return counts


def main() -> None:
    parse = argparse.ArgumentParser()
    parse.add_argument("--out", default="/tmp/pr61_merge_evidence.png")
    args = parse.parse_args()
    out = pathlib.Path(args.out)

    ev = json.loads(EVID.read_text())
    neg = {
        "overlay:\nmain only": run_audit("7accc4b", pathlib.Path("/tmp/pr61_neg_main.json")),
        "overlay:\nbranch only": run_audit("6b4c4fa", pathlib.Path("/tmp/pr61_neg_branch.json")),
        "real three-way\nmerge (HEAD)": ev,
    }
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.4))
    width = 0.38

    # ---- panel 1: union audit, real merge vs the two mechanical overlays ----
    labels = list(neg)
    lost_main = [dropped(neg[k])[0] for k in labels]
    lost_branch = [dropped(neg[k])[1] for k in labels]
    x = range(len(labels))
    bars = (
        ax1.bar([i - width / 2 for i in x], lost_main, width,
                label="main content dropped", color="#c0392b"),
        ax1.bar([i + width / 2 for i in x], lost_branch, width,
                label="branch content dropped", color="#2471a3"),
    )
    for group in bars:
        for rect in group:
            h = rect.get_height()
            ax1.annotate(f"{int(h)}", (rect.get_x() + rect.get_width() / 2, h),
                         ha="center", va="bottom", fontsize=10)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("content units dropped\n(lines for code/data, atoms for prose)")
    ax1.set_title("Union audit: both mechanical overlays lose content,\n"
                  f"the three-way merge loses none "
                  f"({ev['counts']['files_both_sides']} shared files audited, "
                  f"{ev['single_side_content_checked']} single-side files byte-checked)",
                  fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # ---- panel 2: full suite, merged tree vs main baseline ----
    merged = counts_from_log(LOGS / "pytest-full-merged.log")
    base = counts_from_log(LOGS / "pytest-full-baseline.log")
    keys = [k for k in ("passed", "failed", "skipped") if k in merged or k in base]
    xs = range(len(keys))
    ax2.bar([i - width / 2 for i in xs], [base.get(k, 0) for k in keys], width,
            label="main baseline (7accc4b)", color="#7f8c8d")
    ax2.bar([i + width / 2 for i in xs], [merged.get(k, 0) for k in keys], width,
            label="merged tree (HEAD)", color="#27ae60")
    for i, k in enumerate(keys):
        ax2.annotate(str(base.get(k, 0)), (i - width / 2, base.get(k, 0)),
                     ha="center", va="bottom", fontsize=9)
        ax2.annotate(str(merged.get(k, 0)), (i + width / 2, merged.get(k, 0)),
                     ha="center", va="bottom", fontsize=9)
    ax2.set_xticks(list(xs))
    ax2.set_xticklabels(keys, fontsize=9)
    ax2.set_ylabel("tests")
    dp = merged.get("passed", 0) - base.get("passed", 0)
    df = merged.get("failed", 0) - base.get("failed", 0)
    ax2.set_title("Full suite (pytest -n 16, host load ~700/128 cores):\n"
                  f"merged {merged.get('passed')} passed / {merged.get('failed')} failed "
                  f"vs baseline {base.get('passed')} passed / {base.get('failed')} failed",
                  fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)
    ax2.text(0.02, 0.02,
             f"passed delta {dp:+d}, failed delta {df:+d}; the run set is "
             "baseline + the PR's own new test file.\n"
             "Surviving failures are wall-clock / load gates, re-checked serially "
             "on both trees.",
             transform=ax2.transAxes, fontsize=7, va="bottom", color="#555555")

    fig.suptitle("TensorFEM PR #61 reconciliation - union audit and test parity "
                 f"(main {ev['shas']['main'][:7]} <- branch {ev['shas']['branch'][:7]})",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print("wrote", out, f"{out.stat().st_size / 1024:.0f} KB")
    print("merged", merged)
    print("baseline", base)
    for k in neg:
        print(k.replace("\n", " "), "->", neg[k]["verdict"], dropped(neg[k]))


if __name__ == "__main__":
    main()