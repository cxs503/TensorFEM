"""Evidence + figures for the PR #56 (agent/m3148) merge cure, mission 3925.

Everything numeric in the write-up and in the API report is produced here from
files that already exist on disk -- the pytest logs, the git history and the
retention audit -- so no number is transcribed by hand.

Inputs
    artifacts/pytest_main_base_v.log full pytest run on tensorfem_local/main (-v)
    artifacts/pytest_merge_v.log     full pytest run on the merge result  (-v)
    artifacts/pytest_main_base.log   the earlier -q run that aborted in
    artifacts/pytest_merge.log       pytest_sessionfinish (kept as evidence)
    artifacts/pr56-merge-audit.json  two-sided retention audit
Outputs
    artifacts/pr56-merge-evidence.json
    docs/assets/merge_cure_pr56/retention_audit.png
    docs/assets/merge_cure_pr56/test_parity.png

Usage (matplotlib 3.10 from the shared TensorLBM venv, Agg backend):
    MPLBACKEND=Agg /root/TensorLBM/.venv/bin/python scripts/merge_cure_pr56_evidence.py
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
FIG = ROOT / "docs" / "assets" / "merge_cure_pr56"
OLD_BRANCH_SHA = "de6195a4349bf31c615cbbbccccdefe5fc458433"
MAIN_SHA = "7a8f4afeb6dd79c2ad3be480a6117ed2550c2347"
BASE_SHA = "7accc4b9cfb45369958c432a8aa2f0ebbb846a5b"

SUMMARY_RE = re.compile(
    r"^(?P<body>(?:\d+ (?:passed|failed|skipped|xfailed|xpassed|error|errors|warning|warnings)"
    r"(?:, )?)+) in (?P<secs>[0-9.]+)s"
)


VERBOSE_STATUS = ("PASSED", "FAILED", "SKIPPED", "ERROR", "XFAIL", "XPASS")


def verbose_counts(text: str) -> dict[str, int]:
    """Count per-test status lines written by `pytest -v` (survives a crashed
    sessionfinish, so it is the cross-check for the summary line)."""
    out = {k: 0 for k in VERBOSE_STATUS}
    for line in text.splitlines():
        for status in VERBOSE_STATUS:
            if f"] {status} tests/" in line:
                out[status] += 1
                break
    return {k: v for k, v in out.items() if v}


def failed_ids(text: str) -> list[str]:
    ids = []
    for line in text.splitlines():
        if "] FAILED tests/" in line:
            ids.append(line.split("] FAILED ", 1)[1].strip())
    return sorted(ids)


def parse_pytest_log(path: pathlib.Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    counts: dict[str, int] = {}
    seconds = None
    for line in reversed(text.splitlines()):
        line = line.strip().strip("=").strip()
        m = SUMMARY_RE.match(line)
        if not m:
            continue
        seconds = float(m.group("secs"))
        for num, word in re.findall(r"(\d+) (\w+)", m.group("body")):
            if word == "warning" or word == "warnings":
                continue
            counts[word] = int(num)
        break
    passed = counts.get("passed", 0)
    failed = counts.get("failed", 0) + counts.get("error", 0) + counts.get("errors", 0)
    skipped = counts.get("skipped", 0)
    xfailed = counts.get("xfailed", 0)
    xpassed = counts.get("xpassed", 0)
    return {
        "log": str(path.relative_to(ROOT)),
        "counts": counts,
        "verbose_counts": verbose_counts(text),
        "failed_tests": failed_ids(text),
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "xfailed": xfailed,
        "xpassed": xpassed,
        "executed": passed + failed,
        "seconds": seconds,
        "summary_line": next(
            (ln.strip().strip("=").strip() for ln in reversed(text.splitlines())
             if SUMMARY_RE.match(ln.strip().strip("=").strip())),
            None,
        ),
    }


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def retention_figures(audit: dict) -> list[dict]:
    FIG.mkdir(parents=True, exist_ok=True)
    rows = [r for r in audit["per_file"] if not r.get("binary")]
    rows.sort(key=lambda r: -((r.get("main_added") or 0) + (r.get("branch_added") or 0)))
    top = rows[:16]
    labels = [r["path"] if len(r["path"]) <= 46 else "..." + r["path"][-43:] for r in top]
    main_added = [r.get("main_added") or 0 for r in top]
    branch_added = [r.get("branch_added") or 0 for r in top]
    main_kept = [r.get("main_kept") or 0 for r in top]
    branch_kept = [r.get("branch_kept") or 0 for r in top]

    fig, ax = plt.subplots(figsize=(9.2, 6.4), dpi=105)
    y = range(len(top))
    ax.barh([i + 0.2 for i in y], main_added, height=0.36, color="#3b6ea5", label="main: lines added vs merge base")
    ax.barh([i - 0.2 for i in y], branch_added, height=0.36, color="#c46a3f", label="agent/m3148: lines added vs merge base")
    ax.scatter([i + 0.2 for i in y], main_kept, color="black", s=14, zorder=5, label="kept in merge result")
    ax.scatter([i - 0.2 for i in y], branch_kept, color="black", s=14, zorder=5)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=6.6, family="monospace")
    ax.invert_yaxis()
    ax.set_xlabel("lines added by each side (dots = lines still present after the merge)")
    ax.set_title(
        "PR #56 two-sided retention audit: %d files, %d silently dropped lines"
        % (audit["checked_files"], audit["dropped_total"]),
        fontsize=10,
    )
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    p1 = FIG / "retention_audit.png"
    fig.savefig(p1)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=105)
    return [{"name": "retention_audit", "path": str(p1.relative_to(ROOT)), "fig": fig, "ax": ax}]


def test_parity_figure(base: dict, merged: dict) -> dict:
    FIG.mkdir(parents=True, exist_ok=True)
    cats = ["passed", "skipped", "xfailed", "xpassed", "failed"]
    b = [base[c] for c in cats]
    m = [merged[c] for c in cats]
    x = range(len(cats))
    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=105)
    ax.bar([i - 0.2 for i in x], b, width=0.38, color="#3b6ea5", label="main (7a8f4af) baseline")
    ax.bar([i + 0.2 for i in x], m, width=0.38, color="#c46a3f", label="PR #56 merge result")
    for i, (vb, vm) in enumerate(zip(b, m)):
        ax.text(i - 0.2, vb, str(vb), ha="center", va="bottom", fontsize=7)
        ax.text(i + 0.2, vm, str(vm), ha="center", va="bottom", fontsize=7)
        if vb:
            ax.text(i, max(vb, vm) * 1.06, "ratio %.4f" % (vm / vb), ha="center", fontsize=6.6, style="italic")
    ax.set_xticks(list(x))
    ax.set_xticklabels(cats)
    ax.set_ylabel("tests")
    ax.set_ylim(0, max(max(b), max(m)) * 1.18)
    ax.set_title("Core suite: merge result vs main baseline (no regressions allowed)", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    p = FIG / "test_parity.png"
    fig.savefig(p)
    plt.close(fig)
    return {"name": "test_parity", "path": str(p.relative_to(ROOT))}


def main() -> int:
    audit = json.loads((ART / "pr56-merge-audit.json").read_text(encoding="utf-8"))
    base = parse_pytest_log(ART / "pytest_main_base_v.log")
    merged = parse_pytest_log(ART / "pytest_merge_v.log")
    aborted = {
        "note": "first tally attempt (-q, xdist -n 16) aborted inside "
                "pytest_sessionfinish with OSError: cannot send (already closed?); "
                "no summary line was written, so the tallies above come from the "
                "repeat run with -v, which also yields per-test records",
        "logs": [str(p.relative_to(ROOT)) for p in
                 (ART / "pytest_merge.log", ART / "pytest_main_base.log")],
        "has_summary_merge": parse_pytest_log(ART / "pytest_merge.log")["summary_line"] is not None,
        "has_summary_base": parse_pytest_log(ART / "pytest_main_base.log")["summary_line"] is not None,
    }
    for side in (base, merged):
        summary = {k: side[k] for k in ("passed", "failed", "skipped", "xfailed", "xpassed")}
        vc = side["verbose_counts"]
        side["summary_matches_verbose"] = (
            vc.get("PASSED", 0) == summary["passed"]
            and vc.get("FAILED", 0) == summary["failed"]
            and vc.get("SKIPPED", 0) == summary["skipped"]
        )

    figs = retention_figures(audit)[0]
    plt.close("all")
    parity = test_parity_figure(base, merged)

    gate = {
        "no_silent_content_loss": audit["dropped_total"] == 0,
        "no_regression_vs_main": merged["failed"] <= base["failed"]
        and merged["passed"] >= base["passed"]
        and merged["failed_tests"] == base["failed_tests"],
        "summary_matches_verbose": base["summary_matches_verbose"] and merged["summary_matches_verbose"],
        "executed_not_below_baseline": merged["executed"] >= base["executed"],
    }

    evidence = {
        "mission": 3925,
        "repository": "TensorFEM",
        "pull_request": 56,
        "branch": "agent/m3148",
        "shas": {
            "merge_base": BASE_SHA,
            "main": MAIN_SHA,
            "branch_old": OLD_BRANCH_SHA,
            "branch_new": git("rev-parse", "HEAD"),
            "merge_commit_parents": [
                p for p in git("log", "-1", "--format=%P", "HEAD").split() if p
            ],
        },
        "conflict_files": [
            "CHANGELOG.md",
            "docs/benchmark-qualification-status.md",
        ],
        "retention_audit": {
            "checked_files": audit["checked_files"],
            "dropped_total": audit["dropped_total"],
            "superseded_claims": [
                {"path": r["path"], "side": s,
                 "sentences": r[f"{s}_sentence_audit"]["superseded_sentences"],
                 "reason": "see docs/pr56-merge-cure-m3925.md"}
                for r in audit["per_file"]
                for s in ("main", "branch")
                if r.get(f"{s}_sentence_audit")
                and r[f"{s}_sentence_audit"]["superseded_sentences"]
            ],
        },
        "tests": {"main_baseline": base, "merge_result": merged, "aborted_first_attempt": aborted},
        "failure_parity": base["failed_tests"] == merged["failed_tests"],
        "gate": gate,
        "figures": [figs["path"], parity["path"]],
    }
    out = ART / "pr56-merge-evidence.json"
    out.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"gate": gate, "tests": {"base": base["counts"], "merged": merged["counts"]},
                      "branch_new": evidence["shas"]["branch_new"]}, ensure_ascii=False, indent=2))
    return 0 if all(gate.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())