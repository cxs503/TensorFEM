#!/usr/bin/env python
"""Figures and machine-readable summary for the PR #77 merge resolution.

Everything plotted here is computed, never transcribed: the merge-side numbers
come from ``scripts/audit_changelog_merge.py`` run against the merge commit
itself, and the physics panel re-reads this branch's committed evidence file
(``benchmarks/plate_uniform_load_evidence.json``), which is the content the PR
carries.

Outputs
-------
``docs/assets/pr77_merge/merge_audit.png``     content preservation + structural audit
``docs/assets/pr77_merge/plate_gate.png``      branch's 3 % anchors, re-read from evidence
``benchmarks/pr77_merge_audit.json``           the same numbers, machine-readable
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets" / "pr77_merge"
EVIDENCE = ROOT / "benchmarks" / "plate_uniform_load_evidence.json"
OUT_JSON = ROOT / "benchmarks" / "pr77_merge_audit.json"


def auditor():
    spec = importlib.util.spec_from_file_location(
        "audit_changelog_merge", ROOT / "scripts" / "audit_changelog_merge.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git_show(rev: str, path: str = "CHANGELOG.md") -> list[str]:
    out = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=ROOT, check=True,
                         capture_output=True, text=True)
    return out.stdout.splitlines()


def main() -> dict:
    A = auditor()
    ASSETS.mkdir(parents=True, exist_ok=True)

    parents = subprocess.run(["git", "rev-list", "--parents", "-n", "1", "HEAD"],
                             cwd=ROOT, check=True, capture_output=True,
                             text=True).stdout.split()
    merge_sha, p_branch, p_main = parents[0], parents[1], parents[2]

    resolved = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8").splitlines()
    audit = {
        "merge_commit": merge_sha,
        "parents": {"branch": p_branch, "main": p_main},
        "resolved": A.structural_audit(resolved),
        "preservation": {
            side: A.loss_audit(resolved, git_show(rev), allow_dropped=("## Unreleased",))
            for side, rev in (("branch", p_branch), ("main", p_main))
        },
        "simulated_naive_merge": A.simulate_machine_merge(merge_sha),
    }

    # --- panel data: each side's own state, then the naive merge, then the result
    states = [
        ("branch\n" + p_branch[:7], A.structural_audit(git_show(p_branch)), git_show(p_branch)),
        ("main\n" + p_main[:7], A.structural_audit(git_show(p_main)), git_show(p_main)),
    ] + [("naive\ntextual merge", audit["simulated_naive_merge"]["structural"], None)] + [
        ("resolved\n" + merge_sha[:7], audit["resolved"], resolved)]

    labels = [s[0] for s in states]
    findings = [len(s[1]["findings"]) for s in states]
    colours = ["#b03030" if f else "#2a7a3a" for f in findings]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)

    ax = axes[0]
    x = np.arange(len(states))
    ax.bar(x, findings, color=colours, width=0.62)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("structural findings (0 = clean)")
    ax.set_title("Changelog layout audit per state", fontsize=10)
    for xi, (lab, (rep), _) in zip(x, states):
        ax.annotate(f"U:{rep['unreleased_headings']}  cook:{rep.get('cook_membrane_bullets')}",
                    (xi, len(rep["findings"])), ha="center", va="bottom", fontsize=7)
    ax.set_ylim(0, max(findings) + 1.6)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    ref_lines = {
        "branch": sum(1 for l in git_show(p_branch) if l.strip()),
        "main": sum(1 for l in git_show(p_main) if l.strip()),
        "resolved": sum(1 for l in resolved if l.strip()),
    }
    keep = sum(1 for l in resolved if l.strip())
    lost = {k: audit["preservation"][k]["lost_lines"] for k in ("branch", "main")}
    unexpected = {k: len(audit["preservation"][k]["unexpected_lost_lines"])
                  for k in ("branch", "main")}
    bars = ax.bar(["branch side\n(non-blank lines)", "main side\n(non-blank lines)",
                   "resolved file"], [ref_lines["branch"], ref_lines["main"], keep],
                  color=["#4a6fa5", "#4a6fa5", "#2a7a3a"], width=0.6)
    ax.bar_label(bars, fontsize=9)
    ax.set_ylim(0, max(ref_lines.values()) * 1.28)
    ax.set_ylabel("non-blank lines")
    ax.set_title("Content preservation against both merge parents", fontsize=10)
    ax.annotate(f"lost vs branch: {lost['branch']} "
                f"(duplicated heading, allow-listed)\n"
                f"unexpected lost vs branch: {unexpected['branch']}\n"
                f"lost vs main: {lost['main']} | unexpected: {unexpected['main']}",
                xy=(0.02, 0.97), xycoords="axes fraction", va="top", fontsize=8,
                bbox=dict(boxstyle="round", fc="#f2f2f2", ec="#888888"))
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("PR #77 merge resolution audit: textual merge preserves lines "
                 "but corrupts the document; the union resolution is clean", fontsize=10)
    f1 = ASSETS / "merge_audit.png"
    fig.savefig(f1, dpi=130)
    plt.close(fig)

    # --- panel 2: the physics the PR carries, re-read from committed evidence
    ev = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    anchors = [a for a in ev["anchors"] if not a["note"].startswith("control:")]
    errors = np.array([a["relative_error"] for a in anchors]) * 100.0
    tol = np.array([a["tolerance"] for a in anchors]) * 100.0
    order = np.argsort(-errors)
    names = [anchors[i]["anchor"] for i in order]

    fig, ax = plt.subplots(figsize=(10.5, 5.4), constrained_layout=True)
    y = np.arange(len(names))
    ax.barh(y, errors[order], color="#2a5d9f", height=0.62, label="relative error")
    ax.plot(tol[order], y, "r|", markersize=14, markeredgewidth=2,
            label="per-anchor gate (0.1 / 1 / 3 %, anchor-specific)")
    ax.set_yticks(y, [n.replace("_", " ")[:58] for n in names], fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlabel("relative error vs exact Navier series / separable Ritz  [%]")
    ax.set_title("Q4 Mindlin plate under uniform pressure: branch anchors "
                 f"(max error {errors.max():.4f} %, declared gate 3 %)", fontsize=10)
    for yi, ei in zip(y, errors[order]):
        ax.annotate(f"{ei:.4g}", (ei, yi), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=7)
    ax.axvline(3.0, color="red", ls="--", lw=1.2, alpha=0.8,
               label="declared 3 % gate")
    ax.grid(axis="x", alpha=0.25, which="both")
    ax.legend(fontsize=8, loc="lower right")
    f2 = ASSETS / "plate_gate.png"
    fig.savefig(f2, dpi=130)
    plt.close(fig)

    summary = {
        "benchmark": "pr77_merge_resolution_audit",
        "merge_commit": merge_sha,
        "branch_old_head": "c441d91",
        "main_merged": p_main,
        "resolved_changelog_lines": len(resolved),
        "resolved_changelog_nonblank": keep,
        "structural_audit": audit["resolved"],
        "preservation_vs_parents": audit["preservation"],
        "naive_textual_merge": {
            "conflict_markers": audit["simulated_naive_merge"]["conflict_markers"],
            "structural_findings": len(audit["simulated_naive_merge"]["structural"]["findings"]),
            "naive_merge_is_structurally_broken":
                audit["simulated_naive_merge"]["naive_merge_is_structurally_broken"],
        },
        "carried_physics": {
            "source": str(EVIDENCE.relative_to(ROOT)),
            "all_passed": ev["all_passed"],
            "anchors": len(anchors),
            "max_relative_error_percent": float(errors.max()),
            "declared_gate_percent": 3.0,
            "per_anchor_gate_percent": sorted({round(float(t), 2) for t in tol}),  # tol already in %
            "anchors_within_own_gate": bool(all(e < t for e, t in zip(errors, tol))),
            "within_3_percent": bool(errors.max() < 3.0),
            "parameters": ev["convergence"]["parameters"],
        },
        "figures": [str(f1.relative_to(ROOT)), str(f2.relative_to(ROOT))],
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in
                      ("merge_commit", "resolved_changelog_lines",
                       "naive_textual_merge", "carried_physics")}, indent=2))
    return summary


if __name__ == "__main__":
    main()