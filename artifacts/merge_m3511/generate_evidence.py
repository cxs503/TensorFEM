"""Build the PR #89 merge evidence: parse pytest logs, emit JSON + markdown + figure.

All numbers in the generated report are parsed from artifacts/merge_m3511/*.log and from
the loss-audit script output, so nothing is transcribed by hand.
"""

import json
import re
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
WT = HERE.parent.parent


def passed(log: Path) -> int:
    text = log.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"(\d+) passed", text)
    if not m:
        raise SystemExit(f"no pass count in {log}")
    if "failed" in text or "error" in text.lower().split("passed")[-1]:
        raise SystemExit(f"non-clean log: {log}")
    return int(m.group(1))


GROUPS = {
    "A_rank_quadrature_core": {
        "files": [
            "tests/test_element_rank_audit.py",
            "tests/test_hex20.py",
            "tests/test_quadratic_solid.py",
            "tests/test_quadratic_solid_rank_audit.py",
            "tests/test_quadratic_plane.py",
        ],
        "merged": HERE / "group_a_merged.log",
        "baseline": HERE / "group_a_main_baseline.log",
        "branch_only": ["tests/test_hex20_distortion.py"],
    },
    "B_solid3d_bbar_locking": {
        "files": [
            "tests/test_solid3d.py",
            "tests/test_continuum_bbar.py",
            "tests/test_incompressible_locking.py",
        ],
        "merged": HERE / "group_b_merged.log",
        "baseline": HERE / "group_b_main_baseline.log",
        "branch_only": [],
    },
}

LOSS_LOG = HERE / "loss_audit.txt"


def parse_loss() -> dict:
    text = LOSS_LOG.read_text(encoding="utf-8")
    files, checked, reworded, lost = {}, 0, 0, 0
    current = None
    for line in text.splitlines():
        m = re.match(r"^\[(.+)\]$", line.strip())
        if m:
            current = m.group(1)
            files[current] = {"checked": 0, "reworded": 0, "lost": 0}
            continue
        m = re.match(r"\s+additions from .*: (\d+) checked, (\d+) reworded, (\d+) lost", line)
        if m and current:
            c, r, l = map(int, m.groups())
            checked += c
            reworded += r
            lost += l
            files[current]["checked"] += c
            files[current]["reworded"] += r
            files[current]["lost"] += l
    if lost:
        raise SystemExit("loss audit reports lost lines")
    if "PASS" not in text:
        raise SystemExit("loss audit did not pass")
    return {"files": files, "checked": checked, "reworded": reworded, "lost": lost}


def sha(ref: str) -> str:
    return subprocess.run(["git", "rev-parse", ref], cwd=WT, capture_output=True, text=True, check=True).stdout.strip()


def main() -> None:
    loss = parse_loss()
    groups = {}
    for name, spec in GROUPS.items():
        groups[name] = {
            "files": spec["files"],
            "branch_only_files": spec["branch_only"],
            "baseline_passed": passed(spec["baseline"]),
            "merged_passed": passed(spec["merged"]),
        }
    base_total = sum(g["baseline_passed"] for g in groups.values())
    merged_total = sum(g["merged_passed"] for g in groups.values())
    branch_only_total = passed(HERE / "branch_only_hex20_distortion.log")
    data = {
        "pr": 89,
        "branch": "agent/m3511",
        "old_branch_sha": "f9bd976b241ff797bc04759c8b35276676f41aa8",
        "merge_commit_sha": sha("HEAD"),
        "merged_onto_sha": sha("HEAD^1"),
        "main_sha": sha("HEAD^2"),
        "groups": groups,
        "totals": {
            "baseline_passed": base_total,
            "merged_passed": merged_total,
            "shared_tests_passed_after_merge": merged_total - branch_only_total,
            "shared_tests_regression": merged_total - branch_only_total - base_total,
        },
        "loss_audit": loss,
    }
    data["totals"]["branch_only_tests"] = branch_only_total
    (HERE / "merge_evidence.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # figure: (a) baseline vs merged tests, (b) per-file line preservation
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    labels = ["A: rank/quadrature\n(shared)", "A: branch-only\nhex20 distortion", "B: solid3D/B-bar/\nlocking (shared)"]
    base = [groups["A_rank_quadrature_core"]["baseline_passed"], 0, groups["B_solid3d_bbar_locking"]["baseline_passed"]]
    merged_only = [0, branch_only_total, 0]
    extra = [
        groups["A_rank_quadrature_core"]["merged_passed"] - groups["A_rank_quadrature_core"]["baseline_passed"] - branch_only_total,
        0,
        groups["B_solid3d_bbar_locking"]["merged_passed"] - groups["B_solid3d_bbar_locking"]["baseline_passed"],
    ]
    x = range(len(labels))
    axes[0].bar(x, base, label="main baseline", color="#4c72b0")
    axes[0].bar(x, merged_only, bottom=base, label="new tests from #89", color="#55a868")
    axes[0].bar(x, extra, bottom=[b + m for b, m in zip(base, merged_only)], label="delta (0 expected)", color="#c44e52")
    axes[0].set_xticks(list(x), labels, fontsize=8)
    axes[0].set_ylabel("pytest tests passed")
    axes[0].set_title(f"tests: merged {merged_total} vs baseline {base_total}")
    axes[0].legend(fontsize=8)
    for i, (b, m, e) in enumerate(zip(base, merged_only, extra)):
        axes[0].text(i, b + m + e + 1.0, f"{b + m + e}", ha="center", fontsize=9)

    files = list(loss["files"])
    kept = [loss["files"][f]["checked"] for f in files]
    lost = [loss["files"][f]["lost"] for f in files]
    short = [f.split("/")[-1] for f in files]
    y = range(len(files))
    axes[1].barh(list(y), kept, color="#4c72b0", label="added lines preserved")
    axes[1].barh(list(y), lost, color="#c44e52", label="added lines lost")
    axes[1].set_yticks(list(y), short, fontsize=8)
    axes[1].set_xlabel("non-blank lines added by either side")
    axes[1].set_title(f"lossless-union audit: {loss['checked']} lines kept, {loss['lost']} lost")
    axes[1].legend(fontsize=8)
    for i, k in enumerate(kept):
        axes[1].text(k + 3, i, str(k), va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "merge_verification.png", dpi=140)
    print(json.dumps({k: data[k] for k in ("old_branch_sha", "merge_commit_sha", "totals")}, ensure_ascii=False, indent=2))
    print("loss:", loss["checked"], "checked /", loss["lost"], "lost")
    print("figure:", HERE / "merge_verification.png")


main()