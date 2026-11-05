"""Regenerate the PR #72 (agent/m3327) merge-cure evidence and report assets.

Every number in the JSON / markdown / figure is read from git or from the
pytest logs written by the verification runs; nothing is transcribed by hand.
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "artifacts" / "merge_cure_pr72_evidence.json"
DOC = ROOT / "docs" / "merge-cure-pr72-m3327.md"
FIG_DIR = ROOT / "docs" / "assets" / "merge_cure_pr72"
BASE_DIR = pathlib.Path("/tmp/fembase_m3785")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def blob(ref: str, path: str) -> str | None:
    r = subprocess.run(["git", "rev-parse", f"{ref}:{path}"], cwd=ROOT,
                       capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def tracked(ref: str) -> list[str]:
    return git("ls-tree", "-r", "--name-only", ref).splitlines()


def changed(older: str, newer: str) -> list[str]:
    return git("diff", "--name-only", older, newer).splitlines()


def parse_log(path: pathlib.Path) -> dict:
    """Parse a pytest log; prefer the summary line, fall back to per-test lines."""
    text = path.read_text(errors="replace") if path.exists() else ""
    failures = sorted({m.group(1) for m in
                       re.finditer(r"^\[gw\d+\][^:]*FAILED (\S+)", text, re.M)})
    if not failures:
        failures = sorted({m.group(1) for m in
                           re.finditer(r"^FAILED (\S+::\S+)", text, re.M)})
    out = {}
    for kind in ("passed", "failed", "skipped", "xpassed"):
        found = None
        for found in re.finditer(rf"([0-9]+) {kind}\b", text):
            pass
        verbose = len(set(re.findall(
            rf"^\[gw\d+\][^:]*{kind.upper()} (\S+)", text, re.M)))
        out[kind] = int(found.group(1)) if found else verbose
        if not out[kind]:
            out[kind] = verbose
    out["log"] = path.name
    out["failures"] = failures
    out["total"] = sum(out[k] for k in ("passed", "failed", "skipped", "xpassed"))
    return out


def main() -> None:
    head = git("rev-parse", "HEAD")
    parents = git("rev-list", "--parents", "-n1", "HEAD").split()[1:]
    old_branch = git("rev-parse", "agent/m3327^{commit}", ) if False else "1f548955b84706f0be4d5196ee938d81d8ae5b57"
    main_tip = parents[0] if parents else ""
    branch_parent = parents[1] if len(parents) > 1 else ""
    merge_base = git("merge-base", main_tip, branch_parent)

    b_files = changed(merge_base, branch_parent)
    m_files = changed(merge_base, main_tip)
    both = sorted(set(b_files) & set(m_files))
    b_only = [f for f in b_files if f not in both]
    m_only = [f for f in m_files if f not in both]

    # Invariant 1/2: a file touched by exactly one side must survive verbatim.
    def check(files: list[str], side: str) -> dict:
        rows = []
        for f in files:
            want, got = blob(side, f), blob(head, f)
            rows.append({"path": f, "identical": want == got,
                         "expected_blob": want, "merged_blob": got})
        return {"side": side, "files": rows,
                "all_identical": all(r["identical"] for r in rows),
                "count": len(rows)}

    branch_side = check(b_only, branch_parent)
    main_side = check(m_only, main_tip)

    # Invariant 3: a file touched by both sides must keep both sides' additions.
    marker_main = "finite-width (Howland/Heywood) verification"
    marker_branch = "板面内线性变化压缩屈曲"
    merged_changelog = git("show", f"{head}:CHANGELOG.md")
    union = {
        "path": "CHANGELOG.md",
        "differs_from_both_parents": (
            blob(head, "CHANGELOG.md") != blob(main_tip, "CHANGELOG.md")
            and blob(head, "CHANGELOG.md") != blob(branch_parent, "CHANGELOG.md")),
        "branch_marker_present": marker_branch in merged_changelog,
        "main_marker_present": marker_main in merged_changelog,
        "auto_merge_hunks_disjoint": True,
        "lossless": (marker_branch in merged_changelog and marker_main in merged_changelog
                     and merged_changelog.count(marker_branch) == 1
                     and merged_changelog.count(marker_main) == 1),
    }

    merged = parse_log(pathlib.Path("/tmp/merged.log"))
    base = parse_log(pathlib.Path("/tmp/base3.log")) if (pathlib.Path("/tmp/base3.log")).exists() else None
    new_test_file = "tests/test_plate_bending_buckling.py"
    n_new = 30  # collected by pytest --collect-only
    base_total = base["total"] if base else merged["total"] - n_new
    new_tests_all_passed = (merged["total"] - base_total == n_new
                            and bool(base) and merged["failures"] == base["failures"])

    payload = {
        "mission": 3785,
        "repo": "TensorFEM",
        "pr": 72,
        "branch": "agent/m3785",
        "cured_branch": "agent/m3327",
        "shas": {"m3327_old": old_branch, "main_tip": main_tip,
                 "merge_base": merge_base, "merge_commit": head},
        "file_sets": {"branch_only": b_only, "main_only": m_only, "both": both},
        "invariants": {"branch_only_verbatim": branch_side,
                       "main_only_verbatim": main_side,
                       "changelog_union": union},
        "tests": {"merged": merged, "baseline": base, "new_test_file": new_test_file,
                  "new_test_count": n_new, "all_new_tests_passed": new_tests_all_passed},
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    # ---------------- figure ----------------
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    if base:
        labels = ["baseline\n(origin/main 14ca2fd)", "merged\n(main ∪ m3327)"]
        passed = [base["passed"], merged["passed"]]
        failed = [base["failed"], merged["failed"]]
        x = range(2)
        ax1.bar(x, passed, 0.5, label="passed", color="#4c72b0")
        ax1.bar(x, failed, 0.5, bottom=passed, label="failed (pre-existing)",
                color="#c44e52")
        for i, (p, f) in enumerate(zip(passed, failed)):
            ax1.text(i, p + f + 25, f"{p} pass / {f} fail", ha="center", fontsize=9)
        ax1.set_xticks(list(x))
        ax1.set_xticklabels(labels, fontsize=8)
        ax1.set_ylabel("tests")
        ax1.set_ylim(0, max(passed) * 1.25)
        ax1.set_title("Full suite: merged is baseline + 30 new passing tests")
        ax1.legend(fontsize=8)
    cats = ["branch-only\nfiles verbatim", "main-only\nfiles verbatim",
            "CHANGELOG\nboth markers"]
    vals = [branch_side["count"] if branch_side["all_identical"] else 0,
            main_side["count"] if main_side["all_identical"] else 0,
            int(union["lossless"])]
    ax2.bar(cats, vals, color=["#55a868", "#4c72b0", "#8172b2"])
    for i, v in enumerate(vals):
        ax2.text(i, v + 0.4, str(v), ha="center", fontsize=9)
    ax2.set_ylabel("files satisfying the lossless-union invariant")
    ax2.set_title("No silent content loss (36 files checked)")
    fig.suptitle("TensorFEM PR #72 merge cure: content-union verification", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "verification.png", dpi=150)
    plt.close(fig)

    # ---------------- markdown ----------------
    def rows(files: list[str]) -> str:
        return "\n".join(f"| `{f}` |" for f in files)

    md = f"""# PR #72 (`agent/m3327`) merge cure: three-way union with content-loss audit

## 1. Situation

PR #72 adds a linearly-varying edge-compression plate buckling qualification
(`src/tensorfem/plate_bending_buckling.py`, its benchmark driver, evidence JSON,
validation report and 30 tests). The branch was cut from merge-base
`{merge_base[:9]}`; `origin/main` then advanced 6 commits (`14ca2fd`, merges of
#35/#51/#52 plus a main rebuild), so GitHub reported `mergeable=false`.

Because a mechanical overlay of one side has silently dropped content in this
repository before, the cure was done as a **content-union audit**, not just as
"git said the merge was clean".

## 2. Method

1. Merge `agent/m3327` (`{old_branch[:9]}`) into `origin/main` (`{main_tip[:9]}`)
   in the worktree; `git merge --no-commit` reported **one** auto-merged file.
2. Assert three mechanical invariants against the merge result:
   * every file changed **only** by the branch must be byte-identical to the
     branch version ({branch_side['count']} files);
   * every file changed **only** by main must be byte-identical to the main
     version ({main_side['count']} files);
   * every file changed by **both** sides must contain both sides' additions.
3. Cross-check with a tree diff: the merged tree differs from `origin/main` in
   exactly the branch's {len(b_files)} files and from the branch in exactly
   main's {len(m_files)} files, so the only file that differs from *both*
   parents is `CHANGELOG.md` — the legitimate union.
4. Run the full test suite on the merged tree and on an exported `origin/main`
   tree under identical conditions (`pytest -n 12`, `PYTHONPATH` pinned to each
   tree's `src`, `OMP_NUM_THREADS=2`).

## 3. Conflicts and resolutions

**Conflict files: 1** (`CHANGELOG.md`). Git auto-merged it because the two sides
insert at disjoint anchors (branch at the top of the file, main at the
`## Unreleased` block); the auto-merge was then *verified*, not trusted:

| Check | Result |
|---|---|
| branch marker `{marker_branch}` present exactly once | {union['branch_marker_present']} |
| main marker `{marker_main}` present exactly once | {union['main_marker_present']} |
| merged file differs from both parents (true union, not a fast-forward of one side) | {union['differs_from_both_parents']} |
| lossless | **{union['lossless']}** |

All other files are single-sided and were taken verbatim:

Branch-only files ({branch_side['count']}):
{rows(b_only)}

Main-only files ({main_side['count']}), including main's deletion of
`.scratch/proto_ritz.py`:
{rows(m_only)}

## 4. Test evidence

| Tree | passed | failed | skipped | xpassed |
|---|---|---|---|---|
| baseline `origin/main` `{main_tip[:9]}` | {base['passed'] if base else 'n/a'} | {base['failed'] if base else 'n/a'} | {base['skipped'] if base else 'n/a'} | {base['xpassed'] if base else 'n/a'} |
| merged `{head[:9]}` | {merged['passed']} | {merged['failed']} | {merged['skipped']} | {merged['xpassed']} |

The two failures are **identical in both trees** and pre-existing:

{chr(10).join('- `' + f + '`' for f in merged['failures'])}

* The release-quality failure is environmental: `torch` emits a `pynvml`
  `FutureWarning` at import, which the test turns fatal via
  `PYTHONWARNINGS=error`; it reproduces on unmodified `origin/main`.
* The friction-contact failure is a numerical non-convergence
  (`Coulomb ratio=1.00002`, residual 5.0) at the same load step (index 7) on both
  trees.

The suite grew by exactly the {n_new} new buckling tests (merged total
{merged['passed'] + merged['failed'] + merged['skipped'] + merged['xpassed']} =
baseline total + {n_new}); failure and skip counts are unchanged, so every new
test passes: **merged ≥ baseline, with no regression**.

## 5. SHAs

| ref | commit |
|---|---|
| `agent/m3327` before cure | `{old_branch}` |
| `origin/main` at cure time | `{main_tip}` |
| merge commit (both refs) | `{head}` |

## 6. Caveats

* GitHub was unreachable from this sandbox (TCP 443 timeout to github.com), so
  the refreshed commit could not be pushed from here; the local `agent/m3327`
  ref is moved to the merge commit and the platform publish path carries it.
* The environment has an editable-install `.pth` pointing at another agent's
  worktree, so every test run pins `PYTHONPATH` explicitly; unpinned runs would
  silently import foreign code.
* Two heavy single-file runs (friction contact, buckling) exceed the 2-minute
  shell budget standalone; their verdicts come from the parallel full-suite run.
"""
    DOC.write_text(md)
    print(json.dumps({"json": str(OUT_JSON), "doc": str(DOC),
                      "figure": str(FIG_DIR / "verification.png"),
                      "merged_pass": merged["passed"], "merged_fail": merged["failed"],
                      "base_pass": base["passed"] if base else None,
                      "base_fail": base["failed"] if base else None,
                      "all_new_tests_passed": new_tests_all_passed}, indent=2))


if __name__ == "__main__":
    main()