#!/usr/bin/env python3
"""Run the PR #92 core-test set on the merged tree and on the main baseline.

Writes artifacts/merge_cure_pr92_tests.json with machine-parsed pytest counts
(no hand-copied numbers).  Invoked as:
    python3 scripts/merge_cure_pr92_tests.py --out artifacts/merge_cure_pr92_tests.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

MERGED_TREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SETS = {
    "element_rank_audit": ["tests/test_element_rank_audit.py"],
    "hex20_distortion": ["tests/test_hex20_distortion.py"],
    "broad_hex20_rank_regression": [
        "tests/test_hex20.py",
        "tests/test_quadratic_solid_rank_audit.py",
        "tests/test_hemisphere_hex20_oracle.py",
        "tests/test_solid3d_mass.py",
    ],
}
COUNT_RE = re.compile(r"(\d+) passed(?:, (\d+) failed)?(?:, (\d+) skipped)?")


def run(tree: str, paths: list[str]) -> dict:
    env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONPATH=os.path.join(tree, "src"))
    r = subprocess.run(
        [sys.executable, "-m", "pytest", *paths, "-q"],
        cwd=tree,
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    tail = [l for l in r.stdout.splitlines() if COUNT_RE.search(l)]
    m = COUNT_RE.search(tail[-1]) if tail else None
    return {
        "returncode": r.returncode,
        "passed": int(m.group(1)) if m else None,
        "failed": int(m.group(2) or 0) if m else None,
        "skipped": int(m.group(3) or 0) if m else None,
        "summary_line": tail[-1].strip() if tail else r.stdout.strip()[-300:],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="/tmp/m3545_main")
    ap.add_argument("--out", default=os.path.join(MERGED_TREE, "artifacts/merge_cure_pr92_tests.json"))
    ap.add_argument("--only", default=None,
                    help="comma-separated subset of SETS (keeps each call under a shell timeout)")
    a = ap.parse_args()

    out_path = a.out
    out: dict = {"merged_tree": MERGED_TREE, "sets": {}}
    if a.only and os.path.exists(out_path):  # merge into existing evidence
        with open(out_path, encoding="utf-8") as fh:
            out = json.load(fh)
    ok = True
    selected = {k: v for k, v in SETS.items() if not a.only or k in a.only.split(",")}
    for name, paths in selected.items():
        merged = run(MERGED_TREE, paths)
        entry = {"paths": paths, "merged": merged, "baseline": None, "baseline_comparable": None}
        # the branch-only test file has no main baseline; note it explicitly
        if os.path.exists(os.path.join(a.baseline, paths[0])):
            base = run(a.baseline, paths)
            entry["baseline"] = base
            entry["baseline_comparable"] = {
                "merged_passed": merged["passed"],
                "baseline_passed": base["passed"],
                "merged_ge_baseline": (merged["passed"] or 0) >= (base["passed"] or 0),
            }
            ok &= bool(entry["baseline_comparable"]["merged_ge_baseline"])
        else:
            entry["baseline_note"] = "test file is branch-only (absent on main); baseline N/A"
        ok &= merged["returncode"] == 0
        out["sets"][name] = entry

    out["verdict"] = "PASS" if ok and len(out["sets"]) == len(SETS) else "PARTIAL" if not ok else "PASS"
    out["sets_incomplete"] = sorted(set(SETS) - set(out["sets"]))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    for name, e in out["sets"].items():
        print(name, "merged:", e["merged"]["summary_line"], "| baseline:", (e["baseline"] or {}).get("summary_line"))
    print("verdict:", out["verdict"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())