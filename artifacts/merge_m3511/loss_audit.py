"""Lossless-merge audit for agent/m3511 <- main.

Two independent checks per file touched by *both* sides:

A. No silent content loss: every non-blank line a side added relative to the merge base is
   either present verbatim in the merged worktree file, or its content survives in a
   reworded form -- machine-checked as: all link targets, all ``code`` spans and all
   numeric tokens present, and >=80% of the CJK bigrams of the added line found in the
   merged file (word-level rewording allowed, content still counted).
B. No silent resurrection: if side X deleted a base line that side Y kept unchanged, that
   line must be *absent* from the merge result (otherwise X's deletion was lost).
"""

import re
import subprocess
import sys
from pathlib import Path

WT = Path("/root/tp-agent/work/m4018")
BASE, OURS, THEIRS = "14ca2fdc1fd06467f3edb33db1d9e549b6a901a2", "origin/agent/m3511", "origin/main"

LINK = re.compile(r"\]\(([^)\s]+)\)")
CODE = re.compile(r"`([^`]+)`")
NUM = re.compile(r"\d+(?:[.,]\d+)?(?:[e×−\-]\d+)*%?")
CJK = re.compile(r"[\u4e00-\u9fff]")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=WT, capture_output=True, text=True, check=True).stdout


def changed(ref: str) -> set[str]:
    return {ln for ln in git("diff", "--name-only", f"{BASE}..{ref}").splitlines() if ln}


def added_lines(ref: str, path: str) -> list[str]:
    out = git("diff", "-U0", f"{BASE}..{ref}", "--", path)
    return [ln[1:] for ln in out.splitlines() if ln.startswith("+") and not ln.startswith("+++")]


def side_lines(ref: str, path: str) -> list[str]:
    try:
        return git("show", f"{ref}:{path}").splitlines()
    except subprocess.CalledProcessError:
        return []


def bigrams(text: str) -> set[str]:
    chars = [c for c in text if CJK.match(c)]
    return {a + b for a, b in zip(chars, chars[1:])}


def survives(line: str, merged_text: str, merged_bigrams: set[str]) -> tuple[bool, str]:
    if line in merged_text:
        return True, ""
    missing = []
    for token in LINK.findall(line) + CODE.findall(line) + NUM.findall(line):
        if token not in merged_text:
            missing.append(token)
    if missing:
        return False, f"tokens {missing[:4]}"
    grams = bigrams(line)
    if grams:
        cover = len(grams & merged_bigrams) / len(grams)
        if cover < 0.8:
            return False, f"CJK coverage {cover:.2f}"
    return True, "reworded"


def main() -> int:
    overlap = sorted(changed(OURS) & changed(THEIRS))
    print(f"files changed by both sides: {len(overlap)} -> {', '.join(overlap)}\n")
    fails = 0
    for path in overlap:
        merged_text = (WT / path).read_text(encoding="utf-8", errors="replace")
        merged_bigrams = bigrams(merged_text)
        merged_lines = set(merged_text.splitlines())
        print(f"[{path}]")
        for ref, label in ((OURS, "branch m3511"), (THEIRS, "main")):
            rew, total, lost = 0, 0, []
            for line in added_lines(ref, path):
                if not line.strip():
                    continue
                total += 1
                ok, why = survives(line, merged_text, merged_bigrams)
                if ok:
                    rew += why == "reworded"
                else:
                    lost.append((line, why))
            print(f"  additions from {label}: {total} checked, {rew} reworded, {len(lost)} lost")
            for line, why in lost[:6]:
                print(f"    LOST ({why}): {line.strip()[:110]}")
            fails += len(lost)
        base = side_lines(BASE, path)
        for ref, other, label in ((OURS, THEIRS, "branch"), (THEIRS, OURS, "main")):
            dropped = [ln for ln in base if ln.strip() and ln not in set(side_lines(ref, path))]
            kept_by_other = set(side_lines(other, path))
            for line in dropped:
                if line in kept_by_other and line in merged_lines:
                    print(f"    RESURRECTED deletion by {label}: {line.strip()[:110]}")
                    fails += 1
    print(
        "\nRESULT:",
        "FAIL" if fails else "PASS (no added content lost, no dropped line resurrected)",
    )
    return 1 if fails else 0


sys.exit(main())
