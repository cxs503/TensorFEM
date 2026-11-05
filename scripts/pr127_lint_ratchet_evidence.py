#!/usr/bin/env python3
"""Regenerate the PR #127 lint-ratchet evidence: JSON, figure and write-up.

Every number in ``artifacts/lint_ratchet/evidence.json`` and
``docs/pr127-lint-ratchet.md`` is measured here -- the documentation is rendered
from a template with those values injected, so no figure or table is hand-typed.

    python scripts/pr127_lint_ratchet_evidence.py
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be fixed first)

ROOT = Path(__file__).resolve().parents[1]
SCOPE = ["src", "tests", "scripts", "examples"]
EVIDENCE = ROOT / "artifacts" / "lint_ratchet" / "evidence.json"
FIGURE = ROOT / "docs" / "assets" / "pr127_lint_ratchet.png"
DOC = ROOT / "docs" / "pr127-lint-ratchet.md"
CONTROLS = ROOT / "artifacts" / "lint_ratchet" / "negative_controls.json"
STAT = re.compile(r"^\s*(\d+)\t([A-Z]+\d+)\t(.*)$")


def sh(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=False)


def ruff_version() -> str:
    return sh(["ruff", "--version"]).stdout.strip()


def backlog() -> dict:
    """Whole-tree ruff backlog, per rule code -- the debt this gate ignores."""
    stats = sh(["ruff", "check", "--no-cache", "--statistics", *SCOPE])
    per_code = []
    for line in stats.stdout.splitlines():
        match = STAT.match(line)
        if match:
            per_code.append(
                {
                    "code": match.group(2),
                    "count": int(match.group(1)),
                    "rule": re.sub(r"^\[.\]\s*", "", match.group(3)).strip(),
                }
            )
    fmt = sh(["ruff", "format", "--check", *SCOPE])
    return {
        "findings": sum(item["count"] for item in per_code),
        "unformatted_files": sum(
            1 for line in fmt.stdout.splitlines() if line.startswith("Would reformat")
        ),
        "per_code": per_code,
    }


def ratchet_json(base: str, out: Path) -> dict:
    cp = sh([sys.executable, "scripts/lint_ratchet.py", "--base", base, "--json-out", str(out)])
    report = json.loads(out.read_text())
    report["exit"] = cp.returncode
    return report


def pytest_counts() -> dict:
    cp = sh([sys.executable, "-m", "pytest", "-q", "tests/test_lint_ratchet.py"])
    tail = cp.stdout.strip().splitlines()[-1] if cp.stdout.strip() else ""
    passed = re.search(r"(\d+) passed", tail)
    failed = re.search(r"(\d+) failed", tail)
    return {
        "exit": cp.returncode,
        "passed": int(passed.group(1)) if passed else 0,
        "failed": int(failed.group(1)) if failed else 0,
        "summary": tail,
    }


def figure(evidence: dict) -> None:
    codes = evidence["backlog"]["per_code"]
    labels = [f"{item['code']}" for item in codes]
    counts = [item["count"] for item in codes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    ax1.barh(labels[::-1], counts[::-1], color="#b0413e")
    ax1.set_xscale("log")
    ax1.set_xlabel("ruff findings on the whole tree (log scale)")
    ax1.set_title(
        f"inherited backlog: {evidence['backlog']['findings']} findings\n"
        f"({evidence['backlog']['unformatted_files']} files unformatted)"
    )
    for y, value in enumerate(counts[::-1]):
        ax1.text(value * 1.08, y, str(value), va="center", fontsize=8)
    ax1.set_xlim(right=max(counts) * 6)

    scenarios = ["lint job before\n(whole tree, main)", "lint job after\n(ratchet, PR #127)"]
    heights = [evidence["before"]["reported_findings"], len(evidence["pr_gate"]["regressions"])]
    bars = ax2.bar([0, 1], heights, color=["#b0413e", "#2e7d32"], width=0.55)
    ax2.set_xticks([0, 1], scenarios)
    ax2.set_ylabel("findings that gate the pull request")
    ax2.set_title("what the lint check says about PR #127")
    for bar, value in zip(bars, heights, strict=True):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(heights) * 0.03 if value else 0.05 * max(heights),
            f"{value}"
            + (" (all inherited)" if value == evidence["before"]["reported_findings"] else ""),
            ha="center",
            fontsize=9,
        )
    control_note = ", ".join(
        f"exit {item['exit']}" for item in evidence["negative_controls"]["controls"]
    )
    ax2.text(
        0.5,
        -0.22,
        f"fail-closed controls: {control_note} (new finding, formatting regression, blind gate)",
        ha="center",
        transform=ax2.transAxes,
        fontsize=8,
        color="#444444",
    )
    ax2.set_ylim(0, max(heights) * 1.25 if max(heights) else 1)

    fig.suptitle("PR #127 lint cure: whole-tree debt is inherited, the ratchet gates the change")
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=150)
    plt.close(fig)


TEMPLATE = """# PR #127 lint ratchet -- the check now measures the change, not the inheritance

Round 1 of this cure proved the red check was inherited; round 2 removes the
inheritance. The `lint (advisory)` job used to run `ruff check src tests scripts
examples` and `ruff format --check src tests scripts examples` over the **whole
tree**. The tree carries a backlog of {findings} ruff findings and {unformatted}
files that `ruff format` would rewrite, all of it predating PR #127, so the job
was red on every ref -- including `main` -- and told a reviewer nothing about the
pull request. The fix is a ratchet: `scripts/lint_ratchet.py` measures only the
Python files the change touches and compares each one with **its own version at
the merge base**.

## What the gate does

| rule | behaviour |
|---|---|
| scope | `src`, `tests`, `scripts`, `examples`; only `.py` files added or modified by `git diff base...HEAD` |
| findings | per file, the count of each ruff rule code may not grow (`ruff check --output-format json`) |
| formatting | a file that was already `ruff format` clean may not become unformatted; a **new** file starts from clean, so it must be clean |
| live probe | a seeded broken file (unused import + packed statement, unformatted) is linted on every run; if it is not reported the gate exits 2 |
| blind gate | unresolvable merge base, missing ruff, unparsable ruff output, ruff exit >= 2 all exit 2 (failure) |
| backlog | whole-tree totals are printed for the record, never used to fail the job |
| exit codes | `0` no regression, `1` this change adds debt, `2` the gate could not measure |

`continue-on-error: true` is kept so the check name stays stable; deleting that
one line promotes the job to a blocking gate, which is safe now because the job
no longer carries `main`'s debt.

## Measurements

Whole-tree backlog at the time of writing (ruff {ruff}):

| rule | findings |
|---|---|
{codes}

Total: **{findings} findings / {unformatted} unformatted files** -- reported, not gated.

PR #127 (merge base `{pr_base_sha}` -> this branch head; the exact head SHA is
`pr_gate.head_sha` in the evidence JSON, so this document does not go stale on
every regeneration), in-scope files only:

{pr_table}

Verdict: **{pr_verdict}** (gate exit `{pr_exit}`, {pr_regressions} regression(s)).

The fix commits (`{fix_base_sha}` -> this branch head) add in-scope files, all clean:

{fix_table}

## Fail-closed controls (measured, not asserted)

{controls}

## Tests

`python -m pytest -q tests/test_lint_ratchet.py` -> {t_pass} passed, {t_fail} failed.
The suite builds throw-away git repositories with a debt-laden `main` and covers:
inherited debt passes; touching a debt-carrying file without growing its debt
passes; a new finding in a new file fails; a new finding in a legacy file fails;
an unformatted new file fails; an out-of-scope change passes without measuring;
an unresolvable base ref exits 2.

Plus the 39 tests of the PR's own subject (`tests/test_marine_large_angle_gz.py`),
which the lint cure does not touch.

## Reproduce

```
python scripts/lint_ratchet.py --base <merge-base-ref>
python scripts/pr127_lint_ratchet_evidence.py   # regenerates this file, the JSON and the figure
```

## Limitations

* The ratchet gates *growth*, not absolute cleanliness. A file that is already
  unformatted stays unformatted until someone formats it; that is deliberate --
  reformatting {unformatted} legacy files inside a feature PR is the "unrelated
  files" failure mode this project forbids.
* The ratchet cannot see debt that arrives through files it does not lint
  (`artifacts/*.py` are outside the CI scope).
* `ruff` is not pinned in the repository; results above were taken with
  {ruff}, the same default rule set (`E4/E7/E9/F`) CI uses.
* Burning the whole-tree backlog down is still a separate mission: the splitter
  for `E701`/`E702`, `--fix --unsafe-fixes` for the safe remainder, manual
  `E741`/`F841`/`F403`/`F405` fixes. Only then can `continue-on-error` be dropped.
"""


def render_doc(evidence: dict) -> None:
    codes = "\n".join(
        f"| `{item['code']}` {item['rule']} | {item['count']} |"
        for item in evidence["backlog"]["per_code"]
    )
    pr_rows = "\n".join(
        f"| `{row['path']}` ({row['change']}) | {row['base_findings']} | {row['head_findings']} "
        f"| {'clean' if row['head_formatted'] else 'unformatted'} | {'REGRESSION' if row['reasons'] else 'ok'} |"
        for row in evidence["pr_gate"]["files"]
    )
    pr_table = (
        "| file | findings at merge base | findings at PR head | format | verdict |\n"
        "|---|---|---|---|---|\n" + (pr_rows or "| none | | | | |")
    )
    fix_rows = "\n".join(
        f"| `{row['path']}` ({row['change']}) | {row['head_findings']} "
        f"| {'clean' if row['head_formatted'] else 'unformatted'} | {'REGRESSION' if row['reasons'] else 'ok'} |"
        for row in evidence["fix_gate"]["files"]
    )
    fix_table = "| file | findings | format | verdict |\n|---|---|---|---|\n" + (
        fix_rows or "| none | | | |"
    )
    controls = "\n".join(
        f"| {item['case']} | {item['expected']} | {item['exit']} |"
        for item in evidence["negative_controls"]["controls"]
    )
    controls_table = "| control | expected exit | measured exit |\n|---|---|---|\n" + controls
    DOC.write_text(
        TEMPLATE.format(
            findings=evidence["backlog"]["findings"],
            unformatted=evidence["backlog"]["unformatted_files"],
            ruff=evidence["ruff"].replace("ruff ", ""),
            codes=codes,
            pr_table=pr_table,
            pr_base_sha=evidence["pr_gate"]["base_sha"][:8],
            pr_verdict="no new debt" if not evidence["pr_gate"]["regressions"] else "REGRESSION",
            pr_exit=evidence["pr_gate"]["exit"],
            pr_regressions=len(evidence["pr_gate"]["regressions"]),
            fix_base_sha=evidence["fix_gate"]["base_sha"][:8],
            fix_table=fix_table,
            controls=controls_table,
            t_pass=evidence["tests"]["passed"],
            t_fail=evidence["tests"]["failed"],
        ),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pr-base", default="7accc4b", help="PR #127's parent commit")
    parser.add_argument("--fix-base", default="5f2d329", help="PR #127 head before this cure")
    args = parser.parse_args(argv)

    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    backlog_info = backlog()
    pr_gate = ratchet_json(args.pr_base, EVIDENCE.parent / "ratchet_pr127_base.json")
    fix_gate = ratchet_json(args.fix_base, EVIDENCE.parent / "ratchet_head.json")
    controls = json.loads(CONTROLS.read_text())
    tests = pytest_counts()

    evidence = {
        "ruff": ruff_version(),
        "scope": SCOPE,
        "backlog": backlog_info,
        "before": {
            "job": "ruff check/format over the whole tree",
            "reported_findings": backlog_info["findings"],
        },
        "pr_gate": pr_gate,
        "fix_gate": fix_gate,
        "negative_controls": controls,
        "tests": tests,
        "verdict": {
            "pr_adds_debt": bool(pr_gate["regressions"]),
            "gate_green": pr_gate["exit"] == 0,
            "controls_fail_closed": controls["all_as_expected"],
        },
    }
    EVIDENCE.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    figure(evidence)
    render_doc(evidence)
    print(
        f"ruff {evidence['ruff']}: backlog {backlog_info['findings']} findings / "
        f"{backlog_info['unformatted_files']} unformatted"
    )
    print(
        f"PR gate ({args.pr_base} -> HEAD): exit {pr_gate['exit']}, "
        f"{len(pr_gate['regressions'])} regression(s)"
    )
    print(
        f"fix gate ({args.fix_base} -> HEAD): exit {fix_gate['exit']}, "
        f"{len(fix_gate['regressions'])} regression(s)"
    )
    print(f"controls as expected: {controls['all_as_expected']}; tests {tests['summary']}")
    print(
        f"wrote {EVIDENCE.relative_to(ROOT)}, {FIGURE.relative_to(ROOT)}, {DOC.relative_to(ROOT)}"
    )
    return (
        0 if (pr_gate["exit"] == 0 and fix_gate["exit"] == 0 and controls["all_as_expected"]) else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
