"""Figure for the ``cohesive3d_verification_tables.{json,md}`` reproducibility gate.

Two panels, both read *programmatically* from the committed artefacts (no number
is typed by hand):

* **left** -- headroom of every ``PASS`` row: ``value / tolerance`` on a log
  axis.  A row at ``1e-16`` has sixteen decades of margin; a row at ``1`` is at
  its predeclared limit.  The two ``KNOWN_LIMIT`` rows are drawn outside the
  axis (they are measurements of a documented failure, not claims).
* **right** -- the drift the gate now catches: the DCB specimen compliance
  ``C_fem(a)`` published by the table against the analytic Williams-root-rotation
  bracket, with the pre-gate committed value at ``a = 40 mm`` (taken from
  ``git show HEAD:benchmarks/cohesive3d_verification_tables.json`` when a base
  revision is available).

Usage::

    python scripts/cohesive3d_tables_gate_figure.py
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
ARTIFACT = ROOT / "benchmarks" / "cohesive3d_verification_tables.json"
OUT = ROOT / "docs" / "assets" / "cohesive3d_tables_gate.png"
REL = "benchmarks/cohesive3d_verification_tables.json"

NUMBER = re.compile(r"[-+]?\d+\.\d+e[-+]\d+")


def _git_show(rel: str) -> dict | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "show", f"HEAD:{rel}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - no git
        return None
    if out.returncode != 0:
        return None
    return json.loads(out.stdout)


def specimen_points(payload: dict) -> list:
    """``(a_mm, c_plain, c_root_11, c_fem)`` parsed from the published rows."""
    points = []
    for row in payload["checks"]:
        match = re.fullmatch(r"specimen_compliance_a(\d+)mm", row["check"])
        if not match:
            continue
        bracket = NUMBER.findall(row["oracle"])
        c_fem = NUMBER.findall(row["measure"])
        points.append(
            (float(match.group(1)), float(bracket[0]), float(bracket[1]), float(c_fem[0]))
        )
    return sorted(points)


def main() -> int:
    payload = json.loads(ARTIFACT.read_text())
    rows = payload["checks"]
    headroom = [
        (row["check"], row["value"] / row["tolerance"])
        for row in rows
        if row["status"] == "PASS" and row["tolerance"] > 0
    ]
    limits = [(row["check"], row["value"]) for row in rows if row["status"] == "KNOWN_LIMIT"]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16.5, 7.6))

    # ---- left: headroom of every PASS row
    labels = [name for name, _ in headroom]
    ratios = [max(ratio, 1e-18) for _, ratio in headroom]
    ypos = range(len(labels))
    colors = ["#2a7f62" if ratio < 1e-3 else "#c9a227" for ratio in ratios]
    ax0.barh(list(ypos), ratios, color=colors, height=0.72)
    ax0.set_yticks(list(ypos))
    ax0.set_yticklabels(labels, fontsize=6.4, family="monospace")
    ax0.set_xscale("log")
    ax0.axvline(1.0, color="#b3261e", lw=1.4, ls="--", label="predeclared tolerance (=1)")
    ax0.set_xlabel("measured error / predeclared tolerance  (log)")
    ax0.set_xlim(1e-18, 10.0)
    ax0.invert_yaxis()
    ax0.set_title(
        f"{len(rows)} published rows: {len(headroom)} PASS vs tolerance, "
        f"{len(limits)} KNOWN_LIMIT (measured, not claimed)",
        fontsize=10,
    )
    ax0.legend(loc="lower right", fontsize=8)
    ax0.grid(axis="x", which="both", alpha=0.25, lw=0.5)
    for idx, (name, ratio) in enumerate(headroom):
        ax0.text(min(ratio * 1.6, 4.0), idx, f"{ratio:.0e}", va="center", fontsize=5.4)

    # ---- right: specimen compliance vs the analytic bracket, and the stale value
    points = specimen_points(payload)
    a = [p[0] for p in points]
    plain = [p[1] for p in points]
    root11 = [p[2] for p in points]
    fem = [p[3] for p in points]
    ax1.fill_between(
        a,
        plain,
        root11,
        color="#9fc7b4",
        alpha=0.45,
        label="acceptance bracket [plain, 1.10 x root rotation]",
    )
    ax1.plot(a, plain, "-o", color="#41522f", lw=1.2, ms=4, label="plain Timoshenko beam")
    ax1.plot(a, root11, "-s", color="#2a6f97", lw=1.2, ms=4, label="1.10 x Williams root rotation")
    ax1.plot(a, fem, "D", color="#b3261e", ms=9, label="C_fem published by the table")

    previous = _git_show(REL)
    stale_marked = False
    if previous is not None:
        for value_a, _, _, value_c in specimen_points(previous):
            if abs(value_c - dict(zip(a, fem))[value_a]) > 1e-12:
                ax1.plot(
                    [value_a],
                    [value_c],
                    "x",
                    color="#7b1fa2",
                    ms=11,
                    mew=2.2,
                    label=f"pre-gate committed C_fem (a={value_a:.0f} mm): {value_c:.6e} m/N",
                )
                stale_marked = True

    ax1.set_xlabel("initial crack length a (mm)")
    ax1.set_ylabel("specimen compliance C (m/N)")
    ax1.set_title(
        "DCB compliance: current code vs the bracket"
        + (
            "\n(the stale committed value is what the freshness gate now rejects)"
            if stale_marked
            else ""
        ),
        fontsize=10,
    )
    ax1.legend(fontsize=7.6, loc="upper left")
    ax1.grid(alpha=0.3, lw=0.5)

    fig.suptitle(
        "3-D cohesive verification tables: "
        f"checks_digest {payload['checks_digest'][:12]}... , "
        "reproducible under --check (rtol 1e-8, atol 1e-10)",
        fontsize=11,
    )
    fig.subplots_adjust(left=0.30, right=0.985, top=0.88, bottom=0.10, wspace=0.32)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160)
    print(f"wrote {OUT.relative_to(ROOT)} ({len(headroom)} PASS rows, {len(points)} specimen rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
