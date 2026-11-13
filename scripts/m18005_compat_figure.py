#!/usr/bin/env python3
"""Render the two measurements of the public-API compatibility gate (m18005).

Left panel: detection contrast -- the same two code-consistent API breaks, seen by
the point-in-time snapshot audit and by the release-to-release gate.
Right panel: how the published top-level API actually grew across the recorded
manifest history, and what the two gate modes charge for that growth.

Reads ``artifacts/m18005_public_api_compat_baseline.json`` (produced by
``scripts/m18005_compat_baseline_compare.py``); no number is typed by hand.

    python scripts/m18005_compat_figure.py --out artifacts/m18005_public_api_compat_gate.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/m18005_public_api_compat_baseline.json"
DEFAULT_OUT = ROOT / "artifacts/m18005_public_api_compat_gate.png"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact", type=Path, default=ARTIFACT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    payload = json.loads(args.artifact.read_text())
    contrast = payload["detection_contrast"]
    history = payload["history"]["transitions"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # --- left: detection contrast -------------------------------------------
    faults = [row["id"] for row in contrast["rows"]]
    snapshot = [1 if row["snapshot_gate_detected"] else 0 for row in contrast["rows"]]
    compat = [1 if row["compat_gate_detected"] else 0 for row in contrast["rows"]]
    positions = range(len(faults))
    width = 0.36
    bars_a = left.bar(
        [p - width / 2 for p in positions],
        snapshot,
        width,
        label="snapshot audit (baseline)",
        color="#b23b3b",
    )
    bars_b = left.bar(
        [p + width / 2 for p in positions],
        compat,
        width,
        label="release-to-release gate",
        color="#2e7d32",
    )
    for bars in (bars_a, bars_b):
        for bar in bars:
            left.annotate(
                "detected" if bar.get_height() else "missed",
                (bar.get_x() + bar.get_width() / 2, 0.02),
                ha="center",
                va="bottom",
                fontsize=8,
                color="white" if bar.get_height() else "black",
            )
    left.set_xticks(list(positions))
    left.set_xticklabels([fault.replace("-", "\n", 1) for fault in faults], fontsize=8)
    left.set_ylim(0, 1.35)
    left.set_yticks([0, 1])
    left.set_yticklabels(["not detected", "detected"])
    left.set_title(
        f"code-consistent API breaks: snapshot {contrast['snapshot_gate_detections']}"
        f"/{contrast['injected_faults']} vs gate {contrast['compat_gate_detections']}"
        f"/{contrast['injected_faults']}",
        fontsize=10,
    )
    left.legend(fontsize=8, loc="upper right")
    left.grid(axis="y", alpha=0.3)

    # --- right: real growth + what each mode charges -------------------------
    labels = [f"{row['base_commit']}\n->{row['head_commit']}" for row in history]
    totals = [row["entries"]["head"] for row in history]
    added = [row["counts"].get("added", 0) for row in history]
    x = list(range(len(history)))
    right.plot(x, totals, "-o", color="#1f4e79", label="top-level names in the manifest")
    right.bar(x, added, color="#9ecae1", alpha=0.8, label="names added in this transition")
    for xi, (total, extra) in enumerate(zip(totals, added)):
        right.annotate(
            str(total),
            (xi, total),
            textcoords="offset points",
            xytext=(0, 6),
            fontsize=8,
            ha="center",
        )
        right.annotate(
            f"+{extra}",
            (xi, extra),
            textcoords="offset points",
            xytext=(0, 3),
            fontsize=7,
            ha="center",
        )
    right.set_xticks(x)
    right.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    right.set_ylabel("names")
    right.set_ylim(0, max(totals) * 1.15)
    right.set_title(
        "shipped API only ever grew (no removal/rename/move/demotion): "
        f"pr-mode PASS {sum(1 for row in history if row['pr_mode']['compliant'])}/{len(history)}, "
        f"release-mode FAIL {sum(1 for row in history if not row['release_mode']['compliant'])}/{len(history)}",
        fontsize=10,
    )
    right.legend(fontsize=8, loc="lower right")
    right.grid(axis="y", alpha=0.3)

    fig.suptitle("TensorFEM public API compatibility gate (mission m18005)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
