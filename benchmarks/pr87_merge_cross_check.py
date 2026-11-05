"""PR #87 merge gate: cross-validate the two independent large-angle implementations.

After the three-way merge of ``agent/m3495`` into ``main`` the repository carries
*two* independent derivations of the intact righting arm of a wall-sided barge:

* ``BoxBarge`` (merged from m3275) -- clips the heeled section and takes the
  buoyancy centroid of the clipped polygon, then projects it into the space
  horizontal;
* ``WallSidedBarge`` (this PR) -- evaluates the wall-sided closed form and
  cross-checks it against its *own* polygon/shoelace oracle, which solves the
  constant-displacement waterplane by bisection.

They share no numerical core: one bisects a free-surface height and projects a
centroid, the other evaluates ``sin(phi)*(GM + 0.5*BM*tan^2 phi)``.  Agreement
to round-off is therefore a genuine consistency proof for the merged module,
not a tautology.  This script measures that agreement and writes
``artifacts/pr87_merge_cross_check.json`` plus a figure.

Run: ``PYTHONPATH=src python3 benchmarks/pr87_merge_cross_check.py``
"""

from __future__ import annotations

import json
import os

from tensorfem.marine_large_angle_stability import BoxBarge, WallSidedBarge

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS = os.path.join(ROOT, "artifacts")
ASSETS = os.path.join(ROOT, "docs", "assets", "pr87_merge_cross_check")

# (name, length, beam, depth, draft, kg) -- geometries that bind the envelope on
# each of the two limits, plus a tender section where BM/GM is large.
CASES = [
    ("slender_deck_limited", 90.0, 20.0, 8.0, 6.0, 4.5),
    ("wide_shallow_low_gm", 80.0, 20.0, 10.0, 4.0, 9.6),
    ("narrow_deep_section", 40.0, 12.0, 4.0, 2.0, 2.1),
]
SAMPLES = 41


def sweep(length: float, beam: float, depth: float, draft: float, kg: float):
    """Both implementations over the shared closed-form envelope."""
    exact = BoxBarge(length=length, beam=beam, depth=depth, draft=draft, kg=kg)
    branch = WallSidedBarge("cross-check", length, beam, depth, draft, kg)
    limit = min(exact.deck_edge_angle_deg, exact.bottom_exposure_angle_deg)
    rows = []
    for index in range(SAMPLES):
        heel = limit * index / (SAMPLES - 1)
        a = exact.gz(heel)                      # centroid route
        b_closed = branch.gz_closed_form(heel)  # closed form
        b_oracle = branch.gz_numerical(heel)    # branch polygon oracle
        rows.append(
            {
                "heel_deg": heel,
                "bbox_barge_centroid": a,
                "wall_sided_closed_form": b_closed,
                "wall_sided_oracle": b_oracle,
                "abs_diff_closed_form": abs(a - b_closed),
                "abs_diff_oracle": abs(a - b_oracle),
                "linear_gz": exact.linear_gz(heel),
            }
        )
    return limit, exact, rows


def build_payload(verbose: bool = True):
    """Compute the cross-check payload.

    Split out of :func:`main` so the test suite can reproduce every number
    without depending on ``artifacts/`` (which the delivery gate filters).
    """
    results = []
    for name, length, beam, depth, draft, kg in CASES:
        limit, barge, rows = sweep(length, beam, depth, draft, kg)
        scale = max(abs(row["bbox_barge_centroid"]) for row in rows) or 1.0
        max_abs = max(row["abs_diff_closed_form"] for row in rows)
        max_abs_oracle = max(row["abs_diff_oracle"] for row in rows)
        results.append(
            {
                "name": name,
                "geometry": {
                    "length": length,
                    "beam": beam,
                    "depth": depth,
                    "draft": draft,
                    "kg": kg,
                },
                "closed_form_limit_deg": limit,
                "limit_reason": (
                    "deck_edge_immersion"
                    if barge.deck_edge_angle_deg <= barge.bottom_exposure_angle_deg
                    else "keel_emergence"
                ),
                "samples": SAMPLES,
                "max_abs_diff_closed_form_vs_centroid": max_abs,
                "max_rel_diff_closed_form_vs_centroid": max_abs / scale,
                "max_abs_diff_oracle_vs_centroid": max_abs_oracle,
                "max_rel_diff_oracle_vs_centroid": max_abs_oracle / scale,
                "max_linear_model_relative_error": max(
                    barge.linear_arm_relative_error(row["heel_deg"]) for row in rows
                ),
                "rows": rows,
            }
        )
        if verbose:
            print(
                f"{name:22s} limit={limit:7.3f} deg ({results[-1]['limit_reason']:20s}) "
                f"|closed-centroid|={max_abs:.3e} m  "
                f"|oracle-centroid|={max_abs_oracle:.3e} m  "
                f"linear model max err={results[-1]['max_linear_model_relative_error'] * 100:.2f}%"
            )

    overall = {
        "max_abs_diff_closed_form_vs_centroid": max(
            item["max_abs_diff_closed_form_vs_centroid"] for item in results
        ),
        "max_rel_diff_closed_form_vs_centroid": max(
            item["max_rel_diff_closed_form_vs_centroid"] for item in results
        ),
        "max_abs_diff_oracle_vs_centroid": max(
            item["max_abs_diff_oracle_vs_centroid"] for item in results
        ),
        "max_rel_diff_oracle_vs_centroid": max(
            item["max_rel_diff_oracle_vs_centroid"] for item in results
        ),
    }
    payload = {
        "schema": "tensorfem.pr87-merge-cross-check.v1",
        "purpose": (
            "Consistency of the two independent large-angle implementations that "
            "coexist after the three-way merge of agent/m3495 into main."
        ),
        "gate": {"max_relative_difference": 1e-9},
        "cases": results,
        "overall": overall,
        "verdict": (
            "PASS"
            if overall["max_rel_diff_closed_form_vs_centroid"] < 1e-9
            and overall["max_rel_diff_oracle_vs_centroid"] < 1e-9
            else "FAIL"
        ),
    }
    return payload


def main() -> int:
    os.makedirs(ARTIFACTS, exist_ok=True)
    os.makedirs(ASSETS, exist_ok=True)
    payload = build_payload()
    results = payload["cases"]
    path = os.path.join(ARTIFACTS, "pr87_merge_cross_check.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("wrote", os.path.relpath(path, ROOT), "verdict:", payload["verdict"])

    _figure(results)
    return 0


def _figure(results) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(8.6, 3.5), dpi=72)
    for item in results:
        heels = [row["heel_deg"] for row in item["rows"]]
        left.plot(
            heels,
            [row["bbox_barge_centroid"] for row in item["rows"]],
            "-",
            linewidth=2.0,
            label=f"{item['name']}: BoxBarge centroid (m3275)",
        )
        left.plot(
            heels,
            [row["wall_sided_closed_form"] for row in item["rows"]],
            "o",
            markersize=3.2,
            fillstyle="none",
            label=f"{item['name']}: WallSidedBarge closed form (PR #87)",
        )
        right.semilogy(
            heels,
            [max(row["abs_diff_closed_form"], 1e-18) for row in item["rows"]],
            "-",
            linewidth=1.8,
            label=f"{item['name']}: |closed form - centroid|",
        )
        right.semilogy(
            heels,
            [max(row["abs_diff_oracle"], 1e-18) for row in item["rows"]],
            "--",
            linewidth=1.2,
            label=f"{item['name']}: |branch oracle - centroid|",
        )
    left.set_xlabel("heel angle [deg]")
    left.set_ylabel("GZ [m]")
    left.set_title("Two independent routes to the same righting arm")
    left.grid(alpha=0.3)
    left.legend(fontsize=6.2, loc="upper left")
    right.axhline(1e-12, color="crimson", linestyle=":", linewidth=1.0,
                  label="1e-12 m round-off band")
    right.set_xlabel("heel angle [deg]")
    right.set_ylabel("absolute difference [m]")
    right.set_title("Agreement is at double-precision round-off")
    right.grid(alpha=0.3, which="both")
    right.legend(fontsize=6.2, loc="lower right")
    fig.tight_layout()
    out = os.path.join(ASSETS, "cross_check.png")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    raise SystemExit(main())