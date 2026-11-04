"""Generate the wall-sided large-angle stability evidence + figures.

Cross-checks the wall-sided closed form (Biran & Lopez-Pulido) against the
independent polygon/shoelace oracle in
``tensorfem.marine_large_angle_stability``, quantifies the error of the
``GM * sin(phi)`` model that ``tensorfem.marine_hydrostatics`` ships for
``|heel| <= 10 deg``, demonstrates the two validity limits (keel emergence and
deck-edge immersion) and verifies the classical angle-of-loll identity.

Writes ``benchmarks/marine_large_angle_stability_evidence.json``, the
auto-generated tables ``docs/assets/marine_large_angle_stability/tables.md``
and PNGs next to them.
"""
from __future__ import annotations

import json
import math
import os

from tensorfem.marine_large_angle_stability import (
    SMALL_ANGLE_LIMIT_DEG,
    WallSidedBarge,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "docs", "assets", "marine_large_angle_stability")
EVIDENCE = os.path.join(ROOT, "benchmarks", "marine_large_angle_stability_evidence.json")
TABLES = os.path.join(ASSETS, "tables.md")

CASES = (
    # name, length, beam, depth, draft, kg   (SI, sea water)
    ("design_barge", 40.0, 12.0, 4.0, 2.0, 2.10),
    ("shallow_draft_barge", 40.0, 12.0, 6.0, 2.0, 2.50),
    ("small_gm_barge", 80.0, 20.0, 10.0, 4.0, 9.60),
    ("negative_gm_barge", 80.0, 20.0, 10.0, 4.0, 10.83),
    ("deck_limited_barge", 30.0, 10.0, 4.0, 2.5, 2.00),
)


def heel_grid(limit_deg: float, step: float = 1.0) -> list[float]:
    """Heels from 0 to the closed-form limit inclusive, plus quarter steps."""

    count = int(math.floor(limit_deg / step))
    grid = [round(index * step, 10) for index in range(count + 1)]
    for index in range(count):
        grid.append(round(index * step + 0.5 * step, 10))
    grid.append(round(limit_deg * (1.0 - 1e-9), 10))
    return sorted(set(grid))


def build_case(entry) -> dict:
    name, length, beam, depth, draft, kg = entry
    barge = WallSidedBarge(str(name), length, beam, depth, draft, kg)
    limit = barge.closed_form_limit_deg
    rows = barge.curve(heel_grid(limit))
    scale = max(abs(row["gz_numerical"]) for row in rows) or 1.0
    agreement = max(
        abs(row["gz_closed_form"] - row["gz_numerical"])
        for row in rows
        if row["gz_closed_form"] is not None
    )
    displacement_error = max(abs(row["displacement_error"]) for row in rows)
    small_rows = [
        row for row in rows
        if row["gz_small_angle"] is not None and row["heel_angle_deg"] > 1e-9
        and barge.gm > 0.0
    ]
    small_error = max((row["small_angle_relative_error"] for row in small_rows), default=None)
    crossing = None
    if barge.gm > 0.0 and small_error is not None:
        for coarse, fine in zip(small_rows, small_rows[1:]):
            e0 = coarse["small_angle_relative_error"]
            e1 = fine["small_angle_relative_error"]
            if e0 < 0.03 <= e1:
                crossing = round(fine["heel_angle_deg"], 6)
                break
    metrics = {
        "max_abs_closed_form_minus_oracle": agreement,
        "max_abs_closed_form_minus_oracle_relative": agreement / scale,
        "max_displacement_error": displacement_error,
        "max_small_angle_relative_error_within_10deg": small_error,
        "first_heel_above_3pct_small_angle_error": crossing,
        "gm_sign": barge.stability_state,
        "closed_form_limit_deg": limit,
        "closed_form_limit_reason": barge.closed_form_limit_reason,
    }
    return {
        "barge": barge.as_dict(),
        "rows": rows,
        "metrics": metrics,
        "barge_object": barge,
    }


def keel_emergence_regime(barge: WallSidedBarge) -> dict:
    """Oracle stays valid past keel emergence; the closed form must not be used."""

    limit = barge.closed_form_limit_deg
    upper = barge.equilibrium_deck_immersion_deg()
    sample = []
    heel = limit + 5.0
    while heel <= upper * 0.99:
        state = barge.section_state(heel)
        numeric = barge.gz_numerical(heel)
        extrapolated = barge.gz_closed_form(heel, strict=False)
        sample.append(
            {
                "heel_angle_deg": round(heel, 6),
                "gz_numerical": numeric,
                "gz_closed_form_extrapolated": extrapolated,
                "extrapolation_relative_deviation": abs(extrapolated - numeric) / abs(numeric),
                "sinkage": float(state["sinkage"]),
                "keel_emerged": bool(state["keel_emerged"]),
                "displacement_error": float(state["area_error"]),
            }
        )
        heel += 5.0
    if not sample or abs(sample[-1]["heel_angle_deg"] - limit) > 1e-9:
        state = barge.section_state(limit)
        numeric = barge.gz_numerical(limit)
        extrapolated = barge.gz_closed_form(limit, strict=False)
        sample.insert(
            0,
            {
                "heel_angle_deg": round(limit, 6),
                "gz_numerical": numeric,
                "gz_closed_form_extrapolated": extrapolated,
                "extrapolation_relative_deviation": abs(extrapolated - numeric) / abs(numeric),
                "sinkage": float(state["sinkage"]),
                "keel_emerged": bool(state["keel_emerged"]),
                "displacement_error": float(state["area_error"]),
            },
        )
    state = barge.section_state(upper * 0.995)
    max_deviation = max(row["extrapolation_relative_deviation"] for row in sample)
    return {
        "barge": barge.name,
        "keel_emergence_limit_deg": barge.keel_emergence_limit_deg,
        "deck_immersion_limit_deg": barge.deck_immersion_limit_deg,
        "equilibrium_deck_immersion_deg": barge.equilibrium_deck_immersion_deg(),
        "samples": sample,
        "max_extrapolation_relative_deviation": max_deviation,
        "oracle_displacement_error_near_deck": float(state["area_error"]),
        "sinkage_near_deck": float(state["sinkage"]),
    }


def loll_identity(barge: WallSidedBarge) -> dict | None:
    """Verify tan^2(phi_loll) = -2 GM / BM against the zero crossing of GZ."""

    if barge.gm >= 0.0:
        return None
    loll = barge.angle_of_loll_deg
    lo, hi = 0.0, barge.closed_form_limit_deg
    for _ in range(400):
        mid = 0.5 * (lo + hi)
        if barge.gz_numerical(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    crossing = 0.5 * (lo + hi)
    return {
        "barge": barge.name,
        "gm": barge.gm,
        "bm": barge.bm,
        "angle_of_loll_deg": loll,
        "numeric_zero_crossing_deg": crossing,
        "absolute_difference_deg": abs(crossing - loll),
        "relative_difference": abs(crossing - loll) / loll,
        "gz_at_loll": barge.gz_numerical(loll),
        "inside_closed_form_range": barge.loll_inside_closed_form_range,
    }


def _markdown_tables(cases, keel, loll) -> str:
    lines = ["<!-- generated by benchmarks/marine_large_angle_stability.py -->", ""]
    lines.append("### Table A — hull parameters and validity limits")
    lines.append("")
    lines.append("| barge | L [m] | B [m] | D [m] | T [m] | KG [m] | GM [m] | BM [m] | deck limit [deg] | keel limit [deg] | closed-form limit [deg] |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for case in cases:
        b = case["barge"]
        lines.append(
            f"| {b['name']} | {b['length']:g} | {b['beam']:g} | {b['depth']:g} | {b['draft']:g} | "
            f"{b['kg']:g} | {b['gm']:.4f} | {b['bm']:.4f} | {b['deck_immersion_limit_deg']:.3f} | "
            f"{b['keel_emergence_limit_deg']:.3f} | {b['closed_form_limit_deg']:.3f} "
            f"({b['closed_form_limit_reason']}) |"
        )
    lines.append("")
    lines.append("### Table B — closed form vs independent polygon oracle")
    lines.append("")
    lines.append("| barge | heels | max abs difference [m] | max relative difference | max displacement error [m^2] |")
    lines.append("|---|---|---|---|---|")
    for case in cases:
        m = case["metrics"]
        lines.append(
            f"| {case['barge']['name']} | {len(case['rows'])} | "
            f"{m['max_abs_closed_form_minus_oracle']:.3e} | "
            f"{m['max_abs_closed_form_minus_oracle_relative']:.3e} | "
            f"{m['max_displacement_error']:.3e} |"
        )
    lines.append("")
    lines.append("### Table C — `GM * sin(phi)` audit (the merged `marine_hydrostatics` model)")
    lines.append("")
    lines.append("| barge | GM [m] | BM/GM | max relative error 0-10 deg | max relative error 0-limit | first heel above 3% [deg] |")
    lines.append("|---|---|---|---|---|---|")
    for case in cases:
        b, m = case["barge"], case["metrics"]
        within = "n/a (GM<=0)" if m["max_small_angle_relative_error_within_10deg"] is None else f"{m['max_small_angle_relative_error_within_10deg']*100:.3f}%"
        full = max(
            (row["small_angle_relative_error"] for row in case["rows"]
             if row["small_angle_relative_error"] is not None and row["heel_angle_deg"] > 1e-9),
            default=None,
        )
        full_txt = "n/a (GM<=0)" if full is None else f"{full*100:.3f}%"
        crossing = m["first_heel_above_3pct_small_angle_error"]
        lines.append(
            f"| {b['name']} | {b['gm']:.4f} | {b['bm']/b['gm']:.3f} | {within} | {full_txt} | "
            f"{crossing if crossing is not None else '-'} |"
        )
    lines.append("")
    lines.append("### Table D — past keel emergence: oracle vs closed-form extrapolation")
    lines.append("")
    lines.append(f"Barge `{keel['barge']}`, keel limit {keel['keel_emergence_limit_deg']:.3f} deg, deck-edge limit at upright draft {keel['deck_immersion_limit_deg']:.3f} deg, equilibrium deck-edge limit {keel['equilibrium_deck_immersion_deg']:.3f} deg.")
    lines.append("")
    lines.append("| heel [deg] | GZ oracle [m] | GZ closed-form extrapolated [m] | deviation | sinkage [m] | keel emerged |")
    lines.append("|---|---|---|---|---|---|")
    for row in keel["samples"]:
        lines.append(
            f"| {row['heel_angle_deg']:.3f} | {row['gz_numerical']:.5f} | "
            f"{row['gz_closed_form_extrapolated']:.5f} | "
            f"{row['extrapolation_relative_deviation']*100:.3f}% | {row['sinkage']:.5f} | "
            f"{'yes' if row['keel_emerged'] else 'no'} |"
        )
    lines.append("")
    if loll is not None:
        lines.append("### Table E — angle of loll identity `tan^2(phi) = -2 GM / BM`")
        lines.append("")
        lines.append("| barge | GM [m] | BM [m] | closed-form loll [deg] | oracle zero crossing [deg] | difference [deg] | GZ at loll [m] |")
        lines.append("|---|---|---|---|---|---|---|")
        lines.append(
            f"| {loll['barge']} | {loll['gm']:.4f} | {loll['bm']:.4f} | {loll['angle_of_loll_deg']:.6f} | "
            f"{loll['numeric_zero_crossing_deg']:.6f} | {loll['absolute_difference_deg']:.3e} | "
            f"{loll['gz_at_loll']:.3e} |"
        )
        lines.append("")
    return "\n".join(lines) + "\n"


def figures(cases, keel) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(ASSETS, exist_ok=True)
    produced = []
    colours = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

    fig, axes = plt.subplots(1, len(cases), figsize=(2.55 * len(cases), 2.7), sharey=False)
    for axis, case, colour in zip(axes, cases, colours):
        b = case["barge"]
        rows = case["rows"]
        heels = [row["heel_angle_deg"] for row in rows]
        axis.plot(heels, [row["gz_closed_form"] for row in rows], "-", color=colour,
                  lw=1.4, label="closed form (1)")
        axis.plot(heels, [row["gz_numerical"] for row in rows], "o", ms=2.6,
                  mfc="none", mec="k", mew=0.6, label="polygon oracle (2)")
        small = [(row["heel_angle_deg"], row["gz_small_angle"]) for row in rows
                 if row["gz_small_angle"] is not None]
        if small:
            axis.plot([h for h, _ in small], [g for _, g in small], ":", color="0.35",
                      lw=1.2, label=r"$GM\sin\phi$ (merged)")
        axis.axhline(0.0, color="0.6", lw=0.6)
        axis.axvline(b["closed_form_limit_deg"], color="0.5", ls="--", lw=0.8)
        axis.set_title(f"{b['name']}\nlimit {b['closed_form_limit_deg']:.1f} deg "
                       f"({b['closed_form_limit_reason'].replace('_', ' ')})", fontsize=7.5)
        axis.set_xlabel("heel [deg]", fontsize=8)
        axis.set_ylabel("GZ [m]", fontsize=8)
        axis.tick_params(labelsize=7)
    axes[0].legend(fontsize=6, loc="upper left")
    fig.tight_layout()
    path = os.path.join(ASSETS, "gz_vs_heel.png")
    fig.savefig(path, dpi=95)
    plt.close(fig)
    produced.append(path)

    fig, axis = plt.subplots(figsize=(4.6, 3.0))
    for case, colour in zip(cases, colours):
        rows = [row for row in case["rows"]
                if row["small_angle_relative_error"] is not None and case["barge"]["gm"] > 0.0]
        if not rows:
            continue
        axis.plot([row["heel_angle_deg"] for row in rows],
                  [100.0 * row["small_angle_relative_error"] for row in rows],
                  "-", color=colour, lw=1.4,
                  label=f"{case['barge']['name']} (BM/GM={case['barge']['bm']/case['barge']['gm']:.2f})")
    axis.axhline(3.0, color="k", ls="--", lw=1.0, label="3% gate")
    axis.axhline(15.0, color="0.4", ls=":", lw=1.0, label="15% gate")
    axis.axvline(SMALL_ANGLE_LIMIT_DEG, color="0.2", ls="-.", lw=1.0, label="declared 10 deg cap")
    axis.set_xlabel("heel [deg]", fontsize=8)
    axis.set_ylabel(r"$|GM\sin\phi - GZ| / |GZ|$  [%]", fontsize=8)
    axis.tick_params(labelsize=7)
    axis.set_yscale("log")
    axis.legend(fontsize=6, loc="upper left")
    axis.set_title("small-angle model error against the exact wall-sided arm", fontsize=8)
    fig.tight_layout()
    path = os.path.join(ASSETS, "small_angle_error.png")
    fig.savefig(path, dpi=95)
    plt.close(fig)
    produced.append(path)
    return produced


def main() -> dict:
    cases = [build_case(entry) for entry in CASES]
    keel_case = next(case for case in cases if case["barge"]["name"] == "shallow_draft_barge")
    keel = keel_emergence_regime(keel_case["barge_object"])
    loll = loll_identity(
        next(case["barge_object"] for case in cases
             if case["barge"]["name"] == "negative_gm_barge")
    )
    os.makedirs(ASSETS, exist_ok=True)
    figures_paths = figures(cases, keel)
    payload = {
        "schema": "tensorfem.marine-large-angle-stability-evidence.v1",
        "generated_by": "benchmarks/marine_large_angle_stability.py",
        "reference": {
            "closed_form": "GZ(phi) = sin(phi) * (GM + 0.5 * BM * tan^2(phi))",
            "lever_decomposition": "GZ = y_B cos(phi) + (KB' - KG) sin(phi), y_B = BM tan(phi), KB' = T/2 + 0.5 BM tan^2(phi)",
            "source": "Biran & Lopez-Pulido, Ship Hydrostatics and Stability (2nd ed.), wall-sided righting-lever identity; Derrett, Ship Stability for Masters and Mates",
            "loll_identity": "tan^2(phi_loll) = -2 GM / BM",
            "small_angle_limit_deg": SMALL_ANGLE_LIMIT_DEG,
        },
        "agreement": {
            "max_abs_closed_form_minus_oracle": max(
                case["metrics"]["max_abs_closed_form_minus_oracle"] for case in cases
            ),
            "max_abs_closed_form_minus_oracle_relative": max(
                case["metrics"]["max_abs_closed_form_minus_oracle_relative"] for case in cases
            ),
            "max_displacement_error": max(
                case["metrics"]["max_displacement_error"] for case in cases
            ),
        },
        "cases": [
            {"barge": case["barge"], "metrics": case["metrics"], "rows": case["rows"]}
            for case in cases
        ],
        "keel_emergence_regime": keel,
        "loll_identity": loll,
        "figures": [os.path.relpath(path, ROOT) for path in figures_paths],
    }
    with open(EVIDENCE, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")
    with open(TABLES, "w", encoding="utf-8") as handle:
        handle.write(_markdown_tables(cases, keel, loll))
    return payload


if __name__ == "__main__":
    result = main()
    print(json.dumps(result["agreement"], indent=2))
    for case in result["cases"]:
        m = case["metrics"]
        print(
            f"{case['barge']['name']:>22} GM={case['barge']['gm']:8.4f} "
            f"limit={m['closed_form_limit_deg']:6.3f} ({m['closed_form_limit_reason']}) "
            f"agree={m['max_abs_closed_form_minus_oracle']:.2e} "
            f"small10={m['max_small_angle_relative_error_within_10deg']}"
        )