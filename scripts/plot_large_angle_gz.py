"""Regenerate the large-angle GZ evidence table and figure.

Writes
------
``benchmarks/large_angle_gz_reference.json``
    Deterministic evidence table: heel sweep with the exact geometric GZ, the
    wall-sided closed-form reference, the residual, the ``KN`` arm and the
    bilge/deck event flags, plus the summary scalars.
``docs/assets/marine_large_angle_gz.png``
    Three panels: GZ families for several ``KG`` with the closed form overlaid
    and the wall-sided band shaded; the residual against the closed form on a
    log scale; the ``KN`` curve with the dynamic-stability area shaded.

Usage
-----
    python scripts/plot_large_angle_gz.py

Everything is analytic (no mesh, no solver iterations), so the script runs in
well under a second and the JSON is reproducible bit-for-bit.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # run without an editable install
    sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorfem.marine_large_angle_gz import (  # noqa: E402
    cross_curves,
    gz_curve,
    wall_sided_dynamic_area,
    wall_sided_gz,
    wall_sided_validity_deg,
)

BARGE = dict(beam=12.0, depth=6.0, draft=2.0)
KG_LIST = (2.1, 3.2, 4.4, 5.6)
SWEEP = [index * 0.5 for index in range(0, 181)]  # 0 .. 90 deg, step 0.5
JSON_OUT = ROOT / "benchmarks" / "large_angle_gz_reference.json"
PNG_OUT = ROOT / "docs" / "assets" / "marine_large_angle_gz.png"


def build_evidence() -> dict:
    beam, depth, draft = BARGE["beam"], BARGE["depth"], BARGE["draft"]
    bilge_limit, deck_limit = wall_sided_validity_deg(beam, depth, draft)
    wall_limit = min(bilge_limit, deck_limit)
    curves = {kg: gz_curve(beam, depth, draft, kg, SWEEP) for kg in KG_LIST}
    reference = curves[KG_LIST[0]]
    rows = []
    for index, angle in enumerate(SWEEP):
        analytic = wall_sided_gz(reference.gm, reference.bm, angle)
        rows.append(
            {
                "heel_deg": angle,
                "gz": reference.gz[index],
                "wall_sided_gz": analytic,
                "abs_residual": abs(reference.gz[index] - analytic),
                "kn": reference.kn[index],
                "xb": reference.xb[index],
                "zb": reference.zb[index],
                "immersed_area": reference.section_area[index],
                "in_wall_sided_band": angle <= wall_limit,
                "bilge_emerged": reference.bilge_emerged[index],
                "deck_immersed": reference.deck_immersed[index],
            }
        )
    in_band = [row for row in rows if row["in_wall_sided_band"]]
    out_band = [row for row in rows if row["heel_deg"] > wall_limit]
    true_deck = next(row["heel_deg"] for row in rows if row["deck_immersed"])
    return {
        "schema": "tensorfem.marine-large-angle-gz.evidence.v1",
        "geometry": {**BARGE, "kg": KG_LIST[0]},
        "hydrostatics": {
            "kb": reference.kb,
            "bm": reference.bm,
            "km": reference.kb + reference.bm,
            "gm": reference.gm,
        },
        "wall_sided_limits_deg": {"bilge": bilge_limit, "deck_estimate": deck_limit, "band_end": wall_limit},
        "true_deck_immersion_deg": true_deck,
        "max_gz": reference.max_gz,
        "angle_of_max_gz_deg": reference.angle_of_max_gz_deg,
        "dynamic_stability_area": reference.dynamic_stability_area,
        "dynamic_stability_wall_sided_exact_over_band": wall_sided_dynamic_area(
            reference.gm, reference.bm, wall_limit
        ),
        "gz_at_90_deg": reference.gz[-1],
        "gz_at_90_deg_analytic": 0.5 * depth - KG_LIST[0],
        "max_abs_residual_in_band": max(row["abs_residual"] for row in in_band),
        "max_abs_gz_in_band": max(abs(row["gz"]) for row in in_band),
        "max_abs_residual_out_of_band": max(row["abs_residual"] for row in out_band),
        "kn_cross_curves": cross_curves(beam, depth, draft, KG_LIST, SWEEP[::10]),
        "rows": rows,
    }


def build_figure(evidence: dict) -> None:
    beam, depth, draft = BARGE["beam"], BARGE["depth"], BARGE["draft"]
    curves = {kg: gz_curve(beam, depth, draft, kg, SWEEP) for kg in KG_LIST}
    reference = curves[KG_LIST[0]]
    limit = evidence["wall_sided_limits_deg"]["band_end"]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    ax = axes[0]
    ax.axvspan(0.0, limit, color="0.85", label=f"wall-sided band (<= {limit:.2f}$^\\circ$)")
    for kg in KG_LIST:
        curve = curves[kg]
        ax.plot(curve.angles_deg, curve.gz, lw=1.8, label=f"exact geometry, KG={kg:g} m")
    ax.plot(
        SWEEP,
        [wall_sided_gz(reference.gm, reference.bm, angle) for angle in SWEEP],
        "k--",
        lw=1.0,
        label="wall-sided closed form (KG=2.1 m)",
    )
    ax.plot(
        [reference.angle_of_max_gz_deg],
        [reference.max_gz],
        "o",
        ms=5,
        color="crimson",
        label=f"max GZ={reference.max_gz:.3f} m @ {reference.angle_of_max_gz_deg:.1f}$^\\circ$",
    )
    ax.set_xlabel("heel angle [deg]")
    ax.set_ylabel("GZ [m]")
    ax.set_title("Barge B=12, D=6, T=2 m: large-angle GZ")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")

    ax = axes[1]
    rows = evidence["rows"]
    ax.semilogy(
        [row["heel_deg"] for row in rows],
        [max(row["abs_residual"], 1e-17) for row in rows],
        lw=1.4,
        label="|GZ_geom - GZ_wall_sided|, KG=2.1 m",
    )
    ax.axvline(limit, color="0.4", ls=":", label=f"band end {limit:.2f}$^\\circ$")
    ax.axhline(1e-12, color="green", ls="--", lw=1.0, label="1e-12 m (machine precision)")
    ax.set_xlabel("heel angle [deg]")
    ax.set_ylabel("absolute residual [m]")
    ax.set_title("Closed form is exact inside the band only")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="lower right")

    ax = axes[2]
    ax.plot(reference.angles_deg, reference.kn, color="tab:blue", lw=1.8, label="KN arm (buoyancy lever)")
    ax.fill_between(
        reference.angles_deg,
        0.0,
        reference.gz,
        color="tab:blue",
        alpha=0.15,
        label=f"dynamic stability area = {reference.dynamic_stability_area:.3f} m$\\cdot$rad",
    )
    ax.plot(reference.angles_deg, reference.gz, color="tab:blue", lw=1.0, ls=":", label="GZ, KG=2.1 m")
    ax.set_xlabel("heel angle [deg]")
    ax.set_ylabel("arm [m]")
    ax.set_title("KN decomposition and dynamic stability")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")

    fig.tight_layout()
    PNG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_OUT, dpi=150)
    plt.close(fig)


def main() -> None:
    evidence = build_evidence()
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    build_figure(evidence)
    print(f"wrote {JSON_OUT.relative_to(ROOT)}")
    print(f"wrote {PNG_OUT.relative_to(ROOT)}")
    print(
        "band end {band_end:.4f} deg, max residual in band {res:.3e} m, "
        "max GZ {gz:.4f} m at {ang:.2f} deg, GZ(90) = {g90:.4f} m".format(
            band_end=evidence["wall_sided_limits_deg"]["band_end"],
            res=evidence["max_abs_residual_in_band"],
            gz=evidence["max_gz"],
            ang=evidence["angle_of_max_gz_deg"],
            g90=evidence["gz_at_90_deg"],
        )
    )
    assert math.isfinite(evidence["max_gz"])


if __name__ == "__main__":
    main()