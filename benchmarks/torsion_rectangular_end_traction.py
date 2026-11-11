"""End-traction attribution and wall-peak recovery for rectangular Saint-Venant torsion.

Closes the two open items of ``docs/torsion-rectangular-section-verification.md`` §5:

* §5.1 the ``a/b = 3`` 3-D twist rate failed to converge under refinement.  The report
  attributed it to the Saint-Venant end zone ("window lies inside the decay length
  ``~ a``") but had no decisive experiment.  Here the same prism is twisted twice: with
  the *physical* linear end traction (the original load case) and with the exact
  Saint-Venant traction.  If the end-zone story is right, the exact load case must
  converge, and the linear one must keep drifting -- and the measured twist rate of the
  linear case must depend on the extraction window while the exact one does not.
* §5.2 the wall-peak shear of a rectangle was never recovered (the surface flow existed
  for ``a == b`` only).  ``surface_peak_shear`` supplies three schemes and this script
  measures them against the closed-form peak at the rate the FE solve itself produced.

Outputs ``docs/assets/torsion_rectangular_end_traction/evidence.json`` plus two figures.
"""
from __future__ import annotations

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorfem.torsion_rectangular import (  # noqa: E402
    _twist_prism,
    rectangular_torsion_bar,
    shape_factors,
    surface_peak_shear,
)

OUT = pathlib.Path("docs/assets/torsion_rectangular_end_traction")
OUT.mkdir(parents=True, exist_ok=True)
WINDOWS = ((0.25, 0.75), (0.30, 0.70), (0.35, 0.65))


def twist_table() -> dict:
    """§5.1: linear vs exact end traction, and window sensitivity, at a/b = 3."""
    rows = []
    for traction in ("linear", "exact"):
        for segments in (3, 4, 5):
            solve = _twist_prism(a=3.0, b=1.0, length=4.0, segments_short=segments,
                                 segments_long=None, young=1e9, poisson=0.3, torque=1.0,
                                 window=WINDOWS[0], terms=401, rtol=1e-12,
                                 end_traction=traction)
            result = rectangular_torsion_bar(a=3.0, b=1.0, length=4.0, segments_short=segments,
                                             end_traction=traction)
            exact = result.exact_twist_rate
            rates = {f"{lo:.2f}-{hi:.2f}": solve.twist_rate_in((lo, hi)) for lo, hi in WINDOWS}
            row = {
                "end_traction": traction,
                "segments_short": segments,
                "n_long": solve.n_long,
                "nz": solve.nz,
                "n_dofs": 3 * len(solve.nodes),
                "twist_rate_error": {
                    key: abs(value / exact - 1.0) for key, value in rates.items()},
                "torsion_constant_relative_error": result.torsion_constant_relative_error,
                "end_traction_residual": result.end_traction_residual,
                "applied_torque": result.applied_torque,
                "centroid_shear_l2_error": result.centroid_shear_l2_error,
            }
            error = row["twist_rate_error"]
            row["window_spread"] = max(error.values()) - min(error.values())
            rows.append(row)
    return {"aspect": 3.0, "b": 1.0, "length": 4.0, "window": "0.25-0.75 (reference)",
            "rows": rows}


def length_study() -> dict:
    """§5.1: does a longer bar cure the physical (linear) load case?"""
    rows = []
    for length in (4.0, 6.0, 9.0):
        for traction in ("linear", "exact"):
            result = rectangular_torsion_bar(a=3.0, b=1.0, length=length, segments_short=3,
                                             end_traction=traction)
            rows.append({"length": length, "aspect_over_length": 3.0 / length,
                         "end_traction": traction,
                         "torsion_constant_relative_error": result.torsion_constant_relative_error,
                         "n_dofs": result.n_dofs})
    return {"aspect": 3.0, "segments_short": 3, "rows": rows}


def wall_peaks() -> dict:
    """§5.2: wall shear at both mid-side points, three recovery schemes."""
    rows = []
    for aspect in (1.5, 2.0, 3.0):
        for segments in (4, 6):
            # a node must sit on the y = 0 symmetry plane, hence an even long-side count
            n_long = int(round(segments * aspect))
            n_long += n_long % 2
            peak = surface_peak_shear(a=aspect, b=1.0, length=6.0, segments_short=segments,
                                      segments_long=n_long)
            rows.append({
                "aspect": aspect, "segments_short": segments,
                "segments_long": peak.segments_long, "n_dofs": peak.n_dofs,
                "exact_long": peak.exact_long, "exact_short": peak.exact_short,
                "direct_long": peak.direct_long, "direct_short": peak.direct_short,
                "gauss_long": peak.gauss_long, "gauss_short": peak.gauss_short,
                "centroid_long": peak.centroid_long, "centroid_short": peak.centroid_short,
                "error_direct_long": peak.relative_error("direct"),
                "error_direct_short": peak.relative_error("direct", "short"),
                "error_gauss_long": peak.relative_error("gauss"),
                "error_gauss_short": peak.relative_error("gauss", "short"),
                "error_centroid_long": peak.relative_error("centroid"),
                "error_centroid_short": peak.relative_error("centroid", "short"),
                "long_over_short_exact": peak.long_over_short_exact,
                "long_over_short_direct": peak.long_over_short_direct,
            })
    series = {str(aspect): {"long_over_short": shape_factors(aspect, 1.0).long_over_short,
                            "c2": shape_factors(aspect, 1.0).c2}
              for aspect in (1.5, 2.0, 3.0)}
    return {"length": 6.0, "rows": rows, "series": series}


def figure_twist(table: dict, length_rows: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), dpi=140)
    for traction, marker in (("linear", "o"), ("exact", "s")):
        rows = [row for row in table["rows"] if row["end_traction"] == traction]
        error = [100.0 * row["torsion_constant_relative_error"] for row in rows]
        spread = [100.0 * row["window_spread"] for row in rows]
        axes[0].errorbar([row["segments_short"] for row in rows], error, yerr=spread,
                         marker=marker, capsize=4, label=f"{traction} traction")
    axes[0].axhline(3.0, linestyle="--", color="grey", linewidth=1.0, label="3 % gate")
    axes[0].set_xlabel("section segments (short side)")
    axes[0].set_ylabel("twist-rate error  [%]")
    axes[0].set_title("a/b = 3, L = 4: error bars = window spread")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    for traction, marker in (("linear", "o"), ("exact", "s")):
        rows = [row for row in length_rows["rows"] if row["end_traction"] == traction]
        axes[1].semilogy([row["length"] for row in rows],
                          [100.0 * row["torsion_constant_relative_error"] for row in rows],
                          marker=marker, label=f"{traction} traction")
    axes[1].axhline(3.0, linestyle="--", color="grey", linewidth=1.0, label="3 % gate")
    axes[1].set_xlabel("bar length L  (a = 3)")
    axes[1].set_ylabel("twist-rate error  [%]")
    axes[1].set_title("physical load case: error vs end-zone distance")
    axes[1].legend()
    axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_end_traction_attribution.png")
    plt.close(fig)


def figure_walls(table: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), dpi=140)
    for scheme, marker in (("direct", "o"), ("gauss", "s"), ("centroid", "^")):
        fine = [row for row in table["rows"] if row["segments_short"] == 6]
        axes[0].plot([row["aspect"] for row in fine],
                     [100.0 * row[f"error_{scheme}_long"] for row in fine],
                     marker=marker, label=f"{scheme} (long side)")
        axes[1].plot([row["aspect"] for row in fine],
                     [100.0 * row[f"error_{scheme}_short"] for row in fine],
                     marker=marker, label=f"{scheme} (short side)")
    for axis, title in zip(axes, ("long-side midpoint (the peak)", "short-side midpoint")):
        axis.axhline(3.0, linestyle="--", color="grey", linewidth=1.0, label="3 % gate")
        axis.set_xlabel("a / b")
        axis.set_ylabel("wall-shear error  [%]")
        axis.set_title(f"{title}, n = 6, L = 6")
        axis.legend(fontsize=7)
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_wall_peak_recovery.png")
    plt.close(fig)


def main() -> None:
    twist = twist_table()
    length_rows = length_study()
    walls = wall_peaks()
    evidence = {"twist_rate": twist, "length_study": length_rows, "wall_peaks": walls}
    (OUT / "evidence.json").write_text(json.dumps(evidence, indent=2))
    figure_twist(twist, length_rows)
    figure_walls(walls)
    print(json.dumps(evidence, indent=2)[:4000])


if __name__ == "__main__":
    main()