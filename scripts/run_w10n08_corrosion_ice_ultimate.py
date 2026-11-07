"""W10-N08: corrosion-ice multi-damage time-dependent ultimate strength study.

Runs the baseline (survey-consistent Melchers wastage) and a documented severe
(Paik-range deep pitting) ice-belt scenario, writes the machine-readable evidence
payload and the report figures.

Usage::

    PYTHONPATH=src python scripts/run_w10n08_corrosion_ice_ultimate.py [--fast]
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from tensorfem.marine_corrosion_ice_ultimate import (
    PanelConfig, PittingSpec, anchor_sensitivity,
    allowable_pitting_depth, allowable_service_life, calibrate_ice_pressure,
    inspection_schedule,
    intact_reference_check, paik_cross_check, panel_realization,
    run_corrosion_ice_strength_assessment,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "w10n08_corrosion_ice_ultimate"
FIGURES = ROOT / "docs" / "assets" / "marine_corrosion_ice"


def _panel_config(samples: int) -> PanelConfig:
    base = PanelConfig(samples=samples)
    return calibrate_ice_pressure(config=base, utilization=0.55)


def _severe_config(samples: int) -> PanelConfig:
    base = PanelConfig(samples=samples)
    pitting = PittingSpec(
        length_m=base.length_m, width_m=base.width_m,
        thickness_m=base.thickness_m, pitting_intensity=0.25,
        depth_model="depth_ratio", depth_ratio=0.55, reference_years=25.0,
        correlation_length_m=0.30, seed=0,
    )
    return calibrate_ice_pressure(
        config=PanelConfig(**{**base.__dict__, "pitting": pitting}),
        utilization=0.55,
    )


def _figure_field_maps(config: PanelConfig, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 3.4), constrained_layout=True)
    for axis, years in zip(axes, (0.0, 25.0)):
        result, field = panel_realization(config=config, years=years)
        grid = np.asarray(field.thickness_grid) * 1000.0
        image = axis.imshow(grid, origin="lower", aspect="auto", cmap="viridis",
                            extent=(0.0, config.length_m, 0.0, config.width_m))
        axis.set_title(
            f"t = {years:.0f} yr, DOP = {result.dop_raster:.3f}, "
            f"sigma_u = {result.sigma_ultimate_mpa:.1f} MPa"
        )
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        figure.colorbar(image, ax=axis, label="effective thickness [mm]")
    figure.suptitle(
        "Rasterised pitting field (cell-mean thickness, Voigt-effective for bending)"
    )
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _figure_paik(rows, path: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dop_axis = np.linspace(0.0, 0.35, 120)
    figure, axis = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    band = anchor_sensitivity([float(value) for value in dop_axis])
    for entry in band:
        axis.plot(dop_axis, entry["reduction"], color="0.75", linewidth=0.8, zorder=1)
    axis.plot(dop_axis, [1.0 - 0.33 * value for value in dop_axis], color="C3",
              linewidth=2.0, label="Paik-type anchor R = 1 - 0.33 DOP (default)")
    axis.plot([row["dop_realized"] for row in rows],
              [row["reduction_computed"] for row in rows], "o-", color="C0",
              label="field solver (this work)")
    axis.set_xlabel("degree of pitting, DOP [-]")
    axis.set_ylabel("R = sigma_u / sigma_u,intact [-]")
    axis.set_title("Pitting reduction: field solver versus Paik-type anchor")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8, loc="lower left")
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return {"anchor_band_min": min(min(entry["reduction"]) for entry in band),
            "anchor_band_max": max(entry["reduction"][0] for entry in band)}


def _figure_history(history: dict, config: PanelConfig, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    years = [row["years"] for row in history["rows"]]
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), constrained_layout=True)
    axes[0].fill_between(
        years,
        [row["sigma_ultimate_mpa_p05"] for row in history["rows"]],
        [row["sigma_ultimate_mpa_p95"] for row in history["rows"]],
        color="C0", alpha=0.18, label="p05-p95 (100 realisations)",
    )
    axes[0].plot(years, [row["sigma_ultimate_mpa_p50"] for row in history["rows"]],
                 "o-", color="C0", label="sigma_u p50")
    axes[0].plot(years, [config.target_safety_factor * row["sigma_ice_mpa_p95"]
                         for row in history["rows"]], "s--", color="C3",
                 label="gamma x sigma_ice, p95")
    axes[0].set_xlabel("service years")
    axes[0].set_ylabel("stress [MPa]")
    axes[0].set_title("Capacity and ice demand")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].plot(years, [row["utilization_p50"] for row in history["rows"]],
                 "o-", color="C2", label="U p50")
    axes[1].plot(years, [row["utilization_p95"] for row in history["rows"]],
                 "^--", color="C4", label="U p95")
    axes[1].axhline(1.0 / config.target_safety_factor, color="C3", linestyle=":",
                    label=f"1/gamma = {1.0 / config.target_safety_factor:.3f}")
    axes[1].set_xlabel("service years")
    axes[1].set_ylabel("U = sigma_ice / sigma_u [-]")
    axes[1].set_title("Utilisation is monotone non-decreasing")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _figure_convergence(payload: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = payload["grid_convergence"]["rows"]
    figure, axis = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    axis.plot([row["elements_equivalent"] for row in rows],
              [row["sigma_ultimate_mpa"] for row in rows], "o-", color="C0")
    axis.set_xlabel("eigen cell count (grid_x x grid_y)")
    axis.set_ylabel("sigma_u [MPa]")
    axis.set_title("Eigen-grid convergence of one pitted realisation")
    axis.grid(alpha=0.3)
    detail = payload["criteria"]["c_degeneracy_dop_zero"]["detail"]
    axis.annotate(
        f"DOP = 0 vs sibling plate-buckling node: "
        f"{100 * detail['relative_error_vs_sibling']:.2f} %",
        xy=(0.03, 0.06), xycoords="axes fraction", fontsize=8,
    )
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _figure_inspection(schedule, life: dict, config: PanelConfig, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    years = [row["years"] for row in schedule]
    margins = [row["margin_ratio"] for row in schedule]
    figure, axis = plt.subplots(figsize=(6.8, 4.0), constrained_layout=True)
    axis.plot(years, margins, "o-", color="C1", label="margin = sigma_u,p05 / (gamma sigma_ice,p95)")
    axis.axhline(1.0, color="C3", linestyle=":", label="limit")
    if life.get("allowable_years") is not None:
        axis.axvline(life["allowable_years"], color="0.3", linestyle="--",
                     label=f"allowable life {life['allowable_years']:.1f} yr")
    axis.set_xlabel("service years")
    axis.set_ylabel("margin ratio [-]")
    axis.set_title("Inspection criterion trajectory")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true",
                        help="reduced sample counts (smoke run)")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--mc-levels", type=int, nargs=2, default=(500, 1000))
    arguments = parser.parse_args()
    warnings.simplefilter("ignore")
    samples = 12 if arguments.fast else arguments.samples
    levels = (60, 120) if arguments.fast else (arguments.mc_levels[0],
                                              arguments.mc_levels[1])

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    baseline = _panel_config(samples)
    severe = _severe_config(samples)
    payload = run_corrosion_ice_strength_assessment(
        config=baseline, samples=samples, mc_levels=levels,
    )
    severe_payload = run_corrosion_ice_strength_assessment(
        config=severe, samples=samples, mc_levels=levels,
    )
    baseline_depth = allowable_pitting_depth(config=baseline, design_years=25.0)
    severe_depth = allowable_pitting_depth(config=severe, design_years=25.0)
    inverse = {
        "baseline": {
            "allowable_depth": baseline_depth,
            "allowable_life": allowable_service_life(
                history=payload["history"], config=baseline),
            "schedule": inspection_schedule(history=payload["history"], config=baseline),
        },
        "severe": {
            "allowable_depth": severe_depth,
            "allowable_life": allowable_service_life(
                history=severe_payload["history"], config=severe),
            "schedule": inspection_schedule(history=severe_payload["history"],
                                            config=severe),
        },
    }

    anchor_rows = paik_cross_check(config=baseline)
    band = _figure_paik(anchor_rows, FIGURES / "paik_reduction.png")
    _figure_field_maps(severe, FIGURES / "pitting_field.png")
    _figure_history(severe_payload["history"], severe, FIGURES / "margin_history.png")
    _figure_convergence(payload, FIGURES / "convergence.png")
    _figure_inspection(inverse["severe"]["schedule"],
                       inverse["severe"]["allowable_life"], severe,
                       FIGURES / "inspection.png")

    summary = {
        "schema": "tensorfem.w10n08-corrosion-ice-study/1.0",
        "baseline": {
            "panel": payload["panel"],
            "ice": payload["ice"],
            "criteria": {name: {key: value for key, value in entry.items()
                                if key not in {"rows", "detail", "anchor_sensitivity"}}
                         for name, entry in payload["criteria"].items()},
            "grid_convergence": payload["grid_convergence"]["max_error_of_coarser_grids"],
            "history_rows": payload["history"]["rows"],
            "passed": payload["passed"],
        },
        "severe": {
            "panel": severe_payload["panel"],
            "ice": severe_payload["ice"],
            "criteria": {name: {key: value for key, value in entry.items()
                                if key not in {"rows", "detail", "anchor_sensitivity"}}
                         for name, entry in severe_payload["criteria"].items()},
            "grid_convergence": severe_payload["grid_convergence"]["max_error_of_coarser_grids"],
            "history_rows": severe_payload["history"]["rows"],
            "passed": severe_payload["passed"],
        },
        "paik_cross_check": anchor_rows,
        "anchor_band": band,
        "intact_reference": intact_reference_check(config=baseline),
        "inverse": inverse,
        "membrane_screening": {
            "note": "clamped-plate large-deflection screening, kept as a "
                    "cross-check only: the family spread shows the one-parameter "
                    "elastic idealisation is not converged",
            "detail": payload["ice"]["plate_screening_mpa"],
            "spread": payload["ice"]["plate_screening_family_spread"],
        },
    }
    (ARTIFACTS / "assessment.json").write_text(json.dumps(summary, indent=1, default=float))
    figure_notes = {
        "figures": [
            "pitting_field.png", "paik_reduction.png", "margin_history.png",
            "convergence.png", "inspection.png",
        ],
    }
    (ARTIFACTS / "figures.json").write_text(json.dumps(figure_notes, indent=1))
    print(json.dumps({
        "passed": summary["baseline"]["passed"] and summary["severe"]["passed"],
        "baseline_criteria": {name: entry["passed"] for name, entry
                              in summary["baseline"]["criteria"].items()},
        "severe_criteria": {name: entry["passed"] for name, entry
                            in summary["severe"]["criteria"].items()},
        "allowable_life_severe": inverse["severe"]["allowable_life"],
        "allowable_depth_severe": inverse["severe"]["allowable_depth"],
        "artifacts": str(ARTIFACTS),
        "figures": str(FIGURES),
    }, indent=1, default=float))
    return summary


if __name__ == "__main__":
    main()