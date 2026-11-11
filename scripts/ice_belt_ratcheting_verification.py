"""Ice-belt ratcheting / Bree shakedown verification sweep (P226W3-N05 / m14467).

Produces ``artifacts/ice_belt_ratcheting/verification.json`` (every number the
report quotes, generated programmatically -- no hand transcription) and four
figures:

1. ``bree_diagram.png``   -- region map from the incremental solver + the exact
   elastic limit ``x + y = 1`` + the closed-form shakedown limit ``x + y/4 = 1``
   + the Melan LP boundary points;
2. ``ratcheting_rate.png`` -- per-cycle growth histories and the step/fibre
   convergence of the steady ratcheting rate;
3. ``deflection_history.png`` -- permanent deflection vs. cycle number for the
   ice-pressure overload ladder with the ``L/150`` serviceability limit;
4. ``ice_spectrum.png`` -- the pressure-area relation and the ice impact block
   spectrum.

Run:  PYTHONPATH=src python scripts/ice_belt_ratcheting_verification.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfem.ice_belt_ratcheting import (
    IceBeltPlate,
    analytic_elastic_x,
    analytic_shakedown_x,
    bree_cycles,
    bree_region,
    deflection_to_membrane_strain,
    first_cycle_plastic_deflection,
    ice_belt_deflection_history,
    ice_belt_service_life,
    ice_collision_spectrum,
    ice_pressure,
    illustrative_class_pressure,
    melan_boundary,
    melan_shakedown,
    membrane_deflection,
)

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "ice_belt_ratcheting"
OUT.mkdir(parents=True, exist_ok=True)

PLATE = IceBeltPlate(span_m=0.35, thickness_m=0.020, young_pa=206.0e9,
                     yield_pa=355.0e6, edge_condition="clamped",
                     deflection_limit_ratio=1.0 / 150.0)
DESIGN_PRESSURE_MPA = 3.0


def section_bree_diagram(payload: dict) -> None:
    xs = np.round(np.linspace(0.05, 0.95, 16), 4)
    ys = np.round(np.linspace(0.05, 2.60, 14), 4)
    regions = [[bree_region(float(x), float(y), cycles=8, steps=6, fibres=41)
                for x in xs] for y in ys]
    counts: dict[str, int] = {}
    for row in regions:
        for v in row:
            counts[v] = counts.get(v, 0) + 1
    payload["bree_diagram"] = {
        "x_grid": xs.tolist(),
        "y_grid": ys.tolist(),
        "regions": regions,
        "region_counts": counts,
        "grid_points": int(sum(counts.values())),
        "elastic_limit": "x + y = 1 (exact)",
        "shakedown_limit": "x + y/4 = 1 (closed form, Melan with zero-mean residual)",
    }

    codes = {"E": 0, "S": 1, "P": 2, "R": 3, "COLLAPSE": 4}
    grid = np.array([[codes[v] for v in row] for row in regions], dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    cmap = matplotlib.colors.ListedColormap(
        ["#e8f4fb", "#a8d5a2", "#f4c97a", "#e8877a", "#4d4d4d"])
    ax.pcolormesh(xs, ys, grid, cmap=cmap, vmin=-0.5, vmax=4.5,
                  shading="nearest", edgecolors="white", linewidth=0.3)
    xx = np.linspace(0.0, 1.0, 200)
    ax.plot(xx, np.maximum(0.0, 1.0 - xx), "k-", lw=2.0,
            label="elastic limit  x + y = 1 (exact)")
    ax.plot(xx, np.maximum(0.0, 4.0 * (1.0 - xx)), "k--", lw=2.0,
            label="shakedown limit  x + y/4 = 1 (closed form)")
    ys_lp = np.linspace(0.05, 2.6, 25)
    ax.plot([melan_boundary(float(y), fibres=41) for y in ys_lp], ys_lp, "o",
            ms=3.2, color="b", label="Melan LP boundary (bisection)")
    legend = [matplotlib.patches.Patch(color=c, label=n) for n, c in
              (("E elastic", "#e8f4fb"), ("S shakedown", "#a8d5a2"),
               ("P plastic cycling", "#f4c97a"), ("R ratcheting", "#e8877a"))]
    ax.legend(handles=legend + ax.get_legend_handles_labels()[0],
              loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_xlabel(r"primary membrane stress ratio  $x=\sigma_p/\sigma_y$")
    ax.set_ylabel(r"secondary stress range ratio  $y=\Delta\sigma_t/\sigma_y$")
    ax.set_title("Bree diagram from the incremental EPP solver\n"
                 "(region = mechanical output, boundaries = analytic anchors)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 2.6)
    fig.tight_layout()
    fig.savefig(OUT / "bree_diagram.png", dpi=160)
    plt.close(fig)


def section_shakedown_checks(payload: dict) -> None:
    rows = []
    worst = 0.0
    for y in (0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.4):
        lp = melan_boundary(float(y), fibres=41)
        closed = analytic_shakedown_x(float(y))
        err = abs(lp - closed)
        worst = max(worst, err)
        rows.append({"y": y, "melan_lp_x_max": lp, "closed_form": closed,
                     "abs_error": err})
    payload["shakedown_boundary"] = {"rows": rows, "max_abs_error": worst}

    xs = (0.15, 0.35, 0.55, 0.75, 0.92)
    ys = (0.2, 0.6, 1.0, 1.5, 2.0, 2.4)
    agree, total, mismatches = 0, 0, []
    for y in ys:
        for x in xs:
            region = bree_region(x, y, cycles=6, steps=6, fibres=41)
            shakes = melan_shakedown(x, y, fibres=41)
            total += 1
            if (region in ("E", "S")) == shakes:
                agree += 1
            else:
                mismatches.append({"x": x, "y": y, "region": region,
                                   "melan_shakedown": shakes})
    payload["incremental_vs_melan"] = {
        "agreement": agree, "total": total, "fraction": agree / total,
        "mismatches": mismatches}

    elastic_rows = []
    for x, y in ((0.10, 0.20), (0.50, 0.40), (0.85, 0.10),
                 (0.20, 0.90), (0.60, 0.45), (0.90, 0.15)):
        res = bree_cycles(x, y, cycles=4, steps=6, fibres=41)
        elastic_rows.append({"x": x, "y": y, "x_plus_y": x + y,
                             "region": res.region,
                             "plastic_strain_peak": res.plastic_strain_peak})
    payload["elastic_limit"] = elastic_rows


def section_ratcheting_rate(payload: dict) -> None:
    cases = [{"x": 0.95, "y": 0.30}, {"x": 0.90, "y": 0.80},
             {"x": 0.85, "y": 1.00}, {"x": 0.75, "y": 1.40},
             {"x": 0.65, "y": 0.90}]
    histories = []
    for c in cases:
        res = bree_cycles(c["x"], c["y"], cycles=14, steps=6, fibres=41)
        histories.append({
            "x": c["x"], "y": c["y"], "region": res.region,
            "growth_per_cycle": res.growth_per_cycle,
            "growth_history": list(res.growth_history),
            "plastic_range": res.plastic_range,
        })
    payload["ratcheting"] = {"cases": histories}

    conv = []
    for steps in (4, 8, 16, 32):
        for fibres in (21, 41, 81):
            res = bree_cycles(0.90, 0.80, cycles=10, steps=steps, fibres=fibres)
            conv.append({"steps": steps, "fibres": fibres,
                         "growth_per_cycle": res.growth_per_cycle})
    payload["ratcheting"]["convergence"] = conv

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    ax = axes[0]
    for h in histories:
        if h["region"] == "R":
            ax.plot(np.arange(1, len(h["growth_history"]) + 1),
                    h["growth_history"], "o-", ms=3,
                    label=f"x={h['x']}, y={h['y']} ({h['region']})")
    ax.set_xlabel("cycle")
    ax.set_ylabel("membrane growth per cycle  $\\Delta e$ (normalised)")
    ax.set_title("Steady ratcheting rate (plateau = steady state)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]
    for fibres, marker in zip((21, 41, 81), ("o", "s", "^")):
        subset = [c for c in conv if c["fibres"] == fibres]
        ax.plot([c["steps"] for c in subset],
                [c["growth_per_cycle"] for c in subset], marker + "-",
                label=f"{fibres} fibres")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("load steps per half cycle")
    ax.set_ylabel("steady growth per cycle")
    ax.set_title("Step / fibre convergence at x=0.90, y=0.80")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "ratcheting_rate.png", dpi=160)
    plt.close(fig)


def section_spectrum(payload: dict) -> None:
    areas = np.array([0.1, 0.25, 0.5, 1.0, 2.0, 5.0])
    pressure = ice_pressure(areas, p_ref_mpa=1.6, area_exponent=0.3)
    class_pressures = {pc: illustrative_class_pressure(pc, 2.0)
                       for pc in range(1, 8)}
    p_design = class_pressures[4]
    spec = ice_collision_spectrum(design_pressure_mpa=DESIGN_PRESSURE_MPA,
                                  impacts_per_voyage=60, voyages=5, n_bins=8,
                                  loaded_area_m2=0.5)
    amps = spec.amplitudes()
    counts = spec.counts()
    payload["ice_pressure"] = {
        "p_ref_mpa": 1.6, "area_exponent": 0.3,
        "areas_m2": areas.tolist(), "pressure_mpa": pressure.tolist(),
        "loglog_slope": float(np.polyfit(np.log(areas), np.log(pressure), 1)[0]),
        "illustrative_class_pressure_pc1_mpa": 2.0,
        "illustrative_class_table_mpa": class_pressures,
        "provenance": "ILLUSTRATIVE class factors; p_ref is a user input",
        "spectrum": {
            "design_pressure_mpa": DESIGN_PRESSURE_MPA,
            "impacts_per_voyage": 60, "voyages": 5,
            "weibull_shape": spec.weibull_shape,
            "bins_mpa": amps.tolist(), "counts": counts.tolist(),
            "total_cycles": spec.total_cycles,
            "weighted_mean_mpa": float(np.sum(counts * amps) / spec.total_cycles),
        },
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    ax = axes[0]
    ax.loglog(areas, pressure, "o-", color="#1f6f9c")
    for a, p in zip(areas[::2], pressure[::2]):
        ax.annotate(f"{p:.2f} MPa", (a, p), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)
    ax.set_xlabel("loaded area A (m$^2$)")
    ax.set_ylabel("average ice pressure p (MPa)")
    ax.set_title("IACS-style pressure-area relation\n"
                 "$p=C_dC_fp_{ref}(A/A_{ref})^{-0.3}$")
    ax.grid(alpha=0.3, which="both")
    ax = axes[1]
    ax.bar(np.arange(len(amps)), counts, color="#c0504d", alpha=0.85)
    ax.set_xticks(np.arange(len(amps)))
    ax.set_xticklabels([f"{a:.2f}" for a in amps], rotation=60, fontsize=7)
    ax.set_xlabel("ice pressure amplitude (MPa)")
    ax.set_ylabel("cycles over 5 voyages")
    ax.set_title(f"Block spectrum (total {spec.total_cycles} cycles,\n"
                 f"mean {np.sum(counts * amps) / spec.total_cycles:.2f} MPa)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "ice_spectrum.png", dpi=160)
    plt.close(fig)
    return spec


def section_deflection(payload: dict, spec) -> None:
    p_yield_mpa = PLATE.yield_pressure_pa() / 1.0e6
    ladder = [2.0, 2.5, 3.0]
    rows = []
    histories = {}
    for p in ladder:
        hist = ice_belt_deflection_history(PLATE, p, cycles=200, fibres=41)
        histories[p] = hist
        rows.append({
            "pressure_mpa": p,
            "secondary_ratio": float(hist.secondary_ratio[0]),
            "first_cycle_set_mm": first_cycle_plastic_deflection(
                PLATE, p * 1e6) * 1e3,
            "deflection_after_200_mm": hist.final_deflection() * 1e3,
            "deflection_over_span": hist.final_ratio(PLATE.span_m),
            "primary_ratio_end": float(hist.primary_ratio[-1]),
            "secondary_ratio_end": float(hist.secondary_ratio[-1]),
            "plateau_reached": hist.plateau_reached,
            "cycles_run": int(hist.cycles[-1]),
            "regions": sorted(set(hist.region)),
        })
    payload["deflection"] = {
        "plate": {
            "span_m": PLATE.span_m, "thickness_m": PLATE.thickness_m,
            "young_pa": PLATE.young_pa, "yield_pa": PLATE.yield_pa,
            "edge_condition": PLATE.edge_condition,
            "deflection_limit_m": PLATE.deflection_limit_m,
            "first_yield_pressure_mpa": p_yield_mpa,
            "yield_deflection_mm":
                PLATE.elastic_deflection(PLATE.yield_pressure_pa()) * 1e3,
        },
        "ladder": rows,
        "membrane_strain_check": {
            "w_mm": 1.0, "strain": deflection_to_membrane_strain(1e-3,
                                                                 PLATE.span_m),
            "round_trip_mm": membrane_deflection(
                deflection_to_membrane_strain(1e-3, PLATE.span_m),
                PLATE.span_m) * 1e3,
        },
    }

    spec_mild = ice_collision_spectrum(design_pressure_mpa=2.5,
                                       impacts_per_voyage=60, voyages=5,
                                       n_bins=8, loaded_area_m2=0.5)
    result = ice_belt_service_life(PLATE, spec, cycles_per_bin=200, fibres=41)
    mild = ice_belt_service_life(PLATE, spec_mild, cycles_per_bin=200, fibres=41)
    payload["service_life_mild"] = {
        "design_pressure_mpa": 2.5,
        "total_deflection_mm": mild.total_deflection_m * 1e3,
        "limit_mm": mild.limit_m * 1e3,
        "utilisation": mild.utilisation,
        "cycles_to_limit": mild.cycles_to_limit,
        "region_counts": mild.region_counts,
        "acceptable": mild.acceptable,
        "per_bin": [{"pressure_mpa": h.pressure_mpa,
                     "cycles": int(h.cycles[-1]),
                     "deflection_mm": h.final_deflection() * 1e3,
                     "plateau": h.plateau_reached} for h in mild.histories],
    }
    payload["service_life"] = {
        "design_pressure_mpa": DESIGN_PRESSURE_MPA,
        "spectrum_total_cycles": spec.total_cycles,
        "per_bin": [{"pressure_mpa": h.pressure_mpa,
                     "cycles": int(h.cycles[-1]),
                     "deflection_mm": h.final_deflection() * 1e3,
                     "plateau": h.plateau_reached}
                    for h in result.histories],
        "total_deflection_mm": result.total_deflection_m * 1e3,
        "limit_mm": result.limit_m * 1e3,
        "utilisation": result.utilisation,
        "cycles_to_limit": result.cycles_to_limit,
        "region_counts": result.region_counts,
        "max_primary_ratio": result.max_primary_ratio,
        "max_secondary_ratio": result.max_secondary_ratio,
        "acceptable": result.acceptable,
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    ax = axes[0]
    for p in ladder:
        h = histories[p]
        ax.plot(h.cycles, h.deflection_m * 1e3, "-",
                label=f"p = {p} MPa  (y0 = {h.secondary_ratio[0]:.2f})")
    ax.axhline(PLATE.deflection_limit_m * 1e3, color="r", ls="--", lw=1.6,
               label="serviceability limit  L/150")
    ax.set_xlabel("ice pressure cycle")
    ax.set_ylabel("permanent deflection (mm)")
    ax.set_title("Plate-level accumulated deflection\n(constant amplitude ladder)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]
    for p in ladder:
        h = histories[p]
        ax.plot(h.cycles, h.primary_ratio, "-",
                label=f"x, p = {p} MPa")
        ax.plot(h.cycles, h.secondary_ratio, "--",
                label=f"y, p = {p} MPa")
    ax.set_xlabel("ice pressure cycle")
    ax.set_ylabel("stress ratios (x primary, y secondary)")
    ax.set_title("Load path in the Bree space\n(membrane action sheds bending)")
    ax.legend(fontsize=6.5, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "deflection_history.png", dpi=160)
    plt.close(fig)


def main() -> dict:
    payload: dict = {"mission": "P226W3-N05 / m14467",
                     "repo": "TensorFEM",
                     "module": "src/tensorfem/ice_belt_ratcheting.py"}
    section_bree_diagram(payload)
    section_shakedown_checks(payload)
    section_ratcheting_rate(payload)
    spec = section_spectrum(payload)
    section_deflection(payload, spec)
    (OUT / "verification.json").write_text(json.dumps(payload, indent=2))

    summary = {
        "elastic_limit_ok": all((r["region"] == "E") == (r["x_plus_y"] < 1.0)
                                for r in payload["elastic_limit"]),
        "shakedown_max_abs_error": payload["shakedown_boundary"]["max_abs_error"],
        "incremental_vs_melan": payload["incremental_vs_melan"]["fraction"],
        "diagram_counts": payload["bree_diagram"]["region_counts"],
        "ratcheting": [(c["x"], c["y"], c["region"], c["growth_per_cycle"])
                       for c in payload["ratcheting"]["cases"]],
        "service": (payload["service_life"]["total_deflection_mm"],
                    payload["service_life"]["utilisation"],
                    payload["service_life"]["cycles_to_limit"]),
    }
    print(json.dumps(summary, indent=2))
    print(f"artifacts written to {OUT}")
    return payload


if __name__ == "__main__":
    main()