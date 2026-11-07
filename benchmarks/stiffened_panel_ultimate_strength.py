"""Benchmark: stiffened-panel ultimate strength, ISO 19906 / CSR screening chain.

Run modes
---------
``python benchmarks/stiffened_panel_ultimate_strength.py``
    Use the cached finite-element traces in
    ``artifacts/stiffened_panel_ultimate/`` (regenerate them with ``--run-fe``)
    and rebuild ``evidence.json`` plus the figures.

``python benchmarks/stiffened_panel_ultimate_strength.py --run-fe gate_a gate_b``
    Execute the named arc-length cases (expensive: minutes per case) and cache
    the traces.

The accepted evidence separates

* the **closed-form chain** (plate/stiffener limiting stresses, Smith progressive
  collapse, Navier plate field, yield-line capacity) -- cheap and exercised by
  ``tests/test_stiffened_panel_ultimate.py``;
* the **numerical chain** (elasto-plastic Shell4 unit cell with a stateful
  arc-length path) -- the only source of a peak load.

Nothing in the closed-form chain consumes a numerical result, so the two remain
independent when compared.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

from tensorfem.stiffened_panel_ultimate import (
    IcePressureModel,
    SQUARE_PLATE_PRESSURE_COEFFICIENT,
    StiffenedPanel,
    StructuralUnit,
    csr_hull_girder_check,
    hull_girder_progressive_collapse,
    ice_wave_combined_check,
    iso19906_ice_belt_check,
    panel_load_end_shortening_curve,
    plate_pressure_bending_stress,
    square_plate_coefficient_error,
    yield_line_collapse_pressure,
)
from tensorfem.stiffened_panel_unit_fe import (
    assemble_panel_unit,
    build_panel_unit,
    trace_panel_unit_path,
    trace_summary,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts" / "stiffened_panel_ultimate"
FIGURES = ROOT / "docs" / "assets" / "stiffened_panel_ultimate"


def ice_belt_panel() -> StiffenedPanel:
    """Typical ice-belt panel: t = 20 mm, spacing 800 mm, frame spacing 2.4 m."""
    return StiffenedPanel(span=2.4, spacing=0.800, plate_thickness=0.020,
                          web_height=0.20, web_thickness=0.010,
                          flange_width=0.080, flange_thickness=0.012)


def square_bay_panel() -> StiffenedPanel:
    """Single-bay panel used for the resolved collapse path (beta ~ 2.25)."""
    return StiffenedPanel(span=0.800, spacing=0.800, plate_thickness=0.012,
                          web_height=0.18, web_thickness=0.009,
                          flange_width=0.080, flange_thickness=0.012)


CASES = {
    # name: (panel factory, mesh, arc-length settings)
    "square_bay": (square_bay_panel, dict(nx=4, ny=4, layers=3),
                   dict(steps=40, ds=2.4e-4, load_fraction=0.80, max_iterations=10)),
    "square_bay_fine": (square_bay_panel, dict(nx=4, ny=4, layers=3),
                        dict(steps=64, ds=1.2e-4, load_fraction=0.08, max_iterations=8)),
    "ice_belt_coarse": (ice_belt_panel, dict(nx=4, ny=2, layers=3),
                        dict(steps=24, ds=4e-4, load_fraction=0.30, max_iterations=10)),
    "ice_belt_coarse_fine": (ice_belt_panel, dict(nx=4, ny=2, layers=3),
                             dict(steps=24, ds=2e-4, load_fraction=0.30,
                                  max_iterations=10)),
}


def run_fe_case(name: str) -> dict[str, object]:
    factory, mesh, controls = CASES[name]
    panel = factory()
    case = build_panel_unit(panel, **mesh)
    ds = controls["ds"]
    load_scale = ds / (controls["load_fraction"] * panel.yield_stress)
    trace = trace_panel_unit_path(case, steps=controls["steps"], step_size=ds,
                                  load_scale=load_scale,
                                  max_iterations=controls["max_iterations"])
    summary = trace_summary(case, trace)
    summary["case"] = name
    summary["controls"] = controls
    summary["mesh"] = mesh
    summary["panel_parameters"] = panel.summary()
    summary["free_dofs"] = int(len(case.free_dofs()))
    summary["element_counts"] = list(case.element_counts)
    return summary


def path_status(trace: dict) -> str:
    """Classify a trace: limit point reached, or truncated (last value a bound only)."""
    stresses = [row["stress_over_yield"] for row in trace["points"]]
    if trace.get("has_peak"):
        return "limit_point"
    if len(stresses) < 3:
        return "truncated"
    increments = [b - a for a, b in zip(stresses, stresses[1:])]
    reference = sorted(increments[:-1])[len(increments[:-1]) // 2]
    return "approaching_limit_point" if reference > 0 and increments[-1] / reference < 0.4 \
        else "truncated"


def fe_error(summary: dict, panel: StiffenedPanel) -> dict[str, float]:
    finite = float(summary["ultimate_over_yield"])
    closed = panel.panel_ultimate_stress / panel.yield_stress
    return {"fe_ultimate_over_yield": finite, "closed_form_over_yield": closed,
            "relative_error": abs(finite / closed - 1.0)}


def hull_girder_case(panel: StiffenedPanel) -> dict[str, object]:
    depth = 8.0
    deck = StructuralUnit("deck", 0.020 * 6.0, depth, panel)
    bottom = StructuralUnit("bottom", 0.024 * 6.0, 0.0, panel)
    side = StructuralUnit("side", 0.014 * 4.0, depth / 2.0, panel)
    units = (deck, bottom, side, StructuralUnit("side_mirror", side.area, depth / 2.0, panel))
    collapse = hull_girder_progressive_collapse(units, yield_stress=panel.yield_stress,
                                                young=panel.young)
    section_modulus = collapse["section_inertia"] / (depth / 2.0)
    # A representative vertical wave bending moment: 1.5e5 kN m per metre breadth
    # scaled to the unit width used here (documented assumption, not a rule value).
    applied = 0.55 * float(collapse["upper_bound_fully_plastic_moment"])
    csr = csr_hull_girder_check(units, yield_stress=panel.yield_stress, young=panel.young,
                                vertical_bending_moment=applied, resistance_factor=1.05)
    return {"units": [{"name": u.name, "area": u.area, "z": u.vertical_position}
                      for u in units],
            "collapse": {k: collapse[k] for k in
                         ("neutral_axis", "section_inertia", "ultimate_moment",
                          "ultimate_curvature", "upper_bound_fully_plastic_moment",
                          "post_peak", "plateau_reached")},
            "rows": [{"curvature": row["curvature"], "moment": row["moment"]}
                     for row in collapse["rows"]],
            "csr_check": {k: csr[k] for k in ("applied_moment", "ultimate_moment",
                                              "design_resistance", "utilisation", "passed")},
            "section_modulus": section_modulus,
            "applied_moment": applied}


def ice_belt_case(panel: StiffenedPanel) -> dict[str, object]:
    # Coefficients are caller supplied (documented assumption); the standard
    # itself must be consulted for design values.
    model = IcePressureModel(reference_pressure=0.6e6, reference_area=1.0,
                             exponent=-0.3, label="assumed representative (not a rule value)")
    check = iso19906_ice_belt_check(panel, model)
    navier = plate_pressure_bending_stress(panel.span, panel.spacing,
                                           panel.plate_thickness, check["ice_pressure"],
                                           young=panel.young, poisson=panel.poisson)
    coefficient = square_plate_coefficient_error(terms=61)
    yield_line = yield_line_collapse_pressure(panel.span, panel.spacing,
                                              panel.plate_thickness, panel.yield_stress)
    square_yield_line = yield_line_collapse_pressure(1.0, 1.0, panel.plate_thickness,
                                                     panel.yield_stress)
    combined = ice_wave_combined_check(
        panel, model, wave_bending_moment=0.55 * panel.panel_ultimate_stress * 1.0,
        section_modulus=1.0)
    # Allowable local pressure implied by the structural side of the chain.
    allowable_factor = 0.8
    stress_per_pressure = check["maximum_bending_stress"] / check["ice_pressure"]
    allowable_pressure_elastic = allowable_factor * panel.yield_stress / stress_per_pressure
    return {"ice_pressure": check["ice_pressure"],
            "allowable_pressure_elastic": allowable_pressure_elastic,
            "allowable_pressure_plastic": check["plastic_collapse_pressure"],
            "allowable_factor": allowable_factor,
            "stress_per_pressure": stress_per_pressure,
            "load_area": check["load_area"],
            "maximum_bending_stress": check["maximum_bending_stress"],
            "allowable_stress": check["allowable_stress"],
            "allowable_utilisation": check["allowable_utilisation"],
            "uls_utilisation": check["uls_utilisation"],
            "plastic_collapse_pressure": check["plastic_collapse_pressure"],
            "governing_mechanism": check["governing_mechanism"],
            "plate_field_coefficient": navier["coefficient_m_over_pb2"],
            "textbook_square_coefficient": SQUARE_PLATE_PRESSURE_COEFFICIENT,
            "series_error": coefficient["relative_error"],
            "yield_line_aspect": yield_line["aspect_ratio"],
            "square_yield_line_ratio": float(square_yield_line["collapse_pressure"])
            / float(square_yield_line["square_plate_closed_form"]),
            "combined_check": {k: combined[k] for k in
                               ("in_plane_utilisation", "bending_utilisation",
                                "combined_utilisation", "interaction", "passed")}}


def path_convergence(coarse: dict, fine: dict, samples: int = 21) -> dict[str, float] | None:
    """Compare two arc-length paths on their common end-shortening range.

    Comparing peaks is only possible when both paths reach a limit point; on a
    monotone branch the honest comparison is the stress discrepancy at matched
    end shortening.
    """
    import numpy as np
    u_coarse = [row["end_shortening"] for row in coarse["points"]]
    s_coarse = [row["stress_over_yield"] for row in coarse["points"]]
    u_fine = [row["end_shortening"] for row in fine["points"]]
    s_fine = [row["stress_over_yield"] for row in fine["points"]]
    low = max(u_coarse[0], u_fine[0])
    high = min(u_coarse[-1], u_fine[-1])
    if high <= low:
        return None
    grid = [low + (high - low) * i / (samples - 1) for i in range(samples)]
    a = np.interp(grid, u_coarse, s_coarse)
    b = np.interp(grid, u_fine, s_fine)
    difference = np.abs(a - b) / np.maximum(np.abs(b), 1e-12)
    return {"overlap_up_to": high, "samples": samples,
            "maximum_relative_difference": float(difference.max()),
            "mean_relative_difference": float(difference.mean())}


def field_snapshot(panel: StiffenedPanel, *, stress_fraction: float,
                   pressure: float, nx: int = 4, ny: int = 4,
                   layers: int = 3) -> dict[str, object]:
    """Linear-elastic Shell4 field under combined compression and ice pressure."""
    case = build_panel_unit(panel, nx=nx, ny=ny, layers=layers)
    free = case.free_dofs()
    zero = torch.zeros(case.n_dofs, dtype=torch.float64)
    _internal, tangent, _stress, _state = assemble_panel_unit(case, zero, case.virgin_state())
    load = stress_fraction * panel.yield_stress * case.compression_load \
        + pressure * case.pressure_load
    displacement = torch.zeros(case.n_dofs, dtype=torch.float64)
    displacement[free] = torch.linalg.solve(tangent[free][:, free], load[free])
    count = (nx + 1) * (ny + 1)
    nodes = case.nodes[:count]
    return {"x": nodes[:, 0].tolist(), "y": nodes[:, 1].tolist(),
            "w": displacement[0:6 * count:6].tolist(), "nx": nx, "ny": ny,
            "tip_w": displacement[case.probe_dof].item(),
            "stress_fraction": stress_fraction, "pressure": pressure}


def build_figures(evidence: dict, outdir: pathlib.Path) -> list[dict[str, str]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    outdir.mkdir(parents=True, exist_ok=True)
    figures: list[dict[str, str]] = []

    # 1. collapse path
    panel = square_bay_panel()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, style in (("square_bay", "-o"), ("square_bay_fine", "-s"),
                        ("ice_belt_coarse", "-^")):
        trace = evidence["finite_element"].get(name)
        if not trace:
            continue
        u = [row["end_shortening"] * 1e3 for row in trace["points"]]
        s = [row["stress_over_yield"] for row in trace["points"]]
        axes[0].plot(u, s, style, markersize=3.5, label=name)
        if trace["has_peak"]:
            axes[0].plot(u[trace["ultimate_step"] - 1], s[trace["ultimate_step"] - 1],
                         "*", markersize=14, label=f"peak {name}")
    axes[0].axhline(panel.panel_ultimate_stress / panel.yield_stress, color="k",
                    linestyle="--", linewidth=1,
                    label=r"closed form $\sigma_u/\sigma_y$")
    axes[0].set_xlabel("end shortening [mm]")
    axes[0].set_ylabel(r"average stress $\sigma/\sigma_y$")
    axes[0].set_title("Shell4 arc-length collapse path")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    betas = [1.0 + 0.02 * i for i in range(101)]
    faulkner = [min(1.0, 2.0 / b - 1.0 / b**2) for b in betas]
    axes[1].plot(betas, faulkner, "k-", label="Faulkner plate ultimate")
    for panel_case, marker in ((ice_belt_panel(), "o"), (square_bay_panel(), "s")):
        axes[1].plot(panel_case.plate_slenderness,
                     panel_case.faulkner_plate_strength / panel_case.yield_stress,
                     marker, markersize=8,
                     label=f"plate component, t={panel_case.plate_thickness*1e3:.0f} mm")
        axes[1].plot(panel_case.plate_slenderness,
                     panel_case.panel_ultimate_stress / panel_case.yield_stress,
                     marker, markerfacecolor="none", markersize=11,
                     label=f"panel (area weighted), t={panel_case.plate_thickness*1e3:.0f} mm")
    for name, trace in evidence["finite_element"].items():
        axes[1].plot(trace["panel_parameters"]["plate_slenderness"],
                     trace["ultimate_over_yield"], "r*", markersize=13,
                     label=f"FE {name}")
    axes[1].set_xlabel(r"plate slenderness $\beta = (b/t)\sqrt{\sigma_y/E}$")
    axes[1].set_ylabel(r"$\sigma_u/\sigma_y$")
    axes[1].set_title("Component and panel limiting strength")
    axes[1].legend(fontsize=6)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    path = outdir / "collapse_path.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figures.append({"name": "collapse_path", "path": str(path),
                    "caption": "Left: Shell4 arc-length collapse paths (average stress vs end "
                               "shortening) with the closed-form panel ultimate. Right: Faulkner "
                               "plate ultimate, area-weighted panel ultimate and the FE peaks "
                               "against plate slenderness."})

    # 2. hull girder moment-curvature
    hull = evidence["hull_girder"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    curvatures = [row["curvature"] for row in hull["rows"]]
    moments = [abs(row["moment"]) / 1e6 for row in hull["rows"]]
    ax.plot([c * 1e3 for c in curvatures], moments, "-o", markersize=3)
    ax.axhline(hull["collapse"]["upper_bound_fully_plastic_moment"] / 1e6, color="k",
               linestyle="--", label="fully plastic upper bound $M_p$")
    ax.axhline(hull["csr_check"]["ultimate_moment"] / 1e6, color="C1", linestyle=":",
               label="$M_u$ (Smith progressive collapse)")
    ax.axhline(hull["csr_check"]["applied_moment"] / 1e6, color="C3", linestyle="-.",
               label="applied design moment")
    ax.set_xlabel(r"curvature $\kappa$ [1/km]")
    ax.set_ylabel("section moment [MN m]")
    ax.set_title("Hull-girder progressive collapse (unit-cell model)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = outdir / "hull_girder_moment_curvature.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figures.append({"name": "hull_girder_mc", "path": str(path),
                    "caption": "Moment-curvature from the Smith incremental-curvature method with "
                               "limiting-stress unit curves, compared with the fully plastic "
                               "upper bound and the applied design moment."})

    # 3. plate field under ice pressure + compression
    field = evidence["field_snapshot"]
    import numpy as _np
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    triang = mtri.Triangulation(_np.array(field["x"]), _np.array(field["y"]))
    contour = ax.tricontourf(triang, _np.array(field["w"]) * 1e3, levels=14, cmap="viridis")
    ax.tricontour(triang, _np.array(field["w"]) * 1e3, levels=14, colors="k", linewidths=0.3)
    fig.colorbar(contour, ax=ax, label="transverse deflection [mm]")
    ax.set_xlabel("x [m] (frame to mid-bay)")
    ax.set_ylabel("y [m] (stiffener to mid-bay)")
    ax.set_title(f"Plate field, $\\sigma={field['stress_fraction']:.2f}\\sigma_y$ + "
                 f"$p_{{ice}}={field['pressure']/1e3:.0f}$ kPa")
    ax.set_aspect("equal")
    fig.tight_layout()
    path = outdir / "plate_field_deflection.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figures.append({"name": "plate_field", "path": str(path),
                    "caption": "Linear-elastic Shell4 plate-field deflection under the combined "
                               "in-plane compression and local ice pressure of the ISO 19906 "
                               "screening case (half-bay model, stiffener at y = 0)."})

    # 4. ice-wave interaction
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    panel = ice_belt_panel()
    pressures = [0.1e6 + 0.2e6 * i for i in range(16)]
    for moment, label in ((0.0, "no wave moment"),
                          (0.5 * panel.panel_ultimate_stress, r"$M=0.5\sigma_u Z$"),
                          (0.8 * panel.panel_ultimate_stress, r"$M=0.8\sigma_u Z$")):
        values = []
        for p in pressures:
            combined = ice_wave_combined_check(
                panel, IcePressureModel(p, 1.0, -0.3), wave_bending_moment=moment,
                section_modulus=1.0)
            values.append(combined["combined_utilisation"])
        ax.plot([p / 1e3 for p in pressures], values, label=label)
    ax.axhline(1.0, color="k", linestyle="--", linewidth=1, label="utilisation limit")
    ax.set_xlabel("local ice pressure [kPa]")
    ax.set_ylabel("combined utilisation [-]")
    ax.set_title("Ice + wave combined screening (quadratic interaction)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = outdir / "ice_wave_interaction.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figures.append({"name": "ice_wave", "path": str(path),
                    "caption": "Combined ice-belt utilisation against local ice pressure for three "
                               "levels of wave-induced hull-girder bending; the quadratic "
                               "interaction surface of :func:`ice_wave_combined_check` is used."})
    return figures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-fe", nargs="*", default=None,
                        help="execute the named arc-length cases and cache their traces")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--cache", default=None,
                        help="write this trace cache instead of the shared one "
                             "(allows concurrent case runs)")
    args = parser.parse_args(argv)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    cache = pathlib.Path(args.cache) if args.cache else ARTIFACTS / "fe_traces.json"
    traces: dict[str, dict] = {}
    sources = [ARTIFACTS / "fe_traces.json"] + sorted(ARTIFACTS.glob("fe_trace_*.json"))
    for source in sources:
        if source.exists():
            traces.update(json.loads(source.read_text()))
    if cache.exists():
        traces.update(json.loads(cache.read_text()))
    if args.run_fe is not None:
        names = args.run_fe or list(CASES)
        for name in names:
            if name not in CASES:
                raise SystemExit(f"unknown case {name}")
            traces[name] = run_fe_case(name)
            cache.write_text(json.dumps(traces, indent=1))

    ice_belt = ice_belt_panel()
    square_bay = square_bay_panel()
    closed_form = {
        "ice_belt": ice_belt.summary(),
        "square_bay": square_bay.summary(),
        "ice_belt_curve": panel_load_end_shortening_curve(ice_belt),
        "series_coefficient_error": square_plate_coefficient_error(terms=61),
        "yield_line_square": yield_line_collapse_pressure(1.0, 1.0, 0.020, ice_belt.yield_stress),
    }

    finite_element = {name: trace for name, trace in traces.items()}
    status = {name: path_status(trace) for name, trace in finite_element.items()}
    # Only traces that reached (or are converging on) a limit point carry an
    # ultimate strength; a truncated path's last load factor is a lower bound.
    errors = {name: fe_error(trace, CASES[name][0]())
              for name, trace in finite_element.items()
              if status[name] in ("limit_point", "approaching_limit_point")}

    convergence = None
    for coarse_name, fine_name in (("ice_belt_coarse", "ice_belt_coarse_fine"),
                                   ("square_bay", "square_bay_fine")):
        if coarse_name in finite_element and fine_name in finite_element:
            coarse, fine = finite_element[coarse_name], finite_element[fine_name]
            path = path_convergence(coarse, fine)
            if path is not None:
                convergence = {
                    "case": coarse_name,
                    "coarse_step": coarse["controls"]["ds"],
                    "fine_step": fine["controls"]["ds"],
                    "coarse_ultimate_over_yield": coarse["ultimate_over_yield"],
                    "fine_ultimate_over_yield": fine["ultimate_over_yield"],
                    "relative_change": path["maximum_relative_difference"],
                    "mean_relative_change": path["mean_relative_difference"],
                    "overlap_up_to": path["overlap_up_to"],
                    "method": "stress at matched end shortening over the common range",
                }
                break

    ice = ice_belt_case(ice_belt)
    hull = hull_girder_case(ice_belt)
    field = field_snapshot(ice_belt, stress_fraction=0.5, pressure=ice["ice_pressure"], nx=4, ny=4)

    criteria = {
        "fe_matches_closed_form_within_10_percent": bool(
            errors and max(row["relative_error"] for row in errors.values()) < 0.10),
        "arc_length_step_convergence_within_15_percent": bool(
            convergence and convergence["relative_change"] < 0.15),
        "collapse_path_shows_peak": bool(
            finite_element and any(t["has_peak"] for t in finite_element.values())),
        "fe_approaches_a_limit_point": bool(
            any(value in ("limit_point", "approaching_limit_point")
                for value in status.values())),
        "plate_field_series_within_5_percent_of_textbook": bool(
            closed_form["series_coefficient_error"]["relative_error"] < 0.05),
        "yield_line_square_matches_24Mp_over_a2": bool(
            abs(closed_form["yield_line_square"]["collapse_pressure"]
                / closed_form["yield_line_square"]["square_plate_closed_form"] - 1.0) < 0.05),
        "smith_1977_experimental_comparison": "WITHHELD",
        "abaqus_riks_comparison": "WITHHELD",
        "iso19906_normative_allowable_pressure": "WITHHELD",
    }
    evidence = {
        "schema": "tensorfem.stiffened-panel-ultimate/1.0",
        "geometry_check": {
            "ice_belt_panel": "t = 20 mm, spacing = 800 mm, span = 2.4 m, web 200x10, "
                              "flange 80x12, E = 206 GPa, sigma_y = 235 MPa",
            "square_bay_panel": "t = 12 mm, spacing = 800 mm, span = 0.8 m, web 180x9, "
                                "flange 80x12",
        },
        "closed_form": closed_form,
        "finite_element": finite_element,
        "fe_vs_closed_form": errors,
        "path_status": status,
        "step_convergence": convergence,
        "iso19906": ice,
        "hull_girder": hull,
        "field_snapshot": field,
        "criteria": criteria,
        "withheld": {
            "smith_1977": "The Smith (1977) stiffened-panel test series is not available in this "
                          "environment (no copy of the paper or its digitised data). The FE/closed-form "
                          "comparison and the mesh/step convergence evidence stand in for it; no "
                          "experimental number is quoted from memory.",
            "abaqus_riks": "No ABAQUS licence or Riks solution is available here. The arc-length "
                           "trajectory is instead checked by step-size convergence and by the elastic "
                           "limit of the same driver against tensorfem.arc_length.solve_arc_length.",
        },
    }
    if not args.skip_figures:
        evidence["figures"] = build_figures(evidence, FIGURES)
    (ARTIFACTS / "evidence.json").write_text(json.dumps(evidence, indent=1))
    print(json.dumps({"criteria": criteria, "errors": errors,
                      "convergence": convergence,
                      "ice": {k: ice[k] for k in ("ice_pressure", "maximum_bending_stress",
                                                  "allowable_utilisation", "uls_utilisation")},
                      "hull": hull["csr_check"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())