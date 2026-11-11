"""Clear the sharp-elliptical-hole tip verification debt (``K_t = 7``).

The Inglis report (``docs/elliptic-hole-inglis-validation.md``) left one debt open:
the *transverse* tip of a hole elongated transverse to the load
(``semi_x/semi_y = 1/3``, ``K_t = 7``) only reached ``8.87 %`` peak error on the
published ladder ``n_theta = n_rho = 24..48`` at ``grading = 3``, with an observed
order close to one -- i.e. an honest FAIL of the 3 % gate, with the note that the
shape "must be locally refined before the peak may be quoted".

This script tests *which* discretisation parameter that local refinement has to
move.  It sweeps two independent mesh controls -- the radial grading strength
``grading`` (exponential clustering towards the hole) and the mesh level ``n`` --
over both hole shapes and correlates the peak error with the physical size of the
first element at the tip, measured in units of the tip curvature radius
``rho_t = semi_x^2 / semi_y``.

Outputs (all numbers recomputed here, nothing transcribed):

* ``benchmarks/elliptic_hole_sharp_tip.json``      machine-readable evidence
* ``docs/assets/elliptic_hole_sharp_tip/*.png``    report figures

    OMP_NUM_THREADS=4 PYTHONPATH=src python scripts/audit_elliptic_hole_sharp_tip.py
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensorfem.elliptic_hole import (  # noqa: E402
    InglisHole,
    solve_case,
    tip_resolution,
)

REPO = pathlib.Path(__file__).resolve().parents[1]
ASSETS = REPO / "docs" / "assets" / "elliptic_hole_sharp_tip"
JSON_PATH = REPO / "benchmarks" / "elliptic_hole_sharp_tip.json"

RHO_OUTER = 5.0
GRADINGS = (3.0, 5.0, 6.0, 9.0)
LEVELS = (24, 32, 48)          # n=64 of this ladder was measured once by hand
#                              (22 s/solve) and is quoted in the write-up only
HOLES = {
    "A_transverse_short": InglisHole(semi_x=1.5, semi_y=1.0),   # K_t = 2.333, rho_t = 2.25
    "B_transverse_sharp": InglisHole(semi_x=0.5, semi_y=1.5),   # K_t = 7.000, rho_t = 1/6
}
GATE = 3.0          # percent, the platform's structural acceptance gate
RESOLVED = 0.01     # delta = first tip element / rho_t, tested sufficient condition
UNRESOLVED = 0.03   # delta above which every tested case fails


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def layer_decay(hole: InglisHole, span: float = 8.0, count: int = 4001) -> dict:
    """How far the exact peak excess decays along the transverse symmetry axis.

    Returns the distances (in units of ``rho_t``) at which the excess over the far
    field drops to 50 % and 10 % of the peak excess, plus a fitted exponential
    length.  This is the physical length scale the mesh has to resolve.
    """
    rho_t = hole.tip_curvature_radius
    peak = hole.stress_concentration * hole.far_field
    ys = np.linspace(hole.semi_y, hole.semi_y + span * rho_t, count)
    values = np.array([hole.stress_cartesian(0.0, float(y))[0] for y in ys])
    excess = (values - hole.far_field) / (peak - hole.far_field)
    d = ys - hole.semi_y
    half = float(d[int(np.argmax(excess < 0.5))] / rho_t)
    tenth = float(d[int(np.argmax(excess < 0.1))] / rho_t)
    # least-squares slope of log(excess) against d/rho_t over the first 50 %
    mask = excess > 0.5
    slope, _ = np.polyfit(d[mask] / rho_t, np.log(excess[mask]), 1)
    return {
        "half_excess_at_rho_t": half,
        "tenth_excess_at_rho_t": tenth,
        "log_slope_per_rho_t": float(slope),
        "decay_length_rho_t": float(-1.0 / slope) if slope < 0 else float("inf"),
    }


def sweep() -> list[dict]:
    rows = []
    for name, hole in HOLES.items():
        for grading in GRADINGS:
            for level in LEVELS:
                case = solve_case(
                    hole, rho_outer=RHO_OUTER, n_theta=level, n_radial=level, grading=grading
                )
                res = case["tip_resolution"]
                rows.append(
                    {
                        "hole": name,
                        "semi_x": hole.semi_x,
                        "semi_y": hole.semi_y,
                        "stress_concentration": hole.stress_concentration,
                        "curvature_radius": hole.tip_curvature_radius,
                        "grading": grading,
                        "level": level,
                        "dofs": case["dofs"],
                        "elements": case["elements"],
                        "peak_fe": case["transverse_tip"]["hoop_fe"],
                        "peak_exact": case["transverse_tip"]["exact"],
                        "peak_error_pct": 100.0 * case["transverse_tip"]["relative_error"],
                        "load_axis_tip_error_pct": 100.0 * case["load_axis_tip"]["relative_error"],
                        "delta_radial": res["radial_over_curvature"],
                        "delta_angular": res["angular_over_curvature"],
                        "aspect": res["aspect"],
                        "displacement_error": case["displacement_error"],
                        "force_balance_relative": case["force_balance_relative"],
                        "recovery_cross_check": case["recovery_cross_check"],
                        "symmetry_edge_error_pct": 100.0
                        * case["symmetry_edge"]["relative_error"],
                    }
                )
    return rows


def rule_report(rows: list[dict]) -> dict:
    """Evaluate the two half-conditions claimed in the write-up."""
    resolved = [
        r for r in rows if r["delta_radial"] <= RESOLVED and r["delta_angular"] <= RESOLVED
    ]
    unresolved = [r for r in rows if r["delta_radial"] >= UNRESOLVED]
    return {
        "resolved_condition": {
            "predicate": f"delta_radial <= {RESOLVED} and delta_angular <= {RESOLVED}",
            "cases": len(resolved),
            "max_error_pct": max((r["peak_error_pct"] for r in resolved), default=float("nan")),
            "all_within_gate": all(r["peak_error_pct"] < GATE for r in resolved),
            "cases_table": [
                {
                    "hole": r["hole"],
                    "grading": r["grading"],
                    "level": r["level"],
                    "delta_radial": r["delta_radial"],
                    "delta_angular": r["delta_angular"],
                    "peak_error_pct": r["peak_error_pct"],
                }
                for r in resolved
            ],
        },
        "unresolved_condition": {
            "predicate": f"delta_radial >= {UNRESOLVED}",
            "cases": len(unresolved),
            "min_error_pct": min(
                (r["peak_error_pct"] for r in unresolved), default=float("nan")
            ),
            "all_fail_gate": all(r["peak_error_pct"] > GATE for r in unresolved),
            "cases_table": [
                {
                    "hole": r["hole"],
                    "grading": r["grading"],
                    "level": r["level"],
                    "delta_radial": r["delta_radial"],
                    "peak_error_pct": r["peak_error_pct"],
                }
                for r in unresolved
            ],
        },
    }


def grading_family(rows: list[dict], hole: str) -> dict:
    """Error-vs-level trend inside the *old* family (grading = 3) and the fixed one."""
    out = {}
    for grading in GRADINGS:
        fam = sorted(
            (r for r in rows if r["hole"] == hole and r["grading"] == grading),
            key=lambda r: r["level"],
        )
        if not fam:
            continue
        levels = np.array([r["level"] for r in fam], dtype=float)
        errors = np.array([r["peak_error_pct"] for r in fam])
        finite = errors > 0
        slope = (
            float(np.polyfit(np.log(levels[finite]), np.log(errors[finite]), 1)[0])
            if finite.sum() >= 2
            else float("nan")
        )
        # level at which the fitted trend would cross the 3 % gate
        if finite.sum() >= 2 and errors[-1] > GATE and slope < 0:
            c = errors[finite] / levels[finite] ** slope
            cross = float((GATE / float(np.exp(np.mean(np.log(c))))) ** (1.0 / slope))
        else:
            cross = None
        out[f"grading_{grading:g}"] = {
            "levels": levels.tolist(),
            "errors_pct": errors.tolist(),
            "delta_radial": [r["delta_radial"] for r in fam],
            "observed_order": -slope if slope == slope else float("nan"),
            "level_for_3pct": cross,
        }
    return out


def richardson(rows: list[dict], hole: str, grading: float) -> dict:
    fam = sorted(
        (r for r in rows if r["hole"] == hole and r["grading"] == grading),
        key=lambda r: r["level"],
    )
    peaks = [r["peak_fe"] for r in fam]
    levels = [r["level"] for r in fam]
    exact = fam[-1]["peak_exact"]
    rates = []
    for k in range(1, len(fam)):
        ratio = abs(peaks[k] - exact) / max(abs(peaks[k - 1] - exact), 1e-300)
        rates.append(float(math.log(ratio) / math.log(levels[k] / levels[k - 1])))
    p = rates[-1] if rates else 1.0
    extrapolated = peaks[-1] + (peaks[-1] - peaks[-2]) / max(
        (levels[-1] / levels[-2]) ** p - 1.0, 1e-12
    )
    return {
        "grading": grading,
        "levels": levels,
        "peaks": peaks,
        "exact": exact,
        "signed_rates": rates,
        "order_used": p,
        "richardson_peak": extrapolated,
        "richardson_error_pct": 100.0 * abs(extrapolated - exact) / abs(exact),
    }


def certified_case(hole_name: str = "B_transverse_sharp", level: int = 48, grading: float = 6.0) -> dict:
    """The recommended production configuration, with the global sanity checks."""
    hole = HOLES[hole_name]
    case = solve_case(hole, rho_outer=RHO_OUTER, n_theta=level, n_radial=level, grading=grading)
    return {
        "hole": hole_name,
        "level": level,
        "grading": grading,
        "dofs": case["dofs"],
        "elements": case["elements"],
        "peak_fe": case["transverse_tip"]["hoop_fe"],
        "peak_exact": case["transverse_tip"]["exact"],
        "peak_error_pct": 100.0 * case["transverse_tip"]["relative_error"],
        "load_axis_tip_error_pct": 100.0 * case["load_axis_tip"]["relative_error"],
        "tip_resolution": case["tip_resolution"],
        "displacement_error": case["displacement_error"],
        "force_balance_relative": case["force_balance_relative"],
        "symmetry_edge_error_pct": 100.0 * case["symmetry_edge"]["relative_error"],
        "recovery_cross_check": case["recovery_cross_check"],
        "sample_rows": case["sample_rows"],
        "hosts": case["transverse_tip"]["hosts"],
    }


def legacy_case(rows: list[dict]) -> dict:
    match = [
        r for r in rows
        if r["hole"] == "B_transverse_sharp" and r["grading"] == 3.0 and r["level"] == 48
    ]
    row = match[0]
    return {
        "hole": row["hole"],
        "level": row["level"],
        "grading": row["grading"],
        "dofs": row["dofs"],
        "peak_error_pct": row["peak_error_pct"],
        "delta_radial": row["delta_radial"],
        "exceeds_gate": row["peak_error_pct"] > GATE,
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def figure_rule(rows: list[dict]) -> str:
    fig, ax = plt.subplots(figsize=(6.6, 4.6), dpi=140)
    markers = {"A_transverse_short": "s", "B_transverse_sharp": "o"}
    for name in HOLES:
        sub = [r for r in rows if r["hole"] == name]
        ax.loglog(
            [100.0 * r["delta_radial"] for r in sub],
            [r["peak_error_pct"] for r in sub],
            markers[name], markersize=6, markerfacecolor="none",
            label=f"{name} ($K_t$={sub[0]['stress_concentration']:.3f})",
        )
    ax.axhline(GATE, color="crimson", linewidth=0.9, linestyle=":")
    ax.text(0.024, GATE * 1.15, "3 % gate", color="crimson", fontsize=8)
    ax.axvline(100 * RESOLVED, color="0.4", linewidth=0.9, linestyle="--")
    ax.text(100 * RESOLVED * 1.1, 0.05, r"$\delta_{r}\leq 1\,\%$", color="0.3", fontsize=8)
    ax.axvline(100 * UNRESOLVED, color="0.4", linewidth=0.9, linestyle="-.")
    ax.text(100 * UNRESOLVED * 1.1, 0.05, r"$\delta_{r}\geq 3\,\%$", color="0.3", fontsize=8)
    ax.set_xlabel(r"first-element thickness at the tip, $\delta_r = h_r/\rho_t$  (%)")
    ax.set_ylabel("peak hoop-stress error (%)")
    ax.set_title("Sharp-tip peak error collapses onto the tip layer resolution")
    ax.grid(which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = ASSETS / "sharp_tip_rule.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path.relative_to(REPO))


def figure_families(rows: list[dict]) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), dpi=140, sharey=True)
    for ax, name in zip(axes, HOLES):
        for grading in GRADINGS:
            fam = sorted(
                (r for r in rows if r["hole"] == name and r["grading"] == grading),
                key=lambda r: r["level"],
            )
            if not fam:
                continue
            ax.semilogy(
                [r["level"] for r in fam],
                [r["peak_error_pct"] for r in fam],
                "o-", label=f"grading={grading:g}",
            )
        exact = HOLES[name].stress_concentration
        ax.axhline(GATE, color="crimson", linewidth=0.9, linestyle=":")
        ax.set_title(f"{name}: $K_t$={exact:.3f}, $\\rho_t$={HOLES[name].tip_curvature_radius:.3f}")
        ax.set_xlabel(r"$n_\theta = n_\rho$")
        ax.grid(which="both", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("peak hoop-stress error (%)")
    fig.suptitle("Refining n alone cannot buy the sharp-tip peak; the grading strength can", fontsize=10)
    fig.tight_layout()
    path = ASSETS / "sharp_tip_families.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path.relative_to(REPO))


def figure_layers(rows: list[dict], decays: dict) -> str:
    hole = HOLES["B_transverse_sharp"]
    rho_t = hole.tip_curvature_radius
    peak = hole.stress_concentration * hole.far_field
    ys = np.linspace(hole.semi_y, hole.semi_y + 5.0 * rho_t, 1201)
    excess = (np.array([hole.stress_cartesian(0.0, float(y))[0] for y in ys]) - hole.far_field) / (
        peak - hole.far_field
    )
    fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=140)
    ax.plot((ys - hole.semi_y) / rho_t, excess, "k-", label="exact excess over far field")
    for grading, style in ((3.0, "C3"), (5.0, "C2"), (6.0, "C0")):
        d = 100.0 * tip_resolution(hole, 48, 48, grading, RHO_OUTER)["radial_over_curvature"]
        ax.axvline(d, color=style, linestyle="--", linewidth=1.0,
                   label=f"first layer, n=48, grading={grading:g} (${d:.2f}\\%\\,\\rho_t$)")
    ax.set_xlabel(r"distance from the tip along the symmetry axis, $(y-\mathrm{semi}_y)/\rho_t$")
    ax.set_ylabel(r"$(\sigma_{xx}-T)/(\sigma_{xx}^{tip}-T)$")
    ax.set_title("The tip boundary layer the mesh must resolve is $O(\\rho_t)$ thick")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = ASSETS / "sharp_tip_layer.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path.relative_to(REPO))


def figure_profiles(certified: dict) -> str:
    hole = HOLES[certified["hole"]]
    rows = certified["sample_rows"]
    y = np.array([r["y_over_semi_y"] for r in rows])
    fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=140)
    ax.plot(y, [r["sigma_xx_exact"] for r in rows], "k-", label="exact")
    ax.plot(y, [r["sigma_xx_fe"] for r in rows], "o", markersize=4,
            label=f"FE, n={certified['level']}, grading={certified['grading']:g}")
    ax.set_xlabel(r"$y/\mathrm{semi}_y$ on the transverse symmetry axis")
    ax.set_ylabel(r"$\sigma_{xx}/T$")
    ax.set_title("Certified configuration: transverse-axis stress profile")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = ASSETS / "sharp_tip_profile.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path.relative_to(REPO))


def markdown_table(rows: list[dict], hole: str) -> str:
    lines = [
        "| grading | n | DOF | $\\delta_r=h_r/\\rho_t$ | $\\delta_\\theta$ | aspect $h_\\theta/h_r$ | peak FE | peak error | load-axis tip error |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(
        (r for r in rows if r["hole"] == hole), key=lambda r: (r["grading"], r["level"])
    ):
        lines.append(
            f"| {r['grading']:g} | {r['level']} | {r['dofs']} | {100 * r['delta_radial']:.3f}% | "
            f"{100 * r['delta_angular']:.3f}% | {r['aspect']:.3f} | {r['peak_fe']:.5f} | "
            f"{r['peak_error_pct']:.3f}% | {r['load_axis_tip_error_pct']:.3f}% |"
        )
    return "\n".join(lines)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    rows = sweep()
    decays = {name: layer_decay(hole) for name, hole in HOLES.items()}
    report = {
        "schema": "fem-elliptic-hole-sharp-tip/1",
        "rho_outer": RHO_OUTER,
        "gradings": list(GRADINGS),
        "levels": list(LEVELS),
        "gate_pct": GATE,
        "resolved_delta": RESOLVED,
        "unresolved_delta": UNRESOLVED,
        "holes": {
            name: {
                "semi_x": hole.semi_x,
                "semi_y": hole.semi_y,
                "stress_concentration": hole.stress_concentration,
                "curvature_radius": hole.tip_curvature_radius,
                "mapping_m": hole.mapping_m,
                "layer_decay": decays[name],
            }
            for name, hole in HOLES.items()
        },
        "sweep": rows,
        "rule": rule_report(rows),
        "families": {name: grading_family(rows, name) for name in HOLES},
        "richardson": {
            name: {f"grading_{g:g}": richardson(rows, name, g) for g in (3.0, 5.0, 6.0)}
            for name in HOLES
        },
        "certified": certified_case(),
        "legacy": legacy_case(rows),
        "figures": {},
    }
    if not report["certified"]["sample_rows"]:
        raise RuntimeError("certified case produced no profile samples")
    report["figures"] = {
        "rule": figure_rule(rows),
        "families": figure_families(rows),
        "layer": figure_layers(rows, decays),
        "profile": figure_profiles(report["certified"]),
    }
    report["markdown_tables"] = {name: markdown_table(rows, name) for name in HOLES}
    report["source_sha256"] = {
        "scripts/audit_elliptic_hole_sharp_tip.py": sha256(pathlib.Path(__file__)),
        "src/tensorfem/elliptic_hole.py": sha256(REPO / "src" / "tensorfem" / "elliptic_hole.py"),
    }
    JSON_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    cert = report["certified"]
    print(f"certified  : {cert['hole']} n={cert['level']} grading={cert['grading']:g} "
          f"dofs={cert['dofs']} peak={cert['peak_fe']:.5f} vs {cert['peak_exact']:.5f} "
          f"-> {cert['peak_error_pct']:.3f}%")
    print(f"legacy     : n={report['legacy']['level']} grading={report['legacy']['grading']:g} "
          f"-> {report['legacy']['peak_error_pct']:.3f}% (exceeds gate: "
          f"{report['legacy']['exceeds_gate']})")
    res = report["rule"]["resolved_condition"]
    unr = report["rule"]["unresolved_condition"]
    print(f"resolved   : {res['cases']} cases, max error {res['max_error_pct']:.3f}%, "
          f"all within gate: {res['all_within_gate']}")
    print(f"unresolved : {unr['cases']} cases, min error {unr['min_error_pct']:.3f}%, "
          f"all fail gate: {unr['all_fail_gate']}")
    for name in HOLES:
        fam = report["families"][name]["grading_3"]
        print(f"{name}: grading=3 observed order {fam['observed_order']:.3f}, "
              f"level needed for 3% in that family: {fam['level_for_3pct']}")
    print(f"json       : {JSON_PATH.relative_to(REPO)}")


if __name__ == "__main__":
    main()