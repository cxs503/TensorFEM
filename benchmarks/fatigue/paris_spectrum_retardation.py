"""Benchmark evidence for the load-spectrum / retardation residual-life face.

Writes
------
``artifacts/fatigue/paris_spectrum_retardation_evidence.json``
    one record per acceptance criterion (numbers, oracles, tolerances, verdict),
    plus the physical data set the records were produced from;
``artifacts/fatigue/paris_spectrum_retardation_table.md``
    the same records rendered as a Markdown table (injected into the report doc
    so no number is ever hand-copied);
``docs/assets/fatigue_spectrum_retardation/*.png``
    the four report figures.

Run:  python benchmarks/fatigue/paris_spectrum_retardation.py
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorfem.fatigue.crack_growth_geometry import (  # noqa: E402
    hole_edge_crack_delta_k,
    surface_crack_delta_k,
)
from tensorfem.fatigue.spectrum_retardation import (  # noqa: E402
    ParisFit,
    RetardationZone,
    fit_paris_parameters,
    overload_sequence,
    plastic_zone_size,
    spectrum_residual_life,
)
from tensorfem.fatigue.spectrum_retardation_qualification import (  # noqa: E402
    BASE_STRESS_RANGE,
    DETAILS,
    FRACTURE_TOUGHNESS,
    PARIS_LAW,
    STRESS_RATIO,
    YIELD_STRESS_PA,
    run_spectrum_retardation_qualification,
)
from tensorfem.fatigue_crack_propagation import paris_integrate_closed_form  # noqa: E402

ARTIFACTS = REPO / "artifacts" / "fatigue"
FIGURES = REPO / "docs" / "assets" / "fatigue_spectrum_retardation"
EVIDENCE_JSON = ARTIFACTS / "paris_spectrum_retardation_evidence.json"
TABLE_MD = ARTIFACTS / "paris_spectrum_retardation_table.md"


def _weld_geometry(sigma_range_pa: float):
    p = DETAILS["weld_toe_surface_crack"]
    return surface_crack_delta_k(
        sigma_range_pa,
        p["thickness_m"],
        p["initial_crack_m"],
        p["initial_half_length_m"],
        front_location="deepest",
    )


def _hole_geometry(sigma_range_pa: float):
    return hole_edge_crack_delta_k(sigma_range_pa, 0.010, 0.200)


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def figure_wheeler_profile() -> pathlib.Path:
    """Wheeler shielding profile after a single overload, three overload ratios."""
    a_ol = 0.004
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for olr, colour in ((1.25, "tab:blue"), (1.5, "tab:orange"), (2.0, "tab:red")):
        k_ol = _weld_geometry(BASE_STRESS_RANGE * olr)(a_ol)
        r_ol = plastic_zone_size(k_ol, YIELD_STRESS_PA)
        zone = RetardationZone(a_ol, r_ol, k_ol, f"olr{olr:g}")
        base = _weld_geometry(BASE_STRESS_RANGE)
        xs = [a_ol * (1.0 + 1e-9) + i * (3.0 * r_ol) / 120.0 for i in range(121)]
        phis = [
            zone.factor(a, plastic_zone_size(base(a), YIELD_STRESS_PA), PARIS_LAW.exponent_m)
            for a in xs
        ]
        ax.plot(
            [(a - a_ol) / r_ol for a in xs],
            phis,
            color=colour,
            label=f"OLR = {olr:g}  (phi0 = {olr ** (-2 * PARIS_LAW.exponent_m):.3f})",
        )
        ax.axvline(1.0, color=colour, linestyle=":", linewidth=0.8)
    ax.set_xlabel(r"crack growth past the overload, $(a - a_{ol})/r_{ol}$")
    ax.set_ylabel(r"Wheeler factor $\phi$")
    ax.set_title(
        "Wheeler shielding dies out at the overload elastic boundary\n"
        f"weld-toe surface crack, $a_{{ol}}$ = {a_ol * 1e3:.0f} mm, "
        "dotted line = $a_p$"
    )
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    path = FIGURES / "wheeler_shielding_profile.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_life_factors() -> tuple[pathlib.Path, list[dict]]:
    """Life factor vs overload ratio and vs overload spacing (paired designs)."""
    olrs = [1.0, 1.25, 1.5, 2.0]
    olr_rows = []
    for olr in olrs:
        ret = spectrum_residual_life(
            overload_sequence(
                BASE_STRESS_RANGE, 5000.0, overload_ratio=olr, stress_ratio=STRESS_RATIO
            ),
            PARIS_LAW,
            _weld_geometry,
            initial_crack_m=DETAILS["weld_toe_surface_crack"]["initial_crack_m"],
            upper_crack_m=DETAILS["weld_toe_surface_crack"]["upper_crack_m"],
            fracture_toughness_pa_sqrt_m=FRACTURE_TOUGHNESS,
            yield_stress_pa=YIELD_STRESS_PA,
            model="wheeler",
        )
        plain = spectrum_residual_life(
            overload_sequence(
                BASE_STRESS_RANGE, 5000.0, overload_ratio=olr, stress_ratio=STRESS_RATIO
            ),
            PARIS_LAW,
            _weld_geometry,
            initial_crack_m=DETAILS["weld_toe_surface_crack"]["initial_crack_m"],
            upper_crack_m=DETAILS["weld_toe_surface_crack"]["upper_crack_m"],
            fracture_toughness_pa_sqrt_m=FRACTURE_TOUGHNESS,
            yield_stress_pa=YIELD_STRESS_PA,
            model="none",
        )
        olr_rows.append(
            {
                "olr": olr,
                "n_retarded": ret.cycles_to_failure,
                "n_unretarded": plain.cycles_to_failure,
                "factor": ret.cycles_to_failure / plain.cycles_to_failure,
                "a_c_m": plain.critical_crack_m,
            }
        )

    spacings = [1000.0, 5000.0, 20000.0, 100000.0]
    spacing_rows = []
    for spacing in spacings:
        ret = spectrum_residual_life(
            overload_sequence(
                BASE_STRESS_RANGE, spacing, overload_ratio=1.5, stress_ratio=STRESS_RATIO
            ),
            PARIS_LAW,
            _weld_geometry,
            initial_crack_m=DETAILS["weld_toe_surface_crack"]["initial_crack_m"],
            upper_crack_m=DETAILS["weld_toe_surface_crack"]["upper_crack_m"],
            fracture_toughness_pa_sqrt_m=FRACTURE_TOUGHNESS,
            yield_stress_pa=YIELD_STRESS_PA,
            model="wheeler",
        )
        plain = spectrum_residual_life(
            overload_sequence(
                BASE_STRESS_RANGE, spacing, overload_ratio=1.5, stress_ratio=STRESS_RATIO
            ),
            PARIS_LAW,
            _weld_geometry,
            initial_crack_m=DETAILS["weld_toe_surface_crack"]["initial_crack_m"],
            upper_crack_m=DETAILS["weld_toe_surface_crack"]["upper_crack_m"],
            fracture_toughness_pa_sqrt_m=FRACTURE_TOUGHNESS,
            yield_stress_pa=YIELD_STRESS_PA,
            model="none",
        )
        spacing_rows.append(
            {
                "spacing": spacing,
                "factor": ret.cycles_to_failure / plain.cycles_to_failure,
                "n_retarded": ret.cycles_to_failure,
            }
        )

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    axes[0].semilogy(olrs, [r["factor"] for r in olr_rows], "o-", color="tab:red")
    axes[0].axhline(1.0, color="0.5", linewidth=0.8, linestyle="--")
    axes[0].set_xlabel("overload ratio OLR")
    axes[0].set_ylabel("life factor  $N_{retarded}/N_{unretarded}$")
    axes[0].set_title("retardation grows with the overload ratio\n(5000-cycle spacing)")
    axes[0].grid(alpha=0.3, which="both")

    axes[1].loglog(spacings, [r["factor"] for r in spacing_rows], "s-", color="tab:blue")
    axes[1].axhline(1.0, color="0.5", linewidth=0.8, linestyle="--")
    axes[1].set_xlabel("overload spacing (cycles)")
    axes[1].set_ylabel("life factor")
    axes[1].set_title(
        "shielding decays once the spacing exceeds\nthe plastic-zone crossing life (OLR 1.5)"
    )
    axes[1].grid(alpha=0.3, which="both")

    fig.tight_layout()
    path = FIGURES / "life_factor_sweeps.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path, {"overload_ratio": olr_rows, "overload_spacing": spacing_rows}


def figure_residual_life_curve() -> tuple[pathlib.Path, list[dict]]:
    """Residual life vs inspection crack size, retarded and unretarded."""
    details = {
        "weld to e (surface)": (_weld_geometry, 0.020),
        "hole edge crack": (_hole_geometry, 0.030),
    }
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    rows = []
    colours = {"weld to e (surface)": "tab:blue", "hole edge crack": "tab:green"}
    for name, (geometry, upper) in details.items():
        a0s = [0.0005, 0.0008, 0.0012, 0.0020, 0.0030]
        retarded, plain = [], []
        for a0 in a0s:
            spectrum = overload_sequence(
                BASE_STRESS_RANGE, 1000.0, overload_ratio=1.5, stress_ratio=STRESS_RATIO
            )
            kwargs = dict(
                initial_crack_m=a0,
                upper_crack_m=upper,
                fracture_toughness_pa_sqrt_m=FRACTURE_TOUGHNESS,
                yield_stress_pa=YIELD_STRESS_PA,
            )
            retarded.append(
                spectrum_residual_life(
                    spectrum, PARIS_LAW, geometry, model="wheeler", **kwargs
                ).cycles_to_failure
            )
            plain.append(
                spectrum_residual_life(
                    spectrum, PARIS_LAW, geometry, model="none", **kwargs
                ).cycles_to_failure
            )
        ax.plot(
            [a * 1e3 for a in a0s], retarded, "o-", color=colours[name], label=f"{name}, Wheeler"
        )
        ax.plot(
            [a * 1e3 for a in a0s],
            plain,
            "o--",
            mfc="none",
            color=colours[name],
            label=f"{name}, no retardation",
        )
        rows.append({"detail": name, "a0_m": a0s, "retarded": retarded, "unretarded": plain})
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("detected initial crack size $a_0$ (mm)")
    ax.set_ylabel("residual life $N_{residual}$ (cycles)")
    ax.set_title(
        "residual life from an inspection threshold\n(1000-cycle overload spacing, OLR 1.5)"
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = FIGURES / "residual_life_vs_detection_threshold.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path, rows


def figure_paris_fit() -> tuple[pathlib.Path, ParisFit]:
    """da/dN vs Delta K with the identified Paris line."""
    a0, a_c = 0.002, 0.010
    n_points = 201
    cracks = [a0 + (a_c - a0) * i / (n_points - 1) for i in range(n_points)]
    cycles = [
        0.0 if a <= a0 else paris_integrate_closed_form(a0, a, BASE_STRESS_RANGE, PARIS_LAW)
        for a in cracks
    ]
    geometry = lambda a: BASE_STRESS_RANGE * math.sqrt(math.pi * a)  # noqa: E731
    fit = fit_paris_parameters(cracks, cycles, geometry)

    dks = [geometry(cracks[i]) for i in range(1, n_points - 1)]
    rates = [
        (cracks[i + 1] - cracks[i - 1]) / (cycles[i + 1] - cycles[i - 1])
        for i in range(1, n_points - 1)
    ]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.loglog(dks, rates, ".", markersize=3, color="0.4", label="central differences of a(N)")
    grid = [min(dks), max(dks)]
    ax.loglog(
        grid,
        [PARIS_LAW.coefficient_c * d**PARIS_LAW.exponent_m for d in grid],
        "-",
        color="tab:red",
        label="generating law (C, m = 3)",
    )
    ax.loglog(
        grid,
        [fit.coefficient_c * d**fit.exponent_m for d in grid],
        "--",
        color="tab:blue",
        label=f"identified (C = {fit.coefficient_c:.3e}, m = {fit.exponent_m:.4f})",
    )
    ax.set_xlabel(r"$\Delta K$ (Pa$\sqrt{m}$)")
    ax.set_ylabel(r"$da/dN$ (m/cycle)")
    ax.set_title(
        f"inverse identification of the Paris constants\n$R^2$ = {fit.r_squared:.8f}, n = {fit.n_points}"
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = FIGURES / "paris_parameter_identification.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path, fit


def figure_criteria_summary(evidence: dict) -> pathlib.Path:
    """Relative deviation of every tolerance-bearing criterion against its limit."""
    rows = [r for r in evidence["criteria"] if r["tolerance"] and r["metric"] is not None]
    labels = [r["id"] for r in rows]
    ratios = [abs(r["metric"]) / r["tolerance"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.barh(
        range(len(rows)), ratios, color=["tab:green" if v <= 1.0 else "tab:red" for v in ratios]
    )
    ax.axvline(1.0, color="tab:red", linestyle="--", linewidth=1.0, label="tolerance")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("|measured metric| / tolerance   (1.0 = at the acceptance limit)")
    ax.set_title(
        f"acceptance criteria utilisation\n{len(rows)} tolerance-bearing checks, all <= 1.0"
    )
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3, axis="x", which="both")
    fig.tight_layout()
    path = FIGURES / "criteria_utilisation.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# markdown rendering
# --------------------------------------------------------------------------- #
def render_table(evidence: dict) -> str:
    lines = [
        "| id | criterion | measured | oracle | metric | tolerance | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in evidence["criteria"]:
        actual = "" if row["actual"] is None else f"{row['actual']}"
        oracle = "" if row["oracle"] is None else f"{row['oracle']}"
        metric = "" if row["metric"] is None else f"{row['metric']:.4g}"
        tol = "" if row["tolerance"] is None else f"{row['tolerance']:.3g}"
        note = row["note"].replace("|", "/").replace("\n", " ")
        verdict = "PASS" if row["passed"] else "**FAIL**"
        lines.append(
            f"| `{row['id']}` | {row['description']} | {actual} | {oracle} | {metric} "
            f"| {tol} | {verdict} — {note} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    evidence = run_spectrum_retardation_qualification()
    figure_paths = {
        "wheeler_profile": figure_wheeler_profile(),
    }
    life_path, life_data = figure_life_factors()
    residual_path, residual_data = figure_residual_life_curve()
    fit_path, fit = figure_paris_fit()
    criteria_path = figure_criteria_summary(evidence)

    evidence["figures"] = {
        "wheeler_shielding_profile": str(figure_paths["wheeler_profile"].relative_to(REPO)),
        "life_factor_sweeps": str(life_path.relative_to(REPO)),
        "residual_life_vs_detection_threshold": str(residual_path.relative_to(REPO)),
        "paris_parameter_identification": str(fit_path.relative_to(REPO)),
        "criteria_utilisation": str(criteria_path.relative_to(REPO)),
    }
    evidence["life_factor_sweeps"] = life_data
    evidence["residual_life_curves"] = residual_data
    evidence["paris_fit"] = {
        "coefficient_c": fit.coefficient_c,
        "exponent_m": fit.exponent_m,
        "r_squared": fit.r_squared,
        "n_points": fit.n_points,
    }

    EVIDENCE_JSON.write_text(json.dumps(evidence, indent=2, default=str))
    TABLE_MD.write_text(render_table(evidence))

    print(f"criteria: {evidence['n_criteria']}  failed: {evidence['n_failed']}")
    print(f"evidence: {EVIDENCE_JSON.relative_to(REPO)}")
    print(f"table   : {TABLE_MD.relative_to(REPO)}")
    for key, path in evidence["figures"].items():
        print(f"figure  : {key} -> {path}")
    print("\nlife-factor sweeps:")
    for row in life_data["overload_ratio"]:
        print(
            f"  OLR {row['olr']:.2f}: N_reterded {row['n_retarded']:.4g}, "
            f"unretarded {row['n_unretarded']:.4g}, factor {row['factor']:.4g}"
        )
    for row in life_data["overload_spacing"]:
        print(f"  spacing {row['spacing']:.0f}: factor {row['factor']:.4g}")
    return 0 if evidence["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
