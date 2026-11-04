"""Finite-width plate with a circular hole: Howland/Heywood verification audit.

Writes ``benchmarks/finite_width_hole_reference.json`` and five figures under
``docs/assets/finite_width_hole/`` by driving the production plane-stress path
through :mod:`tensorfem.finite_width_hole`.

Run: ``OMP_NUM_THREADS=2 PYTHONPATH=src python scripts/audit_finite_width_hole.py``
"""
from __future__ import annotations

import json
import math
import platform as host
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

import torch

from tensorfem import __version__
from tensorfem.finite_width_hole import (
    PETERSON_HOWLAND_KTN,
    FiniteWidthPlate,
    heywood_stress_concentration,
    solve_case,
    starburst_mesh,
    plate_model,
)
from tensorfem.continuum import solve_continuum

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "benchmarks" / "finite_width_hole_reference.json"
ASSETS = ROOT / "docs" / "assets" / "finite_width_hole"
RATIOS = (0.10, 0.20, 0.30, 0.40, 0.50)
LEVELS = (16, 24, 32, 40)
HEADLINE = 40
HALF_LENGTH_OVER_WIDTH = 1.5
LIMIT_RATIOS = (0.02, 0.05)
LIMIT_LEVELS = (32, 40)
LS_STUDY_RATIO = 0.20
LS_STUDY_HALF_LENGTHS = (1.0, 1.5, 2.0, 2.5)
GRADING_STUDY_RATIO = 0.50
GRADINGS = (4.0, 6.0, 8.0)
FIGURES: list[Path] = []


def _slim(case: dict) -> dict:
    return {k: v for k, v in case.items() if not k.startswith("_")}


def _richardson(values: list, factors: list) -> dict:
    """Observed order and extrapolated limit from the last three values."""
    if len(values) < 3:
        return {"observed_order": None, "extrapolated": None}
    _, f1 = factors[-3] / factors[-2], factors[-2] / factors[-1]
    d0, d1 = values[-2] - values[-3], values[-1] - values[-2]
    if d1 == 0.0 or d0 / d1 <= 0.0:
        return {"observed_order": None, "extrapolated": None}
    order = math.log(d0 / d1) / math.log(f1)
    if order <= 0.05:
        return {"observed_order": order, "extrapolated": None}
    ratio = f1 ** order - 1.0
    return {"observed_order": order, "extrapolated": values[-1] + d1 / ratio}


# ---------------------------------------------------------------------------
# Studies
# ---------------------------------------------------------------------------

def mesh_study() -> list:
    cases = []
    for ratio in RATIOS:
        plate = FiniteWidthPlate(diameter=ratio, half_length=HALF_LENGTH_OVER_WIDTH)
        for level in LEVELS:
            started = time.time()
            case = solve_case(plate, n_boundary=level)
            case["wall_seconds"] = time.time() - started
            case["h_over_width"] = 1.0 / level
            cases.append(case)
    return cases


def kirsch_limit_study() -> list:
    cases = []
    for ratio in LIMIT_RATIOS:
        plate = FiniteWidthPlate(diameter=ratio, half_length=HALF_LENGTH_OVER_WIDTH)
        for level in LIMIT_LEVELS:
            case = solve_case(plate, n_boundary=level)
            case["ktg_error_vs_kirsch"] = abs(case["ktg_fe"] - 3.0) / 3.0
            cases.append(case)
    return cases


def saint_venant_study() -> list:
    plate_rows = []
    for ratio in (0.2, 0.5):
        rows = []
        for half_length in LS_STUDY_HALF_LENGTHS:
            case = solve_case(
                FiniteWidthPlate(diameter=ratio, half_length=half_length), n_boundary=HEADLINE
            )
            rows.append(_slim(case))
        cases_plate = FiniteWidthPlate(diameter=ratio, half_length=HALF_LENGTH_OVER_WIDTH)
        designated = [r for r in rows if abs(r["half_length_over_width"] - HALF_LENGTH_OVER_WIDTH) < 1e-9][0]
        spread = max(r["ktn_fe"] for r in rows) - min(r["ktn_fe"] for r in rows)
        plate_rows.append(
            {
                "ratio": ratio,
                "ktn_heywood": cases_plate.heywood_ktn,
                "designated_ktn": designated["ktn_fe"],
                "designated_error": abs(designated["ktn_fe"] - cases_plate.heywood_ktn) / cases_plate.heywood_ktn,
                "band": spread,
                "band_relative": spread / designated["ktn_fe"],
                "rows": rows,
            }
        )
    return plate_rows


def grading_study() -> list:
    plate = FiniteWidthPlate(diameter=GRADING_STUDY_RATIO, half_length=HALF_LENGTH_OVER_WIDTH)
    rows = []
    for grading in GRADINGS:
        case = solve_case(plate, n_boundary=HEADLINE, grading=grading)
        rows.append({"grading": grading, "ktn_fe": case["ktn_fe"],
                     "error_vs_heywood": case["error_vs_heywood"]})
    return rows


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figure_ktn(cases: list) -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ratios = np.linspace(0.0, 0.55, 300)
    ax.plot(ratios, [heywood_stress_concentration(r) for r in ratios], "k-", lw=1.6,
            label="Heywood fit  $K_{tn}=2+(1-d/W)^3$")
    table_r = sorted(PETERSON_HOWLAND_KTN)
    ax.plot(table_r, [PETERSON_HOWLAND_KTN[r] for r in table_r], "s", ms=7, mfc="none",
            mec="tab:orange", mew=1.6, label="Peterson/Howland table (chart, ±1 %)")
    ax.axhline(3.0, color="tab:red", ls=":", lw=1.2, label="Kirsch 1898 ($d/W\\to0$): 3.0")
    headline = {c["ratio"]: c for c in cases if c["n_boundary"] == HEADLINE}
    xs = sorted(headline)
    ys = [headline[r]["ktn_fe"] for r in xs]
    lo = [min(c["ktn_fe"] for c in cases if c["ratio"] == r) for r in xs]
    hi = [max(c["ktn_fe"] for c in cases if c["ratio"] == r) for r in xs]
    ax.errorbar(xs, ys, yerr=[np.array(ys) - np.array(lo), np.array(hi) - np.array(ys)],
                fmt="o", ms=7, color="tab:blue", capsize=4,
                label=f"TensorFEM Q4, $n_\\theta$={HEADLINE}, band = 16..40")
    for r, y in zip(xs, ys):
        ax.annotate(f"{y:.3f}", (r, y), textcoords="offset points", xytext=(7, -12), fontsize=8)
    ax.set_xlabel("hole-to-width ratio  $d/W$")
    ax.set_ylabel("net-section factor  $K_{tn}=\\sigma_{peak}/\\sigma_{net}$")
    ax.set_title("Finite-width plate with a central hole: verification against Howland/Heywood")
    ax.set_xlim(0, 0.55)
    ax.set_ylim(1.9, 3.1)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8.5)
    path = ASSETS / "finite_width_ktn_vs_ratio.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_convergence(cases: list) -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(RATIOS)))
    for color, ratio in zip(colors, RATIOS):
        subset = sorted((c for c in cases if c["ratio"] == ratio), key=lambda c: c["n_boundary"])
        h = [c["h_over_width"] for c in subset]
        values = [c["ktn_fe"] for c in subset]
        ax.plot(h, values, "o-", color=color, ms=5, label=f"$d/W$={ratio:.1f}")
        reference = heywood_stress_concentration(ratio)
        ax.axhline(reference, color=color, ls=":", lw=1.0, alpha=0.8)
        extrapolated = _richardson(values, [1.0 / c["n_boundary"] for c in subset])["extrapolated"]
        if extrapolated is not None:
            ax.plot([0.0], [extrapolated], "*", color=color, ms=12)
    ax.set_xlabel("element size  $h/W$  ($n_\\theta = W/h$)")
    ax.set_ylabel("$K_{tn}$")
    ax.set_xlim(0.0, 0.065)
    ax.set_title("Mesh convergence (dotted = Heywood reference, star = Richardson limit)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    path = ASSETS / "finite_width_convergence.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_field(level: int = HEADLINE) -> Path:
    plate = FiniteWidthPlate(diameter=GRADING_STUDY_RATIO, half_length=HALF_LENGTH_OVER_WIDTH)
    nodes, elements, meta = starburst_mesh(plate, level, level, 6.0)
    model = plate_model(plate, meta)
    result = solve_continuum(model)
    stress = result.stress.detach().cpu().numpy()
    xy = np.asarray(nodes, dtype=np.float64)
    index = np.asarray(elements)
    polygons = [xy[element] for element in index]
    values = stress[:, 0] / plate.tension  # sigma_xx / T  (gross normalisation)
    limit = float(np.percentile(np.abs(values), 99.0)) * 1.6
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    collection = PolyCollection(polygons, array=values, cmap="inferno",
                                edgecolors="face", linewidths=0.15)
    collection.set_clim(0.0, limit)
    ax.add_collection(collection)
    ax.autoscale()
    theta = np.linspace(0.0, math.pi / 2.0, 200)
    ax.plot(plate.hole_radius * np.cos(theta), plate.hole_radius * np.sin(theta), "w-", lw=1.4)
    ax.plot([plate.half_length] * 2, [0.0, plate.half_width], "c--", lw=1.2)
    ax.plot([0.0, plate.half_length], [plate.half_width] * 2, "c:", lw=1.2)
    ax.annotate("$\\sigma_{xx}=T$ (loaded end)", (plate.half_length * 0.62, plate.half_width * 0.1),
                color="white", fontsize=8)
    ax.annotate("free edge", (plate.half_length * 0.45, plate.half_width * 0.93),
                color="cyan", fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlabel("$x/W$")
    ax.set_ylabel("$y/W$")
    ax.set_title(f"$\\sigma_{{xx}}/T$ (quarter model, $d/W$=0.5, $n_\\theta$={level})")
    fig.colorbar(collection, ax=ax, label="$\\sigma_{xx}/T$")
    path = ASSETS / "finite_width_sigma_xx_field.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_profile(cases: list) -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(RATIOS)))
    for color, ratio in zip(colors, RATIOS):
        case = [c for c in cases if c["ratio"] == ratio and c["n_boundary"] == HEADLINE][0]
        profile = np.array(case["hole_edge_hoop_profile_over_tension"])
        ax.plot(profile[:, 0], profile[:, 1], "-", color=color, lw=1.6,
                label=f"$d/W$={ratio:.1f} (FE, $K_{{tn}}$={case['ktn_fe']:.3f})")
    theta = np.linspace(0.0, 90.0, 200)
    rad = np.radians(theta)
    ax.plot(theta, 1.0 - 2.0 * np.cos(2.0 * rad), "k--", lw=1.2,
            label="Kirsch $\\sigma_{\\theta\\theta}/T=1-2\\cos 2\\theta$ ($d/W\\to0$)")
    ax.set_xlabel("hole-edge angle $\\theta$ (deg, 90° = symmetry line, peak)")
    ax.set_ylabel("$\\sigma_{\\theta\\theta}/T$ on the hole edge (gross normalisation)")
    ax.set_title("Hole-edge hoop stress: finite-width families follow the Kirsch shape")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    path = ASSETS / "finite_width_hole_edge_profile.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_saint_venant(study: list) -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for color, block in zip(("tab:blue", "tab:red"), study):
        rows = sorted(block["rows"], key=lambda r: r["half_length_over_width"])
        ax.plot([r["half_length_over_width"] for r in rows], [r["ktn_fe"] for r in rows],
                "o-", color=color, label=f"$d/W$={block['ratio']:.1f} FE ($n_\\theta$={HEADLINE})")
        ax.axhline(block["ktn_heywood"], color=color, ls=":", lw=1.1)
    ax.set_xlabel("half-length  $L/W$  (full plate length $2L$)")
    ax.set_ylabel("$K_{tn}$")
    ax.set_title("Saint-Venant insensitivity (dotted = Heywood reference)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    path = ASSETS / "finite_width_L_sensitivity.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    cases = mesh_study()
    limit = kirsch_limit_study()
    saint_venant = saint_venant_study()
    grading = grading_study()

    headline = []
    for ratio in RATIOS:
        subset = sorted((c for c in cases if c["ratio"] == ratio), key=lambda c: c["n_boundary"])
        values = [c["ktn_fe"] for c in subset]
        rich = _richardson(values, [1.0 / c["n_boundary"] for c in subset])
        best = subset[-1]
        headline.append(
            {
                "ratio": ratio,
                "ktn_fe": best["ktn_fe"],
                "ktg_fe": best["ktg_fe"],
                "ktn_heywood": best["ktn_heywood"],
                "ktn_peterson": best["ktn_peterson"],
                "error_vs_heywood": best["error_vs_heywood"],
                "error_vs_peterson": best["error_vs_peterson"],
                "richardson": rich,
                "richardson_error_vs_heywood": (
                    None if rich["extrapolated"] is None
                    else abs(rich["extrapolated"] - best["ktn_heywood"]) / best["ktn_heywood"]
                ),
                "dofs": best["dofs"],
                "elements": best["elements"],
                "peak_theta_degrees": best["peak_theta_degrees"],
                "load_balance_error": best["load_balance"]["relative_error"],
                "coarse_error": max(c["error_vs_heywood"] for c in subset),
            }
        )

    FIGURES.append(figure_ktn(cases))
    FIGURES.append(figure_convergence(cases))
    FIGURES.append(figure_field())
    FIGURES.append(figure_profile(cases))
    FIGURES.append(figure_saint_venant(saint_venant))

    evidence = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tensorfem_version": __version__,
        "python": host.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "reference": (
            "Kirsch (1898) limit; Howland (1930) series values as tabulated in "
            "Peterson's Stress Concentration Factors (Pilkey & Pilkey, 3rd ed., Chart 4.1); "
            "Heywood (1952) fit K_tn = 2 + (1-d/W)^3"
        ),
        "convention": (
            "K_tn = sigma_peak / sigma_net with sigma_net = T W/(W-d); "
            "K_tg = sigma_peak / T = K_tn (1-d/W)"
        ),
        "model": {
            "quarter_plate": "0<=x<=L, 0<=y<=W/2, quarter hole at origin",
            "symmetry": "u_y=0 on y=0, u_x=0 on x=0",
            "hole_and_lateral_edge": "traction free",
            "loaded_end": "uniform tension T on x=L",
            "element": "production Q4 plane stress (tensorfem.continuum)",
            "mesh": "starburst rays from hole, arc-length boundary distribution, graded to hole",
            "half_length_over_width": HALF_LENGTH_OVER_WIDTH,
            "young_pa": 200.0e9,
            "poisson": 0.3,
            "thickness_m": 0.01,
        },
        "levels": list(LEVELS),
        "ratios": list(RATIOS),
        "headline_level": HEADLINE,
        "headline": headline,
        "cases": [_slim(c) for c in cases],
        "kirsch_limit": [_slim(c) for c in limit],
        "saint_venant": saint_venant,
        "grading": grading,
        "figures": [str(p.relative_to(ROOT)) for p in FIGURES],
        "max_error_vs_heywood": max(row["error_vs_heywood"] for row in headline),
        "max_error_vs_peterson": max(
            row["error_vs_peterson"] for row in headline if row["error_vs_peterson"] is not None
        ),
        "max_load_balance_error": max(row["load_balance_error"] for row in headline),
        "thin_plate_note": (
            "plane stress, small strain, isotropic, thin plate (t/W = 0.01); no contact, "
            "plasticity or geometric nonlinearity is exercised here"
        ),
    }
    OUTPUT.write_text(json.dumps(evidence, indent=2))
    print(json.dumps({k: evidence[k] for k in
                      ("max_error_vs_heywood", "max_error_vs_peterson", "max_load_balance_error")}, indent=2))
    for row in headline:
        print("d/W=%.2f Ktn=%.4f hey=%.3f pet=%s err=%.2f%% richardson=%s" % (
            row["ratio"], row["ktn_fe"], row["ktn_heywood"], row["ktn_peterson"],
            100 * row["error_vs_heywood"],
            "n/a" if row["richardson"]["extrapolated"] is None else "%.4f" % row["richardson"]["extrapolated"]))
    print("wrote", OUTPUT)


if __name__ == "__main__":
    main()