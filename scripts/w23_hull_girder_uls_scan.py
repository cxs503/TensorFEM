"""Run the W23 multi-station hull-girder ULS scan and write the evidence bundle.

Artifacts (paths relative to the repository root):

* ``benchmarks/hull_girder_uls/section_scan.csv``      per-case ULS table
* ``benchmarks/hull_girder_uls/csr_utilisation.json``  full study + criteria + recomputation
* ``benchmarks/hull_girder_uls/grid_convergence.json`` Roache GCI of M_u
* ``docs/assets/w23_hull_girder_uls/*.png``            figures

Every number in the JSON is produced here; the report only quotes these files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # allow running from a bare checkout
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorfem.hull_girder_csr_chain import MILD_STEEL_YIELD  # noqa: E402
from tensorfem.hull_girder_csr_w22 import (  # noqa: E402
    CASE_CATALOGUE,
    analyse_direction as w22_analyse_direction,
    build_scantlings as w22_build_scantlings,
    still_water_and_wave_moments as w22_moments,
)
from tensorfem.hull_girder_uls_scan import (  # noqa: E402
    CONDITION_COMPRESSED_FIBRE,
    FIBRE_LEVELS,
    STATION_FRACTIONS,
    STEP_LEVELS,
    load_distributions,
    recompute_residual,
    run_uls_scan,
)

OUT_DIR = ROOT / "benchmarks" / "hull_girder_uls"
FIG_DIR = ROOT / "docs" / "assets" / "w23_hull_girder_uls"
REGISTER = OUT_DIR / "csr_factor_register.json"
CSV_COLUMNS = [
    "case_id", "hull", "station_x_over_L", "station_x_m", "section_breadth_m",
    "taper_factor", "condition", "compressed_fibre", "gamma_s", "gamma_w",
    "M_sw_Nm", "M_wv_Nm", "lhs_Nm", "M_u_smith_Nm", "M_y_elastic_Nm",
    "M_u_over_M_y", "utilisation", "verdict", "dominant_mechanism",
    "controlling_member", "W_elastic_m3", "grid_gci_fibres_percent",
    "grid_gci_steps_percent", "fibres", "strips", "curvature_steps",
]


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    except Exception:  # pragma: no cover - git absent
        return "unknown"


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


# ---------------------------------------------------------------------------
# pairing audit: the W22 defect and its numerical consequence
# ---------------------------------------------------------------------------
def pairing_audit() -> dict:
    """Re-evaluate the W22 parent hulls with the compressed fibre paired correctly.

    W22 labelled the ``kappa > 0`` direction (bottom compression, physically
    hogging) "sagging" and matched it with the sagging wave moment.  Here both
    capacities are computed on the same sections and both pairings are reported.
    """
    rows = []
    for case in CASE_CATALOGUE:
        members, _ = w22_build_scantlings(case)
        bottom_compression = w22_analyse_direction(members, condition="sagging", strips=8, steps=320)
        deck_compression = w22_analyse_direction(members, condition="hogging", strips=8, steps=320)
        moments = w22_moments(case)
        m_sw_sag = max(moments["M_sw_mid_signed"], 0.0)
        m_sw_hog = max(-moments["M_sw_mid_signed"], 0.0)
        m_ws = moments["M_ws_rule"]
        m_wh = moments["M_wh_rule"]
        rows.append({
            "case": case.key,
            "M_u_bottom_compression_Nm": bottom_compression["ultimate_moment"],
            "M_u_deck_compression_Nm": deck_compression["ultimate_moment"],
            "M_sw_sagging_Nm": m_sw_sag,
            "M_sw_hogging_Nm": m_sw_hog,
            "M_wv_sagging_rule_Nm": m_ws,
            "M_wv_hogging_rule_Nm": m_wh,
            "w22_utilisation_sagging": (m_sw_sag + m_ws) / bottom_compression["ultimate_moment"],
            "w22_utilisation_hogging": (m_sw_hog + m_wh) / deck_compression["ultimate_moment"],
            "corrected_utilisation_sagging": (m_sw_sag + m_ws) / deck_compression["ultimate_moment"],
            "corrected_utilisation_hogging": (m_sw_hog + m_wh) / bottom_compression["ultimate_moment"],
        })
    for row in rows:
        row["delta_sagging"] = row["corrected_utilisation_sagging"] - row["w22_utilisation_sagging"]
        row["delta_hogging"] = row["corrected_utilisation_hogging"] - row["w22_utilisation_hogging"]
    return {
        "defect": (
            "W22 used analyse_direction(condition='sagging') for the one with kappa > 0, i.e. "
            "bottom compression, which is hogging; its docstring calls this 'sagging'. The "
            "sagging wave moment was therefore divided by the bottom-compression capacity."
        ),
        "reference": "src/tensorfem/hull_girder_smith_collapse.py: 'kappa > 0 ... the bottom is compressed'; "
                     "src/tensorfem/hull_girder_csr_w22.py analyse_direction docstring",
        "rows": rows,
        "max_absolute_utilisation_change": max(
            max(abs(row["delta_sagging"]), abs(row["delta_hogging"])) for row in rows),
        "verdict_changes": sum(
            (row["w22_utilisation_sagging"] <= 1.0) != (row["corrected_utilisation_sagging"] <= 1.0)
            or (row["w22_utilisation_hogging"] <= 1.0) != (row["corrected_utilisation_hogging"] <= 1.0)
            for row in rows),
    }


# ---------------------------------------------------------------------------
# factor sensitivity (utilisation is linear in the factors)
# ---------------------------------------------------------------------------
def factor_sensitivity(study: dict) -> dict:
    level_sets = [(1.0, 1.0), (1.0, 1.1), (1.0, 1.2), (1.0, 1.25), (1.0, 1.3),
                  (1.0, 1.4), (1.05, 1.2), (1.1, 1.3)]
    table = []
    for row in study["cases"]:
        entry = {"case_id": row["case_id"], "gamma_w_max_at_gamma_s_1": row["gamma_w_max_at_gamma_s_1"],
                 "gamma_admissible": row["gamma_admissible"], "levels": []}
        for gamma_s, gamma_w in level_sets:
            utilisation = (gamma_s * row["M_sw_component_Nm"]
                           + gamma_w * row["M_wv_component_Nm"]) / row["M_u_smith_Nm"]
            entry["levels"].append({"gamma_s": gamma_s, "gamma_w": gamma_w,
                                    "utilisation": utilisation,
                                    "verdict": "PASS" if utilisation <= 1.0 else "FAIL"})
        table.append(entry)
    return {
        "note": "sensitivity scan only: NONE of these factor values is a transcribed CSR value "
                "(see csr_factor_register.json); they show at which factor level a case breaches unity",
        "levels": [{"gamma_s": s, "gamma_w": w} for s, w in level_sets],
        "cases": table,
        "n_pass_at_unit_factors": sum(
            1 for row in study["cases"] if row["verdict"] == "PASS"),
        "n_cases": len(study["cases"]),
    }


# ---------------------------------------------------------------------------
# artifact writers
# ---------------------------------------------------------------------------
def write_csv(study: dict, grid_lookup: dict, path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in study["cases"]:
            fibre_gci = grid_lookup[row["case_id"]]["fibre_family"]["gci_fine_percent"]
            step_gci = grid_lookup[row["case_id"]]["step_family"]["gci_fine_percent"]
            writer.writerow({
                "case_id": row["case_id"],
                "hull": study["hull_model"]["case"],
                "station_x_over_L": f"{row['station_x_fraction']:.2f}",
                "station_x_m": f"{row['station_x_m']:.17g}",
                "section_breadth_m": f"{row['station_breadth_m']:.17g}",
                "taper_factor": f"{row['taper_factor']:.17g}",
                "condition": row["condition"],
                "compressed_fibre": row["compressed_fibre"],
                "gamma_s": f"{row['gamma_s']:.17g}",
                "gamma_w": f"{row['gamma_w']:.17g}",
                "M_sw_Nm": f"{row['M_sw_component_Nm']:.17g}",
                "M_wv_Nm": f"{row['M_wv_component_Nm']:.17g}",
                "lhs_Nm": f"{row['lhs_Nm']:.17g}",
                "M_u_smith_Nm": f"{row['M_u_smith_Nm']:.17g}",
                "M_y_elastic_Nm": f"{row['M_y_elastic_Nm']:.17g}",
                "M_u_over_M_y": f"{row['M_u_over_M_y']:.17g}",
                "utilisation": f"{row['utilisation']:.17g}",
                "verdict": row["verdict"],
                "dominant_mechanism": row["dominant_mechanism"],
                "controlling_member": (row["controlling_member"] or {}).get("member", ""),
                "W_elastic_m3": f"{row['elastic']['section_modulus_m3']:.17g}",
                "grid_gci_fibres_percent": f"{fibre_gci:.17g}",
                "grid_gci_steps_percent": f"{step_gci:.17g}",
                "fibres": row["fibres"],
                "strips": study["settings"]["reference_strips"],
                "curvature_steps": study["settings"]["reference_steps"],
            })


def _independent_load_path(case_key: str):
    """Re-integrate the beam loads from scratch (separate code path from the scan)."""
    case = {c.key: c for c in CASE_CATALOGUE}[case_key]
    loads = load_distributions(case)
    return loads


def _trapezoid_moment(x, q, station: float) -> float:
    """M(x) = int_0^x V ds - (x/L) int_0^L V ds evaluated at ``station``."""
    step = x[1] - x[0]
    shear = np.concatenate(([0.0], np.cumsum(0.5 * (q[1:] + q[:-1]) * step)))
    moment = np.concatenate(([0.0], np.cumsum(0.5 * (shear[1:] + shear[:-1]) * step)))
    moment = moment - moment[-1] * (x - x[0]) / (x[-1] - x[0])
    return float(np.interp(station, x, moment))


def recompute_from_artifacts(json_path: Path, csv_path: Path) -> dict:
    """Recompute the clause arithmetic and the load effects from the artifacts.

    The load effects are re-integrated with an independent trapezoid path
    (``_trapezoid_moment``) rather than the ``beam_moment_from_load`` helper used
    by the scan, so the residual covers the whole chain
    loads -> demand -> utilisation -> verdict and not only the final division.
    """
    data = json.loads(json_path.read_text())
    loads = _independent_load_path(data["hull_model"]["case"])
    x = loads["x"]
    q_still = None  # recomputed below through the public helper of the module
    from tensorfem.hull_girder_csr_chain import (
        homogeneous_loaded_tanker_weights, solve_equilibrium_draft_trim,
        weight_centroid, weight_per_length,
    )
    case = {c.key: c for c in CASE_CATALOGUE}[data["hull_model"]["case"]]
    form = case.form()
    items, total = homogeneous_loaded_tanker_weights(form)
    equilibrium = solve_equilibrium_draft_trim(form, total, weight_centroid(form, items))
    q_still = equilibrium["buoyancy_N_per_m"] - weight_per_length(form, items)
    m_sw_path = np.array([_trapezoid_moment(x, q_still, station) for station in x])
    json_rows = {row["case_id"]: row for row in data["cases"]}
    json_residuals = {case_id: recompute_residual(row) for case_id, row in json_rows.items()}
    from tensorfem.hull_girder_csr_chain import wave_load_distribution
    wave_path = {}
    for condition, crest in (("sagging", 0.0), ("hogging", form.Lpp / 2.0)):
        q_wave = wave_load_distribution(form, h_w=loads["design_wave_height_m"],
                                       crest_position=crest)
        wave_path[condition] = np.array([_trapezoid_moment(x, q_wave, station) for station in x])
    csv_residuals = {}
    load_residuals = {}
    with csv_path.open() as handle:
        for record in csv.DictReader(handle):
            case_id = record["case_id"]
            station = float(record["station_x_m"])
            condition = record["condition"]
            independent_sw = float(np.interp(station, x, m_sw_path))
            independent_wv = float(np.interp(station, x, wave_path[condition]))
            stored_sw = float(record["M_sw_Nm"])
            stored_wv = float(record["M_wv_Nm"])
            # the stored components are the same-sign parts of the signed moments
            expected_sw = max(independent_sw, 0.0) if condition == "sagging" else max(-independent_sw, 0.0)
            expected_wv = max(independent_wv, 0.0) if condition == "sagging" else max(-independent_wv, 0.0)
            load_residuals[case_id] = max(
                abs(stored_sw - expected_sw) / max(abs(expected_sw), 1.0),
                abs(stored_wv - expected_wv) / max(abs(expected_wv), 1.0),
            )
            gamma_s = float(record["gamma_s"])
            gamma_w = float(record["gamma_w"])
            lhs = gamma_s * float(record["M_sw_Nm"]) + gamma_w * float(record["M_wv_Nm"])
            utilisation = lhs / float(record["M_u_smith_Nm"])
            elastic = MILD_STEEL_YIELD * float(record["W_elastic_m3"])
            residuals = [
                abs(lhs - float(record["lhs_Nm"])) / max(abs(lhs), 1e-30),
                abs(utilisation - float(record["utilisation"])) / utilisation,
                abs(elastic - float(record["M_y_elastic_Nm"])) / float(record["M_y_elastic_Nm"]),
                abs(float(record["M_sw_Nm"]) - json_rows[case_id]["M_sw_component_Nm"])
                / max(float(record["M_sw_Nm"]), 1e-30),
                abs(float(record["utilisation"]) - json_rows[case_id]["utilisation"])
                / json_rows[case_id]["utilisation"],
            ]
            csv_residuals[case_id] = max(residuals)
    return {
        "clause": "gamma_s M_sw + gamma_w M_wv = lhs; utilisation = lhs/M_u; M_y = sigma_y W",
        "json_residuals": json_residuals,
        "csv_residuals": csv_residuals,
        "load_path_residuals": load_residuals,
        "max_load_path_residual": max(load_residuals.values()),
        "max_json_residual": max(json_residuals.values()),
        "max_csv_residual": max(csv_residuals.values()),
        "max_csv_json_consistency": max(csv_residuals.values()),
        "threshold": 1e-6,
        "verdict": "PASS" if max(max(json_residuals.values()), max(csv_residuals.values()),
                                max(load_residuals.values())) < 1e-6 else "FAIL",
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _save(fig, name: str) -> Path:
    path = FIG_DIR / name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_utilisation(study: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for axis, condition in zip(axes, ("sagging", "hogging")):
        rows = [r for r in study["cases"] if r["condition"] == condition]
        station = [r["station_x_fraction"] for r in rows]
        axis.plot(station, [r["utilisation"] for r in rows], "o-", label="utilisation (unit factors)")
        axis.plot(station, [r["gamma_w_max_at_gamma_s_1"] for r in rows], "s--",
                  label=r"$\gamma_w^{max}$ at $\gamma_s=1$")
        axis.axhline(1.0, color="crimson", linestyle=":", linewidth=1.2, label="unity (CSR limit)")
        axis.set_title(f"{condition}: {CONDITION_COMPRESSED_FIBRE[condition]} in compression")
        axis.set_xlabel("x / L")
        axis.set_ylabel("(-)")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    fig.suptitle("W23 ULS utilisation along the hull girder (unit-factor reference value)")
    fig.tight_layout()
    return _save(fig, "utilisation_vs_station.png")


def plot_capacity(study: dict) -> Path:
    fig, axis = plt.subplots(figsize=(9.0, 4.8))
    styles = {"sagging": "o-", "hogging": "s-"}
    for condition, style in styles.items():
        rows = [r for r in study["cases"] if r["condition"] == condition]
        station = [r["station_x_fraction"] for r in rows]
        axis.plot(station, [r["M_u_smith_Nm"] / 1e9 for r in rows], style,
                  label=f"M_u Smith, {condition}")
        axis.plot(station, [r["M_y_elastic_Nm"] / 1e9 for r in rows], style,
                  alpha=0.45, label=f"M_y elastic, {condition}")
        axis.plot(station, [r["lhs_Nm"] / 1e9 for r in rows], "x:", color="crimson",
                  label=f"demand (unit factors), {condition}")
    axis.set_xlabel("x / L")
    axis.set_ylabel("M (GN.m)")
    axis.set_title("Ultimate (Smith) vs elastic (yield) capacity and demand")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return _save(fig, "capacity_vs_station.png")


def plot_moment_curvature(study: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for axis, condition in zip(axes, ("sagging", "hogging")):
        rows = [r for r in study["cases"] if r["condition"] == condition]
        for row in rows:
            curvature = [point[0] for point in row["curve"]]
            moment = [point[1] / 1e9 for point in row["curve"]]
            axis.plot(curvature, moment, label=f"x/L={row['station_x_fraction']:.2f}")
        axis.set_title(f"Smith moment-curvature, {condition}")
        axis.set_xlabel("curvature (1/m)")
        axis.set_ylabel("M (GN.m)")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)
    fig.suptitle("W23 progressive collapse curves at six stations")
    fig.tight_layout()
    return _save(fig, "moment_curvature.png")


def plot_convergence(study: dict) -> Path:
    levels = study["settings"]["fibre_levels"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for axis, condition in zip(axes, ("sagging", "hogging")):
        rows = [r for r in study["grid_convergence"] if r["condition"] == condition]
        for row in rows:
            values = [row["fibre_family"]["values"][str(level)] / 1e9 for level in levels]
            axis.semilogx(levels, values, "o-", label=f"x/L={row['station_x_fraction']:.2f}")
        axis.set_xlabel("fibres per deep member (-)")
        axis.set_ylabel("M_u (GN.m)")
        axis.set_title(f"fibre convergence, {condition}")
        axis.grid(alpha=0.3, which="both")
        axis.legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, "grid_convergence.png")


def plot_pairing_audit(audit: dict) -> Path:
    rows = audit["rows"]
    labels = [row["case"] for row in rows]
    index = range(len(rows))
    fig, axis = plt.subplots(figsize=(8.6, 4.2))
    width = 0.2
    axis.bar([i - 1.5 * width for i in index], [r["w22_utilisation_sagging"] for r in rows], width,
             label="W22 sagging")
    axis.bar([i - 0.5 * width for i in index], [r["corrected_utilisation_sagging"] for r in rows],
             width, label="corrected sagging (deck compression)")
    axis.bar([i + 0.5 * width for i in index], [r["w22_utilisation_hogging"] for r in rows], width,
             label="W22 hogging")
    axis.bar([i + 1.5 * width for i in index], [r["corrected_utilisation_hogging"] for r in rows],
             width, label="corrected hogging (bottom compression)")
    axis.set_xticks(list(index))
    axis.set_xticklabels(labels, rotation=12)
    axis.axhline(1.0, color="crimson", linestyle=":")
    axis.set_ylabel("utilisation (-)")
    axis.set_title("W22 pairing defect: utilisation before/after correct fibre pairing")
    axis.legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, "pairing_audit.png")


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stations", type=str, default=",".join(str(s) for s in STATION_FRACTIONS))
    parser.add_argument("--fibre-levels", type=str, default=",".join(str(s) for s in FIBRE_LEVELS))
    parser.add_argument("--step-levels", type=str, default=",".join(str(s) for s in STEP_LEVELS))
    parser.add_argument("--gamma-s", type=float, default=1.0)
    parser.add_argument("--gamma-w", type=float, default=1.0)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    stations = tuple(float(part) for part in args.stations.split(","))
    fibre_levels = tuple(int(part) for part in args.fibre_levels.split(","))
    step_levels = tuple(int(part) for part in args.step_levels.split(","))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    study = run_uls_scan(stations=stations, fibre_levels=fibre_levels, step_levels=step_levels,
                         gamma_s=args.gamma_s, gamma_w=args.gamma_w,
                         factor_register_path=REGISTER, repo_root=ROOT)
    study["pairing_audit"] = pairing_audit()
    study["factor_sensitivity"] = factor_sensitivity(study)
    study["sensitivity"] = study["factor_sensitivity"]

    grid_lookup = {row["case_id"]: row for row in study["grid_convergence"]}
    csv_path = OUT_DIR / "section_scan.csv"
    write_csv(study, grid_lookup, csv_path)

    grid_payload = {
        "study": study["study"],
        "mission_id": study["mission_id"],
        "repository": study["repository"],
        "criterion": "Roache GCI of M_u < 5% on three grids; observed order reported",
        "safety_factor": study["settings"]["gci_safety_factor"],
        "reference": {"strips": study["settings"]["reference_strips"],
                      "curvature_steps": study["settings"]["reference_steps"]},
        "axes": {"fibres_per_deep_member": list(fibre_levels),
                 "curvature_steps": list(step_levels)},
        "cases": study["grid_convergence"],
        "max_gci_percent_fibres": max(r["fibre_family"]["gci_fine_percent"]
                                      for r in study["grid_convergence"]),
        "max_gci_percent_curvature_steps": max(r["step_family"]["gci_fine_percent"]
                                               for r in study["grid_convergence"]),
        "max_order_fibres": max(r["fibre_family"]["order_observed"]
                                for r in study["grid_convergence"]
                                if r["fibre_family"]["order_observed"] is not None),
        "min_order_fibres": min(r["fibre_family"]["order_observed"]
                                for r in study["grid_convergence"]
                                if r["fibre_family"]["order_observed"] is not None),
    }
    grid_payload["verdict"] = "PASS" if max(
        grid_payload["max_gci_percent_fibres"],
        grid_payload["max_gci_percent_curvature_steps"]) < 5.0 else "FAIL"
    (OUT_DIR / "grid_convergence.json").write_text(json.dumps(grid_payload, indent=2) + "\n")

    json_path = OUT_DIR / "csr_utilisation.json"
    study["provenance"] = {
        "git_commit": _git_commit(),
        "runner": "scripts/w23_hull_girder_uls_scan.py",
        "runner_sha256": _sha256(Path(__file__)),
        "module": "src/tensorfem/hull_girder_uls_scan.py",
        "module_sha256": _sha256(ROOT / "src" / "tensorfem" / "hull_girder_uls_scan.py"),
        "engine": "src/tensorfem/hull_girder_smith_collapse.py",
        "engine_sha256": _sha256(ROOT / "src" / "tensorfem" / "hull_girder_smith_collapse.py"),
        "csr_factor_register": "benchmarks/hull_girder_uls/csr_factor_register.json",
        "csr_factor_register_sha256": _sha256(REGISTER),
        "inputs": {
            "hull_form": "kvlcc2 (src/tensorfem/hull_girder_csr_chain.py:KVLCC2_PARTICULARS)",
            "stations": list(stations), "fibre_levels": list(fibre_levels),
            "step_levels": list(step_levels),
        },
        "figures": [],
    }
    # first pass writes the study so the residuals are computed from the artifact itself
    json_path.write_text(json.dumps(study, indent=2) + "\n")
    study["recomputation"] = recompute_from_artifacts(json_path, csv_path)

    for criterion in study["criteria"]:
        if criterion["id"] == "W23-C2a":
            criterion["value"]["max_residual"] = study["recomputation"]["max_json_residual"]
            criterion["value"]["max_residual_csv"] = study["recomputation"]["max_csv_residual"]
            criterion["verdict"] = study["recomputation"]["verdict"]

    if not args.no_figures:
        figures = [plot_utilisation(study), plot_capacity(study),
                   plot_moment_curvature(study), plot_convergence(study),
                   plot_pairing_audit(study["pairing_audit"])]
        study["provenance"]["figures"] = [
            {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)} for path in figures]
    json_path.write_text(json.dumps(study, indent=2) + "\n")

    print(f"cases: {len(study['cases'])}")
    for row in study["cases"]:
        print(f"{row['case_id']:>28} M_u={row['M_u_smith_Nm']/1e9:7.3f} "
              f"M_y={row['M_y_elastic_Nm']/1e9:7.3f} GN.m "
              f"U={row['utilisation']:.3f} {row['verdict']:>4} "
              f"({row['dominant_mechanism'].split(' (')[0]})")
    for criterion in study["criteria"]:
        print(f"{criterion['id']}: {criterion['verdict']}")
    print(f"max recomputation residual: {study['recomputation']['max_json_residual']:.3e}")
    print(f"max GCI: {grid_payload['max_gci_percent_fibres']:.3f}% (fibres), "
          f"{grid_payload['max_gci_percent_curvature_steps']:.3f}% (steps)")
    print(f"wrote {csv_path}, {json_path}, {OUT_DIR / 'grid_convergence.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())