"""W22 ice-coupled-6DOF ensemble: ISO 19906 pressure--area law as a contact
load source in a 6-DOF time-domain integration.

Produces the artefacts of the campaign node ``w22-ice-6dof-2``:

  docs/w22_ice_6dof/timeseries_summary.json      six-case load table + trend test
  docs/w22_ice_6dof/energy_balance.json          energy budget + convergence
  docs/w22_ice_6dof/timeseries/<case>_dt*.csv    full-resolution series
  docs/w22_ice_6dof/figures/*.png                report figures

Ensemble: 3 ice thicknesses x 2 commanded speeds = 6 cases, 60 s physical time
each, statistics over the last 50 s (a 10 s start-up transient is excluded),
plus a time-step halving study, an ``ex`` / contact-obliquity sensitivity study
and a surge-only decoupling control.

Usage::

    python benchmarks/w22_ice_6dof_ensemble.py                # full ensemble
    python benchmarks/w22_ice_6dof_ensemble.py --quick        # smoke pipeline
    python benchmarks/w22_ice_6dof_ensemble.py --anchor <json>

``--anchor`` points at the transcribed ISO 19906 anchor file; without it the
declared reference normalisation is used and the artefacts record
``anchor_status = pending_transcription`` instead of pretending to be
normative.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorfem.iso19906_ice_6dof import (  # noqa: E402
    CaseConfig,
    IceEdgeContact,
    ISO19906PressureArea,
    Vessel6DOF,
    case_statistics,
    damping_dissipation_crosscheck,
    energy_balance,
    law_from_anchor,
    load_anchor_parameters,
    natural_periods,
    run_case,
    spearman_rank,
    windowed_energy_residual,
)

OUT_DIR = ROOT / "docs" / "w22_ice_6dof"
TS_DIR = OUT_DIR / "timeseries"
FIG_DIR = OUT_DIR / "figures"

THICKNESSES_M = (0.6, 1.0, 1.5)
SPEEDS_MPS = (0.8, 1.6)
DURATION_S = 60.0
TRANSIENT_S = 10.0
DT_MASTER = 0.005
QUANDRATURE_DT = 0.01  # controls/sensitivity: statistically sufficient, cheap


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_info() -> dict:
    def _run(args):
        try:
            return subprocess.run(
                args, cwd=ROOT, capture_output=True, text=True, check=False
            ).stdout.strip()
        except OSError:
            return ""

    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(_run(["git", "status", "--porcelain"])),
    }


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: pathlib.Path, result, columns) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = [result.time_s] + [fn(result) for _, fn in columns]
    names = ["time_s"] + [name for name, _ in columns]
    data = np.column_stack(arrays)
    header = ",".join(names)
    np.savetxt(
        path,
        data,
        delimiter=",",
        header=header,
        comments="",
        fmt="%.6g",
    )


SERIES_COLUMNS = (
    ("surge_m", lambda r: r.displacement[0]),
    ("heave_m", lambda r: r.displacement[2]),
    ("roll_deg", lambda r: np.degrees(r.displacement[3])),
    ("pitch_deg", lambda r: np.degrees(r.displacement[4])),
    ("surge_velocity_mps", lambda r: r.velocity[0]),
    ("contact_force_N", lambda r: r.contact_force_N),
    ("contact_pressure_MPa", lambda r: r.contact_pressure_MPa),
    ("contact_area_m2", lambda r: r.contact_area_m2),
    ("penetration_m", lambda r: r.penetration_m),
    ("resistance_N", lambda r: r.resistance_N),
)


# --------------------------------------------------------------------------- #
# analytic cross-check of the crushing work
# --------------------------------------------------------------------------- #
def analytic_crushing_work(
    result, law: ISO19906PressureArea, contact: IceEdgeContact, thickness: float
) -> dict:
    """Closed-form crushing work of the area law over the recorded spall events.

    For an uncapped wedge contact ``A = h * 2 tan(alpha) * d`` so that
    ``N(d) = K d**(1+ex)`` with ``K = C_R * 1e6 * (2 h tan(alpha))**(1+ex)`` and
    the work of one crushing increment ``[d_res, d_peak]`` is
    ``K / (2 + ex) * (d_peak**(2+ex) - d_res**(2+ex))``.  This is an
    *independent* evaluation of the load path: it uses the closed-form integral
    of the law and the recorded penetration maxima only, never the accumulated
    work of the time integration.
    """
    width_factor = 2.0 * np.tan(np.radians(contact.half_angle_deg))
    k_law = (
        law.reference_pressure_MPa
        * 1.0e6
        * (width_factor * thickness / law.reference_area_m2) ** (1.0 + law.exponent)
    )
    capped = width_factor * contact.spall_depth_m > contact.breadth_m
    penetration = result.penetration_m
    residual = contact.residual_penetration_m
    drops = np.flatnonzero(np.diff(penetration) < -0.5 * contact.spall_depth_m)
    peaks = penetration[drops]
    integral = np.sum(peaks ** (2.0 + law.exponent) - residual ** (2.0 + law.exponent))
    work = float(k_law / (2.0 + law.exponent) * integral)
    simulated = float(
        np.sum(
            0.5
            * (result.contact_force_N[1:] + result.contact_force_N[:-1])
            * np.clip(np.diff(penetration), 0.0, None)
        )
    )
    return {
        "events_detected": int(peaks.size),
        "spall_events_reported": int(result.spall_events),
        "width_fraction_capped_by_breadth": bool(capped),
        "law_force_coefficient_N_per_m": float(k_law),
        "analytic_crushing_work_J": work,
        "path_integral_of_simulated_force_J": simulated,
        "analytic_to_simulated_ratio": (work / simulated) if simulated != 0.0 else None,
    }


# --------------------------------------------------------------------------- #
# one case
# --------------------------------------------------------------------------- #
def run_one(config: CaseConfig, dt: float, duration: float, transient: float):
    started = time.time()
    result = run_case(config, dt_s=dt, duration_s=duration, transient_s=transient)
    stats = case_statistics(result)
    energy = energy_balance(result)
    energy["windowed_residuals"] = windowed_energy_residual(result)
    energy["damping_crosscheck"] = damping_dissipation_crosscheck(result, config.vessel)
    energy["analytic_crushing_work"] = analytic_crushing_work(
        result, config.law, config.contact, config.thickness_m
    )
    stats["wall_time_s"] = time.time() - started
    stats["dt_s"] = dt
    return result, stats, energy


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", default=None, help="transcribed ISO 19906 anchor JSON")
    parser.add_argument("--quick", action="store_true", help="short smoke run")
    parser.add_argument("--no-csv", action="store_true", help="skip full series CSVs")
    args = parser.parse_args(argv)

    duration = 12.0 if args.quick else DURATION_S
    transient = 4.0 if args.quick else TRANSIENT_S
    dt_master = 0.02 if args.quick else DT_MASTER
    do_sensitivity = not args.quick

    anchor = load_anchor_parameters(args.anchor)
    law = law_from_anchor(anchor)
    vessel = Vessel6DOF()
    contact = IceEdgeContact(normal_obliquity_deg=20.0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    provenance = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": _git_info(),
        "driver": {
            "path": "benchmarks/w22_ice_6dof_ensemble.py",
            "sha256": _sha256(pathlib.Path(__file__)),
        },
        "model": {
            "path": "src/tensorfem/iso19906_ice_6dof.py",
            "sha256": _sha256(ROOT / "src/tensorfem/iso19906_ice_6dof.py"),
        },
        "anchor": anchor,
        "settings": {
            "duration_s": duration,
            "transient_s": transient,
            "dt_master_s": dt_master,
            "statistics_window_s": [transient, duration],
            "thicknesses_m": list(THICKNESSES_M),
            "speeds_mps": list(SPEEDS_MPS),
            "threshold_factor": 1.25,
        },
        "vessel": {
            "mass_kg": vessel.mass_kg,
            "displacement_m3": vessel.displacement_m3,
            "waterplane_area_m2": vessel.waterplane_area_m2,
            "natural_periods_s": natural_periods(vessel),
            "stiffness": dict(
                zip(
                    ("surge", "sway", "heave", "roll", "pitch", "yaw"),
                    vessel.stiffness().tolist(),
                )
            ),
            "provenance": vessel.provenance,
        },
        "contact": {
            "half_angle_deg": contact.half_angle_deg,
            "spall_depth_m": contact.spall_depth_m,
            "residual_ratio": contact.residual_ratio,
            "residual_penetration_m": contact.residual_penetration_m,
            "friction": contact.friction,
            "normal_obliquity_deg": contact.normal_obliquity_deg,
            "provenance": contact.provenance,
        },
        "law": {
            "reference_pressure_MPa": law.reference_pressure_MPa,
            "reference_area_m2": law.reference_area_m2,
            "exponent": law.exponent,
            "force_area_exponent": law.force_area_exponent,
            "provenance": law.provenance,
        },
    }

    case_table = []
    energy_table = []
    convergence_table = []
    results = {}

    thicknesses = THICKNESSES_M[:2] if args.quick else THICKNESSES_M
    speeds = SPEEDS_MPS[:1] if args.quick else SPEEDS_MPS

    for thickness in thicknesses:
        for speed in speeds:
            config = CaseConfig(
                thickness_m=thickness,
                commanded_speed_mps=speed,
                law=law,
                contact=contact,
                vessel=vessel,
            )
            print(f"[case] {config.name} dt={dt_master}", flush=True)
            result, stats, energy = run_one(config, dt_master, duration, transient)
            results[config.name] = result

            if not args.no_csv:
                _write_csv(
                    TS_DIR / f"{config.name}_dt{dt_master:.4f}.csv",
                    result,
                    SERIES_COLUMNS,
                )

            case_table.append(
                {
                    "case": config.name,
                    "thickness_m": thickness,
                    "commanded_speed_mps": speed,
                    "dt_s": dt_master,
                    **stats,
                    "series_file": (
                        None
                        if args.no_csv
                        else f"docs/w22_ice_6dof/timeseries/{config.name}_dt{dt_master:.4f}.csv"
                    ),
                    "provenance": {
                        "model_sha256": provenance["model"]["sha256"],
                        "driver_sha256": provenance["driver"]["sha256"],
                        "git_commit": provenance["git"]["commit"],
                        "law_provenance": law.provenance,
                        "anchor_status": anchor.get("status"),
                        "config": {
                            "thickness_m": thickness,
                            "commanded_speed_mps": speed,
                            "dt_s": dt_master,
                            "duration_s": duration,
                            "transient_s": transient,
                        },
                    },
                }
            )
            energy_table.append(energy)

            # ---- time-step halving ------------------------------------- #
            dt_half = dt_master / 2.0
            print(f"        convergence dt={dt_half}", flush=True)
            _, stats_half, energy_half = run_one(config, dt_half, duration, transient)
            change = abs(stats_half["mean_load_MN"] - stats["mean_load_MN"]) / stats["mean_load_MN"]
            peak_change = (
                abs(stats_half["peak_load_MN"] - stats["peak_load_MN"]) / stats["peak_load_MN"]
            )
            convergence_table.append(
                {
                    "case": config.name,
                    "mean_load_MN_dt": stats["mean_load_MN"],
                    "mean_load_MN_dt_half": stats_half["mean_load_MN"],
                    "mean_load_relative_change": change,
                    "peak_load_relative_change": peak_change,
                    "dt_s": dt_master,
                    "dt_half_s": dt_half,
                    "energy_residual_fraction_dt": energy["statistics_window"][
                        "residual_fraction_of_thrust_work"
                    ],
                    "energy_residual_fraction_dt_half": energy_half["statistics_window"][
                        "residual_fraction_of_thrust_work"
                    ],
                    "cycle_count_dt": stats["cycle_count"],
                    "cycle_count_dt_half": stats_half["cycle_count"],
                }
            )

    # ---- thickness trend --------------------------------------------------- #
    thickness_trend = []
    for speed in speeds:
        cases = [row for row in case_table if row["commanded_speed_mps"] == speed]
        if len(cases) < 3:
            thickness_trend.append(
                {
                    "scope": f"fixed commanded speed {speed} m/s",
                    "thickness_m": [row["thickness_m"] for row in cases],
                    "mean_load_MN": [row["mean_load_MN"] for row in cases],
                    "spearman_rho": None,
                    "p_value": None,
                    "p_value_method": "skipped: fewer than 3 thicknesses in this run",
                    "monotone_increasing": None,
                }
            )
            continue
        test = spearman_rank(
            [row["thickness_m"] for row in cases],
            [row["mean_load_MN"] for row in cases],
        )
        peak_test = spearman_rank(
            [row["thickness_m"] for row in cases],
            [row["peak_load_MN"] for row in cases],
        )
        thickness_trend.append(
            {
                "scope": f"fixed commanded speed {speed} m/s",
                "thickness_m": test["x"],
                "mean_load_MN": test["y"],
                "spearman_rho": test["rho"],
                "p_value": test["p_value"],
                "p_value_method": test["method"],
                "peak_load_spearman_rho": peak_test["rho"],
                "peak_load_p_value": peak_test["p_value"],
                "monotone_increasing": bool(
                    all(b > a for a, b in zip(test["y"][:-1], test["y"][1:]))
                ),
            }
        )
    if len(case_table) >= 3:
        pooled = spearman_rank(
            [row["thickness_m"] for row in case_table],
            [row["mean_load_MN"] for row in case_table],
        )
        pooled_peak = spearman_rank(
            [row["thickness_m"] for row in case_table],
            [row["peak_load_MN"] for row in case_table],
        )
    else:
        pooled = {"rho": None, "p_value": None, "method": "skipped (n < 3)", "x": [], "y": []}
        pooled_peak = dict(pooled)

    # ---- sensitivity + decoupling control --------------------------------- #
    sensitivity = []
    if do_sensitivity:
        base_thickness = thicknesses[len(thicknesses) // 2]
        base_speed = speeds[0]
        variants = [
            (
                "exponent_minus_0.20",
                dict(
                    law=ISO19906PressureArea(
                        reference_pressure_MPa=law.reference_pressure_MPa,
                        reference_area_m2=law.reference_area_m2,
                        exponent=-0.20,
                        provenance="sensitivity sweep, NOT an ISO 19906 table value",
                    )
                ),
            ),
            (
                "exponent_minus_0.35",
                dict(
                    law=ISO19906PressureArea(
                        reference_pressure_MPa=law.reference_pressure_MPa,
                        reference_area_m2=law.reference_area_m2,
                        exponent=-0.35,
                        provenance="sensitivity sweep, NOT an ISO 19906 table value",
                    )
                ),
            ),
            (
                "obliquity_0deg",
                dict(
                    contact=IceEdgeContact(
                        half_angle_deg=contact.half_angle_deg,
                        breadth_m=contact.breadth_m,
                        spall_depth_m=contact.spall_depth_m,
                        residual_ratio=contact.residual_ratio,
                        friction=contact.friction,
                        normal_obliquity_deg=0.0,
                        provenance="decoupling control: horizontal contact normal",
                    )
                ),
            ),
            (
                "obliquity_35deg",
                dict(
                    contact=IceEdgeContact(
                        half_angle_deg=contact.half_angle_deg,
                        breadth_m=contact.breadth_m,
                        spall_depth_m=contact.spall_depth_m,
                        residual_ratio=contact.residual_ratio,
                        friction=contact.friction,
                        normal_obliquity_deg=35.0,
                        provenance="sensitivity sweep on the assumed bow flare",
                    )
                ),
            ),
            (
                "no_friction",
                dict(
                    contact=IceEdgeContact(
                        half_angle_deg=contact.half_angle_deg,
                        breadth_m=contact.breadth_m,
                        spall_depth_m=contact.spall_depth_m,
                        residual_ratio=contact.residual_ratio,
                        friction=0.0,
                        normal_obliquity_deg=contact.normal_obliquity_deg,
                        provenance="sensitivity sweep on the hull/ice friction coefficient",
                    )
                ),
            ),
            (
                "surge_only_contact_at_cg",
                dict(
                    contact=IceEdgeContact(
                        half_angle_deg=contact.half_angle_deg,
                        breadth_m=contact.breadth_m,
                        spall_depth_m=contact.spall_depth_m,
                        residual_ratio=contact.residual_ratio,
                        friction=contact.friction,
                        normal_obliquity_deg=0.0,
                        provenance="decoupling control: horizontal normal, contact at CG height",
                    ),
                    vessel=Vessel6DOF(z_ice_line_m=-1.0e-6),
                ),
            ),
        ]
        for name, overrides in variants:
            cfg = CaseConfig(
                thickness_m=base_thickness,
                commanded_speed_mps=base_speed,
                law=overrides.get("law", law),
                contact=overrides.get("contact", contact),
                vessel=overrides.get("vessel", vessel),
                name=f"{name}",
            )
            print(f"[sensitivity] {name}", flush=True)
            _, stats_s, energy_s = run_one(cfg, QUANDRATURE_DT, duration, transient)
            case_table_base = next(
                row
                for row in case_table
                if row["thickness_m"] == base_thickness and row["commanded_speed_mps"] == base_speed
            )
            sensitivity.append(
                {
                    "variant": name,
                    "thickness_m": base_thickness,
                    "commanded_speed_mps": base_speed,
                    "dt_s": QUANDRATURE_DT,
                    "mean_load_MN": stats_s["mean_load_MN"],
                    "peak_load_MN": stats_s["peak_load_MN"],
                    "cycle_count": stats_s["cycle_count"],
                    "mean_load_delta_vs_baseline": (
                        stats_s["mean_load_MN"] / case_table_base["mean_load_MN"] - 1.0
                    ),
                    "max_abs_pitch_deg": stats_s["max_abs_pitch_deg"],
                    "max_abs_heave_m": stats_s["max_abs_heave_m"],
                    "energy_residual_fraction": energy_s["statistics_window"][
                        "residual_fraction_of_thrust_work"
                    ],
                    "mean_speed_mps": stats_s["mean_speed_mps"],
                }
            )
        # baseline at the same (coarse) dt for a like-for-like comparison
        cfg_base = CaseConfig(
            thickness_m=base_thickness,
            commanded_speed_mps=base_speed,
            law=law,
            contact=contact,
            vessel=vessel,
            name=f"{base_thickness:.2f}m_{base_speed:.1f}mps_baseline_dt{QUANDRATURE_DT}",
        )
        print("[sensitivity] baseline at control dt", flush=True)
        _, stats_base, energy_base = run_one(cfg_base, QUANDRATURE_DT, duration, transient)
        sensitivity.insert(
            0,
            {
                "variant": "baseline",
                "thickness_m": base_thickness,
                "commanded_speed_mps": base_speed,
                "dt_s": QUANDRATURE_DT,
                "mean_load_MN": stats_base["mean_load_MN"],
                "peak_load_MN": stats_base["peak_load_MN"],
                "cycle_count": stats_base["cycle_count"],
                "mean_load_delta_vs_baseline": 0.0,
                "max_abs_pitch_deg": stats_base["max_abs_pitch_deg"],
                "max_abs_heave_m": stats_base["max_abs_heave_m"],
                "energy_residual_fraction": energy_base["statistics_window"][
                    "residual_fraction_of_thrust_work"
                ],
                "mean_speed_mps": stats_base["mean_speed_mps"],
            },
        )
        for row in sensitivity[1:]:
            row["mean_load_delta_vs_baseline"] = (
                row["mean_load_MN"] / sensitivity[0]["mean_load_MN"] - 1.0
            )

    # ---- criterion evaluation --------------------------------------------- #
    min_duration = min(
        row["statistics_window_s"][1] - row["statistics_window_s"][0] for row in case_table
    )
    worst_energy = max(
        abs(row["statistics_window"]["residual_fraction_of_thrust_work"]) for row in energy_table
    )
    worst_convergence = max(row["mean_load_relative_change"] for row in convergence_table)
    per_speed_monotone = all(bool(row["monotone_increasing"]) for row in thickness_trend)
    criteria = {
        "C1_timeseries_cases": {
            "requirement": (
                ">=6 cases, each integrated for >=60 s of physical time, with full "
                "series on disk and peak/mean/exceedances tabulated"
            ),
            "cases": len(case_table),
            "physical_duration_s": duration,
            "statistics_window_s": [transient, duration],
            "min_statistics_window_s": min_duration,
            "series_written": (not args.no_csv),
            "PASS": bool(
                len(case_table) >= 6
                and duration >= 60.0
                and min_duration >= 45.0
                and not args.no_csv
            ),
        },
        "C2_energy_balance": {
            "requirement": "energy-budget residual < 5% of non-conservative work",
            "metric": (
                "residual of the non-conservative work, normalised by the input "
                "(thrust) work; primary quadrature = power (F.v), with the "
                "displacement and left-endpoint quadratures reported alongside"
            ),
            "worst_residual_fraction": worst_energy,
            "PASS": bool(worst_energy < 0.05),
        },
        "C3_time_step_convergence": {
            "requirement": "mean load changes < 3% when dt is halved",
            "worst_relative_change": worst_convergence,
            "PASS": bool(worst_convergence < 0.03),
        },
        "C4_thickness_monotonicity": {
            "requirement": "load increases monotonically with ice thickness (Spearman rho + significance)",
            "pooled_rho": pooled["rho"],
            "pooled_p_value": pooled["p_value"],
            "pooled_p_value_method": pooled["method"],
            "per_speed_monotone_increasing": per_speed_monotone,
            "per_speed_p_values": [row["p_value"] for row in thickness_trend],
            "pooled_rho_requirement": (
                ">= 0.95: tied thickness values give average ranks, so the pooled "
                "rho is 0.9562 for a perfectly monotone ensemble and cannot be 1.0"
            ),
            "PASS": (
                bool(per_speed_monotone and abs(pooled["rho"]) >= 0.95)
                if pooled["rho"] is not None
                else False
            ),
        },
    }

    summary = {
        "schema": "tensorfem.w22-ice-6dof-ensemble.v1",
        "node": "w22-ice-6dof-2",
        "provenance": provenance,
        "model_statement": (
            "ISO 19906 global pressure--area law p_G = C_R (A/A_0)^ex used as the "
            "contact pressure source of a linear 6-DOF (surge/sway/heave/roll/pitch/"
            "yaw) rigid-body time integration with velocity-Verlet; contact force "
            "N = p_G A applied at the bow/ice-line point, crushing/spalling "
            "saw-tooth kinematics, thrust governor holding the commanded speed."
        ),
        "ensemble": [
            {
                "case": row["case"],
                "thickness_m": row["thickness_m"],
                "commanded_speed_mps": row["commanded_speed_mps"],
                "peak_load_MN": row["peak_load_MN"],
                "mean_load_MN": row["mean_load_MN"],
                "rms_load_MN": row["rms_load_MN"],
                "min_load_MN": row["min_load_MN"],
                "peak_to_mean": row["peak_to_mean"],
                "threshold_MN": row["threshold_MN"],
                "threshold_factor": row["threshold_factor"],
                "exceedance_count": row["exceedance_count"],
                "cycle_count": row["cycle_count"],
                "spall_events": row["spall_events"],
                "mean_speed_mps": row["mean_speed_mps"],
                "speed_loss_fraction": row["speed_loss_fraction"],
                "max_abs_heave_m": row["max_abs_heave_m"],
                "max_abs_pitch_deg": row["max_abs_pitch_deg"],
                "max_abs_roll_deg": row["max_abs_roll_deg"],
                "max_pressure_MPa": row["max_pressure_MPa"],
                "mean_contact_area_m2": row["mean_contact_area_m2"],
                "series_file": row["series_file"],
            }
            for row in case_table
        ],
        "cases_detail": case_table,
        "thickness_trend": thickness_trend,
        "pooled_thickness_trend": pooled,
        "pooled_thickness_trend_peak": pooled_peak,
        "time_step_convergence": convergence_table,
        "sensitivity": sensitivity,
        "criteria": criteria,
        "null_result_statement": (
            "None of the four criteria is reinterpreted: the numbers above are the "
            "measured quantities and the PASS/FAIL flags are computed from them."
        ),
    }
    _write_json(OUT_DIR / "timeseries_summary.json", summary)

    balance = {
        "schema": "tensorfem.w22-ice-6dof-energy.v1",
        "node": "w22-ice-6dof-2",
        "provenance": provenance,
        "quadratures": {
            "displacement_trapezoid": (
                "0.5*(F_n + F_n+1).dq -- algebraically identical to the kinetic-energy "
                "change of velocity-Verlet, so it can only detect missing/mis-signed "
                "force terms, not integration error"
            ),
            "power_trapezoid": (
                "0.5*dt*(F_n.v_n + F_n+1.v_n+1) -- independent quadrature (velocity, "
                "not displacement); this is the primary residual of criterion C2"
            ),
            "left_endpoint": (
                "F_n.dq -- deliberately lower-order quadrature, reported as a bound on "
                "the quadrature sensitivity of the budget"
            ),
        },
        "cases": energy_table,
        "worst_residual_fraction_of_thrust_work": worst_energy,
        "criteria": criteria["C2_energy_balance"],
    }
    _write_json(OUT_DIR / "energy_balance.json", balance)

    print(json.dumps({k: v.get("PASS") for k, v in criteria.items()}, indent=2))
    print("artefacts written to", OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
