"""W23 ``ice-coupled-6dof`` driver: zero-ice RAO check, ice-impact momentum
ledger and peak statistics, all injected into artefacts.

Artefacts (paths relative to the repository root):

    benchmarks/ice_coupled_6dof/zero_ice_rao_check.json
    benchmarks/ice_coupled_6dof/momentum_conservation.csv
    benchmarks/ice_coupled_6dof/peak_weibull_fit.json
    benchmarks/ice_coupled_6dof/ensemble_summary.json
    benchmarks/ice_coupled_6dof/figures/*.png

Every number quoted in ``papers/reports/w23_ice_coupled_6dof_3.md`` is read back
from these files by ``make_report.py`` -- nothing is transcribed by hand.

Usage::

    python benchmarks/ice_coupled_6dof/run_w23_ice_coupled_6dof.py            # full run
    python benchmarks/ice_coupled_6dof/run_w23_ice_coupled_6dof.py --quick    # smoke
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
import subprocess
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from cummins_ice_6dof import (  # noqa: E402
    Cummins6DOF,
    IceField,
    KernelTerm,
    ShipModel,
    fit_rao_time_domain,
    harmonic_amplitude,
    weibull_mle,
)

OUT = HERE
FIG = HERE / "figures"

CONCENTRATIONS = (0.3, 0.5, 0.7)
# pre-declared, distinct per concentration so that the three Poisson checks
# are independent realisations (shared seeds would make them correlated)
SEEDS = tuple(7001 + k for k in range(10))
TARGET_EVENTS_PER_CONCENTRATION = 9000
ICE_DIAMETER_MEDIAN_M = 3.0
COMMANDED_SPEED_MPS = 5.0
DT_ICE = 0.01
DT_RAO = 0.02
RAO_FREQUENCIES = tuple(np.round(np.geomspace(0.15, 1.5, 12), 6))


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def sha256_of(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def damping_ratio(model: ShipModel, omega: float, dof: str) -> float:
    idx = {"heave": 2, "pitch": 4}[dof]
    a = float(model.radiation[dof].added_mass(omega))
    b = float(model.radiation[dof].damping(omega)) + float(model.viscous[idx])
    m = float(model.mass_rigid[idx]) + a
    c = float(model.restoring[idx])
    return b / (2.0 * math.sqrt(m * c))


def kernel_table(model: ShipModel, omegas) -> list:
    rows = []
    for om in omegas:
        a = model.added_mass_vector(om)
        b = model.damping_vector(om)
        rows.append({
            "omega_rad_s": float(om),
            "added_mass_heave_kg": float(a[2]),
            "added_mass_heave_over_mass": float(a[2] / model.mass_kg),
            "damping_heave_Ns_m": float(b[2]),
            "added_mass_pitch_kgm2": float(a[4]),
            "added_mass_pitch_over_inertia": float(a[4] / model.mass_rigid[4]),
            "damping_pitch_Nms_rad": float(b[4]),
            "damping_ratio_heave": float(damping_ratio(model, om, "heave")),
            "damping_ratio_pitch": float(damping_ratio(model, om, "pitch")),
        })
    return rows


def kramers_kronig_check(model: ShipModel, omegas) -> list:
    """Independently integrate the kernel to re-derive A(w), B(w)."""
    tau = np.linspace(0.0, 120.0, 240001)
    out = []
    for dof, idx in (("heave", 2), ("pitch", 4)):
        k = model.radiation[dof].kernel(tau)
        for om in omegas:
            a_num = float(model.radiation[dof].added_mass_inf) - float(
                np.trapezoid(k * np.sin(om * tau), tau) / om)
            b_num = float(np.trapezoid(k * np.cos(om * tau), tau))
            out.append({
                "dof": dof,
                "omega_rad_s": float(om),
                "added_mass_closed_form": float(model.radiation[dof].added_mass(om)),
                "added_mass_kernel_integral": a_num,
                "damping_closed_form": float(model.radiation[dof].damping(om)),
                "damping_kernel_integral": b_num,
                "rel_error_added_mass": abs(a_num / float(model.radiation[dof].added_mass(om)) - 1.0),
                "rel_error_damping": abs(b_num / float(model.radiation[dof].damping(om)) - 1.0),
            })
    return out


def geometric_encounter_rate(field: IceField, model: ShipModel, speed: float,
                             duration: float, rng) -> dict:
    """Independent Monte-Carlo verification of the swept-area rate lambda.

    Floes are placed uniformly in a corridor of length ``V T`` and width
    ``B + D_max``; a floe counts as an encounter when its centre lies inside the
    swept strip of width ``B + D``.  This does not use the exponential
    inter-arrival sampling used by the simulator.
    """
    length = speed * duration
    width = model.breadth_m + 4.0 * field.mean_floe_diameter
    area = length * width
    n_expected_floes = field.number_density() * area
    n_floes = int(rng.poisson(n_expected_floes))
    d = rng.lognormal(math.log(field.median_floe_diameter_m), field.lognormal_sigma, n_floes)
    y = rng.uniform(-0.5 * width, 0.5 * width, n_floes)
    hit = np.abs(y) <= 0.5 * (model.breadth_m + d)
    count = int(np.count_nonzero(hit))
    lam_theory = field.encounter_rate(speed, model.breadth_m)
    return {
        "corridor_length_m": float(length),
        "corridor_width_m": float(width),
        "floes_placed": int(n_floes),
        "encounters_counted": count,
        "duration_s": float(duration),
        "rate_geometric_hz": count / duration,
        "rate_swept_area_hz": float(lam_theory),
        "rel_deviation": abs(count / duration / lam_theory - 1.0),
    }


# --------------------------------------------------------------------------- #
# 1. zero-ice limit against the frequency-domain RAO
# --------------------------------------------------------------------------- #
def run_rao_check(model: ShipModel, quick: bool = False) -> dict:
    freqs = RAO_FREQUENCIES if not quick else RAO_FREQUENCIES[::3]
    rows = []
    t0 = time.time()
    for om in freqs:
        zeta = max(damping_ratio(model, om, "heave"), damping_ratio(model, om, "pitch"))
        periods = int(min(60, max(15, math.ceil(3.0 / max(zeta, 1e-3)))))
        sim = fit_rao_time_domain(model, om, periods, DT_RAO)
        a3 = harmonic_amplitude(sim["time"], sim["heave"], om)
        a5 = harmonic_amplitude(sim["time"], sim["pitch"], om)
        fd = model.rao_frequency_domain(om)
        fd3 = fd["amplitude_abs"][2]
        fd5 = fd["amplitude_abs"][4]
        dev3 = a3 / fd3 - 1.0
        dev5 = a5 / fd5 - 1.0
        rows.append({
            "omega_rad_s": float(om),
            "period_s": float(2.0 * math.pi / om),
            "periods_simulated": periods,
            "damping_ratio_used": float(zeta),
            "rao_heave_frequency_domain_m_per_m": float(fd3),
            "rao_heave_time_domain_m_per_m": float(a3),
            "rao_pitch_frequency_domain_rad_per_m": float(fd5),
            "rao_pitch_time_domain_rad_per_m": float(a5),
            "rel_deviation_heave": float(dev3),
            "rel_deviation_pitch": float(dev5),
            "rel_deviation_max": float(max(abs(dev3), abs(dev5))),
            "pass_5pct": bool(max(abs(dev3), abs(dev5)) < 0.05),
        })
    worst = max(rows, key=lambda r: r["rel_deviation_max"])

    # time-step refinement at three frequencies: the deviation must shrink
    convergence = []
    for om in (freqs[0], freqs[len(freqs) // 2], freqs[-1]):
        for dt in (DT_RAO, DT_RAO / 2.0):
            zeta = max(damping_ratio(model, om, "heave"), damping_ratio(model, om, "pitch"))
            periods = int(min(60, max(15, math.ceil(3.0 / max(zeta, 1e-3)))))
            sim = fit_rao_time_domain(model, om, periods, dt)
            a3 = harmonic_amplitude(sim["time"], sim["heave"], om)
            a5 = harmonic_amplitude(sim["pitch"], sim["pitch"], om) if False else \
                harmonic_amplitude(sim["time"], sim["pitch"], om)
            fd = model.rao_frequency_domain(om)
            convergence.append({
                "omega_rad_s": float(om),
                "dt_s": float(dt),
                "rel_deviation_heave": float(a3 / fd["amplitude_abs"][2] - 1.0),
                "rel_deviation_pitch": float(a5 / fd["amplitude_abs"][4] - 1.0),
            })

    # negative control: time-domain kernel with one term removed while the
    # frequency-domain reference keeps the full kernel.  A test with no
    # resolving power would still report a tiny deviation here.
    mutated = ShipModel()
    for dof in ("heave", "pitch", "roll", "surge", "sway", "yaw"):
        rd = mutated.radiation[dof]
        mutated.radiation[dof] = type(rd)(rd.added_mass_inf, rd.terms[1:])
    control = []
    for om in (freqs[len(freqs) // 4], freqs[-1]):
        zeta = max(damping_ratio(model, om, "heave"), damping_ratio(model, om, "pitch"))
        periods = int(min(60, max(15, math.ceil(3.0 / max(zeta, 1e-3)))))
        sim = fit_rao_time_domain(mutated, om, periods, DT_RAO)
        a3 = harmonic_amplitude(sim["time"], sim["heave"], om)
        fd = model.rao_frequency_domain(om)
        control.append({
            "omega_rad_s": float(om),
            "rel_deviation_heave": float(a3 / fd["amplitude_abs"][2] - 1.0),
            "note": "time-domain kernel has one exponential term removed; a real "
                    "mismatch must show up as a large deviation",
        })

    return {
        "case": "zero_ice_limit_time_vs_frequency_domain",
        "git_commit": git_commit(),
        "model": {
            "length_m": model.length_m, "breadth_m": model.breadth_m,
            "draught_m": model.draught_m,
            "mass_kg": model.mass_kg,
            "natural_frequency_rad_s": model.natural_frequency,
            "added_mass_inf": model.radiation["heave"].added_mass_inf,
            "kernel_terms_heave": [[t.gain, t.alpha, t.beta]
                                   for t in model.radiation["heave"].terms],
            "kernel_terms_pitch": [[t.gain, t.alpha, t.beta]
                                   for t in model.radiation["pitch"].terms],
            "provenance": model.provenance,
        },
        "excitation": "Froude-Krylov + Smith correction, no diffraction, "
                      "identical in time and frequency domain, unit wave amplitude",
        "dt_s": DT_RAO,
        "frequencies": rows,
        "worst_point": worst,
        "criterion": {"metric": "max over heave/pitch of |RAO_td/RAO_fd - 1|",
                      "threshold": 0.05,
                      "pass": bool(all(r["pass_5pct"] for r in rows))},
        "kernel_kramers_kronig": kramers_kronig_check(model, (0.2, 0.5, 0.7, 1.0)),
        "added_mass_damping_table": kernel_table(model, (0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0)),
        "time_step_refinement": convergence,
        "negative_control_kernel_mismatch": control,
        "runtime_s": time.time() - t0,
    }


# --------------------------------------------------------------------------- #
# 2. random ice field ensemble
# --------------------------------------------------------------------------- #
def run_ice_ensemble(model: ShipModel, quick: bool = False) -> tuple:
    seeds = SEEDS if not quick else SEEDS[:2]
    target = TARGET_EVENTS_PER_CONCENTRATION if not quick else 400
    cases = []
    all_events = []
    peaks = {c: {"heave": [], "pitch": []} for c in CONCENTRATIONS}
    inter_arrivals = {c: [] for c in CONCENTRATIONS}
    geo_check = []
    t0 = time.time()

    for ci, conc in enumerate(CONCENTRATIONS):
        field = IceField(concentration=conc, median_floe_diameter_m=ICE_DIAMETER_MEDIAN_M)
        lam = field.encounter_rate(COMMANDED_SPEED_MPS, model.breadth_m)
        per_case_events = target / len(seeds)
        duration = per_case_events / lam
        nsteps = max(200, int(round(duration / DT_ICE)))
        duration = nsteps * DT_ICE
        geo_check.append(dict(concentration=conc,
                              **geometric_encounter_rate(field, model,
                                                         COMMANDED_SPEED_MPS,
                                                         20000.0,
                                                         np.random.default_rng(900 + ci))))
        case_seeds = [7001 + 1000 * ci + k for k in range(len(seeds))]
        for k, seed in enumerate(case_seeds):
            sim = Cummins6DOF(model)
            out = sim.run(duration, DT_ICE, ice=field, seed=seed,
                          commanded_speed=COMMANDED_SPEED_MPS,
                          initial_velocity=COMMANDED_SPEED_MPS)
            events = out["events"]
            arrivals = sorted(ev["t_arrival_s"] for ev in events)
            gaps = np.diff(arrivals).tolist() if len(arrivals) > 1 else []
            inter_arrivals[conc].extend(gaps)
            peaks[conc]["heave"].extend(ev["peak_heave_m"] for ev in events)
            peaks[conc]["pitch"].extend(ev["peak_pitch_rad"] for ev in events)
            for ev in events:
                ev["case_id"] = f"C{conc:.1f}_s{seed}"
                ev["concentration"] = conc
                ev["seed"] = seed
                all_events.append(ev)
            speed = out["surge_m"] / np.maximum(out["time"], 1e-9)
            tail = out["time"] >= 0.5 * duration
            mean_speed = float(np.mean(np.gradient(out["surge_m"][tail], out["time"][tail])))
            cases.append({
                "case_id": f"C{conc:.1f}_s{seed}",
                "concentration": conc,
                "seed": seed,
                "duration_s": float(duration),
                "dt_s": DT_ICE,
                "steps": nsteps,
                "theta_ratio": float(conc / field.number_density()) if False else None,
                "poisson_rate_hz": float(out["poisson_rate_hz"]),
                "poisson_expected": float(out["poisson_expected"]),
                "encounters": int(out["encounter_count"]),
                "encounters_completed": int(out["completed_contacts"]),
                "rel_deviation_vs_poisson": float(
                    out["encounter_count"] / out["poisson_expected"] - 1.0),
                "max_abs_resid_heave": float(max((e["resid_heave"] for e in events), default=float("nan"))),
                "max_abs_resid_pitch": float(max((e["resid_pitch"] for e in events), default=float("nan"))),
                "mean_speed_mps": mean_speed,
                "peak_heave_m": float(np.max(np.abs(out["heave_m"][tail]))),
                "peak_pitch_deg": float(np.degrees(np.max(np.abs(out["pitch_rad"][tail])))),
                "runtime_s": None,
            })
    elapsed = time.time() - t0
    for c in cases:
        c.pop("theta_ratio", None)
    # inter-arrival statistics: exponential (CV = 1) is the Poisson signature
    for conc in CONCENTRATIONS:
        g = np.array(inter_arrivals[conc])
        if g.size > 10:
            lam_emp = 1.0 / float(np.mean(g))
            lam_theory = IceField(concentration=conc,
                                  median_floe_diameter_m=ICE_DIAMETER_MEDIAN_M
                                  ).encounter_rate(COMMANDED_SPEED_MPS, model.breadth_m)
            # Kolmogorov-Smirnov statistic against Exp(lam_theory)
            xs = np.sort(g)
            cdf_emp = np.arange(1, xs.size + 1) / xs.size
            cdf_th = 1.0 - np.exp(-lam_theory * xs)
            ks = float(np.max(np.abs(cdf_emp - cdf_th)))
            ks_crit_5pct = 1.36 / math.sqrt(xs.size)
            inter_arrivals[conc] = {
                "n_gaps": int(g.size),
                "mean_gap_s": float(np.mean(g)),
                "cv": float(np.std(g) / np.mean(g)),
                "empirical_rate_hz": lam_emp,
                "theory_rate_hz": float(lam_theory),
                "ks_statistic": ks,
                "ks_critical_5pct": float(ks_crit_5pct),
                "exponential_not_rejected": bool(ks < ks_crit_5pct),
            }
        else:
            inter_arrivals[conc] = {"n_gaps": int(g.size)}
    return cases, all_events, peaks, inter_arrivals, geo_check, elapsed


def poisson_check(cases, concentrations=CONCENTRATIONS) -> list:
    out = []
    for conc in concentrations:
        sel = [c for c in cases if c["concentration"] == conc]
        observed = sum(c["encounters"] for c in sel)
        expected = sum(c["poisson_expected"] for c in sel)
        duration = sum(c["duration_s"] for c in sel)
        rate = sel[0]["poisson_rate_hz"]
        dev = observed / expected - 1.0
        sd = 1.0 / math.sqrt(max(expected, 1.0))
        out.append({
            "concentration": conc,
            "cases": len(sel),
            "total_duration_s": float(duration),
            "rate_hz": float(rate),
            "expected_count": float(expected),
            "observed_count": int(observed),
            "rel_deviation": float(dev),
            "poisson_relative_sd": float(sd),
            "z_score": float(dev / sd) if sd > 0 else float("nan"),
            "pass_3pct": bool(abs(dev) < 0.03),
        })
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    model = ShipModel()
    print("[1/3] zero-ice RAO check ...", flush=True)
    rao = run_rao_check(model, quick=args.quick)
    (OUT / "zero_ice_rao_check.json").write_text(json.dumps(rao, indent=2) + "\n")

    print("[2/3] ice ensemble ...", flush=True)
    cases, events, peaks, inter, geo, elapsed = run_ice_ensemble(model, quick=args.quick)
    poisson = poisson_check(cases)

    fields = ["case_id", "concentration", "seed", "t_arrival_s", "t_start_s",
              "duration_s", "diameter_m", "thickness_m", "floe_mass_kg",
              "azimuth_deg", "arm_x_m", "v_rel_in_mps", "v_rel_out_mps",
              "restitution", "p_in_kgmps", "J_contact_Ns", "dp_hull_kgmps",
              "dp_floe_kgmps", "dp_fluid_kgmps", "ext_impulse_heave_Ns",
              "resid_heave", "resid_pitch", "resid_max", "cluster_size",
              "peak_heave_m", "peak_pitch_rad", "mutation"]
    with (OUT / "momentum_conservation.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for ev in events:
            writer.writerow({k: ev.get(k) for k in fields})

    print("[3/3] peak statistics ...", flush=True)
    weib = {}
    for conc in CONCENTRATIONS:
        ph = np.array([p for p in peaks[conc]["heave"] if p > 0.0])
        pp = np.array([p for p in peaks[conc]["pitch"] if p > 0.0])
        weib[f"concentration_{conc:.1f}"] = {
            "heave": weibull_mle(ph) if ph.size >= 5 else {"n": int(ph.size)},
            "pitch": weibull_mle(pp) if pp.size >= 5 else {"n": int(pp.size)},
        }
    pooled_h = np.array([p for c in CONCENTRATIONS for p in peaks[c]["heave"] if p > 0.0])
    pooled_p = np.array([p for c in CONCENTRATIONS for p in peaks[c]["pitch"] if p > 0.0])
    weib["pooled"] = {"heave": weibull_mle(pooled_h), "pitch": weibull_mle(pooled_p)}
    weib["definition"] = (
        "per-impact peak |heave| / |pitch| over the contact window plus "
        "1.5 s of free decay; 2-parameter Weibull by MLE; R^2 is the coefficient "
        "of determination of the Weibull probability plot "
        "ln(-ln(1-F)) vs ln(x) with Blom plotting positions"
    )
    weib["criterion"] = {
        "metric": "Weibull probability-plot R^2 for heave and pitch, per concentration",
        "threshold": 0.95,
        "pass_heave": bool(min(v["heave"].get("weibull_plot_r2", 0.0)
                               for k, v in weib.items() if k.startswith("concentration")) >= 0.95),
        "pass_pitch": bool(min(v["pitch"].get("weibull_plot_r2", 0.0)
                               for k, v in weib.items() if k.startswith("concentration")) >= 0.95),
    }
    weib["git_commit"] = git_commit()
    (OUT / "peak_weibull_fit.json").write_text(json.dumps(weib, indent=2) + "\n")

    resid = np.array([e["resid_max"] for e in events])
    summary = {
        "git_commit": git_commit(),
        "node": "w23-ice-coupled-6dof-3",
        "settings": {
            "concentrations": list(CONCENTRATIONS),
            "seeds": {str(c): [7001 + 1000 * i + k
                               for k in range(len(SEEDS if not args.quick else SEEDS[:2]))]
                      for i, c in enumerate(CONCENTRATIONS)},
            "target_events_per_concentration": TARGET_EVENTS_PER_CONCENTRATION
            if not args.quick else 400,
            "floe_diameter_median_m": ICE_DIAMETER_MEDIAN_M,
            "commanded_speed_mps": COMMANDED_SPEED_MPS,
            "dt_s": DT_ICE,
            "iso19906_law": {
                "C_R_MPa": 1.0, "A_0_m2": 1.0, "ex": -0.28,
                "anchor_status": "declared reference normalisation, no normative "
                                 "transcription available in this checkout; the contact "
                                 "force is exactly linear in C_R so every criterion is "
                                 "invariant to C_R, and ex is reported as a declared value",
            },
        },
        "cases": cases,
        "poisson_check": poisson,
        "inter_arrival_statistics": inter,
        "geometric_rate_verification": geo,
        "ensemble_totals": {
            "cases": len(cases),
            "impacts": len(events),
            "simulated_time_s": float(sum(c["duration_s"] for c in cases)),
            "resid_max_over_all_impacts": float(resid.max()),
            "resid_p99": float(np.percentile(resid, 99)),
            "resid_median": float(np.median(resid)),
            "fraction_resid_below_1pct": float(np.mean(resid < 0.01)),
            "impacts_resid_above_1pct": int(np.sum(resid >= 0.01)),
            "mean_restitution": float(np.nanmean([e["restitution"] for e in events])),
            "mean_impulse_Ns": float(np.mean([e["J_contact_Ns"] for e in events])),
            "max_impulse_Ns": float(np.max([e["J_contact_Ns"] for e in events])),
        },
        "ensemble_runtime_s": float(elapsed),
        "run_started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # failure table: every case is reported, including cases above threshold
    for c in cases:
        c["resid_pass_1pct"] = bool(c["max_abs_resid_heave"] < 0.01
                                    and c["max_abs_resid_pitch"] < 0.01)
    (OUT / "ensemble_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps({
        "rao_pass": rao["criterion"]["pass"],
        "rao_worst_dev": rao["worst_point"]["rel_deviation_max"],
        "rao_worst_omega": rao["worst_point"]["omega_rad_s"],
        "resid_max": summary["ensemble_totals"]["resid_max_over_all_impacts"],
        "poisson": [(p["concentration"], p["rel_deviation"], p["pass_3pct"]) for p in poisson],
        "weibull": {k: (v["heave"].get("weibull_plot_r2"), v["pitch"].get("weibull_plot_r2"))
                    for k, v in weib.items() if k.startswith("concentration")},
        "cases": len(cases), "impacts": len(events),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())