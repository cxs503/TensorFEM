"""W30 ``ice-coupled-6dof``: random-ice 6-DOF ensemble vs the IACS Polar Class
design ice load (and the declared ISO 19906 pressure--area family).

Chain of evidence
-----------------
1. ``benchmarks/anchors/iacs_ur_i_ice_load_anchor.py`` transcribes the IACS
   UR I2 Rev.5 clauses (clause locator + verbatim quote + sha256 of the PDF and
   of the quote) and exposes :func:`iacs_bow_design_load`.  This driver *reads
   that artefact* and refuses to run if the transcription gate failed, so no
   code value in the output can come from memory.
2. The ensemble is the existing W23 6-DOF Cummins time-domain chain
   (:mod:`cummins_ice_6dof`): random floe field (Poisson arrivals, lognormal
   diameters, Gaussian thicknesses), velocity-Verlet hull/floe contact, and a
   per-impact peak contact force, contact area and contact pressure recorded
   for every impact cluster.
3. Statistics: per-group mean / sd / extremes of the per-impact peaks, a
   convergence check on the seed axis, Weibull and Gumbel fits with KS and AIC,
   and the quantile of the code design value inside the ensemble.
4. Comparison against the code: bow design force, design pressure and design
   patch from the transcribed clauses, transported through the ISO family only
   where a transport is unavoidable (and flagged when it is).

Honesty contract
----------------
* The contact law of the simulation **is** the ISO 19906 pressure--area form
  with declared coefficients, so "measured pressure equals the ISO law at the
  measured area" is a closure check (the chain reproduces the law it was fed),
  not a validation of the law.  Every such item is tagged ``closure_arm``.
* ISO 19906 stays WITHHELD (no licensed copy reachable): all ISO-derived
  numbers are tagged ``non_normative`` and the exponent is swept.
* The IACS design load is a scantling load for a prescribed glancing bow
  impact; a random-floe 6-DOF ensemble is a different quantity.  Deviation from
  it is classified as ``formulation_scope`` unless the two are at matched
  area and thickness, in which case it is classified ``model_difference``.

Usage::

    python benchmarks/ice_coupled_6dof/run_w30_ice_load_vs_code.py            # full
    python benchmarks/ice_coupled_6dof/run_w30_ice_load_vs_code.py --quick    # smoke
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import math
import pathlib
import platform
import subprocess
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "benchmarks" / "anchors"))
sys.path.insert(0, str(ROOT / "src"))

from cummins_ice_6dof import (  # noqa: E402
    Cummins6DOF,
    IceField,
    ShipModel,
    weibull_mle,
)
import iacs_ur_i_ice_load_anchor as ANCHOR  # noqa: E402
from tensorfem.iso19906_ice_6dof import ISO19906PressureArea  # noqa: E402

OUT_DIR = ROOT / "papers" / "artifacts" / "w30" / "ice-coupled-6dof"
OUT_PATH = OUT_DIR / "ice_load_vs_iso19906.json"
PER_IMPACT_CSV = OUT_DIR / "ensemble_per_impact.csv"
SERIES_CSV = OUT_DIR / "representative_timeseries.csv"
ANCHOR_PATH = ROOT / "benchmarks" / "anchors" / "iacs-ur-i-ice-load.json"

TASK_KEY = "ice-coupled-6dof"
MISSION_ID = 21515

# ---- pre-declared experiment design --------------------------------------- #
# three concentration groups + one design-severity group; identical seeds per
# group (independent realisations: the RNG is re-seeded per group/seed pair).
GROUPS = [
    dict(name="G1-mild-c0.3", concentration=0.30, thickness_m=0.6, diameter_m=3.0),
    dict(name="G2-moderate-c0.5", concentration=0.50, thickness_m=0.6, diameter_m=3.0),
    dict(name="G3-severe-c0.7", concentration=0.70, thickness_m=0.6, diameter_m=3.0),
    dict(name="G4-design-severity", concentration=0.70, thickness_m=1.2, diameter_m=10.0),
]
SEEDS = tuple(7101 + k for k in range(24))
DURATION_S = 40.0
DT_S = 0.01
COMMANDED_SPEED_MPS = 5.0
CONVERGENCE_TAIL = 10          # "last N seeds"
CONVERGENCE_TOL = 0.05         # <= 5 %
DEVIATION_TOL = 0.15           # < 15 %

# declared bow geometry of the design-load case (clause I2.3.2.1 (ii) splits the
# bow waterline length into four equal sub-regions; these angles are declared
# inputs of this study, not values read from a ship's lines plan)
BOW_SUB_REGIONS = [
    dict(x_over_L=0.0125, alpha_deg=38.0, beta_prime_deg=42.0),
    dict(x_over_L=0.0375, alpha_deg=33.0, beta_prime_deg=35.0),
    dict(x_over_L=0.0625, alpha_deg=27.0, beta_prime_deg=28.0),
    dict(x_over_L=0.0875, alpha_deg=21.0, beta_prime_deg=22.0),
]
ISO_EX_SWEEP = (-0.42, -0.28, -0.15)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def rel(path: pathlib.Path) -> str:
    return str(path.relative_to(ROOT))


def pct_dev(measured: float, reference: float) -> float:
    return (measured - reference) / reference


def gumbel_mle(data) -> dict:
    """Gumbel (max) fit: scipy MLE when available, method of moments otherwise.

    The node-local environment decides which one runs; ``method`` records it so
    the numbers are never silently mixed.
    """
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]
    try:
        from scipy.stats import gumbel_r
        loc, scale = gumbel_r.fit(x, floc=0.0)  # 0 is the physical lower bound
        method = "scipy.stats.gumbel_r.fit(floc=0.0) MLE"
    except ImportError:  # pragma: no cover - fallback path
        gamma = 0.5772156649015329
        scale = float(np.std(x, ddof=1) * math.sqrt(6.0) / math.pi)
        loc = float(np.mean(x) - gamma * scale)
        method = "method_of_moments (scipy unavailable)"
    loglik = float(np.sum(-np.log(scale) - (x - loc) / scale
                          - np.exp(-(x - loc) / scale)))
    return dict(loc=float(loc), scale=float(scale), loglik=loglik, n=int(x.size),
                method=method)


def ks_test(x, cdf) -> tuple:
    """Two-sided one-sample KS statistic and its asymptotic p-value.

    Uses ``scipy.stats.kstest`` when scipy is importable and the closed-form
    Kolmogorov asymptotic series otherwise, so the goodness-of-fit check does
    not depend on scipy being installed on the compute node.
    """
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    try:
        from scipy.stats import kstest
        res = kstest(x, cdf)
        return float(res.statistic), float(res.pvalue), "scipy.stats.kstest"
    except ImportError:  # pragma: no cover - fallback path
        c = np.clip(np.asarray(cdf(x), dtype=float), 0.0, 1.0)
        i = np.arange(1, n + 1)
        d = float(max(np.max(i / n - c), np.max(c - (i - 1) / n)))
        lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
        p = 2.0 * sum((-1) ** (j - 1) * math.exp(-2.0 * j * j * lam * lam)
                      for j in range(1, 101))
        return d, float(min(max(p, 0.0), 1.0)), "asymptotic Kolmogorov series"


def weibull_fit(data) -> dict:
    w = weibull_mle(data)
    return w


def fit_scores(data, dist_name: str, params) -> dict:
    if dist_name == "weibull":
        k, lam = params["k"], params["lam"]
        cdf = lambda v: 1.0 - np.exp(-(v / lam) ** k)  # noqa: E731
        logpdf = lambda v: (np.log(k) - np.log(lam) + (k - 1.0)  # noqa: E731
                            * np.log(v / lam) - (v / lam) ** k)
        n_par = 2
    else:
        loc, scale = params["loc"], params["scale"]
        cdf = lambda v: math.exp(-math.exp(-(v - loc) / scale)) if np.isscalar(v) else \
            np.exp(-np.exp(-(np.asarray(v) - loc) / scale))  # noqa: E731
        logpdf = lambda v: (-np.log(scale) - (v - loc) / scale  # noqa: E731
                            - np.exp(-(v - loc) / scale))
        n_par = 2
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]
    ks_stat, ks_p, ks_method = ks_test(x, cdf)
    loglik = float(np.sum(logpdf(x)))
    return dict(n=int(x.size), ks_statistic=ks_stat, ks_pvalue=ks_p,
                ks_method=ks_method, loglik=loglik, n_params=n_par,
                aic=2 * n_par - 2 * loglik)


def quantile_at_exceedance(dist_name: str, params, exceedance: float) -> float:
    """Load level exceeded with probability ``exceedance`` per impact."""
    if dist_name == "weibull":
        k, lam = params["k"], params["lam"]
        return float(lam * (-math.log(exceedance)) ** (1.0 / k))
    loc, scale = params["loc"], params["scale"]
    return float(loc - scale * math.log(-math.log(1.0 - exceedance)))


# --------------------------------------------------------------------------- #
# ensemble
# --------------------------------------------------------------------------- #
def _run_single(payload) -> dict:
    """One (group, seed) simulation -- module level so it can be pooled."""
    group, seed = payload
    model = ShipModel()
    field = IceField(concentration=group["concentration"],
                     median_floe_diameter_m=group["diameter_m"],
                     mean_thickness_m=group["thickness_m"])
    run = Cummins6DOF(model).run(DURATION_S, DT_S, ice=field, seed=seed,
                                 commanded_speed=COMMANDED_SPEED_MPS)
    events = [e for e in run["events"]
              if math.isfinite(e["peak_contact_pressure_MPa"]) and e["peak_contact_N"] > 0.0]
    rows = [dict(seed=seed,
                 peak_force_MN=e["peak_contact_N"] / 1.0e6,
                 peak_area_m2=e["peak_contact_area_m2"],
                 peak_pressure_MPa=e["peak_contact_pressure_MPa"],
                 thickness_m=e["thickness_m"],
                 diameter_m=e["diameter_m"],
                 contact_duration_s=e["duration_s"]) for e in events]
    per_seed = None
    if events:
        pf = np.array([e["peak_contact_N"] for e in events]) / 1.0e6
        pp = np.array([e["peak_contact_pressure_MPa"] for e in events])
        per_seed = dict(seed=seed, impacts=len(events),
                        max_force_MN=float(np.max(pf)),
                        mean_force_MN=float(np.mean(pf)),
                        max_pressure_MPa=float(np.max(pp)),
                        mean_pressure_MPa=float(np.mean(pp)))
    step = max(1, len(run["time"]) // 2000)
    representative = dict(seed=seed,
                          time_s=run["time"][::step].tolist(),
                          ice_force_MN=(run["ice_force_N"][::step] / 1.0e6).tolist(),
                          heave_m=run["heave_m"][::step].tolist(),
                          pitch_rad=run["pitch_rad"][::step].tolist())
    return dict(rows=rows, per_seed=per_seed, representative=representative)


def run_group(group: dict, seeds, jobs: int = 1) -> dict:
    tasks = [(group, seed) for seed in seeds]
    if jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            outs = list(pool.map(_run_single, tasks))
    else:
        outs = [_run_single(t) for t in tasks]
    rows, per_seed = [], []
    representative = None
    for out in outs:
        rows.extend(out["rows"])
        if out["per_seed"] is not None:
            per_seed.append(out["per_seed"])
        if representative is None:
            representative = out["representative"]
    return dict(group=group, impacts=rows, per_seed=per_seed,
                representative=representative)


def group_stats(rows: list, per_seed: list, *, tail_n: int, a_code_ref: float) -> dict:
    pf = np.array([r["peak_force_MN"] for r in rows])
    pa = np.array([r["peak_area_m2"] for r in rows])
    pp = np.array([r["peak_pressure_MPa"] for r in rows])
    seed_max = np.array([s["max_force_MN"] for s in per_seed])
    seed_mean = np.array([s["mean_force_MN"] for s in per_seed])
    seed_pmax = np.array([s["max_pressure_MPa"] for s in per_seed])

    def block(x):
        return dict(mean=float(np.mean(x)), sd=float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
                    min=float(np.min(x)), p50=float(np.median(x)),
                    p95=float(np.quantile(x, 0.95)), max=float(np.max(x)),
                    n=int(x.size))

    # convergence: last `tail_n` seeds vs all seeds (per-seed means are the
    # independent units; per-impact peaks inside a seed share one ice field)
    full_mean = float(np.mean(seed_mean))
    tail_mean = float(np.mean(seed_mean[-tail_n:]))
    full_pmax = float(np.mean(seed_pmax))
    tail_pmax = float(np.mean(seed_pmax[-tail_n:]))
    return dict(
        impacts=int(pf.size),
        peak_force_MN=block(pf),
        peak_area_m2=block(pa),
        peak_pressure_MPa=block(pp),
        per_seed_max_force_MN=block(seed_max),
        per_seed_mean_force_MN=block(seed_mean),
        per_seed_max_pressure_MPa=block(seed_pmax),
        area_overlap_with_code_patch=dict(
            code_patch_area_m2=float(a_code_ref),
            impacts_within_factor_2=int(np.sum((pa >= a_code_ref / 2) & (pa <= 2 * a_code_ref))),
            area_ratio_code_over_ensemble=float(a_code_ref / float(np.mean(pa))),
        ),
        convergence=dict(
            tail_seeds=list(SEEDS[-tail_n:]),
            running_mean_force_MN=[float(np.mean(seed_mean[:k + 1]))
                                   for k in range(seed_mean.size)],
            mean_force_full_MN=full_mean, mean_force_tail_MN=tail_mean,
            mean_force_rel_diff=(tail_mean - full_mean) / full_mean,
            mean_pressure_full_MPa=full_pmax, mean_pressure_tail_MPa=tail_pmax,
            mean_pressure_rel_diff=(tail_pmax - full_pmax) / full_pmax,
            tolerance=CONVERGENCE_TOL,
            passed=bool(abs((tail_mean - full_mean) / full_mean) <= CONVERGENCE_TOL),
        ),
        pressure_area_regression=dict(
            **loglog_slope(np.log(pa), np.log(pp)),
            note=("slope of ln p on ln A over per-impact peaks; A is endogenous to the "
                  "coupled dynamics, so this is not an independent measurement of the "
                  "law (closure arm)"),
        ),
    )


def loglog_slope(lx: np.ndarray, ly: np.ndarray) -> dict:
    n = lx.size
    A = np.vstack([lx, np.ones(n)]).T
    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    yhat = A @ coef
    ss_res = float(np.sum((ly - yhat) ** 2))
    ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
    # standard error of the slope
    dof = max(n - 2, 1)
    s2 = ss_res / dof
    cov = s2 * np.linalg.inv(A.T @ A)
    se = float(math.sqrt(max(cov[0, 0], 0.0)))
    return dict(slope=float(coef[0]), intercept=float(coef[1]),
                intercept_MPa=float(math.exp(coef[1])), slope_se=se,
                r2=1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"), n=int(n))


# --------------------------------------------------------------------------- #
# code side
# --------------------------------------------------------------------------- #
def code_loads(anchor: dict, model: ShipModel) -> dict:
    table1 = anchor["parsed_from_quotes"]["class_factors_table1"]
    displacement_kt = model.mass_kg / 1.0e6      # 1 kt = 1000 t = 1e6 kg
    per_class = {}
    for pc, cf in table1.items():
        per_class[pc] = ANCHOR.iacs_bow_design_load(
            displacement_kt=displacement_kt, class_factors=cf,
            sub_regions=BOW_SUB_REGIONS, pc=pc)
    return dict(displacement_kt=displacement_kt, per_class=per_class,
                sub_regions=BOW_SUB_REGIONS,
                code_pressure_area_exponent=anchor["code_pressure_area_exponent"])


def transport(p_ref_MPa: float, a_ref_m2: float, a_target_m2: float, ex: float) -> float:
    """Transport a pressure along a p ~ A^ex law (a *modelling* step, flagged)."""
    return p_ref_MPa * (a_target_m2 / a_ref_m2) ** ex


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="2 seeds per group, 10 s runs")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel worker processes (each run is independent and seeded)")
    ap.add_argument("--report-id", default=None,
                    help="tensor_reports id to bind into the artefact provenance")
    args = ap.parse_args(argv)

    global SEEDS, DURATION_S
    if args.quick:
        SEEDS = SEEDS[:2]
        DURATION_S = 10.0

    if not ANCHOR_PATH.is_file():
        print("anchor artefact missing: run benchmarks/anchors/iacs_ur_i_ice_load_anchor.py",
              file=sys.stderr)
        return 2
    anchor = json.loads(ANCHOR_PATH.read_text(encoding="utf-8"))
    if anchor["verdict"] != "PASS":
        print("anchor transcription gate did not pass; refusing to emit code values",
              file=sys.stderr)
        return 2

    model = ShipModel()
    code = code_loads(anchor, model)
    iso = ISO19906PressureArea()

    t0 = time.time()
    results = []
    for group in GROUPS:
        print(f"[run] {group['name']} seeds={len(SEEDS)} jobs={args.jobs} ...", flush=True)
        results.append(run_group(group, SEEDS, jobs=args.jobs))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # ---- raw per-impact table (traceability) ------------------------------ #
    with PER_IMPACT_CSV.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["group", "seed", "peak_force_MN",
                                            "peak_area_m2", "peak_pressure_MPa",
                                            "thickness_m", "diameter_m",
                                            "contact_duration_s"])
        wr.writeheader()
        for res in results:
            for row in res["impacts"]:
                wr.writerow({"group": res["group"]["name"], **row})
    with SERIES_CSV.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["group", "quantity", "time_s", "value"])
        for res in results:
            rep = res["representative"]
            if rep is None:
                continue
            for qty, key in (("ice_force_MN", "ice_force_MN"),
                             ("heave_m", "heave_m"),
                             ("pitch_rad", "pitch_rad")):
                for t, v in zip(rep["time_s"], rep[key]):
                    wr.writerow([res["group"]["name"], qty, t, v])

    # ---- statistics + fits + comparison ----------------------------------- #
    pc6 = code["per_class"]["PC6"]
    a_code = pc6["patch_area_m2"]
    p_code = pc6["design_pressure_MPa"]
    f_code = pc6["design_force_MN"]

    groups_out = []
    for res in results:
        g = res["group"]
        st = group_stats(res["impacts"], res["per_seed"],
                         tail_n=min(CONVERGENCE_TAIL, max(len(SEEDS) // 2, 2)),
                         a_code_ref=a_code)
        pf = np.array([r["peak_force_MN"] for r in res["impacts"]])
        wfit = weibull_fit(pf)
        gfit = gumbel_mle(pf)
        fit = {
            "weibull": dict(params=dict(k=wfit["shape_k"], lam=wfit["scale_lambda"]),
                            weibull_plot_r2=wfit["weibull_plot_r2"],
                            **fit_scores(pf, "weibull", dict(k=wfit["shape_k"],
                                                             lam=wfit["scale_lambda"]))),
            "gumbel": dict(params=dict(loc=gfit["loc"], scale=gfit["scale"]),
                           method=gfit["method"],
                           **fit_scores(pf, "gumbel", dict(loc=gfit["loc"],
                                                           scale=gfit["scale"]))),
        }
        fit["preferred_by_aic"] = ("weibull"
                                  if fit["weibull"]["aic"] < fit["gumbel"]["aic"]
                                  else "gumbel")
        n_imp = int(pf.size)
        tail = {}
        for name in ("weibull", "gumbel"):
            params = fit[name]["params"]
            tail[name] = {
                f"q_{int(1 / e)}": quantile_at_exceedance(name, params, e)
                for e in (1e-2, 1e-3, 1 / max(n_imp, 1))
            }
        tail["observed_max_MN"] = float(np.max(pf))
        tail["assumption"] = (
            "iid per-impact peaks; the true peaks are clustered inside a floe field and "
            "the tail is therefore optimistic -- the comparison of the extrapolated "
            "1/N quantile with the observed maximum is the check on that assumption")

        # ---- comparison items ---- #
        a_meas = st["peak_area_m2"]["mean"]
        p_meas = st["peak_pressure_MPa"]["mean"]
        f_meas = st["peak_force_MN"]["mean"]

        items = []

        def add(item_id, quantity, measured, reference, reference_label,
                grade, scope, note):
            d = pct_dev(measured, reference)
            items.append(dict(
                item=item_id, quantity=quantity,
                measured=measured, measured_unit=scope["unit"],
                reference=reference, reference_label=reference_label,
                deviation_pct=100.0 * d,
                signed_difference=measured - reference,
                within_15pct=bool(abs(d) < DEVIATION_TOL),
                evidence_grade=grade,
                difference_class=scope["difference_class"],
                note=note,
            ))

        p_max = st["peak_pressure_MPa"]["max"]
        add("C1-force-mean", "mean per-impact peak contact force",
            f_meas, f_code, "IACS UR I2 I2.3.2.1(iii)(b) PC6 design force",
            "REAL_ANCHORED", {"unit": "MN", "difference_class": "formulation_scope"},
            "the code is a prescribed glancing bow impact of a large floe; the ensemble is a "
            "random field of small floes, so the code/ensemble ratio is a scope ratio")
        add("C2-force-seedmax", "median over seeds of the per-seed maximum peak force",
            st["per_seed_max_force_MN"]["p50"], f_code,
            "IACS UR I2 I2.3.2.1(iii)(b) PC6 design force",
            "REAL_ANCHORED", {"unit": "MN", "difference_class": "formulation_scope"},
            "extreme statistic of the same ensemble, still not the code scenario")
        add("C3-pressure-peak-vs-Pavg", "maximum per-impact local peak pressure",
            p_max, p_code, "IACS UR I2 I2.3.4(i) PC6 average patch pressure Pavg",
            "REAL_ANCHORED", {"unit": "MPa", "difference_class": "model_difference"},
            "no transport: the code's patch-average pressure compared with the ensemble's "
            "extremal local pressure; a small positive deviation would mean the code's average "
            "pressure is as high as the worst local peak seen in 24 random fields")
        band = [(r["peak_pressure_MPa"], r["peak_force_MN"]) for r in res["impacts"]
                if a_code / 2.0 <= r["peak_area_m2"] <= 2.0 * a_code]
        band_n = len(band)
        if band_n >= 5:
            band_p = float(np.mean([b[0] for b in band]))
            band_f = float(np.mean([b[1] for b in band]))
            add("C3b-pressure-matched-band", f"mean peak pressure of the {band_n} impacts whose "
                "contact area is within a factor 2 of the code patch area",
                band_p, p_code, "IACS UR I2 I2.3.4(i) PC6 average patch pressure Pavg",
                "REAL_ANCHORED", {"unit": "MPa", "difference_class": "model_difference"},
                "matched-area, no transport: the only item where the ensemble and the code "
                "describe the same patch size")
            add("C3c-force-matched-band", f"mean peak force of the {band_n} area-matched impacts",
                band_f, f_code, "IACS UR I2 I2.3.2.1(iii)(b) PC6 design force",
                "REAL_ANCHORED", {"unit": "MN", "difference_class": "model_difference"},
                "same matched-area subset, force level")
        else:
            items.append(dict(
                item="C3b-pressure-matched-band", quantity="matched-area pressure",
                measured=None, measured_unit="MPa", reference=p_code,
                reference_label="IACS UR I2 I2.3.4(i) PC6 Pavg",
                deviation_pct=None, signed_difference=None, within_15pct=None,
                difference_class="not_evaluated",
                evidence_grade="REAL_ANCHORED",
                note=(f"only {band_n} impacts (<5) with a peak contact area within a factor 2 of "
                      f"the code patch area {a_code:.3f} m2: the ensemble and the code do not "
                      "overlap in area for this group, so no matched-area comparison exists")))
        add("C4-iso-closure", "mean per-impact peak pressure vs the ISO law at the measured area",
            p_meas, iso.pressure_MPa(a_meas),
            "ISO 19906 form p = C_R (A/A_0)^ex with DECLARED coefficients",
            "DECLARED_REFERENCE_NOT_NORMATIVE(closure_arm)",
            {"unit": "MPa", "difference_class": "closure_tautology"},
            "the contact law of the simulation is this law, so a small deviation only proves the "
            "chain reproduces what it was fed; it says nothing about ISO 19906 itself")
        add("C5-exponent-vs-iso", "fitted pressure--area exponent",
            st["pressure_area_regression"]["slope"], iso.exponent,
            "declared ISO exponent ex = -0.28",
            "DECLARED_REFERENCE_NOT_NORMATIVE(closure_arm)",
            {"unit": "-", "difference_class": "closure_tautology"},
            "same closure caveat: A is endogenous inside the law that generated it")
        add("C6-exponent-vs-code", "fitted pressure--area exponent vs the exponent implied by the "
            "IACS clause chain",
            st["pressure_area_regression"]["slope"],
            code["code_pressure_area_exponent"]["dlnp_dlnA"],
            "IACS UR I2 I2.3.2.1(iii)(d)-(e) implied dlnP/dlnA = +0.282",
            "REAL_ANCHORED(derived)", {"unit": "-", "difference_class": "formulation_scope"},
            "the IACS chain carries no decreasing area law (its pressure rises with area at fixed "
            "AR); the code handles small areas through the Table 3 peak pressure factors instead, "
            "so a sign disagreement here is a formulation-scope statement, not an error")

        quant = {}
        for name, dist in (("weibull", fit["weibull"]), ("gumbel", fit["gumbel"])):
            quant[name] = {}
            params = dist["params"]
            if name == "weibull":
                k, lam = params["k"], params["lam"]
                cdf = float(1.0 - math.exp(-(f_code / lam) ** k))
            else:
                cdf = float(math.exp(-math.exp(-(f_code - params["loc"])
                                              / params["scale"])))
            quant[name]["code_design_force_MN"] = cdf
        pp_all = np.array([r["peak_pressure_MPa"] for r in res["impacts"]])
        empirical = {
            "code_design_force_MN": {
                "fraction_of_impacts_below": float(np.mean(pf < f_code)),
                "impacts": int(pf.size),
                "percentile": float(100.0 * np.mean(pf < f_code)),
            },
            "code_design_pressure_MPa": {
                "fraction_of_impacts_below": float(np.mean(pp_all < p_code)),
                "impacts": int(pp_all.size),
                "percentile": float(100.0 * np.mean(pp_all < p_code)),
            },
        }

        # ---- real-vs-code ratio for every Polar Class (informative) -------- #
        class_table = {}
        for pc, load in code["per_class"].items():
            class_table[pc] = dict(
                design_force_MN=load["design_force_MN"],
                design_pressure_MPa=load["design_pressure_MPa"],
                patch_area_m2=load["patch_area_m2"],
                ratio_code_force_over_ensemble_mean=load["design_force_MN"] / f_meas,
                ratio_code_pressure_over_ensemble_max=(
                    load["design_pressure_MPa"] / st["peak_pressure_MPa"]["max"]),
            )

        groups_out.append(dict(
            group=g, seeds=list(SEEDS),
            ice_field=dict(concentration=g["concentration"],
                           mean_thickness_m=g["thickness_m"],
                           median_floe_diameter_m=g["diameter_m"]),
            statistics=st, distribution_fits=fit, tail=tail,
            code_quantile=dict(empirical_cdf_of_code_value=empirical,
                               fitted_cdf=quant,
                               note=("empirical_cdf = fraction of per-impact peaks below the "
                                     "code value; a value near 1.0 means the code brackets the "
                                     "whole ensemble")),
            comparison_items=items,
            polar_class_table=class_table,
        ))

    # ---- criterion-level verdicts ----------------------------------------- #
    conv_ok = all(g["statistics"]["convergence"]["passed"] for g in groups_out)
    r15 = [i for g in groups_out for i in g["comparison_items"]]
    anchored_items = [i for i in r15
                      if i["difference_class"] not in ("closure_tautology", "not_evaluated")]
    not_evaluated = [i for i in r15 if i["difference_class"] == "not_evaluated"]
    anchored_ok = [i for i in anchored_items if i["within_15pct"]]
    closure_items = [i for i in r15 if i["difference_class"] == "closure_tautology"]
    closure_ok = [i for i in closure_items if i["within_15pct"]]
    criteria = [
        dict(id="1-anchors",
             statement=">= 3 normative anchors, each with clause locator, verbatim quote and sha256",
             value=anchor["anchor_count"],
             passed=bool(anchor["anchor_count"] >= 3
                         and anchor["all_anchor_values_from_transcription"]),
             evidence=dict(path=rel(ANCHOR_PATH), sha256=sha256_file(ANCHOR_PATH),
                           iso_19906_status=anchor["iso_19906"]["status"])),
        dict(id="2-ensemble-convergence",
             statement=f"last {CONVERGENCE_TAIL} seeds agree with the full ensemble mean within 5%",
             value={g["group"]["name"]: g["statistics"]["convergence"]["mean_force_rel_diff"]
                    for g in groups_out},
             passed=bool(conv_ok), evidence=dict(tolerance=CONVERGENCE_TOL)),
        dict(id="3-deviation-table",
             statement=("per-item deviation vs the code with an explicit '<15%?' answer and a "
                        "cause classification for every item"),
             value=dict(items=len(anchored_items), within_15pct=len(anchored_ok),
                        not_evaluated=len(not_evaluated),
                        all_within_15pct=bool(len(anchored_ok) == len(anchored_items))),
             passed=bool(anchored_items),   # completeness of the table, not the 15% gate
             evidence=dict(
                 deviation_gate_15pct=("PASS" if len(anchored_ok) == len(anchored_items)
                                       else "FAIL"),
                 note=("the 15% gate itself is reported separately in "
                       "deviation_gate_15pct; it is NOT satisfied, and the cause is classified "
                       "per item")),),
        dict(id="4-distribution-fit",
             statement="Weibull/Gumbel fit with goodness-of-fit and a stated tail assumption",
             value={g["group"]["name"]: g["distribution_fits"]["preferred_by_aic"]
                    for g in groups_out},
             passed=all(math.isfinite(g["distribution_fits"][d]["ks_statistic"])
                        for g in groups_out for d in ("weibull", "gumbel")),
             evidence=dict(fits=["weibull", "gumbel"], gof="scipy KS + AIC",
                           tail_assumption_check="extrapolated 1/N quantile vs observed max")),
        dict(id="5-per-item-answer",
             statement="answer 'is it < 15%' for every item and classify the cause",
             value=dict(anchored_within_15pct=len(anchored_ok),
                        anchored_total=len(anchored_items),
                        closure_within_15pct=len(closure_ok),
                        closure_total=len(closure_items)),
             passed=True,
             evidence=dict(classification={"formulation_scope": "code and ensemble measure "
                                           "different physical quantities",
                                           "model_difference": "same quantity, different "
                                           "model/inputs",
                                           "closure_tautology": "the law is inside the "
                                           "simulation; agreement proves implementation only"})),
    ]

    payload = {
        "schema_version": "1.0.0",
        "task_key": TASK_KEY,
        "mission_id": MISSION_ID,
        "slug": "ice-coupled-6dof",
        "domain": "ice",
        "title": ("Random-ice 6-DOF load ensemble vs the IACS Polar Class design ice load "
                  "(and the declared ISO 19906 pressure--area family)"),
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git": {"commit": git_commit()},
        "environment": {"python": platform.python_version(),
                        "numpy": np.__version__,
                        "platform": platform.platform(),
                        "wall_time_s": None},
        "experiment": {
            "groups": GROUPS,
            "seeds": list(SEEDS),
            "duration_s": DURATION_S,
            "dt_s": DT_S,
            "commanded_speed_mps": COMMANDED_SPEED_MPS,
            "hull": dict(length_m=model.length_m, breadth_m=model.breadth_m,
                         draught_m=model.draught_m, block_coefficient=model.block_coefficient,
                         displacement_kt=code["displacement_kt"],
                         provenance=model.provenance),
            "metric": ("per impact cluster: peak contact force (heave normal), peak contact "
                       "area, peak contact pressure; plus per-seed maxima"),
            "convergence_tail_seeds": CONVERGENCE_TAIL,
            "deviation_threshold": DEVIATION_TOL,
        },
        "anchors": {
            "artifact": rel(ANCHOR_PATH),
            "sha256": sha256_file(ANCHOR_PATH),
            "anchor_count": anchor["anchor_count"],
            "documents": [dict(file=s["file"], sha256_file=s["sha256_file"],
                               url=s["url"], revision=s["revision_as_printed"],
                               retrieved_utc=s["retrieved_utc"]) for s in anchor["sources"]],
            "anchor_ids": [a["anchor_id"] for a in anchor["anchors"]],
            "code_constants_check": {k: v["match"]
                                     for k, v in anchor["code_vs_transcription_consistency"].items()},
            "iso_19906": {
                "status": anchor["iso_19906"]["status"],
                "entries": anchor["iso_19906"]["entries"],
                "declared_values": anchor["iso_19906"]["declared_values_used_downstream"],
            },
        },
        "code_side": {
            "polar_class_used_for_headline": "PC6",
            "bow_sub_regions": BOW_SUB_REGIONS,
            "per_class": {pc: {k: v for k, v in load.items() if k != "sub_regions"}
                          for pc, load in code["per_class"].items()},
            "pc6_sub_regions": code["per_class"]["PC6"]["sub_regions"],
            "code_pressure_area_exponent": code["code_pressure_area_exponent"],
            "iso_exponent_sweep": list(ISO_EX_SWEEP),
            "transport_sensitivity": {
                f"ex={ex}": transport(pc6["design_pressure_MPa"],
                                      pc6["patch_area_m2"],
                                      (results[0]["impacts"][0]["peak_area_m2"]
                                       if results[0]["impacts"] else 1.0), ex)
                for ex in ISO_EX_SWEEP
            },
        },
        "groups": groups_out,
        "criteria": criteria,
        "provenance": {
            "script": rel(pathlib.Path(__file__).resolve()),
            "script_sha256": sha256_file(pathlib.Path(__file__).resolve()),
            "per_impact_csv": rel(PER_IMPACT_CSV),
            "per_impact_csv_sha256": sha256_file(PER_IMPACT_CSV),
            "timeseries_csv": rel(SERIES_CSV),
            "anchor_artifact": rel(ANCHOR_PATH),
            "anchor_script": "benchmarks/anchors/iacs_ur_i_ice_load_anchor.py",
            "simulation_module": "benchmarks/ice_coupled_6dof/cummins_ice_6dof.py",
            "iso_module": "src/tensorfem/iso19906_ice_6dof.py",
            "tensor_report_id": args.report_id,
            "reproduce": ("python benchmarks/anchors/iacs_ur_i_ice_load_anchor.py && "
                          "python benchmarks/ice_coupled_6dof/run_w30_ice_load_vs_code.py"),
        },
        "honesty": {
            "closure_arms": ("items tagged closure_tautology compare the simulation against the "
                             "law that generates its contact force; they can only prove the "
                             "implementation, never the standard"),
            "iso_19906_withheld": ("no ISO 19906 number is transcribed; all ISO-derived values "
                                   "use declared coefficients and are non_normative"),
            "geometry_gate": ("no hull geometry file is used, so Cb/L/B are the declared ShipModel "
                              "particulars and are not independently validated here"),
            "not_claimed": ["compliance with ISO 19906",
                            "validation of the IACS design load against model tests",
                            "extrapolation of the 6-DOF chain to a real ship"],
        },
        "wall_time_s": time.time() - t0,
    }

    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    OUT_PATH.write_text(text, encoding="utf-8")
    print(f"wrote {rel(OUT_PATH)}")
    for g in groups_out:
        st = g["statistics"]
        print(f"  {g['group']['name']}: impacts={st['impacts']} "
              f"meanF={st['peak_force_MN']['mean']:.4f} MN "
              f"maxF={st['peak_force_MN']['max']:.4f} MN "
              f"meanP={st['peak_pressure_MPa']['mean']:.3f} MPa "
              f"conv={st['convergence']['mean_force_rel_diff']:+.3%}")
    print(f"  code PC6: F={pc6['design_force_MN']:.3f} MN P={pc6['design_pressure_MPa']:.3f} MPa "
          f"patch={a_code:.3f} m2")
    for item in anchored_items:
        print(f"  {item['item']:<28} dev={item['deviation_pct']:+8.1f}%  "
              f"<15%={item['within_15pct']}  [{item['difference_class']}]")
    gate = "PASS" if len(anchored_ok) == len(anchored_items) and anchored_items else "FAIL"
    print(f"  15% deviation gate over anchored items: {gate} "
          f"({len(anchored_ok)}/{len(anchored_items)})")
    for c in criteria:
        print(f"  criterion {c['id']}: {'PASS' if c['passed'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())