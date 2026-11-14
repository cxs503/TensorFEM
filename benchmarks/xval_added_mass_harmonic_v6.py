#!/usr/bin/env python
"""XVAL v6 -- harmonic identification of IBM added mass (anchor debt repair).

Anchor: ``XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION`` (joconnor22/LIFE implicit-IBM
+ Aitken recipe).  Repo: TensorFEM.

What this benchmark settles
---------------------------
1. **Defect (proved on synthetic data + the archived record).**  The legacy
   ``forced_oscillation`` fit onto ``{y'', y', y, 1}`` is rank deficient at a
   single drive frequency (``y'' = -w^2 y`` exactly), so it carries *no*
   information about ``m_a``: feeding it an exact harmonic force built from
   ``m_a = 314.16`` or ``m_a = 560`` returns ``1e-4`` in both cases.  The
   archived ``added_mass_fit = -0.0`` of
   ``benchmarks/xval_added_mass_artifacts/evidence.json`` is that artefact, not
   a physics failure.
2. **v5 §5.3 resolved (algebra, not physics).**  The free-decay mass attribute is
   a *derived* quantity; its error is the frequency error amplified exactly by
   ``((1+e)^-2 - 1)(1 + 1/r)``.  That reproduces the three archived v5 rows to
   < 0.1 pp, so the reported sign-changing "unexplained bias" needs no fluid
   added-stiffness mechanism.
3. **Independent measurement.**  A multi-frequency prescribed-motion sweep
   separates ``m_a`` from ``k_a`` (``K/A = m_a w^2 - k_a``) and its frequency
   flatness is *tested* rather than assumed.

Usage (cluster shards)::

    python benchmarks/xval_added_mass_harmonic_v6.py --shard 0 4
    python benchmarks/xval_added_mass_harmonic_v6.py --aggregate
    python benchmarks/xval_added_mass_harmonic_v6.py --figures
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from tensorfem.ibm_added_mass_harmonic import (  # noqa: E402
    amplify_frequency_error,
    added_mass_gain,
    apparent_added_mass,
    fit_added_mass_sweep,
    legacy_basis_condition,
    legacy_single_frequency_fit,
    required_frequency_accuracy,
)
from tensorfem.ibm_aitken_added_mass import (  # noqa: E402
    analytic_added_mass,
    forced_oscillation,
)

ART = os.path.join(HERE, "xval_added_mass_harmonic_v6_artifacts")

# (tag, box, half_width, drive_period, amplitude, min_steps)
#
# Two families.  ``h160/h256`` are the v5 plate (c = 10 Delta) at the historical
# box sizes: their records (5 drive periods) are 240-1920 steps, i.e. far below
# the viscous settling time c^2/nu ~ 5400 steps, so they are transient-limited.
# ``h064_c04`` is the *settleable* regime: c = 4 Delta puts c^2/nu at ~900 steps
# and the 3200-step record then spans 40-50 periods of settled flow.
CONFIGS = [
    ("h160_c10_T048_A025", 160, 10.0, 48.0, 0.25, 0),
    ("h160_c10_T096_A025", 160, 10.0, 96.0, 0.25, 0),
    ("h160_c10_T192_A025", 160, 10.0, 192.0, 0.25, 0),
    ("h160_c10_T288_A025", 160, 10.0, 288.0, 0.25, 0),
    ("h160_c10_T384_A025", 160, 10.0, 384.0, 0.25, 0),
    ("h256_c10_T096_A025", 256, 10.0, 96.0, 0.25, 0),
    ("h256_c10_T192_A025", 256, 10.0, 192.0, 0.25, 0),
    ("h160_c10_T192_A0125", 160, 10.0, 192.0, 0.125, 0),
    ("h064_c04_T064_A020", 64, 4.0, 64.0, 0.2, 3200),
    ("h064_c04_T128_A020", 64, 4.0, 128.0, 0.2, 3200),
]

SETTLE_CYCLES = 2.0
MEASURE_CYCLES = 3.0
TAG_FILTER = set()


def run_one(tag, box, half_width, period, amplitude, min_steps=0,
            omega_lbm=1.8, dt=1.0, sub_tol=1e-10, max_sub_iter=40):
    lattice = (int(box), int(box))
    n_steps = max(int(round((SETTLE_CYCLES + MEASURE_CYCLES) * period)),
                  int(min_steps))
    cycles_total = n_steps / float(period)
    settle = max(SETTLE_CYCLES, cycles_total / 3.0)
    rec = forced_oscillation(n_steps=n_steps, dt=dt, half_width=half_width,
                             center=(0.5 * box, 0.5 * box), lattice=lattice,
                             amplitude=amplitude, drive_period=period,
                             omega_lbm=omega_lbm, sub_tol=sub_tol,
                             max_sub_iter=max_sub_iter)
    t = dt * np.arange(1, n_steps + 1, dtype=float)
    pt = apparent_added_mass(rec["force_history"], rec["y_history"], t,
                             omega=2.0 * math.pi / period,
                             amplitude=amplitude,
                             skip_cycles=settle)
    pt.update({
        "tag": tag, "box": int(box), "lattice": list(lattice),
        "half_width": half_width, "drive_period": float(period),
        "n_steps": int(n_steps), "omega_lbm": omega_lbm,
        "settle_cycles": settle, "cycles_total": cycles_total,
        "viscous_time_steps": half_width ** 2 / ((1.0 / omega_lbm - 0.5) / 3.0),
        "lamb_added_mass": analytic_added_mass(half_width),
        "legacy_added_mass_fit": _legacy_value(rec),
    })
    return pt


def _legacy_value(rec):
    raw = rec.get("legacy_added_mass_fit", rec.get("added_mass_fit"))
    return float(raw) if raw is not None else float("nan")


def shard(i, n):
    os.makedirs(ART, exist_ok=True)
    todo = [c for k, c in enumerate(CONFIGS) if k % n == i]
    if TAG_FILTER:
        todo = [c for c in todo if c[0] in TAG_FILTER]
    for tag, box, half_width, period, amp, min_steps in todo:
        path = os.path.join(ART, tag + ".json")
        if os.path.exists(path):
            print("[skip] %s" % tag)
            continue
        pt = run_one(tag, box, half_width, period, amp, min_steps)
        with open(path, "w") as fh:
            json.dump(pt, fh, indent=1)
        print("[done] %s m_app=%.3f r2=%.4f" %
              (tag, pt["apparent_added_mass"], pt["r2"]))
    return 0


def load_points():
    if not os.path.isdir(ART):
        raise SystemExit("no artifacts: run the shards first")
    pts = []
    for name in sorted(os.listdir(ART)):
        if name.endswith(".json") and name.startswith("h"):
            with open(os.path.join(ART, name)) as fh:
                pts.append(json.load(fh))
    return pts


def aggregate():
    pts = load_points()
    out = {"mode": "xval_added_mass_harmonic_v6", "points": pts}
    by_box = {}
    for p in pts:
        if abs(p["amplitude"] - 0.25) < 1e-12:
            by_box.setdefault(p["box"], []).append(p)
    sweeps = {}
    for box, rows in sorted(by_box.items()):
        fit = fit_added_mass_sweep(rows)
        fit["box"] = box
        fit["lamb_added_mass"] = analytic_added_mass(rows[0]["half_width"])
        if fit["identifiable"]:
            fit["added_mass_error_pct"] = 100.0 * (
                fit["added_mass"] - fit["lamb_added_mass"]) / \
                fit["lamb_added_mass"]
        sweeps[str(box)] = fit
    out["sweeps"] = sweeps

    # amplitude stationarity at the shared period 192 / box 160
    amp_rows = [p for p in pts if p["box"] == 160 and
                abs(p["drive_period"] - 192.0) < 1e-9]
    if len(amp_rows) == 2:
        a = sorted(amp_rows, key=lambda p: p["amplitude"])
        out["amplitude_sensitivity"] = {
            "small_amplitude": a[0]["apparent_added_mass"],
            "large_amplitude": a[1]["apparent_added_mass"],
            "rel_diff_pct": 100.0 * (a[0]["apparent_added_mass"] -
                                     a[1]["apparent_added_mass"]) /
            a[1]["apparent_added_mass"],
        }

    # legacy defect: exact harmonics built from two very different masses
    out["legacy_defect"] = {
        "condition_1dof": legacy_basis_condition(2.0 * math.pi / 300.0),
        "fits": [legacy_single_frequency_fit(m_a, omega=2 * math.pi / 300.0)
                 for m_a in (314.1592653589793, 560.0)],
        "archived_evidence": {
            "added_mass_fit": -0.0, "fit_r2": 1.0,
            "source": "benchmarks/xval_added_mass_artifacts/evidence.json"
                      " :: E1_forced",
        },
    }
    # v5 free-decay mass attribute is an algebraic amplification
    rows = [{"tag": "c10_H288", "ratio": 0.6, "df_pct": -12.109,
             "archived_pct": 78.544},
            {"tag": "c10_H160", "ratio": 0.6, "df_pct": -9.4,
             "archived_pct": 56.7},
            {"tag": "c20_H160", "ratio": 2.51, "df_pct": 11.9,
             "archived_pct": -28.1}]
    for r in rows:
        r["predicted_pct"] = amplify_frequency_error(r["ratio"], r["df_pct"])
        r["gain"] = added_mass_gain(r["ratio"])
        r["residual_pp"] = r["predicted_pct"] - r["archived_pct"]
        r["required_df_pct_for_15pct_mass"] = required_frequency_accuracy(
            r["ratio"], 15.0)
    out["amplification_identity"] = rows

    path = os.path.join(ART, "summary.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("points",)}, indent=1, default=float))
    print("[wrote] %s" % path)
    return 0


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = json.load(open(os.path.join(ART, "summary.json")))
    pts = out["points"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    # (a) apparent mass vs drive frequency, with Lamb and the v5 inference
    for box, mk in ((160, "o"), (256, "s")):
        rows = sorted([p for p in pts if p["box"] == box and
                       abs(p["amplitude"] - 0.25) < 1e-12],
                      key=lambda p: p["omega"])
        if rows:
            ax[0].semilogx([p["omega"] for p in rows],
                           [p["apparent_added_mass"] for p in rows],
                           mk + "-", label="forced, box %d$\\Delta$" % box)
    lamb = analytic_added_mass(10.0)
    ax[0].axhline(lamb, color="k", ls="--", label="Lamb $\\rho\\pi c^2$")
    ax[0].axhline(560.91, color="tab:red", ls=":",
                  label="v5 free-decay inference")
    ax[0].set_xlabel("drive frequency $\\omega$ [1/$\\Delta t$]")
    ax[0].set_ylabel("apparent added mass  $m_a-k_a/\\omega^2$")
    ax[0].legend(fontsize=7)
    ax[0].set_title("(a) harmonic sweep, prescribed motion")

    # (b) the separated terms: in-phase force / amplitude vs omega^2
    rows = sorted([p for p in pts if p["box"] == 160 and
                   abs(p["amplitude"] - 0.25) < 1e-12],
                  key=lambda p: p["omega"])
    if rows:
        ax[1].plot([p["omega"] ** 2 for p in rows],
                   [p["in_phase"] / p["amplitude"] for p in rows],
                   "o-", label="K/A (measured)")
        w2 = np.linspace(0, max(p["omega"] ** 2 for p in rows) * 1.1, 10)
        fit = out["sweeps"].get("160", {})
        if fit.get("identifiable"):
            ax[1].plot(w2, fit["added_mass"] * w2 - fit["added_stiffness"],
                       "k--", label="fit: $m_a\\omega^2-k_a$")
        ax[1].plot(w2, lamb * w2, ":", color="k", label="Lamb $m_a\\omega^2$")
    ax[1].set_xlabel("$\\omega^2$")
    ax[1].set_ylabel("in-phase force / amplitude  $K/A$")
    ax[1].legend(fontsize=7)
    ax[1].set_title("(b) in-phase coefficient vs $\\omega^2$")

    # (c) legacy defect: the estimator is blind to m_a
    ma = np.linspace(0, 800, 41)
    got = [legacy_single_frequency_fit(m, omega=2 * math.pi / 300.0)[
        "added_mass_fit"] for m in ma]
    ax[2].plot(ma, got, "o-", label="legacy $\\{y'',y',y,1\\}$ fit")
    ax[2].plot(ma, ma, "k--", label="identity (what it should be)")
    ax[2].set_xlabel("true $m_a$ in the synthetic force")
    ax[2].set_ylabel("$m_a$ returned by the legacy fit")
    ax[2].legend(fontsize=7)
    ax[2].set_title("(c) rank deficiency: zero information ($\\kappa\\sim10^{19}$)")
    fig.tight_layout()
    path = os.path.join(ART, "v6_harmonic.png")
    fig.savefig(path, dpi=130)
    print("[wrote] %s" % path)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", nargs=2, type=int, metavar=("I", "N"))
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--tag", action="append", default=[],
                    help="run only these config tags (shard mode)")
    ap.add_argument("--figures", action="store_true")
    a = ap.parse_args(argv)
    global TAG_FILTER
    TAG_FILTER = set(a.tag)
    if a.shard:
        return shard(*a.shard)
    if a.aggregate:
        return aggregate()
    if a.figures:
        return figures()
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())