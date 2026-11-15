#!/usr/bin/env python3
"""Artifacts for the v13 sub-iteration-gain round (local, cheap stages only).

The full-length ``relax``/``physics`` stages of :mod:`xval_ibm_subiter_gain`
need a lattice large enough to resolve a period and run ~1120 coupled steps, so
they are submitted to the compute gate (job ``cj3827``).  This script produces
the three artifacts that are cheap enough to regenerate anywhere:

* ``summary.json`` -- the complete 14-configuration gain sweep (production
  driver, residual traces at omega = 1, 1/2, 1/4) plus the short-record
  relaxation comparison;
* ``gain.png`` -- measured gain vs the added-mass prediction and the
  frozen-fluid discrete bound;
* ``traces.png`` -- residual traces of the production driver.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

spec = importlib.util.spec_from_file_location(
    "xval_gain", os.path.join(HERE, "xval_ibm_subiter_gain.py"))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

ART = bench.ART
SHORT = dict(lattice=(48, 36), n_elements=8, n_steps=24)


def short_relax():
    out = {}
    for tag, mode, w in [("aitken", "aitken", 1.0), ("fixed1", "fixed", 1.0),
                         ("fixed005", "fixed", 0.05)]:
        rec = bench.measure_physics(fluid_rho=1.0, rho_s=1.0,
                                    relax_mode=mode, fixed_omega=w, **SHORT)
        out[tag] = {k: v for k, v in rec.items() if k != "tips"}
        out[tag]["tips"] = rec["tips"]
    base = np.asarray(out["aitken"]["tips"], dtype=float)
    for tag in ("fixed1", "fixed005"):
        other = np.asarray(out[tag]["tips"], dtype=float)
        n = min(len(base), len(other))
        dev = float(np.max(np.abs(base[:n] - other[:n])))
        out[tag]["dev_vs_aitken"] = dev
        out[tag]["rel_dev_vs_aitken"] = dev / max(float(np.max(np.abs(base[:n]))), 1e-300)
    return out


def figure(rows, relax):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
    ax = axes[0]
    pred = [r["mu_added_mass"] - 1.0 for r in rows]
    ax.scatter(pred, [max(r["mu_minus_1"], 1e-17) for r in rows], s=46,
               label="shipped beam driver (measured)")
    ax.scatter(pred, [max(r["discrete_bound"], 1e-17) for r in rows], marker="^",
               s=40, label="frozen-fluid bound, eq. (3)")
    ax.axhline(bench.INERT_TOL, color="k", ls="--", lw=1,
               label=r"inert tolerance $10^{-6}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"added-mass prediction $\mu-1=m_a/m_s$ (Lamb plate)")
    ax.set_ylabel(r"measured $\mu-1$")
    ax.set_title("interface gain: measured vs added mass")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    for r in rows[:3]:
        ax.semilogy(range(len(r["traces"][0])), r["traces"][0], "o-",
                    label=rf"$\omega=1$, $\rho_f$={r['rho_f']}, $\rho_s$={r['rho_s']}")
    for w, tr in zip(rows[0]["omegas"][1:], rows[0]["traces"][1:]):
        ax.semilogy(range(len(tr)), tr, "s--", label=rf"$\rho_f$={rows[0]['rho_f']}, $\omega$={w}")
    ax.set_xlabel("sub-iteration")
    ax.set_ylabel("interface residual (dimensionless)")
    ax.set_title(r"residual traces: ratio $=1-\omega$ exactly")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(ART, "xval_ibm_subiter_gain.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


def main():
    os.makedirs(ART, exist_ok=True)
    rows = bench.stage_gain()
    relax = short_relax()
    fig_path = figure(rows, relax)
    summ = {
        "gain": rows,
        "relax_short_record": {k: {kk: vv for kk, vv in v.items() if kk != "tips"}
                              for k, v in relax.items()},
        "relax_short_config": SHORT,
        "figure": fig_path,
        "note": ("full-length relax/physics stages submitted as compute job cj3827; "
                 "this summary holds the 14-config gain sweep and the 24-step "
                 "short-record relaxation comparison"),
    }
    with open(os.path.join(ART, "summary.json"), "w") as fh:
        json.dump(summ, fh, indent=2, default=float)
    print("wrote", os.path.join(ART, "summary.json"))
    for r in rows:
        print(f"rho_f={r['rho_f']} rho_s={r['rho_s']} mu={r['mu_measured']:.6f} "
              f"mu-1={r['mu_minus_1']:.2e} ratios={['%.6f' % x for x in r['ratios']]}")
    for tag in ("aitken", "fixed1", "fixed005"):
        r = relax[tag]
        print(tag, "sub_it", r["sub_iterations"][:4], "maxit", r["maxit_hits"],
              "rel_dev", r.get("rel_dev_vs_aitken"))
    print("figure", fig_path)


if __name__ == "__main__":
    main()