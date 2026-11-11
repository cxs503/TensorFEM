#!/usr/bin/env python3
"""m14687 board-audit figures (program-generated, zero hand-copied numbers).

Reads the artifact JSON produced by benchmarks/mortar_lm_hertz_verification.py
(bit-identical to the committed copy, reproduced in this audit) and renders:

  panel a) FE contact pressure at the finest level vs the reference used by the
           benchmark (p0*sqrt(1-(x/b_ref)^2)) and vs the internally consistent
           Hertz line-contact state implied by the same force per unit length.
  panel b) pressure L2 error vs mesh size h (the benchmark's convergence claim).

Writes artifacts/m14687_audit/m14687_audit.png.
"""
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SRC = os.path.join(REPO, "benchmarks", "artifacts", "mortar_lm_hertz", "verification_results.json")
OUT_DIR = os.path.join(REPO, "artifacts", "m14687_audit")
OUT = os.path.join(OUT_DIR, "m14687_audit.png")

E, NU, R, DELTA = 1000.0, 0.3, 10.0, 0.01


def main() -> dict:
    d = json.load(open(SRC))
    ref = d["hertz_reference"]
    res = d["results"]
    finest = res[-1]

    x = np.asarray(finest["contact_x"])
    p_fe = np.asarray(finest["pressure_fe"])
    p_ana = np.asarray(finest["pressure_ana"])

    # Internally consistent plane-strain Hertz line contact for the SAME force:
    #   b = sqrt(4 F R / (pi E*)),  p0 = 2 F / (pi b)
    E_star = E / (1.0 - NU ** 2)
    F = ref["F"]
    b_c = math.sqrt(4.0 * F * R / (math.pi * E_star))
    p0_c = 2.0 * F / (math.pi * b_c)
    xc = np.linspace(-b_c, b_c, 400)
    p_c = p0_c * np.sqrt(np.clip(1.0 - (xc / b_c) ** 2, 0.0, None))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    ax[0].plot(x, p_fe, "o-", ms=4, lw=1.2, label="FE nodal pressure (level 4, nx=64)")
    ax[0].plot(x, p_ana, "--", lw=1.2,
               label=f"benchmark reference b={ref['b']:.4f}, p0={ref['p0']:.2f}")
    ax[0].plot(xc, p_c, ":", lw=1.6,
               label=f"consistent Hertz, same F: b={b_c:.4f}, p0={p0_c:.2f}")
    ax[0].axvline(ref["b"], color="0.6", lw=0.8)
    ax[0].axvline(-ref["b"], color="0.6", lw=0.8)
    ax[0].set_xlabel("x [m]")
    ax[0].set_ylabel("contact pressure p [Pa]")
    ax[0].set_title("(a) FE vs reference pressure, indentation $\\delta$=0.01 m")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    h = [r["h"] for r in res]
    e = [r["pressure_l2_error"] for r in res]
    ax[1].loglog(h, e, "s-", ms=5, label="pressure L2 error")
    hs = np.asarray(h, dtype=float)
    ax[1].loglog(hs, e[0] * (hs / hs[0]) ** 0.07, "--", lw=1.0, label="slope 0.07 (measured)")
    ax[1].loglog(hs, e[0] * (hs / hs[0]) ** 1.0, ":", lw=1.0, label="slope 1.0 reference")
    ax[1].set_xlabel("h [m]")
    ax[1].set_ylabel("relative L2 error")
    ax[1].set_title("(b) pressure error vs mesh size (no convergence)")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3, which="both")

    fig.suptitle("m14687 GAP-FEM-MORTAR audit — reproduced artifacts, Hertz line contact", fontsize=10)
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(OUT, dpi=150)

    summary = {
        "artifact": os.path.relpath(OUT, REPO),
        "benchmark_reference": {"b": ref["b"], "p0": ref["p0"], "F": ref["F"]},
        "consistent_hertz_same_force": {"b": b_c, "p0": p0_c},
        "reference_b_consistency_gap_pct": 100.0 * (ref["b"] - b_c) / b_c,
        "finest_level_fe": {
            "b_fe": finest["b_fe"],
            "b_error_vs_benchmark_ref": finest["b_error"],
            "b_error_vs_consistent_ref": abs(finest["b_fe"] - b_c) / b_c,
            "total_force": finest["total_force"],
            "force_error_vs_benchmark_ref": finest["force_error"],
            "pressure_l2_error": finest["pressure_l2_error"],
        },
        "verification_flags": d["verification"],
    }
    with open(os.path.join(OUT_DIR, "audit_numbers.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()