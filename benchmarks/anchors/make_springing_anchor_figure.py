"""Figure for the springing-whipping-hydroelastic anchor experiment.

Reads the generated artifact (no numbers are typed here) and renders the one
quantitative statement this experiment can legitimately make: *the same
nondimensional abscissa maps onto frequencies that differ by L/delta* when the
nondimensionalisation convention is unresolved*. That is the reason the anchor
transfer is WITHHELD rather than performed.

    python benchmarks/anchors/make_springing_anchor_figure.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ANCHOR_DIR = Path(__file__).resolve().parent
REPO_ROOT = ANCHOR_DIR.parents[1]
ARTIFACT = ANCHOR_DIR / "springing-whipping-hydroelastic.json"
OUT = REPO_ROOT / "artifacts" / "springing_whipping_anchor_figure.png"


def fixture_spectrum(x):
    """Same synthetic shape as the generator; A is read from the artifact."""
    return x ** 2 * np.exp(-x ** 2)


def main() -> int:
    art = json.loads(ARTIFACT.read_text())
    cal = art["case_caliber"]["quantities"]
    rho, v, L = cal["rho"]["value"], cal["V"]["value"], cal["L"]["value"]
    ds = cal["delta_star"]["value"]
    A = art["dry_run_fixture"]["amplitude_A"]
    q = 0.5 * rho * v ** 2

    ws = np.logspace(-1.5, 1.2, 800)
    s_star = A * fixture_spectrum(ws)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # --- panel 1: the same nondimensional ordinate on two frequency axes ---
    ax = axes[0]
    for l_ref, label, style in ((L, f"convention A: l_ref = L = {L:.0f} m", "-"),
                                (ds, f"convention B: l_ref = delta* = {ds:g} m", "--")):
        f = ws * v / (2 * math.pi * l_ref)
        sp = q ** 2 * (l_ref / v) * s_star * 2 * math.pi
        ax.loglog(f, sp, style, lw=2, label=label)
    ax.set_xlabel("frequency f [Hz]")
    ax.set_ylabel(r"$S_p(f)$ [Pa$^2$/Hz]  (synthetic fixture)")
    ax.set_title("Same nondimensional spectrum, two unresolved conventions")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    ax.annotate("shift = L/delta* = "
                f"{L/ds:.3g}\nin frequency", xy=(0.03, 0.06),
                xycoords="axes fraction", fontsize=9,
                bbox=dict(boxstyle="round", fc="w", ec="0.6"))

    # --- panel 2: compressive-load path not applicable; show sensitivity rows ---
    ax = axes[1]
    rows = [c for c in art["machine_checks"] if c["id"].startswith("CHK7")][0]["rows"]
    labels = [f"delta*={r['delta_star_m']:g} m" for r in rows]
    f_a = [r["f_at_omega_star_min_if_l_ref_is_L_hz"] for r in rows]
    f_b = [r["f_at_omega_star_min_if_l_ref_is_delta_star_hz"] for r in rows]
    x = np.arange(len(rows))
    ax.bar(x - 0.2, f_a, 0.38, label="l_ref = L (hull length)")
    ax.bar(x + 0.2, f_b, 0.38, label="l_ref = delta* (boundary layer)")
    ax.set_yscale("log")
    ax.set_xticks(x, labels)
    ax.set_ylabel(r"f at $\omega^*=1$ [Hz]")
    ax.set_title("Frequency calibration is convention-dominated")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=9)

    fig.suptitle("springing-whipping-hydroelastic: anchor transfer blocked at the "
                 "transcription/convention gate (honest FAIL, 0 claim-eligible anchors)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"figure: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())