"""XVAL screening harness: phase-field AT2 vs pseudopotential LBM (NOT_APPLICABLE).

Anchor: ``XVAL-LBM-NOT-APPLICABLE-PHASE-FIELD-NOT-PSEUDOPOT``.

Produces the evidence bundle for the applicability verdict and the substitute
verification path (the 1-D AT2 profile against its closed-form solution).

Usage::

    PYTHONPATH=src python examples/xval_not_applicable_phase_field.py --out artifacts
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tensorfem.xval_applicability import (
    ACCEPTANCE_REL_TOL,
    not_applicable_verdict,
    solve_at2_1d_profile,
)


def _markdown_rows(verdict: dict) -> list[str]:
    run = verdict["at2_reference_run"]
    rows = [
        "| AT2 1-D profile, L2 relative error "
        f"(n={run['n_elem']}, h/l={run['h_over_l']:.4g}) | "
        f"{run['l2_relative_error']:.6e} | < {ACCEPTANCE_REL_TOL:.0%} |",
        f"| AT2 1-D profile, nodal max relative error | "
        f"{run['nodal_max_relative_error']:.6e} | < {ACCEPTANCE_REL_TOL:.0%} |",
        f"| AT2 convergence order (L2 vs h) | {verdict['at2_observed_l2_order']:.3f} | ~2 |",
    ]
    for hh in verdict["at2_homogeneous_anchor"]:
        rows.append(
            f"| AT2 homogeneous saturation d=2Hl/(Gc+2Hl), H={hh['H']:g} | "
            f"{hh['d_numeric']:.12g} vs {hh['d_reference']:.12g} "
            f"(err {hh['relative_error']:.2e}) | < {ACCEPTANCE_REL_TOL:.0%} |"
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--l", type=float, default=0.05)
    ap.add_argument("--length", type=float, default=1.0)
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    verdict = not_applicable_verdict(ell=args.l, length=args.length)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = out / "xval_not_applicable_phase_field.json"
    serialisable = {k: v for k, v in verdict.items() if k != "at2_reference_run"}
    serialisable["at2_reference_run"] = verdict["at2_reference_run"]
    payload.write_text(json.dumps(serialisable, indent=2), encoding="utf-8")

    print("== screen ==")
    print(f"pairing: {verdict['screen']['left']} vs {verdict['screen']['right']}")
    print(f"verdict: {verdict['screen']['verdict']}  "
          f"blocking failures: {', '.join(verdict['screen']['blocking_failures'])}")
    for p in verdict["screen"]["predicates"]:
        print(f"  [{ 'PASS' if p['passed'] else 'FAIL' }] {p['code']} {p['question']}")
        print(f"         {p['evidence']}")

    print("\n== substitute verification path: AT2 closed-form anchors ==")
    for row in _markdown_rows(verdict):
        print(row)

    print("\n== ladder ==")
    for r in verdict["at2_convergence_ladder"]:
        print(f"  h/l={r['h_over_l']:<7.4g} L2={r['l2_relative_error']:.3e} "
              f"nodal={r['nodal_max_relative_error']:.3e} iters={r['newton_iterations']}")

    print("\n== width control (x(d=1/e)/l) ==")
    for w in verdict["width_scaling"]:
        print(f"  l={w['ell']:.4g} l/L={w['l_over_L']:.3f} -> {w['width_ratio']:.6f}")

    print("\n== acceptance ==")
    for k, v in verdict["acceptance"].items():
        print(f"  {k}: {v}")

    if not args.no_figure:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        run = solve_at2_1d_profile(args.l, args.length, 160)
        fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.3))

        ax = axes[0]
        ax.plot(run["x"], run["d_reference"], "-", color="#1f4e79", lw=2.0,
                label=r"analytic $\cosh((L-x)/l)/\cosh(L/l)$")
        ax.plot(run["x"], run["d_numeric"], "o", ms=3.0, mfc="none", mec="#c00000",
                label=f"TensorFEM AT2 (n={run['n_elem']}, h/l={run['h_over_l']:.3g})")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("damage d [-]")
        ax.set_yscale("log")
        ax.set_title("1-D AT2 equilibrium profile")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
        ax.text(0.35, 0.75,
                f"L2 rel. err = {run['l2_relative_error']:.2e}\n"
                f"nodal max = {run['nodal_max_relative_error']:.2e}",
                transform=ax.transAxes, fontsize=8,
                bbox=dict(fc="white", ec="0.7", alpha=0.9))

        ax = axes[1]
        ladder = verdict["at2_convergence_ladder"]
        h = np.array([r["h_over_l"] for r in ladder])
        l2 = np.array([r["l2_relative_error"] for r in ladder])
        nod = np.array([r["nodal_max_relative_error"] for r in ladder])
        ax.loglog(h, l2, "s-", label="L2 relative error")
        ax.loglog(h, nod, "^-", label="nodal max relative error")
        ref_h = np.array([h.min(), h.max()])
        ax.loglog(ref_h, l2.min() * (ref_h / ref_h.min()) ** 2, "k--", lw=1.0,
                  label="slope 2 reference")
        ax.axhline(ACCEPTANCE_REL_TOL, color="0.4", ls=":", lw=1.0, label="3 % line")
        ax.axvline(1.0, color="0.7", ls="-.", lw=0.8)
        ax.set_xlabel("h / l")
        ax.set_ylabel("relative error")
        ax.set_title(f"h-refinement (fitted order {verdict['at2_observed_l2_order']:.2f})")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, which="both")

        ax = axes[2]
        ls = [w["ell"] for w in verdict["width_scaling"]]
        ws = [w["width_ratio"] for w in verdict["width_scaling"]]
        ax.plot(ls, ws, "o-", color="#2e7d32", label=r"$x(d=1/e)/l$ (numeric)")
        ax.axhline(1.0, color="#1f4e79", ls="--", lw=1.2, label=r"analytic $=1$")
        ax.set_xlabel("regularisation length l [m]")
        ax.set_ylabel(r"$x(d=1/e)/l$")
        ax.set_title("width is controlled by the free parameter l")
        ax.set_ylim(0.90, 1.06)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        fig.suptitle(
            "XVAL " + verdict["anchor"] + ": NOT_APPLICABLE verdict + AT2 substitute anchor",
            fontsize=11,
        )
        fig.tight_layout()
        png = out / "xval_not_applicable_phase_field.png"
        fig.savefig(png, dpi=130)
        print(f"\nwrote {payload} and {png}")

    print(f"\nelapsed {time.time() - t0:.2f} s")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())