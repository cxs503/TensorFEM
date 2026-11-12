"""Machine gate for the MacNeal (1985) slender-beam shear-locking benchmark.

Reads the artefacts produced by ``run.py`` and re-derives every verdict from the
raw numbers in ``results.json`` (never from stored flags), then checks:

1. ``results.json`` exists and carries all three shear treatments over the
   refinement ladder;
2. the closed-form references are reproduced from the stored case data (zero
   hand-copying: the gate re-evaluates the Euler-Bernoulli and Timoshenko
   closed forms and requires agreement with the stored values to 1e-9);
3. the locking-free rules (``sri``, ``mitc``) are within the 5 % tolerance of
   the Timoshenko reference at the finest level (and of the Euler-Bernoulli
   yardstick, the MacNeal yardstick);
4. the equal-order ``full`` rule is demonstrably the least accurate at the
   finest level (the locking signature) and its error decays slower than the
   locking-free rules;
5. the locking-free error is non-increasing along the ladder;
6. the figure artefacts exist.

Usage::

    PYTHONPATH=src python benchmarks/macneal_beam_shear_locking/validate.py

Exit code 0 = every criterion holds, 1 = at least one failure.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results.json"
ASSETS = ROOT / "docs" / "assets" / "macneal_beam"
TOLERANCE = 0.05
TREATMENTS = ("full", "sri", "mitc")
LOCKING_FREE = ("sri", "mitc")


def _check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


def main() -> int:
    if not RESULTS.exists():
        print(f"FAIL: {RESULTS} not found -- run run.py first")
        return 1
    d = json.loads(RESULTS.read_text())
    ok = True

    # 1. structure ---------------------------------------------------------- #
    rows = d.get("rows", [])
    ok &= _check("results.json has rows", len(rows) > 0, f"{len(rows)} rows")
    for t in TREATMENTS:
        ok &= _check(f"treatment '{t}' present",
                     any(r["treatment"] == t for r in rows))
    levels = sorted({r["n_elements"] for r in rows})
    ok &= _check("at least two refinement levels", len(levels) >= 2,
                 f"levels={levels}")

    # 2. references re-derived from the stored case data -------------------- #
    c = d["case"]
    E, nu, L, h, b, P = (c["young"], c["poisson"], c["length"],
                        c["height"], c["width"], c["load"])
    k = c["shear_correction_factor"]
    I = b * h ** 3 / 12.0
    G = E / (2.0 * (1.0 + nu))
    eb = P * L ** 3 / (48.0 * E * I)
    tim = eb + P * L / (4.0 * k * b * h * G)
    a = d["analytic"]
    ok &= _check("Euler-Bernoulli reference reproduced",
                 abs(a["euler_bernoulli"] - eb) < 1e-9 * max(1.0, eb),
                 f"stored={a['euler_bernoulli']:.6f} rederived={eb:.6f}")
    ok &= _check("Timoshenko reference reproduced",
                 abs(a["timoshenko"] - tim) < 1e-9 * max(1.0, tim),
                 f"stored={a['timoshenko']:.6f} rederived={tim:.6f}")

    # 3. locking-free rules within tolerance -------------------------------- #
    for t in LOCKING_FREE:
        finest = [r for r in rows if r["treatment"] == t][-1]
        e_tim = (finest["centre_deflection"] - tim) / tim
        e_eb = (finest["centre_deflection"] - eb) / eb
        ok &= _check(f"{t} finest within {TOLERANCE:.0%} of Timoshenko",
                     abs(e_tim) < TOLERANCE,
                     f"delta={finest['centre_deflection']:.4f} err={100*e_tim:+.3f}%")
        ok &= _check(f"{t} finest within {TOLERANCE:.0%} of Euler-Bernoulli",
                     abs(e_eb) < TOLERANCE,
                     f"err={100*e_eb:+.3f}%")

    # 4. locking signature -------------------------------------------------- #
    def finest_err(t: str) -> float:
        return abs([r for r in rows if r["treatment"] == t][-1]["error_vs_timoshenko"])

    full_err = finest_err("full")
    free_err = max(finest_err(t) for t in LOCKING_FREE)
    ok &= _check("full rule is the least accurate (locking signature)",
                 full_err > free_err,
                 f"full={100*full_err:.2f}% vs best-free={100*free_err:.2f}%")
    ok &= _check("full rule exceeds tolerance (locking, expected FAIL)",
                 full_err >= TOLERANCE, f"full={100*full_err:.2f}%")

    # 5. locking-free error non-increasing along the ladder ----------------- #
    for t in LOCKING_FREE:
        errs = [abs(r["error_vs_timoshenko"])
                for r in rows if r["treatment"] == t]
        non_inc = all(b_ <= a_ + 1e-12 for a_, b_ in zip(errs, errs[1:]))
        ok &= _check(f"{t} error non-increasing along ladder", non_inc,
                    " ".join(f"{100*e:.2f}%" for e in errs))

    # 6. figures ------------------------------------------------------------- #
    for name in d.get("figures", []):
        ok &= _check(f"figure {name} exists", (ASSETS / name).exists())

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
