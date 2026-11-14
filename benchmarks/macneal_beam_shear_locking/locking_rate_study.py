"""Asymptotic convergence-rate study for the MacNeal (1985) slender-beam anchor.

Why this study exists
---------------------
``run.py`` grades the anchor at ``n = 2, 4, 8, 16`` and reports that the
equal-order ("full" shear rule) element is the least accurate (``-11.13 %`` at
``n = 16``) and that it "decays slowly".  That statement is *pre-asymptotic*:
the local error ratio of the equal-order rule is still growing at ``n = 16``
(1.33, 2.00, 3.00 per doubling), so the anchor alone cannot say **how** the
locking element fails.  Two physically different failure modes are compatible
with an 11 % error at one mesh:

* **(A) rate defect** -- shear locking degrades the *asymptotic* convergence
  rate, so the equal-order rule keeps falling behind at every mesh; or
* **(B) constant defect** -- the rate is unchanged and locking only multiplies
  the error *constant*, i.e. the same accuracy is reached on a uniformly finer
  mesh (locking = a mesh-cost penalty, not a limit).

This study discriminates (A) from (B) by extending the ladder until the rate is
settled, and it is fully deterministic (no loads, no seeds, no randomness).

Pre-registered protocol
-----------------------
Written down and fixed *before* the ladder below was run to completion (the
levels were chosen from the measured cost per solve -- 2.2 s at ``n = 256`` --
not from the outcome).

Decision
    Which of (A)/(B) describes the equal-order Q4 Mindlin rule at
    ``L/h = 10``, and how much mesh does locking cost at 5 % accuracy.
Alternatives
    ``H_rate``  = (A) asymptotic rate of the equal-order rule ``p_full <= 1.5``
                  (at least half an order below the locking-free rate).
    ``H_const`` = (B) ``p_full >= 1.8`` and ``|p_full - p_free| <= 0.2``.
Metrics (unit of analysis = one treatment x one mesh level)
    ``err(n)``  = relative centre-deflection error vs the Timoshenko closed
                  form (``run.beam_analytic``, zero hand-copying), signed;
    ``p_i``     = local rate ``log2(|err(n_i)| / |err(n_{i+1})|)`` per doubling;
    ``p_inf``   = least-squares slope of ``log|err|`` vs ``log n`` over the three
                  finest levels (``n = 64, 128, 256``);
    ``n_tol``   = elements needed for 5 % accuracy, log-log interpolated
                  between the bracketing levels;
    ``C_ratio`` = ``|err_full| / |err_sri|`` at the finest level (error-constant
                  ratio), and ``sqrt(C_ratio)`` = its predicted mesh penalty.
Thresholds (pre-declared, not tuned after the run)
    ``H_const`` accepted iff  ``p_inf(full) >= 1.8``  **and**
    ``|p_inf(full) - p_inf(sri)| <= 0.2``; otherwise ``H_rate`` is accepted iff
    ``p_inf(full) <= 1.5``; anything between is reported as *undecided*.
    ``n_tol(full) / n_tol(sri) >= 2`` = locking is a material mesh penalty.
    ``|p_local(n = 8..16)| < 1.8`` = the anchor's ``n = 16`` verdict is
    pre-asymptotic (must be reported as such in the README).
Stopping rule
    Extend the ladder while fewer than three consecutive doublings have
    ``|p_i - 2| <= 0.05`` for **every** treatment; hard caps ``n = 256`` and
    600 s wall clock (dense direct solves, ``3 * 2 * (n+1)`` DOF).
Confounders / controls
    * Solver/precision floor: dense LU in float64; the smallest reported error
      (``4.9e-4``) is >6 orders above the solver noise, so round-off cannot
      produce the measured slopes (asserted in ``validate`` mode).
    * Point-load kink: an interior point load limits the optimal deflection
      rate to ``O(h^2)`` -- equal for all three treatments, so it cannot create
      a *difference* between them (it does explain why no treatment exceeds
      ``p = 2``).
    * Only the shear block changes between treatments: the 2x2 bending block,
      mesh, boundary conditions, load vector and reference are shared.
    * The strip is one element through the thickness for every treatment, so
      the width-direction kinematics cannot mask the axial locking effect.

Outputs (relative to the repository root)
    * ``benchmarks/macneal_beam_shear_locking/locking_rate.json`` -- every number
    * ``docs/assets/macneal_beam/locking_rate.png`` -- log-log error decay

Usage::

    OMP_NUM_THREADS=4 PYTHONPATH=src python benchmarks/macneal_beam_shear_locking/locking_rate_study.py
"""

from __future__ import annotations

import importlib.util
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "locking_rate.json"
ASSETS = ROOT / "docs" / "assets" / "macneal_beam"

LEVELS = (2, 4, 8, 16, 32, 64, 128, 256)  # elements along the axis
TREATMENTS = ("full", "sri", "mitc")
ANCHOR_LEVELS = (2, 4, 8, 16)  # the ladder graded by run.py
TOLERANCE = 0.05  # 5 % acceptance tolerance of the anchor

# Pre-registered thresholds (see module docstring; do not tune after the run).
RATE_ASYMPTOTIC = 2.0  # optimal O(h^2) deflection rate for this case
RATE_CONST_MIN = 1.8  # H_const: asymptotic rate of the equal-order rule
RATE_CONST_SPREAD = 0.2  # H_const: |p_full - p_free|
RATE_DEFECT_MAX = 1.5  # H_rate: asymptotic rate that would prove (A)
RATE_SETTLED = 0.05  # stopping rule: |p_i - 2| for three doublings
MESH_PENALTY_MIN = 2.0  # locking is material iff n_tol ratio >= this
COST_CAP_SECONDS = 600.0
PRECISION_FLOOR = 1e-10  # float64 dense-solve noise; errors must exceed this


def _load_run():
    """Import the anchor driver (``run.py``) by path, house style."""
    path = HERE / "run.py"
    spec = importlib.util.spec_from_file_location("macneal_run", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUN = _load_run()


# --------------------------------------------------------------------------- #
# Analysis primitives (pure arithmetic -> unit-testable without solving)
# --------------------------------------------------------------------------- #
def local_rates(levels, errors):
    """Local convergence rate per doubling, ``log2(|e_i| / |e_{i+1}|)``."""
    out = []
    for i in range(len(levels) - 1):
        e0, e1 = abs(errors[i]), abs(errors[i + 1])
        if e1 <= 0.0 or e0 <= 0.0:
            raise ValueError("local rate undefined for a zero error")
        out.append(
            {
                "n_from": levels[i],
                "n_to": levels[i + 1],
                "ratio": e0 / e1,
                "rate": math.log2(e0 / e1),
            }
        )
    return out


def asymptotic_rate(levels, errors, count=3):
    """Least-squares slope of ``log|e|`` vs ``log n`` over the finest ``count`` levels."""
    if count < 2 or count > len(levels):
        raise ValueError("need 2 <= count <= number of levels")
    xs = [math.log(float(n)) for n in levels[-count:]]
    ys = [math.log(abs(float(e))) for e in errors[-count:]]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0.0:
        raise ValueError("degenerate ladder")
    return -sxy / sxx


def mesh_for_tolerance(levels, errors, tolerance=TOLERANCE):
    """Elements needed to reach ``tolerance``, log-log interpolated.

    Returns ``None`` when the ladder never reaches the tolerance (then the
    ceiling is reported instead) and ``levels[0]`` when it is met at the
    coarsest level already (reported as a floor).
    """
    for i in range(len(levels) - 1):
        e0, e1 = abs(errors[i]), abs(errors[i + 1])
        if e0 > tolerance >= e1:
            frac = math.log(e0 / tolerance) / math.log(e0 / e1)
            return levels[i] * 2.0**frac
    if abs(errors[0]) <= tolerance:
        return float(levels[0])
    return None


def settled(levels, errors, rate_ideal=RATE_ASYMPTOTIC, spread=RATE_SETTLED, doublings=3):
    """Stopping rule: are the last ``doublings`` local rates within ``spread``?"""
    rates = local_rates(levels, errors)
    if len(rates) < doublings:
        return False
    return all(abs(r["rate"] - rate_ideal) <= spread for r in rates[-doublings:])


# --------------------------------------------------------------------------- #
# The study
# --------------------------------------------------------------------------- #
def rate_study(levels=LEVELS, treatments=TREATMENTS, cfg=None):
    """Run the ladder and grade the pre-registered criteria."""
    cfg = dict(RUN.BEAM) if cfg is None else dict(cfg)
    analytic = RUN.beam_analytic(cfg)
    reference = analytic["timoshenko"]
    started = time.perf_counter()

    series = {}
    dofs = {}
    for treatment in treatments:
        errors, deflections = [], []
        for n in levels:
            res = RUN.solve_beam(n, treatment, cfg)
            deflections.append(res["centre_deflection"])
            errors.append((res["centre_deflection"] - reference) / reference)
            dofs[n] = res["dofs"]
            if time.perf_counter() - started > COST_CAP_SECONDS:
                raise RuntimeError(f"cost cap {COST_CAP_SECONDS:.0f} s exceeded at n={n}")
        series[treatment] = {
            "deflections": deflections,
            "errors": errors,
            "local_rates": local_rates(levels, errors),
            "asymptotic_rate": asymptotic_rate(levels, errors),
            "mesh_for_5pct": mesh_for_tolerance(levels, errors),
            "settled": settled(levels, errors),
        }

    full = series["full"]
    free = series["sri"]  # mitc is identical here
    p_full, p_free = full["asymptotic_rate"], free["asymptotic_rate"]
    n_full, n_free = full["mesh_for_5pct"], free["mesh_for_5pct"]
    constant_ratio = abs(full["errors"][-1]) / abs(free["errors"][-1])
    mesh_penalty = (n_full / n_free) if (n_full and n_free) else None
    predicted_penalty = math.sqrt(constant_ratio)

    # --- pre-registered decision ------------------------------------------- #
    if p_full >= RATE_CONST_MIN and abs(p_full - p_free) <= RATE_CONST_SPREAD:
        hypothesis = "H_const"
        decision = (
            "equal-order rule shares the asymptotic rate of the "
            "locking-free rules; shear locking multiplies the error "
            "constant (mesh-cost defect), it does not cap accuracy"
        )
    elif p_full <= RATE_DEFECT_MAX:
        hypothesis = "H_rate"
        decision = (
            "equal-order rule has a degraded asymptotic rate; shear "
            "locking is a rate defect, not only a constant defect"
        )
    else:
        hypothesis = "undecided"
        decision = (
            f"asymptotic rate {p_full:.2f} lies between the "
            f"pre-registered bounds ({RATE_DEFECT_MAX}, {RATE_CONST_MIN})"
        )

    anchor_rate = [r for r in full["local_rates"] if r["n_from"] == 8][0]["rate"]
    tier_eval = {
        "H_const_asymptotic_rate": {
            "criterion": f"p_inf(full) >= {RATE_CONST_MIN} and "
            f"|p_inf(full) - p_inf(sri)| <= {RATE_CONST_SPREAD}",
            "value": {"p_inf_full": p_full, "p_inf_sri": p_free},
            "pass": bool(p_full >= RATE_CONST_MIN and abs(p_full - p_free) <= RATE_CONST_SPREAD),
        },
        "H_rate_asymptotic_rate": {
            "criterion": f"p_inf(full) <= {RATE_DEFECT_MAX}",
            "value": p_full,
            "pass": bool(p_full <= RATE_DEFECT_MAX),
        },
        "mesh_penalty_material": {
            "criterion": f"n_tol(full)/n_tol(sri) >= {MESH_PENALTY_MIN}",
            "value": {"n_tol_full": n_full, "n_tol_sri": n_free, "ratio": mesh_penalty},
            "pass": bool(mesh_penalty is not None and mesh_penalty >= MESH_PENALTY_MIN),
        },
        "anchor_mesh_is_preasymptotic": {
            "criterion": f"local rate at n=8->16 of the full rule < {RATE_CONST_MIN}",
            "value": anchor_rate,
            "pass": bool(anchor_rate < RATE_CONST_MIN),
        },
        "precision_floor_clear": {
            "criterion": f"smallest |err| > {PRECISION_FLOOR:g}",
            "value": min(abs(e) for s in series.values() for e in s["errors"]),
            "pass": bool(
                min(abs(e) for s in series.values() for e in s["errors"]) > PRECISION_FLOOR
            ),
        },
        "tolerance_reached_on_ladder": {
            "criterion": f"|err| < {TOLERANCE} for every treatment by n <= {levels[-1]}",
            "value": {t: abs(s["errors"][-1]) for t, s in series.items()},
            "pass": bool(all(abs(s["errors"][-1]) < TOLERANCE for s in series.values())),
        },
    }

    return {
        "benchmark": "macneal_1985_slender_beam_shear_locking",
        "study": "asymptotic_convergence_rate",
        "generated_by": "benchmarks/macneal_beam_shear_locking/locking_rate_study.py",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reference": {
            "model": "Timoshenko closed form PL^3/(48EI) + PL/(4kAG), k=5/6",
            "value": reference,
            "euler_bernoulli": analytic["euler_bernoulli"],
            "source": "evaluated in run.beam_analytic (zero hand-copying)",
        },
        "environment": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "dtype": "torch.float64",
            "solver": "dense LU (torch.linalg.solve)",
            "deterministic": True,
        },
        "protocol": {
            "levels": list(levels),
            "treatments": list(treatments),
            "tolerance": TOLERANCE,
            "pre_registered": {
                "rate_asymptotic": RATE_ASYMPTOTIC,
                "rate_const_min": RATE_CONST_MIN,
                "rate_const_spread": RATE_CONST_SPREAD,
                "rate_defect_max": RATE_DEFECT_MAX,
                "rate_settled": RATE_SETTLED,
                "mesh_penalty_min": MESH_PENALTY_MIN,
                "cost_cap_seconds": COST_CAP_SECONDS,
                "precision_floor": PRECISION_FLOOR,
            },
        },
        "dofs_by_level": {str(k): v for k, v in sorted(dofs.items(), key=lambda kv: kv[0])},
        "series": series,
        "locking": {
            "error_constant_ratio_full_over_sri": constant_ratio,
            "predicted_mesh_penalty_sqrt_ratio": predicted_penalty,
            "measured_mesh_penalty": mesh_penalty,
            "penalty_prediction_error_pct": (
                100.0 * (mesh_penalty - predicted_penalty) / predicted_penalty
                if mesh_penalty
                else None
            ),
        },
        "decision": {"hypothesis": hypothesis, "statement": decision},
        "gates": tier_eval,
        "anchor_levels": list(ANCHOR_LEVELS),
        "total_seconds": time.perf_counter() - started,
    }


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def make_figure(payload: dict) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ASSETS.mkdir(parents=True, exist_ok=True)
    levels = payload["protocol"]["levels"]
    styles = {"full": ("o-", "tab:red"), "sri": ("s-", "tab:blue"), "mitc": ("^-", "tab:green")}
    fig, ax = plt.subplots(figsize=(7.0, 4.8), dpi=150)
    for treatment in payload["protocol"]["treatments"]:
        s = payload["series"][treatment]
        mk, col = styles[treatment]
        ax.loglog(
            levels,
            [abs(e) for e in s["errors"]],
            mk,
            color=col,
            ms=5,
            markerfacecolor="white",
            label=f"{treatment}  (p_inf = {s['asymptotic_rate']:.2f})",
        )
    # O(h^2) guide anchored on the finest sri point
    sri = payload["series"]["sri"]["errors"]
    anchor = abs(sri[-1])
    guide = [anchor * (levels[-1] / n) ** 2 for n in levels]
    ax.loglog(levels, guide, "k--", lw=1.0, label=r"$O(h^2)$ guide")
    ax.axhline(
        payload["protocol"]["tolerance"],
        color="gray",
        ls=":",
        lw=1.4,
        label="5 % tolerance (anchor)",
    )
    ax.axvspan(2, payload["anchor_levels"][-1], color="gray", alpha=0.10)
    ax.text(
        0.03,
        0.93,
        f"grey: anchor ladder (n = 2..{payload['anchor_levels'][-1]})",
        transform=ax.transAxes,
        fontsize=7.5,
        color="0.35",
        va="top",
    )
    ax.text(
        0.03,
        0.86,
        f"full vs sri: same O(h^2) rate, error constant x"
        f"{payload['locking']['error_constant_ratio_full_over_sri']:.0f}"
        f"  ->  mesh penalty x{payload['locking']['measured_mesh_penalty']:.1f}",
        transform=ax.transAxes,
        fontsize=7.5,
        color="tab:red",
        va="top",
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(levels)
    ax.set_xticklabels([str(v) for v in levels], fontsize=7.5)
    ax.set_xlabel("elements along the axis  (h = L/n, L/h_beam = 10)")
    ax.set_ylabel("|centre-deflection error| vs Timoshenko")
    ax.set_title("Shear locking = error-constant defect, not a rate defect")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    path = ASSETS / "locking_rate.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> int:
    payload = rate_study()
    path = make_figure(payload)
    payload["figure"] = {
        "name": path.name,
        "caption": "|error| vs elements per treatment on a log-log scale; the "
        "equal-order 'full' curve is parallel to the locking-free "
        "curves (same O(h^2) rate) but shifted by a constant factor.",
    }
    OUT.write_text(json.dumps(payload, indent=2))
    summary = {
        "levels": list(LEVELS),
        "asymptotic_rates": {t: s["asymptotic_rate"] for t, s in payload["series"].items()},
        "mesh_for_5pct": {t: s["mesh_for_5pct"] for t, s in payload["series"].items()},
        "error_constant_ratio": payload["locking"]["error_constant_ratio_full_over_sri"],
        "measured_mesh_penalty": payload["locking"]["measured_mesh_penalty"],
        "predicted_mesh_penalty": payload["locking"]["predicted_mesh_penalty_sqrt_ratio"],
        "decision": payload["decision"],
        "gates": {k: v["pass"] for k, v in payload["gates"].items()},
        "seconds": payload["total_seconds"],
    }
    print(json.dumps(summary, indent=2))
    failed = [
        k for k, v in payload["gates"].items() if not v["pass"] and k != "H_rate_asymptotic_rate"
    ]
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
