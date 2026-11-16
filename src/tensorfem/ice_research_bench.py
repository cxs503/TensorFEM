"""ICE-SHOW-RESEARCH-BENCH: ice research bench — problem library + one-click demos + ICE-* task board.

This module is the *showcase* surface of the ice research capability set
(anchor ``ICE-SHOW-RESEARCH-BENCH``).  It does **not** re-derive any solver:
every problem below is a *curated pointer* to an already-verified ice module
in :mod:`tensorfem`, carrying (a) a physical problem statement, (b) the
acceptance criterion (判据) that the module's own test suite enforces, and
(c) the analytic / literature reference the criterion is anchored to.

Three deliverables
------------------
1. **Problem library** — :data:`ICE_PROBLEMS` is a curated list of ≥12 ice
   research problems, each with ``id``, ``title``, ``background``,
   ``criterion`` (the acceptance gate), ``reference`` and ``module``.  The
   library is *self-checking*: :func:`check_problem_library` rejects any
   entry that lacks a criterion or a reference, so the "all problems carry
   a 判据" acceptance is enforced by construction.
2. **One-click demos** — :func:`demo_ice_pressure_area` and
   :func:`demo_ice_plate_flexural_limit` each run a *real* ice solver path
   (analytic pressure-area law; FE first-crack moment field) and emit a
   matplotlib figure plus a machine-readable ``{name, value, reference,
   rel_error, passed}`` verdict row.  Both are deliberately lightweight
   (small mesh / analytic route) so they run in seconds on a shared node.
3. **ICE-* task board** — :func:`ice_task_board` renders the ICE-* task
   registry (the curated problems plus their verification status) as a
   flat table, the "task board view" of the anchor.

Honesty statement
-----------------
* The demos are *demonstrations*, not full verification sweeps: they use
  reduced mesh / analytic routes so they stay lightweight.  The full
  verification lives in the referenced modules' own test suites
  (``tests/test_ice_*``), which this bench does not re-run.
* Every ``rel_error`` reported here is against the *same* reference the
  underlying module uses; a demo that reports ``< 3 %`` is therefore
  evidence the module's own anchor is healthy, not a new independent claim.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

import math


# ---------------------------------------------------------------------------
# 1. Problem library (curated, self-checking)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IceProblem:
    """One curated ice research problem with its acceptance criterion."""
    id: str
    title: str
    background: str
    criterion: str          # the acceptance gate (判据)
    reference: str          # analytic / literature anchor
    module: str             # implementing tensorfem module

    def to_dict(self):
        return asdict(self)


# ≥12 problems, each with a criterion and a reference.  The `module` column
# points at the already-verified ice module whose own test suite enforces the
# criterion (see tests/test_ice_*).
ICE_PROBLEMS: tuple[IceProblem, ...] = (
    IceProblem(
        id="ICE-P01",
        title="Ice pressure-area law (local ice action)",
        background=("Local ice pressure on a hull panel decreases with loaded "
                    "area: p(A) = C_d C_f p_ref (A/A_ref)^-alpha, the IACS "
                    "Polar Class / ISO 19906 local-load structure."),
        criterion=("Three independent evaluation routes (direct, log-space, "
                   "marginal dF/dA) must agree to machine precision; the "
                   "fitted exponent must recover alpha = 0.3."),
        reference="IACS UR I2 / ISO 19906:2010 pressure-area form",
        module="tensorfem.ice_belt_pressure_area",
    ),
    IceProblem(
        id="ICE-P02",
        title="Ice plate flexural (first-crack) limit on a Winkler bed",
        background=("An infinite floating ice sheet under a circular patch "
                    "load: the FE moment field must reproduce the exact "
                    "Kelvin centre moment and the Westergaard/Nevel "
                    "first-crack load."),
        criterion=("FE first-crack load vs Kelvin and vs Nevel must both be "
                   "< 3 %; the centre moment must match the Kelvin anchor "
                   "to < 1 %."),
        reference="Westergaard 1926 / Nevel 1969 / Kerr 1979; ISO 19906",
        module="tensorfem.ice_plate_flexural_limit",
    ),
    IceProblem(
        id="ICE-P03",
        title="Low-temperature steel flow stress (Arctic grades)",
        background=("DH36/EH36/FH36 Arctic-grade steel yield stress as a "
                    "piecewise-linear function of temperature."),
        criterion=("Interpolation between published test points must be "
                   "exact; out-of-range / unknown grades must be rejected."),
        reference="CSR for Polar Ships §5 / DNV-RP-C204",
        module="tensorfem.ice_plate_plasticity",
    ),
    IceProblem(
        id="ICE-P04",
        title="Ilyushin plate plastic limit load",
        background=("Full-plastic limit load of a rectangular plate under "
                    "combined membrane stress resultants (Ilyushin "
                    "closed form)."),
        criterion=("FE limit load vs Ilyushin closed form must be < 10 % "
                   "(perfect-plasticity lower-bound check)."),
        reference="Ilyushin 1948",
        module="tensorfem.ice_plate_plasticity",
    ),
    IceProblem(
        id="ICE-P05",
        title="Grillage ice rule benchmark (IACS Polar Class)",
        background=("Grillage (stiffened panel) ultimate strength under "
                    "patch ice pressure vs the IACS rule plate-thickness / "
                    "stiffener-modulus / frame-modulus formulas."),
        criterion=("Rule plate thickness at p_ult within 25 % of actual; "
                   "safety margin in [0.5, 2.0]; margin monotone in axial "
                   "ratio."),
        reference="IACS UR I2 Polar Class rules",
        module="tensorfem.grillage_ice_rule_benchmark",
    ),
    IceProblem(
        id="ICE-P06",
        title="Ice-belt design-pressure zone partition",
        background=("Longitudinal hull areas x structural load patches x "
                    "ice-belt extent must tile the hull without gaps or "
                    "overlaps."),
        criterion=("Zone partition must tile the hull exactly (sum of zone "
                   "lengths = hull length); belt extent must be contained "
                   "within the draught."),
        reference="IACS UR I2 hull-area division",
        module="tensorfem.ice_belt_pressure_area",
    ),
    IceProblem(
        id="ICE-P07",
        title="Ice-coupled 6-DOF added-mass information gate",
        background=("The ice chain's infinite-frequency heave added mass "
                    "(A_inf = 1.00 m) must be judged against an independent "
                    "2-D potential-flow route with a stated detection "
                    "power."),
        criterion=("Judged quantity read from the chain with its own "
                   "geometry; reference route validated on the exact "
                   "deep-cylinder value; decision rule per ILAC-G8 / "
                   "JCGM 106."),
        reference="ILAC-G8:09/2019 / JCGM 106:2012",
        module="tensorfem.ice_gate_power",
    ),
    IceProblem(
        id="ICE-P08",
        title="Ice-belt collapse / ratcheting pressure chains",
        background=("(pressure -> collapse) and (pressure -> cyclic "
                    "accumulation) chains for an ice-belt plate field."),
        criterion=("Each chain must be verified on an independent "
                   "evaluation route before it may speak for a real "
                   "panel."),
        reference="ISO 19906:2010 ice action",
        module="tensorfem.ice_belt_collapse / ice_belt_ratcheting",
    ),
    IceProblem(
        id="ICE-P09",
        title="Ice slamming / whipping response",
        background=("Slamming impact and whipping vibration of an "
                    "ice-going hull under wave impact."),
        criterion=("Slamming/whipping response must be verified against "
                   "the analytic impact / modal reference."),
        reference="Zhao-Faltinsen wedge / modal beam theory",
        module="tensorfem.ice_slamming_whipping",
    ),
    IceProblem(
        id="ICE-P10",
        title="Propeller ice-milling fatigue chain",
        background=("Fatigue damage accumulation in a propeller under "
                    "repeated ice milling events."),
        criterion=("Fatigue chain must be verified against the "
                   "rainflow-spectrum reference."),
        reference="Rainflow counting / Miner's rule",
        module="tensorfem.propeller_ice_milling_fatigue_chain",
    ),
    IceProblem(
        id="ICE-P11",
        title="Weld fatigue in ice structures",
        background=("Fatigue life of welded ice-structure joints under "
                    "cyclic ice loading."),
        criterion=("Weld fatigue must be verified against the "
                   "S-N / hot-spot reference."),
        reference="DNV-RP-C203 / hot-spot stress",
        module="tensorfem.weld_fatigue_ice",
    ),
    IceProblem(
        id="ICE-P12",
        title="Stiffened-panel ice ultimate strength",
        background=("Ultimate strength of a stiffened ice panel under "
                    "combined ice pressure and axial load."),
        criterion=("Panel ultimate strength vs the analytic / rule "
                   "reference must be within the stated gate."),
        reference="CSR / stiffened-panel collapse",
        module="tensorfem.stiffened_panel_ice_ultimate",
    ),
    IceProblem(
        id="ICE-P13",
        title="Suboff ice collision (appendage impact)",
        background=("Ice impact on a suboff appendage: contact force and "
                    "local ice damage under prescribed impact."),
        criterion=("Contact expectation / ice damage must be verified "
                   "against the impact reference."),
        reference="ISO 19906 impact / contact mechanics",
        module="tensorfem.suboff_ice_collision",
    ),
    IceProblem(
        id="ICE-P14",
        title="Broken-ice floe field (discontinuous ice)",
        background=("Discontinuous ice shells with prescribed cracks, "
                    "buoyancy and fragment diagnostics."),
        criterion=("Floe-field geometry / buoyancy must be verified "
                   "against the fragment-diagnostics reference."),
        reference="Discontinuous ice / fragment kinematics",
        module="tensorfem.broken_ice_floe_field",
    ),
)


def check_problem_library(problems: Sequence[IceProblem] = ICE_PROBLEMS) -> dict:
    """Self-check: every problem must carry a criterion and a reference.

    Returns ``{n_problems, n_with_criterion, n_with_reference, passed}``.
    ``passed`` is False if any entry lacks a criterion or a reference (the
    "all problems carry a 判据" acceptance is enforced by construction).
    """
    n = len(problems)
    n_crit = sum(1 for p in problems if p.criterion and p.criterion.strip())
    n_ref = sum(1 for p in problems if p.reference and p.reference.strip())
    ids = [p.id for p in problems]
    dup = len(ids) != len(set(ids))
    return {
        "n_problems": n,
        "n_with_criterion": n_crit,
        "n_with_reference": n_ref,
        "unique_ids": not dup,
        "passed": n >= 12 and n_crit == n and n_ref == n and not dup,
    }


# ---------------------------------------------------------------------------
# 2. One-click demos (lightweight, real solver paths, emit figures)
# ---------------------------------------------------------------------------

def _verdict_row(name: str, value: float, reference: float, gate: float) -> dict:
    """One machine-readable verdict row (programmatic, no hand transcription)."""
    if reference == 0:
        raise ValueError("zero-reference verdicts must be registered as invariants")
    rel = abs(float(value) - float(reference)) / abs(float(reference))
    return {
        "name": name,
        "value": float(value),
        "reference": float(reference),
        "rel_error": rel,
        "gate": gate,
        "passed": rel < gate,
    }


def demo_ice_pressure_area(
    areas: Sequence[float] = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0),
    p_ref: float = 2.0e6,
    area_ref: float = 1.0,
    alpha: float = 0.3,
    out_png: str | None = None,
) -> dict:
    """One-click demo: ice pressure-area law, three routes, log-log figure.

    Runs the *real* :class:`tensorfem.ice_belt_pressure_area.PressureAreaLaw`
    (direct / log-space / marginal-dF-dA routes) and emits a log-log
    pressure-vs-area figure.  The verdict is the route-agreement residual
    (machine precision) and the recovered exponent vs ``alpha``.
    """
    from .ice_belt_pressure_area import (
        PressureAreaLaw,
        pressure_area_curve_rows,
        fit_loglog_exponent,
    )

    law = PressureAreaLaw(c_d=1.0, c_f=1.0, p_ref=p_ref, area_ref=area_ref,
                          exponent=alpha)
    rows = pressure_area_curve_rows(law, list(areas))
    fit = fit_loglog_exponent(law, list(areas))

    # ---- figure: log-log pressure vs area, three routes ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    A = [r["area_m2"] for r in rows]
    p_direct = [r["pressure_mpa_direct"] for r in rows]
    p_log = [r["pressure_mpa_logspace"] for r in rows]
    p_marg = [r["pressure_mpa_marginal"] for r in rows]
    ax.loglog(A, p_direct, "o-", label="direct", color="#2b6cb0")
    ax.loglog(A, p_log, "s--", label="log-space", color="#dd6b20")
    ax.loglog(A, p_marg, "^-", label="marginal dF/dA", color="#38a169")
    ax.set_xlabel(r"loaded area $A$ [m$^2$]")
    ax.set_ylabel(r"ice pressure $p$ [MPa]")
    ax.set_title("Ice pressure-area law: three independent routes")
    ax.legend()
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=120)
    plt.close(fig)

    # ---- verdict: route agreement (residual vs 0, an invariant) + exponent ----
    max_route_diff = max(r["rel_diff_logspace"] for r in rows)
    # route agreement is a *residual* against machine zero: report it as an
    # invariant (must be < 1e-9), not as a relative-error vs a reference.
    verdicts = [
        {
            "name": "route_agreement_logspace",
            "value": float(max_route_diff),
            "reference": "machine zero (invariant)",
            "rel_error": float(max_route_diff),
            "gate": 1e-9,
            "passed": max_route_diff < 1e-9,
        },
        _verdict_row("recovered_exponent", fit["fitted_exponent"], -alpha, 1e-9),
    ]
    return {
        "demo": "ice_pressure_area",
        "n_points": len(rows),
        "figure": out_png,
        "verdicts": verdicts,
        "all_passed": all(v["passed"] for v in verdicts),
    }


def demo_ice_plate_flexural_limit(
    patch_radius: float = 0.5,
    flexural_strength: float = 6.0e5,
    cells: int = 40,
    out_png: str | None = None,
) -> dict:
    """One-click demo: ice plate first-crack limit vs Kelvin / Nevel.

    Runs the *real* :func:`tensorfem.ice_plate_flexural_limit.first_crack_load`
    (FE moment field + independent Rankine criterion) and emits a bar figure
    of the FE limit vs the two analytic references.  The verdict is the
    relative error vs Kelvin and vs Nevel (both must be < 3 %).
    """
    from .ice_plate_flexural_limit import first_crack_load

    r = first_crack_load(
        patch_radius=patch_radius,
        flexural_strength=flexural_strength,
        young=7.0e9, poisson=0.30, thickness=0.30,
        foundation=1025.0 * 9.81,
        cells=cells,
    )

    # ---- figure: FE limit vs Kelvin / Nevel ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    labels = ["FE (Rankine)", "Kelvin", "Nevel"]
    vals = [r["first_crack_load_fem"], r["first_crack_load_kelvin"],
            r["first_crack_load_nevel"]]
    bars = ax.bar(labels, vals, color=["#2b6cb0", "#dd6b20", "#38a169"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.02, f"{v:.3g}",
                ha="center", fontsize=9)
    ax.set_ylabel(r"first-crack load $P_f$ [N]")
    ax.set_title(f"Ice plate first-crack limit (patch radius {patch_radius} m)")
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=120)
    plt.close(fig)

    verdicts = [
        _verdict_row("fe_vs_kelvin", r["first_crack_load_fem"],
                     r["first_crack_load_kelvin"], 0.03),
        _verdict_row("fe_vs_nevel", r["first_crack_load_fem"],
                     r["first_crack_load_nevel"], 0.03),
    ]
    return {
        "demo": "ice_plate_flexural_limit",
        "patch_radius": patch_radius,
        "figure": out_png,
        "verdicts": verdicts,
        "all_passed": all(v["passed"] for v in verdicts),
    }


# ---------------------------------------------------------------------------
# 3. ICE-* task board view
# ---------------------------------------------------------------------------

def ice_task_board(problems: Sequence[IceProblem] = ICE_PROBLEMS) -> dict:
    """Render the ICE-* task board: curated problems + library self-check.

    Returns a flat table (``rows``) of ``{id, title, criterion, reference,
    module}`` plus the library ``check`` — the "task board view" of the
    anchor.
    """
    check = check_problem_library(problems)
    return {
        "check": check,
        "rows": [p.to_dict() for p in problems],
    }


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    """CLI: print the problem library self-check and run both demos."""
    import json
    import tempfile
    from pathlib import Path

    check = check_problem_library()
    print(json.dumps(check, indent=2))
    with tempfile.TemporaryDirectory() as td:
        p1 = Path(td) / "pressure_area.png"
        p2 = Path(td) / "flexural_limit.png"
        d1 = demo_ice_pressure_area(out_png=str(p1))
        d2 = demo_ice_plate_flexural_limit(out_png=str(p2))
        print("demo1:", d1["all_passed"], "figure", p1.exists())
        print("demo2:", d2["all_passed"], "figure", p2.exists())
    return 0
