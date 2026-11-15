"""GAP-ICE-GATE-POWER: evidence run + figures for the ice-chain added-mass gate.

Produces (all numbers program-injected, nothing hand-copied):

* ``benchmarks/ice_gate_power/gate_evidence.json`` -- the gate certificate, the
  two mirror-route convergence studies, the deep-cylinder self-verification, the
  B/T sensitivity sweep and the detection-rate table;
* ``benchmarks/ice_gate_power/fig1_power_curve.png``   -- detection rate vs true
  deviation, guard-banded rule vs naive rule;
* ``benchmarks/ice_gate_power/fig2_convergence.png``   -- panel convergence of the
  two reference routes (log-log, fitted observed order);
* ``benchmarks/ice_gate_power/fig3_band_vs_controls.png`` -- declared value and
  control values against the accepted reference band.

Run:  python benchmarks/ice_gate_power/run_gate_evidence.py
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorfem import ice_gate_power as gate  # noqa: E402


def _power_table(u_rel, tol, k, deltas):
    return [{
        "delta": d,
        "guard_band": gate.detection_power_analytic(d, tol, u_rel, k),
        "guard_band_mc": gate.detection_power(d, tol, u_rel, k, n_mc=100_000),
        "naive_rule": gate.detection_power_analytic(d, tol, u_rel, 0.0),
        "false_accept_guard_band": gate.false_accept_probability(d, tol, u_rel, k),
    } for d in deltas]


def _sensitivity(per_edge=160, draughts=(4.0, 6.0, 8.0)):
    """Reference band and verdict for several B/T of the same hull breadth."""
    rows = []
    for t in draughts:
        ov = {"draught_m": t}
        v = gate.gate_verdict(per_edge=per_edge, dofs=("heave",), n_mc=20_000,
                              geometry_overrides=ov)
        rows.append({
            "draught_m": t,
            "B_over_T": v["geometry"]["breadth_m"] / t,
            "roll_coupling_changed_mass_kg": v["geometry"]["mass_kg"],
            "declared_t_per_m": v["declared"]["heave"]["value_t_per_m"],
            "route_omega0_t_per_m": v["reference"]["repo_route_omega0_kg_per_m"] / 1e3,
            "route_infinity_t_per_m": v["reference"]["repo_route_infinity_kg_per_m"] / 1e3,
            "deviation_heave": v["deviations"]["heave"],
            "decision": v["decision"]["heave"],
            "control_band_half_width_rel": v["reference"]["band_half_width_rel"],
        })
    return rows


def main() -> dict:
    verdict = gate.gate_verdict(per_edge=160, n_mc=200_000)

    conv = {
        "omega0": gate.convergence_study(per_edges=(40, 80, 160, 320, 640),
                                         free_surface="omega0"),
        "infinity": gate.convergence_study(per_edges=(40, 80, 160, 320, 640),
                                           free_surface="infinity"),
    }
    deltas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.2557, 0.30, 0.40, 0.625, 1.0]
    u_sensitivity = [{
        "u_rel": u,
        "acceptance_limit": gate.acceptance_limit(gate.DEFAULT_TOL, u, gate.DEFAULT_K),
        "rejection_limit": gate.rejection_limit(gate.DEFAULT_TOL, u, gate.DEFAULT_K),
        "critical_deviation_95": gate.critical_deviation(gate.DEFAULT_TOL, u, gate.DEFAULT_K),
        "false_accept_at_tolerance": gate.false_accept_probability(
            gate.DEFAULT_TOL, gate.DEFAULT_TOL, u, gate.DEFAULT_K),
    } for u in (0.005, 0.010, 0.020, 0.029, 0.050)]
    evidence = {
        "gap_slug": gate.GAP_SLUG,
        "verdict": verdict,
        "convergence": conv,
        "power_table": _power_table(gate.DEFAULT_U_REL, gate.DEFAULT_TOL,
                                    gate.DEFAULT_K, deltas),
        "sensitivity_B_over_T": _sensitivity(),
        "sensitivity_u_rel": u_sensitivity,
        "deep_cylinder": gate.validate_reference_routes(depths=(2.0, 4.0, 8.0)),
        "repro": "python benchmarks/ice_gate_power/run_gate_evidence.py",
    }

    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "gate_evidence.json").write_text(json.dumps(evidence, indent=2))

    _figures(evidence)
    print(json.dumps({
        "pass": verdict["pass"],
        "decision_heave": verdict["decision"]["heave"],
        "declared_t_per_m": verdict["declared"]["heave"]["value_t_per_m"],
        "reference_t_per_m": verdict["reference"]["primary_t_per_m"],
        "deviation": verdict["deviations"]["heave"],
        "power_at_audited": verdict["power"]["at_audited_deviation"],
        "critical_deviation_95": verdict["power"]["critical_deviation_95"],
        "controls_ok": all(c["passed"] for c in verdict["controls"]),
    }, indent=2))
    return evidence


def _figures(evidence):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v = evidence["verdict"]
    tol, u, k = v["rule"]["tol"], v["rule"]["u_rel"], v["rule"]["k"]
    acc, rej = v["rule"]["acceptance_limit"], v["rule"]["rejection_limit"]

    # ---- fig 1: detection rate (statistical power) of the criterion ----------
    d = np.linspace(0.0, 0.7, 281)
    gb = [gate.detection_power_analytic(x, tol, u, k) for x in d]
    nv = [gate.detection_power_analytic(x, tol, u, 0.0) for x in d]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(d, gb, lw=2.0, label=f"guard-banded (u={u:.3f}, k={k:g})")
    ax.plot(d, nv, lw=1.4, ls="--", label="naive |dev|<=tol")
    ax.axvline(tol, color="0.4", lw=0.9, ls=":")
    ax.axvline(rej, color="C3", lw=0.9, ls=":")
    audited = abs(v["deviations"]["heave"])
    ax.plot([audited], [v["power"]["at_audited_deviation"]], "o", ms=6, color="C2",
            label=f"audited deviation {audited:.1%} (power {v['power']['at_audited_deviation']:.2f})")
    ax.axhline(0.95, color="0.7", lw=0.8, ls="-.")
    ax.annotate(f"tol={tol:.2f}", (tol, 0.06), rotation=90, fontsize=8)
    ax.annotate(f"rejection limit={rej:.3f}", (rej, 0.06), rotation=90, fontsize=8)
    ax.set_xlabel("true relative deviation of the declared value $\\delta$")
    ax.set_ylabel("detection rate  $P(\\mathrm{reject}\\,|\\,\\delta)$")
    ax.set_title("GAP-ICE-GATE-POWER: information content of the criterion")
    ax.set_ylim(-0.02, 1.03)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(HERE / "fig1_power_curve.png", dpi=140)
    plt.close(fig)

    # ---- fig 2: convergence of the two reference routes ---------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for key, style in (("omega0", "o-"), ("infinity", "s-")):
        c = evidence["convergence"][key]
        ref = c["richardson_kg_per_m"]
        err = [abs(x / ref - 1.0) for x in c["values_kg_per_m"]]
        ax.loglog(c["per_edges"], err, style, lw=1.4,
                  label=f"{key} route (p_obs={c['observed_order']:.2f})")
        p = c["observed_order"]
        pe = np.asarray(c["per_edges"], float)
        ax.loglog(pe, err[0] * (pe / pe[0]) ** (-p), ":", lw=0.9, color="0.6")
    ax.set_xlabel("panels per rectangle edge")
    ax.set_ylabel("|A - A_Richardson| / A_Richardson")
    ax.set_title("reference-route panel convergence (2-D single-layer BEM)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "fig2_convergence.png", dpi=140)
    plt.close(fig)

    # ---- fig 3: declared value vs band vs controls --------------------------
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ref = v["reference"]["primary_t_per_m"]
    lo = v["reference"]["band"]["min_kg_per_m"] / 1e3
    hi = v["reference"]["band"]["max_kg_per_m"] / 1e3
    ax.axhspan(lo, hi, color="C0", alpha=0.18,
               label=f"repo routes band [{lo:.1f}, {hi:.1f}] t/m")
    ax.axhline(ref, color="C0", lw=1.6, label=f"reference #19234 {ref:.1f} t/m")
    ax.axhline(ref * (1 - tol), color="0.5", lw=0.9, ls="--")
    ax.axhline(ref * (1 + tol), color="0.5", lw=0.9, ls="--",
               label=f"+/-{tol:.0%} tolerance")
    names = [c["name"] for c in v["controls"]]
    vals = [c["value_t_per_m"] for c in v["controls"]]
    colors = ["C2" if c["got"] == "accept" else "C3" for c in v["controls"]]
    ax.barh(range(len(vals)), vals, color=colors, alpha=0.85)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("added mass per unit length [t/m]")
    ax.set_title("green = accepted, red = rejected (all controls behave as predeclared)")
    ax.grid(alpha=0.3, axis="x")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(HERE / "fig3_band_vs_controls.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()