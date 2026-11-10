"""ASTM E399 specimen suite for the AT2 phase-field fracture model.

Cases
-----
ct_lefm   : C(T) energy-release rate of a slit crack vs the ASTM E399 K calibration
ct_peak   : C(T) rigid-pin displacement control, peak load vs ASTM E399 + Irwin(Gc)
seb_peak  : SE(B) three-point bending, peak load vs ASTM E399 SE(B)
sent_peak : single-edge-notched tension, peak stress vs the Griffith anchors
paths     : crack path extracted from the CT / SEB / SENT damage fields
lconv     : peak load as the regularisation length l -> 0 (mesh converges with l)
staggered : alternating (displacement block / phase-field block) vs monolithic Newton

Run ``python benchmarks/phase_field_fracture/run_astm_specimens.py --case all``.
Results (numbers + audit trail) go to ``artifacts/phase_field_fracture/astm_specimens.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tensorfem.phase_field_fracture import PhaseFieldSolver, energies  # noqa: E402
from tensorfem.phase_field_specimens import (  # noqa: E402
    astm_ct_factor,
    astm_ct_k,
    astm_ct_predict_load,
    astm_seb_k,
    astm_seb_predict_load,
    configure_threads,
    ct_setup,
    damage_path,
    fit_limit,
    griffith_stress,
    irwin_k_from_gc,
    ledger_residuals,
    lock_circle_damage,
    peak_load,
    initial_damage_field,
    relax_damage,
    pin_load_forces,
    run_control_path,
    seb_setup,
    sent_critical_stress,
    sent_setup,
)

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "artifacts", "phase_field_fracture")
E, NU, GC = 1000.0, 0.3, 1.0e-3


def _clean(records):
    return [{k: v for k, v in r.items() if k != "newton"} for r in records]


def _ledger_summary(records):
    led = ledger_residuals(records)
    return {"max_residual_percent": float(100.0 * led.max()),
            "mean_residual_percent": float(100.0 * led.mean()),
            "max_irreversibility_violations": int(max(r["irreversibility_violations"]
                                                       for r in records)),
            "all_converged": bool(all(r["converged"] for r in records)),
            "max_newton_iters": int(max(r["newton_iters"] for r in records))}


# ----------------------------------------------------------------------------------
# C1  C(T) energy-release rate vs ASTM E399
# ----------------------------------------------------------------------------------
def case_ct_lefm(alphas=(0.35, 0.45, 0.55, 0.65), l_values=(0.08, 0.04), P=1.0, W=1.0):
    from tensorfem.phase_field_fracture import sharp_crack_energy_release

    out = []
    for l in l_values:
        h = 0.5 * l
        for alpha in alphas:
            specs = [ct_setup(a_over_w=alpha + da / W, W=W, l=l, h=h, E=E, nu=NU, Gc=GC)
                     for da in (-h, 0.0, h)]
            base = specs[1]
            cu, cl = base.meta["hole_centers"]
            r = base.meta["hole_radius"]
            forces, counts = [], []
            for s in specs:
                fu, _ = pin_load_forces(s.nodes, s.elements, s.problem.void_elements, cu, r,
                                        P, (0, 1))
                fl, _ = pin_load_forces(s.nodes, s.elements, s.problem.void_elements, cl, r,
                                        -P, (0, 1))
                forces.append(fu + fl)
                counts.append(int(len(s.problem.void_elements)))
            res = sharp_crack_energy_release([s.problem for s in specs], forces[1], base.fixed, h)
            k_fem = math.sqrt(max(res["G"], 0.0) / (1.0 - NU ** 2) * E)
            k_astm = astm_ct_k(P, base.B, W, alpha)
            out.append({"l": l, "h": h, "a_over_W": alpha,
                        # one removed element per step keeps the finite difference clean
                        "void_counts_ok": bool(counts[0] + 1 == counts[1] == counts[2] - 1),
                        "G": float(res["G"]), "K_fem": float(k_fem),
                        "K_astm": float(k_astm),
                        "error_percent": float(100.0 * (k_fem - k_astm) / k_astm),
                        "elements": int(base.problem.n_elem),
                        "strain_energy": [d["strain_energy"] for d in res["details"]]})
    worst = max(abs(r["error_percent"]) for r in out)
    return {"case": "ct_lefm", "P": P, "W": W, "runs": out,
            "max_abs_error_percent": float(worst),
            "astm_factor_at_0p5": float(astm_ct_factor(0.5))}


# ----------------------------------------------------------------------------------
# phase-field propagation driver (rigid actuator, displacement control)
# ----------------------------------------------------------------------------------
def run_specimen(spec, *, two_sided: bool, n_step: int, delta_max: float,
                 d_lock_circles=(), scheme: str = "monolithic", max_iter: int = 20,
                 tol: float = 1e-9, d_stop: float = 0.95, drop_stop: float = 0.45,
                 profiles: bool = False):
    p = spec.problem
    solver = PhaseFieldSolver(p, fixed=dict(spec.fixed),
                              initial_damage=initial_damage_field(spec))
    solver.fix_damage_nodes(spec.notch, 1.0)
    for c, radius in d_lock_circles:
        solver.fix_damage_nodes(lock_circle_damage(spec.nodes, c, 0.0, radius), 0.0)
    relax_iters, relax_ok, relax_res = relax_damage(solver)
    up = list(spec.control_dofs)
    low = list(spec.control2_dofs)
    values = [delta_max * (k + 1) / n_step for k in range(n_step)]
    sets = []
    for v in values:
        step = {int(d): float(v) for d in up}
        if two_sided:
            step.update({int(d): -float(v) for d in low})
        sets.append(step)
    if scheme == "alternating":
        max_iter = max(max_iter, 400)
    trace = []

    def stop(solver_, rec):
        trace.append({"value": rec["value"], "d_max": rec["d_max_free"],
                      "force": abs(rec["reaction"])})
        if rec["d_max_free"] > d_stop:
            return True
        if not rec["converged"]:
            return True
        forces = [t["force"] for t in trace]
        fmax = max(forces)
        return bool(len(forces) > 2 and fmax > 0 and forces[-1] < (1.0 - drop_stop) * fmax)

    t0 = time.time()
    records = run_control_path(p, solver, sets, measure_dofs=up, scheme=scheme,
                               max_iter=max_iter, tol=tol, abscissa=values, stop=stop)
    elapsed = time.time() - t0
    # the peak is measured on the equilibrated part of the path only
    converged = [r for r in records if r["converged"]] or records
    peak = peak_load(converged)
    peak["n_nonconverged"] = int(sum(1 for r in records if not r["converged"]))
    en = energies(p, solver.u, solver.d, solver.history)
    out = {"records": _clean(records), "peak": peak, "ledger": _ledger_summary(records),
           "relaxation": {"iters": int(relax_iters), "converged": bool(relax_ok),
                          "residual": float(relax_res)},
           "elapsed_s": float(elapsed), "specimen": spec.summary(),
           "d_max_final": float(solver.d.max()),
           "energy_final": {"elastic": en["elastic"], "fracture": en["fracture"]}}
    if scheme == "alternating" and records:
        out["staggered_residual_max"] = float(max(r["newton"][-1] for r in records
                                                  if r.get("newton")))
        out["staggered_iters_max"] = int(max(r["newton_iters"] for r in records))
        out["staggered_residual_history_last"] = [r["newton"] for r in records[-1:]][0]
    if profiles:
        out["field"] = {"nodes": spec.nodes.tolist(), "d": solver.d.tolist()}
    return out, solver, spec


def case_ct_peak(alpha=0.5, l=0.04, n_step=60, delta_max=0.0028, scheme="monolithic"):
    spec = ct_setup(a_over_w=alpha, W=1.0, l=l, h=0.5 * l, E=E, nu=NU, Gc=GC)
    lock = [(c, 1.8 * spec.meta["hole_radius"]) for c in spec.meta["hole_centers"]]
    run, solver, spec = run_specimen(spec, two_sided=True, n_step=n_step, delta_max=delta_max,
                                     d_lock_circles=lock, scheme=scheme, max_iter=20,
                                     profiles=True)
    k_ic = irwin_k_from_gc(GC, E, NU)
    p_pred = astm_ct_predict_load(k_ic, spec.B, spec.W, spec.alpha)
    k_pf = astm_ct_k(run["peak"]["peak"], spec.B, spec.W, spec.alpha)
    path = damage_path(spec.nodes, solver.d, (spec.meta["crack_tip"][0], 0.0), (1.0, 0.0),
                       max(1e-6, spec.W - spec.meta["crack_tip"][0]), n_bands=20)
    run.update({"case": "ct_peak", "a_over_W": spec.alpha, "K_Ic": float(k_ic),
                "P_pred_astm": float(p_pred), "P_peak": run["peak"]["peak"],
                "K_peak": float(k_pf),
                "error_percent": float(100.0 * (run["peak"]["peak"] - p_pred) / p_pred),
                "crack_path": path})
    return run


def case_seb_peak(alpha=0.45, l=0.06, n_step=48, delta_max=0.010, scheme="monolithic"):
    spec = seb_setup(a_over_w=alpha, W=1.0, l=l, h=0.5 * l, E=E, nu=NU, Gc=GC)
    # the roller contacts are idealised as rigid: lock the phase field under the
    # loading roller and the two supports, otherwise the linear-elastic contact
    # singularity nucleates damage at the load point instead of the notch
    lock = [((0.0, spec.W), 0.22 * spec.W),
            ((-0.5 * spec.meta["span"], 0.0), 0.22 * spec.W),
            ((0.5 * spec.meta["span"], 0.0), 0.22 * spec.W)]
    run, solver, spec = run_specimen(spec, two_sided=False, n_step=n_step,
                                     delta_max=delta_max, d_lock_circles=lock,
                                     scheme=scheme, max_iter=20, profiles=True)
    k_ic = irwin_k_from_gc(GC, E, NU)
    p_pred = astm_seb_predict_load(k_ic, spec.B, spec.W, spec.alpha)
    k_pf = astm_seb_k(run["peak"]["peak"], spec.B, spec.W, spec.alpha)
    path = damage_path(spec.nodes, solver.d, (0.0, spec.a), (0.0, 1.0),
                       max(1e-6, spec.W - spec.a), n_bands=16)
    run.update({"case": "seb_peak", "a_over_W": spec.alpha, "K_Ic": float(k_ic),
                "P_pred_astm": float(p_pred), "P_peak": run["peak"]["peak"],
                "K_peak": float(k_pf),
                "error_percent": float(100.0 * (run["peak"]["peak"] - p_pred) / p_pred),
                "crack_path": path})
    return run


def case_sent_peak(alpha=0.3, l=0.04, n_step=60, delta_max=0.0040, scheme="monolithic"):
    spec = sent_setup(a_over_w=alpha, W=1.0, l=l, h=0.5 * l, E=E, nu=NU, Gc=GC)
    run, solver, spec = run_specimen(spec, two_sided=False, n_step=n_step,
                                     delta_max=delta_max, scheme=scheme, max_iter=20,
                                     profiles=True)
    sigma_peak = run["peak"]["peak"] / (spec.B * spec.W)
    k_ic = irwin_k_from_gc(GC, E, NU)
    sigma_fin = sent_critical_stress(GC, E, NU, spec.a, spec.W)
    sigma_inf = griffith_stress(GC, E, NU, spec.a)
    path = damage_path(spec.nodes, solver.d, (spec.a, 0.0), (1.0, 0.0),
                       max(1e-6, spec.W - spec.a), n_bands=20)
    run.update({"case": "sent_peak", "a_over_W": spec.alpha, "K_Ic": float(k_ic),
                "sigma_peak": float(sigma_peak), "sigma_griffith_finite": float(sigma_fin),
                "sigma_griffith_infinite": float(sigma_inf),
                "error_percent_finite": float(100.0 * (sigma_peak - sigma_fin) / sigma_fin),
                "error_percent_infinite": float(100.0 * (sigma_peak - sigma_inf) / sigma_inf),
                "crack_path": path})
    return run


def case_staggered(alpha=0.3, l=0.05, n_step=8, delta_max=0.00095):
    """Alternating vs monolithic on the same (sub-critical to initiating) path.

    ``delta_max`` is 0.8 of the Griffith critical grip displacement
    (``sigma_c * 2H / E'``), so the step sequence stays in the regime where both
    schemes are well conditioned; the drifting behaviour of the alternating loop
    on strongly damaging steps is reported separately as a limitation.
    """
    spec = sent_setup(a_over_w=alpha, W=1.0, l=l, h=0.5 * l, E=E, nu=NU, Gc=GC)
    alt, solver_a, spec_a = run_specimen(spec, two_sided=False, n_step=n_step,
                                         delta_max=delta_max, scheme="alternating",
                                         max_iter=200, drop_stop=1.0)
    spec2 = sent_setup(a_over_w=alpha, W=1.0, l=l, h=0.5 * l, E=E, nu=NU, Gc=GC)
    mon, solver_m, _ = run_specimen(spec2, two_sided=False, n_step=n_step,
                                    delta_max=delta_max, scheme="monolithic",
                                    max_iter=25, drop_stop=1.0)
    ratio = []
    for a, m in zip(alt["records"], mon["records"]):
        ratio.append(abs(a["reaction"] - m["reaction"]) / max(abs(m["reaction"]), 1e-30))
    en_a = energies(spec_a.problem, solver_a.u, solver_a.d, solver_a.history)
    en_m = energies(spec2.problem, solver_m.u, solver_m.d, solver_m.history)
    return {"case": "staggered", "a_over_W": alpha, "l": l,
            "delta_max": delta_max, "n_step": n_step,
            # the residual history is kept out of the cleaned records, so it is
            # taken from the summary fields written by run_specimen
            "alternating": {"records": alt["records"], "ledger": alt["ledger"],
                            "residual_max": float(alt.get("staggered_residual_max", float("nan"))),
                            "iters_max": int(alt.get("staggered_iters_max", -1))},
            "monolithic": {"records": mon["records"], "ledger": mon["ledger"]},
            "reaction_relative_difference_max": float(max(ratio)),
            "energy_relative_difference_total":
                float(abs(en_a["total"] - en_m["total"]) / max(abs(en_a["total"]), 1e-30)),
            "energy_total_alternating": float(en_a["total"]),
            "energy_total_monolithic": float(en_m["total"])}


def case_lconv(l_values=(0.08, 0.05, 0.04), alpha=0.3, n_step=60, delta_max=0.0040):
    runs = []
    finest = min(l_values)
    for l in l_values:
        spec = sent_setup(a_over_w=alpha, W=1.0, l=l, h=0.5 * l, E=E, nu=NU, Gc=GC)
        run, solver, spec = run_specimen(spec, two_sided=False, n_step=n_step,
                                         delta_max=delta_max, max_iter=25,
                                         profiles=(l == finest))
        sigma = run["peak"]["peak"] / (spec.B * spec.W)
        sigma_fin = sent_critical_stress(GC, E, NU, spec.a, spec.W)
        entry = {"l": float(l), "h": float(0.5 * l), "elements": int(spec.problem.n_elem),
                 "sigma_peak": float(sigma), "sigma_griffith": float(sigma_fin),
                 "error_percent": float(100.0 * (sigma - sigma_fin) / sigma_fin),
                 "ledger": run["ledger"], "peak": run["peak"],
                 "elapsed_s": run["elapsed_s"]}
        if l == finest:
            entry["crack_path"] = damage_path(spec.nodes, solver.d, (spec.a, 0.0), (1.0, 0.0),
                                              max(1e-6, spec.W - spec.a), n_bands=20)
            entry["field"] = run["field"]
        runs.append(entry)
    fit = fit_limit([r["l"] for r in runs], [r["sigma_peak"] for r in runs], power=1.0)
    sig_griffith = sent_critical_stress(GC, E, NU, alpha * 1.0, 1.0)
    return {"case": "lconv", "a_over_W": alpha, "runs": runs, "fit": fit,
            "sigma_griffith": float(sig_griffith),
            "limit_error_percent": float(100.0 * (fit["limit"] - sig_griffith) / sig_griffith)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="all")
    ap.add_argument("--out", default=os.path.join(OUT, "astm_specimens.json"))
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    configure_threads(args.threads)
    os.makedirs(OUT, exist_ok=True)
    summary = {"meta": {"E": E, "nu": NU, "Gc": GC, "K_Ic": irwin_k_from_gc(GC, E, NU),
                        "threads": args.threads, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}}
    t0 = time.time()

    def save():
        with open(args.out, "w") as fh:
            json.dump(summary, fh, indent=1, default=float)
        print("written", args.out, flush=True)

    cases = args.case.split(",") if args.case != "all" else ["ct_lefm", "ct_peak", "seb_peak",
                                                             "sent_peak", "staggered", "lconv"]
    for name in cases:
        t1 = time.time()
        if name == "ct_lefm":
            summary["ct_lefm"] = case_ct_lefm()
            print("C1", summary["ct_lefm"]["max_abs_error_percent"], flush=True)
        elif name == "ct_peak":
            summary["ct_peak"] = case_ct_peak()
            print("C2", summary["ct_peak"]["error_percent"], flush=True)
        elif name == "seb_peak":
            summary["seb_peak"] = case_seb_peak()
            print("C3", summary["seb_peak"]["error_percent"], flush=True)
        elif name == "sent_peak":
            summary["sent_peak"] = case_sent_peak()
            print("C4", summary["sent_peak"]["error_percent_finite"], flush=True)
        elif name == "staggered":
            summary["staggered"] = case_staggered()
            print("C5", summary["staggered"]["reaction_relative_difference_max"], flush=True)
        elif name == "lconv":
            summary["lconv"] = case_lconv()
            print("C6", summary["lconv"]["fit"], flush=True)
        summary.setdefault("timing", {})[name] = float(time.time() - t1)
        save()
    summary["timing"]["total"] = float(time.time() - t0)
    save()


if __name__ == "__main__":
    main()