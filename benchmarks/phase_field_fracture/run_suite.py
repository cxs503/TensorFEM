"""Phase-field fracture verification suite (AT2, 2-D plane strain).

Cases
-----
sent_ki : single-edge-notched tension, K_I vs the Tada finite-width solution (C1)
tpb     : ASTM E399 SE(B) three-point bending, fracture load vs K_Ic (C2)
kink    : inclined crack, kink angle vs maximum tangential stress / experiment (C3)
lconv   : compact-band-width l convergence of the apparent G_c (C4)
          plus energy-ledger residual (C5), history irreversibility (C6)
          and Newton convergence order (C7)

Run ``python benchmarks/phase_field_fracture/run_suite.py --case all``.
Results are written to ``artifacts/phase_field_fracture/summary.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tensorfem.phase_field_fracture import (  # noqa: E402
    PhaseFieldProblem,
    PhaseFieldSolver,
    assemble,
    crack_line_elements,
    energies,
    nodes_on_segment,
    problem_strain_energy,
    run_load_path,
    sharp_crack_energy_release,
    structured_quad_mesh,
)

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "artifacts", "phase_field_fracture")


# ----------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------
def opinion_force(nodes, edge_nodes, traction, thickness=1.0):
    """Consistent nodal forces for a uniform traction on a straight edge."""
    f = np.zeros(2 * len(nodes))
    idx = np.asarray(edge_nodes)
    order = np.argsort(nodes[idx, 0] if abs(nodes[idx[0], 0] - nodes[idx[-1], 0]) > 1e-12
                       else nodes[idx, 1])
    idx = idx[order]
    for k in range(len(idx) - 1):
        L = float(np.linalg.norm(nodes[idx[k + 1]] - nodes[idx[k]]))
        f[2 * idx[k]:2 * idx[k] + 2] += 0.5 * L * thickness * np.asarray(traction)
        f[2 * idx[k + 1]:2 * idx[k + 1] + 2] += 0.5 * L * thickness * np.asarray(traction)
    return f


def solve_elastic(problem, load, fixed, n_increment=6, max_iter=25):
    """Incremental dead-load elastic solve with ``d`` frozen at zero."""
    bc = dict(fixed)
    for nd in range(problem.n_nodes):
        bc[2 * problem.n_nodes + nd] = 0.0
    for nd in getattr(problem, "orphan_nodes", []):
        bc[2 * int(nd)] = 0.0
        bc[2 * int(nd) + 1] = 0.0
        bc[2 * problem.n_nodes + int(nd)] = 0.0
    solver = PhaseFieldSolver(problem, fixed=bc, load=np.zeros(2 * problem.n_nodes))
    for k in range(n_increment):
        solver.load = load * (k + 1) / n_increment
        rec = solver.newton_step(max_iter=max_iter, tol=1e-11)
    return solver, rec


def sent_tada(alpha):
    """Tada single-edge-notch tension shape factor (H/W >= 1, error < 1 %)."""
    return 1.12 - 0.231 * alpha + 10.55 * alpha ** 2 - 21.72 * alpha ** 3 + 30.39 * alpha ** 4


def astm_seb(alpha):
    """ASTM E399 SE(B) shape factor, span S = 4W."""
    return 3.0 * math.sqrt(alpha) / (2.0 * (1.0 + 2.0 * alpha) * (1.0 - alpha) ** 1.5) * (
        1.99 - alpha * (1.0 - alpha) * (2.15 - 3.93 * alpha + 2.7 * alpha ** 2))


def mts_kink_angle(alpha_beta):
    """Maximum tangential stress kink angle (rad) for a crack inclined at ``beta``."""
    k2 = 1.0 / math.tan(alpha_beta)          # K_II / K_I for uniaxial tension
    return 2.0 * math.atan2(1.0 - math.sqrt(1.0 + 8.0 * k2 ** 2), 4.0 * k2)


# ----------------------------------------------------------------------------------
# C1  single-edge-notched tension: K_I vs Tada
# ----------------------------------------------------------------------------------
def case_sent_ki(nx=60, a_over_w=0.3, E=1000.0, nu=0.3, sigma=1.0):
    W, H = 1.0, 1.0
    h = W / nx
    a = a_over_w * W

    def build(a_crack):
        # an odd number of rows puts y = 0 at an element centre line, so the slit is
        # one element thick (thickness h) instead of two
        nodes, elems = structured_quad_mesh(0.0, W, -H, H, nx, 2 * nx + 1)
        vid = crack_line_elements(nodes, elems, (0.0, 0.0), (a_crack, 0.0), tol=0.3 * h)
        return PhaseFieldProblem(nodes, elems, E, nu, Gc=1e-3, l=4.0 * h,
                                 thickness=1.0, void_elements=vid), nodes

    problems = [build(a - h)[0], build(a)[0], build(a + h)[0]]
    nodes = build(a)[1]
    top = np.where(np.abs(nodes[:, 1] - H) < 1e-12)[0]
    bot = np.where(np.abs(nodes[:, 1] + H) < 1e-12)[0]
    f = opinion_force(nodes, top, (0.0, sigma)) + opinion_force(nodes, bot, (0.0, -sigma))
    bl = int(np.argmin(nodes[:, 0] ** 2 + (nodes[:, 1] + H) ** 2))
    br = int(np.argmin((nodes[:, 0] - W) ** 2 + (nodes[:, 1] + H) ** 2))
    fixed = {2 * bl: 0.0, 2 * bl + 1: 0.0, 2 * br: 0.0}

    results = []
    for problem in problems:
        solver, rec = solve_elastic(problem, f, fixed)
        U = problem_strain_energy(problem, solver.u, solver.d)
        W_ext = float(f[: 2 * problem.n_nodes] @ solver.u)
        results.append({"U": U, "W": W_ext, "ratio": U / max(W_ext, 1e-30),
                        "residual": rec.residual_norms[-1] if rec.residual_norms else 0.0,
                        "converged": rec.converged})
    potentials = [r["U"] - r["W"] for r in results]
    dPi = (potentials[2] - potentials[0]) / (2.0 * h)
    G = -dPi / 1.0
    E_prime = E / (1.0 - nu ** 2)
    K_fem = math.sqrt(max(G, 0.0) * E_prime)
    alpha = a / W
    K_ana = sigma * math.sqrt(math.pi * a) * sent_tada(alpha)
    return {
        "case": "sent_ki",
        "a_over_W": alpha,
        "elements": int(problems[1].n_elem),
        "G": G,
        "K_fem": K_fem,
        "K_analytic": K_ana,
        "error_percent": 100.0 * (K_fem - K_ana) / K_ana,
        "solves": results,
    }


# ----------------------------------------------------------------------------------
# phase-field propagation cases
# ----------------------------------------------------------------------------------
def plateau_setup(W=1.0, S=4.0, a_over_w=0.45, l=0.02, h=0.02, E=1000.0, nu=0.3, Gc=1e-3):
    """SE(B) specimen: span S, depth W, initial notch a from the bottom mid-span."""
    nx = int(round(S / h))
    ny = int(round(W / h))
    nodes, elems = structured_quad_mesh(-S / 2.0, S / 2.0, 0.0, W, nx, ny)
    a = a_over_w * W
    notch = nodes_on_segment(nodes, (0.0, 0.0), (0.0, a), tol=1e-9)
    problem = PhaseFieldProblem(nodes, elems, E, nu, Gc=Gc, l=l, thickness=1.0)
    # supports: bottom edge under the rollers at x = +-S/2
    bl = int(np.argmin((nodes[:, 0] + S / 2) ** 2 + nodes[:, 1] ** 2))
    br = int(np.argmin((nodes[:, 0] - S / 2) ** 2 + nodes[:, 1] ** 2))
    top = int(np.argmin(nodes[:, 0] ** 2 + (nodes[:, 1] - W) ** 2))
    fixed = {2 * bl: 0.0, 2 * bl + 1: 0.0, 2 * br + 1: 0.0, 2 * top: 0.0}
    return problem, nodes, notch, fixed, top, a


def sent_setup(W=1.0, H=1.0, a_over_w=0.3, l=0.02, h=0.02, E=1000.0, nu=0.3, Gc=1e-3):
    """Square SENT specimen (Miehe geometry): notch from the left edge at mid height."""
    nx = int(round(W / h))
    ny = int(round(2 * H / h))
    nodes, elems = structured_quad_mesh(0.0, W, -H, H, nx, ny)
    a = a_over_w * W
    notch = nodes_on_segment(nodes, (0.0, 0.0), (a, 0.0), tol=1e-9)
    problem = PhaseFieldProblem(nodes, elems, E, nu, Gc=Gc, l=l, thickness=1.0)
    bl = int(np.argmin(nodes[:, 0] ** 2 + (nodes[:, 1] + H) ** 2))
    br = int(np.argmin((nodes[:, 0] - W) ** 2 + (nodes[:, 1] + H) ** 2))
    top = np.where(np.abs(nodes[:, 1] - H) < 1e-12)[0]
    fixed = {2 * bl: 0.0, 2 * bl + 1: 0.0, 2 * br + 1: 0.0}
    return problem, nodes, notch, fixed, top, a


def run_sent_propagation(l, h, a_over_w=0.3, delta=2.0e-5, n_step=30, Gc=1e-3, E=1000.0,
                         nu=0.3, scheme="staggered", d_crit=0.5, checkpoint=None):
    problem, nodes, notch, fixed, top, a = sent_setup(l=l, h=h, a_over_w=a_over_w, Gc=Gc, E=E, nu=nu)
    solver = PhaseFieldSolver(problem, fixed=fixed)
    solver.fix_damage_nodes(notch, 1.0)
    top = np.sort(top[np.argsort(nodes[top, 0])])
    control = [2 * int(n) + 1 for n in top]
    values = [delta * (k + 1) for k in range(n_step)]
    trace = []

    ahead = int(np.argmin(np.abs(nodes[:, 0] - (a + 2.0 * h)) + 2.0 * np.abs(nodes[:, 1])))
    free = np.ones(problem.n_nodes, dtype=bool)
    free[notch] = False

    def stop(solver_, rec):
        d_ahead = float(solver_.d[ahead])
        trace.append({"value": rec["value"], "d_ahead": d_ahead,
                      "d_max_free": float(solver_.d[free].max()),
                      "reaction": rec["reaction"]})
        return d_ahead > d_crit

    records = run_load_path(problem, solver, control, values, scheme=scheme,
                            max_iter=8, tol=1e-8, stop=stop)
    for rec, tr in zip(records, trace):
        rec.update({k: v for k, v in tr.items() if k != "value"})
    P = records[-1]["reaction"]
    return {"problem": problem, "solver": solver, "records": records, "P_crit": P,
            "delta_crit": records[-1]["value"], "a": a, "nodes": nodes, "trace": trace}


def case_tpb(l=0.04, h=0.04, a_over_w=0.45, n_step=25, delta=3.0e-5, Gc=1e-3, E=1000.0, nu=0.3):
    problem, nodes, notch, fixed, top, a = plateau_setup(l=l, h=h, a_over_w=a_over_w,
                                                         Gc=Gc, E=E, nu=nu)
    W = 1.0
    S = 4.0
    solver = PhaseFieldSolver(problem, fixed=fixed)
    solver.fix_damage_nodes(notch, 1.0)
    values = [-delta * (k + 1) for k in range(n_step)]
    free = np.ones(problem.n_nodes, dtype=bool)
    free[notch] = False
    records = run_load_path(problem, solver, [2 * top + 1], values, scheme="staggered",
                            max_iter=15, tol=1e-8,
                            stop=lambda s, r: float(s.d[free].max()) > 0.6)
    P = -records[-1]["reaction"]
    alpha = a / W
    K_pf = P * S / (1.0 * W ** 1.5) * astm_seb(alpha)
    K_ic = math.sqrt(Gc * E / (1.0 - nu ** 2))
    return {"case": "tpb", "P_crit": P, "K_phase_field": K_pf, "K_ic": K_ic,
            "error_percent": 100.0 * (K_pf - K_ic) / K_ic, "a_over_W": alpha,
            "elements": int(problem.n_elem), "records": records}


def case_kink(beta_deg=45.0, a=0.35, l=0.04, h=0.02, n_step=40, delta=3.0e-5,
              Gc=1e-3, E=1000.0, nu=0.3):
    W = H = 1.0
    nx = int(round(W / h))
    ny = int(round(H / h))
    nodes, elems = structured_quad_mesh(0.0, W, 0.0, H, nx, ny)
    beta = math.radians(beta_deg)
    direction = np.array([math.cos(beta), math.sin(beta)])
    p0 = np.array([0.0, 0.25])
    p1 = p0 + a * direction
    notch = nodes_on_segment(nodes, p0, p1, tol=0.6 * h)
    problem = PhaseFieldProblem(nodes, elems, E, nu, Gc=Gc, l=l, thickness=1.0)
    bottom = np.where(nodes[:, 1] < 1e-12)[0]
    bl = int(np.argmin(nodes[:, 0] ** 2 + nodes[:, 1] ** 2))
    br = int(np.argmin((nodes[:, 0] - W) ** 2 + nodes[:, 1] ** 2))
    fixed = {2 * bl: 0.0, 2 * bl + 1: 0.0, 2 * br: 0.0}
    top = np.sort(np.where(nodes[:, 1] > H - 1e-12)[0])
    solver = PhaseFieldSolver(problem, fixed=fixed)
    solver.fix_damage_nodes(notch, 1.0)
    control = [2 * int(n) + 1 for n in top]
    values = [delta * (k + 1) for k in range(n_step)]
    records = run_load_path(problem, solver, control, values, scheme="staggered",
                            max_iter=8, tol=1e-8,
                            stop=lambda s, r: float(s.d[np.setdiff1d(np.arange(problem.n_nodes), notch)].max()) > 0.9)
    # kink angle: fit the damage ridge beyond the initial notch tip
    d = solver.d
    hot = np.where(d > 0.7)[0]
    pts = nodes[hot]
    along = (pts - p1) @ direction
    perp = (pts - p1) @ np.array([-direction[1], direction[0]])
    mask = (along > 0.02) & (along < min(0.25, a))
    if mask.sum() < 5:
        theta = float("nan")
    else:
        A = np.vstack([along[mask], np.ones(mask.sum())]).T
        slope = np.linalg.lstsq(A, perp[mask], rcond=None)[0][0]
        theta = math.atan(slope)
    theta_mts = mts_kink_angle(beta)
    return {"case": "kink", "beta_deg": beta_deg, "theta_phase_field_deg": math.degrees(theta),
            "theta_mts_deg": math.degrees(theta_mts),
            "deviation_deg": abs(math.degrees(theta) - math.degrees(theta_mts)),
            "elements": int(problem.n_elem),
            "n_damage_nodes": int(mask.sum()), "records": records,
            "d_field": d.tolist(), "xy": nodes.tolist()}


def case_lconv(l_values=(0.08, 0.05, 0.032), h_values=None, a_over_w=0.3, n_step=30,
               delta=2.0e-5, Gc=1e-3, E=1000.0, nu=0.3):
    out = []
    for l in l_values:
        h = l / 2.0 if h_values is None else h_values[l]
        run = run_sent_propagation(l=l, h=h, a_over_w=a_over_w, delta=delta, n_step=n_step,
                                   Gc=Gc, E=E, nu=nu)
        records = run["records"]
        delta_c = run["delta_crit"]
        record_trace = run["trace"]
        # applied far-field strain -> stress -> LEFM K -> G
        H = 1.0
        sigma = E / (1.0 - nu ** 2) * delta_c / H
        alpha = run["a"] / 1.0
        K = sigma * math.sqrt(math.pi * run["a"]) * sent_tada(alpha)
        G_app = K ** 2 / (E / (1.0 - nu ** 2))
        ledger = [abs(r["work"] - r["total"]) / max(r["work"], 1e-30) for r in records]
        viol = sum(r["irreversibility_violations"] for r in records)
        out.append({"l": l, "h": h, "elements": int(run["problem"].n_elem),
                    "delta_crit": delta_c, "P_crit": run["P_crit"], "sigma_c": sigma,
                    "G_app": G_app, "Gc": Gc, "G_ratio": G_app / Gc,
                    "deviation_percent": 100.0 * (G_app - Gc) / Gc,
                    "ledger_max_residual_percent": 100.0 * max(ledger),
                    "irreversibility_violations": int(viol),
                    "max_newton_iters": int(max(r["newton_iters"] for r in records)),
                    "newton_history": [r["newton"] for r in records],
                    "trace": record_trace,
                    "records": [{k: v for k, v in r.items() if k != "newton"} for r in records]})
    return {"case": "lconv", "runs": out}


def newton_order(problem, solver, control, values, scheme="monolithic"):
    """Residual history of the last load step for the convergence-order check."""
    rec = run_load_path(problem, solver, control, values, scheme=scheme, max_iter=12, tol=1e-12)
    return rec[-1]["newton"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="all")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    summary = {}
    if args.case in ("sent", "all"):
        summary["sent_ki"] = case_sent_ki()
        print("C1", json.dumps({k: summary["sent_ki"][k] for k in
                                ("G", "K_fem", "K_analytic", "error_percent")}, indent=1))
    if args.case in ("tpb", "all"):
        summary["tpb"] = case_tpb()
        print("C2", json.dumps({k: summary["tpb"][k] for k in
                               ("P_crit", "K_phase_field", "K_ic", "error_percent")}, indent=1))
    if args.case in ("kink", "all"):
        summary["kink"] = case_kink()
        print("C3", json.dumps({k: summary["kink"][k] for k in
                                ("theta_phase_field_deg", "theta_mts_deg", "deviation_deg")}, indent=1))
    if args.case in ("lconv", "all"):
        summary["lconv"] = case_lconv()
        print("C4", json.dumps([{k: r[k] for k in ("l", "G_app", "deviation_percent",
                                                   "ledger_max_residual_percent",
                                                   "irreversibility_violations")}
                                for r in summary["lconv"]["runs"]], indent=1))
    path = args.out or os.path.join(OUT, "summary.json")
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print("written", path)


if __name__ == "__main__":
    main()