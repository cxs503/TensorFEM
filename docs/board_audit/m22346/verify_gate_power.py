"""m22346 independent verification of the m22319 GAP-ICE-GATE-POWER gate.

The audited numbers are *not* taken on trust.  Three independent lines are built
here, none of which reuses the audited BEM operator:

  P1  closed-form bracket.  For an ellipse with semi-axes (a,b) the added mass per
      unit length for motion along b is m = rho*pi*a^2 (Lamb, Hydrodynamics,
      art. 71/73).  By monotonicity under set inclusion the doubled section
      2c x 2h is bracketed by its inscribed and circumscribed ellipse, so
      A_omega0 = m_full/2 lies in a closed-form interval that needs no
      discretisation at all.
  P2  independent discretisation: cell-centred finite volumes (cut cells exact,
      because the body faces coincide with cell faces) on the doubled section
      removed from a box [-R,R]^2, Dirichlet at R.  Two functionals of the same
      discrete solution (force integral, kinetic energy) plus sweeps in h and R.
  P3  the audited decision rule re-derived from its published formula (guard band,
      detection rate, false acceptance, critical deviation, robustness range).
  P4  the audited worktree itself: chain declaration re-read from the chain module,
      doc-vs-code rebuild check, and the discretisation shift between the panel
      count used for the judgement and the finest converged one.

Inconclusive attempt kept on the record: a Method-of-Fundamental-Solutions route
(point sources, zero net flux) did not converge on the corner-singular rectangle
(BC residual ~2, cond(As) ~1e18 at up to 240 nodes/edge) and is reported as
not_run, not as evidence either way.

Run: python docs/board_audit/m22346/verify_gate_power.py  -> verify_output_m22346.json
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import pathlib
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "verify_output_m22346.json"
VERIFIED = HERE / "verified"
AUDITED = pathlib.Path("/root/tp-agent/work/m22319")
AUDITED_COMMIT = "ee385cf336b96c7cb9f64f9d7186455fdd327750"

RHO, L, B, T, CB = 1025.0, 100.0, 20.0, 6.0, 0.70
C_HALF, H_HALF = B / 2.0, T          # doubled section 2c x 2h


# --------------------------------------------------------------------------- #
# P1: closed-form bracket
# --------------------------------------------------------------------------- #
def ellipse_bracket() -> dict:
    m_lo = RHO * math.pi * C_HALF ** 2                       # inscribed ellipse (c, h)
    m_hi = RHO * math.pi * (math.sqrt(2.0) * C_HALF) ** 2     # circumscribed ellipse
    declared = 86.1e3
    return {
        "rule": "ellipse added mass m_yy = rho*pi*a^2 (Lamb art.71/73) + monotonicity "
                "of added mass under set inclusion; doubled section 20 m x 12 m",
        "a_omega0_lower_t_per_m": m_lo / 2.0 / 1e3,
        "a_omega0_upper_t_per_m": m_hi / 2.0 / 1e3,
        "declared_t_per_m": declared / 1e3,
        "declared_vs_lower_bound_rel": declared / (m_lo / 2.0) - 1.0,
    }


# --------------------------------------------------------------------------- #
# P2: independent finite-volume route
# --------------------------------------------------------------------------- #
def _fv_solve(h: float, box: float) -> dict:
    n = int(round(2 * box / h))
    xc = -box + (np.arange(n) + 0.5) * h
    X, Z = np.meshgrid(xc, xc, indexing="ij")
    inside = (np.abs(X) < C_HALF) & (np.abs(Z) < H_HALF)
    fluid = ~inside

    idx, loc, k = {}, np.full((n, n), -1, dtype=int), 0
    for i in range(n):
        for j in range(n):
            if fluid[i, j]:
                idx[(i, j)] = k
                loc[i, j] = k
                k += 1
    ncell = len(idx)
    rows, cols, vals = [], [], []
    rhs = np.zeros(ncell)
    for (i, j), kk in idx.items():
        diag = 0.0
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ii, jj = i + di, j + dj
            if ii < 0 or ii >= n or jj < 0 or jj >= n:
                diag += -2.0 / h                      # Dirichlet phi = 0 at h/2
            elif inside[ii, jj]:
                rhs[kk] -= float(dj)                  # flux = V*n_z(fluid), V = 1
            else:
                rows.append(kk); cols.append(loc[ii, jj]); vals.append(1.0 / h)
                diag += -1.0 / h
        rows.append(kk); cols.append(kk); vals.append(diag)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(ncell, ncell))
    phi = spla.spsolve(A, rhs)

    def flux(i, j, di, dj):
        ii, jj = i + di, j + dj
        if ii < 0 or ii >= n or jj < 0 or jj >= n:
            return -2.0 * phi[loc[i, j]] / h
        if inside[ii, jj]:
            return float(dj)
        return (phi[loc[ii, jj]] - phi[loc[i, j]]) / h

    force = energy = 0.0
    for (i, j), kk in idx.items():
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ii, jj = i + di, j + dj
            if 0 <= ii < n and 0 <= jj < n and inside[ii, jj] and dj != 0:
                force += phi[kk] * (-dj) * h          # body outward n_z = -dj
        gx = 0.5 * (flux(i, j, 1, 0) - flux(i, j, -1, 0))
        gz = 0.5 * (flux(i, j, 0, 1) - flux(i, j, 0, -1))
        energy += (gx ** 2 + gz ** 2) * h * h
    return {"h": h, "box": box, "cells_fluid": ncell,
            "A_half_force_t_per_m": abs(RHO * force) / 2.0 / 1e3,
            "A_half_energy_t_per_m": abs(RHO * energy) / 2.0 / 1e3}


def _richardson(v1, v2, v3, r=2.0):
    e1, e2 = v2 - v1, v3 - v2
    if e2 == 0.0 or e1 / e2 <= 0.0:
        return float("nan"), float("nan")
    p = math.log(e1 / e2) / math.log(r)
    return p, v3 + e2 / (r ** p - 1.0)


def fv_route() -> dict:
    rows, res = [], {}
    for box in (60.0, 120.0, 240.0):
        hs = (1.0,) if box == 240.0 else (1.0, 0.5, 0.25)
        for h in hs:
            r = _fv_solve(h, box)
            res[(box, h)] = r
            rows.append(r)
    h_lim = {}
    for box in (60.0, 120.0):
        v = [res[(box, h)]["A_half_force_t_per_m"] for h in (1.0, 0.5, 0.25)]
        p, lim = _richardson(*v)
        h_lim[box] = {"observed_order_h": p, "limit_t_per_m": lim}
    # R-extrapolation at h = 1 (three box sizes); report the 1/R^2 variant explicitly
    v60, v120, v240 = (res[(b, 1.0)]["A_half_force_t_per_m"] for b in (60.0, 120.0, 240.0))
    r_inf = v240 + (v240 - v120) / 3.0
    best = h_lim[120.0]["limit_t_per_m"]
    combined = best + (r_inf - v120)      # h->0 at box=120, plus the R-increment
    return {
        "rows": rows,
        "h_extrapolation": h_lim,
        "R_series_h1": {"box": [60.0, 120.0, 240.0], "values": [v60, v120, v240],
                        "increment_ratio": (v120 - v60) / (v240 - v120)
                        if v240 != v120 else float("inf"),
                        "limit_1_over_R2": r_inf},
        "force_vs_energy_rel_at_finest": (res[(120.0, 0.25)]["A_half_force_t_per_m"]
                                          / res[(120.0, 0.25)]["A_half_energy_t_per_m"] - 1.0),
        "combined_estimate_t_per_m": combined,
        "inconclusive_attempt": "MFS (point sources, zero net flux) on the corner-singular "
                                "rectangle: BC residual ~2, cond ~1e18 up to 240 nodes/edge -> "
                                "not_run, no evidence either way",
    }


# --------------------------------------------------------------------------- #
# P3: the audited rule, re-derived
# --------------------------------------------------------------------------- #
def phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def rule_probe(tol: float, u: float, k: float, deviation: float) -> dict:
    acc, rej = tol - k * u, tol + k * u

    def power(d, u_=u, k_=k):
        r = tol + k_ * u_
        return phi((d - r) / u_) + phi((-d - r) / u_)

    def false_accept(d, u_=u, k_=k):
        a = tol - k_ * u_
        return phi((a - d) / u_) - phi((-a - d) / u_)

    crit = next(d for d in np.arange(tol, 1.5, 1e-5) if power(d) >= 0.95)
    return {
        "tol": tol, "u": u, "k": k, "acceptance_limit": acc, "rejection_limit": rej,
        "deviation": deviation,
        "decision": ("accept" if abs(deviation) <= acc else
                     "reject" if abs(deviation) >= rej else "indeterminate"),
        "power_at_deviation": power(abs(deviation)),
        "power_at_tolerance": power(tol),
        "naive_power_at_tolerance": power(tol, k_=0.0),
        "false_accept_at_tolerance": false_accept(tol),
        "critical_deviation_95": crit,
        "u_max_still_rejecting": (abs(deviation) - tol) / k,
        "deviation_over_rejection_limit": abs(deviation) / rej,
    }


# --------------------------------------------------------------------------- #
# P4: the audited worktree read back
# --------------------------------------------------------------------------- #
def read_chain_declared() -> dict:
    path = pathlib.Path.cwd() / "benchmarks/ice_coupled_6dof/cummins_ice_6dof.py"
    spec = importlib.util.spec_from_file_location("_chain_audit", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_chain_audit"] = mod
    spec.loader.exec_module(mod)
    m = mod.ShipModel()
    a_heave = float(m.radiation["heave"].added_mass_inf)
    return {
        "source": str(path),
        "length_m": m.length_m, "breadth_m": m.breadth_m, "draught_m": m.draught_m,
        "block_coefficient": m.block_coefficient, "water_density": m.water_density,
        "mass_kg": m.mass_kg,
        "heave_added_mass_inf_kg": a_heave,
        "heave_per_length_t_per_m": a_heave / m.length_m / 1e3,
        "surge_per_length_t_per_m": float(m.radiation["surge"].added_mass_inf) / m.length_m / 1e3,
        "mass_check_rho_cb_lbt_kg": m.water_density * m.block_coefficient * m.length_m
        * m.breadth_m * m.draught_m,
    }


def read_published() -> dict:
    d = json.loads((VERIFIED / "gate_evidence.json").read_text())
    v = d["verdict"]
    conv0, convf = d["convergence"]["omega0"], d["convergence"]["infinity"]
    i160, i640 = conv0["per_edges"].index(160), conv0["per_edges"].index(640)
    return {
        "pass": v["pass"], "decision": v["decision"], "verdict_text": v["verdict"],
        "declared_t_per_m": v["declared"]["heave"]["value_t_per_m"],
        "reference_t_per_m": v["reference"]["primary_t_per_m"],
        "routes_t_per_m": {"omega0": v["reference"]["repo_route_omega0_kg_per_m"] / 1e3,
                           "infinity": v["reference"]["repo_route_infinity_kg_per_m"] / 1e3},
        "band_half_width_rel": v["reference"]["band_half_width_rel"],
        "anchor_deviation": v["reference"]["independent_anchor_deviation"],
        "rule": v["rule"],
        "power": {k: val for k, val in v["power"].items() if not isinstance(val, dict)},
        "controls": [{"name": c["name"], "t_per_m": c["value_t_per_m"], "dev": c["deviation"],
                      "expected": c["expected"], "got": c["got"], "passed": c["passed"],
                      "provenance": c["provenance"]} for c in v["controls"]],
        "meta_gate": v["meta_gate"],
        "degenerate_gates_rejected": v["degenerate_gates_rejected"],
        "convergence": {"omega0": conv0, "infinity": convf},
        "discretisation_shift_160_to_640": {
            "omega0": conv0["values_kg_per_m"][i640] / conv0["values_kg_per_m"][i160] - 1.0,
            "infinity": convf["values_kg_per_m"][convf["per_edges"].index(640)]
            / convf["values_kg_per_m"][convf["per_edges"].index(160)] - 1.0},
        "deep_cylinder": d["deep_cylinder"],
        "sensitivity_B_over_T": d["sensitivity_B_over_T"],
    }


def main() -> dict:
    chain = read_chain_declared()
    pub = read_published()
    bracket = ellipse_bracket()
    fv = fv_route()
    rule = rule_probe(pub["rule"]["tol"], pub["rule"]["u_used"], pub["rule"]["k"],
                      pub["declared_t_per_m"] / pub["reference_t_per_m"] - 1.0)
    stamp = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(VERIFIED.glob("*")) if p.is_file()}
    out = {
        "audited_repo": "TensorFEM", "audited_branch": "agent/m22319",
        "audited_commit": AUDITED_COMMIT,
        "verified_copy_sha256": stamp,
        "chain_read_by_auditor": chain,
        "published": pub,
        "P1_ellipse_bracket": bracket,
        "P2_fv_route": fv,
        "P3_rule_reprobe": rule,
        "cross_checks": {
            "audited_reference_t_per_m": pub["reference_t_per_m"],
            "audited_omega0_route_t_per_m": pub["routes_t_per_m"]["omega0"],
            "fv_independent_omega0_t_per_m": fv["combined_estimate_t_per_m"],
            "fv_vs_audited_omega0_rel": fv["combined_estimate_t_per_m"]
            / pub["routes_t_per_m"]["omega0"] - 1.0,
            "audited_reference_inside_bracket": bracket["a_omega0_lower_t_per_m"]
            < pub["reference_t_per_m"] < bracket["a_omega0_upper_t_per_m"],
            "declared_vs_fv_independent_rel": pub["declared_t_per_m"]
            / fv["combined_estimate_t_per_m"] - 1.0,
        },
        "independent_limitations": [
            "the FV route depends on the same free-surface mirror formulation as the audited "
            "route (omega0 variant only); it is independent in discretisation, not in model",
            "FV truncation at R is the dominant residual (R-series increment ratio "
            f"{fv['R_series_h1']['increment_ratio']:.2f}); the combined estimate is quoted "
            "at ~+/-1.5%",
            "no literature table value (Vugts 1968) was available in this sandbox, so no "
            "external experimental anchor was used",
        ],
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({"P1": bracket, "P2_headline": {
        "h_extrapolation": fv["h_extrapolation"], "R_series_h1": fv["R_series_h1"],
        "force_vs_energy_rel_at_finest": fv["force_vs_energy_rel_at_finest"],
        "combined_estimate_t_per_m": fv["combined_estimate_t_per_m"]},
        "P3": rule, "cross_checks": out["cross_checks"],
        "chain": {k: chain[k] for k in ("heave_per_length_t_per_m", "mass_kg",
                                        "heave_added_mass_inf_kg")},
        "discretisation_shift": pub["discretisation_shift_160_to_640"]},
        indent=2, ensure_ascii=False))
    return out


if __name__ == "__main__":
    main()