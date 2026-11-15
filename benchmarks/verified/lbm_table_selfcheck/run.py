"""XVAL-LBM-TABLE-DRIVEN-LATTICE-SELFCHECK -- compute stage.

Table-driven lattice-Boltzmann velocity sets (D1Q3, D2Q9, D3Q15, D3Q19, D3Q27)
are declared as pure data and audited by an exact (rational-arithmetic)
self-check; the audit is then attacked with corrupted tables, and the same
lattice-agnostic kernels are run against three closed-form references
(Poiseuille channel, Taylor-Green vortex, equilibrium moments).

Predeclared gates (fixed before the final sweep; see
docs/XVAL-LBM-TABLE-DRIVEN-LATTICE-SELFCHECK.md):

  A1  every shipped table passes every invariant with max |error| == 0
      (exact rational arithmetic, no float tolerance)
  A2  >= 95 % of single-entry weight corruptions are rejected, and the
      rank-4-only corruption (mass + 2nd moment preserved) is rejected by
      ``fourth_moment`` alone
  A3  the table-driven equilibrium reproduces the Navier-Stokes moments
      (mass, momentum, 2nd, 3rd) with exact zero error
  A4  the channel interior curvature recovers g/(2 nu_table) to < 0.1 %
  A5  the fitted no-slip planes are resolution-consistent (|offset(32) -
      offset(16)| < 0.05 lattice units) and the profile then matches the
      closed-form parabola to < 1 % at ny >= 16
  A6  Taylor-Green measures nu within 3 % of cs2 (tau - 1/2) and the field
      within 5 % of the analytic decay
  A7  the 3-D channel reproduces A4 for D3Q19 through the *same* kernels

Run:  python benchmarks/verified/lbm_table_selfcheck/run.py [--fast]
Out:  benchmarks/verified/lbm_table_selfcheck/result.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np  # noqa: E402

from tensorfem.lbm_lattice_tables import (  # noqa: E402
    LatticeTable,
    fit_wall_planes,
    get_table,
    interior_curvature,
    mutate_speed_class,
    mutate_weight,
    selfcheck,
    table_names,
    equilibrium_exact,
    run_channel_poiseuille,
    run_taylor_green,
)

# Qian, D'Humieres & Lallemand (1992) values, typed independently of
# LATTICE_SPECS so that the two routes to the same table can disagree.
LITERATURE_D2Q9 = {
    (0, 0): Fraction(4, 9),
    (1, 0): Fraction(1, 9), (0, 1): Fraction(1, 9),
    (-1, 0): Fraction(1, 9), (0, -1): Fraction(1, 9),
    (1, 1): Fraction(1, 36), (-1, -1): Fraction(1, 36),
    (1, -1): Fraction(1, 36), (-1, 1): Fraction(1, 36),
}
LITERATURE_D3Q19 = {
    (0, 0, 0): Fraction(1, 3),
    **{(1, 0, 0): Fraction(1, 18), (-1, 0, 0): Fraction(1, 18),
       (0, 1, 0): Fraction(1, 18), (0, -1, 0): Fraction(1, 18),
       (0, 0, 1): Fraction(1, 18), (0, 0, -1): Fraction(1, 18)},
    **{(1, 1, 0): Fraction(1, 36), (-1, -1, 0): Fraction(1, 36),
       (1, -1, 0): Fraction(1, 36), (-1, 1, 0): Fraction(1, 36),
       (1, 0, 1): Fraction(1, 36), (-1, 0, -1): Fraction(1, 36),
       (1, 0, -1): Fraction(1, 36), (-1, 0, 1): Fraction(1, 36),
       (0, 1, 1): Fraction(1, 36), (0, -1, -1): Fraction(1, 36),
       (0, 1, -1): Fraction(1, 36), (0, -1, 1): Fraction(1, 36)},
}


def _tables():
    return {name: get_table(name) for name in table_names()}


def audit_tables(tables):
    out = {}
    for name, table in tables.items():
        report = selfcheck(table)
        out[name] = report.as_dict()
    return out


def cross_check_literature(tables):
    out = {}
    for name, ref in (("D2Q9", LITERATURE_D2Q9), ("D3Q19", LITERATURE_D3Q19)):
        table = tables[name]
        generated = {c: w for c, w in zip(table.velocities, table.weights)}
        common = set(generated) & set(ref)
        diffs = [abs(generated[c] - ref[c]) for c in common]
        out[name] = {
            "reference": "Qian, D'Humieres & Lallemand (1992)",
            "entries_generated": len(generated),
            "entries_reference": len(ref),
            "entries_common": len(common),
            "max_abs_weight_diff": str(max(diffs) if diffs else Fraction(1)),
            "identical": bool(
                common == set(generated) == set(ref) and max(diffs, default=Fraction(1)) == 0
            ),
        }
    return out


def _rref(rows, ncols):
    """Exact reduced row echelon form over the rationals."""
    m = [list(r) for r in rows]
    pivots = []
    r = 0
    for cidx in range(ncols):
        piv = next((i for i in range(r, len(m)) if m[i][cidx] != 0), None)
        if piv is None:
            continue
        m[r], m[piv] = m[piv], m[r]
        scale = m[r][cidx]
        m[r] = [x / scale for x in m[r]]
        for i in range(len(m)):
            if i != r and m[i][cidx] != 0:
                f = m[i][cidx]
                m[i] = [a - f * b for a, b in zip(m[i], m[r])]
        pivots.append(cidx)
        r += 1
        if r == len(m):
            break
    return m[:r], pivots


def class_shift_analysis(table: LatticeTable):
    """Is the rank-4 condition independent of mass + 2nd moment?

    Within the subspace of *speed-class symmetric* weight shifts (the only
    shifts that keep opposite symmetry) the table has ``n_classes`` free
    parameters.  ``mass = 1`` and ``sum w c_a c_b = cs2 delta_ab`` give two
    linear constraints; this routine reports whether a non-trivial shift
    exists that satisfies both while breaking the mixed 4th-rank moment
    ``sum w c_x^2 c_y^2 = cs2^2``.
    """
    classes = sorted({sum(ci * ci for ci in c) for c in table.velocities})
    n = len(classes)
    rows = []
    for speed in classes:
        n_speed = sum(1 for c in table.velocities if sum(ci * ci for ci in c) == speed)
        n_x2 = sum(1 for c in table.velocities
                   if sum(ci * ci for ci in c) == speed and c[0] * c[0] == 1)
        n_mixed = sum(1 for c in table.velocities
                      if sum(ci * ci for ci in c) == speed
                      and c[0] * c[0] == 1
                      and len(c) > 1 and c[1] * c[1] == 1)
        rows.append([Fraction(n_speed), Fraction(n_x2), Fraction(n_mixed)])
    mass_row = [r[0] for r in rows]        # sum_i w_i
    second_row = [r[1] for r in rows]      # sum_i w_i c_x^2
    fourth_row = [r[2] for r in rows]      # sum_i w_i c_x^2 c_y^2
    reduced, pivots = _rref([mass_row, second_row], n)
    # free parameters = columns without a pivot
    free = [c for c in range(n) if c not in pivots]
    # for each free parameter, build the solution vector with that param = 1
    baselines = []
    for f in free:
        vec = [Fraction(0)] * n
        vec[f] = Fraction(1)
        for row, piv in zip(reduced, pivots):
            vec[piv] = -row[f]
        baselines.append(vec)
    if table.dim < 2:
        return {
            "speed_classes": classes,
            "n_classes": n,
            "rank_mass_second": len(pivots),
            "free_dimension": len(free),
            "fourth_moment_shift_per_free_direction": [],
            "fourth_moment_independent_of_mass_and_second": False,
            "note": "1-D: the mixed rank-4 moment is not defined",
        }
    fourth_change = [sum(a * b for a, b in zip(fourth_row, v)) for v in baselines]
    independent = any(x != 0 for x in fourth_change)
    return {
        "speed_classes": classes,
        "n_classes": n,
        "rank_mass_second": len(pivots),
        "free_dimension": len(free),
        "fourth_moment_shift_per_free_direction": [str(x) for x in fourth_change],
        "fourth_moment_independent_of_mass_and_second": bool(independent),
    }


def mutation_study(tables, delta=Fraction(1, 1000)):
    """Reject-rate of the self-check against corrupted tables."""
    out = {}
    for name, table in tables.items():
        total = 0
        detected = 0
        only_rank4 = 0
        catching = {}
        for i in range(table.q):
            for sign in (+1, -1):
                bad = mutate_weight(table, i, sign * delta)
                report = selfcheck(bad)
                total += 1
                if not report.passed:
                    detected += 1
                    for key in report.failures:
                        catching[key] = catching.get(key, 0) + 1
                    if report.failures == ["fourth_moment"]:
                        only_rank4 += 1
        out[name] = {
            "perturbations": total,
            "detected": detected,
            "reject_rate": detected / total,
            "detected_by": catching,
            "rank4_only_rejections": only_rank4,
        }

    # a corruption that keeps mass, first and second moments exactly and is
    # only visible through the rank-4 isotropy of the moment tensor
    d = Fraction(1, 1000)
    rank4_only = {}
    algebra = {}
    # For every 2-D/3-D table the mass + 2nd-moment constraints leave one free
    # speed-class direction, and the 4th-rank mixed moment is non-zero along it;
    # these vectors are that direction (derived by the algebra below, verified
    # here by the check set).
    candidate = {
        "D1Q3": None,
        "D2Q9": {0: 4 * d, 1: -2 * d, 2: d},
        "D3Q15": {0: 16 * d, 1: -4 * d, 3: d},
        "D3Q19": {0: 12 * d, 1: -4 * d, 2: d},
        "D3Q27": {1: -d, 2: d, 3: -3 * d / 4},
    }
    for name, table in tables.items():
        algebra[name] = class_shift_analysis(table)
        changes = candidate.get(name)
        if not changes:
            continue
        bad = mutate_speed_class(table, changes)
        report = selfcheck(bad)
        rank4_only[name] = {
            "speed_class_shift": {str(k): str(v) for k, v in changes.items()},
            "failures": report.failures,
            "isotropy_rank": report.isotropy_rank,
            "moment_errors": {c.key: str(c.max_abs_error) for c in report.checks},
            "rejected": not report.passed,
        }
    out["_rank4_only_corruption"] = rank4_only
    out["_class_shift_algebra"] = algebra
    return out


def equilibrium_moments(tables):
    """Exact moment closure of the table-driven equilibrium."""
    rho = Fraction(7, 5)
    out = {}
    for name in ("D2Q9", "D3Q19", "D3Q27"):
        table = tables[name]
        dim = table.dim
        u = tuple(Fraction(k, 20) for k in range(1, dim + 1))
        f = equilibrium_exact(table, rho, u)
        cs2 = table.cs2
        vel = table.velocities
        mass = sum(f)
        mom = [sum(fi * Fraction(c[a]) for fi, c in zip(f, vel)) for a in range(dim)]
        second = [
            sum(fi * Fraction(c[a]) * Fraction(c[b]) for fi, c in zip(f, vel))
            for a in range(dim) for b in range(dim)
        ]
        third = [
            sum(fi * Fraction(c[a]) * Fraction(c[b]) * Fraction(c[c_]) for fi, c in zip(f, vel))
            for a in range(dim) for b in range(dim) for c_ in range(dim)
        ]
        usq = sum(ui * ui for ui in u)
        exp_mass = rho
        exp_mom = [rho * u[a] for a in range(dim)]
        exp_second = [
            cs2 * rho * (1 if a == b else 0) + rho * u[a] * u[b]
            for a in range(dim) for b in range(dim)
        ]
        exp_third = []
        for a in range(dim):
            for b in range(dim):
                for c_ in range(dim):
                    delta = ((1 if a == b else 0) * u[c_]
                             + (1 if a == c_ else 0) * u[b]
                             + (1 if b == c_ else 0) * u[a])
                    exp_third.append(cs2 * rho * delta)
        err = max(
            [abs(mass - exp_mass)]
            + [abs(m - e) for m, e in zip(mom, exp_mom)]
            + [abs(m - e) for m, e in zip(second, exp_second)]
            + [abs(m - e) for m, e in zip(third, exp_third)]
        )
        out[name] = {
            "rho": str(rho),
            "u": [str(x) for x in u],
            "usq": str(usq),
            "max_abs_moment_error": str(err),
            "exact": err == 0,
            "sum_f_equals_rho": str(mass - exp_mass),
        }
    return out


def channel_study(tables, fast: bool):
    table = tables["D2Q9"]
    sweep = [(16, 0.8), (32, 0.8)] if fast else [
        (16, 0.8), (32, 0.8), (64, 0.8), (32, 1.0), (32, 0.65), (32, 1.5)
    ]
    runs = []
    for ny, tau in sweep:
        runs.append(run_channel_poiseuille(
            table, ny=ny, tau=tau, force=4e-5, max_steps=400_000, tol=1e-14,
            check_every=200, force_scheme="guo",
        ))
    # same generic kernels, 3-D table, walls in y
    three_d = run_channel_poiseuille(
        tables["D3Q19"], ny=17, tau=0.8, force=4e-5, max_steps=200_000,
        tol=1e-14, check_every=200, force_scheme="guo",
    )
    d1 = run_channel_poiseuille(
        tables["D1Q3"], ny=17, tau=0.8, force=4e-5, max_steps=200_000,
        tol=1e-14, check_every=200, force_scheme="guo",
    ) if False else None  # D1Q3 has no transverse wall direction (guarded)
    return {"D2Q9_sweep": runs, "D3Q19_channel": three_d, "D1Q3_channel": d1}


def taylorgreen_study(tables, fast: bool):
    table = tables["D2Q9"]
    cases = [(64, 1, 1e-3, 0.65, 400)] if fast else [
        (64, 1, 1e-3, 0.65, 400),
        (64, 2, 1e-3, 0.65, 250),
        (64, 4, 1e-3, 0.65, 120),
        (64, 2, 5e-3, 0.65, 250),
        (64, 2, 1e-3, 0.90, 500),
    ]
    runs = []
    for n, periods, u0, tau, steps in cases:
        runs.append(run_taylor_green(
            table, n=n, u0=u0, tau=tau, steps=steps, probe_every=5, periods=periods,
        ))
    return runs


def gates(audit, mutation, equilibrium, channel, tgv, literature):
    names = list(audit)
    gate_a1 = all(audit[n]["passed"] and all(
        Fraction(c["max_abs_error"]) == 0 for c in audit[n]["checks"]) for n in names)
    rejects = [mutation[n]["reject_rate"] for n in names]
    rank4 = mutation["_rank4_only_corruption"]
    algebra = mutation["_class_shift_algebra"]
    gate_a2 = (min(rejects) >= 0.95
               and all(v["failures"] == ["fourth_moment"] for v in rank4.values())
               and all(v["rejected"] for v in rank4.values())
               and all(algebra[n]["fourth_moment_independent_of_mass_and_second"]
                       for n in algebra if n != "D1Q3"))
    gate_a3 = all(v["exact"] for v in equilibrium.values())
    fine = [r for r in channel["D2Q9_sweep"] if r["tau"] == 0.8 and r["ny"] in (16, 32)]
    gate_a4 = all(r["curvature_rel_err"] < 1e-3 for r in channel["D2Q9_sweep"])
    offsets = {r["ny"]: 0.5 * (abs(r["wall_offset_bot"]) + abs(r["wall_offset_top"]))
               for r in fine}
    offset_drift = abs(offsets[32] - offsets[16]) if len(offsets) == 2 else float("nan")
    gate_a5 = (offset_drift < 0.05
               and all(r["rel_err_profile_fitted"] < 0.01 for r in fine))
    gate_a6 = all(r["nu_rel_err"] < 0.03 and r["field_rel_err"] < 0.05 for r in tgv)
    gate_a7 = channel["D3Q19_channel"]["curvature_rel_err"] < 1e-3
    gate_lit = all(v["identical"] for v in literature.values())
    return {
        "A1_tables_pass_exact": {"passed": bool(gate_a1)},
        "A2_mutation_reject_rate": {
            "passed": bool(gate_a2),
            "min_reject_rate": min(rejects),
            "rank4_only": {k: v["failures"] for k, v in rank4.items()},
            "class_shift_independence": {
                k: v["fourth_moment_independent_of_mass_and_second"]
                for k, v in algebra.items()},
        },
        "A3_equilibrium_moments_exact": {"passed": bool(gate_a3)},
        "A4_channel_curvature": {
            "passed": bool(gate_a4),
            "max_rel_err": max(r["curvature_rel_err"] for r in channel["D2Q9_sweep"]),
        },
        "A5_wall_offset_and_profile": {
            "passed": bool(gate_a5),
            "wall_offset_ny16": offsets.get(16),
            "wall_offset_ny32": offsets.get(32),
            "offset_drift": offset_drift,
            "max_profile_err_fitted": max(r["rel_err_profile_fitted"] for r in fine),
            "max_profile_err_apriori_node_center": max(
                r["rel_err_profile_wall_center"] for r in fine),
            "max_profile_err_apriori_half_way": max(
                r["rel_err_profile_half_way"] for r in fine),
        },
        "A6_taylorgreen": {
            "passed": bool(gate_a6),
            "max_nu_rel_err": max(r["nu_rel_err"] for r in tgv),
            "max_field_rel_err": max(r["field_rel_err"] for r in tgv),
        },
        "A7_3d_channel_same_kernels": {"passed": bool(gate_a7)},
        "L1_literature_table_identity": {"passed": bool(gate_lit)},
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="short sweep for smoke runs")
    parser.add_argument("--out", default=os.path.join(HERE, "result.json"))
    args = parser.parse_args(argv)

    tables = _tables()
    audit = audit_tables(tables)
    literature = cross_check_literature(tables)
    mutation = mutation_study(tables)
    equilibrium = equilibrium_moments(tables)
    channel = channel_study(tables, args.fast)
    tgv = taylorgreen_study(tables, args.fast)
    gate = gates(audit, mutation, equilibrium, channel, tgv, literature)

    result = {
        "anchor": "XVAL-LBM-TABLE-DRIVEN-LATTICE-SELFCHECK",
        "fast": args.fast,
        "tables": {n: {"q": t.q, "dim": t.dim, "cs2": str(t.cs2)} for n, t in tables.items()},
        "audit": audit,
        "literature_cross_check": literature,
        "mutation": mutation,
        "equilibrium_moments": equilibrium,
        "channel": channel,
        "taylorgreen": tgv,
        "gates": gate,
        "all_gates_passed": all(v["passed"] for v in gate.values()),
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    print(json.dumps({k: v for k, v in gate.items()}, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())