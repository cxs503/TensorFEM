"""Applied benchmark *reference-model* selection for the cantilever beam.

Practise the anchor ``XVAL-LBM-BENCHMARK-METHOD-SELECTION`` inside TensorFEM:
before a beam benchmark may quote a reference value, the *reference model* has
to be selected by a predeclared rule, and the selection has to be reproducible
from raw measurements.

Selection problem
-----------------
Cantilever, rectangular section ``b x h``, span ``L``, uniform load ``q``.
Where a beam record says "compared with ``q L**4 / (8 E I)``", it has tacitly
selected the **Euler-Bernoulli (EB)** model as its reference.  That reference
ignores transverse shear.  Is it admissible for the geometry being simulated?

Candidates (cost = number of extra inputs / dofs the reference needs)

================  ====  ================================================
candidate         cost  why it might be chosen
================  ====  ================================================
``EB-closed``      1.0  one scalar formula ``q L^4/(8 E I)``; no ``G``,
                        no ``A``, no shear coefficient
``TIMO-closed``    3.0  adds ``G``, ``A``, ``kappa`` (section-dependent)
``TIMO-fem-n2``    6.0  six dofs assembled and solved
================  ====  ================================================

Reference ("truth") = the Timoshenko closed field, **verified inside this run**
against mesh-refined FEM of the shear-flexible element: the verification is
what keeps the selection non-circular.

Probes (all relative errors against the verified truth)

* ``w_tip``      tip deflection      - the classical benchmark quantity
* ``psi_tip``    tip section rotation - shear-insensitive control probe
                  (``psi' = M/EI`` in both models, so any shear signal here
                  would be a bug, not physics)
* ``w_mid``      mid-span deflection - carries the *largest* shear share,
                  so it is the binding probe for admissibility

Protocol controls
-----------------
Two measurement protocols are compared at every state: **nodal** (probe placed
on a mesh node, so the FEM value is the model value) and **field-sampled** on a
single coarse element (probe interpolated inside the element).  The difference
measures how much discretisation error inflates the *inferred* model deficit.
That inflation is what silently biases selection in published comparisons.

Usage::

    PYTHONPATH=$PWD/src python3 benchmarks/beam_reference_method_selection.py
    PYTHONPATH=$PWD/src python3 benchmarks/beam_reference_method_selection.py --quick
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import torch

from tensorfem.advanced_beam import timoshenko_stiffness
from tensorfem.beam_foundation import eb_beam_stiffness
from tensorfem.method_selection import (
    MethodCandidate,
    certify_selection,
    select_method,
)

HERE = pathlib.Path(__file__).resolve().parent
RECORD_PATH = HERE / "beam_reference_method_selection_record.json"

# ---------------------------------------------------------------------------
# case constants
# ---------------------------------------------------------------------------
E = 210.0e9          # Pa, steel
NU = 0.3
G = E / (2.0 * (1.0 + NU))
KAPPA = 5.0 / 6.0    # repo default for advanced_beam.solve_timoshenko_cantilever
L = 1.0              # m
B = 0.1              # m, in-plane width
Q = 1000.0           # N/m, uniform load

SLENDERNESS = (2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0, 40.0)
QUICK_SLENDERNESS = (3.0, 5.0, 10.0, 20.0)
TOLERANCES = (0.03, 0.15)   # repo certification line; owner unacceptable line
NODAL_MESH = 2              # mid-span is a node, both elements nodal-exact
MID_DOF = 2 * (NODAL_MESH // 2)   # dof of node NODAL_MESH//2 = mid-span
FIELD_MESH = 1              # coarse single-element field sampling


def section(h: float) -> tuple[float, float, float]:
    """(A, I, h) of the ``b x h`` rectangle."""
    area = B * h
    inertia = B * h**3 / 12.0
    return area, inertia, h


def eb_exact_w(x: float, h: float) -> float:
    _, inertia, _ = section(h)
    return Q * x**2 * (6.0 * L**2 - 4.0 * L * x + x**2) / (24.0 * E * inertia)


def eb_exact_psi(x: float, h: float) -> float:
    _, inertia, _ = section(h)
    return Q * x * (3.0 * L**2 - 3.0 * L * x + x**2) / (6.0 * E * inertia)


def timo_exact_w(x: float, h: float) -> float:
    area, _, _ = section(h)
    return eb_exact_w(x, h) + Q * (2.0 * L * x - x**2) / (2.0 * KAPPA * G * area)


def timo_exact_psi(x: float, h: float) -> float:
    # section rotation is shear-free in Timoshenko: psi' = M/EI
    return eb_exact_psi(x, h)


def shear_share_tip(h: float) -> float:
    """``q L^2/(2 kappa G A) / w_tip^Timo`` -- the analytic EB model deficit."""
    area, inertia, _ = section(h)
    bending = Q * L**4 / (8.0 * E * inertia)
    shear = Q * L**2 / (2.0 * KAPPA * G * area)
    return shear / (bending + shear)


# ---------------------------------------------------------------------------
# FEM routes
# ---------------------------------------------------------------------------
def fem_solve(kind: str, h: float, elements: int):
    area, inertia, _ = section(h)
    ndof = 2 * (elements + 1)
    K = torch.zeros((ndof, ndof), dtype=torch.float64)
    f = torch.zeros(ndof, dtype=torch.float64)
    length = L / elements
    ke = (
        eb_beam_stiffness(E, inertia, length)
        if kind == "EB"
        else timoshenko_stiffness(E, inertia, G, area, KAPPA, length)
    )
    for e in range(elements):
        dofs = torch.tensor([2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3])
        K[dofs[:, None], dofs] += ke
        f[dofs] += Q * length * torch.tensor(
            [0.5, length / 12.0, 0.5, -length / 12.0], dtype=torch.float64
        )
    free = torch.arange(2, ndof)
    u = torch.zeros(ndof, dtype=torch.float64)
    u[free] = torch.linalg.solve(K[free[:, None], free], f[free])
    return u


def sampled_w(u, elements: int, x: float) -> float:
    """Hermite interpolation of the deflection field at ``x`` (field protocol)."""
    length = L / elements
    e = min(int(x / length), elements - 1)
    xi = (x - e * length) / length
    w1, t1, w2, t2 = (float(u[2 * e]), float(u[2 * e + 1]),
                      float(u[2 * e + 2]), float(u[2 * e + 3]))
    return (
        (1 - 3 * xi**2 + 2 * xi**3) * w1
        + length * (xi - 2 * xi**2 + xi**3) * t1
        + (3 * xi**2 - 2 * xi**3) * w2
        + length * (-xi**2 + xi**3) * t2
    )


def rel(got: float, ref: float) -> float:
    return abs(got - ref) / abs(ref)


# ---------------------------------------------------------------------------
# 1. reference verification: FEM must reproduce the closed forms
# ---------------------------------------------------------------------------
def verify_references(h: float, mesh_levels=(1, 2, 4, 8)) -> dict:
    """Timoshenko closed field vs mesh-refined shear-flexible FEM (non-circularity)."""
    rows = []
    targets = {
        "w_tip": (timo_exact_w(L, h), lambda u, n: float(u[2 * n])),
        "psi_tip": (timo_exact_psi(L, h), lambda u, n: float(u[2 * n + 1])),
        "w_mid": (timo_exact_w(L / 2, h), lambda u, n: sampled_w(u, n, L / 2)),
    }
    for n in mesh_levels:
        u = fem_solve("TIMO", h, n)
        row = {"elements": n}
        for probe, (ref, take) in targets.items():
            row[probe] = rel(take(u, n), ref)
        rows.append(row)
    worst = max(max(r["w_tip"], r["psi_tip"], r["w_mid"]) for r in rows if r["elements"] >= 2)
    return {
        "slenderness": L / h,
        "closed_form": {
            "w_tip": timo_exact_w(L, h),
            "psi_tip": timo_exact_psi(L, h),
            "w_mid": timo_exact_w(L / 2, h),
        },
        "fem_rel_error": rows,
        "worst_rel_error_refined": worst,
        "verified": worst < 1e-10,
    }


def eb_route_consistency(h: float) -> dict:
    """EB closed form vs EB-FEM nodal values: the two routes must agree."""
    u = fem_solve("EB", h, NODAL_MESH)
    return {
        "w_tip": rel(float(u[2 * NODAL_MESH]), eb_exact_w(L, h)),
        "psi_tip": rel(float(u[2 * NODAL_MESH + 1]), eb_exact_psi(L, h)),
        "w_mid": rel(float(u[MID_DOF]), eb_exact_w(L / 2, h)),
    }


# ---------------------------------------------------------------------------
# 2. per-state probe measurements under the predeclared protocol
# ---------------------------------------------------------------------------
def measure_state(h: float) -> dict:
    area, inertia, _ = section(h)
    # nodal protocol (mid-span is a node of the 2-element mesh)
    u_eb = fem_solve("EB", h, NODAL_MESH)
    u_timo = fem_solve("TIMO", h, NODAL_MESH)
    # field protocol (single element, probe interpolated at L/2)
    u_eb_c = fem_solve("EB", h, FIELD_MESH)
    u_timo_c = fem_solve("TIMO", h, FIELD_MESH)

    truth = {
        "w_tip": timo_exact_w(L, h),
        "psi_tip": timo_exact_psi(L, h),
        "w_mid": timo_exact_w(L / 2, h),
    }
    nodal = {
        "EB-closed": {
            "w_tip": rel(eb_exact_w(L, h), truth["w_tip"]),
            "psi_tip": rel(eb_exact_psi(L, h), truth["psi_tip"]),
            "w_mid": rel(eb_exact_w(L / 2, h), truth["w_mid"]),
        },
        "TIMO-closed": {"w_tip": 0.0, "psi_tip": 0.0, "w_mid": 0.0},
        "TIMO-fem-n2": {
            "w_tip": rel(float(u_timo[2 * NODAL_MESH]), truth["w_tip"]),
            "psi_tip": rel(float(u_timo[2 * NODAL_MESH + 1]), truth["psi_tip"]),
            "w_mid": rel(float(u_timo[MID_DOF]), truth["w_mid"]),
        },
    }
    field = {
        "EB-fem-n1-field": {
            "w_tip": rel(float(u_eb_c[2 * FIELD_MESH]), truth["w_tip"]),
            "psi_tip": rel(float(u_eb_c[2 * FIELD_MESH + 1]), truth["psi_tip"]),
            "w_mid": rel(sampled_w(u_eb_c, FIELD_MESH, L / 2), truth["w_mid"]),
        },
        "TIMO-fem-n1-field": {
            "w_tip": rel(float(u_timo_c[2 * FIELD_MESH]), truth["w_tip"]),
            "psi_tip": rel(float(u_timo_c[2 * FIELD_MESH + 1]), truth["psi_tip"]),
            "w_mid": rel(sampled_w(u_timo_c, FIELD_MESH, L / 2), truth["w_mid"]),
        },
    }
    return {
        "slenderness": L / h,
        "h_m": h,
        "area_m2": area,
        "inertia_m4": inertia,
        "analytic_shear_share_tip": shear_share_tip(h),
        "truth": truth,
        "probes_nodal": nodal,
        "probes_field": field,
        # protocol inflation on the *same* model: field / nodal inferred deficit
        "protocol_inflation_w_mid": (
            field["EB-fem-n1-field"]["w_mid"] / nodal["EB-closed"]["w_mid"]
            if nodal["EB-closed"]["w_mid"] > 0 else float("inf")
        ),
    }


COSTS = {"EB-closed": 1.0, "TIMO-closed": 3.0, "TIMO-fem-n2": 6.0}


def selection_for(state: dict, tolerance: float, probe_set: tuple[str, ...],
                  source: str = "probes_nodal"):
    table = state[source]
    cands = [
        MethodCandidate(
            name=name,
            cost=COSTS.get(name, 6.0),
            error={p: table[name][p] for p in probe_set},
        )
        for name in table
        if name in COSTS
    ]
    record = select_method(cands, tolerance)
    cert = certify_selection(record)
    if not cert["certified"]:
        raise SystemExit(f"selection record failed certification: {cert['mismatches']}")
    return record, cert


# ---------------------------------------------------------------------------
# 3. break-even slenderness, two independent routes
# ---------------------------------------------------------------------------
def closed_form_breakeven(tolerance: float, probe: str = "w_mid") -> float:
    """Analytic ``L/h`` where the EB model deficit on ``probe`` hits ``tolerance``.

    ``shear share / (bending + shear) = tol  =>  (h/L)^2 = tol/(K*(1-tol))``
    with ``K`` the probe's shear-share coefficient.
    """
    k = K_COEFFICIENT[probe]
    s = tolerance / (1.0 - tolerance)
    return math.sqrt(k / s)


K_COEFFICIENT = {
    # s = K * (h/L)^2 evaluated with NU=0.3, kappa=5/6
    "w_tip": 2.0 * (1.0 + NU) / (3.0 * KAPPA),      # 1.04
    "w_mid": 1.4117647058823529 * (1.0 + NU) / KAPPA,  # 2.2024
}


def measured_breakeven(tolerance: float, probe: str = "w_mid",
                       lo: float = 1.5, hi: float = 60.0, iters: int = 60) -> float:
    """Bisection on the *FE-route* EB deficit: no closed form, no ``K`` coefficient.

    Both sides of the ratio come out of ``fem_solve`` (assembly + solve at
    ``NODAL_MESH``, where both elements are nodal-exact), so this route shares
    no algebra with :func:`closed_form_breakeven`.
    """
    def deficit(slenderness: float) -> float:
        h = L / slenderness
        u_eb = fem_solve("EB", h, NODAL_MESH)
        u_ti = fem_solve("TIMO", h, NODAL_MESH)
        index = MID_DOF if probe == "w_mid" else 2 * NODAL_MESH
        return rel(float(u_eb[index]), float(u_ti[index]))

    if deficit(lo) <= tolerance:
        raise ValueError("tolerance not bracketed at the low end")
    if deficit(hi) > tolerance:
        raise ValueError("tolerance not bracketed at the high end")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if deficit(mid) > tolerance:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run(slenderness=SLENDERNESS, tolerances=TOLERANCES) -> dict:
    states = [measure_state(L / ratio) for ratio in slenderness]

    ref_checks = [verify_references(L / ratio) for ratio in slenderness]
    if not all(c["verified"] for c in ref_checks):
        bad = [c["slenderness"] for c in ref_checks if not c["verified"]]
        raise SystemExit(f"reference closed form NOT reproduced by FEM at L/h={bad}")
    eb_checks = [{"slenderness": L / ratio, **eb_route_consistency(L / ratio)}
                 for ratio in slenderness]
    worst_eb = max(max(r["w_tip"], r["psi_tip"], r["w_mid"]) for r in eb_checks)
    if worst_eb > 1e-10:
        raise SystemExit(f"EB closed form and EB-FEM disagree: worst {worst_eb:.3e}")

    # selection tables
    tables = {}
    for tol in tolerances:
        for probe_set, label in (
            (("w_tip", "psi_tip", "w_mid"), "nodal_all_probes"),
            (("w_tip",), "tip_only"),
        ):
            key = f"{label}|tol={tol}"
            rows = []
            for state in states:
                record, _ = selection_for(state, tol, probe_set)
                rows.append({
                    "slenderness": state["slenderness"],
                    "selected": record.selected,
                    "admissible": list(record.admissible),
                    "worst_error": record.worst[record.selected] if record.selected else None,
                    "worst_probe": (record.worst_probe[record.selected]
                                    if record.selected else None),
                    "margin_of_selected": (max(record.margin[record.selected].values())
                                           if record.selected else None),
                })
            breakeven = min(r["slenderness"] for r in rows if r["selected"] == "EB-closed")
            tables[key] = {
                "tolerance": tol,
                "probes": list(probe_set),
                "rows": rows,
                "first_EB_admissible_slenderness": breakeven,
            }

    # break-even cross-check (nodal protocol, binding probe w_mid)
    breakeven = {}
    for tol in tolerances:
        cf = closed_form_breakeven(tol, "w_mid")
        cf_tip = closed_form_breakeven(tol, "w_tip")
        meas = measured_breakeven(tol, "w_mid")
        breakeven[f"tol={tol}"] = {
            "closed_form_w_mid": cf,
            "closed_form_w_tip": cf_tip,
            "measured_bisection_w_mid": meas,
            "rel_diff_between_routes": rel(meas, cf),
            "route_agreement_lt_3pct": rel(meas, cf) < 0.03,
        }

    # protocol sensitivity: same model pair, coarse field sampling
    protocol_rows = []
    for state in states:
        eb_nodal = state["probes_nodal"]["EB-closed"]
        eb_field = state["probes_field"]["EB-fem-n1-field"]
        protocol_rows.append({
            "slenderness": state["slenderness"],
            "w_mid_nodal": eb_nodal["w_mid"],
            "w_mid_field_n1": eb_field["w_mid"],
            "inflation_factor": state["protocol_inflation_w_mid"],
            "w_tip_nodal": eb_nodal["w_tip"],
            "w_tip_field_n1": eb_field["w_tip"],
        })

    # mesh convergence of the field-sampled probe at the binding slenderness
    h_ref = L / 10.0
    conv = []
    for n in (1, 2, 4, 8, 16):
        u = fem_solve("EB", h_ref, n)
        conv.append({
            "elements": n,
            "w_mid_rel_error": rel(sampled_w(u, n, L / 2), timo_exact_w(L / 2, h_ref)),
        })

    return {
        "anchor": "XVAL-LBM-BENCHMARK-METHOD-SELECTION (practice applied in TensorFEM)",
        "case": {
            "geometry": {"L_m": L, "b_m": B, "section": "rectangular b x h, h = L/(L/h)"},
            "material": {"E_Pa": E, "nu": NU, "G_Pa": G, "kappa": KAPPA},
            "load_N_per_m": Q,
            "slenderness_sweep": list(slenderness),
            "tolerances": list(tolerances),
            "probes": {
                "w_tip": "tip deflection",
                "psi_tip": "tip section rotation (shear-insensitive control)",
                "w_mid": "mid-span deflection (largest shear share -> binding)",
            },
        },
        "reference_verification": {"states": ref_checks, "worst_refined_rel_error": max(
            c["worst_rel_error_refined"] for c in ref_checks)},
        "eb_route_consistency": {"states": eb_checks, "worst_rel_error": worst_eb},
        "states": states,
        "selections": tables,
        "breakeven": breakeven,
        "protocol_sensitivity": protocol_rows,
        "field_probe_convergence_at_slenderness_10": conv,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--write", action="store_true", default=True)
    args = ap.parse_args(argv)
    slenderness = QUICK_SLENDERNESS if args.quick else SLENDERNESS
    payload = run(slenderness)
    payload["quick"] = bool(args.quick)
    if args.write:
        RECORD_PATH.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {RECORD_PATH.relative_to(HERE.parent)}")
    for key, block in payload["breakeven"].items():
        print(f"[{key}] break-even L/h: closed-form {block['closed_form_w_mid']:.3f} "
              f"| measured {block['measured_bisection_w_mid']:.3f} "
              f"| rel diff {block['rel_diff_between_routes']:.2e}")
    for key, block in payload["selections"].items():
        print(f"[{key}] first EB-admissible L/h = {block['first_EB_admissible_slenderness']:.3f} "
              f"| probes {block['probes']}")
    infl = [r["inflation_factor"] for r in payload["protocol_sensitivity"]]
    print(f"protocol inflation factor (w_mid, field n=1 / nodal): "
          f"min {min(infl):.3f} max {max(infl):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())