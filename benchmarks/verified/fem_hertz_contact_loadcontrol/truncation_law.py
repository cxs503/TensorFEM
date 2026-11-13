#!/usr/bin/env python3
"""Independent cross-check of the indentation error of the Hertz anchor.

Why this exists
---------------
The load-controlled Hertz certificate (``run.py`` in this directory) reports a
``delta`` error that *saturates* at about -1.0 % when the mesh is refined.  A
saturating error is not discretisation noise, so the certificate alone cannot
say where it comes from.  This script re-derives the same error **without any
contact algorithm**: the exact Hertz surface tractions are prescribed on the same
axisymmetric mesh, the linear system ``K u = f`` is solved once (no ALM, no
active set, no load iteration), and the central deflection is compared with

    delta_Hertz = a^2 / R          (equivalently w(0) = (1/(pi E*)) int p/rho dA)

Consequences that the certificate can then state:

* the contact algorithm (augmented Lagrangian, active set) contributes nothing
  to the indentation error: a linear static solve reproduces it;
* the finite-domain term is the same O(1/D) law measured by the contact solves,
  which cross-validates the extrapolation with an independent code path;
* the residual that survives at D -> infinity is a property of the spatial
  discretisation of the half-space compliance (bilinear axisymmetric quads),
  and it is reported together with its own mesh trend so it is not oversold.

Outputs
-------
``truncation_law.json``            numbers used by the docs/report
``docs/assets/.../hertz_loadcontrol_static_compliance.png``  figure
"""
from __future__ import annotations

import json
import os
import platform
import sys

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "src"))

from tensorfem.hertz_halfspace_loadcontrol import (  # noqa: E402
    assemble_axisymmetric_stiffness,
    graded_depth,
    hertz_load_reference,
    mesh_from_axes,
    surface_ring_areas,
    two_zone_axis,
)

R, E, NU = 10.0, 1.0e5, 0.3
RATIO = 0.05
LN2 = np.log(2.0)


def static_compliance_case(h_rel: float, depth_factor: float, n_outer_band: int = 26) -> dict:
    """Prescribe the exact Hertz tractions on the FE mesh and solve once."""
    e_star = E / (1.0 - NU**2)
    a_ref = RATIO * R
    load = 4.0 * e_star * a_ref**3 / (3.0 * R)
    ref = hertz_load_reference(load, R, E, NU)
    a = ref.contact_radius
    h = h_rel * a
    r_axis = two_zone_axis(3.0 * a, depth_factor * a, int(round(3.0 * a / h)), n_outer_band)
    z_axis = graded_depth(depth_factor * a, 4.0 * a, 24, 16)
    nodes, quads = mesh_from_axes(r_axis, z_axis)
    nz = len(z_axis) - 1
    kmat = assemble_axisymmetric_stiffness(nodes, quads, E, NU)
    top = np.arange(0, len(nodes), nz + 1)
    radii = nodes[top, 0]
    areas = surface_ring_areas(radii)
    cz = 2 * top + 1

    inside = radii <= a
    p = np.where(inside, ref.peak_pressure * np.sqrt(np.clip(1.0 - (radii / a) ** 2, 0.0, None)), 0.0)
    f = np.zeros(2 * len(nodes))
    f[cz] = -p * areas

    bottom = np.arange(nz, len(nodes), nz + 1)
    outer = np.arange(len(nodes) - (nz + 1), len(nodes))
    fixed = np.unique(np.concatenate([2 * bottom, 2 * bottom + 1, 2 * outer,
                                      2 * top[np.isclose(radii, 0.0)]]))
    free = np.setdiff1d(np.arange(2 * len(nodes)), fixed)
    u = np.zeros(2 * len(nodes))
    u[free] = spsolve(csr_matrix(kmat)[free][:, free].tocsc(), f[free])

    w_fe = -float(u[cz[0]])
    return {
        "h_over_a": h_rel,
        "domain_depth_over_a": depth_factor,
        "n_outer_band": n_outer_band,
        "n_elements": int(len(quads)),
        "ndof": int(2 * len(nodes)),
        "prescribed_force_over_P": float(np.sum(p * areas) / load),
        "w_fe_centre": w_fe,
        "delta_hertz": ref.indentation,
        "indentation_error_pct": 100.0 * (w_fe / ref.indentation - 1.0),
        "contact_radius_over_h": float(a / h),
    }


def inverse_depth_fit(rows):
    d = np.array([r["domain_depth_over_a"] for r in rows], float)
    e = np.array([r["indentation_error_pct"] for r in rows], float)
    A = np.stack((np.ones_like(d), 1.0 / d), axis=1)
    coef, *_ = np.linalg.lstsq(A, e, rcond=None)
    resid = e - A @ coef
    return {
        "model": "err_pct(D) = e_inf_pct + c_pct_depth / D",
        "e_inf_pct": float(coef[0]),
        "c_pct_depth": float(coef[1]),
        "max_resid_pp": float(np.abs(resid).max()),
        "points": [{"depth_over_a": float(x), "measured_pct": float(y),
                    "predicted_pct": float(A[i] @ coef)} for i, (x, y) in enumerate(zip(d, e))],
    }


def main() -> int:
    # depth ladder at a fixed surface spacing: isolates the O(1/D) boundary term
    ladder = [static_compliance_case(0.015, d) for d in (24.0, 48.0, 96.0, 192.0)]
    # mesh ladder at a deep domain: isolates the discretisation residual
    mesh = [static_compliance_case(h, 192.0) for h in (0.03, 0.015, 0.0075)]
    # far-field refinement at fixed depth: is the 1/D term the domain size?
    far = [static_compliance_case(0.015, 192.0, 26), static_compliance_case(0.015, 192.0, 52)]

    law = inverse_depth_fit(ladder)
    contact_result = os.path.join(HERE, "result.json")
    contact_law = None
    if os.path.exists(contact_result):
        with open(contact_result) as fh:
            contact_law = json.load(fh).get("truncation_law")

    out = {
        "benchmark": "fem_hertz_contact_loadcontrol",
        "case_pool": "FEM-HERTZ-CONTACT (case pool id 104)",
        "mission": "CASE-FEM-HERTZ-CONTACT",
        "schema": "tensorfem.hertz-truncation-crosscheck.v1",
        "what": "exact Hertz tractions on the FE mesh, single linear solve, no contact algorithm",
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "machine": platform.machine(), "float": "float64"},
        "params": {"R": R, "E": E, "nu": NU, "a_over_R": RATIO,
                   "units": "dimensionless: lengths/R, forces/(E R^2)"},
        "reference": "Hertz 1882; Johnson 1985 Contact Mechanics 4.2 "
                     "DOI:10.1017/cbo9781139171731 (delta = a^2/R)",
        "depth_ladder_h0.015": ladder,
        "depth_law": law,
        "mesh_ladder_d192": mesh,
        "farfield_pair_d192": far,
        "contact_solve_law": contact_law,
        "comparison": {
            "static_truncation_coeff_pct_depth": law["c_pct_depth"],
            "contact_truncation_coeff_pct_depth": (contact_law or {}).get("c_pct_depth"),
            "note": "the two code paths (linear static solve vs load-controlled contact "
                    "solve) share the mesh generator and the stiffness matrix; agreement of "
                    "the truncation coefficient shows the O(1/D) term is the boundary "
                    "condition, not the contact algorithm",
        },
    }
    with open(os.path.join(HERE, "truncation_law.json"), "w") as fh:
        json.dump(out, fh, indent=1)

    print("static compliance, exact Hertz tractions (a/R = 0.05)")
    print(f"  {'h/a':>7} {'D/a':>6} {'elements':>9} {'delta err [%]':>14} {'a/h':>7}")
    for r in ladder + mesh + far:
        print(f"  {r['h_over_a']:7.4f} {r['domain_depth_over_a']:6.1f} {r['n_elements']:9d} "
              f"{r['indentation_error_pct']:+14.4f} {r['contact_radius_over_h']:7.1f}")
    print(f"\n  1/D law: err = {law['e_inf_pct']:+.4f}% "
          f"{law['c_pct_depth']:+.4f}%/D  (max resid {law['max_resid_pp']:.4f} pp)")
    if contact_law:
        print(f"  contact solve law:   e_inf = {contact_law['e_inf_pct']:+.4f}% "
              f"c = {contact_law['c_pct_depth']:+.4f}%/D "
              f"(max resid {contact_law['max_resid_pp']:.4f} pp)")
    e_inf_mesh = [r["indentation_error_pct"] + law["c_pct_depth"] / r["domain_depth_over_a"]
                  for r in mesh]
    print("  mesh trend of the unbounded-domain residual: "
          + ", ".join(f"h/a={r['h_over_a']:g}: {v:+.4f}%" for r, v in zip(mesh, e_inf_mesh)))
    print(f"  far-field band 26 -> 52 columns at D=192a: "
          f"{far[0]['indentation_error_pct']:+.4f}% -> {far[1]['indentation_error_pct']:+.4f}%")

    # ---- figure ---------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        print("matplotlib unavailable, figure skipped:", exc)
        return 0
    figdir = os.path.join(HERE, "..", "..", "..", "docs", "assets", "case-fem-hertz-contact")
    os.makedirs(figdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    d = np.array([r["domain_depth_over_a"] for r in ladder])
    e = np.array([r["indentation_error_pct"] for r in ladder])
    ax.plot(1.0 / d, e, "o-", label="static solve, exact Hertz tractions")
    grid = np.linspace(0.0, (1.0 / d).max() * 1.02, 200)
    ax.plot(grid, law["e_inf_pct"] + law["c_pct_depth"] * grid, "--",
            label=f"1/D law, intercept {law['e_inf_pct']:+.3f}%")
    if contact_law:
        cd = np.array([p[0] for p in contact_law["points"]])
        ce = np.array([p[1] for p in contact_law["points"]])
        ax.plot(1.0 / cd, ce, "s", mfc="none", ms=8, mew=1.5,
                label="contact solve (certificate)")
    md = np.array([1.0 / r["domain_depth_over_a"] for r in mesh])
    ax.plot(md, [r["indentation_error_pct"] for r in mesh], "^", ms=8,
            label="static solve, mesh ladder at D = 192a")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlabel("1 / D   (D = half-space depth / a)")
    ax.set_ylabel("$\\delta$ error vs Hertz [%]")
    ax.set_title("Indentation error is a 1/D boundary term (no contact algorithm involved)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(figdir, "hertz_loadcontrol_static_compliance.png"), dpi=150)
    print("figure written to", os.path.join(figdir, "hertz_loadcontrol_static_compliance.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())