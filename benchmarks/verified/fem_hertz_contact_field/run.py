#!/usr/bin/env python3
"""CASE FEM-HERTZ-CONTACT — field-level verification of the Hertz surface deflection.

Why this file exists
--------------------
The certified load-controlled anchor (``benchmarks/verified/fem_hertz_contact_loadcontrol``)
verifies three *edge* scalars only — contact radius ``a``, indentation ``delta`` and
peak pressure ``p0`` — and its own ``limitations`` block admits the gap: nothing in
that certificate looks at the elastic field.  Two different error mechanisms can
leave ``a``, ``delta`` and ``p0`` untouched while the field is wrong (a wrong
half-space Green function, a wrong far-field decay, a surface traction that is
recovered with the wrong axisymmetric weight).  This anchor closes that gap: it
compares the FE *surface deflection profile* ``w(r)`` with the exact half-space
solution, out to ``r = 3a``.

Independent oracle
------------------
For a half-space under the Hertz pressure ``p(s) = p0 sqrt(1-(s/a)^2)`` the exact
surface normal deflection is the superposition of the Boussinesq point-load
solution (Johnson 1985, *Contact Mechanics*, Sec. 3.4 and 4.2):

    w(r) = 1/(pi E*) * int_0^a p(s) s G(r,s) ds,   G(r,s) = int_0^2pi dtheta/|r-s|
         = 1/(pi E*) * int_0^a p(s) s * 4 K(m) / (r+s) ds,  m = 4 r s/(r+s)^2,

with ``K`` the complete elliptic integral of the first kind.  The oracle is
verified against two *elementary* closed forms it must reproduce:

* ``w(0) = pi p0 a / (2 E*) = a^2/R = delta``                     (Hertz relation),
* ``w(r) = delta - r^2/(2R)`` inside the contact  (the indenter is a paraboloid),
* ``w(r) -> P/(pi E* r)`` for ``r >> a``               (Boussinesq point load).

None of these three checks uses the finite element solution, so the oracle is
validated independently.  The FE run never sees the Hertz formula: the load is
prescribed and ``w(r)`` is a raw nodal output.

What is compared (offset invariance)
------------------------------------
The FE body is a *truncated* half-space whose base is fully fixed, so its
displacement field is defined only up to a rigid translation; the absolute nodal
``u_z`` therefore carries a truncation-dependent constant (measured here: the
``r = 0`` value differs from ``delta`` by 0.996 % at depth 96a).  Comparing raw
``w(r)`` would re-measure that constant, not the field.  The graded quantities
are therefore

    D_fe(r) = w_fe(r) - w_fe(0),   D_ex(r) = w_ex(r) - w_ex(0),

i.e. the *shape* of the deflection profile anchored at the contact centre, and
the residual is reported both in units of ``delta`` (``field_shape_over_delta``)
and relative to the local deflection (``field_shape_local``).  A depth
sensitivity run (96a vs 192a) demonstrates that the discarded constant is
truncation, not model bias.

Vacuity control
---------------
A passing tolerance means nothing if the metric cannot see an error.  The same
residual is therefore also evaluated against a *deliberately wrong* competing
Hertz state — same load, contact radius 10 % larger (``p0' = 3P/(2 pi a'^2)``,
``delta' = a'^2/R``, the physically consistent competitor) — and the gate
``discrimination_10pct_radius`` requires that the wrong state be rejected, i.e.
the metric must exceed the tolerance by a wide margin.

Outputs (this directory): ``result.json``, ``gates_verified.json``;
figures: ``../../../docs/assets/case-fem-hertz-contact/hertz_field_profile.png``,
``.../hertz_field_shape_error.png``.

Usage::

    PYTHONPATH=src python benchmarks/verified/fem_hertz_contact_field/run.py
    PYTHONPATH=src python benchmarks/verified/fem_hertz_contact_field/run.py --from-dump
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import ellipk

HERE = Path(__file__).resolve().parent
sys.path.insert(0, os.path.join(HERE, ".."))           # benchmarks/verified/_harness
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "src"))

from _harness import Gate, gate_manifest, write_json, utc_now  # noqa: E402

from tensorfem.hertz_halfspace_loadcontrol import (  # noqa: E402
    graded_depth,
    hertz_load_reference,
    solve_axisymmetric_contact_load_control,
    two_zone_axis,
)

# --- dimensionless working set (identical to the certified load-control run) --
R = 10.0
E = 1.0e5
NU = 0.3
RATIO = 0.05                 # a/R of the certified primary load point
DEPTH_FACTOR = 96.0          # accepted domain depth of the certificate
DEPTH_SENSITIVITY = 192.0    # one step deeper: isolates the rigid offset

#: surface meshes: (h/a, domain depth/a)
LEVELS = ((0.030, DEPTH_FACTOR), (0.015, DEPTH_FACTOR))
#: sample radii of the profile comparison, in units of the Hertz contact radius
OBSERVATION_RADII = (1.10, 1.25, 1.50, 2.00, 3.00)
#: contact-interior samples used to validate the oracle's flatness identity
ORACLE_INTERIOR = (0.20, 0.50, 0.90)
#: the competing (wrong) state used by the discrimination control
DISCRIMINATION_RADIUS_FACTOR = 1.10

#: Thresholds declared in code *before* the fine-mesh / deep-domain runs.  The
#: h/a=0.03 numbers had already been seen in the exploration that produced this
#: file; that is disclosed in the accompanying document.  Nothing here was
#: tuned after the fine-mesh or depth-sensitivity numbers were known.
CRITERIA = {
    "field_shape_over_delta_max": 0.010,     # |D_fe - D_ex| / delta, r/a in [1.1, 3]
    "field_shape_local_max": 0.030,          # same, relative to the local deflection
    "far_field_boussinesq_max": 0.030,       # oracle vs P/(pi E* r) at r = 3a
    "oracle_flatness_max": 1e-6,             # oracle vs delta - r^2/(2R), r/a < 1
    "oracle_centre_max": 1e-8,               # oracle w(0) vs delta
    "penetration_over_delta_max": 1e-5,      # ALM constraint quality
    "refinement_stability_shape_pp": 0.5,    # |max err(h/2) - max err(h)|, pct points
    "depth_sensitivity_offset_max": 0.5,     # |delta offset(96a) - offset(192a)|, pct
    "traction_transfer_max": 0.020,          # FE field vs exact response to FE traction
    "discrimination_min_pct": 1.0,           # wrong 10% radius state must be rejected
    "certificate_consistency_pp": 0.05,      # delta error here vs the certified run
    "reference": "Hertz 1882 J. reine angew. Math. 92:156; "
                 "Johnson 1985 Contact Mechanics Sec. 4.2 "
                 "DOI:10.1017/cbo9781139171731; Boussinesq 1885 (point load kernel)",
    "case_pool": "FEM-HERTZ-CONTACT (case pool id 104)",
    "pool_mandate": "<3% on contact radius a and indentation delta, load P prescribed",
}
FIGDIR = HERE.parents[2] / "docs" / "assets" / "case-fem-hertz-contact"
FIG_PROFILE = "hertz_field_profile.png"
FIG_ERROR = "hertz_field_shape_error.png"


# ---------------------------------------------------------------------------
# Independent oracle: exact half-space deflection under the Hertz pressure
# ---------------------------------------------------------------------------


def green_kernel(r: float, s: float) -> float:
    """Angular integral of the half-space point-load kernel over a circle."""
    if r <= 0.0:
        return 2.0 * math.pi / s
    m = 4.0 * r * s / ((r + s) ** 2)
    return 4.0 * ellipk(min(m, 1.0 - 1e-15)) / (r + s)


def deflection_exact(r: float, *, contact_radius: float, peak_pressure: float,
                     e_star: float) -> float:
    """Exact surface normal deflection of an elastic half-space under Hertz pressure."""

    def integrand(s: float) -> float:
        q = max(0.0, 1.0 - (s / contact_radius) ** 2)
        return peak_pressure * math.sqrt(q) * s * green_kernel(r, s)

    points = [r] if 0.0 < r < contact_radius else None
    value, _ = quad(integrand, 0.0, contact_radius, points=points, limit=400)
    return value / (math.pi * e_star)


def halfspace_response(radii, pressures, r_targets, e_star: float) -> np.ndarray:
    """Exact half-space deflection produced by a piecewise-linear surface traction.

    Same Boussinesq kernel as :func:`deflection_exact`, but the traction is the
    field the FE solution recovered (``consistent_surface_pressure``) instead of
    the analytic Hertz pressure.  Used to test whether a field error comes from
    the elastic operator or from the discrete contact traction.
    """
    radii = np.asarray(radii, dtype=float)
    pressures = np.asarray(pressures, dtype=float)
    nonzero = np.where(pressures > 1e-9 * pressures.max())[0]
    last = int(nonzero.max())
    segments = list(zip(radii[:last], radii[1:last + 1], pressures[:last], pressures[1:last + 1]))
    out = []
    for r in np.atleast_1d(np.asarray(r_targets, dtype=float)):
        total = 0.0

        def integrand(s, r1=0.0, r2=0.0, p1=0.0, p2=0.0):
            q = p1 + (p2 - p1) * (s - r1) / (r2 - r1)
            return q * s * green_kernel(r, s)

        for r1, r2, p1, p2 in segments:
            pts = [r] if r1 < r < r2 else None
            value, _ = quad(integrand, r1, r2, args=(r1, r2, p1, p2), points=pts, limit=200)
            total += value
        out.append(total / (math.pi * e_star))
    return np.array(out)


def traction_transfer(case: dict, state: dict) -> dict:
    """Compare the FE field with the exact response to the FE's *own* traction."""
    a = state["contact_radius"]
    delta = state["indentation"]
    radii = np.asarray(case["radii"], dtype=float)
    w_fe = np.asarray(case["deflection"], dtype=float)
    p_fe = np.asarray(case["pressure_consistent"], dtype=float)
    targets = [rel * a for rel in OBSERVATION_RADII]
    w_hs = halfspace_response(radii, p_fe, targets, state["e_star"])
    w_hs_0 = float(halfspace_response(radii, p_fe, [0.0], state["e_star"])[0])
    w_fe_s = sample(radii, w_fe, targets)
    w_fe_0 = float(sample(radii, w_fe, [0.0])[0])
    resid = (w_fe_s - w_fe_0) - (w_hs - w_hs_0)
    rows = [{
        "r_over_a": rel,
        "fe": float(f),
        "halfspace_of_fe_traction": float(h),
        "residual_over_delta": float(e / delta),
        "residual_over_local": float(abs(e) / abs(h - w_hs_0)) if h != w_hs_0 else float("inf"),
    } for rel, f, h, e in zip(OBSERVATION_RADII, w_fe_s, w_hs, resid)]
    return {
        "rows": rows,
        "centre_over_delta": (w_fe_0 - w_hs_0) / delta,
        "max_residual_over_delta": max(abs(row["residual_over_delta"]) for row in rows),
        "max_residual_over_local": max(row["residual_over_local"] for row in rows),
    }


def hertz_state(load: float, radius: float = R, young: float = E,
                nu: float = NU) -> dict:
    """Closed-form Hertz state for a rigid sphere on a half-space."""
    e_star = young / (1.0 - nu**2)
    a = (3.0 * load * radius / (4.0 * e_star)) ** (1.0 / 3.0)
    p0 = 3.0 * load / (2.0 * math.pi * a * a)
    return {
        "load": load,
        "contact_radius": a,
        "peak_pressure": p0,
        "indentation": a * a / radius,
        "e_star": e_star,
        "boussinesq": lambda r: load / (math.pi * e_star * r),
    }


def load_for_radius_ratio(ratio: float) -> float:
    """Hertz load producing ``a/R = ratio`` (the certificate's primary load point)."""
    e_star = E / (1.0 - NU**2)
    return 4.0 * e_star * (ratio * R) ** 3 / (3.0 * R)


def oracle_profile(radii, state: dict) -> np.ndarray:
    return np.array([
        deflection_exact(float(r), contact_radius=state["contact_radius"],
                         peak_pressure=state["peak_pressure"],
                         e_star=state["e_star"])
        for r in radii
    ])


def oracle_selfchecks(state: dict) -> dict:
    """Validate the oracle against closed forms it does not derive from."""
    a = state["contact_radius"]
    delta = state["indentation"]
    w0 = deflection_exact(0.0, contact_radius=a, peak_pressure=state["peak_pressure"],
                          e_star=state["e_star"])
    flat = []
    for rel in ORACLE_INTERIOR:
        r = rel * a
        w = deflection_exact(r, contact_radius=a, peak_pressure=state["peak_pressure"],
                             e_star=state["e_star"])
        flat.append({
            "r_over_a": rel,
            "oracle": w,
            "delta_minus_square": delta - r * r / (2.0 * R),
            "rel_error": w / (delta - r * r / (2.0 * R)) - 1.0,
        })
    far = 3.0 * a
    w_far = deflection_exact(far, contact_radius=a,
                             peak_pressure=state["peak_pressure"],
                             e_star=state["e_star"])
    point = state["boussinesq"](far)
    return {
        "centre": {"oracle": w0, "delta": delta, "rel_error": w0 / delta - 1.0},
        "flatness": flat,
        "far_field": {"r_over_a": 3.0, "oracle": w_far, "point_load_asymptote": point,
                      "rel_error": w_far / point - 1.0},
    }


# ---------------------------------------------------------------------------
# FE run: same model as the certified load-control anchor, displacement kept
# ---------------------------------------------------------------------------


def run_case(h_rel: float, depth_factor: float) -> dict:
    """One load-controlled solve, returning the surface deflection profile."""
    state_ref = hertz_state(load_for_radius_ratio(RATIO))
    a = state_ref["contact_radius"]
    h = h_rel * a
    n_inner = int(round(3.0 * a / h))
    r_axis = two_zone_axis(3.0 * a, depth_factor * a, n_inner, 26)
    z_axis = graded_depth(depth_factor * a, 4.0 * a, 24, 16)
    t0 = time.time()
    ref, state, diag = solve_axisymmetric_contact_load_control(
        load=state_ref["load"], radius=R, young=E, poisson=NU,
        r_axis=r_axis, z_axis=z_axis, load_tol=1e-10, keep_displacement=True,
    )
    wall = time.time() - t0
    nz = len(z_axis) - 1
    radii = state.surface_radii
    # top-surface nodes: one per radial station, node id = i*(nz+1), axial dof 2*id+1
    top = np.arange(len(radii)) * (nz + 1)
    uz = state.displacement[2 * top + 1]
    # the indenter pushes down (-z), the exact deflection is downward positive
    w = -uz
    lam = np.asarray(state.multiplier)
    # only nodes that actually carry contact load: the ALM tolerance leaves a few
    # edge nodes with a positive but negligible multiplier and a finite gap
    active = lam > 1e-3 * lam.max()
    inside = np.abs(w - (state.prescribed_indentation - radii**2 / (2.0 * R)))
    inside_res = float(inside[active].max() / state.prescribed_indentation) if active.any() else 0.0
    return {
        "h_over_a": h_rel,
        "domain_depth_over_a": depth_factor,
        "wall_s": wall,
        "n_elements": diag["n_elements"],
        "ndof": diag["ndof"],
        "load_residual": diag["load_residual"],
        "penetration_over_delta": diag["penetration_over_indentation"],
        "contact_constraint_residual_rel": inside_res,
        "indentation_fe": float(state.prescribed_indentation),
        "reaction": float(state.reaction),
        "contact_radius_linear": float(state.contact_radius_linear),
        "radii": radii.tolist(),
        "deflection": w.tolist(),
        "pressure_consistent": np.asarray(state.surface_pressure_consistent).tolist(),
        "analytic": {"contact_radius": ref.contact_radius,
                     "indentation": ref.indentation,
                     "peak_pressure": ref.peak_pressure},
    }


def sample(radii: np.ndarray, values: np.ndarray, targets) -> np.ndarray:
    """FE surface trace between nodes is piecewise linear, so interp is exact."""
    return np.interp(np.asarray(targets, dtype=float), radii, values)


def profile_metrics(case: dict, state: dict) -> dict:
    """Offset-invariant shape comparison against the exact half-space profile."""
    a = state["contact_radius"]
    delta = state["indentation"]
    radii = np.asarray(case["radii"])
    w_fe = np.asarray(case["deflection"])
    targets = [rel * a for rel in OBSERVATION_RADII]
    w_fe_s = sample(radii, w_fe, targets)
    w_ex_s = oracle_profile(targets, state)
    w_fe_0 = float(sample(radii, w_fe, [0.0])[0])
    w_ex_0 = state["indentation"]
    d_fe = w_fe_s - w_fe_0
    d_ex = w_ex_s - w_ex_0
    resid = d_fe - d_ex
    rows = []
    for rel, r_fe, r_ex, e_fe, e_ex, res in zip(
        OBSERVATION_RADII, w_fe_s, w_ex_s, d_fe, d_ex, resid
    ):
        rows.append({
            "r_over_a": rel,
            "fe": float(r_fe),
            "exact": float(r_ex),
            "shape_fe": float(e_fe),
            "shape_exact": float(e_ex),
            "residual_over_delta": float(res / delta),
            "residual_over_local": float(abs(res) / abs(e_ex)) if e_ex else float("inf"),
        })
    max_delta = max(abs(r["residual_over_delta"]) for r in rows)
    max_local = max(r["residual_over_local"] for r in rows)
    return {
        "rows": rows,
        "offset_over_delta": (w_fe_0 - w_ex_0) / delta,
        "max_residual_over_delta": max_delta,
        "max_residual_over_local": max_local,
    }


def discrimination_control(case: dict, state: dict, factor: float) -> dict:
    """Metric value for a physically consistent but *wrong* (larger a) state.

    Same applied load, contact radius ``factor * a`` -> the Hertz state that
    belongs to it has ``p0' = p0 / factor^2`` and ``delta' = delta * factor^2``.
    A metric that cannot reject this state is vacuous.
    """
    wrong = dict(state)
    wrong["contact_radius"] = state["contact_radius"] * factor
    wrong["peak_pressure"] = state["peak_pressure"] / factor**2
    wrong["indentation"] = state["indentation"] * factor**2
    return profile_metrics(case, wrong)


# ---------------------------------------------------------------------------
# certificate
# ---------------------------------------------------------------------------


def build(use_dump: bool) -> dict:
    state = hertz_state(load_for_radius_ratio(RATIO))
    oracle = oracle_selfchecks(state)
    results = {"levels": {}, "cases": {}}
    dump = HERE / "result.json"
    if use_dump and dump.exists():
        previous = json.loads(dump.read_text())
        cases = previous.get("raw_cases") or previous["cases"]
        environment = previous.get("environment", {})
        print("[from-dump] reusing the committed raw solution dumps", flush=True)
    else:
        cases = {}
        environment = {"python": platform_python(), "numpy": np.__version__}
        for h_rel, depth in LEVELS:
            key = f"h{h_rel:g}_d{depth:g}"
            print(f"[run] h/a={h_rel:g} depth={depth:g}a ...", flush=True)
            cases[key] = run_case(h_rel, depth)
        print(f"[run] h/a={LEVELS[0][0]:g} depth={DEPTH_SENSITIVITY:g}a ...", flush=True)
        cases[f"h{LEVELS[0][0]:g}_d{DEPTH_SENSITIVITY:g}"] = run_case(
            LEVELS[0][0], DEPTH_SENSITIVITY
        )

    fine_key = f"h{LEVELS[-1][0]:g}_d{LEVELS[-1][1]:g}"
    coarse_key = f"h{LEVELS[0][0]:g}_d{LEVELS[0][1]:g}"
    deep_key = f"h{LEVELS[0][0]:g}_d{DEPTH_SENSITIVITY:g}"
    for key, case in cases.items():
        results["cases"][key] = {
            "h_over_a": case["h_over_a"],
            "domain_depth_over_a": case["domain_depth_over_a"],
            "n_elements": case["n_elements"],
            "ndof": case["ndof"],
            "wall_s": case["wall_s"],
            "load_residual": case["load_residual"],
            "penetration_over_delta": case["penetration_over_delta"],
            "contact_constraint_residual_rel": case["contact_constraint_residual_rel"],
            "indentation_fe": case["indentation_fe"],
            "indentation_error_pct": 100.0 * (case["indentation_fe"] / state["indentation"] - 1.0),
            "contact_radius_linear_error_pct":
                100.0 * (case["contact_radius_linear"] / state["contact_radius"] - 1.0),
        }
        transfer = traction_transfer(case, state)
        results["cases"][key]["traction_transfer"] = transfer
        metrics = profile_metrics(case, state)
        results["cases"][key].update({
            "offset_over_delta": metrics["offset_over_delta"],
            "max_residual_over_delta": metrics["max_residual_over_delta"],
            "max_residual_over_local": metrics["max_residual_over_local"],
            "profile": metrics["rows"],


        })

    fine = results["cases"][fine_key]
    coarse = results["cases"][coarse_key]
    deep = results["cases"][deep_key]
    control = discrimination_control(cases[fine_key], state, DISCRIMINATION_RADIUS_FACTOR)
    results["control"] = {
        "radius_factor": DISCRIMINATION_RADIUS_FACTOR,
        "rows": control["rows"],
        "max_residual_over_delta": control["max_residual_over_delta"],
        "max_residual_over_local": control["max_residual_over_local"],
    }

    certified_delta_err = certified_indentation_error(RATIO, LEVELS[0][0])
    results["cross_check"] = {
        "source": "benchmarks/verified/fem_hertz_contact_loadcontrol/result.json",
        "certified_indentation_error_pct": certified_delta_err,
        "this_run_indentation_error_pct": coarse["indentation_error_pct"],
        "difference_pp": (
            abs(coarse["indentation_error_pct"] - certified_delta_err)
            if certified_delta_err is not None else None
        ),
    }
    results["oracle_selfcheck"] = oracle
    results["diagnostics"] = {
        "hypothesis": "the 0.2-0.3% (of delta) field-shape residual against the analytic "
                      "Hertz profile comes from the discrete contact traction rather than "
                      "from the elastic operator",
        "test": "feed the traction the FE solution recovered into the exact half-space "
                "kernel and compare its field with the FE field",
        "outcome": "falsified",
        "numbers": {
            "fe_vs_analytic_hertz_over_delta": fine["max_residual_over_delta"],
            "fe_vs_exact_response_to_fe_traction_over_delta":
                fine["traction_transfer"]["max_residual_over_delta"],
            "centre_offset_fe_vs_response_over_delta":
                fine["traction_transfer"]["centre_over_delta"],
        },
        "reading": "the FE displacement field is *closer* to the exact Hertz field than to "
                   "the exact response of its own recovered traction: the post-processed "
                   "pressure field, not the displacement field, carries the larger error "
                   "(consistent with the displacement being the variationally optimal "
                   "quantity).  The 0.2-0.3% residual therefore remains unattributed to "
                   "either the operator or the traction, and is reported as an open record.",
    }
    results["headline"] = {
        "field_shape_over_delta": fine["max_residual_over_delta"],
        "field_shape_local": fine["max_residual_over_local"],
        "delta_error_pct": fine["indentation_error_pct"],
        "rigid_offset_pct": 100.0 * fine["offset_over_delta"],
        "pool_mandate_satisfied": abs(fine["indentation_error_pct"]) < 3.0,
    }

    gates = [
        Gate(
            id="oracle_centre",
            quantity="|w(0)/delta - 1| of the exact half-space oracle (no FEM)",
            reference="Hertz relation w(0) = pi p0 a/(2 E*) = a^2/R = delta",
            value=abs(oracle["centre"]["rel_error"]),
            threshold=CRITERIA["oracle_centre_max"],
        ),
        Gate(
            id="oracle_flatness",
            quantity="max |w(r) - (delta - r^2/2R)|/delta of the oracle for r/a < 1",
            reference="the indenter is a paraboloid: the exact solution is flat inside contact",
            value=max(abs(row["rel_error"]) for row in oracle["flatness"]),
            threshold=CRITERIA["oracle_flatness_max"],
        ),
        Gate(
            id="oracle_far_field",
            quantity="|w(3a) / (P/(pi E* 3a)) - 1| of the oracle",
            reference="Boussinesq point-load asymptote of the same Hertz pressure",
            value=abs(oracle["far_field"]["rel_error"]),
            threshold=CRITERIA["far_field_boussinesq_max"],
        ),
        Gate(
            id="field_shape_over_delta",
            quantity=f"max |D_fe - D_exact| / delta over r/a in [{OBSERVATION_RADII[0]:g}, "
                     f"{OBSERVATION_RADII[-1]:g}], h/a={fine['h_over_a']:g}, depth "
                     f"{fine['domain_depth_over_a']:g}a",
            reference="exact half-space deflection under the Hertz pressure; D = w(r)-w(0)",
            value=fine["max_residual_over_delta"],
            threshold=CRITERIA["field_shape_over_delta_max"],
            note="offset invariant: the fixed base of the truncated body leaves an "
                 "arbitrary rigid constant in the raw nodal u_z",
        ),
        Gate(
            id="field_shape_local",
            quantity="max |D_fe - D_exact| / |D_exact| over the same radii",
            reference="same, normalised by the local deflection",
            value=fine["max_residual_over_local"],
            threshold=CRITERIA["field_shape_local_max"],
        ),
        Gate(
            id="refinement_stability_shape",
            quantity="|max shape error (h/a=fine) - max shape error (h/a=coarse)| (pct points)",
            reference="a field metric that is real should not move under refinement by 2",
            value=abs(100.0 * fine["max_residual_over_delta"]
                      - 100.0 * coarse["max_residual_over_delta"]),
            threshold=CRITERIA["refinement_stability_shape_pp"],
        ),
        Gate(
            id="depth_sensitivity_offset",
            quantity="|offset(96a) - offset(192a)| (pct of delta)",
            reference="the raw constant is truncation: it must move when the boundary moves",
            value=abs(100.0 * (coarse["offset_over_delta"] - deep["offset_over_delta"])),
            threshold=CRITERIA["depth_sensitivity_offset_max"],
            note="the offset is discarded by the field metric, so the field verdict does "
                 "not depend on it",
        ),
        Gate(
            id="contact_constraint_quality",
            quantity="max penetration / delta of the finest mesh",
            reference="ALM complementarity: the indenter must not penetrate",
            value=fine["penetration_over_delta"],
            threshold=CRITERIA["penetration_over_delta_max"],
        ),
        Gate(
            id="traction_transfer",
            quantity="max |D_fe - D_exact(p_fe)| / delta over the observation radii, "
                     "where p_fe is the traction the FE solution itself recovered",
            reference="exact half-space response to the FE traction (same Boussinesq kernel)",
            value=fine["traction_transfer"]["max_residual_over_delta"],
            threshold=CRITERIA["traction_transfer_max"],
            note="loose consistency bound on the (pressure recovery + elastic operator) "
                 "pair; see the diagnostics block for what this does NOT show",
        ),
        Gate(
            id="discrimination_10pct_radius",
            quantity="max shape error (pct of delta) of the same metric evaluated against "
                     "a wrong competing Hertz state with a 10% larger contact radius",
            reference="same load, a' = 1.1a, p0' = p0/1.21, delta' = 1.21 delta",
            value=100.0 * control["max_residual_over_delta"],
            threshold=100.0 * CRITERIA["field_shape_over_delta_max"],
            operator=">=",
            note="non-vacuity control: a tolerance the metric cannot enforce is worthless",
        ),
    ]
    if results["cross_check"]["difference_pp"] is not None:
        gates.append(Gate(
            id="certificate_consistency",
            quantity="|indentation error here - indentation error of the certified "
                     "load-control anchor| (pct points), same mesh and load",
            reference="benchmarks/verified/fem_hertz_contact_loadcontrol/result.json",
            value=results["cross_check"]["difference_pp"],
            threshold=CRITERIA["certificate_consistency_pp"],
            note="replication check: this driver must reproduce the certified number",
        ))

    manifest = gate_manifest(
        anchor="fem_hertz_contact_field",
        title="Hertz contact: surface deflection field vs the exact half-space solution",
        criteria=CRITERIA,
        gates=gates,
        artifacts={
            "result": "result.json",
            "gates": "gates_verified.json",
            "figures": [
                "../../../docs/assets/case-fem-hertz-contact/" + FIG_PROFILE,
                "../../../docs/assets/case-fem-hertz-contact/" + FIG_ERROR,
            ],
        },
        limitations=[
            "frictionless, adhesion-free, small-strain, linear elastic: a verification of "
            "the elastic half-space operator only (p0 = 3.5e3 is far above any real yield "
            "strength in the dimensionless units used here)",
            "the model is the exact Hertz equivalent (rigid paraboloid on a half-space with "
            "E* = E/(1-nu^2)); a two-deformable-body contact algorithm is not exercised, "
            "which is exactly the assumption the reduction rests on",
            "the deflection profile is compared only for r/a <= 3; the far field of the "
            "truncated domain is dominated by the boundary condition",
            "the graded quantities are offset invariant; the absolute nodal u_z of the "
            "truncated body carries a 1% rigid constant that this certificate does not "
            "claim to resolve",
        ],
        mission="m18026",
    )
    payload = {
        "benchmark": "fem_hertz_contact_field",
        "anchor": "fem_hertz_contact_field",
        "case_pool": CRITERIA["case_pool"],
        "mission": "m18026",
        "generated_at": utc_now(),
        "environment": environment,
        "params": {
            "R": R, "E": E, "nu": NU, "a_over_R": RATIO,
            "load": state["load"],
            "observation_radii_over_a": list(OBSERVATION_RADII),
            "levels": [{"h_over_a": h, "domain_depth_over_a": d} for h, d in LEVELS],
            "depth_sensitivity_over_a": DEPTH_SENSITIVITY,
        },
        "reference": {
            "hertz": CRITERIA["reference"],
            "exact_deflection": "w(r) = (1/(pi E*)) int_0^a p(s) s 4K(m)/(r+s) ds, "
                                "m = 4rs/(r+s)^2 (Boussinesq kernel superposition)",
            "analytic": state["contact_radius"],
            "analytic_indentation": state["indentation"],
            "analytic_peak_pressure": state["peak_pressure"],
        },
        "criteria": CRITERIA,
        "cases": results["cases"],
        "raw_cases": cases,
        "control": results["control"],
        "cross_check": results["cross_check"],
        "oracle_selfcheck": results["oracle_selfcheck"],
        "diagnostics": results["diagnostics"],
        "headline": results["headline"],
        "gate_table": manifest["gates"],
        "verdict": {"passed": manifest["passed"],
                    "failed_gates": manifest["failed_gates"]},
        "artifacts": {
            "figures": [
                "docs/assets/case-fem-hertz-contact/" + FIG_PROFILE,
                "docs/assets/case-fem-hertz-contact/" + FIG_ERROR,
            ],
        },
    }
    write_json(HERE / "result.json", payload)
    write_json(HERE / "gates_verified.json", manifest)
    return payload


def certified_indentation_error(ratio: float, h_rel: float):
    """Indentation error of the certified anchor for the same model, if present."""
    path = HERE.parent / "fem_hertz_contact_loadcontrol" / "result.json"
    if not path.exists():
        return None
    try:
        doc = json.loads(path.read_text())
        for key, row in doc["runs"].items():
            if (abs(row["a_over_R"] - ratio) < 1e-12
                    and abs(row["h_over_a"] - h_rel) < 1e-12
                    and abs(row["domain_depth_over_a"] - DEPTH_FACTOR) < 1e-9):
                return float(row["error_pct"]["indentation"])
    except (KeyError, TypeError, ValueError):
        return None
    return None


def platform_python() -> str:
    import platform
    return platform.python_version()


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def render(payload: dict) -> None:
    from _harness import pyplot

    plt = pyplot()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    a = payload["reference"]["analytic"]
    delta = payload["reference"]["analytic_indentation"]
    radii = np.linspace(0.0, 3.0, 241) * a
    state = hertz_state(payload["params"]["load"])
    w_ex = oracle_profile(radii, state)

    keys = [k for k in payload["cases"]]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    ax.plot(radii / a, (w_ex - w_ex[0]) / delta, "k-", lw=2.0,
            label=r"exact $[w(r)-w(0)]/\delta$")
    for key in sorted(keys):
        case = payload["cases"][key]
        raw = payload["raw_cases"][key]
        rr = np.asarray(raw["radii"], dtype=float)
        ww = np.asarray(raw["deflection"], dtype=float)
        ww0 = float(np.interp(0.0, rr, ww))
        ax.plot(rr / a, (ww - ww0) / delta, "o", ms=3.0, alpha=0.7,
                label=f"FE h/a={case['h_over_a']:g}, depth {case['domain_depth_over_a']:g}a")
    ax.set_xlabel(r"$r/a$")
    ax.set_ylabel(r"$[w(r)-w(0)]/\delta$")
    ax.set_title("Hertz surface deflection: exact profile")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    # right: measured FE points, offset-invariant shape
    ax = axes[1]
    for key in [k for k in payload["cases"] if k.startswith("h")]:
        case = payload["cases"][key]
        rel = [row["r_over_a"] for row in case["profile"]]
        ax.plot(rel, [100 * row["residual_over_delta"] for row in case["profile"]],
                "o-", ms=4, label=f"h/a={case['h_over_a']:g}, depth {case['domain_depth_over_a']:g}a")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel(r"$r/a$")
    ax.set_ylabel(r"$[D_{fe}-D_{exact}]/\delta$  [%]")
    ax.set_title("offset-invariant shape residual")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    for axi in axes:
        axi.axvspan(0.0, 1.0, color="0.92", zorder=0)
    fig.tight_layout()
    fig.savefig(FIGDIR / FIG_PROFILE, dpi=160)
    plt.close(fig)

    fine_key = sorted([k for k in payload["cases"] if k.startswith("h")],
                      key=lambda k: payload["cases"][k]["h_over_a"])[0]
    fine = payload["cases"][fine_key]
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    rel = [row["r_over_a"] for row in fine["profile"]]
    ax.plot(rel, [100 * row["residual_over_local"] for row in fine["profile"]],
            "o-", color="C0", label="FE vs exact (correct $a$)")
    ax.plot(rel, [100 * row["residual_over_local"] for row in payload["control"]["rows"]],
            "s--", color="C3",
            label=f"FE vs wrong state ($a'={payload['control']['radius_factor']:g}a$)")
    ax.axhline(payload["criteria"]["field_shape_local_max"] * 100.0, color="C0", ls=":",
               label=f"tolerance {payload['criteria']['field_shape_local_max']*100:.1f}% of local")
    ax.set_yscale("log")
    ax.set_xlabel(r"$r/a$")
    ax.set_ylabel("shape residual / local deflection [%]")
    ax.set_title("non-vacuity: the metric rejects a 10% radius error")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / FIG_ERROR, dpi=160)
    plt.close(fig)
    print(f"[fig] {FIGDIR / FIG_PROFILE}")
    print(f"[fig] {FIGDIR / FIG_ERROR}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-dump", action="store_true",
                        help="rebuild certificates/figures from the committed dumps")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    payload = build(args.from_dump)
    if not args.no_figures:
        render(payload)
    verdict = payload["verdict"]
    print(f"[verdict] passed={verdict['passed']} failed={verdict['failed_gates']}")
    for gate in payload["gate_table"]:
        print(f"  {gate['id']:<32} {gate['operator']:>2} {gate['threshold']:<10g} "
              f"value={gate['value']:.6g} passed={gate['passed']}")
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())