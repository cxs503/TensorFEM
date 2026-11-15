"""Verification of the oblique (rotated) ice-patch layer -- gates ``O1`` .. ``O7``.

Anchor ``GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE``.  The audited axis-aligned
grillage layer is *not* re-verified here (see
``docs/AUDIT-GAP-PP-5-GRILLAGE-ULTIMATE-independent-review.md``); what is verified
is the new oblique-patch extension in ``tensorfem.grillage_oblique_patch``:

``O1``  theta = 0 reproduces the closed-form axis-aligned Navier layer
        (``plate_patch_response``): peak moments, modal load shares, patch force.
``O2``  polygon/quadrature geometry: rotated-square area, clipping to the panel,
        quadrature weights summing to the polygon area.
``O3``  quarter-turn invariance of an isotropic square plate under a square patch
        (``m_x(theta) == m_y(theta + 90 deg)``).
``O4``  independent finite-difference reference driven by the *same* oblique load
        (strip superposition of ``ssss_orthotropic_plate_fd`` solvers): deflection
        peak and bending-moment peak versus the series, with refinement.
``O5``  tributary load path: share conservation, strip partition exactness,
        a patch inside one strip loading exactly that member, and mirror symmetry.
``O6``  tie-back to the audited layer: with ``member_allocation="base_convention"``
        and theta = 0, the oblique layer reproduces ``ultimate_ice_pressure``
        (ultimate pressure and all three utilisations) at four combined cases.
``O7``  orientation sensitivity: ultimate ice pressure versus patch orientation at
        equal patch area, for the case-study (plate-governed) and the
        member-governed geometry -- the machine-readable table.

Run::

    PYTHONPATH=src python benchmarks/grillage_oblique_patch_verification.py

Artifacts land in ``artifacts/grillage_oblique_patch/`` (JSON + PNG).  Numbers in
the case document are read from those files, never transcribed by hand.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorfem.grillage_ice_ultimate import (  # noqa: E402
    CombinedLoadCase,
    GrillageGeometry,
    MemberSection,
    case_study_geometry,
    plate_patch_response,
    ultimate_ice_pressure,
)
from tensorfem.grillage_ice_verification import (  # noqa: E402
    MEMBER_GOVERNED_SCENARIOS,
    member_governed_geometry,
)
from tensorfem.grillage_oblique_patch import (  # noqa: E402
    ObliqueIcePatch,
    oblique_patch_response,
    oblique_ultimate_pressure,
    oblique_utilisations,
    orientation_sensitivity,
    patch_fourier_coefficients,
    polygon_area,
    tributary_areas,
    write_orientation_table,
)
from tensorfem.orthotropic_plate_fd import ssss_orthotropic_plate_fd  # noqa: E402

OUT = Path("artifacts/grillage_oblique_patch")
CASES = ((53.0e6, 0.05e6), (106.0e6, 0.10e6), (160.0e6, 0.15e6))
ANGLES = (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0)


def _rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b) if b else abs(a - b)


GATES: list[dict[str, object]] = []


def gate(name: str, measured: float, tolerance: float, unit: str,
         note: str = "") -> bool:
    passed = bool(measured <= tolerance)
    GATES.append({"gate": name, "measured": float(measured),
                  "tolerance": float(tolerance), "unit": unit,
                  "passed": passed, "note": note})
    print(f"[{'PASS' if passed else 'FAIL'}] {name:28s} {measured:.3e} <= {tolerance:.1e} {unit}")
    return passed


# --------------------------------------------------------------------------- #
# O1 -- theta = 0 must reproduce the audited closed-form layer
# --------------------------------------------------------------------------- #
def gate_o1(geom: GrillageGeometry) -> dict[str, float]:
    span_x = geom.n_spans * geom.frame_spacing
    span_y = geom.frame_span
    dx, dy, d3 = geom.plate_rigidities()
    worst = 0.0
    rows = {}
    for lx, ly in ((1.65, 0.55), (1.05, 0.35), (0.70, 0.35), (2.10, 0.35)):
        for p in (1.0e6, 3.0e6):
            patch = ObliqueIcePatch(0.5 * span_x, 0.5 * span_y, lx, ly, 0.0, p)
            general = oblique_patch_response(
                span_x=span_x, span_y=span_y, d1=dx, d2=dy, d3=d3, patch=patch,
                terms=61, sample=61)
            closed = plate_patch_response(
                span_x=span_x, span_y=span_y, d1=dx, d2=dy, d3=d3, pressure=p,
                patch=(0.5 - 0.5 * lx / span_x, 0.5 + 0.5 * lx / span_x,
                       0.5 - 0.5 * ly / span_y, 0.5 + 0.5 * ly / span_y),
                terms=61, sample=61)
            errs = {k: _rel(float(general[k]), float(closed[k]))
                    for k in ("m_x_max", "m_y_max", "w_max", "load_share_x",
                              "load_share_y", "load_share_torsion", "total_force")}
            key = f"{lx:.2f}x{ly:.2f}_{p / 1e6:.0f}MPa"
            rows[key] = errs
            worst = max(worst, max(errs.values()))
    gate("O1_theta0_vs_closed_form", worst, 1e-8, "-",
         "8 axis-aligned cases: peak moments, shares, patch force")
    return {"worst_relative_error": worst, "cases": rows}


# --------------------------------------------------------------------------- #
# O2 -- geometry and quadrature
# --------------------------------------------------------------------------- #
def gate_o2(geom: GrillageGeometry) -> dict[str, float]:
    span_x = geom.n_spans * geom.frame_spacing
    span_y = geom.frame_span
    # rotated square: area is invariant
    sq = ObliqueIcePatch(0.5 * span_x, 0.5 * span_y, 1.0, 1.0, 37.0, 1.0)
    area_err = _rel(abs(polygon_area(sq.clipped(span_x, span_y))), 1.0)
    # half-out patch: exact clipped area of a rotated square hanging over the edge
    half = ObliqueIcePatch(0.0, 0.5 * span_y, 1.0, 1.0, 45.0, 1.0)
    clipped_area = abs(polygon_area(half.clipped(span_x, span_y)))
    # analytic: the part of the rotated square with x >= 0 (its centre is on x=0)
    # is exactly half of the square
    half_err = _rel(clipped_area, 0.5 * 1.0)
    # force identity: a full-panel patch must load exactly the panel area
    full = ObliqueIcePatch(0.5 * span_x, 0.5 * span_y, span_x, span_y, 0.0, 1.0)
    resp = oblique_patch_response(span_x=span_x, span_y=span_y, d1=1.0, d2=1.0,
                                 d3=1.0, patch=full, terms=61, sample=61)
    force_err = _rel(float(resp["load_area"]), span_x * span_y)
    # patch entirely outside the panel -> zero load, no exception
    outside = ObliqueIcePatch(-5.0, -5.0, 0.5, 0.5, 0.0, 1.0e6)
    out_resp = oblique_patch_response(span_x=span_x, span_y=span_y, d1=1.0,
                                      d2=1.0, d3=1.0, patch=outside,
                                      terms=61, sample=21)
    worst = max(area_err, half_err, force_err)
    gate("O2_polygon_geometry", worst, 1e-9, "-",
         "rotated-square area, edge clipping (half-out = 0.5 A), full-panel force")
    assert float(out_resp["total_force"]) == 0.0
    return {"rotated_square_area_error": area_err,
            "clipped_half_area_error": half_err,
            "full_panel_force_error": force_err,
            "outside_patch_force": float(out_resp["total_force"])}


# --------------------------------------------------------------------------- #
# O3 -- quarter-turn invariance on an isotropic square plate
# --------------------------------------------------------------------------- #
def gate_o3() -> dict[str, float]:
    a = 1.0
    d = 1.0e5
    p = 1.0e6
    rows = []
    worst = 0.0
    for angle in (0.0, 20.0, 45.0, 70.0):
        p1 = ObliqueIcePatch(0.5 * a, 0.5 * a, 0.4 * a, 0.25 * a, angle, p)
        r1 = oblique_patch_response(span_x=a, span_y=a, d1=d, d2=d, d3=d,
                                    patch=p1, terms=61, sample=81, poisson=0.3)
        p2 = ObliqueIcePatch(0.5 * a, 0.5 * a, 0.4 * a, 0.25 * a, angle + 90.0, p)
        r2 = oblique_patch_response(span_x=a, span_y=a, d1=d, d2=d, d3=d,
                                    patch=p2, terms=61, sample=81, poisson=0.3)
        e_mx = _rel(float(r1["m_x_max"]), float(r2["m_y_max"]))
        e_my = _rel(float(r1["m_y_max"]), float(r2["m_x_max"]))
        e_w = _rel(float(r1["w_max"]), float(r2["w_max"]))
        rows.append({"angle_deg": angle, "m_x_vs_rotated_m_y": e_mx,
                     "m_y_vs_rotated_m_x": e_my, "w_max": e_w})
        worst = max(worst, e_mx, e_my, e_w)
    gate("O3_quarter_turn_invariance", worst, 5e-3, "-",
         "isotropic square plate: rotation by 90 deg swaps m_x/m_y exactly")
    return {"worst_relative_error": worst, "cases": rows}


# --------------------------------------------------------------------------- #
# O4 -- independent finite-difference reference under the same oblique load
# --------------------------------------------------------------------------- #
def _strip_rectangles(poly: list[tuple[float, float]], n_strips: int):
    """Approximate a convex load polygon by ``n_strips`` axis-aligned rectangles."""
    ys = [p[1] for p in poly]
    y0, y1 = min(ys), max(ys)
    h = (y1 - y0) / n_strips
    rects = []
    for i in range(n_strips):
        yc = y0 + (i + 0.5) * h
        x_at = []
        for k in range(len(poly)):
            ax, ay = poly[k]
            bx, by = poly[(k + 1) % len(poly)]
            if (ay - yc) * (by - yc) <= 0.0 and abs(by - ay) > 0.0:
                t = (yc - ay) / (by - ay)
                x_at.append(ax + t * (bx - ax))
        if len(x_at) >= 2:
            rects.append((min(x_at), max(x_at), yc - 0.5 * h, yc + 0.5 * h))
    return rects


def _fd_field_superposition(*, span_x: float, span_y: float, d1: float, d2: float,
                            d3: float, poly: list[tuple[float, float]],
                            pressure: float, n_strips: int, nx: int,
                            poisson: float) -> dict[str, object]:
    """Sum the FD fields of the strip rectangles (linearity of the plate problem)."""
    w_sum = None
    xs = ys = None
    area = 0.0
    for x0, x1, y0, y1 in _strip_rectangles(poly, n_strips):
        if x1 - x0 <= 0.0 or y1 - y0 <= 0.0:
            continue
        res = ssss_orthotropic_plate_fd(
            span_x=span_x, span_y=span_y, d1=d1, d2=d2, d3=d3, pressure=pressure,
            patch=(x0 / span_x, x1 / span_x, y0 / span_y, y1 / span_y),
            nx=nx, poisson=poisson, return_field=True)
        w = np.asarray(res["w"], dtype=float)
        w_sum = w.copy() if w_sum is None else w_sum + w
        xs = np.asarray(res["x"], dtype=float)
        ys = np.asarray(res["y"], dtype=float)
        area += (x1 - x0) * (y1 - y0)
    if w_sum is None:
        raise RuntimeError("strip decomposition produced no load")
    # moments from the superposed deflection by central differences
    hx = xs[1] - xs[0]
    hy = ys[1] - ys[0]
    w_xx = np.zeros_like(w_sum)
    w_yy = np.zeros_like(w_sum)
    w_xx[1:-1, 1:-1] = (w_sum[2:, 1:-1] - 2.0 * w_sum[1:-1, 1:-1] + w_sum[:-2, 1:-1]) / hx ** 2
    w_yy[1:-1, 1:-1] = (w_sum[1:-1, 2:] - 2.0 * w_sum[1:-1, 1:-1] + w_sum[1:-1, :-2]) / hy ** 2
    d12 = poisson * math.sqrt(d1 * d2)
    m_x = -(d1 * w_xx + d12 * w_yy)
    m_y = -(d2 * w_yy + d12 * w_xx)
    return {"w": w_sum, "x": xs, "y": ys, "m_x": m_x, "m_y": m_y,
            "load_area": area, "w_max": float(np.max(np.abs(w_sum))),
            "m_x_max": float(np.max(np.abs(m_x[1:-1, 1:-1]))),
            "m_y_max": float(np.max(np.abs(m_y[1:-1, 1:-1])))}


def gate_o4(geom: GrillageGeometry) -> dict[str, object]:
    span_x = geom.n_spans * geom.frame_spacing
    span_y = geom.frame_span
    dx, dy, d3 = geom.plate_rigidities()
    angle = 45.0
    lx, ly = 1.65, 0.55
    p = 2.0e6
    patch = ObliqueIcePatch(0.5 * span_x, 0.5 * span_y, lx, ly, angle, p)
    poly = patch.clipped(span_x, span_y)
    series = oblique_patch_response(span_x=span_x, span_y=span_y, d1=dx, d2=dy,
                                    d3=d3, patch=patch, terms=61, sample=121)
    rows = []
    worst = 0.0
    for n_strips, nx in ((16, 96), (32, 96), (32, 192)):
        fd = _fd_field_superposition(span_x=span_x, span_y=span_y, d1=dx, d2=dy,
                                     d3=d3, poly=poly, pressure=p,
                                     n_strips=n_strips, nx=nx,
                                     poisson=geom.poisson)
        e_w = _rel(float(fd["w_max"]), float(series["w_max"]))
        e_mx = _rel(float(fd["m_x_max"]), float(series["m_x_max"]))
        rows.append({"n_strips": n_strips, "nx": nx, "w_max_error": e_w,
                     "m_x_max_error": e_mx,
                     "load_area_error": _rel(float(fd["load_area"]),
                                             float(polygon_area(poly)))})
        if nx == 192:
            worst = max(e_w, e_mx)
    gate("O4_fd_superposition", worst, 5e-2, "-",
         "45 deg patch, FD strip superposition (mx from central differences)")
    return {"series": {"w_max": series["w_max"], "m_x_max": series["m_x_max"]},
            "cases": rows, "worst_error_finest": worst}


# --------------------------------------------------------------------------- #
# O5 -- tributary load path
# --------------------------------------------------------------------------- #
def gate_o5(geom: GrillageGeometry) -> dict[str, object]:
    span_x = geom.n_spans * geom.frame_spacing
    span_y = geom.frame_span
    stamp = ObliqueIcePatch(0.5 * span_x, 0.5 * span_y, 1.65, 0.55, 30.0, 1.0e6)
    trib = tributary_areas(geom, stamp, span_x, span_y)
    sum_stiff = abs(trib["share_sum_stiffener"] - 1.0)
    sum_frame = abs(trib["share_sum_frame"] - 1.0)
    area_err = _rel(sum(r["area"] for r in trib["stiffener"]), trib["patch_area"])
    # a small patch inside one bay loads exactly one stiffener
    tiny = ObliqueIcePatch(3.0, 1.40, 0.10, 0.10, 0.0, 1.0e6)
    trib_tiny = tributary_areas(geom, tiny, span_x, span_y)
    touched_one = trib_tiny["stiffener_touched"] == 1
    # mirror symmetry of the shares: y -> span_y - y
    mirror = ObliqueIcePatch(0.5 * span_x, span_y - 0.5 * span_y, 1.65, 0.55,
                             30.0, 1.0e6)
    trib_m = tributary_areas(geom, mirror, span_x, span_y)
    shares = [r["share"] for r in trib["stiffener"]]
    shares_m = [r["share"] for r in trib_m["stiffener"]]
    mirror_err = max(abs(a - b) for a, b in zip(shares, shares_m[::-1]))
    worst = max(sum_stiff, sum_frame, area_err, mirror_err)
    gate("O5_tributary_conservation", worst, 1e-12, "-",
         "share sums, strip partition area, mirror symmetry")
    gate("O5_single_strip_patch", 0.0 if touched_one else 1.0, 0.0, "flag",
         "a patch inside one tributary strip touches one stiffener")
    return {"share_sum_error": max(sum_stiff, sum_frame),
            "area_error": area_err, "mirror_error": mirror_err,
            "stiffener_share_30deg": shares, "stiffener_touched": trib["stiffener_touched"],
            "frame_touched": trib["frame_touched"]}


# --------------------------------------------------------------------------- #
# O6 -- tie-back to the audited axis-aligned layer
# --------------------------------------------------------------------------- #
def gate_o6(geom: GrillageGeometry) -> dict[str, object]:
    span_x = geom.n_spans * geom.frame_spacing
    span_y = geom.frame_span
    rows = []
    worst_p = 0.0
    worst_u = 0.0
    for sigma, hydro in CASES + ((0.0, 0.0),):
        for lx, ly in ((1.65, 0.55), (1.05, 0.35)):
            case = CombinedLoadCase(axial_stress=sigma, ice_pressure=0.0,
                                    patch_length=lx, patch_width=ly,
                                    hydrostatic_pressure=hydro)
            base = ultimate_ice_pressure(geom, case, terms=61)
            patch = ObliqueIcePatch(0.5 * span_x, 0.5 * span_y, lx, ly, 0.0, 0.0)
            oblique = oblique_ultimate_pressure(
                geom, patch, axial_stress=sigma, hydrostatic_pressure=hydro,
                terms=61, sample=61, member_allocation="base_convention")
            e_p = _rel(float(oblique["ice_pressure_ultimate"]),
                       float(base["ice_pressure_ultimate"]))
            e_u = max(_rel(float(oblique["utilisations_at_limit"][k]),
                           float(base["utilisations_at_limit"][k]))
                      for k in ("plate_local_bending", "longitudinal_stiffener",
                                "transverse_frame"))
            rows.append({
                "sigma_x_MPa": sigma / 1e6, "hydrostatic_MPa": hydro / 1e6,
                "patch_m": [lx, ly],
                "base_p_ult_MPa": float(base["ice_pressure_ultimate"]) / 1e6,
                "oblique_p_ult_MPa": float(oblique["ice_pressure_ultimate"]) / 1e6,
                "p_ult_relative_error": e_p,
                "utilisation_relative_error": e_u,
                "base_governing": base["governing_member"],
                "oblique_governing": oblique["governing_member"],
            })
            worst_p = max(worst_p, e_p)
            worst_u = max(worst_u, e_u)
    gate("O6_tieback_p_ult", worst_p, 1e-5, "-",
         "theta=0, base_convention vs audited ultimate_ice_pressure (8 cases)")
    gate("O6_tieback_utilisations", worst_u, 1e-5, "-",
         "all three limit states at the ultimate pressure")
    return {"worst_p_ult_error": worst_p, "worst_utilisation_error": worst_u,
            "cases": rows}


# --------------------------------------------------------------------------- #
# O7 -- orientation sensitivity
# --------------------------------------------------------------------------- #
def gate_o7(geom: GrillageGeometry, member_geom: GrillageGeometry) -> dict[str, object]:
    span_x = geom.n_spans * geom.frame_spacing
    span_y = geom.frame_span
    tables = {}
    for label, sig, hyd in (("light", *CASES[0]), ("design", *CASES[1]),
                            ("severe", *CASES[2])):
        table = orientation_sensitivity(
            geom, patch_length=1.65, patch_width=0.55, angles_deg=ANGLES,
            axial_stress=sig, hydrostatic_pressure=hyd, terms=61, sample=61)
        tables[label] = table
        write_orientation_table(table, OUT / f"orientation_{label}.json")
    # member-governed geometry: same sweep, tributary allocation
    name, eta, p_ice, lx, ly, p_h = MEMBER_GOVERNED_SCENARIOS[1]
    member_table = orientation_sensitivity(
        member_geom, patch_length=lx, patch_width=ly, angles_deg=ANGLES,
        axial_stress=eta * member_geom.yield_stress,
        hydrostatic_pressure=p_h, terms=61, sample=61)
    member_table["scenario"] = name
    tables["member_governed"] = member_table
    write_orientation_table(member_table, OUT / "orientation_member_governed.json")

    spreads = {k: float(v["spread_relative_to_angle_0"])
               for k, v in tables.items() if v.get("spread_relative_to_angle_0")}
    for label, spread in spreads.items():
        gate(f"O7_orientation_spread_{label}", spread, 1.0, "-",
             "spread of p_ult over 0..90 deg relative to the 0 deg value")
    del span_x, span_y
    return {"tables": tables, "spreads": spreads}


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def figures(geom: GrillageGeometry, o7: dict[str, object]) -> list[str]:
    paths = []
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for label, marker in (("light", "o"), ("design", "s"), ("severe", "^")):
        rows = o7["tables"][label]["rows"]
        ax.plot([r["angle_deg"] for r in rows],
                [r["ice_pressure_ultimate_MPa"] for r in rows],
                marker=marker, label=f"{label} case")
    ax.set_xlabel("ice patch orientation $\\theta$ [deg]")
    ax.set_ylabel("ultimate ice pressure $p_{ult}$ [MPa]")
    ax.set_title("Orientation sensitivity (1.65 m x 0.55 m patch, equal area)")
    ax.grid(alpha=0.3)
    ax.legend()
    p1 = OUT / "fig_orientation_sensitivity.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    paths.append(str(p1))

    span_x = geom.n_spans * geom.frame_spacing
    span_y = geom.frame_span
    dx, dy, d3 = geom.plate_rigidities()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6))
    for ax, angle in zip(axes, (0.0, 45.0)):
        patch = ObliqueIcePatch(0.5 * span_x, 0.5 * span_y, 1.65, 0.55, angle, 2.0e6)
        res = oblique_patch_response(span_x=span_x, span_y=span_y, d1=dx, d2=dy,
                                     d3=d3, patch=patch, terms=61, sample=161,
                                     return_fields=True)
        f = res["fields"]
        cs = ax.contourf(f["x"], f["y"], f["m_x"] / 1e3, levels=20, cmap="viridis")
        poly = np.asarray(patch.clipped(span_x, span_y) + [patch.clipped(span_x, span_y)[0]])
        ax.plot(poly[:, 0], poly[:, 1], "r-", lw=1.6)
        ax.set_title(f"$m_x$ [kN m/m], $\\theta$={angle:.0f} deg")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        fig.colorbar(cs, ax=ax)
    fig.tight_layout()
    p2 = OUT / "fig_patch_fields.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    paths.append(str(p2))
    return paths


def main(stage: str = "all") -> dict[str, object]:
    """Run the gates, merging into ``verification.json``.

    ``stage`` selects a subset so that a single process stays short: ``core``
    (O1-O5), ``tieback`` (O6) or ``orientation`` (O7 + figures); ``all`` runs
    everything.
    """
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    geom = case_study_geometry()
    member_geom = member_governed_geometry()

    path = OUT / "verification.json"
    previous: dict[str, object] = {}
    if stage != "all" and path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
    results: dict[str, object] = dict(previous.get("results", {}))
    GATES.clear()
    GATES.extend(previous.get("gates", []))

    def _replace(prefix: str) -> None:
        GATES[:] = [g for g in GATES if not str(g["gate"]).startswith(prefix)]

    if stage in ("all", "core"):
        for prefix in ("O1", "O2", "O3", "O4", "O5"):
            _replace(prefix)
        results["O1"] = gate_o1(geom)
        results["O2"] = gate_o2(geom)
        results["O3"] = gate_o3()
        results["O4"] = gate_o4(geom)
        results["O5"] = gate_o5(geom)
    if stage in ("all", "tieback"):
        _replace("O6")
        results["O6"] = gate_o6(geom)
    if stage in ("all", "orientation"):
        _replace("O7")
        results["O7"] = gate_o7(geom, member_geom)
        results["figures"] = figures(geom, results["O7"])

    passed = all(g["passed"] for g in GATES)
    payload = {
        "schema": "tensorfem.grillage_oblique_patch.verification/v1",
        "anchor": "GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE",
        "module": "src/tensorfem/grillage_oblique_patch.py",
        "stage": stage,
        "runtime_s": time.time() - t0,
        "gates": GATES,
        "all_passed": passed,
        "results": results,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(GATES)} gates, all_passed={passed}, runtime={time.time() - t0:.1f}s")
    print(f"artifacts -> {OUT}")
    return payload


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "all")
