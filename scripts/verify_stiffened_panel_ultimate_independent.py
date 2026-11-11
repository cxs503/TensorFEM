"""Independent re-verification of ``tensorfem.stiffened_panel_ultimate``.

Why this file exists
--------------------
``stiffened_panel_ultimate`` is a *closed-form screening chain* for ship
stiffened panels (plate field -> Faulkner plate strength -> Johnson-Ostenfeld
column -> flange/tripping limits -> area-weighted panel ultimate, plus
Navier plate field and rigid-plastic yield-line capacity).  Its own tests
re-state the same formulas the module implements, so they cannot catch a
convention or unit mistake: a formula reproduced from the same algebra always
agrees with itself.  This script re-derives every quantity on a path that is
*independent of the module's code*:

1. **Plate field under uniform pressure** -- a finite-difference solution of the
   Kirchhoff plate (two Poisson solves, ``w = 0`` and ``M = -D grad^2 w = 0`` on
   the simply supported edges) is compared with the module's Navier double
   series and with the analytic one-way strip limit ``M = p b^2 / 8``,
   ``w = 5 p b^4 / (384 D)``.
2. **Rigid-plastic yield-line capacity** -- the virtual work of the pyramid and
   strip mechanisms is integrated *numerically* (area quadrature for the
   external work, gradient jumps sampled across the hinge lines for the
   internal work) instead of by hand algebra.
3. **Component limits** -- Faulkner effective width, Johnson-Ostenfeld column
   curve and the one-edge-free outstand buckling are re-implemented from their
   published definitions and swept against the module over a parameter grid;
   the outstand convention is audited against the section geometry used
   elsewhere in the same class.

Outputs: ``artifacts/stiffened_panel_ultimate_review/verification.json`` and
three figures; nothing here is transcribed by hand into the report.
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np

trapz = getattr(np, "trapezoid", None) or np.trapz
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tensorfem.stiffened_panel_ultimate import (  # noqa: E402
    SQUARE_PLATE_PRESSURE_COEFFICIENT,
    StiffenedPanel,
    plate_pressure_bending_stress,
    simply_supported_plate_series,
    yield_line_collapse_pressure,
)

OUT = pathlib.Path(__file__).resolve().parents[1] / "artifacts" / "stiffened_panel_ultimate_review"
OUT.mkdir(parents=True, exist_ok=True)

E_DEFAULT = 2.06e11
NU = 0.3


# ---------------------------------------------------------------------------
# 1. independent finite-difference plate field
# ---------------------------------------------------------------------------
def _laplacian_2d(nx: int, ny: int, h: float) -> sp.csr_matrix:
    """Dirichlet Laplacian on an (nx+1) x (ny+1) node grid, boundary excluded."""
    n = (nx - 1) * (ny - 1)
    main = np.full(n, -4.0 / h**2)
    east = np.full(n, 1.0 / h**2)
    north = np.full(n, 1.0 / h**2)
    rows = np.arange(n)
    # node index i = (j-1)*(nx-1) + (i-1), i in 1..nx-1, j in 1..ny-1
    east[rows % (nx - 1) == nx - 2] = 0.0
    north[rows // (nx - 1) == ny - 2] = 0.0
    diags = [main, east[:-1], east[:-1], north[:-nx + 1], north[:-nx + 1]]
    offsets = [0, 1, -1, nx - 1, -(nx - 1)]
    return sp.diags(diags, offsets, format="csr")


def fd_plate(alpha: float, thickness: float, pressure: float, young: float,
             poisson: float, n_short: int) -> dict:
    """Kirchhoff plate, simply supported rectangle a x b, uniform pressure.

    Solves ``grad^2 M = -p`` with ``M = 0`` on the edges and then
    ``grad^2 w = -M/D`` with ``w = 0``; both boundary conditions are the exact
    simply-supported ones (``w = 0`` and ``w_nn = 0`` on the edge imply
    ``M = -D grad^2 w = 0``).  The centre moments are evaluated with
    Richardson extrapolation between ``n_short`` and ``2*n_short``.
    """
    b = 1.0
    a = alpha * b
    results = []
    for n in (n_short, 2 * n_short):
        h = b / n
        nx, ny = int(round(a / h)), n
        lap = _laplacian_2d(nx, ny, h)
        rhs_m = -pressure * np.ones(lap.shape[0])
        m_int = spla.spsolve(lap.tocsc(), rhs_m)
        d = young * thickness**3 / (12.0 * (1.0 - poisson**2))
        rhs_w = -m_int / d
        w_int = spla.spsolve(lap.tocsc(), rhs_w)
        w = np.zeros((ny + 1, nx + 1))
        m = np.zeros((ny + 1, nx + 1))
        w[1:ny, 1:nx] = w_int.reshape(ny - 1, nx - 1)
        m[1:ny, 1:nx] = m_int.reshape(ny - 1, nx - 1)
        jc, ic = ny // 2, nx // 2
        w_xx = (w[jc, ic + 1] - 2.0 * w[jc, ic] + w[jc, ic - 1]) / h**2
        w_yy = (w[jc + 1, ic] - 2.0 * w[jc, ic] + w[jc - 1, ic]) / h**2
        mx = -d * (w_xx + poisson * w_yy)
        my = -d * (w_yy + poisson * w_xx)
        results.append({"h": h, "w": w[jc, ic], "mx": mx, "my": my,
                        "m_sum": m[jc, ic]})
    coarse, fine = results
    order = 2.0
    factor = 2.0**order
    extrap = {key: (factor * fine[key] - coarse[key]) / (factor - 1.0)
              for key in ("w", "mx", "my")}
    extrap["h_finest"] = fine["h"]
    extrap["m_sum_finest"] = fine["m_sum"]
    extrap["richardson_gain_w"] = abs(fine["w"] - coarse["w"]) / abs(extrap["w"])
    return extrap


def leg_plate_field() -> dict:
    """Navier series vs independent FD vs the analytic one-way strip limit."""
    thickness, pressure = 0.02, 1.0
    rows = []
    for alpha, n_short in ((1.0, 48), (1.5, 64), (2.0, 96), (3.0, 96), (5.0, 64), (10.0, 32)):
        fd = fd_plate(alpha, thickness, pressure, E_DEFAULT, NU, n_short)
        b = 1.0
        d = E_DEFAULT * thickness**3 / (12.0 * (1.0 - NU**2))
        series = plate_pressure_bending_stress(alpha * b, b, thickness, pressure,
                                               young=E_DEFAULT, terms=201)
        w_strip = 5.0 * pressure * b**4 / (384.0 * d)
        m_strip = pressure * b**2 / 8.0
        rows.append({
            "aspect_ratio": alpha,
            "fd_deflection_m": fd["w"],
            "series_deflection_m": series["centre_deflection"],
            "deflection_rel_diff": abs(series["centre_deflection"] / fd["w"] - 1.0),
            "fd_moment_y": fd["my"],
            "series_moment_y": series["centre_moment_y"],
            "moment_y_rel_diff": abs(series["centre_moment_y"] / fd["my"] - 1.0),
            "fd_moment_x": fd["mx"],
            "series_moment_x": series["centre_moment_x"],
            "moment_x_rel_diff": abs(series["centre_moment_x"] / fd["mx"] - 1.0),
            "fd_moment_sum_vs_M": abs(fd["mx"] + fd["my"] - fd["m_sum_finest"]) / abs(fd["m_sum_finest"]),
            "moment_y_coefficient": series["centre_moment_y"] / (pressure * b**2),
            "one_way_strip_moment_coefficient": m_strip / (pressure * b**2),
            "near_one_way_error": abs(series["centre_moment_y"] / (pressure * b**2) - 0.125),
            "deflection_coefficient_pb4_over_D": series["centre_deflection"] * d / (pressure * b**4),
            "one_way_strip_deflection_coefficient": 5.0 / 384.0,
            "richardson_curl_w": fd["richardson_gain_w"],
        })
    square = simply_supported_plate_series(1.0, 1.0, thickness, 1.0, terms=201)
    return {
        "cases": rows,
        "max_series_vs_fd_deflection_diff": max(r["deflection_rel_diff"] for r in rows),
        "max_series_vs_fd_moment_diff": max(max(r["moment_y_rel_diff"], r["moment_x_rel_diff"]) for r in rows),
        "square_moment_coefficient": square["centre_moment"] / 1.0,
        "square_coefficient_constant": SQUARE_PLATE_PRESSURE_COEFFICIENT,
        "square_coefficient_rel_diff": abs(square["centre_moment"] / SQUARE_PLATE_PRESSURE_COEFFICIENT - 1.0),
        # --- the pre-fix contract, reproduced explicitly ------------------
        "pre_fix_unit_young_contract": legacy_contract(rows[3]),
    }


def legacy_series_deflection(span, width, thickness, pressure, young,
                             poisson=NU, terms=51) -> float:
    """The pre-fix deflection sum, verbatim (`D/E` used in place of `D`)."""
    rigidity = thickness**3 / (12.0 * (1.0 - poisson**2))  # D/E, never multiplied back
    total = 0.0
    for m in range(1, terms + 1, 2):
        for n in range(1, terms + 1, 2):
            denominator = m * n * (m**2 / span**2 + n**2 / width**2) ** 2
            shape = math.sin(m * math.pi / 2) * math.sin(n * math.pi / 2)
            total += 16.0 * pressure / (math.pi**6 * rigidity * denominator) * shape
    return total


def legacy_contract(row) -> dict:
    b, t, p = 1.0, 0.02, 1.0
    legacy = legacy_series_deflection(row["aspect_ratio"] * b, b, t, p, E_DEFAULT)
    return {
        "description": ("before the fix the series carried D/E = t^3/(12(1-nu^2)) into "
                        "the deflection sum without multiplying E back, so the returned "
                        "deflection was E times the physical one in SI units"),
        "aspect_ratio": row["aspect_ratio"],
        "legacy_returned_deflection_m": legacy,
        "independent_fd_deflection_m": row["fd_deflection_m"],
        "ratio": legacy / row["fd_deflection_m"],
        "young_default": E_DEFAULT,
        "ratio_over_young": legacy / row["fd_deflection_m"] / E_DEFAULT,
    }


# ---------------------------------------------------------------------------
# 2. independent rigid-plastic virtual work by numerical integration
# ---------------------------------------------------------------------------
def leg_yield_line() -> dict:
    yield_stress, thickness = 235.0e6, 0.02
    sy, t = yield_stress, thickness
    mp = sy * t**2 / 4.0
    # external work for the pyramid, per unit w0: mean of min(1-p1,1-p2)
    rows = []
    for alpha in (1.0, 1.25, 1.5, 1.6, 2.0, 3.0):
        n = 800
        xs = np.linspace(0.0, alpha, n)
        ys = np.linspace(0.0, 1.0, n)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        p1 = np.abs(2.0 * X / alpha - 1.0)
        p2 = np.abs(2.0 * Y - 1.0)
        w = np.minimum(1.0 - p1, 1.0 - p2)
        volume = trapz(trapz(w, ys, axis=1), xs)  # extra work per unit pressure
        # numeric internal work of the pyramid: four diagonals, jump sampled
        total = 0.0
        for (x0, y0) in ((0.0, 0.0), (alpha, 0.0), (0.0, 1.0), (alpha, 1.0)):
            x1, y1 = alpha / 2.0, 0.5
            ln = math.hypot(x1 - x0, y1 - y0)
            tx, ty = (x1 - x0) / ln, (y1 - y0) / ln
            nxv, nyv = -ty, tx
            tt = np.linspace(0.0, 1.0, 4001)
            px = x0 + (x1 - x0) * tt
            py = y0 + (y1 - y0) * tt
            h, eps = 2.0e-4, 1.0e-7

            def f(x, y):
                return np.minimum(1.0 - np.abs(2.0 * x / alpha - 1.0),
                                  1.0 - np.abs(2.0 * y - 1.0))

            wp = f(px + h * nxv, py + h * nyv)
            wm = f(px - h * nxv, py - h * nyv)
            gp = ((f(px + h * nxv + eps, py + h * nyv) - wp) / eps,
                  (f(px + h * nxv, py + h * nyv + eps) - wp) / eps)
            gm = ((f(px - h * nxv + eps, py - h * nyv) - wm) / eps,
                  (f(px - h * nxv, py - h * nyv + eps) - wm) / eps)
            jump = np.abs((gp[0] - gm[0]) * nxv + (gp[1] - gm[1]) * nyv)
            total += mp * trapz(jump, tt) * ln
        numeric_fan = total / volume  # W_int / W_ext(p=1)
        module = yield_line_collapse_pressure(alpha, 1.0, t, sy)
        closed = 12.0 * mp * (alpha**2 + 1.0) / (alpha**2 * 1.0)
        rows.append({
            "aspect_ratio": alpha,
            "numeric_pyramid_pressure": numeric_fan,
            "closed_form_pyramid_pressure": closed,
            "numeric_vs_closed_rel_diff": abs(numeric_fan / closed - 1.0),
            "module_fan_pressure": module["fan_mechanism_pressure"],
            "module_collapse_pressure": module["collapse_pressure"],
            "module_mechanism": module["governing_mechanism"],
            "module_fan_normalised": module["fan_mechanism_pressure"] * 1.0**2 / mp,
            "module_strip_normalised": module["strip_mechanism_pressure"] * 1.0**2 / mp,
            "collapse_normalised": module["collapse_pressure"] * 1.0**2 / mp,
        })
    # quantify the discontinuity of the module selector at a/b = 1.5
    below = yield_line_collapse_pressure(1.5 - 1e-6, 1.0, t, sy)["collapse_pressure"]
    above = yield_line_collapse_pressure(1.5 + 1e-6, 1.0, t, sy)["collapse_pressure"]
    return {
        "cases": rows,
        "max_numeric_vs_closed_rel_diff": max(r["numeric_vs_closed_rel_diff"] for r in rows),
        "selector_jump_at_1p5_rel": abs(above / below - 1.0),
        "selector_below_at_1p5_normalised": below * 1.0**2 / mp,
        "selector_above_at_1p5_normalised": above * 1.0**2 / mp,
        "one_way_limit_normalised": 8.0,
    }


# ---------------------------------------------------------------------------
# 3. component limits, re-implemented from their definitions
# ---------------------------------------------------------------------------
C_OUTSTAND = 0.425 * math.pi**2 / (12.0 * (1.0 - NU**2))


def ref_faulkner(beta: float, sy: float) -> float:
    return sy if beta <= 1.0 else sy * min(1.0, 2.0 / beta - 1.0 / beta**2)


def ref_johnson_ostenfeld(sy: float, se: float) -> float:
    return sy * (1.0 - sy / (4.0 * se)) if se >= 0.5 * sy else se


def ref_plate_buckling(b: float, t: float, e: float) -> float:
    d = e * t**3 / (12.0 * (1.0 - NU**2))
    return 4.0 * math.pi**2 * d / (b**2 * t)


def leg_components() -> dict:
    sy, e = 235.0e6, E_DEFAULT
    beta_rows, lambda_rows = [], []
    for ratio in (60.0, 80.0, 120.0, 160.0, 200.0, 260.0):
        t = 0.012
        panel = StiffenedPanel(span=2.4, spacing=ratio * t, plate_thickness=t,
                               web_height=0.18, web_thickness=0.009,
                               flange_width=0.08, flange_thickness=0.012,
                               young=e, yield_stress=sy)
        beta = ratio * math.sqrt(sy / e)
        beta_rows.append({
            "b_over_t": ratio, "beta": beta,
            "module_beta": panel.plate_slenderness,
            "reference_faulkner": ref_faulkner(beta, sy),
            "module_faulkner": panel.faulkner_plate_strength,
            "buckling_reference": ref_plate_buckling(panel.spacing, t, e),
            "module_buckling": panel.plate_buckling_stress,
        })
    for span in (0.4, 0.8, 1.6, 2.4, 4.0, 6.0):
        panel = StiffenedPanel(span=span, spacing=0.8, plate_thickness=0.02,
                               web_height=0.20, web_thickness=0.010,
                               flange_width=0.08, flange_thickness=0.012,
                               young=e, yield_stress=sy)
        lambda_rows.append({
            "span": span,
            "module_lambda": panel.column_slenderness,
            "module_euler": panel.column_euler_stress,
            "reference_johnson_ostenfeld": ref_johnson_ostenfeld(sy, panel.column_euler_stress),
            "module_column_limit": panel.column_limiting_stress,
        })
    # flange outstand audit: sweep total flange slenderness
    flange_rows = []
    for ratio in (8.0, 12.0, 16.0, 18.0, 20.0, 25.0, 30.0, 36.0, 40.0):
        t_f = 0.010
        panel = StiffenedPanel(span=2.4, spacing=0.8, plate_thickness=0.020,
                               web_height=0.20, web_thickness=0.010,
                               flange_width=ratio * t_f, flange_thickness=t_f,
                               young=e, yield_stress=sy)
        outstand = 0.5 * (panel.flange_width - panel.web_thickness)
        ref_free_edge = C_OUTSTAND * e / (outstand / t_f) ** 2
        ref_total_width = C_OUTSTAND * e / (panel.flange_width / t_f) ** 2
        # what the pre-fix convention (ratio = total width / t) would have given
        legacy_limit = min(ref_total_width, panel.column_limiting_stress, panel.yield_stress)
        legacy_panel = (panel.plate_area * panel.plate_limiting_stress
                        + panel.stiffener_area * legacy_limit) / panel.total_area
        flange_rows.append({
            "flange_width_over_thickness": ratio,
            "flange_outstand_m": outstand,
            "module_flange_local_stress": panel.flange_local_buckling_stress,
            "reference_with_outstand": ref_free_edge,
            "reference_with_total_width": ref_total_width,
            "module_gap_to_outstand_reference": abs(panel.flange_local_buckling_stress / ref_free_edge - 1.0),
            "over_conservatism_factor": ref_free_edge / ref_total_width,
            "module_panel_ultimate": panel.panel_ultimate_stress,
            "module_panel_ultimate_over_yield": panel.panel_yield_ratio,
            "governing_mode": panel.governing_stiffener_mode,
            "pre_fix_panel_ultimate": legacy_panel,
            "pre_fix_governing_mode": ("flange_local" if legacy_limit == min(
                ref_total_width, panel.column_limiting_stress, panel.yield_stress)
                and legacy_limit < panel.column_limiting_stress else "column_or_yield"),
            "pre_fix_under_prediction_rel": 1.0 - legacy_panel / panel.panel_ultimate_stress,
        })
    ice_belt = StiffenedPanel(span=2.4, spacing=0.8, plate_thickness=0.020,
                              web_height=0.20, web_thickness=0.010,
                              flange_width=0.08, flange_thickness=0.012,
                              young=e, yield_stress=sy)
    return {
        "beta_cases": beta_rows,
        "lambda_cases": lambda_rows,
        "flange_cases": flange_rows,
        "max_faulkner_diff": max(abs(r["module_faulkner"] / r["reference_faulkner"] - 1.0) for r in beta_rows),
        "max_buckling_diff": max(abs(r["module_buckling"] / r["buckling_reference"] - 1.0) for r in beta_rows),
        "max_beta_diff": max(abs(r["module_beta"] / r["beta"] - 1.0) for r in beta_rows),
        "max_column_diff": max(abs(r["module_column_limit"] / r["reference_johnson_ostenfeld"] - 1.0) for r in lambda_rows),
        "max_flange_gap": max(r["module_gap_to_outstand_reference"] for r in flange_rows),
        "documented_panel": ice_belt.summary(),
    }


# ---------------------------------------------------------------------------
def figures(plate: dict, yield_line: dict, components: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    d = E_DEFAULT * 0.02**3 / (12.0 * (1.0 - NU**2))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    alphas = [c["aspect_ratio"] for c in plate["cases"]]
    axes[0].plot(alphas, [c["moment_y_coefficient"] for c in plate["cases"]],
                 "o-", label="Navier series (module)")
    axes[0].plot(alphas, [c["fd_moment_y"] / 1.0 for c in plate["cases"]],
                 "x--", label="finite difference (independent)")
    axes[0].axhline(0.125, color="k", lw=0.8, ls=":", label="one-way strip $p b^2/8$")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("aspect ratio a/b")
    axes[0].set_ylabel(r"$M_y^{\max}/(p b^2)$")
    axes[0].set_title("centre moment: two independent paths")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(alphas, [c["deflection_coefficient_pb4_over_D"] for c in plate["cases"]],
                 "o-", label="Navier series (module, fixed)")
    axes[1].plot(alphas, [c["fd_deflection_m"] * d / 1.0**4 for c in plate["cases"]],
                 "x--", label="finite difference (independent)")
    axes[1].axhline(5.0 / 384.0, color="k", lw=0.8, ls=":", label="one-way strip $5/(384)$")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("aspect ratio a/b")
    axes[1].set_ylabel(r"$w_c\,D/(p b^4)$")
    axes[1].set_title("centre deflection: units restored")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    path = OUT / "plate_field_independent.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    made.append(str(path))

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    a = np.linspace(1.0, 5.0, 200)
    fan = [12.0 * (x**2 + 1.0) / (x**2) for x in a]
    strip = [8.0] * len(a)
    selector = [yield_line_collapse_pressure(x, 1.0, 0.02, 235e6)["collapse_pressure"] * 1.0**2 / (235e6 * 0.02**2 / 4.0)
                for x in a]
    ax.plot(a, fan, "--", label="pyramid mechanism $12(a^2+b^2)/(a^2b^2)M_p$")
    ax.plot(a, strip, ":", label="strip mechanism $8M_p/b^2$")
    ax.plot(a, selector, "-", lw=2, label="module selector")
    ax.annotate("step at a/b = 1.5:\n%.0f %%" % (100 * (yield_line["selector_jump_at_1p5_rel"])),
                xy=(1.52, selector[np.argmin(np.abs(a - 1.52))]), xytext=(2.2, 16),
                arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=8)
    ax.set_xlabel("aspect ratio a/b")
    ax.set_ylabel(r"$p_u b^2/M_p$")
    ax.set_title("rigid-plastic capacity: admissible mechanisms and the selector")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = OUT / "yield_line_selector.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    made.append(str(path))

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ratios = [r["flange_width_over_thickness"] for r in components["flange_cases"]]
    ax.plot(ratios, [r["reference_with_outstand"] / 1e6 for r in components["flange_cases"]],
            "o-", label=r"outstand $(b_f-t_w)/2$, k=0.425")
    ax.plot(ratios, [r["reference_with_total_width"] / 1e6 for r in components["flange_cases"]],
            "x--", label=r"total width $b_f$ (pre-fix convention)")
    ax.plot(ratios, [r["module_flange_local_stress"] / 1e6 for r in components["flange_cases"]],
            "s", ms=4, mfc="none", label="module value")
    ax.axhline(235.0, color="k", lw=0.8, ls=":", label=r"$\sigma_y$ = 235 MPa")
    ax.set_yscale("log")
    ax.set_xlabel(r"$b_f/t_f$ (total flange width over thickness)")
    ax.set_ylabel(r"$\sigma_{cr}$ [MPa]")
    ax.set_title("flange outstand buckling: convention audit")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = OUT / "flange_outstand_audit.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    made.append(str(path))
    return made


def main() -> dict:
    plate = leg_plate_field()
    yield_line = leg_yield_line()
    components = leg_components()
    figs = figures(plate, yield_line, components)
    payload = {
        "schema": "stiffened-panel-ultimate-review/1",
        "repo": "TensorFEM",
        "module": "src/tensorfem/stiffened_panel_ultimate.py",
        "plate_field": plate,
        "yield_line": yield_line,
        "components": components,
        "figures": figs,
    }
    (OUT / "verification.json").write_text(json.dumps(payload, indent=1))
    print(json.dumps({
        "plate_max_series_vs_fd_moment": plate["max_series_vs_fd_moment_diff"],
        "plate_max_series_vs_fd_defl": plate["max_series_vs_fd_deflection_diff"],
        "square_vs_constant": plate["square_coefficient_rel_diff"],
        "yield_numeric_vs_closed": yield_line["max_numeric_vs_closed_rel_diff"],
        "selector_jump": yield_line["selector_jump_at_1p5_rel"],
        "max_faulkner": components["max_faulkner_diff"],
        "max_buckling": components["max_buckling_diff"],
        "max_column": components["max_column_diff"],
        "max_flange_gap": components["max_flange_gap"],
        "panel": components["documented_panel"]["panel_ultimate_over_yield"],
        "figures": figs,
    }, indent=1))
    return payload


if __name__ == "__main__":
    main()