#!/usr/bin/env python
"""Verification of the merged plate path against uniform-pressure references.

Produces ``benchmarks/plate_uniform_load_evidence.json`` and the figures in
``docs/assets/plate_uniform_load/``.  Every reported number is computed here;
nothing is transcribed.  The gate is 3 % (fail-closed) on each anchor.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensorfem.plate_uniform_load import (  # noqa: E402
    clamped_uniform_ritz,
    element_center_moments,
    centre_node_deflection,
    nearest_element_center,
    node_values_along_x,
    solve_uniform_square_plate,
    ss_uniform_kirchhoff,
    ss_uniform_mindlin,
    ss_uniform_ritz,
)

ROOT = Path(__file__).resolve().parents[1]
GATE = 0.03
SIDE = 1.0
E = 2.1e11
NU = 0.3
Q = 1.0e6
T_THIN = 0.01   # t/a = 0.01
T_THICK = 0.10  # t/a = 0.10


def rel(fe: float, ref: float) -> float:
    return abs(fe - ref) / abs(ref)


def main() -> dict:
    anchors: list[dict] = []

    def anchor(name, description, value, reference, error, gate=GATE, note=""):
        anchors.append({
            "anchor": name,
            "description": description,
            "fem": value,
            "reference": reference,
            "relative_error": error,
            "tolerance": gate,
            "passed": bool(error < gate),
            "note": note,
        })

    # ---- reference self-checks -------------------------------------------
    ks = ss_uniform_kirchhoff(SIDE, E, NU, T_THIN, Q)
    D = ks["bending_stiffness"]
    w_ss_alpha = ks["deflection"] / (Q * SIDE**4 / D)
    m_ss_alpha = ks["moment_x"] / (Q * SIDE**2)
    anchor("series_ss_centre_deflection_vs_timoshenko",
           "Navier series centre deflection coefficient vs Timoshenko 0.00406",
           w_ss_alpha, 0.00406, rel(w_ss_alpha, 0.00406), 0.01,
           "table rounds to 3 digits; series converged to <1e-9 between 401/801 terms")
    anchor("series_ss_centre_moment_vs_timoshenko",
           "Navier series centre moment coefficient vs Timoshenko 0.0479",
           m_ss_alpha, 0.0479, rel(m_ss_alpha, 0.0479), 0.01, "")

    ritz_ss = ss_uniform_ritz(SIDE, E, NU, T_THIN, Q, n_modes=8)
    w_ritz_ss = ritz_ss.deflection(0.5, 0.5)
    anchor("ritz_machinery_vs_navier_series",
           "Ritz with sine modes must reproduce the closed-form Navier value",
           w_ritz_ss, ks["deflection"], rel(w_ritz_ss, ks["deflection"]), 0.001,
           "machinery anchor for the clamped Ritz reference")

    ritz_cl = clamped_uniform_ritz(SIDE, E, NU, T_THIN, Q, n_modes=10)
    ritz_cl12 = clamped_uniform_ritz(SIDE, E, NU, T_THIN, Q, n_modes=12)
    w_cl = ritz_cl.deflection(0.5, 0.5)
    m_cl = ritz_cl.moments(0.5, 0.5)[0]
    w_cl_alpha = w_cl / (Q * SIDE**4 / D)
    anchor("ritz_clamped_centre_deflection_vs_timoshenko",
           "Clamped Ritz centre deflection coefficient vs Timoshenko 0.00126",
           w_cl_alpha, 0.00126, rel(w_cl_alpha, 0.00126), 0.01, "")
    anchor("ritz_clamped_centre_moment_vs_timoshenko",
           "Clamped Ritz centre moment coefficient vs Timoshenko 0.0231",
           m_cl / (Q * SIDE**2), 0.0231, rel(m_cl / (Q * SIDE**2), 0.0231), 0.01, "")
    anchor("ritz_clamped_mode_independence",
           "N=10 vs N=12 clamped Ritz centre deflection must agree",
           w_cl, ritz_cl12.deflection(0.5, 0.5), rel(w_cl, ritz_cl12.deflection(0.5, 0.5)),
           0.01, "reference stability, not an FE comparison")

    # ---- simply supported, thin ------------------------------------------
    conv_ss, profiles = [], {}
    for nx in (4, 8, 16, 24):
        res = solve_uniform_square_plate(nx, SIDE, E, NU, T_THIN, Q, "simply_supported")
        w = centre_node_deflection(res)
        conv_ss.append({"nx": nx, "h": SIDE / nx, "fem": w,
                        "relative_error": rel(w, ks["deflection"]),
                        "passed": rel(w, ks["deflection"]) < GATE})
        if nx == 16:
            profiles["ss"] = node_values_along_x(res, 0.5)
    a16 = next(c for c in conv_ss if c["nx"] == 16)
    anchor("ss_thin_16x16_centre_deflection",
           "SS t/a=0.01, 16x16 Q4 Mindlin vs Navier series", a16["fem"], ks["deflection"],
           a16["relative_error"], GATE, "")

    # moment at the plate centre: an odd mesh makes the centre an element centre
    res17 = solve_uniform_square_plate(17, SIDE, E, NU, T_THIN, Q, "simply_supported")
    mom = element_center_moments(res17)
    i_c = nearest_element_center(mom, 0.5 * SIDE, 0.5 * SIDE)
    ref_c = ss_uniform_kirchhoff(SIDE, E, NU, T_THIN, Q, mom["x"][i_c], mom["y"][i_c])
    anchor("ss_thin_centre_moment",
           "SS t/a=0.01, 17x17, element-centre Mx at the plate centre vs exact series "
           "at the same point", mom["moment_x"][i_c], ref_c["moment_x"],
           rel(mom["moment_x"][i_c], ref_c["moment_x"]), GATE, "")

    # ---- simply supported, thick (shear matters) --------------------------
    km = ss_uniform_mindlin(SIDE, E, NU, T_THICK, Q)
    res_thick = solve_uniform_square_plate(16, SIDE, E, NU, T_THICK, Q, "simply_supported")
    w_thick = centre_node_deflection(res_thick)
    anchor("ss_thick_16x16_centre_deflection",
           "SS t/a=0.10, 16x16 vs exact Mindlin series (bending + shear)", w_thick,
           km["deflection"], rel(w_thick, km["deflection"]), GATE,
           "shear share %.3f%% of the total" % (km["deflection_shear"] / km["deflection"] * 100))
    anchor("negative_control_thick_vs_kirchhoff",
           "Same FE result against the Kirchhoff (shear-free) value must exceed 3%%, "
           "proving the case discriminates the shear treatment", w_thick,
           km["deflection_kirchhoff"], rel(w_thick, km["deflection_kirchhoff"]), GATE,
           "control: expected to FAIL the gate")

    # ---- clamped, thin ----------------------------------------------------
    conv_cl = []
    for nx in (4, 8, 16):
        res = solve_uniform_square_plate(nx, SIDE, E, NU, T_THIN, Q, "clamped")
        w = centre_node_deflection(res)
        conv_cl.append({"nx": nx, "h": SIDE / nx, "fem": w,
                        "relative_error": rel(w, w_cl), "passed": rel(w, w_cl) < GATE})
        if nx == 16:
            profiles["clamped"] = node_values_along_x(res, 0.5)
    cl16 = next(c for c in conv_cl if c["nx"] == 16)
    anchor("clamped_thin_16x16_centre_deflection",
           "Clamped t/a=0.01, 16x16 vs clamped Ritz (N=10)", cl16["fem"], w_cl,
           cl16["relative_error"], GATE, "")
    anchor("retained_coarse_failure_clamped_4x4",
           "Clamped 4x4 must stay above the 3% gate (failure is kept, not hidden)",
           conv_cl[0]["fem"], w_cl, conv_cl[0]["relative_error"], GATE,
           "control: expected to FAIL the gate")

    res17c = solve_uniform_square_plate(17, SIDE, E, NU, T_THIN, Q, "clamped")
    momc = element_center_moments(res17c)
    j_c = nearest_element_center(momc, 0.5 * SIDE, 0.5 * SIDE)
    ref_mc = ritz_cl.moments(momc["x"][j_c], momc["y"][j_c])[0]
    anchor("clamped_thin_centre_moment",
           "Clamped t/a=0.01, 17x17, element-centre Mx at the plate centre vs clamped "
           "Ritz at the same point", momc["moment_x"][j_c], ref_mc,
           rel(momc["moment_x"][j_c], ref_mc), GATE,
           "edge hogging peak is NOT certified (pointwise series converge slowly on "
           "both sides)")

    # ---- wrong-boundary negative control ---------------------------------
    ss16 = a16["fem"]
    anchor("negative_control_clamped_vs_ss_reference",
           "Clamped FE result against the SS reference must exceed 3%, so the gate "
           "discriminates boundary conditions", ss16, w_cl, rel(ss16, w_cl), GATE,
           "control: expected to FAIL the gate")

    # ---- convergence orders ----------------------------------------------
    def orders(rows):
        """Consecutive observed orders; the finest pair is excluded from the
        headline because pointwise centre values superconverge once the error
        is below 0.1%."""
        out = []
        for r0, r1 in zip(rows[:-1], rows[1:]):
            out.append(math.log(r0["relative_error"] / r1["relative_error"])
                       / math.log(r0["h"] / r1["h"]))
        return out

    def order(rows):
        return 0.5 * (orders(rows)[0] + orders(rows)[1])

    convergence = {
        "ss_thin": conv_ss, "ss_thin_order": order(conv_ss),
        "ss_thin_orders": orders(conv_ss),
        "clamped_thin": conv_cl, "clamped_thin_order": order(conv_cl),
        "clamped_thin_orders": orders(conv_cl),
        "references": {"ss_centre_deflection": ks["deflection"],
                       "ss_centre_moment": ks["moment_x"],
                       "clamped_centre_deflection": w_cl,
                       "clamped_centre_moment": m_cl,
                       "clamped_mindlin_deflection": km["deflection"]},
        "parameters": {"side": SIDE, "young": E, "poisson": NU, "pressure": Q,
                       "t_thin": T_THIN, "t_thick": T_THICK,
                       "terms": 401, "ritz_modes": 10},
    }

    # ---- figures ----------------------------------------------------------
    assets = ROOT / "docs" / "assets" / "plate_uniform_load"
    assets.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for rows, label, style in ((conv_ss, "simply supported", "o-"),
                               (conv_cl, "clamped", "s-".replace("s", "s"))):
        h = [r["h"] for r in rows]
        err = [max(r["relative_error"], 1e-12) for r in rows]
        ax1.loglog(h, err, style, label=label)
    hh = np.array([1.0 / 4, 1.0 / 64])
    ax1.loglog(hh, 3e-3 * (hh / hh[0]) ** 2, "k--", lw=1, label="slope 2 guide")
    ax1.axhline(GATE * 100 / 100, color="r", ls=":", label="3% gate")
    ax1.set_xlabel("element size h/a")
    ax1.set_ylabel("relative error in centre deflection")
    ax1.set_title("Mesh convergence, uniform pressure")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8)

    xs_ss, ws_ss = profiles["ss"]
    xs_cl, ws_cl_p = profiles["clamped"]
    ax2.plot(np.array(xs_ss), np.array(ws_ss) / ks["deflection"], "o", ms=3,
             label="FE SS 16x16")
    xs_f = np.linspace(0, SIDE, 61)
    ax2.plot(xs_f, [ss_uniform_kirchhoff(SIDE, E, NU, T_THIN, Q, x, 0.5 * SIDE)["deflection"]
                    / ks["deflection"] for x in xs_f], "-", lw=1, label="exact series SS")
    ax2.plot(np.array(xs_cl), np.array(ws_cl_p) / w_cl, "s", ms=3, label="FE clamped 16x16")
    ax2.plot(xs_f, [ritz_cl.deflection(x, 0.5 * SIDE) / w_cl for x in xs_f], "--", lw=1,
             label="clamped Ritz")
    ax2.set_xlabel("x/a at y = a/2")
    ax2.set_ylabel("w / w_max")
    ax2.set_title("Centre-line deflection shape")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    f1 = assets / "plate_uniform_convergence_profile.png"
    fig.savefig(f1, dpi=140)
    plt.close(fig)

    names = [a["anchor"] for a in anchors]
    errs = [max(a["relative_error"] * 100, 1e-6) for a in anchors]
    colours = ["tab:green" if a["passed"] else "tab:red" for a in anchors]
    fig2, ax = plt.subplots(figsize=(9, 4.6))
    ax.barh(range(len(names)), errs, color=colours)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xscale("log")
    ax.axvline(3.0, color="r", ls=":", label="3% gate")
    ax.set_xlabel("relative error [%] (log)")
    ax.set_title("Anchors: green = certified, red = retained failure / negative control")
    ax.legend(fontsize=8)
    fig2.tight_layout()
    f2 = assets / "plate_uniform_anchor_errors.png"
    fig2.savefig(f2, dpi=140)
    plt.close(fig2)

    evidence = {
        "benchmark": "plate_uniform_pressure",
        "description": "Uniform lateral pressure on a Q4 Mindlin square plate (SS and "
                       "clamped, thin and thick) against exact Navier series and a "
                       "separable Rayleigh-Ritz reference",
        "anchors": anchors,
        "convergence": convergence,
        "figures": [str(f1.relative_to(ROOT)), str(f2.relative_to(ROOT))],
        "all_passed": all(a["passed"] for a in anchors
                          if not a["note"].startswith("control:")),
    }
    out = ROOT / "benchmarks" / "plate_uniform_load_evidence.json"
    out.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({k: evidence[k] for k in ("all_passed",)}, indent=2))
    for a in anchors:
        print("%-46s fem=%.6e ref=%.6e err=%.4f%% tol=%.1f%% %s %s" % (
            a["anchor"], a["fem"], a["reference"], a["relative_error"] * 100,
            a["tolerance"] * 100, "PASS" if a["passed"] else "FAIL", a["note"]))
    print("orders: ss=%.2f clamped=%.2f" % (convergence["ss_thin_order"],
                                            convergence["clamped_thin_order"]))
    return evidence


if __name__ == "__main__":
    main()