"""Structural mechanics verification face: Mindlin plate bending + vibration.

Runs the two halves of :mod:`tensorfem.structural_mindlin_face` and writes a
machine-readable report (``artifacts/structural_mindlin_face/report.json``) plus
four figures.  Every reported number is computed in this script; nothing is
transcribed, so the JSON is the provenance for the Markdown report.

Criteria (from the campaign node P187-W1-N02)
    C1  Mindlin centre deflection vs Kirchhoff closed form < 2 % (h/L <= 0.05)
    C2  h/L = 0.2 Mindlin deflection > Kirchhoff (shear deformation)
    C3  mesh convergence order >= 2
    C4  first six plate frequencies vs analytic/variational reference < 3 %
    C5  negative control: density -> 0 gives frequency -> infinity
    C6  end-to-end JSON carries deflection/stress/natural_frequencies/
        convergence_rate/boundary_conditions

Usage
-----
    OMP_NUM_THREADS=4 python benchmarks/structural_mindlin_vibration_face.py
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorfem import structural_mindlin_face as face  # noqa: E402
from tensorfem.cylindrical_shell_modal import dmv_angular_frequency  # noqa: E402

# ---------------------------------------------------------------------------
# Case definition: a steel plate panel of 1 m x 1 m (ship deck / bulkhead scale)
# ---------------------------------------------------------------------------
MATERIAL = dict(young=200.0e9, poisson=0.3, density=7850.0)
LENGTH = 1.0
WIDTH = 1.0
LOAD = 1.0e4          # uniform pressure 10 kPa (order of a design deck load)
MODES = 6

THIN_RATIOS = (0.005, 0.01, 0.02, 0.05)
THICK_RATIO = 0.2

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "structural_mindlin_face"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # pragma: no cover - non-git checkout
        return "unknown"


def _fig_png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    report: dict = {
        "face": "structural_mindlin_vibration",
        "campaign_node": "P187-W1-N02",
        "case": {
            "description": "uniformly loaded, free-vibrating steel plate panel",
            "length_m": LENGTH,
            "width_m": WIDTH,
            "load_Pa": LOAD,
            **{k: v for k, v in MATERIAL.items()},
        },
    }

    # -- C1 / C2: thickness sweep against Kirchhoff and FSDT closed forms ----
    sweep_rows = face.locking_sweep(
        thickness_ratios=THIN_RATIOS + (THICK_RATIO,),
        elements_x=16, elements_y=16, length=LENGTH, width=WIDTH,
        young=MATERIAL["young"], poisson=MATERIAL["poisson"], load=LOAD,
        treatments=("sri", "mitc", "full"),
    )
    thin = [r for r in sweep_rows if r["thickness_ratio"] <= 0.05
            and r["treatment"] in ("sri", "mitc")]
    c1_error = max(abs(r["error_vs_kirchhoff"]) for r in thin)
    thick = [r for r in sweep_rows if r["thickness_ratio"] == THICK_RATIO
             and r["treatment"] == "sri"][0]
    report["deflection"] = {
        "thickness_sweep": sweep_rows,
        "deflection_definition": "centre transverse deflection w(0.5,0.5)",
        "fsdt_over_kirchhoff": {row["thickness_ratio"]: row["fsdt_over_kirchhoff"]
                                for row in sweep_rows
                                if row["treatment"] == "sri"},
    }

    print(f"[face] static thickness sweep done ({time.time() - t_start:.1f}s)",
          flush=True)
    # -- C3: mesh convergence of the centre deflection ----------------------
    conv = face.mesh_convergence(
        elements=(4, 8, 16, 32), thickness_ratio=0.01, length=LENGTH,
        young=MATERIAL["young"], poisson=MATERIAL["poisson"], load=LOAD,
    )
    order = face.observed_convergence_order(
        [r["mesh_size"] for r in conv], [r["relative_error"] for r in conv]
    )
    report["convergence_rate"] = {
        "quantity": "centre deflection, h/L = 0.01, SRI",
        "rows": conv,
        "observed_order": order["observed_order"],
        "pairwise_orders": order["pairwise_orders"],
    }

    print(f"[face] convergence study done ({time.time() - t_start:.1f}s)", flush=True)
    # -- stress: FE bending moment at a Gauss point vs the closed form there -
    stress_rows = []
    for ratio in (0.01, 0.05, 0.2):
        h = ratio * LENGTH
        res = face.solve_plate_static(
            elements_x=16, elements_y=16, length=LENGTH, width=WIDTH,
            thickness=h, young=MATERIAL["young"], poisson=MATERIAL["poisson"],
            load=LOAD, boundary="SS", shear_treatment="sri",
        )
        ref = face.kirchhoff_ss_uniform_load(
            length=LENGTH, width=WIDTH, thickness=h,
            young=MATERIAL["young"], poisson=MATERIAL["poisson"], load=LOAD,
        )
        x, y = res.sample_point
        m_ref = ref.moment_x(
            x, y, length=LENGTH, width=WIDTH, poisson=MATERIAL["poisson"],
            load=LOAD,
        )
        stress_rows.append({
            "thickness_ratio": ratio,
            "sample_point_m": [x, y],
            "moment_x_fe_Nm_per_m": res.moment_x_center,
            "moment_x_kirchhoff_Nm_per_m": m_ref,
            "relative_error": (res.moment_x_center - m_ref) / m_ref,
            "bending_stress_MPa": res.bending_stress_center / 1e6,
        })
    report["stress"] = {
        "quantity": "bending moment M_x at the Gauss point nearest the centre",
        "rows": stress_rows,
        "max_abs_error": max(abs(r["relative_error"]) for r in stress_rows),
    }

    print(f"[face] stress rows done ({time.time() - t_start:.1f}s)", flush=True)
    # -- C4: first six plate frequencies, SS / CC / CF ----------------------
    h_thin = 0.01 * LENGTH
    freqs: dict = {}
    t0 = time.time()
    for bc in ("SS", "CC", "CF"):
        mesh = 24
        res = face.solve_plate_modes(
            elements_x=mesh, elements_y=mesh, length=LENGTH, width=WIDTH,
            thickness=h_thin, young=MATERIAL["young"],
            poisson=MATERIAL["poisson"], density=MATERIAL["density"],
            boundary=bc, modes=MODES, shear_treatment="mitc",
        )
        freqs[bc] = {
            "mesh": mesh,
            "frequencies_hz": res.frequencies_hz.tolist(),
            "n_free_dof": res.n_free_dof,
            "shear_treatment": res.shear_treatment,
            "wall_seconds": time.time() - t0,
        }
        print(f"[face] {bc} modes at n={mesh} done "
              f"({time.time() - t_start:.1f}s)", flush=True)
        t0 = time.time()

    # SS: exact FSDT closed form (Mindlin with rotatory inertia)
    ss_mindlin = face.mindlin_ss_angular_frequencies(
        length=LENGTH, width=WIDTH, thickness=h_thin, young=MATERIAL["young"],
        poisson=MATERIAL["poisson"], density=MATERIAL["density"], modes=MODES,
    )
    ss_kirchhoff = face.kirchhoff_ss_angular_frequencies(
        length=LENGTH, width=WIDTH, thickness=h_thin, young=MATERIAL["young"],
        poisson=MATERIAL["poisson"], density=MATERIAL["density"], modes=MODES,
    )
    # CC: polynomial Rayleigh-Ritz (validated against Leissa's clamped square
    # plate coefficients 35.9853 / 73.394 / 108.2 / 131.6 / 132.2)
    cc_ritz = face.ritz_plate_angular_frequencies(
        length=LENGTH, width=WIDTH, thickness=h_thin, young=MATERIAL["young"],
        poisson=MATERIAL["poisson"], density=MATERIAL["density"],
        boundary="CC", modes=MODES, degree=12,
    )

    def _rel(fe, ref):
        return [(float(a) - float(b)) / float(b) for a, b in zip(fe, ref)]

    ss_fe = np.array(freqs["SS"]["frequencies_hz"])
    cc_fe = np.array(freqs["CC"]["frequencies_hz"])
    freqs["SS"]["reference"] = "exact Mindlin (FSDT) Navier solution"
    freqs["SS"]["reference_hz"] = (ss_mindlin / (2 * math.pi)).tolist()
    freqs["SS"]["kirchhoff_hz"] = (ss_kirchhoff / (2 * math.pi)).tolist()
    freqs["SS"]["relative_error"] = _rel(ss_fe, ss_mindlin / (2 * math.pi))
    freqs["CC"]["reference"] = "polynomial Rayleigh-Ritz (Kirchhoff)"
    freqs["CC"]["reference_hz"] = (cc_ritz / (2 * math.pi)).tolist()
    freqs["CC"]["relative_error"] = _rel(cc_fe, cc_ritz / (2 * math.pi))

    # CF: no closed form is used -- three-mesh Richardson extrapolation, with the
    # observed order taken from the SS convergence study (the same element).
    cf_runs = {}
    for mesh in (6, 12, 24):
        res = face.solve_plate_modes(
            elements_x=mesh, elements_y=mesh, length=LENGTH, width=WIDTH,
            thickness=h_thin, young=MATERIAL["young"],
            poisson=MATERIAL["poisson"], density=MATERIAL["density"],
            boundary="CF", modes=MODES, shear_treatment="mitc",
        )
        cf_runs[mesh] = res.frequencies_hz.tolist()
    cf_extrapolated = []
    cf_gci = []
    p_order = max(2.0, float(order["observed_order"]))
    for k in range(MODES):
        series = [cf_runs[6][k], cf_runs[12][k], cf_runs[24][k]]
        ext = face.richardson_extrapolation(series, refinement=2.0,
                                            observed_order=p_order)
        cf_extrapolated.append(ext["extrapolated"])
        cf_gci.append(ext["gci"])
    freqs["CF"] = {
        "mesh": 24,
        "frequencies_hz": cf_runs[24],
        "mesh_series_hz": {str(k): v for k, v in cf_runs.items()},
        "reference": ("three-mesh Richardson extrapolation (ratio 2, order from "
                      "the SS convergence study); no closed form exists"),
        "reference_hz": cf_extrapolated,
        "relative_error": _rel(cf_runs[24], cf_extrapolated),
        "gci_relative": cf_gci,
    }

    print(f"[face] CF refinement done ({time.time() - t_start:.1f}s)", flush=True)
    # thick plate spectrum at h/L = 0.2 against the exact Mindlin closed form
    h_thick = THICK_RATIO * LENGTH
    thick_modes = face.solve_plate_modes(
        elements_x=16, elements_y=16, length=LENGTH, width=WIDTH,
        thickness=h_thick, young=MATERIAL["young"],
        poisson=MATERIAL["poisson"], density=MATERIAL["density"],
        boundary="SS", modes=MODES, shear_treatment="mitc",
    )
    thick_ref = face.mindlin_ss_angular_frequencies(
        length=LENGTH, width=WIDTH, thickness=h_thick, young=MATERIAL["young"],
        poisson=MATERIAL["poisson"], density=MATERIAL["density"], modes=MODES,
    ) / (2 * math.pi)
    thick_kirchhoff = face.kirchhoff_ss_angular_frequencies(
        length=LENGTH, width=WIDTH, thickness=h_thick, young=MATERIAL["young"],
        poisson=MATERIAL["poisson"], density=MATERIAL["density"], modes=MODES,
    ) / (2 * math.pi)
    freqs["SS_thick_h_over_L_0.2"] = {
        "frequencies_hz": thick_modes.frequencies_hz.tolist(),
        "reference_mindlin_hz": thick_ref.tolist(),
        "reference_kirchhoff_hz": thick_kirchhoff.tolist(),
        "relative_error_vs_mindlin": _rel(thick_modes.frequencies_hz, thick_ref),
        "relative_error_vs_kirchhoff": _rel(thick_modes.frequencies_hz,
                                            thick_kirchhoff),
    }
    report["natural_frequencies"] = freqs

    # -- C5: negative control, density -> 0 must push frequencies to infinity
    control = []
    base_density = MATERIAL["density"]
    for factor in (1.0, 1e-2, 1e-4):
        res = face.solve_plate_modes(
            elements_x=8, elements_y=8, length=LENGTH, width=WIDTH,
            thickness=h_thin, young=MATERIAL["young"],
            poisson=MATERIAL["poisson"], density=base_density * factor,
            boundary="SS", modes=1, shear_treatment="mitc",
        )
        control.append({
            "density": base_density * factor,
            "f1_hz": float(res.frequencies_hz[0]),
            "ratio_to_baseline": None,
        })
    for row in control[1:]:
        row["ratio_to_baseline"] = row["f1_hz"] / control[0]["f1_hz"]
    report["negative_control"] = {
        "rule": "f1 * sqrt(rho) must be invariant; f1 must grow as rho -> 0",
        "rows": control,
        "f1_times_sqrt_rho": [row["f1_hz"] * math.sqrt(row["density"])
                              for row in control],
    }

    print(f"[face] thick spectrum + density control done "
          f"({time.time() - t_start:.1f}s)", flush=True)
    # -- shell half: reference plus an honest statement of the current gap --
    # The existing cylindrical_shell4 path (see tensorfem.cylindrical_shell_modal)
    # is exercised on the standard geometry to document where it stands.  It is
    # *not* claimed as verified: its spectrum falls far below the DMV closed form
    # because the diaphragm BC leaves the axial translation free, so the first
    # "modes" are rigid/near-rigid, and the repository test only bounds them
    # loosely (10 < f < 500 Hz).
    shell_case = dict(radius=0.5, length=1.0, thickness=0.005, young=210e9,
                      poisson=0.3, density=7800.0)
    dmv = sorted(
        (dmv_angular_frequency(n_axial=m, n_circ=n, **shell_case) / (2 * math.pi), m, n)
        for m in (1, 2, 3) for n in range(1, 7)
    )
    shell_fe = face.solve_shell_natural_frequencies(
        n_axial=10, n_circ=24, boundary="SS", modes=6, **shell_case
    )
    fe_hz = shell_fe["frequencies_hz"]
    dmv_min = dmv[0][0]
    report["cylindrical_shell"] = {
        "geometry": shell_case,
        "dmv_reference_hz": [{"frequency_hz": v, "axial_half_waves": m,
                              "circumferential_waves": n} for v, m, n in dmv[:8]],
        "fe_frequencies_hz": fe_hz,
        "fe_free_dof": shell_fe["n_free_dof"],
        "verdict": "NOT_VERIFIED",
        "lowest_fe_frequency_ratio_to_dmv": fe_hz[0] / dmv_min,
        "reason": (
            f"the lowest FE frequencies ({fe_hz[0]:.2e}, {fe_hz[1]:.2e} Hz) are "
            "numerically zero: the diaphragm condition constrains only (v, w, rx) "
            "on the end rings and leaves the axial translation (and drilling "
            "rotation) free, so the first 'modes' are rigid-body modes, not the "
            f"circumferential bending family whose lowest DMV member is {dmv_min:.0f} Hz. "
            f"The first bending-like FE frequency is {fe_hz[2]:.1f} Hz, i.e. "
            f"{fe_hz[2] / dmv_min:.3f} of the DMV value, and the repository test for "
            "this path only bounds the frequencies loosely (10 < f < 500 Hz). The "
            "shell half of C4 is therefore reported as an honest gap, not a pass."
        ),
    }

    # -- criteria roll-up ---------------------------------------------------
    criteria = {
        "C1_thin_deflection_vs_kirchhoff": {
            "threshold": 0.02,
            "value": c1_error,
            "pass": bool(c1_error < 0.02),
            "statement": "max |w_FE - w_Kirchhoff| / w_Kirchhoff over h/L <= 0.05",
        },
        "C2_thick_deflection_exceeds_kirchhoff": {
            "threshold": 0.0,
            "value": thick["deflection_fe"] / thick["deflection_kirchhoff"] - 1.0,
            "pass": bool(thick["deflection_fe"] > thick["deflection_kirchhoff"]),
            "statement": ("h/L = 0.2: w_Mindlin - w_Kirchhoff > 0 (shear "
                          "deformation); FE vs the FSDT closed form is reported "
                          "as the accuracy check"),
            "error_vs_fsdt": thick["error_vs_fsdt"],
        },
        "C3_convergence_order": {
            "threshold": 2.0,
            "value": order["observed_order"],
            "pass": bool(order["observed_order"] >= 2.0),
            "statement": "least-squares slope of log|error| vs log h, n=4..32",
        },
        "C4_plate_frequencies": {
            "threshold": 0.03,
            "value": max(
                max(abs(e) for e in freqs["SS"]["relative_error"]),
                max(abs(e) for e in freqs["CC"]["relative_error"]),
            ),
            "pass": bool(
                max(abs(e) for e in freqs["SS"]["relative_error"]) < 0.03
                and max(abs(e) for e in freqs["CC"]["relative_error"]) < 0.03
            ),
            "statement": ("first six frequencies, SS vs exact Mindlin and CC vs "
                          "Rayleigh-Ritz; CF is extrapolated, its error against "
                          "the extrapolation is reported separately"),
            "CF_max_error_vs_extrapolation": max(abs(e) for e in
                                                 freqs["CF"]["relative_error"]),
            "CF_max_gci": max(cf_gci),
            "shell": "NOT_VERIFIED - see cylindrical_shell section",
        },
        "C5_density_negative_control": {
            "threshold": None,
            "value": [row["ratio_to_baseline"] for row in control[1:]],
            "pass": bool(all(row["f1_hz"] > control[0]["f1_hz"] for row in control[1:])),
            "statement": "f1 must grow as rho -> 0; sqrt(rho) f1 stays constant",
        },
    }
    report["criteria"] = criteria
    report["boundary_conditions"] = {
        "plate": {bc: {"mesh": freqs[bc]["mesh"],
                       "frequencies_hz": freqs[bc]["frequencies_hz"]}
                  for bc in ("SS", "CC", "CF")},
        "shell": {"requested": ["SS", "CC", "CF"],
                  "verified": [],
                  "note": "shell end conditions not verified in this face"},
    }

    # -- figures ------------------------------------------------------------
    figures = []

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ratios = sorted({r["thickness_ratio"] for r in sweep_rows})
    for treatment, style in (("sri", "o-"), ("mitc", "s--"), ("full", "^-")):
        errs = [next(100 * abs(r["error_vs_kirchhoff"]) for r in sweep_rows
                     if r["thickness_ratio"] == ratio_ and r["treatment"] == treatment)
                for ratio_ in ratios]
        ax.plot(ratios, errs, style, label=f"{treatment} shear")
    fsdt = [100 * (next(r["fsdt_over_kirchhoff"] for r in sweep_rows
                        if r["thickness_ratio"] == ratio_ and r["treatment"] == "sri") - 1)
            for ratio_ in ratios]
    ax.plot(ratios, fsdt, "k:", label="shear-deformation effect (FSDT/Kirchhoff-1)")
    ax.axhline(2.0, color="grey", lw=0.8, ls="--", label="C1 2% band")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("thickness ratio h/L")
    ax.set_ylabel("|error vs Kirchhoff| [%]")
    ax.set_title("Thickness sweep: locking control and shear deformation")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    figures.append(("thickness_sweep.png",
                    "Centre-deflection error against the Kirchhoff closed form "
                    "for three shear treatments; full integration locks while SRI "
                    "and MITC stay within C1 for h/L <= 0.05.", _fig_png(fig)))

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    hs = np.array([r["mesh_size"] for r in conv])
    es = np.abs(np.array([r["relative_error"] for r in conv]))
    ax.loglog(hs, es, "o-", label="FE vs FSDT closed form")
    ref = es[0] * (hs / hs[0]) ** 2
    ax.loglog(hs, ref, "k--", label="slope 2 reference")
    ax.set_xlabel("mesh size h [m]")
    ax.set_ylabel("|relative error| of centre deflection")
    ax.set_title(f"Mesh convergence (observed order p = {order['observed_order']:.2f})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    figures.append(("convergence.png",
                    "Centre-deflection error versus mesh size with the fitted "
                    "observed order of accuracy.", _fig_png(fig)))

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.9), sharey=False)
    for ax, bc, title in zip(axes, ("SS", "CC", "CF"),
                             ("SS: exact Mindlin", "CC: Rayleigh-Ritz",
                              "CF: Richardson extrapolation")):
        ref_hz = np.array(freqs[bc]["reference_hz"])
        fe_hz = np.array(freqs[bc]["frequencies_hz"])
        idx = np.arange(1, MODES + 1)
        ax.plot(idx, ref_hz, "ko-", label="reference")
        ax.plot(idx, fe_hz, "s--", label="FE (n=16)" if bc != "CF" else "FE (n=32)")
        for i, (a, b) in enumerate(zip(fe_hz, ref_hz)):
            ax.annotate(f"{(a - b) / b * 100:+.2f}%", (i + 1, b),
                        textcoords="offset points", xytext=(0, -12),
                        ha="center", fontsize=6)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("mode number")
        ax.set_ylabel("frequency [Hz]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("First six plate frequencies, h/L = 0.01 (percentages: FE vs reference)")
    figures.append(("frequencies.png",
                    "First six natural frequencies for SS, CC and CF plates with "
                    "the relative error of every mode.", _fig_png(fig)))

    # deflection field + centreline comparison
    h_plot = 0.02 * LENGTH
    field = face.solve_plate_static(
        elements_x=24, elements_y=24, length=LENGTH, width=WIDTH, thickness=h_plot,
        young=MATERIAL["young"], poisson=MATERIAL["poisson"], load=LOAD,
        boundary="SS", shear_treatment="sri",
    )
    grid = field.deflection_field.reshape(25, 25)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    im = ax1.contourf(np.linspace(0, LENGTH, 25), np.linspace(0, WIDTH, 25),
                      grid * 1e3, levels=18, cmap="viridis")
    fig.colorbar(im, ax=ax1, label="w [mm]")
    ax1.set_title("FE deflection field, SS plate, h/L = 0.02")
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ref = face.kirchhoff_ss_uniform_load(
        length=LENGTH, width=WIDTH, thickness=h_plot,
        young=MATERIAL["young"], poisson=MATERIAL["poisson"], load=LOAD,
    )
    xs = np.linspace(0, LENGTH, 25)
    exact = [ref.deflection(x, WIDTH / 2, length=LENGTH, width=WIDTH,
                            young=MATERIAL["young"], poisson=MATERIAL["poisson"],
                            thickness=h_plot, load=LOAD) for x in xs]
    ax2.plot(xs, grid[:, 12] * 1e3, "o-", label="FE centreline")
    ax2.plot(xs, np.array(exact) * 1e3, "k--", label="Kirchhoff closed form")
    ax2.set_xlabel("x [m] at y = 0.5 m")
    ax2.set_ylabel("w [mm]")
    ax2.set_title("Centreline deflection vs closed form (h/L = 0.02)")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    figures.append(("deflection_field.png",
                    "FE deflection field of the simply supported plate and the "
                    "centreline comparison with the Kirchhoff Navier solution.",
                    _fig_png(fig)))

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.plot([1e-4, 1e-2, 1.0], [r["f1_hz"] / control[0]["f1_hz"] for r in control][::-1],
            "o-", label="f1/f1(rho0)")
    dens = np.array([1e-4, 1e-2, 1.0])
    ax.plot(dens, 1 / np.sqrt(dens), "k--", label="1/sqrt(rho) scaling")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("density ratio rho/rho0")
    ax.set_ylabel("frequency ratio")
    ax.set_title("Negative control: density -> 0 (C5)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    figures.append(("density_control.png",
                    "Negative control: the fundamental frequency follows the exact "
                    "1/sqrt(rho) scaling, so a mass matrix that ignored density "
                    "would fail this test.", _fig_png(fig)))

    for name, caption, png in figures:
        path = OUT / name
        path.write_bytes(base64.b64decode(png))
        report.setdefault("figures", []).append(
            {"name": name, "caption": caption, "path": str(path.relative_to(OUT.parents[1])),
             "sha256": hashlib.sha256(base64.b64decode(png)).hexdigest()}
        )

    # -- provenance ---------------------------------------------------------
    report["provenance"] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds": time.time() - t_start,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "module": "tensorfem.structural_mindlin_face",
        "command": "python benchmarks/structural_mindlin_vibration_face.py",
        "thread_env": {k: os.environ.get(k) for k in
                       ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
    }
    report["criteria"]["C6_end_to_end_json"] = {
        "pass": all(key in report for key in ("deflection", "stress",
                                              "natural_frequencies",
                                              "convergence_rate",
                                              "boundary_conditions")),
        "keys": sorted(report.keys()),
    }

    out_json = OUT / "report.json"
    out_json.write_text(json.dumps(report, indent=2))
    summary = {
        "C1_max_error": criteria["C1_thin_deflection_vs_kirchhoff"]["value"],
        "C2_ratio": criteria["C2_thick_deflection_exceeds_kirchhoff"]["value"],
        "C3_order": criteria["C3_convergence_order"]["value"],
        "C4_max_error": criteria["C4_plate_frequencies"]["value"],
        "C5_pass": criteria["C5_density_negative_control"]["pass"],
        "json": str(out_json),
        "wall_seconds": report["provenance"]["wall_seconds"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())