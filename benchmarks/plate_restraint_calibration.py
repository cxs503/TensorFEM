#!/usr/bin/env python
"""Calibrate the plate edge rotational restraint of an ice-belt panel and measure
what it does to the panel response.

Turns scantlings into ``k_rot`` -> ``zeta`` -> ``beta``
(:mod:`tensorfem.plate_restraint_calibration`), then runs

* the Mindlin Q4 restrained panel (:mod:`tensorfem.plate_elastic_restraint`),
* the exact/variational Kirchhoff references
  (:func:`tensorfem.plate_levy_restraint.ritz_restrained_plate`),

on the JOUR-ICE-STRUCT design panel, for an *assumption matrix* of stiffener and
frame scantlings plus the two historical extremes (SSSS, clamped).  Every number
is written to ``benchmarks/artifacts/plate_restraint_calibration.json``; the
figure is written next to it.

Usage
-----
    python benchmarks/plate_restraint_calibration.py            # full (nx=160)
    python benchmarks/plate_restraint_calibration.py --nx 40 --quick
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorfem.plate_elastic_restraint import (  # noqa: E402
    peak_bending_stress,
    rect_patch_pressure,
    solve_restrained_panel,
)
from tensorfem.plate_levy_restraint import (  # noqa: E402
    flexural_rigidity,
    kirchhoff_strip_end_moment,
    ritz_restrained_plate,
)
from tensorfem.plate_restraint_calibration import (  # noqa: E402
    StiffenerSection,
    panel_restraint,
    shear_modulus,
    torsion_beam_restraint,
)

ARTIFACTS = ROOT / "benchmarks" / "artifacts"
BASELINE = ROOT / "tests" / "data" / "panel_patch_load_reference.json"
INF = math.inf

# ---- the paper's design panel (JOUR-ICE-STRUCT) ---------------------------
L_PANEL = 0.8          # frame spacing  [m]
S_PANEL = 0.7          # longitudinal spacing [m]
T_PLATE = 0.022        # plating [m]
E_STEEL = 2.06e11
NU = 0.3
P_REF = 1.0e6          # patch pressure [Pa]
PATCH_C = 0.1 * L_PANEL  # patch chord = 0.1 l  [m]

# ---- member options -------------------------------------------------------
LONG_HP = StiffenerSection(0.210, 0.010, 0.040, 0.022, name="HP220x10 (web+bulb)")
LONG_FB = StiffenerSection(0.200, 0.012, name="FB200x12")
FRAME_T = StiffenerSection(0.400, 0.012, 0.100, 0.018, name="T400x12+100x18")
FRAME_SPAN_GLOBAL = 3.2  # girder spacing: the frame twists over several spaces


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _patch_pressure():
    return rect_patch_pressure(P_REF, 0.5 * (L_PANEL - PATCH_C),
                               0.5 * (L_PANEL + PATCH_C), 0.0, S_PANEL)


def _panel(nx: int, beta_x: float, beta_y: float, *, clamped: bool = False) -> dict:
    ny = int(round(nx * S_PANEL / L_PANEL))
    kwargs = dict(boundary="clamped") if clamped else dict(beta_x=beta_x, beta_y=beta_y)
    res = solve_restrained_panel(nx, ny, L_PANEL, S_PANEL, E_STEEL, NU, T_PLATE,
                                 _patch_pressure(), **kwargs)
    return peak_bending_stress(res)


def _ritz(nu: float, kx: float, ky: float, modes: int) -> dict:
    D = flexural_rigidity(E_STEEL, nu, T_PLATE)
    sol = ritz_restrained_plate(L_PANEL, S_PANEL, D, nu, kx, ky, P_REF,
                                patch=(0.5 * (L_PANEL - PATCH_C),
                                       0.5 * (L_PANEL + PATCH_C), 0.0, S_PANEL),
                                m_modes=modes, n_modes=modes)
    mx, my = sol.moments_centre()
    return {"w_centre": sol.w_centre(), "Mx_centre": mx, "My_centre": my,
            "sigma_x_centre": 6.0 * mx / T_PLATE**2,
            "sigma_vm_centre": 6.0 / T_PLATE**2
            * math.sqrt(mx * mx - mx * my + my * my), "modes": modes}


def _variants() -> dict:
    """Assumption matrix: scantlings/end-fixity/span -> calibrated beta."""
    out = {}
    combos = {
        # upper bound of fixity: stiffest member, torsionally clamped ends, and
        # the frame held at every longitudinal it crosses
        "upper": dict(longitudinal=LONG_HP, frame=FRAME_T,
                      longitudinal_ends=INF, frame_ends=INF, frame_span=None),
        # same, but the bulb flange split into the library's two-wall tree model
        "upper_Jsplit": dict(longitudinal=LONG_HP, frame=FRAME_T,
                             longitudinal_ends=INF, frame_ends=INF, frame_span=None,
                             flange_split=True),
        # mid: clamped stiffener, but the frame twists over its real hold
        # spacing (girder spacing), i.e. much softer in torsion
        "mid_frame_global": dict(longitudinal=LONG_HP, frame=FRAME_T,
                                 longitudinal_ends=INF, frame_ends=INF,
                                 frame_span=FRAME_SPAN_GLOBAL),
        # lower bound: flat-bar longitudinal free at one frame, frame twisting
        # over the girder spacing with one end free
        "lower": dict(longitudinal=LONG_FB, frame=FRAME_T,
                      longitudinal_ends=(INF, 0.0), frame_ends=(INF, 0.0),
                      frame_span=FRAME_SPAN_GLOBAL),
        # lower bound with the bare thin-wall J (no rectangle correction)
        "lower_thinwall_J": dict(longitudinal=LONG_FB, frame=FRAME_T,
                                 longitudinal_ends=(INF, 0.0), frame_ends=(INF, 0.0),
                                 frame_span=FRAME_SPAN_GLOBAL, correction=False),
    }
    for name, cfg in combos.items():
        cfg = dict(cfg)
        flange_split = cfg.pop("flange_split", False)
        correction = cfg.pop("correction", True)
        pr = panel_restraint(young=E_STEEL, poisson=NU, plate_thickness=T_PLATE,
                             side_x=L_PANEL, side_y=S_PANEL,
                             correction=correction, **cfg)
        rec = pr.as_dict()
        rec["variant"] = name
        rec["assumption"] = {k: (v if not isinstance(v, tuple) else list(v))
                             for k, v in cfg.items() if k not in ("longitudinal", "frame")}
        rec["assumption"]["longitudinal"] = cfg["longitudinal"].name
        rec["assumption"]["frame"] = cfg["frame"].name
        rec["flange_split"] = bool(flange_split)
        rec["correction"] = bool(correction)
        j_hi = cfg["longitudinal"].torsion_constant()
        j_lo = cfg["longitudinal"].torsion_constant(flange_split=True) \
            if not cfg["longitudinal"].is_flat_bar else j_hi
        rec["J_longitudinal_hi"] = j_hi
        rec["J_longitudinal_lo"] = j_lo
        rec["J_longitudinal_thinwall"] = cfg["longitudinal"].torsion_constant(
            correction=False)
        rec["J_frame"] = cfg["frame"].torsion_constant()
        out[name] = {"pr": pr, "rec": rec}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=160)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--modes", type=int, default=40)
    ap.add_argument("--out", type=Path,
                    default=ARTIFACTS / "plate_restraint_calibration.json")
    args = ap.parse_args()

    t_start = time.time()
    baseline = json.loads(BASELINE.read_text())
    row = next(r for r in baseline["plate_vs_strip"]
               if r["lambda_c_over_l"] == 0.1 and r["patch_mode"] == "full_breadth")
    sigma_baseline = row["sigma_plate_MPa"]
    D = flexural_rigidity(E_STEEL, NU, T_PLATE)
    G = shear_modulus(E_STEEL, NU)

    doc: dict = {
        "case": {
            "name": "JOUR-ICE-STRUCT design panel (patch load, full breadth)",
            "panel_x_m": L_PANEL, "panel_y_m": S_PANEL, "plate_m": T_PLATE,
            "young_Pa": E_STEEL, "poisson": NU, "shear_Pa": G,
            "D_Nm": D, "patch_pressure_Pa": P_REF, "patch_chord_m": PATCH_C,
            "baseline_sigma_vm_MPa": sigma_baseline,
            "baseline_source": {
                "file": "tests/data/panel_patch_load_reference.json",
                "sha256": _sha256(BASELINE),
                "upstream": baseline["provenance"]["source_repository_file"],
                "method": baseline["provenance"]["method_of_baseline"],
            },
            "nx": args.nx,
        },
        "provenance": {
            "git_sha": _git_sha(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "generated_unix": time.time(),
            "module": "tensorfem.plate_restraint_calibration",
        },
    }

    # ---- 1. member J and its idealisation band ----------------------------
    doc["sections"] = {
        "longitudinal_HP220x10": {
            "J_corrected_m4": LONG_HP.torsion_constant(),
            "J_flange_split_m4": LONG_HP.torsion_constant(flange_split=True),
            "J_thinwall_m4": LONG_HP.torsion_constant(correction=False),
            "GJ_corrected_Nm2": G * LONG_HP.torsion_constant(),
        },
        "longitudinal_FB200x12": {
            "J_corrected_m4": LONG_FB.torsion_constant(),
            "J_thinwall_m4": LONG_FB.torsion_constant(correction=False),
            "GJ_corrected_Nm2": G * LONG_FB.torsion_constant(),
        },
        "frame_T400x12_100x18": {
            "J_corrected_m4": FRAME_T.torsion_constant(),
            "J_thinwall_m4": FRAME_T.torsion_constant(correction=False),
            "GJ_corrected_Nm2": G * FRAME_T.torsion_constant(),
        },
    }

    # ---- 2. calibration assumption matrix --------------------------------
    variants = _variants()
    doc["calibration"] = {k: v["rec"] for k, v in variants.items()}

    # ---- 3. panel response per variant -----------------------------------
    responses: dict = {}
    for name, item in variants.items():
        pr, rec = item["pr"], item["rec"]
        fe = _panel(args.nx, pr.beta_x, pr.beta_y)
        rz = _ritz(NU, pr.k_rot_x, pr.k_rot_y, args.modes)
        responses[name] = {
            "beta_x": pr.beta_x, "beta_y": pr.beta_y,
            "sigma_vm_centre_MPa_FE": fe["centre_von_mises"] / 1e6,
            "w_centre_m_FE": fe["w_centre"],
            "Mx_centre_Nm_per_m_FE": fe["centre_Mx"],
            "sigma_vm_centre_MPa_Ritz": rz["sigma_vm_centre"] / 1e6,
            "w_centre_m_Ritz": rz["w_centre"],
            "Mx_centre_Nm_per_m_Ritz": rz["Mx_centre"],
            "FE_vs_Ritz_sigma_pct": 100.0 * (fe["centre_von_mises"] / 1e6
                                             / (rz["sigma_vm_centre"] / 1e6) - 1.0),
            "FE_vs_Ritz_w_pct": 100.0 * (fe["w_centre"] / rz["w_centre"] - 1.0),
            "sigma_change_vs_SSSS_pct": 100.0 * (fe["centre_von_mises"] / 1e6
                                                 / sigma_baseline - 1.0),
        }
    doc["response"] = responses

    # ---- 4. the two historical extremes ----------------------------------
    extremes = {}
    ssss_fe = _panel(args.nx, 0.0, 0.0)
    clamp_fe = _panel(args.nx, 0.0, 0.0, clamped=True)
    extremes["ssss"] = {
        "sigma_vm_centre_MPa_FE": ssss_fe["centre_von_mises"] / 1e6,
        "w_centre_m_FE": ssss_fe["w_centre"],
        "vs_baseline_pct": 100.0 * (ssss_fe["centre_von_mises"] / 1e6
                                    / sigma_baseline - 1.0),
    }
    extremes["clamped"] = {
        "sigma_vm_centre_MPa_FE": clamp_fe["centre_von_mises"] / 1e6,
        "w_centre_m_FE": clamp_fe["w_centre"],
        "vs_baseline_pct": 100.0 * (clamp_fe["centre_von_mises"] / 1e6
                                    / sigma_baseline - 1.0),
    }
    doc["extremes"] = extremes

    # ---- 5. beta scan (machine readable band) ----------------------------
    betas = [0.0, 0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0] if not args.quick else [0.0, 0.5, 1.0]
    scan = []
    for beta in betas:
        fe = _panel(args.nx, beta, beta)
        scan.append({"beta": beta, "sigma_vm_centre_MPa": fe["centre_von_mises"] / 1e6,
                     "w_centre_m": fe["w_centre"]})
    doc["beta_scan_diagonal"] = scan

    # ---- 6. Ritz mode convergence at one calibrated state ----------------
    conv = []
    ref = variants["mid_frame_global"]["rec"]
    for modes in ((20, 30, 40, 60) if not args.quick else (20, 30)):
        rz = _ritz(NU, ref["k_rot_x_Nm_per_rad_per_m"],
                   ref["k_rot_y_Nm_per_rad_per_m"], modes)
        conv.append({"modes": modes, "sigma_vm_centre_MPa": rz["sigma_vm_centre"] / 1e6,
                     "w_centre_m": rz["w_centre"]})
    doc["ritz_convergence"] = conv

    # ---- 7. strip limit cross-check --------------------------------------
    strip = {}
    for name, item in variants.items():
        rec = item["rec"]
        k = rec["k_rot_y_Nm_per_rad_per_m"]
        strip[name] = {
            "k_rot_y": k,
            "M_end_exact": kirchhoff_strip_end_moment(P_REF, S_PANEL, D, k),
            "M_end_clamped": kirchhoff_strip_end_moment(P_REF, S_PANEL, D, INF),
            "M_end_simple": kirchhoff_strip_end_moment(P_REF, S_PANEL, D, 0.0),
        }
    doc["strip_limit"] = strip

    # ---- 8. torsion-beam sanity ------------------------------------------
    doc["torsion_beam_limits"] = {
        "12GJ/L2_longitudinal_HP": torsion_beam_restraint(
            G * LONG_HP.torsion_constant(), L_PANEL, INF, INF).k_rot,
        "3GJ/L2_longitudinal_FB": torsion_beam_restraint(
            G * LONG_FB.torsion_constant(), L_PANEL, INF, 0.0).k_rot,
    }

    # ---- 9. error verdicts ------------------------------------------------
    fe_ssss = extremes["ssss"]["sigma_vm_centre_MPa_FE"]
    verdicts = []

    def _verdict(label: str, value_pct: float, tol: float) -> None:
        verdicts.append({"check": label, "error_pct": value_pct, "tolerance_pct": tol,
                         "pass": abs(value_pct) <= tol})

    _verdict("FE beta=0 centre sigma_vM vs Navier N=401 baseline", 
             100.0 * (fe_ssss / sigma_baseline - 1.0), 0.5)
    # FE (Mindlin, nx=160) vs Ritz (Kirchhoff, 40x40 modes): two different plate
    # theories, so the comparison is a consistency check on the k_rot -> beta
    # convention, not a convergence test; 5% is the stated model-difference band
    for name, rec in responses.items():
        _verdict(f"calibrated '{name}': Mindlin FE vs Kirchhoff Ritz (sigma_vM)",
                 rec["FE_vs_Ritz_sigma_pct"], 5.0)
        _verdict(f"calibrated '{name}': Mindlin FE vs Kirchhoff Ritz (w)",
                 rec["FE_vs_Ritz_w_pct"], 5.0)
    _verdict("Ritz mode convergence (finest two retained sets, sigma_vM)",
             100.0 * (conv[-2]["sigma_vm_centre_MPa"]
                      / conv[-1]["sigma_vm_centre_MPa"] - 1.0), 1.0)
    doc["verdicts"] = verdicts
    doc["runtime_s"] = time.time() - t_start

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2))
    print(f"wrote {args.out}  ({doc['runtime_s']:.1f} s)")

    if not args.quick:
        _figure(doc, args.out.with_name("plate_restraint_calibration.png"))
    return 0


def _figure(doc: dict, path: Path) -> None:  # pragma: no cover - plotting
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    ax = axes[0]
    scan = doc["beta_scan_diagonal"]
    b = [p["beta"] for p in scan]
    s = [p["sigma_vm_centre_MPa"] for p in scan]
    ax.plot(b, s, "o-", color="#1f4e79", label="Mindlin FE, $\\beta_x=\\beta_y$ scan")
    base = doc["case"]["baseline_sigma_vm_MPa"]
    ax.axhline(base, ls="--", color="#888", lw=1.2,
               label=f"Navier N=401 SSSS baseline {base:.2f} MPa")
    calib = doc["response"]
    for name, style in (("upper", "s"), ("mid_frame_global", "D"), ("lower", "^")):
        if name in calib:
            rec = calib[name]
            ax.plot([rec["beta_x"]], [rec["sigma_vm_centre_MPa_FE"], ], style,
                    color="#c00000", ms=7, mfc="none",
                    label="calibrated cases" if name == "upper" else None)
            ax.annotate(name, (rec["beta_x"], rec["sigma_vm_centre_MPa_FE"]),
                        textcoords="offset points", xytext=(6, 6), fontsize=7,
                        color="#c00000")
    ax.set_xlabel("embedding coefficient $\\beta$ (corner / edge, diagonal)")
    ax.set_ylabel("centre $\\sigma_{vM}$ [MPa]")
    ax.set_title("panel response band (0.8 m x 0.7 m x 22 mm, 1 MPa patch)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    names = [n for n in ("lower", "lower_thinwall_J", "mid_frame_global",
                         "upper_Jsplit", "upper") if n in calib]
    xpos = np.arange(len(names))
    beta_y = [calib[n]["beta_y"] for n in names]
    beta_x = [calib[n]["beta_x"] for n in names]
    ax.bar(xpos - 0.2, beta_y, width=0.38, color="#2e75b6",
           label="$\\beta_y$ (longitudinals)")
    ax.bar(xpos + 0.2, beta_x, width=0.38, color="#a9c4e4",
           label="$\\beta_x$ (frames)")
    ax.set_xticks(xpos)
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=7)
    ax.set_ylabel("calibrated $\\beta$")
    ax.set_ylim(0, 1.0)
    ax.set_title("scantling idealisation band")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    print(f"wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())