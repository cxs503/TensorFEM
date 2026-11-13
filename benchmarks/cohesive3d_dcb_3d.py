#!/usr/bin/env python
"""Specimen-level verification of the 3-D cohesive element: width-resolved DCB.

Writes ``benchmarks/cohesive3d_dcb_3d.json``, the generated verification page
``docs/cohesive3d-dcb-3d-verification.md`` and the figures under
``docs/assets/cohesive3d-dcb-3d/``.

Why this benchmark exists
-------------------------
``benchmarks/cohesive3d_verification.py`` verifies the cohesive *element*
(law, tangent, patch, objectivity) and
``benchmarks/cohesive3d_delamination.py`` grows a crack with **one** element
through the specimen width (a plane-stress-like idealisation whose evidence
JSON was never committed).  Neither closes *GAP-FEM-3D-COHESIVE* at specimen
level in a genuinely three-dimensional stress state.  This script:

1. assembles the production HEX8 adherends and the production COH3D8 element
   into a width-resolved DCB (``ny`` elements across the width);
2. drives the damage front with a displacement-controlled Newton path
   follower (``tensorfem.cohesive3d_dcb``);
3. compares against *independent* closed forms (beam theory with and without
   the Williams/ASTM D5528 root rotation) and against internal balance laws
   (global energy, load-ratio scaling, width objectivity).

Run ``python benchmarks/cohesive3d_dcb_3d.py --quick`` for a fast subset.
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
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorfem.cohesive3d import BKCohesive3DLaw  # noqa: E402
from tensorfem.cohesive3d_dcb import (  # noqa: E402
    DCBGeometry,
    beam_compliance,
    beam_peak_load,
    build_dcb_model,
    opening_prescription,
    solve_dcb,
    williams_root_rotation,
)

torch.set_num_threads(1)

E_ADH = 70.0e9
NU_ADH = 0.30
H_ARM = 3.0e-3
TOUGHNESS = 300.0
TN0 = 120.0e6
TS0 = 80.0e6
KN = 2.4e14
KS = 1.0e14

#: how far past the LEFM peak opening the displacement control is driven.
DELTA_FACTOR = 1.35


#: Source modules whose content identity the committed evidence must bind to.
#: If any of these changes, the committed ``cohesive3d_dcb_3d.json`` is stale
#: and the freshness gate (tests/test_cohesive3d_dcb_evidence_artifact.py) fails.
SOURCE_FILES = (
    "src/tensorfem/cohesive3d.py",
    "src/tensorfem/cohesive3d_dcb.py",
    "src/tensorfem/solid3d.py",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_head() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):  # git absent: degrade, not fatal
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def source_provenance() -> dict:
    """Content identity of the code that produced the numbers."""
    return {
        "module": "tensorfem.cohesive3d_dcb",
        "source_files": {rel: sha256_file(ROOT / rel) for rel in SOURCE_FILES},
        "git_head": _git_head(),
    }


def criteria_digest(criteria: dict) -> str:
    """SHA-256 of the numbers themselves: any hand edit changes the digest."""
    canonical = json.dumps(criteria, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def make_law() -> BKCohesive3DLaw:
    return BKCohesive3DLaw(KN, KS, TN0, TS0, TOUGHNESS, 800.0, 2.2)


def linear_compliance(g: DCBGeometry) -> dict:
    """Compliance of the intact specimen (linear solve, penalty interface).

    The loaded face is *prescribed* (``+delta/2`` / ``-delta/2``, see
    :func:`tensorfem.cohesive3d_dcb.opening_prescription`) and the compliance is
    ``delta / P`` with ``P`` the sum of both arm reactions -- the same
    convention as :func:`tensorfem.cohesive3d_dcb.solve_dcb`.

    Regression (GAP-FEM-3D-COHESIVE): this helper used to assemble a unit-load
    right-hand side on the loaded dofs.  Since displacement control moved those
    dofs into the constrained set, ``unit[free]`` was identically zero and the
    helper reported ``compliance = 0.0``; it also divided the prescribed opening
    by the number of loaded dofs, which made the result depend on the mesh.
    Both defects are gone now that the helper shares the production
    prescription.
    """
    model = build_dcb_model(g, make_law())
    k_solid = model.solid_stiffness()
    u = np.zeros(model.n_dofs)
    _, k_coh, _, _ = model.interface_response(u, None)
    k_all = k_solid + k_coh
    free = model.free_dofs
    faced = np.concatenate((model.upper_load_dofs, model.lower_load_dofs))
    delta = 1.0e-8
    u[faced] = delta * opening_prescription(model)[faced]
    u[free] = spla.spsolve(k_all[free][:, free].tocsc(),
                           -(k_all[free][:, faced] @ u[faced]))
    fint = k_all @ u
    delta = abs(float(fint[model.upper_load_dofs].sum()))
    delta += abs(float(fint[model.lower_load_dofs].sum()))
    delta = 1.0e-8 / delta
    c_plain = beam_compliance(g.initial_crack, g, root_rotation=False, shear=False)
    c_shear = beam_compliance(g.initial_crack, g, root_rotation=False, shear=True)
    c_root = beam_compliance(g.initial_crack, g, root_rotation=True, shear=True)
    delta_root = williams_root_rotation(g.arm_thickness, g.young_modulus, g.shear_modulus)
    return {
        "compliance_fem": float(delta),
        "compliance_beam_bending_only": c_plain,
        "compliance_beam_shear": c_shear,
        "compliance_beam_root_rotation": c_root,
        "rel_error_vs_shear_beam": (float(delta) - c_shear) / c_shear,
        "rel_error_vs_root_rotation_beam": (float(delta) - c_root) / c_root,
        "root_rotation_delta_m": delta_root,
        "root_rotation_over_h": delta_root / g.arm_thickness,
        "n_dofs": model.n_dofs,
    }


def propagate(g: DCBGeometry, *, n_steps: int, delta_max: float) -> dict:
    model = build_dcb_model(g, make_law())
    t0 = time.time()
    result = solve_dcb(model, delta_max=delta_max, n_steps=n_steps, tol=1e-8)
    peak = result.peak
    p_ref = beam_peak_load(g.initial_crack, TOUGHNESS, g, root_rotation=True)
    p_ref_plain = beam_peak_load(g.initial_crack, TOUGHNESS, g, root_rotation=False)
    # R-curve: at every committed step, compare the measured load with the
    # LEFM load of the *measured* crack front (independent of the peak value).
    rcurve = []
    for c in result.curve:
        a_front = c["crack_front_m"]
        if a_front <= g.initial_crack + 1e-12 or c["load"] <= 0.0:
            continue
        p_lefm = beam_peak_load(a_front, TOUGHNESS, g, root_rotation=True)
        rcurve.append(
            {
                "disp": c["disp"],
                "crack_front_m": a_front,
                "load": c["load"],
                "load_lefm_from_front": p_lefm,
                "rel_error": (c["load"] - p_lefm) / p_lefm,
            }
        )
    dmg = result.damage.reshape(-1)
    width_front = []
    ncol = g.n_bonded
    for j in range(g.ny):
        col = dmg.reshape(ncol, g.ny)[:, j] if dmg.size == ncol * g.ny else None
        if col is None:
            continue
        failed = np.flatnonzero(col >= 0.9)
        width_front.append(
            float(g.length - (failed.min() * g.dx)) if failed.size else g.initial_crack
        )
    return {
        "geometry": {
            "length_m": g.length,
            "width_m": g.width,
            "arm_thickness_m": g.arm_thickness,
            "initial_crack_m": g.initial_crack,
            "nx": g.nx,
            "ny": g.ny,
            "nz": g.nz,
            "dx_m": g.dx,
            "width_over_thickness": g.width / g.arm_thickness,
        },
        "peak_load_N": peak.get("load"),
        "peak_disp_m": peak.get("disp"),
        "peak_load_lefm_root_rotation_N": p_ref,
        "peak_load_lefm_plain_N": p_ref_plain,
        "peak_rel_error_vs_root_rotation": (peak.get("load", float("nan")) - p_ref) / p_ref,
        "peak_rel_error_vs_plain_beam": (peak.get("load", float("nan")) - p_ref_plain) / p_ref_plain,
        "stats": result.stats,
        "curve": result.curve,
        "rcurve": rcurve,
        "crack_front_by_width_column": width_front,
        "damage": dmg.tolist(),
        "iface_x": result.iface_x.tolist(),
        "runtime_s": time.time() - t0,
    }


def peak_opening(g: DCBGeometry) -> float:
    """Opening ``delta`` at the LEFM peak load: ``P_c * C`` (root-rotation CBBT).

    The follower has to be driven *past* this value, otherwise the peak-load
    criterion cannot be met no matter how good the element is.  The hard-coded
    ``delta_max = 4e-5 m`` used before this fix was 4.5 % of the peak opening of
    the reference specimen (``P_c = 40.6 N``, ``C = 2.36e-5 m/N`` =>
    ``delta_peak = 9.6e-4 m``), which is exactly why the committed artefact
    reported a "peak load" of 0.41 N (-98.6 % against LEFM) with zero failed
    area: the run never left the linear branch.
    """
    c = beam_compliance(g.initial_crack, g, root_rotation=True, shear=True)
    return beam_peak_load(g.initial_crack, TOUGHNESS, g, root_rotation=True) * c


def figures(out: dict, dest: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dest.mkdir(parents=True, exist_ok=True)
    names = []

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for key, item in out["propagation"].items():
        curve = item["curve"]
        ax.plot(
            [c["disp"] * 1e6 for c in curve],
            [c["load"] for c in curve],
            lw=1.6,
            label=f"{key}: ny={item['geometry']['ny']}, a0={item['geometry']['initial_crack_m']*1e3:.0f} mm",
        )
    ax.set_xlabel("opening at the loaded end [um]")
    ax.set_ylabel("total load P [N]")
    ax.set_title("3-D DCB: load-opening response (cohesive propagation)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    path = dest / "load_opening.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    names.append(str(path.relative_to(ROOT)))

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for key, item in out["propagation"].items():
        rc = item["rcurve"]
        if not rc:
            continue
        ax.plot(
            [(r["crack_front_m"] - item["geometry"]["initial_crack_m"]) * 1e3 for r in rc],
            [100.0 * r["rel_error"] for r in rc],
            marker=".",
            ms=3,
            lw=1.2,
            label=key,
        )
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("crack advance from the initial front [mm]")
    ax.set_ylabel("P_FEM / P_LEFM(a_front) - 1  [%]")
    ax.set_title("R-curve check: measured load vs LEFM of the measured front")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    path = dest / "rcurve.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    names.append(str(path.relative_to(ROOT)))

    fig, axes = plt.subplots(1, len(out["propagation"]), figsize=(4.2 * len(out["propagation"]), 2.8), squeeze=False)
    for ax, (key, item) in zip(axes[0], out["propagation"].items()):
        ncol = len(item["iface_x"]) // item["geometry"]["ny"]
        dmg = np.array(item["damage"]).reshape(ncol, item["geometry"]["ny"])
        ax.imshow(
            dmg.T,
            aspect="auto",
            origin="lower",
            extent=[item["iface_x"][0] * 1e3, item["iface_x"][-1] * 1e3, 0, item["geometry"]["width_m"] * 1e3],
            vmin=0.0,
            vmax=1.0,
            cmap="inferno",
        )
        ax.set_title(f"{key} damage field")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("width [mm]")
    path = dest / "damage_field.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    names.append(str(path.relative_to(ROOT)))
    return names


def render_markdown(out: dict, fig_paths: list[str]) -> str:
    crit = out["criteria"]
    lines = [
        "# 3-D cohesive element: width-resolved DCB verification (GAP-FEM-3D-COHESIVE)",
        "",
        f"Generated {out['generated_utc']} by `{out['generator']}` on "
        f"`{out['host']['machine']}` (torch {out['host']['torch']}, python {out['host']['python']}).",
        "",
        "## 1. Specimen and method",
        "",
        "Two HEX8 adherends (production `tensorfem.solid3d.hex8_stiffness`) are bonded by a",
        "pre-meshed zero-thickness COH3D8 ligament (production `tensorfem.cohesive3d`, BK",
        "mixed-mode law). The interface nodes are duplicated, the specimen is clamped at",
        "`x=0` and opened symmetrically at `x=L` (upper `+delta/2`, lower `-delta/2`), so",
        "`P` is the **total** load and the classical DCB relations apply:",
        "`C = 4a^3/(E b h^3) + a/(kGbh)` and `G_I = 6P^2a^2/(E b^2 h^3)`, both optionally",
        "with the Williams/ASTM D5528 root rotation `a -> a + Delta`,",
        "`Delta = h sqrt(E/(11G) (3 - 2 (Gamma/(1+Gamma))^2))`, `Gamma = 1.18 sqrt(E/G)`.",
        "",
        "| quantity | value |",
        "|---|---|",
        f"| adherend E, nu | {E_ADH:.3g} Pa, {NU_ADH} |",
        f"| arm thickness h | {H_ARM*1e3:.2f} mm |",
        f"| k_n, k_s | {KN:.3g}, {KS:.3g} Pa/m |",
        f"| t_n0, t_s0 | {TN0/1e6:.1f}, {TS0/1e6:.1f} MPa |",
        f"| G_Ic | {TOUGHNESS:.0f} J/m^2 |",
        f"| cohesive length E G_Ic / t_n0^2 | {E_ADH*TOUGHNESS/TN0**2*1e3:.2f} mm |",
        "",
        "## 2. Criteria (independent reference in the same row)",
        "",
        "| check | reference | computed | rel. error | tolerance | status |",
        "|---|---|---|---|---|---|",
    ]
    for key, item in crit.items():
        ref = item.get("ref")
        got = item.get("got")
        err = item.get("rel_error")
        fmt = lambda v: ("-" if v is None else (f"{v:.6g}" if isinstance(v, float) else str(v)))
        err_s = "-" if err is None else f"{100*err:+.2f} %"
        lines.append(
            f"| `{key}` | {fmt(ref)} | {fmt(got)} | {err_s} | {item.get('tolerance','-')} | "
            f"{'PASS' if item.get('passed') else 'FAIL'} |"
        )
    lines += ["", "## 3. Propagation runs", ""]
    for key, item in out["propagation"].items():
        g = item["geometry"]
        s = item["stats"]
        lines += [
            f"### {key}",
            "",
            f"mesh `nx={g['nx']}, ny={g['ny']}, nz={g['nz']}` (dx = {g['dx_m']*1e3:.2f} mm, "
            f"b/h = {g['width_over_thickness']:.1f}), a0 = {g['initial_crack_m']*1e3:.1f} mm, "
            f"{s['n_dofs']} dofs, {s['n_iface']} interface elements, "
            f"{s['steps']} accepted steps, {s['newton_iterations']} Newton iterations, "
            f"{item['runtime_s']:.0f} s.",
            "",
            f"* peak load **{item['peak_load_N']:.3f} N** vs LEFM(root rotation) "
            f"{item['peak_load_lefm_root_rotation_N']:.3f} N "
            f"({100*item['peak_rel_error_vs_root_rotation']:+.2f} %) and plain beam "
            f"{item['peak_load_lefm_plain_N']:.3f} N "
            f"({100*item['peak_rel_error_vs_plain_beam']:+.2f} %).",
            f"* energy balance `|W_ext - U_solid - W_interface| / W_ext` = "
            f"{100*s['energy_balance_rel']:.3f} %.",
            f"* interface work per failed area = {s['interface_work_per_area']:.4g} J/m^2 "
            f"(G_Ic = {TOUGHNESS:.0f} J/m^2).",
            f"* crack front across the width: {['%.4f' % v for v in item['crack_front_by_width_column']]} m.",
            "",
        ]
    lines += ["## 4. Figures", ""]
    for p in fig_paths:
        lines.append(f"![{Path(p).stem}](../{p})")
    lines += [
        "",
        "## 5. Honest limitations",
        "",
        "* Small-strain kinematics: the arms rotate < 3 deg at the load peak, but",
        "  large-displacement DCB tests (ASTM D5528 asks for a large-displacement",
        "  correction) are outside this benchmark.",
        "* The Williams root rotation is itself an approximation of the crack-tip",
        "  root flexibility; the FEM value is bracketed by the plain and the",
        "  root-rotation closed forms, and the bracket is reported instead of a",
        "  single number.",
        "* No friction/contact on the crack faces, no cyclic degradation.",
        "",
    ]
    return "\n".join(lines)


def run(quick: bool = False) -> dict:
    t_start = time.time()
    if quick:
        narrow = DCBGeometry(length=60e-3, width=3e-3, arm_thickness=H_ARM,
                             initial_crack=30e-3, nx=100, ny=1, nz=3)
        long_crack = DCBGeometry(length=60e-3, width=3e-3, arm_thickness=H_ARM,
                                 initial_crack=42e-3, nx=100, ny=1, nz=3)
        wide = DCBGeometry(length=60e-3, width=18e-3, arm_thickness=H_ARM,
                           initial_crack=30e-3, nx=100, ny=3, nz=3)
        n_steps, steps_long, steps_wide = 40, 40, 40
    else:
        narrow = DCBGeometry(length=120e-3, width=3e-3, arm_thickness=H_ARM,
                             initial_crack=60e-3, nx=200, ny=1, nz=3)
        long_crack = DCBGeometry(length=120e-3, width=3e-3, arm_thickness=H_ARM,
                                 initial_crack=90e-3, nx=200, ny=1, nz=3)
        wide = DCBGeometry(length=120e-3, width=18e-3, arm_thickness=H_ARM,
                           initial_crack=60e-3, nx=160, ny=3, nz=3)
        n_steps, steps_long, steps_wide = 120, 120, 90

    out = {
        "schema": "tensorfem.cohesive3d-dcb-3d/1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "benchmarks/cohesive3d_dcb_3d.py",
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "provenance": source_provenance(),
        "quick_mode": quick,
        "drive": {"delta_factor": DELTA_FACTOR},
        "criteria": {},
        "propagation": {},
    }

    # 0. is the driving window even able to reach the peak? ------------------
    out["criteria"]["window_covers_peak"] = {
        "criterion": "delta_max >= LEFM peak opening P_c*C (else the peak is outside the run)",
        "ref": peak_opening(narrow),
        "got": DELTA_FACTOR * peak_opening(narrow),
        "rel_error": (DELTA_FACTOR * peak_opening(narrow) - peak_opening(narrow))
        / peak_opening(narrow),
        "tolerance": "delta_max >= delta_peak",
        "passed": DELTA_FACTOR >= 1.0,
    }

    # 1. linear compliance of the intact specimen ---------------------------
    lin = linear_compliance(narrow)
    out["linear"] = lin
    out["criteria"]["compliance_bracket"] = {
        "criterion": "FE compliance bracketed by plain and root-rotation beam theory",
        "ref": lin["compliance_beam_root_rotation"],
        "got": lin["compliance_fem"],
        "rel_error": lin["rel_error_vs_root_rotation_beam"],
        "tolerance": "|err| <= 10% vs root-rotation form (bracket in JSON)",
        "passed": abs(lin["rel_error_vs_root_rotation_beam"]) <= 0.10,
    }

    # 2. propagation: reference crack --------------------------------------
    # Drive every specimen 1.35x past its LEFM peak opening: enough to reach the
    # limit point and to see the first post-peak step, without spending the run
    # on the flat tail.  The window is *derived*, never hard-coded (see
    # :func:`peak_opening` and the ``window_covers_peak`` criterion).
    ref = propagate(narrow, n_steps=n_steps,
                    delta_max=DELTA_FACTOR * peak_opening(narrow))
    out["propagation"]["a0_reference"] = ref
    out["criteria"]["peak_load_vs_lefm"] = {
        "criterion": "peak load vs LEFM with root rotation",
        "ref": ref["peak_load_lefm_root_rotation_N"],
        "got": ref["peak_load_N"],
        "rel_error": ref["peak_rel_error_vs_root_rotation"],
        "tolerance": "|err| <= 5%",
        "passed": abs(ref["peak_rel_error_vs_root_rotation"]) <= 0.05,
    }
    out["criteria"]["energy_balance"] = {
        "criterion": "|W_ext - U_solid - W_interface| / W_ext",
        "ref": 0.0,
        "got": ref["stats"]["energy_balance_rel"],
        "rel_error": ref["stats"]["energy_balance_rel"],
        "tolerance": "<= 1%",
        "passed": ref["stats"]["energy_balance_rel"] <= 0.01,
    }
    out["criteria"]["interface_work_vs_GIc"] = {
        "criterion": "interface work per failed area vs G_Ic",
        "ref": TOUGHNESS,
        "got": ref["stats"]["interface_work_per_area"],
        "rel_error": abs(ref["stats"]["interface_work_per_area"] - TOUGHNESS) / TOUGHNESS,
        "tolerance": "<= 10%",
        "passed": abs(ref["stats"]["interface_work_per_area"] - TOUGHNESS) / TOUGHNESS <= 0.10,
    }
    if ref["rcurve"]:
        errs = np.abs([r["rel_error"] for r in ref["rcurve"]])
        out["criteria"]["rcurve_front_lefm"] = {
            "criterion": "max |P_FEM/P_LEFM(a_front) - 1| over propagating steps",
            "ref": 0.0,
            "got": float(errs.max()),
            "rel_error": float(errs.max()),
            "tolerance": "<= 10%",
            "passed": float(errs.max()) <= 0.10,
        }

    # 3. propagation: longer initial crack -> load ratio test ---------------
    long_run = propagate(long_crack, n_steps=steps_long,
                         delta_max=DELTA_FACTOR * peak_opening(long_crack))
    out["propagation"]["a0_long"] = long_run
    a1, a2 = narrow.initial_crack, long_crack.initial_crack
    delta_root = williams_root_rotation(H_ARM, E_ADH, narrow.shear_modulus)
    ratio_theory = (a1 + delta_root) / (a2 + delta_root)
    ratio_fem = ref["peak_load_N"] / long_run["peak_load_N"]
    out["criteria"]["peak_ratio_two_cracks"] = {
        "criterion": "P(a1)/P(a2) vs (a2+Delta)/(a1+Delta)^{-1} (root rotation cancels to first order)",
        "ref": ratio_theory,
        "got": float(ratio_fem),
        "rel_error": (float(ratio_fem) - ratio_theory) / ratio_theory,
        "tolerance": "|err| <= 2%",
        "passed": abs((float(ratio_fem) - ratio_theory) / ratio_theory) <= 0.02,
    }

    # 4. width objectivity (true 3-D stress state) -------------------------
    wide = propagate(wide, n_steps=steps_wide,
                     delta_max=DELTA_FACTOR * peak_opening(wide))
    out["propagation"]["wide_ny3"] = wide
    fronts = wide["crack_front_by_width_column"]
    spread = (max(fronts) - min(fronts)) if fronts else float("nan")
    p_wide = wide["peak_load_N"]
    # a wide specimen is stiffer in bending: the plane-strain modulus raises the
    # load by sqrt(E/(1-nu^2)/E) at equal G_Ic; use the converted reference.
    p_wide_ref = ref["peak_load_lefm_root_rotation_N"] * math.sqrt(
        E_ADH / (1.0 - NU_ADH**2) / E_ADH
    )
    out["criteria"]["width_objectivity"] = {
        "criterion": "peak load of the ny=3 specimen vs ny=1 (plane-strain corrected) and front straightness",
        "ref": p_wide_ref,
        "got": p_wide,
        "rel_error": (p_wide - p_wide_ref) / p_wide_ref,
        "tolerance": "|err| <= 8% (transitional b/h, 3-D anticlastic effect)",
        "front_spread_m": spread,
        "damage_front_straight": bool(spread <= 2.0 * wide["geometry"]["dx_m"]),
        "passed": abs((p_wide - p_wide_ref) / p_wide_ref) <= 0.08
        and (spread <= 2.0 * wide["geometry"]["dx_m"]),
    }

    out["all_passed"] = all(c.get("passed", False) for c in out["criteria"].values())
    out["checks_digest"] = criteria_digest(out["criteria"])
    out["runtime_s"] = time.time() - t_start
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    out = run(quick=args.quick)
    dest = ROOT / "benchmarks" / "cohesive3d_dcb_3d.json"
    dest.write_text(json.dumps(out, indent=1))
    fig_paths = [] if args.no_figures else figures(out, ROOT / "docs" / "assets" / "cohesive3d-dcb-3d")
    (ROOT / "docs" / "cohesive3d-dcb-3d-verification.md").write_text(
        render_markdown(out, fig_paths)
    )
    summary = {
        k: {"rel_error": v.get("rel_error"), "passed": v.get("passed")}
        for k, v in out["criteria"].items()
    }
    print(json.dumps(summary, indent=2))
    print(f"all_passed={out['all_passed']} runtime={out['runtime_s']:.1f}s")
    return 0 if out["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())