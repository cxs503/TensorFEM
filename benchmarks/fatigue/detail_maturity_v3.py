"""Benchmark driver of the ship-hull detail fatigue maturity package (P106W3A2-N02).

Runs the complete global-to-detail fatigue chain across the three acceptance
joint families (T-joint, butt, corner/bracket toe), writes the machine-readable
ledger and the figures of the report, and prints the criterion table.

The driver is deterministic (no random seeds) and needs no network.

Usage::

    python benchmarks/fatigue/detail_maturity_v3.py [--out artifacts/detail_fatigue_v3]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from tensorfem.fatigue import (
    BeamSection,
    beam_nominal_stress,
    fatigue_life,
    global_to_detail_fatigue,
    hot_spot_stress,
    solve_weld_detail,
)
from tensorfem.fatigue.detail_maturity import (
    JOINT_FAMILY,
    _json_ready,
    family_geometry,
    gci_hotspot,
    power_law_response,
    run_maturity_assessment,
    scf_vs_iiw,
    toe_mesh_resolution,
)
from tensorfem.fatigue.weld_toe import (
    maddox_stress_concentration,
    nii_stress_concentration,
)


def _joint_family_scf_table(level: int = 1) -> list[dict[str, object]]:
    """SCF of every joint family member against its closed-form anchor."""
    rows = []
    for t, r in JOINT_FAMILY:
        for joint_type in ("t_fillet", "butt", "corner"):
            detail = family_geometry(t, r, joint_type=joint_type, load_case="plate_axial")
            sol = solve_weld_detail(detail, level=level)
            if joint_type == "t_fillet":
                anchor = nii_stress_concentration(t, r, detail.attachment_height_m)
                anchor_name = "nii"
            elif joint_type == "butt":
                anchor = maddox_stress_concentration(t, r)
                anchor_name = "maddox"
            else:
                anchor = None
                anchor_name = "none"
            rows.append(
                {
                    "joint_type": joint_type,
                    "t_m": t,
                    "r_m": r,
                    "t_over_r": t / r,
                    "kt_peak": sol.kt_toe,
                    "anchor": anchor_name,
                    "anchor_value": anchor,
                    "anchor_error": (
                        (sol.kt_toe - anchor) / anchor if anchor is not None else None
                    ),
                    "n_elements": sol.n_elements,
                }
            )
    return rows


def _hotspot_chain_table(level: int = 1) -> list[dict[str, object]]:
    """Hot-spot stress and S-N life of the worked hull-girder detail."""
    t = 0.020
    section = BeamSection(area_m2=t * 1.0, I_m4=1.0 * t**3 / 12.0, c_m=0.5 * t)
    sigma_nom = beam_nominal_stress(
        axial_force_n=60.0e6 * section.area_m2,
        bending_moment_nm=20.0e6 * section.Z_m3,
        section=section,
    )
    rows = []
    for joint_type in ("t_fillet", "butt", "corner"):
        detail = family_geometry(t, 0.002, joint_type=joint_type, load_case="plate_axial")
        sol = solve_weld_detail(detail, level=level)
        lin = hot_spot_stress(sol, t, method="linear")
        quad = hot_spot_stress(sol, t, method="quadratic")
        # the stress range is the absolute value of the hot-spot stress
        # (fatigue is driven by the stress *range*, not the mean)
        sigma_hot_lin = abs(lin.stress_ratio * sigma_nom)
        sigma_hot_quad = abs(quad.stress_ratio * sigma_nom)
        rows.append(
            {
                "joint_type": joint_type,
                "sigma_nom_pa": sigma_nom,
                "scf_toe": sol.kt_toe,
                "scf_hotspot_linear": lin.stress_ratio,
                "scf_hotspot_quadratic": quad.stress_ratio,
                "sigma_hot_linear_pa": sigma_hot_lin,
                "sigma_hot_quadratic_pa": sigma_hot_quad,
                "life_air_cycles": fatigue_life(sigma_hot_lin, environment="air", thickness_m=t),
                "life_cp_cycles": fatigue_life(sigma_hot_lin, environment="seawater_cp", thickness_m=t),
                "life_free_cycles": fatigue_life(sigma_hot_lin, environment="seawater_free", thickness_m=t),
            }
        )
    return rows


def _plot(out: Path, ledger: dict[str, object], hotspot_chain: list[dict[str, object]]) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs: list[Path] = []
    out.mkdir(parents=True, exist_ok=True)

    # 1) SCF family: FE notch vs closed-form anchors, all three joint types
    scf_rows = ledger["scf_family"]["rows"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for joint, marker in (("t_fillet", "o"), ("butt", "s"), ("corner", "^")):
        sel = [row for row in scf_rows if row["joint_type"] == joint]
        if joint == "corner":
            ax.plot([row["t_over_r"] for row in sel], [row["kt_peak"] for row in sel],
                    marker, label=f"{joint} (no closed form)", alpha=0.8)
        else:
            ax.plot([row["anchor_value"] for row in sel], [row["kt_peak"] for row in sel],
                    marker, label=f"{joint} FE vs closed form", alpha=0.8)
    lims = [1.0, 3.0]
    ax.plot(lims, lims, "k--", lw=0.8, label="1:1")
    ax.set_xlabel("closed-form SCF (Nii / Maddox)")
    ax.set_ylabel("FE notch SCF")
    ax.set_title("Notch SCF against closed-form anchors (plate axial load)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    path = out / "scf_vs_anchor.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figs.append(path)

    # 2) S-N curve D with the assessed detail range
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ranges = np.logspace(np.log10(20e6), np.log10(300e6), 100)
    for env in ("air", "seawater_cp", "seawater_free"):
        ax.loglog(ranges / 1e6, [fatigue_life(v, environment=env) for v in ranges], label=env)
    ax.set_xlabel("stress range $\\Delta\\sigma$ [MPa]")
    ax.set_ylabel("cycles to failure N")
    ax.set_title("DNV-RP-C203 curve D (m1 = 3, m2 = 5)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    path = out / "sn_curve.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figs.append(path)

    # 3) mesh convergence of the hot-spot stress
    conv = ledger["convergence"]["levels"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    h = [row["h_m"] * 1e6 for row in conv]
    hs = [row["hotspot_ratio"] for row in conv]
    ax.plot(h, hs, "o-")
    for hh, vv, n in zip(h, hs, [row["n_elements"] for row in conv]):
        ax.annotate(f"{n} el.", (hh, vv), textcoords="offset points", xytext=(4, 6), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("toe surface element size [um]")
    ax.set_ylabel("extrapolated hot-spot SCF")
    ax.set_title("Mesh convergence of the hot-spot stress (3 levels)")
    ax.grid(True, which="both", alpha=0.3)
    path = out / "mesh_convergence.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figs.append(path)

    # 4) hot-spot chain: sigma_hot = SCF * sigma_nom for all three joint types
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    joints = [row["joint_type"] for row in hotspot_chain]
    scf_lin = [abs(row["scf_hotspot_linear"]) for row in hotspot_chain]
    scf_quad = [abs(row["scf_hotspot_quadratic"]) for row in hotspot_chain]
    x = np.arange(len(joints))
    w = 0.35
    ax.bar(x - w / 2, scf_lin, w, label="linear (0.5t/1.5t)")
    ax.bar(x + w / 2, scf_quad, w, label="quadratic (0.4t/0.9t/1.4t)")
    ax.set_xticks(x)
    ax.set_xticklabels(joints)
    ax.set_ylabel("hot-spot SCF (abs)")
    ax.set_title("Hot-spot SCF of the three joint families (t = 20 mm, r = 2 mm)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    path = out / "hotspot_chain.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figs.append(path)
    return figs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/detail_fatigue_v3")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ledger = run_maturity_assessment()
    scf_table = _joint_family_scf_table()
    hotspot_chain = _hotspot_chain_table()

    payload = {
        "node": "P106W3A2-N02",
        "repository": "TensorFEM",
        "package": "tensorfem.fatigue.detail_maturity",
        "joint_family": [
            {"t_m": t, "r_m": r, "t_over_r": t / r} for t, r in JOINT_FAMILY
        ],
        "ledger": _json_ready(ledger),
        "scf_family_table": _json_ready(scf_table),
        "hotspot_chain": _json_ready(hotspot_chain),
    }
    ledger_path = out / "ledger.json"
    ledger_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    figures = [] if args.no_figures else _plot(out, _json_ready(ledger), hotspot_chain)

    print(f"ledger   -> {ledger_path}")
    for path in figures:
        print(f"figure   -> {path}")
    for name, row in ledger["criteria"].items():
        print(f"{name:24s} pass={str(row['pass']):5s} {row['requirement']}")
    print(f"all_pass = {ledger['all_pass']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
