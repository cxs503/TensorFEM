"""Benchmark driver of the welded-detail fatigue maturity package (P106W2-N10).

Runs :func:`tensorfem.fatigue.detail_maturity.run_maturity_assessment`, writes the
machine-readable ledger and the figures of the report, and prints the criterion
table.  The driver is deterministic (no random seeds) and needs no network.

Usage::

    python benchmarks/fatigue/detail_maturity_v2.py [--out artifacts/detail_fatigue_v2]
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
    detail_fatigue_assessment,
    fatigue_life,
    hot_spot_stress,
    solve_weld_detail,
)
from tensorfem.fatigue.detail_maturity import (
    JOINT_FAMILY,
    _json_ready,
    family_geometry,
    run_maturity_assessment,
)


def _anchor_gap_table(level: int = 1) -> list[dict[str, object]]:
    """FE/anchor ratio of the T-joint family under the two load cases.

    ``carried`` is the historical default (plate axial traction *and* attachment
    traction); ``plate_axial`` is the load-free attachment case.  The table is the
    before/after evidence of the load-case defect fixed in this work package.
    """
    rows = []
    for t, r in JOINT_FAMILY:
        for load_case in ("carried", "plate_axial"):
            detail = family_geometry(t, r, load_case=load_case)
            sol = solve_weld_detail(detail, level=level)
            from tensorfem.fatigue.weld_toe import nii_stress_concentration

            anchor = nii_stress_concentration(t, r, detail.attachment_height_m)
            rows.append(
                {
                    "t_m": t,
                    "r_m": r,
                    "t_over_r": t / r,
                    "load_case": load_case,
                    "kt_peak": sol.kt_toe,
                    "nii": anchor,
                    "ratio": sol.kt_toe / anchor,
                }
            )
    return rows


def _spectrum_life(scf: float, sigma_nom_pa: float, t: float) -> dict[str, float]:
    """Hot-spot stress range -> DNV curve D life in three environments."""
    sigma_hot = scf * sigma_nom_pa
    return {
        env: fatigue_life(sigma_hot, environment=env, thickness_m=t)
        for env in ("air", "seawater_cp", "seawater_free")
    }


def _plot(out: Path, ledger: dict[str, object]) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs: list[Path] = []
    out.mkdir(parents=True, exist_ok=True)

    # 1) radius sweep: notch SCF and structural ratio
    controls = ledger["radius_controls"]
    rows = controls["rows"]
    x = [row["radius_over_t"] for row in rows]
    kt = [row["kt_peak"] for row in rows]
    st = [row["structural_ratio"] for row in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.loglog(x, kt, "o-", label="FE notch SCF (peak surface)")
    ax.loglog(x, st, "s-", label="FE structural SCF (IIW 0.4t/0.9t)")
    rr = np.logspace(math.log10(min(x)), math.log10(max(x)), 60)
    amp = controls["decay_amplitude"]
    expo = controls["decay_exponent"]
    if math.isfinite(amp) and math.isfinite(expo):
        ax.loglog(rr, 1.0 + amp * rr ** (-expo), "--", lw=1,
                  label=fr"fit $1+{amp:.3f}(r/t)^{{-{expo:.2f}}}$")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.axhspan(1.2, 1.5, color="tab:orange", alpha=0.12, label="IIW structural band 1.2-1.5")
    ax.set_xlabel("r / t")
    ax.set_ylabel("stress concentration factor")
    ax.set_title("Weld-toe radius sweep (t = 20 mm, plate axial load)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    path = out / "radius_sweep.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figs.append(path)

    # 2) hot-spot surface path of the load-carrying joint
    carrying = family_geometry(0.020, 0.002, load_case="attachment_supported")
    sol = solve_weld_detail(carrying, level=1)
    xs = (np.asarray(sol.surface_x_m) - sol.x_toe_m) / carrying.plate_thickness_m
    sig = np.asarray(sol.surface_sigma_xx_pa) / sol.nominal_stress_pa
    order = np.argsort(xs)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(xs[order], sig[order], "-", label="FE surface $\\sigma_{xx}/\\sigma_{nom}$")
    for f in (0.5, 1.0, 1.5):
        ax.axvline(f, color="grey", ls=":", lw=0.8)
    lin = hot_spot_stress(sol, carrying.plate_thickness_m, method="linear")
    ax.axhline(lin.stress_ratio, color="tab:red", ls="--", lw=1,
               label=f"extrapolated hot spot = {lin.stress_ratio:.2f}")
    ax.set_xlim(-1.0, 4.0)
    ax.set_xlabel("(x - x_toe) / t")
    ax.set_ylabel("surface stress / nominal")
    ax.set_title("Load-carrying T-joint, surface path and IIW stations")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    path = out / "surface_path.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figs.append(path)

    # 3) SCF family: FE notch vs closed form, and structural ratio vs band
    scf_rows = ledger["scf_family"]["rows"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for joint, marker in (("t_fillet", "o"), ("butt", "s")):
        sel = [row for row in scf_rows if row["joint_type"] == joint]
        ax.plot([row["anchor_value"] for row in sel], [row["kt_peak"] for row in sel],
                marker, label=f"{joint} FE vs closed form", alpha=0.8)
    lims = [1.2, 2.5]
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

    # 4) S-N curve D with the assessed detail range
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ranges = np.logspace(np.log10(20e6), np.log10(300e6), 100)
    for env in ("air", "seawater_cp", "seawater_free"):
        ax.loglog(ranges / 1e6, [fatigue_life(v, environment=env) for v in ranges], label=env)
    ax.axvline(lin.stress_ratio * 1.0e6, color="tab:red", ls="--", lw=1,
               label="assessed $\\sigma_{hot}$ (1 MPa nominal)")
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

    # 5) mesh convergence of the hot-spot stress
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
    return figs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/detail_fatigue_v2")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ledger = run_maturity_assessment()
    anchors = _anchor_gap_table()

    # a worked detail: T-joint of the acceptance family in a hull-girder stress field
    detail = family_geometry(0.020, 0.002, load_case="plate_axial")
    section = BeamSection(area_m2=0.020 * 1.0, I_m4=0.020**3 / 12.0, c_m=0.010)
    # hull-girder detail: 60 MPa axial + 20 MPa bending at the detail section
    sigma_nom = beam_nominal_stress(
        axial_force_n=1.2e6 * 1.0, bending_moment_nm=20.0e6 * section.Z_m3 * 1.0,
        section=section,
    )
    assessment = detail_fatigue_assessment(
        detail, nominal_stress_range_pa=sigma_nom, environment="air", level=1
    )
    lives = _spectrum_life(
        assessment["kt_hotspot_linear"], sigma_nom, 0.020
    )

    payload = {
        "node": "P106W2-N10",
        "repository": "TensorFEM",
        "package": "tensorfem.fatigue.detail_maturity",
        "joint_family": [
            {"t_m": t, "r_m": r, "t_over_r": t / r} for t, r in JOINT_FAMILY
        ],
        "ledger": _json_ready(ledger),
        "anchor_gap_before_after": _json_ready(anchors),
        "worked_example": {
            "sigma_nom_pa": sigma_nom,
            "assessment": _json_ready(assessment),
            "lives_by_environment": lives,
        },
    }
    ledger_path = out / "ledger.json"
    ledger_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    figures = [] if args.no_figures else _plot(out, _json_ready(ledger))

    print(f"ledger   -> {ledger_path}")
    for path in figures:
        print(f"figure   -> {path}")
    for name, row in ledger["criteria"].items():
        print(f"{name:24s} pass={str(row['pass']):5s} {row['requirement']}")
    print(f"all_pass = {ledger['all_pass']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())