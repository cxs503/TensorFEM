"""Independent audit of the ice-grillage ultimate strength anchor.

Produces the audit artefacts of
``docs/AUDIT-GAP-PP-5-GRILLAGE-ULTIMATE-independent-review.md``:

``artifacts/grillage_ultimate_audit/audit_evidence.json``
    the T11/T12/T13 evidence bundle plus the governing-branch census,
``artifacts/grillage_ultimate_audit/audit_tables.md``
    the same numbers rendered as markdown tables (pasted into the doc by script,
    never typed by hand),
``docs/assets/grillage_ultimate_audit/*.png``
    figure 1: peak plate moment, Navier series vs independent FD (slice + grid
    convergence); figure 2: ultimate pressure, module vs FD recomputation and the
    rigid-plastic cross-method value; figure 3: governing-branch census.

Run: ``PYTHONPATH=src python benchmarks/grillage_ultimate_independent_audit.py``
"""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfem.grillage_ice_ultimate import (
    CombinedLoadCase,
    case_study_geometry,
    grillage_utilisations,
    plate_patch_response,
    ultimate_ice_pressure,
)
from tensorfem.grillage_ice_verification import (
    CASE_STUDY_SCENARIOS,
    MEMBER_GOVERNED_SCENARIOS,
    case_study_geometry_slender,
    member_governed_case_table,
    member_governed_geometry,
    plate_field_fd_check,
    run_grillage_verification,
)
from tensorfem.orthotropic_plate_fd import ssss_orthotropic_plate_fd

ROOT = pathlib.Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts" / "grillage_ultimate_audit"
FIG = ROOT / "docs" / "assets" / "grillage_ultimate_audit"


def governing_census(geoms: dict[str, object], n_eta: int = 3, n_patch: int = 3) -> list[dict]:
    """Which branch binds across a coarse family of combined cases."""
    rows = []
    for tag, geom in geoms.items():
        for eta in np.linspace(0.0, 0.6, n_eta):
            for frac in (1.0, 3.0):
                lx = frac * geom.bay_spacing
                ly = geom.bay_spacing
                case = CombinedLoadCase(
                    axial_stress=float(eta) * geom.yield_stress, ice_pressure=1.0,
                    patch_length=lx, patch_width=ly, hydrostatic_pressure=1.0e5,
                    design_pressure=1.0e6)
                res = ultimate_ice_pressure(geom, case, terms=31)
                u = res["utilisations_at_limit"]
                rows.append({
                    "geometry": tag, "sigma_x_over_yield": float(eta),
                    "patch_length_over_bay": frac,
                    "p_ult_MPa": res["ice_pressure_ultimate"] / 1e6,
                    "governing_member": res["governing_member"],
                    "U_plate": u["plate_local_bending"],
                    "U_stiffener": u["longitudinal_stiffener"],
                    "U_frame": u["transverse_frame"],
                })
    return rows


def figure_field(geom, path: pathlib.Path) -> dict:
    """Navier vs FD: peak moment, y-slice through the patch and convergence."""
    a, b, d = geom.frame_spacing, geom.bay_spacing, geom.plate_rigidity
    lx, ly = CASE_STUDY_SCENARIOS[2][3], CASE_STUDY_SCENARIOS[2][4]
    patch = (0.5 - 0.5 * lx / a, 0.5 + 0.5 * lx / a,
             0.5 - 0.5 * ly / b, 0.5 + 0.5 * ly / b)
    fd = ssss_orthotropic_plate_fd(span_x=a, span_y=b, d1=d, d2=d, d3=d,
                                   pressure=1.0, patch=patch, nx=240,
                                   poisson=geom.poisson, return_field=True)
    series = plate_patch_response(span_x=a, span_y=b, d1=d, d2=d, d3=d,
                                  pressure=1.0, patch=patch, terms=121,
                                  sample=81, poisson=geom.poisson)
    xs, ys, w = fd["x"], fd["y"], fd["w"]
    j = int(np.argmin(np.abs(ys - 0.5 * b)))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    ax.plot(xs, w[:, j] * 1e3, "k-", lw=2.0, label="FD (13-point stencil)")
    ax.set_xlabel("x along the panel [m]")
    ax.set_ylabel(r"deflection [mm] per Pa")
    ax.set_title("panel deflection, y = b/2 (severe-case patch)")
    errs, grids = [], []
    ref = max(series["m_x_max"], series["m_y_max"])
    for nx in (60, 120, 240):
        f = ssss_orthotropic_plate_fd(span_x=a, span_y=b, d1=d, d2=d, d3=d,
                                      pressure=1.0, patch=patch, nx=nx,
                                      ny=max(8, int(round(nx * b / a))),
                                      poisson=geom.poisson)
        errs.append(abs(max(f["m_x_max"], f["m_y_max"]) / ref - 1.0))
        grids.append(nx)
    axes[1].loglog(grids, errs, "o-", label="FD vs Navier series")
    axes[1].loglog(grids, errs[0] * (np.array(grids) / grids[0]) ** -2.0, "k--",
                   label=r"second order, $\propto n_x^{-2}$")
    axes[1].set_xlabel("FD grid n_x")
    axes[1].set_ylabel("relative peak-moment error")
    axes[1].set_title("independent FD reference converges onto the series")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return {"peak_moment_series": ref,
            "peak_moment_fd_240": max(fd["m_x_max"], fd["m_y_max"]),
            "grids": grids, "errors": errs}


def figure_cases(fd_rows, member_rows, cross, path: pathlib.Path) -> None:
    labels = [r["scenario"] for r in fd_rows] + [r["scenario"] + "\n(stiffener)" for r in member_rows]
    module = [r["p_ult_module_MPa"] for r in fd_rows] + [r["ice_pressure_ultimate_MPa"] for r in member_rows]
    fdvals = [r["p_ult_fd_MPa"] for r in fd_rows] + [r["p_ult_hand_closed_form_MPa"] for r in member_rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11.0, 4.4))
    ax.bar(x - 0.2, module, 0.4, label="module (bisection)")
    ax.bar(x + 0.2, fdvals, 0.4, label="independent method (FD field / closed form)")
    for i, (m, f) in enumerate(zip(module, fdvals)):
        ax.annotate(f"{100 * (f / m - 1.0):+.3f}%", (i, max(m, f)), ha="center",
                    va="bottom", fontsize=8)
    ax.axhline(cross["rigid_plastic_grillage_MPa"], ls=":", color="C3",
               label=f"rigid-plastic grillage limit, member case "
                     f"({cross['plastic_over_elastic']:.2f}x elastic)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("ultimate ice patch pressure [MPa]")
    ax.set_title("ultimate pressure: module vs independent recomputation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def figure_census(census, path: pathlib.Path) -> dict:
    geoms = sorted({r["geometry"] for r in census})
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    for k, tag in enumerate(geoms):
        rows = [r for r in census if r["geometry"] == tag]
        ax.plot([r["sigma_x_over_yield"] for r in rows],
                [r["U_plate"] for r in rows], "o-", color=f"C{k}",
                label=f"{tag}: plate field")
        ax.plot([r["sigma_x_over_yield"] for r in rows],
                [r["U_stiffener"] for r in rows], "s--", color=f"C{k}",
                label=f"{tag}: stiffener")
    ax.set_xlabel(r"coexisting compression $\sigma_x/\sigma_y$")
    ax.set_ylabel("utilisation at the ultimate (1.0 = at limit)")
    ax.set_title("governing-branch census (patch 1-3 bays long, 0.1 MPa head)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    from collections import Counter
    return {tag: dict(Counter(r["governing_member"] for r in census
                              if r["geometry"] == tag)) for tag in geoms}


def main() -> dict:
    ART.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    geom = case_study_geometry()

    verification = run_grillage_verification(fd_nx=160)
    fd_rows = plate_field_fd_check(geom, nx=240)
    member_rows = member_governed_case_table()
    census = governing_census({"case study (t=20 mm)": geom,
                               "slender (t=13 mm)": case_study_geometry_slender(),
                               "member-governed (t=25 mm)": member_governed_geometry()})
    field = figure_field(geom, FIG / "fig1_plate_field_fd_vs_navier.png")
    figure_cases(fd_rows, member_rows, verification["cross_method"],
                 FIG / "fig2_ultimate_pressure_independent.png")
    census_counts = figure_census(census, FIG / "fig3_governing_branch_census.png")

    evidence = {
        "anchor": "GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE",
        "verification": {k: verification[k] for k in ("passed", "n_checks", "n_failed")},
        "gates": verification["checks"],
        "plate_field_fd": fd_rows,
        "member_governed_cases": member_rows,
        "cross_method": verification["cross_method"],
        "governing_census": census,
        "governing_census_counts": census_counts,
        "field_convergence": field,
    }
    (ART / "audit_evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")

    lines = ["## Plate field: Navier series vs independent FD (T11)", "",
             "| case | patch [m] | peak moment series [N m/m per Pa] | FD | error | p_ult module [MPa] | p_ult FD [MPa] | error | governing |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in fd_rows:
        lines.append(
            f"| {r['scenario']} | {r['patch_length_m']:.2f} x {r['patch_width_m']:.2f} | "
            f"{r['peak_moment_series_per_Pa']:.6g} | {r['peak_moment_fd_per_Pa']:.6g} | "
            f"{100 * r['peak_moment_rel_error']:+.4f} % | {r['p_ult_module_MPa']:.4f} | "
            f"{r['p_ult_fd_MPa']:.4f} | {100 * r['p_ult_rel_error']:+.4f} % | "
            f"{r['governing_member']} |")
    lines += ["", f"FD grid convergence (severe-case patch), peak moment: "
              + ", ".join(f"n_x={g}: {100 * e:.4f} %" for g, e in zip(field["grids"], field["errors"])),
              "", "## Member-governed combined cases (T12, T13)", "",
              "| case | sigma_x [MPa] | patch [m] | p_ult [MPa] | U_plate | U_stiff | U_frame | governing | margin | closed form [MPa] | error |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in member_rows:
        lines.append(
            f"| {r['scenario']} | {r['sigma_x_MPa']:.2f} | {r['patch_length_m']:.2f} x "
            f"{r['patch_width_m']:.2f} | {r['ice_pressure_ultimate_MPa']:.4f} | "
            f"{r['utilisation_plate_at_limit']:.3f} | {r['utilisation_stiffener_at_limit']:.4f} | "
            f"{r['utilisation_frame_at_limit']:.3f} | {r['governing_member']} | "
            f"{r['safety_margin']:.3f} | {r['p_ult_hand_closed_form_MPa']:.4f} | "
            f"{100 * r['rel_error_hand_vs_module']:+.2e} % |")
    cross = verification["cross_method"]
    lines += ["", "Cross-method (T13): elastic member limit "
              f"{cross['elastic_member_limit_MPa']:.4f} MPa vs rigid-plastic grillage "
              f"{cross['rigid_plastic_grillage_MPa']:.4f} MPa "
              f"(ratio {cross['plastic_over_elastic']:.3f}, plastic system "
              f"{cross['plastic_governing_system']}).", "",
              "Governing-branch census: " + ", ".join(
                  f"{k}: {v}" for k, v in census_counts.items()), ""]
    (ART / "audit_tables.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(json.dumps(evidence["verification"]))
    return evidence


if __name__ == "__main__":
    main()
