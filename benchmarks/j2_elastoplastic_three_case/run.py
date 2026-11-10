"""Three-case J2 elastoplastic qualification run (campaign node p337w1-n07).

Produces, in one command:

* ``artifacts/j2_elastoplastic_three_case/report.json`` -- every criterion
  number plus the full tables (nothing hand-copied: the JSON *is* the evidence),
* the figures referenced by ``docs/j2-elastoplastic-three-case.md``.

Cases (SI units, material card E = 210 GPa, nu = 0.3, sigma_y = 250 MPa,
H = 1000 MPa):

  A  uniaxial tension   L = 100 mm, A = 10 x 10 mm, eps = 0..5 %
  B  pure shear         thin-wall tube L = 100 mm, R = 50 mm, t = 2 mm
  C  thick-walled tube  R_i = 50 mm, R_o = 100 mm, L = 200 mm, p internal

Usage::

    python benchmarks/j2_elastoplastic_three_case/run.py            # full run (~5 min)
    python benchmarks/j2_elastoplastic_three_case/run.py --quick    # smoke (~1 min)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorfem.j2_elastoplastic_three_case import (  # noqa: E402
    CylinderSpec,
    TorsionSpec,
    UniaxialSpec,
    plastic_pressure_levels,
    run_qualification,
)

ARTIFACTS = ROOT / "artifacts" / "j2_elastoplastic_three_case"


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if value != 0.0 and abs(value) < 1e-3:
            return f"{value:.3e}"
        return f"{value:.{digits}f}"
    return str(value)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="coarse meshes / fewer levels (smoke test)")
    parser.add_argument("--out", default=None, help="output directory override")
    parser.add_argument("--figures-only", action="store_true",
                        help="re-render the figures from an existing report.json")
    args = parser.parse_args(argv)
    torch.set_num_threads(6)

    out = Path(args.out) if args.out else ARTIFACTS
    out.mkdir(parents=True, exist_ok=True)

    if args.figures_only:
        payload = json.loads((out / "report.json").read_text())
        payload["figures"] = make_figures(payload, out)
        (out / "report.json").write_text(json.dumps(payload, indent=2))
        print(f"figures refreshed in {out}: {[f['name'] for f in payload['figures']]}")
        return 0

    uniaxial = UniaxialSpec()
    torsion = TorsionSpec()
    cylinder = CylinderSpec()
    if args.quick:
        strains = [0.002, 0.01, 0.03, 0.05]
        shears = [0.0009, 0.0021, 0.006, 0.025]
        meshes = (8, 16, 32)
        bar_mesh = (6, 1, 1)
        steps = 6
    else:
        strains = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
        shears = [0.0004, 0.0009, 0.0015, 0.0021, 0.003, 0.006, 0.012, 0.025, 0.05]
        meshes = (16, 32, 64)
        bar_mesh = (8, 1, 1)
        steps = 8

    start = time.time()
    report = run_qualification(
        uniaxial=uniaxial, torsion=torsion, cylinder=cylinder,
        uniaxial_strains=strains, shear_levels=shears,
        cylinder_meshes=meshes, bar_mesh=bar_mesh, steps=steps)
    elapsed = time.time() - start

    payload = {
        "node": "p337w1-n07-fem-elastoplastic",
        "benchmark": "j2_elastoplastic_three_case",
        "generated_by": "benchmarks/j2_elastoplastic_three_case/run.py",
        "quick": bool(args.quick),
        "wall_seconds": elapsed,
        "torch_version": torch.__version__,
        "geometry": {
            "uniaxial": {"length": uniaxial.length, "width": uniaxial.width},
            "torsion": {"length": torsion.length, "radius": torsion.radius,
                        "thickness": torsion.thickness},
            "cylinder": {"inner_radius": cylinder.inner_radius,
                         "outer_radius": cylinder.outer_radius,
                         "length": cylinder.length,
                         "pressure_levels": plastic_pressure_levels(cylinder)},
        },
        "report": report,
    }
    (out / "report.json").write_text(json.dumps(payload, indent=2))

    figures = make_figures(payload, out)
    payload["figures"] = figures
    (out / "report.json").write_text(json.dumps(payload, indent=2))

    print(f"wall time {elapsed:.1f} s; artifacts in {out}")
    print()
    print(f"{'criterion':28s} {'value':>12s} {'band':>8s}  verdict")
    for name, entry in report["criteria"].items():
        value = entry["value"]
        band = entry["tolerance"]
        verdict = {True: "PASS", False: "FAIL", None: "n/a"}[entry["passed"]]
        if isinstance(band, float) and band <= 1.0:
            shown_band = f"{band:.0%}"
        else:
            shown_band = _fmt(band)
        print(f"{name:28s} {_fmt(value):>12s} {shown_band:>8s}  {verdict}")
    print()
    print(f"{'case':34s} {'worst number'}")
    tube = report["case_C_cylinder"]["plastic"]
    print(f"{'A bar sigma(eps) max rel err':34s} "
          f"{_fmt(report['case_A_uniaxial']['tet4_bar']['max_relative_error_element'])}")
    print(f"{'A yield stress read back [MPa]':34s} "
          f"{_fmt(report['case_A_uniaxial']['tet4_bar']['yield_stress_estimate'] / 1e6, 3)}")
    print(f"{'B tau_y [MPa]':34s} {_fmt(report['case_B_shear']['material_point']['tau_y'] / 1e6, 4)}")
    print(f"{'C r_p max rel err':34s} {_fmt(tube['max_r_p_relative_error'])}")
    print(f"{'C u_r(r) max rel err':34s} {_fmt(tube['max_u_profile_relative_error'])}")
    worst_gci = max((r["gci"]["relative_gci"] for r in tube["convergence"].values()
                     if r.get("gci")), default=float("nan"))
    print(f"{'C GCI (worst, u_r(a))':34s} {_fmt(worst_gci)}")
    print(f"{'C pi-plane max deviation':34s} "
          f"{_fmt(report['pi_plane']['max_relative_deviation'])}")
    failed = [name for name, entry in report["criteria"].items() if entry["passed"] is False]
    print()
    print("FAILED:", failed if failed else "none")
    return 0 if not failed else 1


def make_figures(payload: dict, out: Path) -> list[dict]:
    """Render the four evidence figures; returns the figure manifest."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - matplotlib is a dev dependency
        print(f"figures skipped: {exc}")
        return []

    report = payload["report"]
    card = report["material_card"]
    figures = []

    # -- figure 1: uniaxial sigma-eps -----------------------------------------
    point = report["case_A_uniaxial"]["material_point"]["levels"]
    bar = report["case_A_uniaxial"]["tet4_bar"]["levels"]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot([r["strain"] * 100 for r in bar], [r["stress_reference"] / 1e6 for r in bar],
            "-", lw=1.6, label="material card (bilinear closed form)")
    ax.plot([r["strain"] * 100 for r in point], [r["stress_kernel"] / 1e6 for r in point],
            "o", ms=4, label="material point (radial return)")
    ax.plot([r["strain"] * 100 for r in bar], [r["stress_element"] / 1e6 for r in bar],
            "s", ms=4, mfc="none", label="TET4 bar (FE, load-cell)")
    yield_strain = card["yield_strain"] * 100
    ax.axvline(yield_strain, color="0.6", ls="--", lw=1)
    ax.annotate(f"$\\varepsilon_y$ = {yield_strain:.3f} %\n$\\sigma_y$ = {card['yield_stress']/1e6:.0f} MPa",
                xy=(yield_strain, card["yield_stress"] / 1e6), xytext=(1.4, 268),
                fontsize=8, arrowprops={"arrowstyle": "->", "color": "0.4"})
    ax.set_xlabel("axial strain $\\varepsilon$ [%]")
    ax.set_ylabel("axial stress $\\sigma$ [MPa]")
    ax.set_title("case A - uniaxial tension: FE vs material card")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    figures.append(_save(fig, out, "fig1_uniaxial_stress_strain.png",
                         "Uniaxial tension: bilinear material card, material-point "
                         "radial return and the TET4 bar coincide to machine precision."))

    # -- figure 2: pure shear ------------------------------------------------
    shear = report["case_B_shear"]["material_point"]["levels"]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot([r["gamma"] * 100 for r in shear], [r["tau_reference"] / 1e6 for r in shear],
            "-", lw=1.6, label="closed form $\\tau(\\gamma)$")
    ax.plot([r["gamma"] * 100 for r in shear], [r["tau_kernel"] / 1e6 for r in shear],
            "o", ms=4, label="radial return (pure shear strain state)")
    tau_y = report["case_B_shear"]["material_point"]["tau_y"] / 1e6
    ax.axhline(tau_y, color="0.6", ls="--", lw=1)
    ax.annotate(f"$\\tau_y=\\sigma_y/\\sqrt{{3}}$ = {tau_y:.2f} MPa", xy=(0.2, tau_y),
                xytext=(1.2, tau_y + 6), fontsize=8,
                arrowprops={"arrowstyle": "->", "color": "0.4"})
    ax.set_xlabel("engineering shear strain $\\gamma$ [%]")
    ax.set_ylabel("shear stress $\\tau$ [MPa]")
    ax.set_title("case B - pure shear (thin-wall tube idealisation)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    figures.append(_save(fig, out, "fig2_pure_shear.png",
                         "Pure shear: the implemented return map reproduces the closed form "
                         "and the Mises shear yield stress sigma_y/sqrt(3)."))

    # -- figure 3: plastic zone + radial displacement ------------------------
    tube = report["case_C_cylinder"]["plastic"]
    rows = [row for row in tube["levels"] if row["r_p_reference"] is not None]
    finest = max(row["n_r"] for row in rows)
    top = [row for row in rows if row["n_r"] == finest]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    ax = axes[0]
    ax.plot([row["pressure"] / 1e6 for row in top],
            [row["r_p_reference"] * 1e3 for row in top], "-", lw=1.6,
            label="Hill 1950 (closed form)")
    ax.plot([row["pressure"] / 1e6 for row in top],
            [row["r_p_fem"] * 1e3 for row in top], "o", ms=4, label=f"FE, $n_r$={finest}")
    ax.set_xlabel("internal pressure $p$ [MPa]")
    ax.set_ylabel("plastic radius $r_p$ [mm]")
    ax.set_title("case C - plastic-zone growth")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax = axes[1]
    target = top[len(top) // 2]
    profile = target["u_profile"]
    ax.plot([p["r"] * 1e3 for p in profile], [p["u_reference"] * 1e6 for p in profile],
            "-", lw=1.6, label="Hill 1950 (quadrature)")
    ax.plot([p["r"] * 1e3 for p in profile], [p["u_computed"] * 1e6 for p in profile],
            "s", ms=4, mfc="none", label=f"FE, $n_r$={finest}")
    ax.set_xlabel("radius $r$ [mm]")
    ax.set_ylabel("radial displacement $u_r$ [$\\mu$m]")
    ax.set_title(f"case C - $u_r(r)$ at $p/k$ = {target['pressure_over_k']:.2f}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    figures.append(_save(fig, out, "fig3_cylinder_plastic_zone.png",
                         "Thick-walled cylinder: plastic radius versus pressure and the radial "
                         "displacement field against Hill's plane-strain solution."))

    # -- figure 4: pi-plane circularity --------------------------------------
    directions = report["pi_plane"]["directions"]
    fig = plt.figure(figsize=(4.6, 4.2))
    ax = fig.add_subplot(111, projection="polar")
    theta = [math.radians(row["theta_deg"]) for row in directions]
    radius = [row["equivalent_stress_at_onset"] / card["yield_stress"] for row in directions]
    ax.plot(theta + theta[:1], radius + radius[:1], "o-", ms=4, lw=1.2)
    ax.set_title("case C6 - yield surface in the $\\pi$ plane\n"
                 f"max deviation {report['pi_plane']['max_relative_deviation']:.1e}",
                 fontsize=10)
    ax.set_yticklabels([])
    figures.append(_save(fig, out, "fig4_pi_plane.png",
                         "von Mises yield surface probed along six deviatoric directions: "
                         "q/sigma_y = 1 on a circle to machine precision."))

    # -- figure 5: GCI convergence -------------------------------------------
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for key, block in tube["convergence"].items():
        if not block.get("gci"):
            continue
        levels = block["levels"]
        ax.loglog([level["mesh_size"] * 1e3 for level in levels],
                  [abs(level["value"]) * 1e6 for level in levels], "o-", ms=4, lw=1.0,
                  label=f"$p/k$ = {float(key)/ (250e6/math.sqrt(3)):.2f}, "
                        f"GCI {block['gci']['relative_gci']:.1e}")
    ax.set_xlabel("radial element size $h$ [mm]")
    ax.set_ylabel("$u_r(a)$ [$\\mu$m]")
    ax.set_title("case C - h-refinement of the bore displacement")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    figures.append(_save(fig, out, "fig5_gci.png",
                         "Grid-convergence study: bore displacement versus element size for the "
                         "five pressure levels, with the Roache GCI of the finest pair."))
    return figures


def _save(fig, out: Path, name: str, caption: str) -> dict:
    import matplotlib.pyplot as plt
    path = out / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    try:
        shown = str(path.relative_to(ROOT))
    except ValueError:  # --out outside the repository (ad-hoc runs)
        shown = str(path)
    return {"name": name, "caption": caption, "path": shown}


if __name__ == "__main__":
    raise SystemExit(main())