"""Ice three-point bend fracture benchmark driven by cohesive interface elements.

Case definition (ice plate bending, span/depth = 4 as in ice beam tests)

    E  = 5.0 GPa          nu = 0.30
    sigma_f = 1.5 MPa     G_f = 1.0 J/m^2
    span L = 0.40 m       depth d = 0.10 m      thickness b = 0.10 m
    l_ch = E G_f / sigma_f^2 = 2.22 mm  ->  l_ch/d = 0.022

Machine-checkable anchors
-------------------------
1. peak load vs the flexural-strength limit ``P = 2 b d^2 sigma_f / (3 L)``;
2. dissipated fracture energy vs ``G_f * b * a_crack`` (a_crack = fully failed
   interface height);
3. global energy balance ``W_ext - U_elastic - W_diss``;
4. mesh convergence of the peak load;
5. brittleness sweep ``l_ch / d`` showing the cohesive peak load approaching the
   strength limit as the material becomes more brittle.

Usage
-----
    python benchmarks/ice_cohesive_bending_fracture.py run --nx 32 --ny 60 --tag m32x60
    python benchmarks/ice_cohesive_bending_fracture.py run --gf 4.0 --nx 32 --ny 60 --tag gf4
    python benchmarks/ice_cohesive_bending_fracture.py report

Each ``run`` writes ``artifacts/ice_cohesive_bend/<tag>.json`` and ``<tag>.npz``;
``report`` merges every recorded case into ``artifacts/ice_cohesive_bend/report.json``
plus CSV tables and the figures used by the documentation.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tensorfem.cohesive_bending import (  # noqa: E402
    BilinearCohesiveLaw,
    build_three_point_bend,
    flexural_strength_load,
    solve_cmod_control,
    timoshenko_midspan_deflection,
)

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "artifacts", "ice_cohesive_bend")

ICE = dict(
    young=5.0e9,
    poisson=0.30,
    strength=1.5e6,
    fracture_energy=1.0,
    span=0.16,
    depth=0.04,
    thickness=0.04,
)


def run_case(
    *,
    tag: str,
    nx: int,
    ny: int,
    young: float = ICE["young"],
    poisson: float = ICE["poisson"],
    strength: float = ICE["strength"],
    fracture_energy: float = ICE["fracture_energy"],
    span: float = ICE["span"],
    depth: float = ICE["depth"],
    thickness: float = ICE["thickness"],
    initial_crack: float = 0.0,
    max_steps: int = 400,
) -> dict:
    """Solve one case and persist its history and the anchor metrics."""
    law = BilinearCohesiveLaw.from_material(young, strength, fracture_energy)
    model = build_three_point_bend(
        span=span, depth=depth, thickness=thickness, young_modulus=young,
        poisson_ratio=poisson, law=law, nx=nx, ny=ny,
    )
    t0 = time.time()
    result = solve_cmod_control(
        model,
        initial_step=0.4 * law.damage_opening,
        cmod_max=60.0 * law.failure_opening,
        max_steps=max_steps,
        growth=1.3,
        load_tolerance=0.05,
    )
    elapsed = time.time() - t0

    analytic = flexural_strength_load(
        span=span, depth=depth, thickness=thickness, tensile_strength=strength
    )
    crack = float(result.crack_length[-1])
    gf_area = fracture_energy * thickness * crack
    diss = float(result.dissipated_envelope[-1])
    residual = float(result.external_work[-1] - result.strain_energy[-1] - result.dissipated_path[-1])
    data = dict(
        tag=tag,
        law=dict(
            young=young, poisson=poisson, strength=strength, fracture_energy=fracture_energy,
            cohesive_length=young * fracture_energy / strength**2,
            normal_stiffness=law.normal_stiffness,
            damage_opening=law.damage_opening,
            failure_opening=law.failure_opening,
        ),
        geometry=dict(span=span, depth=depth, thickness=thickness, span_over_depth=span / depth),
        mesh=dict(nx=nx, ny=ny, n_dofs=model.n_dofs, n_segments=model.n_segments,
                  segment_height=float(model.segment_heights[0]),
                  segment_over_lch=float(model.segment_heights[0] / (young * fracture_energy / strength**2))),
        solution=dict(steps=int(len(result.load)), elapsed_s=elapsed, stop_reason=result.stop_reason,
                      converged=bool(result.converged)),
        peak=dict(load_kn=result.peak_load / 1e3, analytic_load_kn=analytic / 1e3,
                  relative_error_pct=100.0 * (result.peak_load - analytic) / analytic,
                  deflection_mm=1e3 * result.peak_deflection,
                  timoshenko_deflection_mm=1e3 * timoshenko_midspan_deflection(
                      span=span, depth=depth, thickness=thickness, young_modulus=young,
                      poisson_ratio=poisson, load=result.peak_load)),
        fracture=dict(crack_height_mm=1e3 * crack,
                      process_zone_height_mm=1e3 * float(result.process_zone_length[-1]),
                      crack_over_depth=crack / depth,
                      dissipated_j=diss,
                      dissipated_path_j=float(result.dissipated_path[-1]),
                      gf_area_j=gf_area,
                      energy_relative_error_pct=100.0 * (diss - gf_area) / gf_area if gf_area > 0 else None,
                      external_work_j=float(result.external_work[-1]),
                      strain_energy_j=float(result.strain_energy[-1]),
                      balance_residual_j=residual,
                      balance_residual_pct=100.0 * residual / diss if diss > 0 else None),
        damage_profile=[float(v) for v in result.damage_profile],
        segment_y_mm=[1e3 * float(v) for v in result.segment_centres[:, 1]],
    )
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    base = os.path.join(ARTIFACT_DIR, tag)
    with open(base + ".json", "w") as handle:
        json.dump(data, handle, indent=1)
    np.savez(
        base + ".npz",
        load=result.load, deflection=result.deflection, cmod=result.cmod,
        crack=result.crack_length, zone=result.process_zone_length,
        diss_env=result.dissipated_envelope, diss_path=result.dissipated_path,
        wext=result.external_work, uel=result.strain_energy,
        damage=result.damage_profile, seg_y=result.segment_centres[:, 1],
    )
    print(json.dumps({k: data[k] for k in ("tag", "peak", "fracture", "mesh")}, indent=1))
    return data


def _load_cases():
    cases = []
    for path in sorted(glob.glob(os.path.join(ARTIFACT_DIR, "*.json"))):
        if os.path.basename(path) == "report.json":
            continue
        with open(path) as handle:
            cases.append(json.load(handle))
    return cases


def build_report() -> dict:
    """Merge every recorded case into the benchmark report and draw figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = _load_cases()
    if not cases:
        raise SystemExit("no cases recorded under " + ARTIFACT_DIR)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    # ---------------------------------------------------------------- P-delta
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for case in cases:
        npz = np.load(os.path.join(ARTIFACT_DIR, case["tag"] + ".npz"))
        ax.plot(1e3 * npz["deflection"], npz["load"] / 1e3, lw=1.6, label=case["tag"])
    reference = cases[0]["peak"]["analytic_load_kn"]
    ax.axhline(reference, ls="--", c="k", lw=1.0,
               label=f"flexural limit {reference:.2f} kN")
    ax.set_xlabel("load-point deflection [mm]")
    ax.set_ylabel("total load [kN]")
    ax.set_title("Ice three-point bend: cohesive load-displacement")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(ARTIFACT_DIR, "load_deflection.png"), dpi=160)
    plt.close(fig)

    # ------------------------------------------------------ crack growth / R
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for case in cases:
        npz = np.load(os.path.join(ARTIFACT_DIR, case["tag"] + ".npz"))
        ax.plot(1e3 * npz["crack"], npz["load"] / 1e3, lw=1.6, label=case["tag"])
    ax.set_xlabel("crack height a [mm]")
    ax.set_ylabel("total load [kN]")
    ax.set_title("Crack growth resistance (P-a)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(ARTIFACT_DIR, "crack_resistance.png"), dpi=160)
    plt.close(fig)

    # -------------------------------------------------- damage kymograph
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    tag = cases[0]["tag"]
    npz = np.load(os.path.join(ARTIFACT_DIR, tag + ".npz"))
    ax.plot(npz["damage"], 1e3 * npz["seg_y"], "o-", ms=3)
    ax.set_xlabel("damage d at the interface (0 = intact, 1 = failed)")
    ax.set_ylabel("height above tensile face [mm]")
    ax.set_title(f"Crack profile along the symmetry plane ({tag})")
    ax.set_xlim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(ARTIFACT_DIR, "damage_profile.png"), dpi=160)
    plt.close(fig)

    # ------------------------------------------------ energy accounting
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(1e3 * npz["deflection"], npz["wext"], label="external work")
    ax.plot(1e3 * npz["deflection"], npz["uel"], label="elastic energy")
    ax.plot(1e3 * npz["deflection"], npz["diss_env"], label="dissipated (envelope)")
    ax.plot(1e3 * npz["deflection"], npz["diss_path"], "--", label="dissipated (path)")
    ax.set_xlabel("load-point deflection [mm]")
    ax.set_ylabel("energy [J]")
    ax.set_title("Energy balance")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(ARTIFACT_DIR, "energy_balance.png"), dpi=160)
    plt.close(fig)

    # -------------------------------------- mesh convergence and brittleness
    mesh_cases = [c for c in cases if c["tag"].startswith("mesh")]
    if mesh_cases:
        fig, ax = plt.subplots(figsize=(6.0, 4.2))
        seg = [c["mesh"]["segment_over_lch"] for c in mesh_cases]
        err = [c["peak"]["relative_error_pct"] for c in mesh_cases]
        ax.plot(seg, err, "o-")
        ax.axhline(0.0, c="k", lw=0.8)
        ax.axhspan(-5.0, 5.0, color="tab:green", alpha=0.12, label="+-5 % gate")
        ax.set_xlabel("interface segment height / l_ch")
        ax.set_ylabel("peak load error vs flexural limit [%]")
        ax.set_title("Mesh convergence of the peak load")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(ARTIFACT_DIR, "mesh_convergence.png"), dpi=160)
        plt.close(fig)

    brittle = [c for c in cases if c["tag"].startswith("lch")]
    if brittle:
        fig, ax = plt.subplots(figsize=(6.0, 4.2))
        ratio = [c["law"]["cohesive_length"] / c["geometry"]["depth"] for c in brittle]
        err = [c["peak"]["relative_error_pct"] for c in brittle]
        ax.semilogx(ratio, err, "o-")
        ax.axhline(0.0, c="k", lw=0.8)
        ax.axhspan(-5.0, 5.0, color="tab:green", alpha=0.12, label="+-5 % gate")
        ax.set_xlabel("l_ch / d (brittleness parameter)")
        ax.set_ylabel("peak load error vs flexural limit [%]")
        ax.set_title("Size effect: cohesive peak load vs strength limit")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(ARTIFACT_DIR, "brittleness_sweep.png"), dpi=160)
        plt.close(fig)

    # ------------------------------------------------------------- tables
    csv_path = os.path.join(ARTIFACT_DIR, "cases.csv")
    keys = ["tag", "mesh.nx", "mesh.ny", "mesh.segment_over_lch", "law.cohesive_length",
            "peak.load_kn", "peak.analytic_load_kn", "peak.relative_error_pct",
            "peak.deflection_mm", "fracture.crack_height_mm", "fracture.dissipated_j",
            "fracture.gf_area_j", "fracture.energy_relative_error_pct",
            "fracture.balance_residual_pct", "solution.steps", "solution.elapsed_s"]
    with open(csv_path, "w") as handle:
        handle.write(",".join(keys) + "\n")
        for case in cases:
            row = []
            for key in keys:
                value = case
                for part in key.split("."):
                    value = value[part] if isinstance(value, dict) else ""
                row.append(f"{value:.6g}" if isinstance(value, float) else str(value))
            handle.write(",".join(row) + "\n")

    report = dict(
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        n_cases=len(cases),
        cases=cases,
        figures=sorted(
            os.path.basename(p) for p in glob.glob(os.path.join(ARTIFACT_DIR, "*.png"))
        ),
    )
    with open(os.path.join(ARTIFACT_DIR, "report.json"), "w") as handle:
        json.dump(report, handle, indent=1)
    print(f"report.json written with {len(cases)} cases; figures: {report['figures']}")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="solve one case")
    run.add_argument("--tag", required=True)
    run.add_argument("--nx", type=int, default=32)
    run.add_argument("--ny", type=int, default=60)
    run.add_argument("--young", type=float, default=ICE["young"])
    run.add_argument("--poisson", type=float, default=ICE["poisson"])
    run.add_argument("--strength", type=float, default=ICE["strength"])
    run.add_argument("--gf", type=float, default=ICE["fracture_energy"])
    run.add_argument("--span", type=float, default=ICE["span"])
    run.add_argument("--depth", type=float, default=ICE["depth"])
    run.add_argument("--thickness", type=float, default=ICE["thickness"])
    run.add_argument("--max-steps", type=int, default=400)

    sub.add_parser("report", help="merge cases into report.json + figures")

    args = parser.parse_args(argv)
    if args.command == "run":
        run_case(
            tag=args.tag, nx=args.nx, ny=args.ny, young=args.young, poisson=args.poisson,
            strength=args.strength, fracture_energy=args.gf, span=args.span,
            depth=args.depth, thickness=args.thickness, max_steps=args.max_steps,
        )
    else:
        build_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())