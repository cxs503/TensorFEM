#!/usr/bin/env python3
"""Run the TensorFEM basic analytic verification ladder and write its evidence.

Outputs (all paths relative to the repository root by default):

* ``artifacts/verification_ladder/ladder_report.json`` -- machine-readable
  criteria table (computed / reference / relative error / tolerance / pass),
  convergence rows and the paths of every figure produced here;
* ``docs/assets/basic-verification-ladder/*.png`` -- figures used by
  ``docs/basic-verification-ladder.md``.

Usage::

    python scripts/run_basic_verification_ladder.py            # ~3 min
    python scripts/run_basic_verification_ladder.py --quick    # test-sized mesh

The script is deterministic (no random numbers, float64 end to end).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorfem.verification_ladder import (  # noqa: E402
    CRITERIA,
    beam_interior_error,
    cantilever_tip_deflection,
    euler_bernoulli_static_solve,
    navier_ss_deflection,
    plate_rigidity,
    run_verification_ladder,
    ss_udl_deflection,
)
from tensorfem.modal import (  # noqa: E402
    simply_supported_beam_modes,
    simply_supported_exact_angular_frequency,
)
from tensorfem.plate_mindlin import solve_plate_on_winkler_foundation  # noqa: E402
from tensorfem.plate_vibration import solve_plate_vibration  # noqa: E402


def _grid(nodes: torch.Tensor, values: torch.Tensor) -> tuple[list, list, list, int]:
    """Reshape unstructured-but-structured-mesh node values onto a square grid."""
    xs = sorted({round(float(v), 12) for v in nodes[:, 0]})
    ys = sorted({round(float(v), 12) for v in nodes[:, 1]})
    nx, ny = len(xs), len(ys)
    if nx * ny != nodes.shape[0]:
        raise RuntimeError("node set is not a complete tensor grid")
    xi = {v: i for i, v in enumerate(xs)}
    yi = {v: i for i, v in enumerate(ys)}
    grid = [[0.0] * nx for _ in range(ny)]
    for (x, y), w in zip(nodes.tolist(), values.tolist()):
        grid[yi[round(y, 12)]][xi[round(x, 12)]] = w
    return xs, ys, grid, nx


def convergence_studies(elements: list[int], modal_elements: list[int]) -> dict:
    """Mesh refinement rows for the plate static and the modal rungs."""
    E, nu, h = 200.0e9, 0.3, 0.01
    side, q = 1.0, 1.0e3
    D = plate_rigidity(E, nu, h)
    navier_11 = navier_ss_deflection(0.5, 0.5, side, D, q, terms=11)
    navier_ref = navier_ss_deflection(0.5, 0.5, side, D, q, terms=401)
    static_rows = []
    for nx in elements:
        row = {"elements_per_side": nx, "boundaries": {}}
        for boundary in ("simply_supported", "clamped"):
            started = time.time()
            result = solve_plate_on_winkler_foundation(
                nx, side, E, nu, h, 0.0, q, boundary=boundary
            )
            w = result.displacement.view(-1, 3)[:, 0]
            centre = int(torch.argmin(
                (result.nodes[:, 0] - 0.5).abs() + (result.nodes[:, 1] - 0.5).abs()
            ))
            value = float(w[centre])
            if boundary == "simply_supported":
                error = value / navier_11 - 1.0
            else:
                from tensorfem.plate_uniform_load import clamped_uniform_ritz

                ritz = clamped_uniform_ritz(
                    side=side, young=E, poisson=nu, thickness=h, pressure=q
                )
                error = value / ritz.deflection(0.5, 0.5) - 1.0
            row["boundaries"][boundary] = {
                "centre_deflection": value,
                "relative_error": error,
                "seconds": round(time.time() - started, 1),
            }
        static_rows.append(row)

    plate_modal_rows = []
    rho = 7850.0
    for nx in modal_elements:
        result = solve_plate_vibration(
            nx, nx, length=1.0, young=E, poisson=nu, thickness=h, density=rho,
            boundary="simple", modes=3,
        )
        pairs = ((1, 1), (1, 2), (2, 1))
        refs_hz = [
            math.pi**2 * math.sqrt(D / (rho * h)) * (m * m + n * n) / (2.0 * math.pi)
            for (m, n) in pairs
        ]
        plate_modal_rows.append({
            "elements_per_side": nx,
            "modes": [f"({m},{n})" for (m, n) in pairs],
            "frequencies_hz": [float(f) for f in result.frequencies_hz],
            "reference_hz": refs_hz,
            "relative_errors": [
                float(a) / b - 1.0
                for a, b in zip(result.frequencies_hz, refs_hz)
            ],
        })

    EI_beam = 200.0e9 * 0.05 * 0.02**3 / 12.0
    rho_a = 7850.0 * 0.05 * 0.02
    beam_rows = []
    for ne in (4, 8, 16, 32):
        result = simply_supported_beam_modes(2.0, ne, EI_beam, rho_a, modes=5)
        refs = [
            simply_supported_exact_angular_frequency(2.0, EI_beam, rho_a, mode=m)
            for m in range(1, 6)
        ]
        beam_rows.append({
            "elements": ne,
            "relative_errors": [
                float(a) / b - 1.0 for a, b in zip(result.angular_frequencies, refs)
            ],
        })
    beam_static_errors = [
        beam_interior_error(2.0, ne, EI_beam, 1.0e4) for ne in (2, 4, 8)
    ]
    return {
        "beam_static_interior_errors": beam_static_errors,
        "plate_static": static_rows,
        "plate_modal": plate_modal_rows,
        "beam_modal": beam_rows,
        "navier_reference_bias": {
            "terms_11": navier_11,
            "terms_401": navier_ref,
            "relative_bias": navier_11 / navier_ref - 1.0,
        },
    }


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 130,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "savefig.bbox": "tight",
    })
    return plt


def make_figures(report: dict, studies: dict, outdir: Path) -> list[dict]:
    plt = _style()
    outdir.mkdir(parents=True, exist_ok=True)
    figures: list[dict] = []

    # --- Figure 1: beam statics, field curve + error ------------------------
    L1 = report["details"]["L1"]
    L = L1["model"]["length_m"]
    q = L1["model"]["uniform_load_N_per_m"]
    EI = L1["model"]["EI_Nm2"]
    P = L1["model"]["tip_load_N"]
    xi = [i / 200.0 * L for i in range(201)]
    w_ref = [ss_udl_deflection(x, L, q, EI) for x in xi]
    w_ana_field = w_ref
    ss = euler_bernoulli_static_solve(
        L, L1["model"]["elements"], EI, uniform_load=q, support="pinned-pinned"
    )
    cant = euler_bernoulli_static_solve(
        L, L1["model"]["elements"], EI, tip_load=P, support="clamped-free"
    )
    w_fem = [ss.deflection(x) for x in xi]
    w_cant_fem = [cant.deflection(x) for x in xi]
    w_cant_ref = [
        cantilever_tip_deflection(x, P, EI) for x in xi
    ]  # tip-loaded cantilever exact field

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    ax = axes[0]
    ax.plot([x / L for x in xi], [1e3 * v for v in w_ana_field], "--",
            label="analytic  qx(L³−2Lx²+x³)/24EI", lw=1.6)
    ax.plot([x / L for x in xi], [1e3 * v for v in w_fem], "-",
            label=f"FEM Hermite, {L1['model']['elements']} elements", lw=1.2)
    ax.plot([x / L for x in xi], [1e3 * v for v in w_cant_ref], "--",
            color="C2", label="analytic  Px²(3L−x)/6EI (cantilever)", lw=1.6)
    ax.plot([x / L for x in xi], [1e3 * v for v in w_cant_fem], "-",
            color="C3", label="FEM clamped–free, tip load", lw=1.2)
    ax.set_xlabel("x / L")
    ax.set_ylabel("w  [mm]")
    ax.set_title("L1 beam deflection: FEM vs closed form")
    ax.legend(fontsize=7, loc="lower right")
    ax = axes[1]
    err = [
        abs(f - a) / max(abs(a), 1e-30) for f, a in zip(w_fem, w_ana_field)
        if abs(a) > 1e-9 * max(abs(v) for v in w_ana_field)
    ]
    ax.semilogy([x / L for x, e in zip(xi, err) if e > 0], err, lw=1.2, color="C0")
    ax.axhline(CRITERIA["L1-SS-UDL"], color="k", ls=":", lw=1,
               label="criterion 0.5 %")
    ax.set_xlabel("x / L")
    ax.set_ylabel("|w_FEM − w_exact| / w_exact")
    ax.set_title("L1 pointwise relative error (uniform load)")
    ax.legend(fontsize=7)
    fig.suptitle("TensorFEM verification ladder — L1 Euler–Bernoulli beam", y=1.02)
    path = outdir / "l1-beam-deflection.png"
    fig.savefig(path)
    plt.close(fig)
    figures.append({"name": path.name, "caption":
                    "L1 梁验证：均布载简支梁与端部集中载悬臂梁的挠度曲线（FEM 与闭式解重合）及逐点相对误差。"})

    # --- Figure 2: simply supported plate deflection fields ----------------
    fields = report["details"]["L2"]["fields"]
    nodes = torch.tensor(fields["nodes"])
    w_ss = torch.tensor(fields["w_simply_supported"])
    w_cl = torch.tensor(fields["w_clamped"])
    D = fields["rigidity"]
    q = fields["pressure"]
    side = fields["side"]
    xs, ys, grid_fem, nx = _grid(nodes, w_ss)
    w_ana = [[navier_ss_deflection(x, y, side, D, q, terms=51) for x in xs] for y in ys]
    peak = max(abs(v) for row in w_ana for v in row)
    diff = [[100.0 * (grid_fem[j][i] - w_ana[j][i]) / peak
             for i in range(len(xs))] for j in range(len(ys))]

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.3))
    X, Y = [[v for v in row] for row in [xs] * len(ys)], [[y] * len(xs) for y in ys]
    for ax, data, title, cmap in (
        (axes[0], w_ana, "Navier series (51 terms)", "viridis"),
        (axes[1], grid_fem, "FEM Q4 Mindlin, thin limit a/h=100", "viridis"),
    ):
        cs = ax.contourf(X, Y, [[1e3 * v for v in row] for row in data], levels=16, cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal")
        fig.colorbar(cs, ax=ax, shrink=0.85, label="w [mm]")
    cs = axes[2].contourf(X, Y, diff, levels=16, cmap="RdBu_r",
                          vmin=-1.0, vmax=1.0)
    axes[2].set_title("FEM − analytic [% of peak]")
    axes[2].set_xlabel("x [m]")
    axes[2].set_ylabel("y [m]")
    axes[2].set_aspect("equal")
    fig.colorbar(cs, ax=axes[2], shrink=0.85, label="%")
    fig.suptitle("L2 simply supported square plate under uniform pressure", y=1.03)
    path = outdir / "l2-plate-ss-field.png"
    fig.savefig(path)
    plt.close(fig)
    figures.append({"name": path.name, "caption":
                    "L2 简支方板均布压力挠度场：Navier 级数解、FEM 云图与两者之差（占峰值百分比）。"})

    # --- Figure 3: clamped plate field + section profiles ------------------
    xs_c, ys_c, grid_cl, _ = _grid(nodes, w_cl)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))
    Xc = [[v for v in row] for row in [xs_c] * len(ys_c)]
    Yc = [[y] * len(xs_c) for y in ys_c]
    cs = axes[0].contourf(Xc, Yc, [[1e3 * v for v in row] for row in grid_cl],
                          levels=16, cmap="magma")
    axes[0].set_title("clamped square, FEM centre deflection")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    axes[0].set_aspect("equal")
    fig.colorbar(cs, ax=axes[0], shrink=0.85, label="w [mm]")
    mid = len(ys_c) // 2
    axes[1].plot(xs_c, [1e3 * v for v in grid_cl[mid]], "o-", ms=3,
                 label="clamped, y = a/2")
    axes[1].plot(xs_c, [1e3 * v for v in grid_fem[mid]], "s-", ms=3,
                 label="simply supported, y = a/2")
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("w [mm]")
    axes[1].set_title("mid-line deflection profiles")
    axes[1].legend(fontsize=7)
    fig.suptitle("L2 plate fields used by the ladder", y=1.02)
    path = outdir / "l2-plate-clamped-field.png"
    fig.savefig(path)
    plt.close(fig)
    figures.append({"name": path.name, "caption":
                    "L2 固支方板挠度云图与两类边界沿板中线的挠度剖面（对照 Timoshenko 表值 0.00126）。"})

    # --- Figure 4: modal comparison ---------------------------------------
    L3 = report["details"]["L3"]
    beam_modes = L3["beam"]["modes"]
    plate_modes = L3["plate"]["modes"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6))
    ax = axes[0]
    idx = [m["mode"] for m in beam_modes]
    ax.plot(idx, [m["reference_hz"] for m in beam_modes], "o--", label="analytic")
    ax.plot(idx, [m["computed_hz"] for m in beam_modes], "s-", label="FEM")
    ax.set_yscale("log")
    ax.set_xlabel("mode n")
    ax.set_ylabel("frequency [Hz]")
    ax.set_title("L3 simply supported beam, 5 modes")
    ax.legend(fontsize=7)
    ax = axes[1]
    labels = [f"({m['m']},{m['n']})" for m in plate_modes]
    pos = range(len(labels))
    width = 0.36
    ax.bar([p - width / 2 for p in pos], [m["reference_hz"] for m in plate_modes],
           width, label="Navier analytic")
    ax.bar([p + width / 2 for p in pos], [m["computed_hz"] for m in plate_modes],
           width, label="FEM Q4 Mindlin")
    for p, m in zip(pos, plate_modes):
        ax.text(p, max(m["computed_hz"], m["reference_hz"]) * 1.02,
                f"{100 * m['relative_error']:+.2f}%", ha="center", fontsize=7)
    ax.set_xticks(list(pos))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Navier mode (m,n)")
    ax.set_ylabel("frequency [Hz]")
    ax.set_title("L3 simply supported square plate, first 3 modes")
    ax.legend(fontsize=7)
    fig.suptitle("L3 modal verification", y=1.02)
    path = outdir / "l3-modal-comparison.png"
    fig.savefig(path)
    plt.close(fig)
    figures.append({"name": path.name, "caption":
                    "L3 模态验证：简支梁前 5 阶与简支方板前 3 阶频率的 FEM／解析对比（标注相对误差）。"})

    # --- Figure 5: convergence -------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.3))
    rows = studies["plate_static"]
    h = [1.0 / r["elements_per_side"] for r in rows]
    for boundary, label, color in (("simply_supported", "SS plate (vs Navier 11×11)", "C0"),
                                   ("clamped", "clamped plate (vs Ritz)", "C3")):
        axes[0].loglog(h, [abs(r["boundaries"][boundary]["relative_error"]) for r in rows],
                       "o-", label=label, color=color)
    axes[0].axhline(1e-2, color="k", ls=":", lw=1)
    axes[0].set_xlabel("h / a")
    axes[0].set_ylabel("relative error in centre deflection")
    axes[0].set_title("L2 plate static convergence")
    axes[0].legend(fontsize=6)
    pm = studies["plate_modal"]
    hh = [1.0 / r["elements_per_side"] for r in pm]
    for k, label in enumerate(["mode (1,1)", "mode (1,2)/(2,1)", "mode (2,2)"]):
        axes[1].loglog(hh, [abs(r["relative_errors"][k]) for r in pm], "o-", label=label)
    axes[1].axhline(2e-2, color="k", ls=":", lw=1)
    axes[1].set_xlabel("h / a")
    axes[1].set_ylabel("relative frequency error")
    axes[1].set_title("L3 plate modal convergence")
    axes[1].legend(fontsize=6)
    bm = studies["beam_modal"]
    hb = [1.0 / r["elements"] for r in bm]
    for k in range(5):
        axes[2].loglog(hb, [abs(r["relative_errors"][k]) for r in bm], "o-",
                       label=f"mode {k + 1}")
    axes[2].axhline(1e-2, color="k", ls=":", lw=1)
    axes[2].set_xlabel("h / L")
    axes[2].set_ylabel("relative frequency error")
    axes[2].set_title("L3 beam modal convergence")
    axes[2].legend(fontsize=6)
    fig.suptitle("Mesh-refinement evidence behind the ladder", y=1.02)
    path = outdir / "convergence.png"
    fig.savefig(path)
    plt.close(fig)
    figures.append({"name": path.name, "caption":
                    "网格收敛证据：板静力（简支/固支）、板模态与梁模态误差随 h 的下降。"})

    return figures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elements", type=int, default=16,
                        help="plate static elements per side used by the gated rungs")
    parser.add_argument("--modal-elements", type=int, default=16,
                        help="plate modal elements per side")
    parser.add_argument("--beam-elements", type=int, default=16)
    parser.add_argument("--quick", action="store_true",
                        help="coarse meshes for a fast smoke run")
    parser.add_argument("--outdir", type=Path,
                        default=ROOT / "artifacts" / "verification_ladder")
    parser.add_argument("--figdir", type=Path,
                        default=ROOT / "docs" / "assets" / "basic-verification-ladder")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    if args.quick:
        args.elements, args.modal_elements, args.beam_elements = 12, 12, 8

    started = time.time()
    report = run_verification_ladder(
        plate_elements=args.elements,
        plate_modal_elements=args.modal_elements,
        beam_elements=args.beam_elements,
        progress=lambda message: print(f"[ladder] {message}", flush=True),
    )
    studies = convergence_studies([8, 12, 16], [8, 12, 16, 20])
    figures = make_figures(report.as_dict(), studies, args.figdir)

    payload = report.as_dict()
    payload["convergence"] = studies
    payload["figures"] = figures
    payload["artifacts"] = {
        "report_json": str((args.json or args.outdir / "ladder_report.json")
                           .relative_to(ROOT)),
        "figures_dir": str(args.figdir.relative_to(ROOT)),
        "doc": "docs/basic-verification-ladder.md",
        "tests": "tests/test_verification_ladder.py",
    }
    payload["runtime_seconds"] = round(time.time() - started, 1)
    payload["mesh"] = {
        "plate_static_elements": args.elements,
        "plate_modal_elements": args.modal_elements,
        "beam_elements": args.beam_elements,
    }
    out = args.json or args.outdir / "ladder_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    print(f"[ladder] all_passed = {payload['all_passed']}")
    for check in payload["checks"]:
        print(f"[ladder] {check['criterion']:>18s}  "
              f"{check['relative_error_percent']:+.4f}%  "
              f"(tol {100 * check['tolerance']:.1f}%)  "
              f"{'PASS' if check['passed'] else 'FAIL'}")
    print(f"[ladder] wrote {out} in {payload['runtime_seconds']} s")
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())