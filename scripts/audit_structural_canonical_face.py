"""Audit of the canonical structural verification face: JSON evidence and figures.

Writes ``benchmarks/structural_canonical_face.json`` (gates C1-C6 with every
number, the aspect-ratio table, the refinement ladder, the Kirsch ladder) and the
figures under ``docs/assets/structural_canonical/``.

Run: ``PYTHONPATH=src python scripts/audit_structural_canonical_face.py``

Do **not** pin ``OMP_NUM_THREADS`` to 1 or 2 on this platform: flexiblas 3.3.0
(loongarch64) then returns a wrong-but-plausible spectrum for the reduced plate
matrices.  ``tensorfem.plate_vibration`` now refuses such output via an
eigenpair residual check, but the audit is meant to run on a healthy solver.
"""
from __future__ import annotations

import json
import math
import platform as host
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import numpy as np

from tensorfem import __version__
from tensorfem.structural_canonical_face import (
    GATE_LINEAR_ORDER,
    STRIP_ALPHA,
    TIMOSHENKO_ALPHA_TABLE,
    convergence_order,
    kirsch_q8_hole_edge_hoop_stress,
    linear_element_order_study,
    pairwise_orders,
    rectangular_plate_navier,
    run_canonical_face,
    solve_cantilever_point_load,
    solve_kirsch_q8_concentration,
    solve_plate_first_frequency,
    solve_rectangular_plate_kirchhoff,
)
from tensorfem.modal import cantilever_beam_modes
from tensorfem.plate_vibration import (
    BOUNDARY_SIMPLE,
    EIGEN_RESIDUAL_TOLERANCE,
    solve_plate_vibration,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "benchmarks" / "structural_canonical_face.json"
ASSETS = ROOT / "docs" / "assets" / "structural_canonical"
KIRSCH_LADDER = ((12, 2), (16, 2), (16, 4), (24, 4), (24, 8))
PLATE_LEVELS = (16, 24, 32)
FIGURES: list[Path] = []


def _save(fig, name: str) -> None:
    path = ASSETS / name
    fig.savefig(path, dpi=160)
    plt.close(fig)
    FIGURES.append(path)


def _beam_figure(beam) -> None:
    x = beam.nodes() / beam.length
    fig, (ax, err) = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True,
                                  gridspec_kw={"height_ratios": [2.2, 1.0]})
    ax.plot(x, beam.exact_transverse() * 1e3, "-", lw=2.0, label="exact  $Px^2(3L-x)/(6EI)$")
    ax.plot(x, beam.transverse * 1e3, "o", ms=5, mfc="none", label="FE, 8 Hermite elements")
    ax.set_ylabel("deflection  $w$  [mm]")
    ax.set_title("Cantilever, concentrated tip load $P$: $w_{tip}=PL^3/(3EI)$")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", frameon=False)
    err.semilogy(x[1:], np.abs(beam.transverse[1:] - beam.exact_transverse()[1:])
                 / np.abs(beam.exact_transverse()[1:]), "s-", ms=4, color="tab:red")
    err.set_xlabel("$x/L$")
    err.set_ylabel("rel. error")
    err.set_title("nodal relative error (round-off: the load is applied at a node)", fontsize=9)
    err.grid(alpha=0.3)
    _save(fig, "beam_static.png")


def _plate_aspect_figure(rows) -> None:
    ratios = np.linspace(1.0, 6.0, 60)
    series = np.array([float(rectangular_plate_navier(
        np.array([[0.5 * r, 0.5]]), length=r, width=1.0, young=12.0, thickness=1.0,
        poisson=0.3, pressure=1.0, terms=81)[0]) * 12.0 / (12.0 * (1.0 - 0.09))
        for r in ratios])
    table = {k: v for k, v in TIMOSHENKO_ALPHA_TABLE.items() if math.isfinite(k)}
    fig, (ax, err) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    ax.plot(ratios, series, "-", lw=2.0, color="0.25", label="Navier series (this module)")
    ax.axhline(STRIP_ALPHA, ls=":", color="tab:green", lw=1.6, label="strip limit $5/384$")
    ax.plot(list(table), list(table.values()), "s", ms=6, mfc="none", color="tab:blue",
            label="Timoshenko table 8")
    ax.plot([r["aspect_ratio"] for r in rows], [r["alpha_fe"] for r in rows], "o",
            ms=7, color="tab:red", label="BFS plate, $w_c D/(qb^4)$")
    ax.set_xlabel("aspect ratio $a/b$")
    ax.set_ylabel(r"$\alpha = w_c D/(q b^4)$")
    ax.set_title("Simply supported rectangular plate: aspect-ratio coefficient")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    labels = [f"{r['aspect_ratio']:g}" for r in rows]
    ax2 = err
    ax2.bar(labels, [100.0 * r["relative_error"] for r in rows], color="tab:red", alpha=0.7,
            label="FE vs Navier")
    ax2.axhline(100.0 * 0.02, ls="--", color="0.3", lw=1.2, label="gate 2 %")
    ax2.set_xlabel("aspect ratio $a/b$")
    ax2.set_ylabel("centre-deflection error [%]")
    ax2.set_title("C2 gate")
    ax2.grid(alpha=0.3, axis="y")
    ax2.legend(frameon=False, fontsize=8)
    _save(fig, "plate_aspect.png")


def _plate_field_figure(solution) -> None:
    """Deflection field: FE, Navier and their difference on a 2:1 rectangular plate."""
    X, Y, W = solution.grid()
    fine_x = np.linspace(0.0, solution.length, 121)
    fine_y = np.linspace(0.0, solution.width, 61)
    FX, FY = np.meshgrid(fine_x, fine_y)
    exact = solution.navier(np.stack([FX.ravel(), FY.ravel()], axis=1)).reshape(FX.shape)
    # bilinear interpolation of the FE grid onto the fine comparison grid
    ny, nx = W.shape
    gx = np.linspace(0.0, solution.length, nx)
    gy = np.linspace(0.0, solution.width, ny)
    interp = np.empty_like(exact)
    for j in range(ny - 1):
        rows = (fine_y >= gy[j]) & (fine_y <= gy[j + 1])
        t = (fine_y[rows] - gy[j]) / (gy[j + 1] - gy[j])
        for k in range(nx - 1):
            cols = (fine_x >= gx[k]) & (fine_x <= gx[k + 1])
            u = (fine_x[cols] - gx[k]) / (gx[k + 1] - gx[k])
            lower = W[j, k] * (1 - u) + W[j, k + 1] * u
            upper = W[j + 1, k] * (1 - u) + W[j + 1, k + 1] * u
            interp[np.ix_(rows, cols)] = (lower[None, :] * (1 - t[:, None])
                                          + upper[None, :] * t[:, None])
    scale = 1e6 / solution.pressure  # deflection per unit pressure, micrometres
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for ax, data, title in (
        (axes[0], interp * scale, "BFS FE, a/b = %.1f" % solution.aspect_ratio),
        (axes[1], exact * scale, "Navier series"),
        (axes[2], (interp - exact) / exact.max() * 100.0, "(FE - exact)/exact$_{max}$ [%]"),
    ):
        cmap = "viridis" if ax is not axes[2] else "RdBu_r"
        limits = {} if ax is not axes[2] else {"vmin": -3.0, "vmax": 3.0}
        cs = ax.contourf(FX, FY, data, levels=18, cmap=cmap, **limits)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9)
        fig.colorbar(cs, ax=ax, shrink=0.85)
    axes[0].set_xlabel("x [m]")
    axes[1].set_xlabel("x [m]")
    axes[2].set_xlabel("x [m]")
    fig.suptitle("Uniformly loaded simply supported rectangular plate: deflection field",
                 fontsize=10)
    _save(fig, "plate_field.png")


def _modal_figure() -> None:
    beam = cantilever_beam_modes(1.0, 8, 2000.0, 8.0, 1)
    nodes = np.linspace(0.0, 1.0, 9)
    shape = beam.modes.numpy()[0::2, 0]
    shape = shape / np.abs(shape).max()
    # exact cantilever first mode: cosh(bx) - cos(bx) - sigma (sinh(bx) - sin(bx)),
    # sigma = (cosh z + cos z)/(sinh z + sin z), z = 1.875104
    z = 1.8751040687119611
    sigma = (math.cosh(z) + math.cos(z)) / (math.sinh(z) + math.sin(z))
    xs = np.linspace(0.0, 1.0, 200)
    exact = (np.cosh(z * xs) - np.cos(z * xs)
             - sigma * (np.sinh(z * xs) - np.sin(z * xs)))
    exact = exact / np.abs(exact).max()
    plate = solve_plate_vibration(24, 12, length=2.0, width=1.0, young=200.0e9,
                                  poisson=0.3, thickness=0.01, density=7850.0,
                                  boundary=BOUNDARY_SIMPLE, modes=1)
    px = plate.nodes.numpy()
    pw = plate.modes.numpy()[0::3, 0]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.0))
    ax.plot(xs, exact, "-", lw=2.0, color="0.3", label="exact Bernoulli mode 1")
    ax.plot(nodes, shape, "o", ms=6, mfc="none", label="Hermite FE, 8 elements")
    ax.set_xlabel("$x/L$")
    ax.set_ylabel("normalised mode shape")
    ax.set_title("Cantilever mode 1, $f_1$ = (1.8751)$^2$/(2$\\pi L^2$)$\\sqrt{EI/\\rho A}$",
                 fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    cs = ax2.tricontourf(px[:, 0], px[:, 1], pw / np.abs(pw).max(), levels=16, cmap="magma")
    ax2.set_aspect("equal")
    ax2.set_title("SSSS rectangular plate, mode (1,1) from the Q4 eigen-path", fontsize=9)
    fig.colorbar(cs, ax=ax2, shrink=0.85, label="normalised $w$")
    _save(fig, "modal.png")


def _kirsch_figure(field, ladder) -> None:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax.plot(np.degrees(field["theta"]), field["exact"], "-", lw=2.0, color="0.3",
            label=r"exact $\sigma_{\theta\theta}/T = 1-2\cos 2\theta$")
    ax.plot(np.degrees(field["theta"]), field["computed"], "o", ms=5, mfc="none",
            color="tab:red", label="Q8 nodal recovery")
    ax.axhline(3.0, ls=":", color="tab:green", lw=1.4)
    ax.set_xlabel(r"$\theta$ [deg]")
    ax.set_ylabel(r"$\sigma_{\theta\theta}/T$")
    ax.set_title("Hole-edge hoop stress, uniaxial tension", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    ax2.plot([row["n_theta"] for row in ladder],
             [row["factor"] for row in ladder], "o-", label=r"$K_t$ from Q8 recovery")
    ax2.axhline(3.0, ls="-", color="0.3", lw=1.4, label="exact $K_t=3$")
    ax2.axhspan(2.9, 3.1, color="tab:green", alpha=0.15, label=r"gate $3.0\pm0.1$")
    ax2.set_xlabel(r"$n_\theta$ (with the associated $n_{radial}$ of the ladder)")
    ax2.set_ylabel("$K_t$")
    ax2.set_title("C5 gate: stress-concentration factor vs mesh", fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.legend(frameon=False, fontsize=8)
    _save(fig, "kirsch_edge.png")


def _convergence_figure(rows) -> None:
    h = np.array([row.element_size for row in rows])
    e = np.array([row.relative_l2_error for row in rows])
    order = convergence_order(rows)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.loglog(h, e, "o-", label=f"bilinear Q4 Mindlin plate, fitted p = {order:.2f}")
    ax.loglog(h, e[0] * (h / h[0]) ** 2, "--", color="0.4", label="slope 2 reference")
    for row in rows:
        ax.annotate(f"{row.n_dof} dof", (row.element_size, row.relative_l2_error),
                    textcoords="offset points", xytext=(4, -10), fontsize=7)
    ax.set_xlabel("element size $h/b$")
    ax.set_ylabel(r"relative $L_2$ error of $w$")
    ax.set_title("C6: linear-element convergence of the deflection field", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, "convergence.png")


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    face = run_canonical_face(
        plate_elements=(16, 16),
        convergence_levels=(4, 8, 12, 16, 20),
        kirsch_n_theta=24,
        kirsch_n_radial=8,
    )

    # extra plate meshes so the aspect-ratio arm shows a mesh ladder, not one point
    aspect_extra = {}
    for ratio in (1.0, 2.0, 3.0):
        aspect_extra[str(ratio)] = []
        for level in PLATE_LEVELS:
            solution = solve_rectangular_plate_kirchhoff(
                length=ratio, width=1.0, elements_x=int(level * ratio), elements_y=level
            )
            aspect_extra[str(ratio)].append(
                {
                    "elements": [int(level * ratio), level],
                    "n_dof": solution.n_dof,
                    "alpha_fe": solution.alpha,
                    "relative_error": solution.centre_error(),
                }
            )

    kirsch_ladder = []
    for n_theta, n_radial in KIRSCH_LADDER:
        result = solve_kirsch_q8_concentration(n_theta=n_theta, n_radial=n_radial)
        kirsch_ladder.append(
            {
                "n_theta": result.n_theta,
                "n_radial": result.n_radial,
                "n_node": result.n_node,
                "factor": result.factor,
                "absolute_error": result.absolute_error,
                "displacement_relative_l2": result.displacement_relative_l2,
            }
        )
    field = kirsch_q8_hole_edge_hoop_stress(n_theta=24, n_radial=8)
    plate_frequency = solve_plate_first_frequency(elements_x=32, elements_y=16)

    rows = face["arms"]["C6"]["rows"]
    ladder = linear_element_order_study((4, 8, 12, 16, 20))
    assert [row.level for row in ladder] == [row["level"] for row in rows]

    beam = solve_cantilever_point_load(elements=8)
    _beam_figure(beam)
    _plate_aspect_figure(face["arms"]["C2"]["rows"])
    _plate_field_figure(
        solve_rectangular_plate_kirchhoff(length=2.0, width=1.0, elements_x=32, elements_y=16)
    )
    _modal_figure()
    _kirsch_figure(field, kirsch_ladder)
    _convergence_figure(ladder)

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": {"platform": host.platform(), "machine": host.machine(),
                 "python": host.python_version(),
                 "torch_threads": torch.get_num_threads(),
                 "eigen_residual_tolerance": EIGEN_RESIDUAL_TOLERANCE},
        "tensorfem_version": __version__,
        "command": "PYTHONPATH=src python scripts/audit_structural_canonical_face.py",
        "gates": {
            "C1_beam_tip_deflection": {"gate": "relative error < 1 %",
                                       "value": face["arms"]["C1"]["relative_error"],
                                       "pass": face["arms"]["C1"]["pass"]},
            "C2_plate_centre_deflection": {
                "gate": "relative error < 2 % (every aspect ratio)",
                "worst": max(row["relative_error"] for row in face["arms"]["C2"]["rows"]),
                "pass": face["arms"]["C2"]["pass"]},
            "C3_beam_first_frequency": {"gate": "relative error < 3 %",
                                        "value": face["arms"]["C3"]["relative_error"],
                                        "pass": face["arms"]["C3"]["pass"]},
            "C4_plate_first_frequency": {"gate": "relative error < 3 %",
                                         "value": plate_frequency["relative_error"],
                                         "pass": face["arms"]["C4"]["pass"]},
            "C5_stress_concentration": {"gate": "|K_t - 3| < 0.1",
                                        "value": face["arms"]["C5"]["computed_factor"],
                                        "pass": face["arms"]["C5"]["pass"]},
            "C6_linear_element_order": {"gate": "observed L2 order >= 1.9",
                                        "value": face["arms"]["C6"]["observed_order"],
                                        "pass": face["arms"]["C6"]["pass"]},
        },
        "all_pass": face["all_pass"],
        "arms": face["arms"],
        "plate_aspect_mesh_ladder": aspect_extra,
        "kirsch_ladder": kirsch_ladder,
        "kirsch_edge_hoop_stress": {
            "theta_deg": (np.degrees(field["theta"]).tolist()),
            "computed": field["computed"].tolist(),
            "exact": field["exact"].tolist(),
        },
        "kirsch_edge_max_abs_deviation": float(
            np.abs(field["computed"] - field["exact"]).max()
        ),
        "plate_first_frequency_fine_mesh": plate_frequency,
        "pairwise_orders": pairwise_orders(ladder),
        "gate_linear_order": GATE_LINEAR_ORDER,
        "figures": [str(path.relative_to(ROOT)) for path in FIGURES],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")
    for key, gate in payload["gates"].items():
        print(f"  {key:28s} value={gate.get('value', gate.get('worst'))} pass={gate['pass']}")


if __name__ == "__main__":
    main()
