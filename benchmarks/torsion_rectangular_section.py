"""Aspect-ratio verification of the rectangular Saint-Venant torsion model.

Closes the open item recorded in ``docs/torsion-boundary-stress.md`` §5: every
published check of the closed-form rectangular solution used a *square* section, so
the ``a != b`` (shape-factor) branch and the long-side/short-side difference of the
peak shear stress were unverified.

Three mutually independent evidence streams are produced:

1. the closed-form series of ``tensorfem.torsion_bar``, reported in hand-book
   normalisation ``J = c1 a b^3``, ``tau_max = T / (c2 a b^2)``;
2. a second-order finite-difference solution of the defining PDE
   ``lap(psi) = -2`` (:func:`tensorfem.torsion_rectangular.prandtl_finite_difference`),
   Richardson-extrapolated;
3. the hand-book coefficient table (Timoshenko & Goodier; Roark).

and one 3-D finite-element stream:

4. a HEX8 twist solve of a genuinely rectangular prism
   (:func:`tensorfem.torsion_rectangular.rectangular_torsion_bar`) at three mesh
   levels, compared with the closed form through the measured twist rate, the
   centroid shear-stress L2 error and the torque-conservation identity.

Run with::

    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python benchmarks/torsion_rectangular_section.py

Writes ``docs/assets/torsion_rectangular_section/`` (JSON evidence + PNG figures).
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorfem.torsion_rectangular import (classical_rectangle_coefficients,  # noqa: E402
                                           prandtl_finite_difference, rectangular_torsion_bar,
                                           richardson, shape_factors,
                                           thin_strip_torsion_constant)

ASSETS = ROOT / "docs" / "assets" / "torsion_rectangular_section"
ASPECTS = [row.aspect for row in classical_rectangle_coefficients()]
FE_ASPECTS = [1.5, 2.0, 3.0]
FE_SEGMENTS = [3, 4, 5, 6]


def finite_difference_pair(aspect: float, *, base: int | None = None):
    """Two-level FD solve plus its Richardson extrapolation."""
    short = base if base is not None else max(24, int(round(96.0 / math.sqrt(aspect))))
    coarse = prandtl_finite_difference(aspect, 1.0, n_short=short)
    fine = prandtl_finite_difference(aspect, 1.0, n_short=2 * short)
    return {
        "n_short_coarse": coarse.n_short,
        "n_short_fine": fine.n_short,
        "j_fine": fine.torsion_constant,
        "c1_fine": fine.c1,
        "c1_extrapolated": richardson(fine.c1, coarse.c1),
        "c2_extrapolated": richardson(fine.c2, coarse.c2),
        "grad_long_mid": fine.grad_long_mid,
        "grad_short_mid": fine.grad_short_mid,
    }


def analytic_sweep():
    rows = []
    for aspect in ASPECTS:
        table = classical_rectangle_coefficients(aspect)
        series = shape_factors(aspect, 1.0)
        fd = finite_difference_pair(aspect)
        rows.append({
            "aspect": aspect,
            "table": {"c1": table.c1, "c2": table.c2},
            "series": {"c1": series.c1, "c2": series.c2,
                       "grad_long_mid": series.grad_long_mid,
                       "grad_short_mid": series.grad_short_mid,
                       "long_over_short": series.long_over_short,
                       "peak_scan": series.peak_scan,
                       "peak_location": list(series.peak_location)},
            "finite_difference": fd,
            "error_series_vs_table_pct": {
                "c1": 100.0 * (series.c1 / table.c1 - 1.0),
                "c2": 100.0 * (series.c2 / table.c2 - 1.0),
            },
            "error_fd_vs_series_pct": {
                "c1": 100.0 * (fd["c1_extrapolated"] / series.c1 - 1.0),
                "c2": 100.0 * (fd["c2_extrapolated"] / series.c2 - 1.0),
                "grad_long_mid": 100.0 * (fd["grad_long_mid"] / series.grad_long_mid - 1.0),
            },
        })
        print(f"a/b={aspect:5.1f}  c1 series {series.c1:.6f} table {table.c1:.3f} "
              f"fd {fd['c1_extrapolated']:.6f} | c2 series {series.c2:.6f} table {table.c2:.3f} "
              f"fd {fd['c2_extrapolated']:.6f} | long/short {series.long_over_short:.4f}")
    return rows


def convergence_order():
    """Observed FD order for a/b = 3 (the scheme must be second order)."""
    levels = [prandtl_finite_difference(3.0, 1.0, n_short=n) for n in (20, 40, 80)]
    c1 = [level.c1 for level in levels]
    e1, e2 = c1[0] - c1[1], c1[1] - c1[2]
    return {"levels": [level.n_short for level in levels], "c1": c1,
            "observed_order": math.log2(abs(e1 / e2)),
            "extrapolated": richardson(c1[2], c1[1])}


def finite_element_sweep():
    rows = []
    for aspect in FE_ASPECTS:
        for segments in FE_SEGMENTS:
            started = time.time()
            result = rectangular_torsion_bar(a=aspect, b=1.0, length=4.0,
                                             segments_short=segments)
            rows.append({
                "aspect": aspect,
                "segments_short": result.segments_short,
                "segments_long": result.segments_long,
                "n_elements": result.n_elements,
                "n_nodes": result.n_nodes,
                "n_dofs": result.n_dofs,
                "applied_torque": result.applied_torque,
                "torque_error": abs(result.applied_torque - 1.0),
                "measured_torsion_constant": result.measured_torsion_constant,
                "exact_torsion_constant": result.exact_torsion_constant,
                "j_error_pct": 100.0 * result.torsion_constant_relative_error,
                "shear_l2_error_pct": 100.0 * result.centroid_shear_l2_error,
                "peak_centroid_fe": result.peak_centroid_fe,
                "peak_series_sampled": result.peak_series_sampled,
                "minutes": (time.time() - started) / 60.0,
            })
            print(f"FE a/b={aspect:4.1f} seg={segments:2d} dof={result.n_dofs:5d} "
                  f"J err {rows[-1]['j_error_pct']:6.2f}%  L2 {rows[-1]['shear_l2_error_pct']:6.2f}%  "
                  f"T err {rows[-1]['torque_error']:.1e}  ({rows[-1]['minutes'] * 60:.1f}s)")
    return rows


def length_study():
    """Why the a/b = 3 twist-rate error is not monotone in the section mesh.

    With cubic elements the axial spacing is tied to the section spacing, so varying the
    length at a fixed section mesh isolates Saint-Venant end effects from mesh
    refinement: ``L/a = 1.33`` (a stubby prism) versus ``L/a = 2.67``.
    """
    rows = []
    for length in (4.0, 6.0, 8.0):
        result = rectangular_torsion_bar(a=3.0, b=1.0, length=length, segments_short=4)
        rows.append({"aspect": 3.0, "length": length, "length_over_a": length / 3.0,
                     "segments_short": result.segments_short, "n_dofs": result.n_dofs,
                     "j_error_pct": 100.0 * result.torsion_constant_relative_error})
        print(f"length study L={length:4.1f} (L/a={length / 3.0:4.2f}) dof={result.n_dofs:5d} "
              f"J err {rows[-1]['j_error_pct']:6.3f}%")
    return rows


def thin_strip_check():
    out = []
    for aspect in (10.0, 50.0, 200.0):
        series = shape_factors(aspect, 1.0, samples=801)
        strip = thin_strip_torsion_constant(aspect, 1.0)
        out.append({"aspect": aspect, "series_c1": series.c1,
                    "thin_strip_c1": strip / (aspect * 1.0**3),
                    "relative_error_pct": 100.0 * (series.c1 / (strip / (aspect * 1.0**3)) - 1.0)})
    return out


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    analytic = analytic_sweep()
    fe = finite_element_sweep()
    evidence = {
        "description": "Aspect-ratio verification of rectangular Saint-Venant torsion",
        "analytic_sweep": analytic,
        "finite_element_sweep": fe,
        "finite_difference_convergence": convergence_order(),
        "length_study": length_study(),
        "thin_strip_limit": thin_strip_check(),
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__},
    }
    path = ASSETS / "rectangular_evidence.json"
    path.write_text(json.dumps(evidence, indent=2))
    print(f"wrote {path.relative_to(ROOT)}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"matplotlib unavailable ({exc}); skipping figures")
        return

    aspects = [row["aspect"] for row in analytic]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for axis, key, label in ((axes[0], "c1", r"$c_1 = J/(ab^3)$"),
                             (axes[1], "c2", r"$c_2$,  $\tau_{max}=T/(c_2ab^2)$")):
        axis.plot(aspects, [row["series"][key] for row in analytic], "-o", ms=4,
                  label="closed-form series (TensorFEM)")
        axis.plot(aspects, [row["table"][key] for row in analytic], "s", mfc="none", ms=7,
                  label="hand-book table")
        axis.plot(aspects, [row["finite_difference"][f"{key}_extrapolated"] for row in analytic],
                  "^", ms=5, mfc="none", label="FD PDE + Richardson")
        axis.axhline(1.0 / 3.0, color="0.5", ls=":", lw=1.0, label=r"thin-strip limit $1/3$")
        axis.set_xlabel("aspect ratio a/b")
        axis.set_ylabel(label)
        axis.set_xscale("log")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    fig.suptitle("Rectangular torsion shape factors: three independent sources")
    fig.tight_layout()
    fig.savefig(ASSETS / "shape_factors_vs_aspect.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.6, 4.2))
    for key, marker in (("c1", "o"), ("c2", "s")):
        axis.plot(aspects, [abs(row["error_series_vs_table_pct"][key]) for row in analytic],
                  marker + "-", ms=4, label=f"|series - table| {key}")
        axis.plot(aspects, [abs(row["error_fd_vs_series_pct"][key]) for row in analytic],
                  marker + "--", ms=4, mfc="none", label=f"|FD - series| {key}")
    axis.axhline(1.0, color="0.4", ls=":", lw=1.0, label="1 %")
    axis.axhline(3.0, color="r", ls="--", lw=1.0, label="3 % gate")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("aspect ratio a/b")
    axis.set_ylabel("relative deviation [%]")
    axis.grid(alpha=0.3, which="both")
    axis.legend(fontsize=8)
    axis.set_title("Deviation from the hand-book table and from the PDE reference")
    fig.tight_layout()
    fig.savefig(ASSETS / "error_vs_aspect.png", dpi=150)
    plt.close(fig)

    # field cloud: shear magnitude over the a/b = 3 cross-section, plus the two mid-side
    # profiles that show where the peak actually sits.
    aspect = 3.0
    grid = 241
    xs = torch.linspace(-0.5 + 1e-6, 0.5 - 1e-6, grid, dtype=torch.float64)
    ys = torch.linspace(-aspect / 2 + 1e-6, aspect / 2 - 1e-6, int(grid * aspect),
                        dtype=torch.float64)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    from tensorfem.torsion_bar import torsion_shear_stress
    tx, ty = torsion_shear_stress(X.reshape(-1), Y.reshape(-1), aspect, 1.0, terms=401)
    magnitude = torch.sqrt(tx**2 + ty**2).reshape(X.shape).numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4),
                             gridspec_kw={"width_ratios": [1.0, 1.5]})
    contour = axes[0].contourf(X.numpy(), Y.numpy(), magnitude, levels=24, cmap="viridis")
    fig.colorbar(contour, ax=axes[0], label=r"$|\tau|/(G\theta)$")
    axes[0].plot([0.5], [0.0], "r*", ms=14, label="long-side midpoint (peak)")
    axes[0].plot([0.0], [aspect / 2], "wx", ms=9, mew=2, label="short-side midpoint")
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("x / b")
    axes[0].set_ylabel("y / b")
    axes[0].set_title("a/b = 3: shear magnitude")
    axes[0].legend(fontsize=7, loc="upper right")

    mid = int(grid * aspect) // 2
    axes[1].plot(Y[mid].numpy(), magnitude[mid], "-", label="along the long side (y = 0)")
    axes[1].plot(X[:, int(grid) // 2].numpy(), magnitude[:, int(grid) // 2], "--",
                 label="along the short side (x = 0)")
    peak_row = shape_factors(aspect, 1.0)
    axes[1].axhline(peak_row.grad_long_mid, color="r", ls=":", lw=1.0,
                    label=r"closed form $\tau_{max}/(G\theta)$")
    axes[1].axhline(peak_row.grad_short_mid, color="C2", ls=":", lw=1.0,
                    label=r"short-side midpoint")
    axes[1].set_xlabel("coordinate / b")
    axes[1].set_ylabel(r"$|\tau|/(G\theta)$")
    axes[1].set_title("Peak location: long-side midpoint, ratio "
                      f"{peak_row.long_over_short:.3f}")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ASSETS / "shear_field_aspect3.png", dpi=150)
    plt.close(fig)
    print(f"wrote 3 figures to {ASSETS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()