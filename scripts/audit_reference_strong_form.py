#!/usr/bin/env python
"""Run the strong-form reference audit and publish its evidence.

Writes, all from the same in-memory result (no hand-copied numbers):

* ``docs/reference-strong-form-audit.json`` -- the raw residuals;
* ``docs/assets/reference-strong-form/*.png`` -- field cloud, control curves and
  the metric bar chart;
* the markdown tables inside ``docs/reference-strong-form-audit.md`` between the
  ``<!--METRICS-->`` / ``<!--CONTROLS-->`` markers.

Usage::

    PYTHONPATH=src python scripts/audit_reference_strong_form.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from tensorfem.kirsch_hole import KirschHole  # noqa: E402
from tensorfem.lame_reference import lame_state  # noqa: E402
from tensorfem.reference_strong_form_audit import (  # noqa: E402
    INFORMATIONAL,
    PASS_TOLERANCE,
    kirsch_stress,
    polar_samples,
    run_audit_suite,
)

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ASSETS = DOCS / "assets" / "reference-strong-form"


def field_panel(ax) -> None:
    """Kirsch hoop stress over the annulus, with the audit sample points."""
    hole = KirschHole()
    grid = torch.linspace(-4.0, 4.0, 401, dtype=torch.float64)
    xx, yy = torch.meshgrid(grid, grid, indexing="ij")
    points = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
    sxx, syy, sxy = (t.numpy() for t in (
        kirsch_stress(hole)(points)[:, 0],
        kirsch_stress(hole)(points)[:, 1],
        kirsch_stress(hole)(points)[:, 2],
    ))
    r = np.hypot(points[:, 0].numpy(), points[:, 1].numpy())
    theta = np.arctan2(points[:, 1].numpy(), points[:, 0].numpy())
    cos, sin = np.cos(theta), np.sin(theta)
    srr = sxx * cos**2 + syy * sin**2 + 2.0 * sxy * cos * sin
    stt = sxx * sin**2 + syy * cos**2 - 2.0 * sxy * cos * sin
    del srr
    field = np.where(r >= hole.radius, stt / hole.far_field, np.nan).reshape(xx.shape)
    image = ax.imshow(field, origin="lower", extent=(-4, 4, -4, 4), cmap="inferno",
                      vmin=0, vmax=3.2)
    samples = polar_samples(1.12, 4.0, 121, seed=0).numpy()
    ax.plot(samples[:, 0], samples[:, 1], ".", ms=1.6, color="#7fe3ff",
            label="audit samples (121)")
    ax.set_title(r"Kirsch $\sigma_{\theta\theta}/T$ (exact) and audit samples")
    ax.set_xlabel("x/a")
    ax.set_ylabel("y/a")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.6)
    plt.colorbar(image, ax=ax, fraction=0.046)


def lame_panel(ax) -> None:
    """Lame hoop stress of a pressurised cylinder."""
    grid = torch.linspace(-2.0, 2.0, 401, dtype=torch.float64)
    xx, yy = torch.meshgrid(grid, grid, indexing="ij")
    points = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
    r = torch.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    state = lame_state(r, inner_radius=1.0, outer_radius=2.0, inner_pressure=5.0,
                       outer_pressure=1.0, young_modulus=200e9, poisson_ratio=0.3,
                       plane="strain")
    field = torch.where(r >= 1.0, state.hoop_stress, torch.full_like(r, float("nan")))
    image = ax.imshow(field.numpy().reshape(xx.shape), origin="lower", extent=(-2, 2, -2, 2),
                      cmap="viridis")
    samples = polar_samples(1.05, 1.9, 121, seed=1).numpy()
    ax.plot(samples[:, 0], samples[:, 1], ".", ms=1.6, color="#ff9f1c",
            label="audit samples (121)")
    ax.set_title(r"Lamé $\sigma_{\theta\theta}$ (MPa), $p_i=5$, $p_o=1$")
    ax.set_xlabel("r/a")
    ax.set_ylabel("r/a")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.6)
    plt.colorbar(image, ax=ax, fraction=0.046)


def plot_fields(result: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    field_panel(axes[0])
    lame_panel(axes[1])
    worst = result["verdict"]["references"]["kirsch_hole"]["worst_value"]
    fig.suptitle(
        "Closed-form reference fields used for accuracy claims "
        f"(worst gated residual {worst:.1e} vs tolerance {result['verdict']['tolerance']:.0e})",
        fontsize=9,
    )
    fig.tight_layout()
    path = ASSETS / "reference-fields-and-samples.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_controls(result: dict) -> Path:
    stencil = result["controls"]["kirsch_stencil"]["table"]
    perturbations = result["controls"]["kirsch_perturbation"]
    baseline = perturbations["baseline"]
    amplitudes = sorted(
        (float(key), value) for key, value in perturbations.items() if key != "baseline"
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    steps = np.array([row["step"] for row in stencil])
    residuals = np.array([row["residual"] for row in stencil])
    axes[0].loglog(steps, residuals, "o-", label="measured residual")
    reference = residuals[0] * (steps / steps[0]) ** 4
    axes[0].loglog(steps, reference, "--", label=r"$h^{4}$ reference slope")
    axes[0].axhline(PASS_TOLERANCE, color="crimson", ls=":",
                    label=f"pass gate {PASS_TOLERANCE:.0e}")
    axes[0].set_xlabel("difference step h/a")
    axes[0].set_ylabel("normalised equilibrium residual")
    axes[0].set_title(
        f"Stencil control: observed order {result['stencil_order']:.2f} "
        f"(stencil is {4}th order)"
    )
    axes[0].legend(fontsize=7)

    eps = np.array([value for value, _ in amplitudes])
    values = np.array([res for _, res in amplitudes])
    axes[1].loglog(eps, values, "s-", color="#e07a5f", label="corrupted field")
    axes[1].loglog(eps, eps, "--", color="0.4", label="1:1 detection line")
    axes[1].axhline(baseline, color="seagreen", ls=":", label="clean baseline")
    axes[1].axhline(0.03, color="navy", ls="-.", label="typical FE gate 3%")
    axes[1].set_xlabel(r"injected relative corruption $\epsilon$")
    axes[1].set_ylabel("detected residual")
    axes[1].set_title(
        f"Perturbation control: detection floor {result['detection_floor']:.0e}"
    )
    axes[1].invert_xaxis()
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    path = ASSETS / "residual-controls.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_metrics(result: dict) -> Path:
    reports = result["reports"]
    labels, values, colours = [], [], []
    for report in reports:
        for key, value in report["metrics"].items():
            if key in INFORMATIONAL:
                continue
            labels.append(f"{report['name']}\n{key}")
            values.append(max(abs(value), 1e-16))
            colours.append("#3d5a80" if report["passed"] else "crimson")
    order = np.argsort(values)[::-1]
    fig, ax = plt.subplots(figsize=(8.6, 0.24 * len(labels) + 1.8))
    ax.barh([labels[i] for i in order], [values[i] for i in order],
            color=[colours[i] for i in order])
    ax.set_xscale("log")
    ax.axvline(PASS_TOLERANCE, color="crimson", ls="--",
               label=f"pass gate {PASS_TOLERANCE:.0e}")
    ax.axvline(0.03, color="navy", ls="-.", label="typical FE gate 3%")
    ax.set_xlabel("normalised residual")
    ax.set_title("Strong-form residuals of the closed-form reference layer")
    ax.tick_params(axis="y", labelsize=6)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = ASSETS / "metric-summary.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def metrics_table(result: dict) -> str:
    rows = ["| reference | metric | value | gate | status |", "|---|---|---:|---:|---|"]
    for report in result["reports"]:
        for key, value in report["metrics"].items():
            if key in INFORMATIONAL:
                rows.append(f"| `{report['name']}` | `{key}` | {value:.3e} | info | "
                            f"{'ok' if abs(value) <= PASS_TOLERANCE else 'reported'} |")
            else:
                status = "PASS" if abs(value) <= PASS_TOLERANCE else "FAIL"
                rows.append(f"| `{report['name']}` | `{key}` | {value:.3e} | "
                            f"{PASS_TOLERANCE:.0e} | {status} |")
    verdict = result["verdict"]
    rows.append("")
    rows.append(f"Suite verdict: **{'PASS' if verdict['passed'] else 'FAIL'}** "
                f"at tolerance {verdict['tolerance']:.0e}.")
    return "\n".join(rows)


def controls_table(result: dict) -> str:
    rows = ["| injected corruption $\\epsilon$ | detected residual | residual/$\\epsilon$ |",
            "|---:|---:|---:|"]
    table = result["controls"]["kirsch_perturbation"]
    for key in sorted((k for k in table if k != "baseline"), key=lambda k: float(k), reverse=True):
        value = table[key]
        rows.append(f"| {key} | {value:.3e} | {value / float(key):.3f} |")
    rows.append(f"| clean baseline | {table['baseline']:.3e} | -- |")
    rows.append("")
    rows.append(f"Detection floor: **{result['detection_floor']:.0e}** relative; "
                f"stencil order **{result['stencil_order']:.2f}** "
                "(4th-order stencil, so the residual is truncation-limited).")
    return "\n".join(rows)


def inject(path: Path, metrics: str, controls: str) -> None:
    text = path.read_text()
    text = text.replace("<!--METRICS-->", metrics).replace("<!--CONTROLS-->", controls)
    path.write_text(text)


def main() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)
    result = run_audit_suite()
    (DOCS / "reference-strong-form-audit.json").write_text(
        json.dumps(result, indent=1, sort_keys=True) + "\n"
    )
    for path in (plot_fields(result), plot_controls(result), plot_metrics(result)):
        print(f"figure: {path.relative_to(ROOT)}")
    doc = DOCS / "reference-strong-form-audit.md"
    if doc.exists():
        inject(doc, metrics_table(result), controls_table(result))
        print(f"injected tables into {doc.relative_to(ROOT)}")
    print(metrics_table(result))
    print(controls_table(result))
    print("verdict:", "PASS" if result["verdict"]["passed"] else "FAIL")
    return 0 if result["verdict"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())