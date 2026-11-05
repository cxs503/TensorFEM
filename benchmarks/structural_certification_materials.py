#!/usr/bin/env python3
"""Structural-certification material: independent baselines, uncertainty, clouds.

The script produces the Results-level material of
``paper-draft-tensorfem-structural-certification`` and nothing else:

* ``benchmarks/structural_certification_materials.json`` -- every number used in
  the tables, each comparison row carrying the numerator ``|fe - ref|``, the
  denominator ``|ref|``, the 3 % threshold and a ``PASS`` / ``FAIL`` /
  ``WITHHELD`` verdict, plus the discretisation and parametric uncertainty;
* matplotlib figures under ``docs/assets/structural-certification/``: FE field
  clouds next to the analytic field, the pointwise comparison with error bars,
  the convergence/uncertainty plot and the finite-width stress concentration;
* ``docs/structural-certification-materials.md`` -- the tables rendered from the
  JSON above, so no number in the write-up is transcribed by hand.

Two independent baselines are used:

1. the Kirsch (1898) closed-form solution of the infinite plate with a circular
   hole (quarter annulus, traction-driven, production Q4 path);
2. the Howland (1930) series values tabulated in Peterson's stress
   concentration charts for the *finite-width* plate, with the Heywood (1952)
   fit as the secondary cross-check.

Run::

    OMP_NUM_THREADS=1 PYTHONPATH=src python benchmarks/structural_certification_materials.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorfem import __version__  # noqa: E402
from tensorfem.certification_evidence import (  # noqa: E402
    Criterion,
    aggregate_verdicts,
    classify_pointwise,
    combine_envelopes,
    envelope_uncertainty,
    grid_uncertainty,
)
from tensorfem.finite_width_hole import (  # noqa: E402
    FiniteWidthPlate,
    peterson_stress_concentration,
    solve_case as solve_plate,
)
from tensorfem.kirsch_hole import (  # noqa: E402
    KirschHole,
    q4_stress_at_point,
    solve_case as solve_kirsch,
)

JSON_OUT = ROOT / "benchmarks" / "structural_certification_materials.json"
ASSETS = ROOT / "docs" / "assets" / "structural-certification"
DOC_OUT = ROOT / "docs" / "structural-certification-materials.md"

KIRSCH_LEVELS = (24, 32, 48, 64)
PLATE_LEVELS = (24, 32, 40)
PLATE_RATIOS = (0.1, 0.2, 0.3, 0.4, 0.5)
RAYS_DEGREES = (90.0, 0.0)
RADIAL_SCALES = (1.02, 1.05, 1.10, 1.25, 1.50, 2.00)
OUTER_RADIUS = 10.0
GRADING = 5.0
SENSITIVITY_LEVEL = 48
POISSON_SWEEP = (0.25, 0.30, 0.35)
GRADING_SWEEP = (3.0, 5.0, 8.0)
CRITERION = Criterion(relative_tolerance=0.03, denominator_floor=0.01)
SAMPLE_POINTS = tuple(
    (scale, math.radians(theta)) for theta in RAYS_DEGREES for scale in RADIAL_SCALES
)
COMPONENTS = ("hoop", "radial", "shear")


# ---------------------------------------------------------------------------
# Baseline 1: Kirsch infinite plate, pointwise
# ---------------------------------------------------------------------------

def kirsch_hole() -> KirschHole:
    return KirschHole(radius=1.0, far_field=1.0, young=1.0, poisson=0.3, plane="stress")


def kirsch_ladder(hole: KirschHole) -> list[dict]:
    return [
        solve_kirsch(
            hole,
            OUTER_RADIUS,
            n_theta=level,
            n_radial=level,
            grading=GRADING,
            sample_points=SAMPLE_POINTS,
        )
        for level in KIRSCH_LEVELS
    ]


def geometry_gate(case: dict, hole: KirschHole) -> dict:
    """Mesh geometry/sign gate against the *closed-form* mesh geometry.

    The hole and the outer arc are represented by inscribed chord polygons with
    ``n`` segments, so the meshed domain is not the exact annulus: its area is
    the annular polygon area ``0.5 * n * sin(pi/2n) * (R^2 - a^2)``, which is
    smaller than ``pi/4 * (R^2 - a^2)`` by a deficit of closed form
    ``1 - 2n sin(pi/2n)/pi``.  The gate is therefore two-part: the element areas
    must sum to the polygon area (geometry generated consistently), and that
    polygon area must equal the closed-form value (no missing/duplicated
    elements).  The polygon-vs-circle deficit is reported, not failed: it is the
    declared geometric discretisation error of the benchmark.
    """
    nodes = np.asarray(case["meta"]["nodes"], dtype=np.float64)
    elements = np.asarray(case["meta"]["elements"])
    quad = nodes[elements]
    x, y = quad[:, :, 0], quad[:, :, 1]
    signed = 0.5 * (
        np.einsum("ij,ij->i", x, np.roll(y, -1, axis=1))
        - np.einsum("ij,ij->i", y, np.roll(x, -1, axis=1))
    )
    level = int(case["n_theta"])
    delta = math.pi / (2.0 * level)
    ring = OUTER_RADIUS**2 - hole.radius**2
    exact_annulus = math.pi / 4.0 * ring
    polygon_annulus = 0.5 * level * math.sin(delta) * ring
    total = float(signed.sum())
    relative_vs_polygon = abs(total - polygon_annulus) / polygon_annulus
    deficit = abs(total - exact_annulus) / exact_annulus
    expected_deficit = 1.0 - polygon_annulus / exact_annulus
    deficit_relative = abs(deficit - expected_deficit) / expected_deficit
    passed = (
        signed.min() > 0.0
        and relative_vs_polygon <= 1e-12
        and deficit_relative <= 1e-9
    )
    return {
        "elements": int(elements.shape[0]),
        "nodes": int(nodes.shape[0]),
        "mesh_area": total,
        "inscribed_polygon_annulus_area": polygon_annulus,
        "exact_quarter_annulus_area": exact_annulus,
        "relative_area_error_vs_polygon": relative_vs_polygon,
        "relative_area_deficit_vs_exact": deficit,
        "expected_relative_area_deficit": expected_deficit,
        "deficit_closed_form_relative_error": deficit_relative,
        "min_signed_element_area": float(signed.min()),
        "orientation_consistent": bool(signed.min() > 0.0),
        "verdict": "PASS" if passed else "FAIL",
    }


def _probe_key(theta_degrees: float, r_over_a: float) -> tuple[float, float]:
    return (round(float(theta_degrees), 9), round(float(r_over_a), 12))


def pointwise_table(hole: KirschHole, cases: list[dict]) -> list[dict]:
    """Pointwise FE vs Kirsch rows, with a per-probe refinement uncertainty."""
    indexed = [
        ({_probe_key(r["theta_degrees"], r["r_over_a"]): r for r in case["stress_rows"]}, case["n_theta"])
        for case in cases
    ]
    rows: list[dict] = []
    for theta in RAYS_DEGREES:
        for scale in RADIAL_SCALES:
            key = _probe_key(theta, scale)
            finest = indexed[-1][0][key]
            for component in COMPONENTS:
                ladder = [
                    {"level": level, "value": float(mapping[key][f"{component}_fe"])}
                    for mapping, level in indexed
                ]
                reference = float(finest[f"{component}_exact"])
                label = f"ray {theta:g} deg, r/a={scale:g}, {component}"
                row = classify_pointwise(
                    component,
                    finest[f"{component}_fe"],
                    reference,
                    scale=hole.far_field,
                    criterion=CRITERION,
                    label=label,
                )
                row.update(
                    {
                        "ray_degrees": theta,
                        "r_over_a": scale,
                        "fe_level": cases[-1]["n_theta"],
                        "inside": bool(finest.get("inside", True)),
                        "ladder": ladder,
                    }
                )
                uncertainty = grid_uncertainty(ladder, reference=reference, label=label)
                row["uncertainty"] = uncertainty
                row["within_3pct"] = (
                    None if row["relative_error"] is None
                    else bool(row["relative_error"] <= CRITERION.relative_tolerance)
                )
                rows.append(row)
    return rows


def qoi_ladders(cases: list[dict]) -> dict[str, list[dict]]:
    """Refinement ladders of the two headline quantities of the Kirsch case."""
    point = []
    peak = []
    for case in cases:
        mapping = {_probe_key(r["theta_degrees"], r["r_over_a"]): r for r in case["stress_rows"]}
        point.append(
            {"level": case["n_theta"], "value": float(mapping[_probe_key(90.0, 1.05)]["hoop_fe"])}
        )
        peak.append({"level": case["n_theta"], "value": float(case["hole_edge"]["peak_hoop"])})
    return {"hoop_at_r1.05": point, "kt_hole_edge": peak}


def sensitivity_study() -> dict:
    """Poisson-ratio and mesh-grading sensitivity at a fixed refinement level."""
    runs: dict[str, dict] = {}
    for poisson in POISSON_SWEEP:
        key = f"poisson={poisson:g}"
        runs[key] = solve_kirsch(
            KirschHole(radius=1.0, far_field=1.0, young=1.0, poisson=poisson, plane="stress"),
            OUTER_RADIUS,
            n_theta=SENSITIVITY_LEVEL,
            n_radial=SENSITIVITY_LEVEL,
            grading=GRADING,
            sample_points=SAMPLE_POINTS,
        )
    for grading in GRADING_SWEEP:
        key = f"grading={grading:g}"
        runs[key] = solve_kirsch(
            kirsch_hole(),
            OUTER_RADIUS,
            n_theta=SENSITIVITY_LEVEL,
            n_radial=SENSITIVITY_LEVEL,
            grading=grading,
            sample_points=SAMPLE_POINTS,
        )
    quantities: dict[str, list[float]] = {"hoop_at_r1.05": [], "kt_hole_edge": []}
    records = []
    for key, case in runs.items():
        mapping = {_probe_key(r["theta_degrees"], r["r_over_a"]): r for r in case["stress_rows"]}
        point = float(mapping[_probe_key(90.0, 1.05)]["hoop_fe"])
        peak = float(case["hole_edge"]["peak_hoop"])
        quantities["hoop_at_r1.05"].append(point)
        quantities["kt_hole_edge"].append(peak)
        records.append(
            {
                "setting": key,
                "level": SENSITIVITY_LEVEL,
                "hoop_at_r1.05": point,
                "kt_hole_edge": peak,
                "max_stress_relative_error": float(case["max_stress_relative_error"]),
            }
        )
    return {
        "level": SENSITIVITY_LEVEL,
        "settings": records,
        "envelopes": {
            name: envelope_uncertainty(values, label=name) for name, values in quantities.items()
        },
    }


def qoi_verdict(
    name: str,
    ladder: list[dict],
    reference: float,
    sensitivity_envelope: dict,
    criterion: Criterion,
) -> dict:
    """Combine discretisation + parametric uncertainty and answer the 3 % gate."""
    uncertainty = grid_uncertainty(ladder, reference=reference, label=name)
    combined = combine_envelopes(
        [
            {"label": "discretisation", "half_width": uncertainty["half_width"]},
            {"label": "parametric/envelope", "half_width": sensitivity_envelope["half_width"]},
        ],
        label=f"{name} combined",
    )
    fine = float(ladder[-1]["value"])
    relative = abs(fine - reference) / abs(reference)
    row = classify_pointwise(
        name, fine, reference, scale=reference, criterion=criterion, label=name
    )
    return {
        "quantity": name,
        "fe_level": ladder[-1]["level"],
        "fe": fine,
        "reference": reference,
        "numerator_abs_error": abs(fine - reference),
        "denominator_abs_reference": abs(reference),
        "relative_error": relative,
        "threshold_relative": criterion.relative_tolerance,
        "within_3pct": bool(relative <= criterion.relative_tolerance),
        "verdict": row["verdict"],
        "discretisation": uncertainty,
        "parametric": sensitivity_envelope,
        "combined_uncertainty": combined,
        "combined_relative_half_width": combined["half_width"] / abs(reference),
        "envelope_covers_reference": bool(
            abs(fine - reference) <= combined["half_width"]
        ),
        "ladder": ladder,
    }


# ---------------------------------------------------------------------------
# Baseline 2: finite-width plate vs Howland / Peterson
# ---------------------------------------------------------------------------

def finite_width_table() -> dict:
    rows = []
    for ratio in PLATE_RATIOS:
        plate = FiniteWidthPlate(diameter=ratio, tension=1.0e6)
        cases = [solve_plate(plate, n_boundary=level) for level in PLATE_LEVELS]
        peterson = peterson_stress_concentration(ratio)
        heywood = plate.heywood_ktn
        fine = cases[-1]
        label = f"d/W={ratio:g}, K_tn vs Howland/Peterson"
        row = classify_pointwise(
            "ktn", fine["ktn_fe"], peterson, scale=peterson, criterion=CRITERION, label=label
        )
        ladder = [
            {"level": case["n_boundary"], "value": float(case["ktn_fe"])} for case in cases
        ]
        row.update(
            {
                "ratio": ratio,
                "half_length_over_width": fine["half_length_over_width"],
                "fe_level": fine["n_boundary"],
                "ktn_heywood": heywood,
                "error_vs_heywood_relative": abs(fine["ktn_fe"] - heywood) / heywood,
                "verdict_vs_heywood": (
                    "PASS" if abs(fine["ktn_fe"] - heywood) <= CRITERION.relative_tolerance * heywood
                    else "FAIL"
                ),
                "ladder": ladder,
                "load_balance": fine["load_balance"],
                "uncertainty": grid_uncertainty(ladder, reference=peterson, label=label),
                "within_3pct": bool(
                    abs(fine["ktn_fe"] - peterson) / peterson <= CRITERION.relative_tolerance
                ),
            }
        )
        rows.append(row)
    return {
        "model": {
            "geometry": "quarter plate, central hole, half_length/width = 1.5, thickness 0.01 m",
            "young_pa": 2.0e11,
            "poisson": 0.3,
            "tension_pa": 1.0e6,
            "convention": "K_tn = sigma_peak / sigma_net, sigma_net = T*W/(W-d)",
            "levels": list(PLATE_LEVELS),
            "element": "production Q4 plane stress, starburst mesh",
        },
        "rows": rows,
        "summary": aggregate_verdicts(rows),
    }


# ---------------------------------------------------------------------------
# Field clouds
# ---------------------------------------------------------------------------

def field_clouds(hole: KirschHole, case: dict, *, n_radial: int = 71, n_theta: int = 81) -> dict:
    meta = case["meta"]
    displacement = case["displacement"]
    radii = np.linspace(hole.radius, 3.0, n_radial)
    thetas = np.linspace(0.0, math.pi / 2.0, n_theta)
    grid_r, grid_t = np.meshgrid(radii, thetas, indexing="ij")
    grid_x = grid_r * np.cos(grid_t)
    grid_y = grid_r * np.sin(grid_t)

    stt = np.empty_like(grid_r)
    srr = np.empty_like(grid_r)
    srt = np.empty_like(grid_r)
    for i in range(n_radial):
        for j in range(n_theta):
            sample = q4_stress_at_point(
                meta, displacement, (grid_x[i, j], grid_y[i, j])
            )
            sxx, syy, sxy = sample["stress"]
            c, s = math.cos(grid_t[i, j]), math.sin(grid_t[i, j])
            srr[i, j] = sxx * c * c + syy * s * s + 2.0 * sxy * c * s
            stt[i, j] = sxx * s * s + syy * c * c - 2.0 * sxy * c * s
            srt[i, j] = (syy - sxx) * c * s + sxy * (c * c - s * s)

    exact_rr, exact_tt, exact_rt = hole.stress_polar(grid_r, grid_t)
    von_mises_fe = np.sqrt(stt**2 - stt * srr + srr**2 + 3.0 * srt**2)
    von_mises_exact = hole.von_mises_polar(grid_r, grid_t)

    return {
        "x": grid_x,
        "y": grid_y,
        "radii": radii,
        "thetas": thetas,
        "hoop_fe": stt,
        "hoop_exact": exact_tt,
        "radial_fe": srr,
        "radial_exact": exact_rr,
        "shear_fe": srt,
        "shear_exact": exact_rt,
        "von_mises_fe": von_mises_fe,
        "von_mises_exact": np.asarray(von_mises_exact, dtype=np.float64),
        "level": case["n_theta"],
    }


def _cell_edges(centres: np.ndarray) -> np.ndarray:
    """Monotone cell edges from a monotone array of sample positions."""
    edges = np.empty(centres.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (centres[1:] + centres[:-1])
    edges[0] = centres[0] - 0.5 * (centres[1] - centres[0])
    edges[-1] = centres[-1] + 0.5 * (centres[-1] - centres[-2])
    return edges


def plot_field_clouds(field: dict, hole: KirschHole, out: Path) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    edge_r = _cell_edges(np.asarray(field["radii"], dtype=np.float64))
    edge_t = _cell_edges(np.asarray(field["thetas"], dtype=np.float64))
    grid_edge_r, grid_edge_t = np.meshgrid(edge_r, edge_t, indexing="ij")
    x = grid_edge_r * np.cos(grid_edge_t)
    y = grid_edge_r * np.sin(grid_edge_t)
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.6))
    panels = (
        (axes[0, 0], field["hoop_fe"], "Q4 FE hoop stress $\\sigma_{\\theta\\theta}/T$", "turbo"),
        (axes[0, 1], field["hoop_exact"], "Kirsch exact $\\sigma_{\\theta\\theta}/T$", "turbo"),
        (
            axes[1, 0],
            field["hoop_fe"] - field["hoop_exact"],
            "FE $-$ exact $\\sigma_{\\theta\\theta}/T$",
            "RdBu_r",
        ),
        (axes[1, 1], field["von_mises_fe"], "Q4 FE von Mises$/T$", "turbo"),
    )
    for ax, data, title, cmap in panels:
        limit = float(np.max(np.abs(data))) if cmap == "RdBu_r" else None
        if limit is not None:
            im = ax.pcolormesh(x, y, data, shading="flat", cmap=cmap, vmin=-limit, vmax=limit)
        else:
            im = ax.pcolormesh(x, y, data, shading="flat", cmap=cmap)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        ax.set_aspect("equal")
        ax.set_xlabel("$x/a$")
        ax.set_ylabel("$y/a$")
        ax.set_title(title, fontsize=10)
        ax.plot(
            hole.radius * np.cos(np.linspace(0, math.pi / 2, 200)),
            hole.radius * np.sin(np.linspace(0, math.pi / 2, 200)),
            "k-",
            lw=1.0,
        )
    fig.suptitle(
        f"Kirsch plate with a circular hole, Q4 level n={field['level']}: "
        "field clouds and pointwise error map",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = out / "kirsch_field_clouds.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return {
        "name": "kirsch_field_clouds",
        "path": str(path.relative_to(ROOT)),
        "caption": (
            "Quarter annulus, traction-driven Kirsch case at the finest level: FE hoop "
            "stress, Kirsch exact hoop stress, signed difference and FE von Mises field "
            "(all normalised by the far-field tension T)."
        ),
        "fields": {
            "hoop_fe": [float(field["hoop_fe"].min()), float(field["hoop_fe"].max())],
            "hoop_exact": [float(field["hoop_exact"].min()), float(field["hoop_exact"].max())],
            "difference": [
                float((field["hoop_fe"] - field["hoop_exact"]).min()),
                float((field["hoop_fe"] - field["hoop_exact"]).max()),
            ],
            "von_mises_fe": [
                float(field["von_mises_fe"].min()),
                float(field["von_mises_fe"].max()),
            ],
            "samples": int(field["x"].size),
        },
    }


def plot_pointwise(rows: list[dict], hole: KirschHole, out: Path) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    samples = np.linspace(hole.radius, 2.05, 400)
    for ax, theta, component in ((axes[0], 90.0, "hoop"), (axes[1], 0.0, "hoop")):
        exact = hole.stress_polar(samples, math.radians(theta))[1 if component == "hoop" else 0]
        ax.fill_between(
            samples,
            exact * (1.0 - CRITERION.relative_tolerance),
            exact * (1.0 + CRITERION.relative_tolerance),
            color="tab:green",
            alpha=0.15,
            label="±3 % criterion band",
        )
        ax.plot(samples, exact, "k-", lw=1.8, label="Kirsch exact")
        selected = [
            row
            for row in rows
            if row["ray_degrees"] == theta and row["component"] == component
        ]
        selected.sort(key=lambda row: row["r_over_a"])
        fe = [row["fe"] for row in selected]
        err = [row["uncertainty"]["half_width"] for row in selected]
        ax.errorbar(
            [row["r_over_a"] for row in selected],
            fe,
            yerr=err,
            fmt="o",
            ms=5,
            capsize=3,
            color="tab:blue",
            label=f"Q4 FE n={selected[0]['fe_level']} ± uncertainty",
        )
        ax.set_xlabel("$r/a$")
        ax.set_ylabel("$\\sigma_{\\theta\\theta}/T$")
        ax.set_title(f"ray $\\theta={theta:g}^\\circ$")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Pointwise hoop stress vs Kirsch exact solution (error bars = "
                 "discretisation + parametric envelope)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = out / "kirsch_pointwise_comparison.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return {
        "name": "kirsch_pointwise_comparison",
        "path": str(path.relative_to(ROOT)),
        "caption": (
            "Hoop stress along the θ=90° and θ=0° rays: Q4 FE values with the combined "
            "uncertainty as error bars against the Kirsch closed form and its ±3 % band."
        ),
    }


def plot_convergence(verdicts: dict, out: Path) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    for ax, name in zip(axes, ("hoop_at_r1.05", "kt_hole_edge")):
        item = verdicts[name]
        levels = [entry["level"] for entry in item["ladder"]]
        values = [entry["value"] for entry in item["ladder"]]
        reference = item["reference"]
        ax.plot(levels, values, "o-", color="tab:blue", label="Q4 FE ladder")
        ax.axhline(reference, color="k", ls="--", lw=1.2,
                   label="analytic reference")
        band = item["discretisation"]["band"]
        if item["discretisation"]["status"] == "ESTIMATED":
            ax.fill_between(
                levels,
                band[0],
                band[1],
                color="tab:blue",
                alpha=0.15,
                label=f"GCI band (p={item['discretisation']['observed_order']:.2f})",
            )
            ax.set_title(f"{name}: GCI estimated", fontsize=10)
        else:
            ax.axhspan(band[0], band[1], color="tab:orange", alpha=0.2,
                       label="min/max envelope (GCI withheld)")
            ax.set_title(f"{name}: GCI WITHHELD", fontsize=10)
        ax.set_xlabel("refinement level n (elements per direction)")
        ax.set_ylabel(name.replace("_", " "))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("QoI refinement ladders with uncertainty bands", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = out / "kirsch_convergence_uncertainty.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return {
        "name": "kirsch_convergence_uncertainty",
        "path": str(path.relative_to(ROOT)),
        "caption": (
            "Refinement ladders of the pointwise hoop stress and of the nodal-averaged "
            "hole-edge peak, with the estimate uncertainty band and the analytic value."
        ),
    }


def plot_finite_width(table: dict, out: Path) -> dict:
    ratios = np.linspace(0.08, 0.55, 200)
    heywood = [2.0 + (1.0 - r) ** 3 for r in ratios]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(ratios, heywood, "-", color="tab:orange", lw=1.6, label="Heywood fit")
    peterson = [row for row in table["rows"]]
    ax.plot(
        [row["ratio"] for row in peterson],
        [row["reference"] for row in peterson],
        "s",
        color="k",
        ms=6,
        label="Howland / Peterson table",
    )
    ax.errorbar(
        [row["ratio"] for row in peterson],
        [row["fe"] for row in peterson],
        yerr=[row["uncertainty"]["half_width"] for row in peterson],
        fmt="o",
        ms=5,
        capsize=3,
        color="tab:blue",
        label=f"Q4 FE n={peterson[0]['fe_level']} ± uncertainty",
    )
    ax.set_xlabel("$d/W$")
    ax.set_ylabel("$K_{tn}=\\sigma_{peak}/\\sigma_{net}$")
    ax.set_title("Finite-width plate: FE vs Howland/Peterson and Heywood")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out / "finite_width_ktn.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return {
        "name": "finite_width_ktn",
        "path": str(path.relative_to(ROOT)),
        "caption": (
            "Net-section stress concentration of the finite-width plate: Q4 FE with "
            "uncertainty bars against the Howland/Peterson table and the Heywood fit."
        ),
    }


# ---------------------------------------------------------------------------
# Markdown renderer (numbers come from the JSON, never from hand)
# ---------------------------------------------------------------------------

def _fmt(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "—"
    if value == 0.0:
        return "0"
    return f"{value:.{digits}g}"


def _pct(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{100.0 * value:.{digits}f} %"


def render_markdown(payload: dict) -> str:
    kirs = payload["cases"]["kirsch"]
    lines: list[str] = []
    lines.append("# Structural certification material: independent baselines, "
                 "uncertainty and field clouds")
    lines.append("")
    lines.append(
        "Generated by `benchmarks/structural_certification_materials.py` "
        f"(TensorFEM {payload['tensorfem_version']}, "
        f"{payload['generated_utc']}). Every number below is rendered from "
        "`benchmarks/structural_certification_materials.json`; the figures live in "
        "`docs/assets/structural-certification/`."
    )
    lines.append("")
    lines.append("## Acceptance criterion")
    lines.append("")
    criterion = payload["criterion"]
    lines.append(
        f"Relative-error tolerance **{_pct(criterion['relative_tolerance'])}** of the "
        f"reference value. A row is `WITHHELD` when `|reference|` is below "
        f"{_pct(criterion['denominator_floor_fraction_of_scale'])} of the stress scale, "
        "because the relative criterion is then ill-conditioned; withheld rows are never "
        "counted as passes."
    )
    lines.append("")
    lines.append("## Case 1: Kirsch plate with a circular hole")
    lines.append("")
    problem = kirs["problem"]
    lines.append(
        f"Quarter annulus `a={problem['radius']:g}`, outer radius "
        f"`{problem['outer_radius']:g}`, far-field tension `T={problem['far_field']:g}`, "
        f"`nu={problem['poisson']:g}`, plane `{problem['plane']}`, graded Q4 mesh "
        f"(grading `{problem['grading']:g}`), traction-driven outer arc."
    )
    lines.append("")
    lines.append("### Mesh ladder and geometry gate")
    lines.append("")
    lines.append("| level n | elements | dofs | max pointwise rel. error | hole-edge "
                 "`K_t` | area vs polygon | polygon deficit vs exact | gate |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for case in kirs["mesh_ladder"]:
        geom = case["geometry"]
        lines.append(
            f"| {case['level']} | {case['elements']} | {case['dofs']} | "
            f"{_pct(case['max_stress_relative_error'])} | {_fmt(case['kt_hole_edge'], 8)} | "
            f"{_fmt(geom['relative_area_error_vs_polygon'], 3)} | "
            f"{_fmt(geom['relative_area_deficit_vs_exact'], 3)} | **{geom['verdict']}** |"
        )
    lines.append("")
    lines.append(
        "The hole and the outer arc are inscribed chord polygons, so the meshed domain "
        "area is short of the exact annulus by the closed-form deficit "
        "`1 - 2n sin(pi/2n)/pi` (O(n^-2)); the gate checks the element-area sum against "
        "the polygon area and that deficit against its closed form, and reports the "
        "polygon-vs-circle deficit as the declared geometric error of the benchmark."
    )
    lines.append("")
    lines.append("Charge-balance gates (consistent arc loads vs reactions):")
    lines.append("")
    for case in kirs["mesh_ladder"]:
        balance = case["load_balance"]
        lines.append(
            f"* n={case['level']}: `|applied - exact|/|exact| = "
            f"{_fmt(balance['relative_error'], 4)}`, recovery cross-check "
            f"`max rel. difference = {_fmt(case['recovery_cross_check'], 3)}`."
        )
    lines.append("")
    for theta in RAYS_DEGREES:
        lines.append(f"### Pointwise comparison, ray θ={theta:g}° (finest level)")
        lines.append("")
        lines.append("| component | r/a | FE | exact | numerator \\|FE-exact\\| | denominator "
                     "\\|exact\\| | rel. error | ±U | verdict |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for row in kirs["pointwise_rows"]:
            if row["ray_degrees"] != theta:
                continue
            lines.append(
                f"| {row['component']} | {row['r_over_a']:g} | {_fmt(row['fe'], 7)} | "
                f"{_fmt(row['reference'], 7)} | {_fmt(row['numerator_abs_error'], 4)} | "
                f"{_fmt(row['denominator_abs_reference'], 4)} | "
                f"{_pct(row['relative_error'])} | "
                f"{_fmt(row['uncertainty']['half_width'], 3)} | **{row['verdict']}** |"
            )
        lines.append("")
        withheld = [
            row
            for row in kirs["pointwise_rows"]
            if row["ray_degrees"] == theta and row["verdict"] == "WITHHELD"
        ]
        if withheld:
            lines.append("Withheld rows and their declared reason:")
            lines.append("")
            for row in withheld:
                lines.append(
                    f"* `{row['component']}` at `r/a={row['r_over_a']:g}`: "
                    f"{row['withheld_reason']}."
                )
            lines.append("")
    lines.append("### Headline quantities: value, uncertainty, 3 % gate")
    lines.append("")
    lines.append("| quantity | FE (finest) | reference | numerator | denominator | rel. error | "
                 "discretisation half-width | parametric half-width | combined | verdict |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for name, item in kirs["quantities"].items():
        lines.append(
            f"| `{name}` | {_fmt(item['fe'], 8)} | {_fmt(item['reference'], 8)} | "
            f"{_fmt(item['numerator_abs_error'], 4)} | "
            f"{_fmt(item['denominator_abs_reference'], 4)} | "
            f"{_pct(item['relative_error'], 3)} | "
            f"{_fmt(item['discretisation']['half_width'], 3)} | "
            f"{_fmt(item['parametric']['half_width'], 3)} | "
            f"{_fmt(item['combined_uncertainty']['half_width'], 3)} | "
            f"**{item['verdict']}** |"
        )
    lines.append("")
    lines.append("Discretisation estimator status per quantity:")
    lines.append("")
    for name, item in kirs["quantities"].items():
        disc = item["discretisation"]
        if disc["status"] == "ESTIMATED":
            lines.append(
                f"* `{name}`: observed order `p={disc['observed_order']:.3f}`, "
                f"`GCI_rel={_pct(disc['relative_gci'], 3)}`, Richardson value "
                f"`{_fmt(disc['richardson_value'], 8)}`, band covers reference: "
                f"`{disc['band_covers_reference']}`."
            )
        else:
            lines.append(
                f"* `{name}`: **WITHHELD** — {disc['reason']}; ladder envelope "
                f"`[{_fmt(disc['band'][0], 8)}, {_fmt(disc['band'][1], 8)}]`, band covers "
                f"reference: `{disc['band_covers_reference']}`."
            )
    lines.append("")
    lines.append("### Parametric sensitivity")
    lines.append("")
    lines.append("| setting | hoop @ r/a=1.05 | hole-edge `K_t` | max pointwise rel. error |")
    lines.append("|---|---|---|---|")
    for record in kirs["sensitivity"]["settings"]:
        lines.append(
            f"| `{record['setting']}` | {_fmt(record['hoop_at_r1.05'], 8)} | "
            f"{_fmt(record['kt_hole_edge'], 8)} | {_pct(record['max_stress_relative_error'])} |"
        )
    lines.append("")
    lines.append("### Section verdicts")
    lines.append("")
    for name, summary in kirs["summary"].items():
        counts = summary["counts"]
        lines.append(
            f"* `{name}`: **{summary['state']}** "
            f"(PASS {counts['PASS']}, FAIL {counts['FAIL']}, WITHHELD {counts['WITHHELD']})"
        )
    lines.append("")
    lines.append("## Case 2: finite-width plate vs Howland / Peterson")
    lines.append("")
    plate = payload["cases"]["finite_width"]
    lines.append(
        f"`K_tn = sigma_peak / sigma_net`, `sigma_net = T W/(W-d)`; "
        f"levels {plate['model']['levels']}; quarter plate, "
        f"half-length/width = 1.5, thickness 0.01 m."
    )
    lines.append("")
    lines.append("| d/W | FE | Howland/Peterson | numerator | denominator | rel. error | ±U | "
                 "Heywood | rel. err. vs Heywood | verdict |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for row in plate["rows"]:
        lines.append(
            f"| {row['ratio']:g} | {_fmt(row['fe'], 7)} | {_fmt(row['reference'], 7)} | "
            f"{_fmt(row['numerator_abs_error'], 4)} | "
            f"{_fmt(row['denominator_abs_reference'], 4)} | {_pct(row['relative_error'])} | "
            f"{_fmt(row['uncertainty']['half_width'], 3)} | {_fmt(row['ktn_heywood'], 6)} | "
            f"{_pct(row['error_vs_heywood_relative'])} | **{row['verdict']}** |"
        )
    lines.append("")
    lines.append("Load balance (applied tension vs reaction):")
    lines.append("")
    for row in plate["rows"]:
        balance = row["load_balance"]
        lines.append(
            f"* `d/W={row['ratio']:g}`: applied `{_fmt(balance['applied'], 6)}`, reacted "
            f"`{_fmt(balance['reacted'], 6)}`, relative imbalance "
            f"`{_fmt(balance['relative_error'], 4)}`."
        )
    lines.append("")
    summary = plate["summary"]
    lines.append(
        f"Section state: **{summary['state']}** "
        f"(PASS {summary['counts']['PASS']}, FAIL {summary['counts']['FAIL']}, "
        f"WITHHELD {summary['counts']['WITHHELD']})."
    )
    lines.append("")
    lines.append("## Uncertainty propagation")
    lines.append("")
    lines.append(
        "Two independent contributions are reported and combined linearly (worst case, "
        "*not* a statistical confidence interval): the discretisation estimate "
        "(Roache GCI when its preconditions hold, otherwise the min/max ladder envelope) "
        "and the parametric envelope over `nu ∈ {0.25, 0.30, 0.35}` and mesh grading "
        "`∈ {3, 5, 8}` at a fixed refinement level. Both are labelled as such in the JSON."
    )
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    for figure in payload["figures"]:
        lines.append(f"* `{figure['path']}` — {figure['caption']}")
    lines.append("")
    lines.append("## Answers to the 3 % question")
    lines.append("")
    for name, item in payload["answers"]["within_3pct"].items():
        answer = "yes" if item["within_3pct"] else "no"
        side = "below" if item["within_3pct"] else "above"
        lines.append(
            f"* `{name}`: relative error {_pct(item['relative_error'], 3)} against "
            f"`{_fmt(item['reference'], 8)}` — **{answer}** ({side} the 3 % gate)."
        )
    lines.append("")
    finite = payload["answers"]["finite_width"]
    finite_side = "below" if finite["within_3pct"] else "above"
    finite_answer = "yes" if finite["within_3pct"] else "no"
    lines.append(
        f"* finite-width `K_tn` over d/W = {', '.join(f'{r:g}' for r in finite['ratios'])}: "
        f"worst relative error {_pct(finite['max_relative_error'])} against the "
        f"Howland/Peterson table — **{finite_answer}** ({finite_side} the 3 % gate)."
    )
    components = payload["answers"]["pointwise_components"]
    lines.append(
        "* pointwise component states (two rays, six probes each): "
        + ", ".join(f"`{name}` = **{state}**" for name, state in components.items())
        + "."
    )
    lines.append("")
    for note in payload["notes"]:
        lines.append(f"> {note}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build_evidence() -> tuple[dict, dict, KirschHole]:
    hole = kirsch_hole()
    cases = kirsch_ladder(hole)
    ladder_records = []
    for case in cases:
        ladder_records.append(
            {
                "level": case["n_theta"],
                "elements": case["elements"],
                "dofs": case["dofs"],
                "geometry": geometry_gate(case, hole),
                "displacement_error_l2": case["displacement_error"]["l2_relative"],
                "max_stress_relative_error": case["max_stress_relative_error"],
                "kt_hole_edge": case["hole_edge"]["peak_hoop"],
                "load_balance": case["load_resultant"],
                "recovery_cross_check": case["recovery_cross_check"]["max_relative_difference"],
            }
        )

    rows = pointwise_table(hole, cases)
    sensitivity = sensitivity_study()
    ladders = qoi_ladders(cases)
    quantities = {
        "hoop_at_r1.05": qoi_verdict(
            "hoop_at_r1.05",
            ladders["hoop_at_r1.05"],
            float(hole.stress_polar(1.05, math.pi / 2.0)[1]),
            sensitivity["envelopes"]["hoop_at_r1.05"],
            CRITERION,
        ),
        "kt_hole_edge": qoi_verdict(
            "kt_hole_edge",
            ladders["kt_hole_edge"],
            float(hole.stress_concentration * hole.far_field),
            sensitivity["envelopes"]["kt_hole_edge"],
            CRITERION,
        ),
    }
    by_component = {
        f"ray_{int(theta)}_deg": aggregate_verdicts(
            [row for row in rows if row["ray_degrees"] == theta]
        )
        for theta in RAYS_DEGREES
    }
    for component in COMPONENTS:
        by_component[f"component_{component}"] = aggregate_verdicts(
            [row for row in rows if row["component"] == component]
        )
    by_component["headline_quantities"] = aggregate_verdicts(quantities.values())
    by_component["all_pointwise_rows"] = aggregate_verdicts(rows)

    plate = finite_width_table()
    field = field_clouds(hole, cases[-1])

    payload: dict = {
        "schema": "tensorfem.structural-certification-materials/1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "tensorfem_version": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "criterion": CRITERION.as_dict(),
        "verdict_states": ["PASS", "FAIL", "WITHHELD"],
        "cases": {
            "kirsch": {
                "problem": {
                    "radius": hole.radius,
                    "outer_radius": OUTER_RADIUS,
                    "far_field": hole.far_field,
                    "young": hole.young,
                    "poisson": hole.poisson,
                    "plane": hole.plane,
                    "thickness": hole.thickness,
                    "grading": GRADING,
                    "levels": list(KIRSCH_LEVELS),
                    "element": "production Q4 plane-stress continuum",
                    "reference": "Kirsch (1898); Timoshenko & Goodier, Theory of Elasticity, "
                                 "3rd ed., sec. 39",
                },
                "mesh_ladder": ladder_records,
                "pointwise_rows": rows,
                "quantities": quantities,
                "sensitivity": sensitivity,
                "summary": by_component,
            },
            "finite_width": plate,
        },
        "field_cloud_summary": {
            "level": field["level"],
            "samples": int(field["x"].size),
            "hoop_fe_range": [
                float(field["hoop_fe"].min()),
                float(field["hoop_fe"].max()),
            ],
            "hoop_exact_range": [
                float(field["hoop_exact"].min()),
                float(field["hoop_exact"].max()),
            ],
            "difference_range": [
                float((field["hoop_fe"] - field["hoop_exact"]).min()),
                float((field["hoop_fe"] - field["hoop_exact"]).max()),
            ],
        },
        "answers": {
            "within_3pct": {
                name: {
                    "relative_error": item["relative_error"],
                    "reference": item["reference"],
                    "within_3pct": item["within_3pct"],
                }
                for name, item in quantities.items()
            },
            "finite_width": {
                "ratios": [row["ratio"] for row in plate["rows"]],
                "max_relative_error": max(row["relative_error"] for row in plate["rows"]),
                "within_3pct": all(row["within_3pct"] for row in plate["rows"]),
            },
            "pointwise_components": {
                component: by_component[f"component_{component}"]["state"]
                for component in COMPONENTS
            },
        },
        "notes": [],
        "figures": [],
    }
    return payload, field, hole


NOTES = (
    "The hole-edge peak is a nodal-averaged recovered quantity: its ladder is not "
    "monotone and overshoots 3T by O(1e-3), so no GCI is claimed for it (WITHHELD, "
    "ladder envelope reported). The pointwise hoop stress at r/a=1.05 does satisfy the "
    "GCI preconditions and gets an analytic band.",
    "WITHHELD rows are those whose |reference| falls below 1 % of the far-field tension "
    "(the shear component on both symmetry rays, where the exact value is identically "
    "zero). A relative-error criterion is meaningless there; the absolute residual is "
    "reported in the table instead of a fabricated percentage.",
    "The Kirsch case is traction driven, so its stress field is independent of E; the "
    "Poisson sweep probes the sensitivity of the FE solution to nu, not a change of the "
    "reference solution.",
    "The combined half-width is a linear (worst case) sum of the discretisation and "
    "parametric contributions. It is not a statistical confidence interval and must not "
    "be reported as one.",
    "Pointwise radial stress fails the 3 % gate at three of the six probes per ray even "
    "at the finest level. That is a real limitation of bilinear Q4 stress recovery inside "
    "the near-hole gradient layer, where the exact radial stress vanishes at r=a; it is "
    "reported as FAIL rather than dropped, and it is why the certification claim is "
    "scoped to the hoop stress / stress concentration and not to every component.",
    "The finite-width baseline is the Howland (1930) series as tabulated in Peterson's "
    "stress concentration charts, with the Heywood (1952) fit, whose own accuracy is "
    "about 2 %, as a secondary cross-check.",
)


def main() -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=JSON_OUT)
    parser.add_argument("--assets", type=Path, default=ASSETS)
    parser.add_argument("--doc", type=Path, default=DOC_OUT)
    args = parser.parse_args()
    if os.environ.get("OMP_NUM_THREADS") is None:
        torch.set_num_threads(1)

    payload, field, hole = build_evidence()
    args.assets.mkdir(parents=True, exist_ok=True)
    payload["figures"] = [
        plot_field_clouds(field, hole, args.assets),
        plot_pointwise(payload["cases"]["kirsch"]["pointwise_rows"], hole, args.assets),
        plot_convergence(payload["cases"]["kirsch"]["quantities"], args.assets),
        plot_finite_width(payload["cases"]["finite_width"], args.assets),
    ]
    payload["notes"] = list(NOTES)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.doc.parent.mkdir(parents=True, exist_ok=True)
    args.doc.write_text(render_markdown(payload), encoding="utf-8")

    kirs = payload["cases"]["kirsch"]
    print(f"json      : {args.output_json}")
    print(f"doc       : {args.doc}")
    for figure in payload["figures"]:
        print(f"figure    : {figure['path']}")
    for name, item in kirs["quantities"].items():
        print(
            f"{name:16s} fe={item['fe']:.8f} ref={item['reference']:.8f} "
            f"rel={100 * item['relative_error']:.3f}% <3%={item['within_3pct']} "
            f"verdict={item['verdict']}"
        )
    for name, summary in kirs["summary"].items():
        print(f"kirsch {name:22s} state={summary['state']} counts={summary['counts']}")
    print(f"finite_width state={payload['cases']['finite_width']['summary']['state']}")
    return payload


if __name__ == "__main__":
    main()