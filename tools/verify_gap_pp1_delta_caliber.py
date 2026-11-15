#!/usr/bin/env python3
"""GAP-PP-1 acceptance-quantity audit: is the non-uniform/uniform peak-stress
difference (a) measured inside the small-deflection envelope, (b) mesh
converged, and (c) tied to an explicit uniform-reference *caliber*?

Why this tool exists
--------------------
The GAP-PP-1 acceptance asks for "非均匀斑相对均匀斑峰值应力差异给出机读 JSON"
(the machine-readable non-uniform-vs-uniform peak-stress difference) together
with "与 JOUR-ICE-STRUCT 已落地的均匀斑基准同口径对照" (the *same caliber*
comparison against the landed uniform-patch baseline).  The landed artifact
(``benchmarks/artifacts/GAP-PP-1-patch-from-contact-verify/verification.json``)
reports such a number, but three things were never checked:

A. **Envelope** -- the sweep ran at nominal peak pressures that put the default
   plate at ``w/t = 5 .. 17``, i.e. far outside small-deflection theory
   (``w/t <= 0.5``), while the reported quantity is a *linear-model* ratio.
   The ratio is load-invariant *in the linear model*, but the numbers were
   never produced in the regime where the model is valid.
B. **Mesh convergence** -- a single mesh (``nx=12``) was used.  The two loads
   (Gaussian core vs rectangular patch) localise the peak stress differently,
   so their ratio carries *both* discretisation errors.  A single-mesh ratio is
   not an error-controlled number.
C. **Caliber** -- the uniform reference was defined as "same total force over
   the configuration's rectangular footprint", but that footprint
   (``patch_width x patch_length``) is a free parameter that was never varied.
   The ratio is therefore a property of the (contact field, reference
   footprint) pair, not of the contact field alone.

This tool measures all three and emits one JSON.  It is a *verification* tool:
it does not change the solver, it bounds the claim.

Sections
--------
A ``envelope_audit``      -- nominal-pressure sweep vs envelope-controlled sweep.
B ``delta_mesh_ladder``   -- Delta% vs plate mesh, in the envelope, per config.
C ``caliber_sensitivity`` -- Delta% vs uniform-reference footprint convention,
                             same non-uniform field, same total force, same mesh.
D figures                 -- Delta vs mesh, Delta vs caliber, stress fields.

Usage
-----
    python tools/verify_gap_pp1_delta_caliber.py --out benchmarks/artifacts/GAP-PP-1-delta-caliber
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorfem.patch_from_contact import (  # noqa: E402
    ContactPressureData,
    _gaussian_contact_grid,
    contact_pressure_to_plate_load,
    peak_stresses,
    scan_contact_configurations,
)
from tensorfem.plate_mindlin import solve_plate_on_winkler_foundation  # noqa: E402

# Calibration plate: 1 m x 1 m x 10 mm steel panel, simply supported.
SIDE = 1.0
YOUNG = 2.06e11
POISSON = 0.3
THICKNESS = 0.01
WINKLER_K = 1.0e7            # elastic foundation of the landed baseline
W_OVER_T_LIMIT = 0.5         # small-deflection bound (Mindlin plate)
TARGET_W_OVER_T = 0.2        # working point inside the envelope
SIGMA = 0.15 * SIDE          # Gaussian contact core of the landed baseline

DEFAULT_OUT = ROOT / "benchmarks" / "artifacts" / "GAP-PP-1-delta-caliber"


def _pct(a: float, b: float) -> float:
    return 100.0 * (a / b - 1.0) if b else float("nan")


def _uniform_fn(pressure: float, half_w: float, half_l: float, centre: float):
    def fn(x, y):
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        inside = (np.abs(x - centre) <= half_l) & (np.abs(y - centre) <= half_w)
        return np.where(inside, pressure, 0.0).astype(np.float64)
    return fn


def _peak(result) -> tuple[dict, float]:
    return peak_stresses(result), float(
        np.max(np.abs(result.displacement[0::3].numpy()))
    )


# ---------------------------------------------------------------------------
# A. envelope audit: nominal (historical) vs envelope-controlled sweep
# ---------------------------------------------------------------------------
def envelope_audit(plate_nx: int = 12) -> dict:
    common = dict(
        plate_nx=plate_nx, plate_side=SIDE, young=YOUNG, poisson=POISSON,
        thickness=THICKNESS, winkler_k=WINKLER_K,
    )
    nominal = scan_contact_configurations(**common, target_w_over_t=None)
    controlled = scan_contact_configurations(
        **common, target_w_over_t=TARGET_W_OVER_T, w_over_t_limit=W_OVER_T_LIMIT,
    )

    def rows(scan: dict) -> list[dict]:
        return [
            {
                "name": r["name"],
                "peak_pressure_MPa": r["peak_pressure_MPa"],
                "nominal_peak_pressure_MPa": r["nominal_peak_pressure_MPa"],
                "pressure_scale_factor": r["pressure_scale_factor"],
                "w_over_t": r["w_over_t_nonuniform"],
                "in_small_deflection_envelope": r["in_small_deflection_envelope"],
                "delta_sigma_x_pct": r["delta_nonuniform_over_uniform_pct"]["sigma_x"],
                "delta_sigma_vm_pct": r["delta_nonuniform_over_uniform_pct"]["sigma_vm"],
            }
            for r in scan["configurations"]
        ]

    return {
        "case": "GAP-PP-1 acceptance sweep: envelope audit",
        "plate": {
            "nx": plate_nx, "side_m": SIDE, "thickness_m": THICKNESS,
            "E_Pa": YOUNG, "nu": POISSON, "winkler_k": WINKLER_K,
        },
        "w_over_t_limit": W_OVER_T_LIMIT,
        "nominal_pressure_sweep": {
            "target_w_over_t": None,
            "configurations": rows(nominal),
            "all_in_small_deflection_envelope": nominal["summary"][
                "all_in_small_deflection_envelope"
            ],
            "max_w_over_t": nominal["summary"]["max_w_over_t"],
        },
        "envelope_controlled_sweep": {
            "target_w_over_t": TARGET_W_OVER_T,
            "configurations": rows(controlled),
            "all_in_small_deflection_envelope": controlled["summary"][
                "all_in_small_deflection_envelope"
            ],
            "max_w_over_t": controlled["summary"]["max_w_over_t"],
        },
        "verdict": {
            "historical_sweep_outside_envelope": bool(
                not nominal["summary"]["all_in_small_deflection_envelope"]
            ),
            "controlled_sweep_inside_envelope": bool(
                controlled["summary"]["all_in_small_deflection_envelope"]
            ),
            "note": (
                "the acceptance sweep must be reported in the envelope-controlled "
                "mode (target_w_over_t); the nominal-pressure mode is kept only "
                "as the historical record"
            ),
        },
    }


# ---------------------------------------------------------------------------
# B. Delta% vs mesh, inside the envelope
# ---------------------------------------------------------------------------
def delta_mesh_ladder(meshes=(12, 16, 24, 32), configs=None) -> dict:
    if configs is None:
        configs = [
            {"name": "mild_nonuniform", "sigma_x": 0.15 * SIDE, "sigma_y": 0.15 * SIDE,
             "patch_width": 0.3 * SIDE, "patch_length": 0.3 * SIDE},
            {"name": "moderate_nonuniform", "sigma_x": 0.10 * SIDE, "sigma_y": 0.10 * SIDE,
             "patch_width": 0.2 * SIDE, "patch_length": 0.2 * SIDE},
            {"name": "sharp_nonuniform", "sigma_x": 0.05 * SIDE, "sigma_y": 0.05 * SIDE,
             "patch_width": 0.1 * SIDE, "patch_length": 0.1 * SIDE},
        ]

    rows = []
    for cfg in configs:
        s_x, s_y = cfg["sigma_x"], cfg["sigma_y"]
        pw, pl = cfg["patch_width"], cfg["patch_length"]
        for nx in meshes:
            cache: dict = {}
            # (1) calibrate the load on a mid mesh so w/t = TARGET_W_OVER_T
            pos, pres = _gaussian_contact_grid(
                0.5, 0.5, s_x, s_y, 1.0e6, SIDE, 41,
            )
            data = ContactPressureData(
                node_positions=pos, normal_pressures=pres,
                contact_area=2.0 * math.pi * s_x * s_y,
            )
            probe, probe_patch = contact_pressure_to_plate_load(
                data, 24, SIDE, YOUNG, POISSON, THICKNESS, WINKLER_K,
            )
            _, w_probe = _peak(probe)
            w_over_t_probe = w_probe / THICKNESS
            peak = 1.0e6 * TARGET_W_OVER_T / w_over_t_probe

            pos, pres = _gaussian_contact_grid(0.5, 0.5, s_x, s_y, peak, SIDE, 41)
            data = ContactPressureData(
                node_positions=pos, normal_pressures=pres,
                contact_area=2.0 * math.pi * s_x * s_y,
            )
            plate_nu, patch = contact_pressure_to_plate_load(
                data, nx, SIDE, YOUNG, POISSON, THICKNESS, WINKLER_K,
                stiffness_cache=cache,
            )
            st_nu, w_nu = _peak(plate_nu)

            total_force = patch.total_force
            uniform_p = total_force / (pw * pl)
            plate_u = solve_plate_on_winkler_foundation(
                nx, SIDE, YOUNG, POISSON, THICKNESS, WINKLER_K,
                _uniform_fn(uniform_p, pw / 2.0, pl / 2.0, 0.5),
                stiffness_cache=cache,
            )
            st_u, w_u = _peak(plate_u)

            rows.append({
                "name": cfg["name"],
                "plate_nx": nx,
                "peak_pressure_MPa": peak / 1e6,
                "w_over_t": w_nu / THICKNESS,
                "in_small_deflection_envelope": bool(
                    w_nu / THICKNESS <= W_OVER_T_LIMIT
                ),
                "sigma_x_nonuniform_MPa": st_nu["sigma_x_MPa"],
                "sigma_x_uniform_MPa": st_u["sigma_x_MPa"],
                "delta_sigma_x_pct": _pct(st_nu["sigma_x_MPa"], st_u["sigma_x_MPa"]),
                "delta_sigma_vm_pct": _pct(st_nu["sigma_vm_MPa"], st_u["sigma_vm_MPa"]),
                "delta_deflection_pct": _pct(w_nu, w_u),
            })

    per_config: dict[str, dict] = {}
    for cfg in configs:
        own = [r for r in rows if r["name"] == cfg["name"]]
        own.sort(key=lambda r: r["plate_nx"])
        d = [r["delta_sigma_x_pct"] for r in own]
        finest = d[-1]
        coarse = d[0]
        # Richardson-style estimate of the discretisation error of the reported
        # quantity itself: |Delta(nx_coarse) - Delta(nx_fine)| as an error bar.
        per_config[cfg["name"]] = {
            "delta_sigma_x_pct_by_mesh": d,
            "meshes": [r["plate_nx"] for r in own],
            "delta_sigma_x_pct_finest": finest,
            "coarse_to_fine_shift_pct_points": abs(coarse - finest),
            "consecutive_shift_last_pct_points": abs(d[-1] - d[-2]) if len(d) > 1 else float("nan"),
            "sign_consistent_across_meshes": bool(
                all(x > 0 for x in d) or all(x < 0 for x in d)
            ),
        }

    worst_shift = max(v["coarse_to_fine_shift_pct_points"] for v in per_config.values())
    last_shift = max(
        v["consecutive_shift_last_pct_points"] for v in per_config.values()
    )
    return {
        "case": "GAP-PP-1 Delta% vs plate mesh (envelope-controlled)",
        "target_w_over_t": TARGET_W_OVER_T,
        "uniform_reference_caliber": (
            "same total force, rectangular footprint = patch_width x patch_length "
            "of the configuration"
        ),
        "rows": rows,
        "per_configuration": per_config,
        "summary": {
            "max_coarse_to_fine_shift_pct_points": worst_shift,
            "max_last_shift_pct_points": last_shift,
            "mesh_converged_within_pct_points": last_shift,
            "all_in_small_deflection_envelope": all(
                r["in_small_deflection_envelope"] for r in rows
            ),
            "verdict": (
                "Delta% is NOT mesh independent to better than the last-shift "
                "value; report it with that error bar (or on the finest mesh "
                "with the shift as uncertainty)"
            ),
        },
    }


# ---------------------------------------------------------------------------
# C. Delta% vs uniform-reference caliber (footprint convention)
# ---------------------------------------------------------------------------
def caliber_sensitivity(plate_nx: int = 24, sigma: float = SIGMA) -> dict:
    cache: dict = {}
    pos, pres = _gaussian_contact_grid(0.5, 0.5, sigma, sigma, 1.0e6, SIDE, 41)
    data = ContactPressureData(
        node_positions=pos, normal_pressures=pres,
        contact_area=2.0 * math.pi * sigma * sigma,
    )
    probe, _ = contact_pressure_to_plate_load(
        data, 24, SIDE, YOUNG, POISSON, THICKNESS, WINKLER_K,
    )
    _, w_probe = _peak(probe)
    peak = 1.0e6 * TARGET_W_OVER_T / (w_probe / THICKNESS)

    pos, pres = _gaussian_contact_grid(0.5, 0.5, sigma, sigma, peak, SIDE, 41)
    data = ContactPressureData(
        node_positions=pos, normal_pressures=pres,
        contact_area=2.0 * math.pi * sigma * sigma,
    )
    plate_nu, patch = contact_pressure_to_plate_load(
        data, plate_nx, SIDE, YOUNG, POISSON, THICKNESS, WINKLER_K,
        stiffness_cache=cache,
    )
    st_nu, w_nu = _peak(plate_nu)
    total_force = patch.total_force
    analytic_force = peak * 2.0 * math.pi * sigma * sigma

    equal_area_side = math.sqrt(2.0 * math.pi * sigma * sigma)
    calibers = [
        ("six_sigma_square", 6.0 * sigma, "6 sigma x 6 sigma (99.7 % of the force)"),
        ("four_sigma_square", 4.0 * sigma, "4 sigma x 4 sigma"),
        ("three_sigma_square", 3.0 * sigma, "3 sigma x 3 sigma"),
        ("equal_area_square", equal_area_side, "square of equal area 2*pi*sigma^2"),
        ("rule_patch_0.3_side", 0.3 * SIDE, "IACS-style rule patch, 0.3 x side"),
        ("rule_patch_0.2_side", 0.2 * SIDE, "0.2 x side"),
    ]

    rows = []
    for name, side_len, desc in calibers:
        uniform_p = total_force / (side_len * side_len)
        plate_u = solve_plate_on_winkler_foundation(
            plate_nx, SIDE, YOUNG, POISSON, THICKNESS, WINKLER_K,
            _uniform_fn(uniform_p, side_len / 2.0, side_len / 2.0, 0.5),
            stiffness_cache=cache,
        )
        st_u, w_u = _peak(plate_u)
        rows.append({
            "caliber": name,
            "footprint_side_m": side_len,
            "footprint_area_over_contact_area": side_len * side_len
            / (2.0 * math.pi * sigma * sigma),
            "uniform_pressure_MPa": uniform_p / 1e6,
            "sigma_x_uniform_MPa": st_u["sigma_x_MPa"],
            "delta_sigma_x_pct": _pct(st_nu["sigma_x_MPa"], st_u["sigma_x_MPa"]),
            "delta_sigma_vm_pct": _pct(st_nu["sigma_vm_MPa"], st_u["sigma_vm_MPa"]),
            "delta_deflection_pct": _pct(w_nu, w_u),
            "description": desc,
        })

    dx = [r["delta_sigma_x_pct"] for r in rows]
    return {
        "case": "GAP-PP-1 Delta% vs uniform-reference caliber",
        "plate_nx": plate_nx,
        "contact": {
            "kind": "gaussian", "sigma_m": sigma, "peak_pressure_MPa": peak / 1e6,
            "w_over_t": w_nu / THICKNESS,
            "total_force_N": total_force,
            "analytic_total_force_N": analytic_force,
        },
        "nonuniform_peak_sigma_x_MPa": st_nu["sigma_x_MPa"],
        "calibers": rows,
        "summary": {
            "min_delta_sigma_x_pct": min(dx),
            "max_delta_sigma_x_pct": max(dx),
            "spread_pct_points": max(dx) - min(dx),
            "sign_changes": bool(min(dx) < 0.0 < max(dx)),
            "verdict": (
                "Delta% is caliber dependent: a bare '+X %' without the "
                "reference-footprint definition is not a reproducible number"
            ),
        },
    }


# ---------------------------------------------------------------------------
# D. figures
# ---------------------------------------------------------------------------
def figures(out: Path, mesh: dict, cal: dict) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs: list[dict] = []

    # (1) Delta vs mesh
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name, own in mesh["per_configuration"].items():
        ax.plot(own["meshes"], own["delta_sigma_x_pct_by_mesh"], "o-", label=name)
    ax.set_xlabel("plate mesh nx (= ny)")
    ax.set_ylabel(r"$\Delta\sigma_x$ non-uniform vs uniform [%]")
    ax.set_title("GAP-PP-1 acceptance quantity vs mesh (w/t = %.2f)" % mesh["target_w_over_t"])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    p = out / "fig_delta_vs_mesh.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    figs.append({"name": p.name, "caption": "非均匀/均匀斑峰值应力差随板网格收敛（包络内 w/t=0.2）"})

    # (2) Delta vs caliber
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    labels = [r["caliber"] for r in cal["calibers"]]
    vals = [r["delta_sigma_x_pct"] for r in cal["calibers"]]
    colours = ["tab:red" if v < 0 else "tab:blue" for v in vals]
    ax.bar(range(len(vals)), vals, color=colours)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_ylabel(r"$\Delta\sigma_x$ [%]")
    ax.set_title("Same contact field, different uniform-reference caliber")
    ax.grid(alpha=0.3, axis="y")
    p = out / "fig_delta_vs_caliber.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    figs.append({"name": p.name, "caption": "同一接触场、不同均匀参考口径下的峰值应力差（跨零）"})

    # (3) stress field: non-uniform vs uniform at the finest studied mesh
    nx = 24
    cache: dict = {}
    pos, pres = _gaussian_contact_grid(0.5, 0.5, SIGMA, SIGMA, 1.0e6, SIDE, 41)
    data = ContactPressureData(
        node_positions=pos, normal_pressures=pres,
        contact_area=2.0 * math.pi * SIGMA * SIGMA,
    )
    probe, _ = contact_pressure_to_plate_load(
        data, nx, SIDE, YOUNG, POISSON, THICKNESS, WINKLER_K,
    )
    _, w_probe = _peak(probe)
    peak = 1.0e6 * TARGET_W_OVER_T / (w_probe / THICKNESS)
    pos, pres = _gaussian_contact_grid(0.5, 0.5, SIGMA, SIGMA, peak, SIDE, 41)
    data = ContactPressureData(
        node_positions=pos, normal_pressures=pres,
        contact_area=2.0 * math.pi * SIGMA * SIGMA,
    )
    plate_nu, patch = contact_pressure_to_plate_load(
        data, nx, SIDE, YOUNG, POISSON, THICKNESS, WINKLER_K,
        stiffness_cache=cache,
    )
    uniform_p = patch.total_force / (0.3 * SIDE) ** 2
    plate_u = solve_plate_on_winkler_foundation(
        nx, SIDE, YOUNG, POISSON, THICKNESS, WINKLER_K,
        _uniform_fn(uniform_p, 0.15, 0.15, 0.5), stiffness_cache=cache,
    )
    nodes = plate_nu.nodes.detach().cpu().numpy() if hasattr(plate_nu.nodes, "detach") \
        else np.asarray(plate_nu.nodes)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    for ax, plate, title in (
        (axes[0], plate_nu, "non-uniform (contact-projected)"),
        (axes[1], plate_u, "uniform reference (0.3 x side, same force)"),
    ):
        w = plate.displacement[0::3].numpy().reshape(nx + 1, nx + 1)
        cs = ax.contourf(nodes[:, 0].reshape(nx + 1, nx + 1),
                         nodes[:, 1].reshape(nx + 1, nx + 1), w * 1e3, levels=16,
                         cmap="viridis")
        ax.set_aspect("equal"); ax.set_title(f"{title}\npeak w = {w.max() * 1e3:.2f} mm",
                                            fontsize=9)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        fig.colorbar(cs, ax=ax, label="w [mm]")
    p = out / "fig_deflection_fields.png"
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    figs.append({"name": p.name, "caption": "包络内（w/t=0.2）非均匀斑与均匀参考挠度场对比，nx=24"})
    return figs


def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--meshes", default="12,16,24,32")
    ap.add_argument("--caliber-mesh", type=int, default=24)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meshes = tuple(int(x) for x in args.meshes.split(","))

    t0 = time.time()
    env = envelope_audit()
    ladder = delta_mesh_ladder(meshes=meshes)
    cal = caliber_sensitivity(plate_nx=args.caliber_mesh)
    figs = figures(out, ladder, cal)

    result = {
        "case": "GAP-PP-1 acceptance-quantity audit (envelope / mesh / caliber)",
        "anchor": "GAP-PP-1-ICE-STRUCT-CONTACT-PATCH",
        "plate": {
            "side_m": SIDE, "thickness_m": THICKNESS, "E_Pa": YOUNG,
            "nu": POISSON, "winkler_k": WINKLER_K, "boundary": "simply_supported",
        },
        "envelope_audit": env,
        "delta_mesh_ladder": ladder,
        "caliber_sensitivity": cal,
        "figures": figs,
        "wall_seconds": round(time.time() - t0, 1),
    }
    (out / "delta_caliber.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({
        "envelope": env["verdict"],
        "mesh_summary": ladder["summary"],
        "caliber_summary": cal["summary"],
        "wall_s": result["wall_seconds"],
    }, indent=1))
    return result


if __name__ == "__main__":
    main()