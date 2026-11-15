#!/usr/bin/env python3
"""End-to-end demo: CZM contact pressure → plate load projection → plate stress.

GAP-PP-1-ICE-STRUCT-CONTACT-PATCH 验收算例。

演示完整的冰-结构接触→斑载荷→板格应力链：
1. 构造 CZM 接触压力数据（高斯分布模拟冰缘接触）
2. 投影到板格节点载荷向量（精确保守投影）
3. 求解 Mindlin 板得到应力场
4. 与均匀斑基准比较，输出机读 JSON

Usage
-----
    python examples/ice_struct_contact/end_to_end_demo.py --out outputs/ice_struct_demo
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensorfem.patch_from_contact import (
    ContactPressureData,
    PatchLoadResult,
    contact_pressure_to_plate_load,
    gaussian_patch_pressure,
    peak_stresses,
    scan_contact_configurations,
)


ARCHIVE = (
    Path(__file__).resolve().parents[2]
    / "benchmarks" / "artifacts" / "GAP-PP-1-patch-from-contact-verify" / "verification.json"
)
W_OVER_T_LIMIT = 0.5   # small-deflection validity bound for a Mindlin plate


def h3_anchor_check() -> dict:
    """Derive the H3-ICE-PLATE-FEM consistency check from the archived cross-check.

    The previous version of this demo carried the numbers as literals
    (``gaussian_patch_fem_vs_navier_pct = -0.41`` and
    ``same_order_of_magnitude = True``), i.e. the acceptance "check" was a
    constant that could never fail.  It is now read from the archived
    verification artifact and the boolean is computed from it; the single-mesh
    nature of the archived number is recorded explicitly (see
    ``tools/verify_gap_pp1_adversarial.py`` section A for the mesh ladder).
    """
    out = {
        "h3_anchor_level_pct": None,
        "h3_anchor_meaning": "FEM-vs-Kelvin error of the H3-ICE-PLATE-FEM anchor",
        "gaussian_patch_fem_vs_navier_pct": None,
        "same_order_of_magnitude": None,
        "source": str(ARCHIVE.relative_to(ARCHIVE.parents[3])) if ARCHIVE.exists() else None,
        "single_mesh_caveat": (
            "archived value is one mesh (nx=16) taken from the archived artifact; the "
            "mesh-converged error is reported by tools/verify_gap_pp1_adversarial.py"
        ),
    }
    if not ARCHIVE.exists():  # pragma: no cover - defensive
        out["note"] = "archive missing: consistency check NOT performed"
        return out
    data = json.loads(ARCHIVE.read_text())
    cross = data["navier_crosscheck"]
    anchor = float(cross["h3_anchor_pct"])
    row = next(r for r in cross["rows"] if r["patch"].startswith("gaussian"))
    value = float(row["mine_vs_navier_pct"])
    out.update({
        "h3_anchor_level_pct": anchor,
        "gaussian_patch_fem_vs_navier_pct": value,
        "plate_nx": row["plate_nx"],
        # "same order of magnitude" = within a factor of 3 of the anchor
        "same_order_of_magnitude": bool(abs(value) <= 3.0 * anchor),
    })
    return out


def build_gaussian_contact(
    centre_x: float, centre_y: float,
    sigma_x: float, sigma_y: float,
    peak_pressure: float,
    plate_side: float,
    n_per_axis: int,
) -> ContactPressureData:
    """Construct a Gaussian contact pressure field on a tensor-product grid."""
    lo_x = max(0.0, centre_x - 3.0 * sigma_x)
    hi_x = min(plate_side, centre_x + 3.0 * sigma_x)
    lo_y = max(0.0, centre_y - 3.0 * sigma_y)
    hi_y = min(plate_side, centre_y + 3.0 * sigma_y)
    xs = np.linspace(lo_x, hi_x, n_per_axis)
    ys = np.linspace(lo_y, hi_y, n_per_axis)
    xx, yy = np.meshgrid(xs, ys)
    pos = np.column_stack([xx.ravel(), yy.ravel()])
    pres = gaussian_patch_pressure(
        pos[:, 0], pos[:, 1], centre_x, centre_y, peak_pressure, sigma_x, sigma_y,
    )
    return ContactPressureData(
        node_positions=pos,
        normal_pressures=pres,
        contact_area=2.0 * math.pi * sigma_x * sigma_y,
    )


# ---------------------------------------------------------------------------
# Small-deflection-valid end-to-end case (m22084 increment)
# ---------------------------------------------------------------------------
# The original demo load (5 MPa peak) puts the plate at w/t ~ 14, i.e. far
# outside small-deflection theory: the absolute peak stresses are then
# meaningless even though the non-uniform/uniform *ratio* is load-invariant in
# the linear model.  This case keeps the same geometry and contact shape but
# scales the peak pressure so that w/t < 0.5, and benchmarks the FEM chain
# against the independent Kirchhoff Navier series (free SSSS plate, k = 0) --
# the same caliber as the H3-ICE-PLATE-FEM anchor (0.339 %).
VALID_PEAK_PRESSURE_PA = 0.05e6   # -> w/t ~= 0.32 on the free plate (nx=24)
VALID_W_OVER_T_LIMIT = 0.5
H3_ANCHOR_PCT = 0.339             # FEM-vs-Kelvin error of the H3-ICE-PLATE-FEM anchor


def _gaussian_contact_patch_ic(pos, pres, peak, sigma_x, sigma_y):
    """Build the independent Navier-side ContactPatch from a contact grid."""
    import torch
    import tensorfem.ice_panel_patch_contact as IC

    xg = np.unique(np.round(pos[:, 0], 12))
    yg = np.unique(np.round(pos[:, 1], 12))
    grid_p = pres.reshape(yg.size, xg.size).T  # (len(x), len(y))
    return IC.ContactPatch(
        x=torch.as_tensor(xg, dtype=torch.float64),
        y=torch.as_tensor(yg, dtype=torch.float64),
        pressure=torch.as_tensor(grid_p, dtype=torch.float64),
        total_force=float(peak * 2.0 * math.pi * sigma_x * sigma_y),
        contact_area=float(abs(xg[-1] - xg[0]) * abs(yg[-1] - yg[0])),
        peak_pressure=float(peak),
        centroid=(0.5 * (xg[0] + xg[-1]), 0.5 * (yg[0] + yg[-1])),
        approach=0.0,
        opening=torch.zeros((len(xg), len(yg)), dtype=torch.float64),
        residual=0.0,
        config={"kind": "valid_case"},
    )


def run_valid_case(
    plate_nx: int = 24,
    plate_side: float = 1.0,
    meshes=(16, 24, 32),
    contact_grid: int = 33,
) -> dict:
    """Small-deflection-valid end-to-end case with a live Navier benchmark.

    Chain: Gaussian CZM-style contact pressure -> exact sub-cell projection ->
    Mindlin Q4 SRI plate (free, simply supported) -> peak face stress,
    compared with (a) a uniform patch of equal total force and footprint and
    (b) the independent Kirchhoff Navier series for both patches.
    """
    import tensorfem.ice_panel_patch_contact as IC
    from tensorfem.plate_mindlin import solve_plate_on_winkler_foundation

    E, nu, t, k = 2.06e11, 0.3, 0.01, 0.0
    cx, cy, sx, sy = 0.5, 0.5, 0.15, 0.15
    peak_p = VALID_PEAK_PRESSURE_PA

    lo_x, hi_x = max(0.0, cx - 3.0 * sx), min(plate_side, cx + 3.0 * sx)
    lo_y, hi_y = max(0.0, cy - 3.0 * sy), min(plate_side, cy + 3.0 * sy)
    xs = np.linspace(lo_x, hi_x, contact_grid)
    ys = np.linspace(lo_y, hi_y, contact_grid)
    xx, yy = np.meshgrid(xs, ys)
    pos = np.column_stack([xx.ravel(), yy.ravel()])
    pres = gaussian_patch_pressure(pos[:, 0], pos[:, 1], cx, cy, peak_p, sx, sy)
    contact_data = ContactPressureData(
        node_positions=pos,
        normal_pressures=pres,
        contact_area=2.0 * math.pi * sx * sy,
    )
    analytic_force = peak_p * 2.0 * math.pi * sx * sy

    # (1) projected non-uniform patch
    plate_proj, patch_proj = contact_pressure_to_plate_load(
        contact_data, plate_nx, plate_side, E, nu, t, k,
    )
    stress_proj = peak_stresses(plate_proj)
    w_proj = float(np.max(np.abs(plate_proj.displacement[0::3].numpy())))

    # (2) uniform reference: same total force, same 6-sigma footprint
    patch_w, patch_l = 6.0 * sx, 6.0 * sy
    uniform_p = patch_proj.total_force / (patch_w * patch_l)

    def uniform_fn(x, y):
        in_patch = (
            (x >= cx - patch_w / 2) & (x <= cx + patch_w / 2) &
            (y >= cy - patch_l / 2) & (y <= cy + patch_l / 2)
        )
        return np.where(in_patch, uniform_p, 0.0)

    plate_uni = solve_plate_on_winkler_foundation(
        plate_nx, plate_side, E, nu, t, k, uniform_fn,
    )
    stress_uni = peak_stresses(plate_uni)
    w_uni = float(np.max(np.abs(plate_uni.displacement[0::3].numpy())))

    # (3) Navier-series reference (independent analytic path, k = 0)
    gauss_patch = _gaussian_contact_patch_ic(pos, pres, peak_p, sx, sy)
    uni_patch = IC.uniform_patch_pressure(
        total_force=patch_proj.total_force, centre=(cx, cy),
        half_x=patch_w / 2.0, half_y=patch_l / 2.0,
        panel=(plate_side, plate_side), grid=contact_grid,
    )

    def navier_sigma(patch, x, y):
        mom = IC.navier_rect_moments(
            patch, plate_side, plate_side, young=E, poisson=nu, thickness=t,
            x=x, y=y,
        )
        return 6.0 * abs(mom["moment_x"]) / t**2 / 1e6

    nav_proj = navier_sigma(gauss_patch, *stress_proj["sigma_x_at_m"])
    nav_uni = navier_sigma(uni_patch, *stress_uni["sigma_x_at_m"])

    def pct(a, b):
        return 100.0 * (a / b - 1.0) if b else float("nan")

    # (4) mesh ladder for the convergence curve
    ladder = []
    for nx in meshes:
        sol, pr = contact_pressure_to_plate_load(
            contact_data, nx, plate_side, E, nu, t, k,
        )
        st = peak_stresses(sol)
        nav = navier_sigma(gauss_patch, *st["sigma_x_at_m"])
        ladder.append({
            "plate_nx": nx,
            "sigma_x_MPa": st["sigma_x_MPa"],
            "navier_sigma_x_MPa": nav,
            "fem_vs_navier_pct": pct(st["sigma_x_MPa"], nav),
        })

    finest = ladder[-1]
    err_finest = abs(finest["fem_vs_navier_pct"])
    return {
        "case": "GAP-PP-1 valid end-to-end case (small-deflection regime)",
        "plate": {
            "nx": plate_nx, "side_m": plate_side, "thickness_m": t,
            "E_Pa": E, "nu": nu, "winkler_k": k, "boundary": "simply_supported",
        },
        "contact": {
            "kind": "gaussian", "centre_m": [cx, cy], "sigma_m": [sx, sy],
            "peak_pressure_MPa": peak_p / 1e6,
            "contact_area_m2": contact_data.contact_area,
            "n_contact_nodes": len(pres),
            "analytic_total_force_N": analytic_force,
        },
        "projected_patch": {
            "total_force_N": patch_proj.total_force,
            "force_error_rel": (patch_proj.total_force - analytic_force) / analytic_force,
            "load_balance_rel": patch_proj.metadata.get("load_balance_rel"),
            "projection_method": patch_proj.metadata.get("projection_method"),
        },
        "peak_stress_nonuniform": stress_proj,
        "peak_stress_uniform": stress_uni,
        "peak_deflection_nonuniform_m": w_proj,
        "peak_deflection_uniform_m": w_uni,
        "delta_nonuniform_over_uniform_pct": {
            "sigma_vm": pct(stress_proj["sigma_vm_MPa"], stress_uni["sigma_vm_MPa"]),
            "sigma_x": pct(stress_proj["sigma_x_MPa"], stress_uni["sigma_x_MPa"]),
            "deflection": pct(w_proj, w_uni),
            "note": (
                "machine-readable non-uniform-vs-uniform peak stress difference "
                "required by the GAP-PP-1 acceptance"
            ),
        },
        "navier_benchmark": {
            "nonuniform_fem_vs_navier_pct": pct(stress_proj["sigma_x_MPa"], nav_proj),
            "uniform_fem_vs_navier_pct": pct(stress_uni["sigma_x_MPa"], nav_uni),
            "navier_sigma_x_nonuniform_MPa": nav_proj,
            "navier_sigma_x_uniform_MPa": nav_uni,
            "mesh_ladder": ladder,
        },
        "small_deflection_validity": {
            "peak_deflection_over_thickness": w_proj / t,
            "limit": VALID_W_OVER_T_LIMIT,
            "valid": bool(w_proj / t <= VALID_W_OVER_T_LIMIT),
        },
        "h3_anchor_check": {
            "h3_anchor_level_pct": H3_ANCHOR_PCT,
            "fem_vs_navier_pct_finest": finest["fem_vs_navier_pct"],
            "same_order_of_magnitude": bool(err_finest <= 3.0 * H3_ANCHOR_PCT),
            "basis": (
                "live FEM-vs-Navier peak bending stress error at the finest mesh "
                "of the ladder; 'same order of magnitude' = within a factor of 3 "
                "of the H3-ICE-PLATE-FEM anchor (0.339 %)"
            ),
        },
    }


def run_demo(plate_nx: int = 12, plate_side: float = 1.0, out: Path | None = None):
    """Run the end-to-end demo and return results dict."""
    # Material / geometry
    E = 2.06e11      # Pa
    nu = 0.3
    t = 0.01         # m
    k = 1.0e7        # N/m³ (Winkler foundation)

    # Contact configuration: mild non-uniformity
    cx, cy = 0.5, 0.5
    sx, sy = 0.15, 0.15
    peak_p = 5.0e6   # Pa

    contact_data = build_gaussian_contact(cx, cy, sx, sy, peak_p, plate_side, n_per_axis=21)

    # End-to-end solve
    plate_result, patch_result = contact_pressure_to_plate_load(
        contact_data, plate_nx, plate_side, E, nu, t, k,
    )

    # Peak stresses
    stress = peak_stresses(plate_result)

    # Uniform reference (same total force, same footprint)
    patch_width = 6.0 * sx
    patch_length = 6.0 * sy
    uniform_p = patch_result.total_force / (patch_width * patch_length)

    def uniform_fn(x, y):
        in_patch = (
            (x >= cx - patch_width / 2) & (x <= cx + patch_width / 2) &
            (y >= cy - patch_length / 2) & (y <= cy + patch_length / 2)
        )
        return np.where(in_patch, uniform_p, 0.0)

    from tensorfem.plate_mindlin import solve_plate_on_winkler_foundation
    plate_uni = solve_plate_on_winkler_foundation(
        plate_nx, plate_side, E, nu, t, k, uniform_fn,
    )
    stress_uni = peak_stresses(plate_uni)

    # Deflections
    w_proj = float(np.max(np.abs(plate_result.displacement[0::3].numpy())))
    w_uni = float(np.max(np.abs(plate_uni.displacement[0::3].numpy())))

    def pct(a, b):
        return 100.0 * (a / b - 1.0) if b else float("nan")

    result = {
        "case": "GAP-PP-1-ICE-STRUCT-CONTACT-PATCH end-to-end demo",
        "plate": {
            "nx": plate_nx,
            "side_m": plate_side,
            "thickness_m": t,
            "E_Pa": E,
            "nu": nu,
            "winkler_k": k,
            "boundary": "simply_supported",
        },
        "contact": {
            "kind": "gaussian",
            "centre_m": [cx, cy],
            "sigma_m": [sx, sy],
            "peak_pressure_MPa": peak_p / 1e6,
            "contact_area_m2": contact_data.contact_area,
            "n_contact_nodes": len(contact_data.normal_pressures),
        },
        "projected_patch": {
            "total_force_N": patch_result.total_force,
            "peak_pressure_MPa": patch_result.peak_pressure / 1e6,
            "mean_pressure_MPa": patch_result.mean_pressure / 1e6,
            "pressure_nonuniformity": patch_result.pressure_nonuniformity,
            "load_balance_rel": patch_result.metadata.get("load_balance_rel", None),
            "projection_method": patch_result.metadata.get("projection_method", None),
        },
        "peak_stress_nonuniform": stress,
        "peak_stress_uniform": stress_uni,
        "peak_deflection_nonuniform_m": w_proj,
        "peak_deflection_uniform_m": w_uni,
        "delta_nonuniform_over_uniform_pct": {
            "sigma_vm": pct(stress["sigma_vm_MPa"], stress_uni["sigma_vm_MPa"]),
            "sigma_x": pct(stress["sigma_x_MPa"], stress_uni["sigma_x_MPa"]),
            "deflection": pct(w_proj, w_uni),
        },
        "h3_anchor_check": h3_anchor_check(),
        "small_deflection_validity": {
            "peak_deflection_over_thickness": w_proj / t,
            "limit": W_OVER_T_LIMIT,
            "valid": bool(w_proj / t <= W_OVER_T_LIMIT),
            "note": (
                "the demo load puts the plate at w/t >> 1: the absolute peak stresses are "
                "outside small-deflection theory, while the non-uniform/uniform ratio is a "
                "load-invariant property of the linear model (exact in linear theory)"
            ),
        },
    }

    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "end_to_end_demo.json").write_text(json.dumps(result, indent=2))

    return result


def main():
    parser = argparse.ArgumentParser(description="End-to-end ice-structure contact demo")
    parser.add_argument(
        "--out", type=str,
        default=str(Path(__file__).resolve().parents[2] / "benchmarks" / "artifacts"
                    / "GAP-PP-1-end-to-end-demo"),
        help="Output directory (committed artifact path; 'outputs/' is gitignored)",
    )
    parser.add_argument("--nx", type=int, default=12, help="Plate mesh resolution")
    args = parser.parse_args()

    out = Path(args.out) if args.out else None
    result = run_demo(plate_nx=args.nx, out=None)
    valid = run_valid_case()
    result["valid_case"] = valid
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "end_to_end_demo.json").write_text(json.dumps(result, indent=2))

    print("=== GAP-PP-1-ICE-STRUCT-CONTACT-PATCH End-to-End Demo ===")
    print(f"Plate: {result['plate']['nx']}x{result['plate']['nx']}, side={result['plate']['side_m']}m, t={result['plate']['thickness_m']*1000}mm")
    print(f"Contact: Gaussian, peak={result['contact']['peak_pressure_MPa']:.1f} MPa, σ=({result['contact']['sigma_m'][0]},{result['contact']['sigma_m'][1]})m")
    print(f"Projected force: {result['projected_patch']['total_force_N']:.1f} N")
    print(f"Load balance rel: {result['projected_patch']['load_balance_rel']:.2e}")
    print(f"Peak stress (non-uniform): σ_x={result['peak_stress_nonuniform']['sigma_x_MPa']:.2f} MPa")
    print(f"Peak stress (uniform):     σ_x={result['peak_stress_uniform']['sigma_x_MPa']:.2f} MPa")
    print(f"Delta σ_x: {result['delta_nonuniform_over_uniform_pct']['sigma_x']:.2f}%")
    h3 = result["h3_anchor_check"]
    print(f"H3 anchor check: {'PASS' if h3['same_order_of_magnitude'] else 'FAIL'}")
    print(f"  small-deflection valid: {result['small_deflection_validity']['valid']} "
          f"(w/t = {result['small_deflection_validity']['peak_deflection_over_thickness']:.1f})")
    print(f"  (Gaussian FEM vs Navier: {h3['gaussian_patch_fem_vs_navier_pct']:.2f}% at nx={h3.get('plate_nx')} "
          f"vs H3 anchor {h3['h3_anchor_level_pct']}%)")

    vc = result["valid_case"]
    print(f"Valid case: w/t = {vc['small_deflection_validity']['peak_deflection_over_thickness']:.3f} "
          f"(limit {VALID_W_OVER_T_LIMIT}), valid={vc['small_deflection_validity']['valid']}")
    print(f"  FEM vs Navier (finest): {vc['h3_anchor_check']['fem_vs_navier_pct_finest']:+.3f}% "
          f"vs H3 anchor {H3_ANCHOR_PCT}% -> "
          f"{'PASS' if vc['h3_anchor_check']['same_order_of_magnitude'] else 'FAIL'}")
    print(f"  Delta sigma_x (non-uniform vs uniform): "
          f"{vc['delta_nonuniform_over_uniform_pct']['sigma_x']:+.2f}%")

    if out is not None:
        print(f"\nResults written to {out / 'end_to_end_demo.json'}")


if __name__ == "__main__":
    main()
