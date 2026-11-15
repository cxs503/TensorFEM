# -*- coding: utf-8 -*-
"""One-shot patcher: insert run_valid_case() into the GAP-PP-1 end-to-end demo."""
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "examples" / "ice_struct_contact" / "end_to_end_demo.py"
src = p.read_text()

anchor = "def run_demo(plate_nx: int = 12, plate_side: float = 1.0, out: Path | None = None):"
assert anchor in src, "run_demo anchor not found"

code = '''# ---------------------------------------------------------------------------
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

    pos, pres = _gaussian_contact_grid(cx, cy, sx, sy, peak_p, plate_side, contact_grid)
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


'''
src = src.replace(anchor, code + anchor)

old_main = """    out = Path(args.out) if args.out else None
    result = run_demo(plate_nx=args.nx, out=out)
"""
new_main = """    out = Path(args.out) if args.out else None
    result = run_demo(plate_nx=args.nx, out=None)
    valid = run_valid_case()
    result["valid_case"] = valid
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "end_to_end_demo.json").write_text(json.dumps(result, indent=2))
"""
assert old_main in src
src = src.replace(old_main, new_main)

old_print = '''    if out is not None:
        print(f"\\nResults written to {out / 'end_to_end_demo.json'}")'''
new_print = '''    vc = result["valid_case"]
    print(f"Valid case: w/t = {vc['small_deflection_validity']['peak_deflection_over_thickness']:.3f} "
          f"(limit {VALID_W_OVER_T_LIMIT}), valid={vc['small_deflection_validity']['valid']}")
    print(f"  FEM vs Navier (finest): {vc['h3_anchor_check']['fem_vs_navier_pct_finest']:+.3f}% "
          f"vs H3 anchor {H3_ANCHOR_PCT}% -> "
          f"{'PASS' if vc['h3_anchor_check']['same_order_of_magnitude'] else 'FAIL'}")
    print(f"  Delta sigma_x (non-uniform vs uniform): "
          f"{vc['delta_nonuniform_over_uniform_pct']['sigma_x']:+.2f}%")

    if out is not None:
        print(f"\\nResults written to {out / 'end_to_end_demo.json'}")'''
assert old_print in src
src = src.replace(old_print, new_print)

p.write_text(src)
print("patched OK")
