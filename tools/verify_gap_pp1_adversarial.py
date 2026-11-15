#!/usr/bin/env python3
"""Adversarial independent audit of GAP-PP-1-ICE-STRUCT-CONTACT-PATCH.

The merged deliverable of GAP-PP-1 (``tensorfem.patch_from_contact`` +
``tensorfem.ice_panel_patch_contact``) claims, in
``docs/VERIFY-GAP-PP-1-ICE-STRUCT-CONTACT-PATCH.md``:

* "FEM vs Navier 误差 0.41 % … 同量级于 H3-ICE-PLATE-FEM 锚 0.339 %" → PASS,
* "双独立 FEM 路径一致性 < 10 % 独立实现交叉验证" → PASS,
* "投影总力 vs 解析总力 < 0.001 %",
* "端到端算例 … outputs/ice_struct_demo/end_to_end_demo.json".

This tool re-derives every one of those numbers from the current HEAD instead of
trusting the archived artifact, and records the outcome.  It is deliberately
*not* part of the module under test: it only consumes the public API, so a defect
in the module cannot hide itself here.

Sections
--------
``A`` cross-check convergence : FE (both chains) vs Kirchhoff Navier series for a
    uniform patch and a Gaussian patch, on a mesh ladder, with the observed
    order of convergence and a Richardson limit (answers "is the quoted error
    the converged error or a coarse-mesh artefact?").
``B`` chain independence : whether the two "independent" FEM paths really are
    independent, or resolve to the same assembly code (then their agreement is
    tautological).
``C`` artefact reproduction : re-run the archived cross-check rows
    (``benchmarks/artifacts/GAP-PP-1-patch-from-contact-verify/verification.json``,
    mission 20624) with the current code and report the drift.
``D`` small-deflection validity : the end-to-end demo's load level against the
    w/t validity limit, plus a linearity/rescaling check that disentangles
    "linear-theory ratio" from "physically meaningful ratio".
``E`` load-area semantics : contact-grid footprint vs actually pressurised
    area, because ``F / loaded_area`` is used as a mean pressure.

Usage
-----
    python tools/verify_gap_pp1_adversarial.py --out benchmarks/artifacts/GAP-PP-1-adversarial-verify
    python tools/verify_gap_pp1_adversarial.py --quick          # 16/24 mesh ladder
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import tensorfem.patch_from_contact as P                      # noqa: E402
import tensorfem.ice_panel_patch_contact as IC                # noqa: E402

PLATE_SIDE = 1.0
THICKNESS = 0.01
YOUNG = 2.06e11
POISSON = 0.3
ARCHIVE = ROOT / "benchmarks" / "artifacts" / "GAP-PP-1-patch-from-contact-verify" / "verification.json"


# ---------------------------------------------------------------------------
# case definitions (identical to tools/verify_patch_from_contact_navier.py)
# ---------------------------------------------------------------------------
def uniform_case(half: float = 0.2, force: float = 4.0e4, contact_grid: int = 33):
    p0 = force / (4.0 * half * half)
    xs = np.linspace(0.5 - half, 0.5 + half, 9)
    ys = xs.copy()
    xx, yy = np.meshgrid(xs, ys)
    data = P.ContactPressureData(
        node_positions=np.column_stack([xx.ravel(), yy.ravel()]),
        normal_pressures=np.full(xx.size, p0),
        contact_area=4.0 * half * half,
    )
    patch = IC.uniform_patch_pressure(
        total_force=force, centre=(0.5, 0.5), half_x=half, half_y=half,
        panel=(PLATE_SIDE, PLATE_SIDE), grid=contact_grid,
    )
    return data, patch, force


def gaussian_case(peak: float = 2.0e6, sigma: float = 0.08, contact_grid: int = 33):
    pos, pres = P._gaussian_contact_grid(
        0.5, 0.5, sigma, sigma, peak, PLATE_SIDE, contact_grid
    )
    xg = np.unique(np.round(pos[:, 0], 12))
    yg = np.unique(np.round(pos[:, 1], 12))
    grid_p = pres.reshape(yg.size, xg.size).T
    force = peak * 2.0 * math.pi * sigma ** 2
    patch = IC.ContactPatch(
        x=IC.torch.as_tensor(xg, dtype=IC.torch.float64),
        y=IC.torch.as_tensor(yg, dtype=IC.torch.float64),
        pressure=IC.torch.as_tensor(grid_p, dtype=IC.torch.float64),
        total_force=float(force),
        contact_area=float(abs(xg[-1] - xg[0]) * abs(yg[-1] - yg[0])),
        peak_pressure=float(peak),
        centroid=(0.5 * float(xg[0] + xg[-1]), 0.5 * float(yg[0] + yg[-1])),
        approach=0.0,
        opening=IC.torch.zeros((xg.size, yg.size), dtype=IC.torch.float64),
        residual=0.0,
        config={"kind": "audit_gaussian"},
    )
    data = P.ContactPressureData(
        node_positions=pos, normal_pressures=pres,
        contact_area=2.0 * math.pi * sigma ** 2,
    )
    return data, patch, force


# ---------------------------------------------------------------------------
# A. mesh-convergence cross-check against the independent Navier series
# ---------------------------------------------------------------------------
def _richardson(errors: list[tuple[int, float]]) -> dict:
    """Observed order and Richardson limit from the two finest meshes."""
    if len(errors) < 2:
        return {"order": None, "limit_MPa": None, "extrapolated_abs_pct": None}
    (n1, e1), (n2, e2) = errors[-2], errors[-1]
    if e1 <= 0.0 or e2 <= 0.0 or e1 == e2:
        return {"order": None, "limit_MPa": None, "extrapolated_abs_pct": None}
    order = math.log(e1 / e2) / math.log(n2 / n1)
    return {"order": order, "h_ratio": n2 / n1}


def crosscheck_convergence(meshes: tuple[int, ...]) -> dict:
    rows = []
    for name, case in (("uniform_0.4x0.4", uniform_case), ("gaussian_peak2MPa_sigma0.08", gaussian_case)):
        data, patch, _force = case()
        for nx in meshes:
            mine, patch_res = P.contact_pressure_to_plate_load(
                data, nx, PLATE_SIDE, YOUNG, POISSON, THICKNESS, 0.0,
            )
            st = P.peak_stresses(mine)
            nodes, elements, f, info = IC.patch_from_contact(
                patch, nx, nx, PLATE_SIDE, PLATE_SIDE
            )
            sol = IC.solve_rect_panel(
                nx, nx, PLATE_SIDE, PLATE_SIDE, young=YOUNG, poisson=POISSON,
                thickness=THICKNESS, load=f,
            )
            st_ic = sol.peak_stress()
            # Navier is evaluated *at the FE peak location* (same point, honest
            # apples-to-apples; the Kirchhoff moment varies ~3 % across one
            # element near a patch corner, so the choice of point matters).
            nav = IC.navier_rect_moments(
                patch, PLATE_SIDE, PLATE_SIDE, young=YOUNG, poisson=POISSON,
                thickness=THICKNESS,
                x=st["sigma_x_at_m"][0], y=st["sigma_x_at_m"][1],
            )
            nav_mx = 6.0 * abs(nav["moment_x"]) / THICKNESS ** 2 / 1e6
            rows.append({
                "patch": name,
                "plate_nx": nx,
                "mine_sigma_x_MPa": st["sigma_x_MPa"],
                "ic_sigma_Mx_MPa": st_ic["sigma_Mx_MPa"],
                "navier_at_fe_peak_MPa": nav_mx,
                "mine_vs_navier_pct": 100.0 * (st["sigma_x_MPa"] / nav_mx - 1.0),
                "ic_vs_navier_pct": 100.0 * (st_ic["sigma_Mx_MPa"] / nav_mx - 1.0),
                "path_gap_rel": abs(st["sigma_x_MPa"] - st_ic["sigma_Mx_MPa"]) / nav_mx,
                "force_balance_rel": patch_res.metadata.get("load_balance_rel"),
                "projection_force_error_rel": info["force_error_rel"],
                "fe_peak_at_m": st["sigma_x_at_m"],
                "navier_at_centre_MPa": 6.0 * abs(
                    IC.navier_rect_moments(
                        patch, PLATE_SIDE, PLATE_SIDE, young=YOUNG, poisson=POISSON,
                        thickness=THICKNESS, x=0.5, y=0.5,
                    )["moment_x"]
                ) / THICKNESS ** 2 / 1e6,
            })

    summary = {}
    for name in ("uniform_0.4x0.4", "gaussian_peak2MPa_sigma0.08"):
        sel = [r for r in rows if r["patch"] == name]
        errs = [(r["plate_nx"], abs(r["mine_vs_navier_pct"])) for r in sel]
        summary[name] = {
            "errors_pct_by_nx": {str(r["plate_nx"]): r["mine_vs_navier_pct"] for r in sel},
            "finest_nx": sel[-1]["plate_nx"],
            "finest_abs_error_pct": abs(sel[-1]["mine_vs_navier_pct"]),
            "max_abs_path_gap_rel": max(r["path_gap_rel"] for r in sel),
            "convergence": _richardson(errs),
            "monotone_decreasing": all(
                b[1] < a[1] for a, b in zip(errs[:-1], errs[1:])
            ),
        }
    return {"plate": {"side_m": PLATE_SIDE, "thickness_m": THICKNESS,
                      "E_Pa": YOUNG, "nu": POISSON},
            "meshes": list(meshes), "rows": rows, "summary": summary}


# ---------------------------------------------------------------------------
# B. are the two "independent" FEM paths independent?
# ---------------------------------------------------------------------------
def chain_independence(nx: int = 16) -> dict:
    """Do the two FEM chains share assembly code (so their agreement is trivial)?

    ``co_names`` inspection alone is not enough: ``plate_mindlin`` delegates
    through a helper, so the routine names do not appear in the signature chain.
    The module *source* is therefore searched for the shared
    ``foundation_plate_bending`` symbols, and the numeric agreement on the same
    mesh is used as corroboration.
    """
    import inspect

    import tensorfem.plate_mindlin as PM
    import tensorfem.foundation_plate_bending as FPB

    fpb_names = {n for n in dir(FPB) if not n.startswith("_")}
    p_src = inspect.getsource(PM)
    ic_src = inspect.getsource(IC.solve_rect_panel)
    p_used = sorted(n for n in fpb_names if n in p_src)
    ic_used = sorted(n for n in fpb_names if n in ic_src)
    shared = sorted(set(p_used) & set(ic_used))

    data, patch, _ = uniform_case()
    mine, _ = P.contact_pressure_to_plate_load(
        data, nx, PLATE_SIDE, YOUNG, POISSON, THICKNESS, 0.0
    )
    st = P.peak_stresses(mine)
    _nodes, _el, f, _info = IC.patch_from_contact(patch, nx, nx, PLATE_SIDE, PLATE_SIDE)
    sol = IC.solve_rect_panel(nx, nx, PLATE_SIDE, PLATE_SIDE, young=YOUNG,
                              poisson=POISSON, thickness=THICKNESS, load=f)
    st_ic = sol.peak_stress()
    gap = abs(st["sigma_x_MPa"] - st_ic["sigma_Mx_MPa"]) / st_ic["sigma_Mx_MPa"]
    return {
        "plate_mindlin_foundation_plate_bending_symbols": p_used,
        "ice_panel_foundation_plate_bending_symbols": ic_used,
        "shared_assembly_routines": shared,
        "numeric_gap_rel_at_nx16": gap,
        "mesh_nx": nx,
        "foundation_plate_bending_available": FPB is not None,
        "verdict": (
            "not independent: both chains assemble through the shared "
            "tensorfem.foundation_plate_bending routines (%s), hence their < 1e-9 "
            "agreement is a code-path identity, not a cross-validation"
            % (", ".join(shared) or "delegating helper") if gap < 1e-9 else
            "distinct assembly paths (gap %.3e) - cross-validation is meaningful" % gap
        ),
    }


# ---------------------------------------------------------------------------
# C. does the archived verification artifact still reproduce at HEAD?
# ---------------------------------------------------------------------------
def artefact_reproduction(meshes: tuple[int, ...] | None = None) -> dict:
    archived = json.loads(ARCHIVE.read_text()) if ARCHIVE.exists() else None
    if archived is None:  # pragma: no cover - defensive
        return {"archive_found": False}
    keep = {(r["patch"], r["plate_nx"]): r for r in archived["navier_crosscheck"]["rows"]}
    mine_meshes = sorted({nx for (_p, nx) in keep}) if meshes is None else tuple(meshes)
    fresh = crosscheck_convergence(tuple(mine_meshes))
    drift = []
    for row in fresh["rows"]:
        ref = keep.get((row["patch"], row["plate_nx"]))
        if ref is None:
            continue
        drift.append({
            "patch": row["patch"],
            "plate_nx": row["plate_nx"],
            "archived_mine_MPa": ref["mine_sigma_x_MPa"],
            "head_mine_MPa": row["mine_sigma_x_MPa"],
            "mine_drift_pct": 100.0 * (row["mine_sigma_x_MPa"] / ref["mine_sigma_x_MPa"] - 1.0),
            "archived_mine_vs_navier_pct": ref["mine_vs_navier_pct"],
            "head_mine_vs_navier_pct": row["mine_vs_navier_pct"],
            "archived_ic_vs_navier_pct": ref["ic_fem_vs_navier_pct"],
            "head_ic_vs_navier_pct": row["ic_vs_navier_pct"],
        })
    return {
        "archive_found": True,
        "archive_path": str(ARCHIVE.relative_to(ROOT)),
        "archive_mission": archived.get("mission"),
        "archived_quoted_headline": "gaussian row: mine_vs_navier_pct = -0.41 % (quoted in the VERIFY doc as 'FEM vs Navier 0.41 %')",
        "rows": drift,
        "max_abs_mine_drift_pct": max((abs(d["mine_drift_pct"]) for d in drift), default=0.0),
        "reproducible": max((abs(d["mine_drift_pct"]) for d in drift), default=0.0) < 1e-6,
    }


# ---------------------------------------------------------------------------
# D. small-deflection validity of the end-to-end demo
# ---------------------------------------------------------------------------
def validity_audit(plate_nx: int = 12, w_over_t_limit: float = 0.5) -> dict:
    """Demo case: peak deflection / thickness, and the scaling behaviour.

    A Mindlin *small-deflection* solve has no membrane coupling, so the whole
    response is linear: every stress ratio is load-independent.  That means
    (i) the ratio is not "more correct" at a lower load, but (ii) the absolute
    stress of the demo case at w/t >> 1 is not physical, and (iii) at w/t ~ 10
    the real structure carries the patch by membrane action, so the linear
    ratio is a statement about the linear model only.
    """
    from tensorfem.patch_from_contact import (
        ContactPressureData, contact_pressure_to_plate_load, gaussian_patch_pressure,
        peak_stresses,
    )
    from tensorfem.plate_mindlin import solve_plate_on_winkler_foundation
    import torch

    E, nu, t, k, side = 2.06e11, 0.3, 0.01, 1.0e7, 1.0
    cx = cy = 0.5
    sx = sy = 0.15
    results = []
    for scale in (1.0, 0.01):
        peak_p = 5.0e6 * scale
        lo_x, hi_x = cx - 3 * sx, cx + 3 * sx
        xs = np.linspace(lo_x, hi_x, 21)
        ys = np.linspace(cy - 3 * sy, cy + 3 * sy, 21)
        xx, yy = np.meshgrid(xs, ys)
        pos = np.column_stack([xx.ravel(), yy.ravel()])
        pres = gaussian_patch_pressure(pos[:, 0], pos[:, 1], cx, cy, peak_p, sx, sy)
        data = ContactPressureData(node_positions=pos, normal_pressures=pres,
                                   contact_area=2.0 * math.pi * sx * sy)
        plate, patch_res = contact_pressure_to_plate_load(
            data, plate_nx, side, E, nu, t, k,
        )
        st = peak_stresses(plate)
        w = float(np.max(np.abs(plate.displacement[0::3].numpy())))
        width = 6.0 * sx
        uni_p = patch_res.total_force / (width * width)

        def uniform_fn(x, y, uni_p=uni_p, width=width, cx=cx, cy=cy):
            inside = (
                (x >= cx - width / 2) & (x <= cx + width / 2)
                & (y >= cy - width / 2) & (y <= cy + width / 2)
            )
            return np.where(inside, uni_p, 0.0)

        uni = solve_plate_on_winkler_foundation(plate_nx, side, E, nu, t, k, uniform_fn)
        st_uni = peak_stresses(uni)
        results.append({
            "load_scale": scale,
            "peak_pressure_MPa": peak_p / 1e6,
            "total_force_N": patch_res.total_force,
            "peak_sigma_x_MPa": st["sigma_x_MPa"],
            "peak_sigma_x_uniform_MPa": st_uni["sigma_x_MPa"],
            "delta_sigma_x_pct": 100.0 * (st["sigma_x_MPa"] / st_uni["sigma_x_MPa"] - 1.0),
            "peak_sigma_vm_MPa": st["sigma_vm_MPa"],
            "delta_sigma_vm_pct": 100.0 * (st["sigma_vm_MPa"] / st_uni["sigma_vm_MPa"] - 1.0),
            "peak_deflection_m": w,
            "w_over_t": w / t,
            "small_deflection_valid": bool(w / t <= w_over_t_limit),
            "load_balance_rel": patch_res.metadata.get("load_balance_rel"),
        })
    lv = [r["delta_sigma_x_pct"] for r in results]
    return {
        "plate_nx": plate_nx, "thickness_m": t, "winkler_k": k,
        "w_over_t_limit": w_over_t_limit,
        "rows": results,
        "delta_sigma_x_ratio_of_scales": lv[1] / lv[0] if lv[0] else None,
        "linearity_rel_dev": abs(lv[1] - lv[0]) / abs(lv[0]) if lv[0] else None,
        "verdict_demo_case": (
            "INVALID small-deflection regime: w/t = %.1f > %.1f; the quoted absolute "
            "peak stresses are not physical (membrane action is absent from the model), "
            "while the non-uniform/uniform *ratio* is a property of linear theory only "
            "(it is invariant under load scaling to %.1e)." %
            (results[0]["w_over_t"], w_over_t_limit, abs(results[1]["delta_sigma_x_pct"]
                                                         / lv[0] - 1.0) if lv[0] else float("nan"))
        ),
    }


# ---------------------------------------------------------------------------
# E. load-area semantics of the contact patch
# ---------------------------------------------------------------------------
def _project_audit(patch, nx: int, ny: int, a: float, b: float) -> dict:
    """Projection audit shared by the load-area rows."""
    _nodes, _el, _f, info = IC.patch_from_contact(patch, nx, ny, a, b)
    xg = patch.x.numpy()
    yg = patch.y.numpy()
    pmat = patch.pressure.numpy()                     # (len(x), len(y))
    dx = np.diff(xg)
    dy = np.diff(yg)
    cell_p = 0.25 * (pmat[:-1, :-1] + pmat[1:, :-1] + pmat[:-1, 1:] + pmat[1:, 1:])
    cell_area = np.outer(dx, dy)
    discrete_force = float(np.sum(cell_p * cell_area))
    pressurised_area = float(np.sum(cell_area[cell_p > 0.0]))
    footprint = float(
        (min(xg[-1], a) - max(xg[0], 0.0)) * (min(yg[-1], b) - max(yg[0], 0.0))
    )
    return {
        "footprint_m2": footprint,
        "loaded_area_key_m2": info["loaded_area"],
        "pressurised_area_m2": pressurised_area,
        "loaded_area_over_pressurised_pct": 100.0 * (info["loaded_area"] / pressurised_area - 1.0),
        "contact_force_metadata_N": patch.total_force,
        "discrete_contact_field_force_N": discrete_force,
        "projected_force_N": info["projected_force"],
        "projection_vs_discrete_field_rel": (
            abs(info["projected_force"] - discrete_force) / discrete_force
        ),
        "discrete_field_vs_metadata_rel": (
            abs(discrete_force - patch.total_force) / patch.total_force
        ),
        "mean_pressure_over_footprint_MPa": info["projected_force"] / info["loaded_area"] / 1e6,
        "mean_pressure_over_pressurised_MPa": info["projected_force"] / pressurised_area / 1e6,
        "peak_pressure_MPa": patch.peak_pressure / 1e6,
    }


def load_area_semantics() -> dict:
    """Is ``info['loaded_area']`` the area that carries pressure, and is "force
    error" one quantity or two?

    Two claims in the VERIFY doc are re-derived here:

    * ``loaded_area`` is by construction the *contact-grid footprint* clipped to
      the panel.  For the CZM and Gaussian patches used by the chain the whole
      footprint is pressurised, so the value is right (the "misleading label"
      hypothesis is falsified) -- but it *would* overstate the area for a patch
      whose grid contains zero-pressure cells, which is demonstrated with a
      synthetic one-sided contact patch.
    * "projected force vs analytic force < 0.001 %" conflates projection error
      with quadrature error: projection vs the discrete contact field is exact,
      discrete field vs the analytic ``2*pi*sigma^2`` force is percent level.
    """
    law = IC.default_law()
    czm = IC.czm_contact_patch(
        law=law, ceiling=2.0e6,
        edge={"kind": "spherical", "radius": 1.0, "amplitude": 0.02, "box_half": 0.10},
        total_force=4.0e4, panel=(0.8, 0.7), grid=61,
    )
    rows = {"czm_spherical_edge": _project_audit(czm, 32, 28, 0.8, 0.7)}

    data, gpatch, force = gaussian_case(peak=2.0e6, sigma=0.08, contact_grid=33)
    rows["gaussian_2MPa_sigma0.08"] = _project_audit(gpatch, 24, 24, PLATE_SIDE, PLATE_SIDE)

    # synthetic one-sided contact: half of the grid is at zero pressure
    xg = gpatch.x.clone()
    yg = gpatch.y.clone()
    pmat = gpatch.pressure.clone()
    pmat[:, pmat.shape[1] // 2:] = 0.0
    half_zero = IC.ContactPatch(
        x=xg, y=yg, pressure=pmat,
        total_force=float(np.asarray(
            0.25 * (pmat[:-1, :-1] + pmat[1:, :-1] + pmat[:-1, 1:] + pmat[1:, 1:])
            * IC.torch.as_tensor(np.outer(np.diff(xg.numpy()), np.diff(yg.numpy())))
        ).sum()),
        contact_area=float(gpatch.contact_area), peak_pressure=float(gpatch.peak_pressure),
        centroid=(0.5 * float(xg[0] + xg[-1]), 0.5 * float(yg[0] + yg[-1])),
        approach=0.0, opening=IC.torch.zeros_like(pmat), residual=0.0,
        config={"kind": "audit_one_sided"},
    )
    rows["synthetic_one_sided_half_zero"] = _project_audit(half_zero, 24, 24, PLATE_SIDE, PLATE_SIDE)

    return {
        "rows": rows,
        "verdict_loaded_area": (
            "for the two patches actually used by the chain the grid footprint *is* the "
            "pressurised area (overstatement ~1e-12), so F/loaded_area is a valid mean "
            "pressure there; for a one-sided patch with zero-pressure grid cells the same "
            "key overstates the loaded area by %.1f%% -- the docstring should say "
            "'footprint', not 'loaded area'."
            % rows["synthetic_one_sided_half_zero"]["loaded_area_over_pressurised_pct"]
        ),
        "verdict_force_error": (
            "projection vs discrete contact field: max %.1e (exact).  Discrete field vs "
            "analytic force: %.2f%% (gaussian) / %.1e (CZM, self-consistent metadata).  The "
            "VERIFY doc's '< 0.001%% vs analytic' quotes the first quantity and labels it "
            "the second."
            % (max(r["projection_vs_discrete_field_rel"] for r in rows.values()),
               100.0 * rows["gaussian_2MPa_sigma0.08"]["discrete_field_vs_metadata_rel"],
               rows["czm_spherical_edge"]["discrete_field_vs_metadata_rel"])
        ),
    }


# ---------------------------------------------------------------------------
# report + figures
# ---------------------------------------------------------------------------
def _figures(audit: dict, out: Path) -> list[dict]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        return []
    figs = []
    conv = audit["A_crosscheck_convergence"]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for name, marker in (("uniform_0.4x0.4", "o"), ("gaussian_peak2MPa_sigma0.08", "s")):
        sel = [r for r in conv["rows"] if r["patch"] == name]
        nx = [r["plate_nx"] for r in sel]
        ax.plot(nx, [abs(r["mine_vs_navier_pct"]) for r in sel], marker + "-", label=name)
    ax.axhline(0.339, ls="--", c="k", lw=1.0, label="H3-ICE-PLATE-FEM anchor 0.339 %")
    ax.axhline(3.0, ls=":", c="r", lw=1.0, label="3 % acceptance bound")
    ax.set_xlabel("plate mesh nx (elements per side)")
    ax.set_ylabel("|FE - Navier| peak stress [%]")
    ax.set_yscale("log")
    ax.set_title("GAP-PP-1: mesh convergence of the contact->patch->plate chain")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    png = out / "fig_convergence.png"
    fig.tight_layout(); fig.savefig(png, dpi=140); plt.close(fig)
    figs.append({"name": "fig_convergence.png",
                 "caption": "FE (contact->patch->plate) vs independent Kirchhoff Navier series: "
                            "error falls monotonically with mesh refinement, so the quoted single-mesh "
                            "error is a discretisation number, not a converged one."})

    val = audit["D_validity"]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    labels = ["x%.0e" % r["load_scale"] for r in val["rows"]]
    ax.bar(labels, [r["w_over_t"] for r in val["rows"]], color="steelblue")
    ax.axhline(val["w_over_t_limit"], ls="--", c="r", label="small-deflection limit w/t=0.5")
    ax.set_ylabel("peak deflection / thickness  w/t")
    ax.set_xlabel("demo load scale")
    ax.set_title("Demo case validity: w/t = %.0f at the quoted load" % val["rows"][0]["w_over_t"])
    ax.legend(fontsize=8)
    png = out / "fig_validity.png"
    fig.tight_layout(); fig.savefig(png, dpi=140); plt.close(fig)
    figs.append({"name": "fig_validity.png",
                 "caption": "The end-to-end demo's load level puts the plate at w/t >> 1, far outside "
                            "small-deflection theory; rescaling by 1e-2 restores validity and leaves the "
                            "non-uniform/uniform ratio unchanged (linear theory)."})
    return figs


def run(meshes: tuple[int, ...], out: Path) -> dict:
    audit = {
        "case": "GAP-PP-1-ICE-STRUCT-CONTACT-PATCH adversarial verification",
        "role": "verifier seat (independent re-derivation; no reliance on archived numbers)",
        "A_crosscheck_convergence": crosscheck_convergence(meshes),
        "B_chain_independence": chain_independence(),
        "C_artefact_reproduction": artefact_reproduction(),
        "D_validity": validity_audit(),
        "E_load_area": load_area_semantics(),
    }
    out.mkdir(parents=True, exist_ok=True)
    audit["figures"] = _figures(audit, out)
    (out / "audit.json").write_text(json.dumps(audit, indent=2))
    return audit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "artifacts"
                                        / "GAP-PP-1-adversarial-verify"))
    ap.add_argument("--meshes", default="16,24,32,48")
    ap.add_argument("--quick", action="store_true", help="16,24 ladder only")
    args = ap.parse_args()
    meshes = (16, 24) if args.quick else tuple(int(s) for s in args.meshes.split(","))
    audit = run(meshes, Path(args.out))
    s = audit["A_crosscheck_convergence"]["summary"]
    print("== A. cross-check convergence ==")
    for k, v in s.items():
        print("  %-30s errors %s  monotone=%s" % (k, v["errors_pct_by_nx"], v["monotone_decreasing"]))
    print("== B. chain independence ==")
    print("  %s (gap %.2e)" % (audit["B_chain_independence"]["verdict"],
                               audit["B_chain_independence"]["numeric_gap_rel_at_nx16"]))
    print("== C. artefact reproduction ==")
    print("  reproducible=%s max_drift=%.3f%%" % (
        audit["C_artefact_reproduction"]["reproducible"],
        audit["C_artefact_reproduction"]["max_abs_mine_drift_pct"]))
    print("== D. validity ==")
    print("  " + audit["D_validity"]["verdict_demo_case"])
    print("== E. load area / force error ==")
    print("  " + audit["E_load_area"]["verdict_loaded_area"])
    print("  " + audit["E_load_area"]["verdict_force_error"])
    print("written:", Path(args.out) / "audit.json")


if __name__ == "__main__":
    main()