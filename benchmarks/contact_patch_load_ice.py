"""Benchmark: ice contact-patch loads on a hull panel.

Gap closed: ``GAP-PP-1-ICE-STRUCT-CONTACT-PATCH`` -- the contact pressure field
from a damageable crush-interface solve is projected onto the panel load array
(``patch_from_contact``) and drives the Mindlin plate solver, with the panel
deflection fed back into the interface (two-way coupling).  The study also
sweeps how the *non-uniformity* of the patch changes the peak panel stress.

Runs three things and writes ``artifacts/contact_patch_load/study.json``:

1. ``verification`` -- the simply supported square plate under uniform load
   against the Timoshenko & Woinowsky-Krieger (1959) coefficients, and three
   load patterns (uniform, central patch, parabolic ridge) against the exact
   Navier double-series solution at two mesh resolutions.
2. ``coupling`` -- the ice/panel two-way fixed point on a clamped marine panel
   bay for two contact footprints, five gap (ice-edge imperfection) amplitudes
   and two interface residual ratios, with a mesh-refinement check.
3. ``identities`` -- force conservation, work conjugacy of the transfer pair,
   reaction equilibrium and strain-energy/work equality, recorded per case.

Every number in the rendered report comes from this JSON (no hand copying).

Usage
-----
    python benchmarks/contact_patch_load_ice.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

torch.set_num_threads(4)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, ".."))

from tensorfem.contact_patch_load import (  # noqa: E402
    ContactFootprint,
    CrushInterfaceLaw,
    PanelSpec,
    PlateModel,
    navier_series_reference,
    patch_from_contact,
    plate_contact_interpolation,
    principal_moments,
    solve_contact_plate_coupling,
)

# Timoshenko & Woinowsky-Krieger, "Theory of Plates and Shells" (2nd ed. 1959),
# Table 8 / Art. 30: uniformly loaded simply supported square plate, nu = 0.3,
# w_max = 0.00406235 q a^4 / D and M_max = 0.0479 q a^2.
LITERATURE_UNIFORM_W = 0.00406235
LITERATURE_UNIFORM_M = 0.0479

OUT_DIR = os.path.join(HERE, "..", "artifacts", "contact_patch_load")


# --------------------------------------------------------------------------
# analytic pressure fields used as verification loads
# --------------------------------------------------------------------------
def pressure_uniform(x, y):
    return np.ones_like(x)


def pressure_patch(x, y, x0=0.25, x1=0.75, y0=0.25, y1=0.75, q=1.0):
    inside = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    return np.where(inside, q, 0.0)


def pressure_ridge(x, y, x0=0.25, x1=0.75, amp=1.0, q=1.0):
    """``q (1 + amp (1 - ((x - xc)/(half width))^2))`` on a full-width band.

    A smooth high-pressure core straddling the plate centre line: the
    non-uniformity is controlled by ``amp`` alone, so the sweep isolates the
    effect of the pressure *shape* at constant footprint.
    """
    xc = 0.5 * (x0 + x1)
    half = 0.5 * (x1 - x0)
    shape = 1.0 - amp * ((x - xc) / half) ** 2
    return np.where((x >= x0) & (x <= x1), q * shape, 0.0)


def load_from_pressure(nodes, elements, fn, x0, x1, y0, y1, nx, ny):
    """Discretise an analytic pressure field on a contact grid and project it."""
    foot = ContactFootprint.grid(x0, x1, y0, y1, nx, ny)
    p = torch.as_tensor(np.asarray(
        fn(foot.points[:, 0].numpy(), foot.points[:, 1].numpy())), dtype=torch.float64)
    return patch_from_contact(foot.points, p, nodes, elements,
                              point_weights=foot.weights)


# --------------------------------------------------------------------------
# 1. verification against Timoshenko coefficients and the Navier series
# --------------------------------------------------------------------------
def verification(a: float, t: float, young: float, poisson: float,
                 meshes, modes=(199, 299)) -> dict:
    D = young * t ** 3 / (12.0 * (1.0 - poisson ** 2))
    cases = {
        "uniform": dict(fn=pressure_uniform, grid=(0.0, a, 0.0, a, 40, 40)),
        "central_patch": dict(fn=lambda x, y: pressure_patch(x, y, 0.25 * a, 0.75 * a,
                                                            0.25 * a, 0.75 * a, 4.0),
                              grid=(0.0, a, 0.0, a, 40, 40)),
        "parabolic_ridge": dict(fn=lambda x, y: pressure_ridge(x, y, 0.25 * a, 0.75 * a,
                                                              1.0, 1.5),
                                grid=(0.25 * a, 0.75 * a, 0.0, a, 20, 40)),
    }
    out = {"plate": {"side": a, "thickness": t, "young": young,
                     "poisson": poisson, "rigidity": D},
           "literature": {"w_coefficient": LITERATURE_UNIFORM_W,
                          "m_coefficient": LITERATURE_UNIFORM_M,
                          "source": "Timoshenko & Woinowsky-Krieger (1959) Table 8, nu=0.3"},
           "cases": {}}
    for name, spec in cases.items():
        entry = {"load": {"grid": spec["grid"]}, "meshes": {}, "series": {}}
        for modes_n in modes:
            ref = navier_series_reference(
                a, a, young=young, poisson=poisson, thickness=t,
                pressure_fn=spec["fn"], modes=modes_n, model="mindlin",
                quadrature=48)
            entry["series"][str(modes_n)] = {
                "w_centre": ref["w_centre"], "Mx_centre": ref["Mx_centre"],
                "m_peak_interior": ref["m_peak"],
            }
        for n in meshes:
            from tensorfem.ice_panel_fsi import rect_mesh
            nodes, elems = rect_mesh(a, a, n, n)
            load = load_from_pressure(nodes, elems, spec["fn"], *spec["grid"])
            model = PlateModel(nodes, elems, young=young, poisson=poisson,
                               thickness=t, boundary="simply_supported")
            resp = model.solve(dof_load=load.dof_load)
            ref = navier_series_reference(
                a, a, young=young, poisson=poisson, thickness=t,
                pressure_fn=spec["fn"], modes=modes[0], model="mindlin",
                quadrature=48, points=resp.gauss_xy.numpy())
            m_fe = np.asarray(resp.gauss_moments)
            m_ref = np.stack((ref["Mx"], ref["My"], ref["Mxy"]), axis=-1)
            p_fe = principal_moments(torch.as_tensor(m_fe)).numpy()
            p_ref = principal_moments(torch.as_tensor(m_ref)).numpy()
            # Reissner boundary layer: within ~4 thicknesses of a hard simple
            # support the Q4 field and the Kirchhoff series legitimately differ
            # (the FE imposes w = theta_t = 0 there), so the L2 metric is taken
            # over the interior; the full-field value is kept as a diagnostic.
            pts = resp.gauss_xy.numpy()
            dist = np.minimum.reduce([pts[:, 0] - 0.0, a - pts[:, 0],
                                      pts[:, 1] - 0.0, a - pts[:, 1]])
            interior = dist > 4.0 * t
            mask = np.abs(p_ref).max(axis=1) > 0.1 * np.max(np.abs(p_ref))
            # normalised L2 error over the full moment tensor (all three
            # components, scaled by the peak): robust where Mx crosses zero,
            # unlike a pointwise relative error
            rel_l2_full = math.sqrt(float(np.sum((m_fe - m_ref) ** 2))
                                    / float(np.sum(m_ref ** 2)))
            rel_l2 = math.sqrt(float(np.sum((m_fe[interior] - m_ref[interior]) ** 2))
                               / float(np.sum(m_ref[interior] ** 2)))
            entry["meshes"][str(n)] = {
                "n_elements": int(elems.shape[0]),
                "total_force": load.total_force,
                "w_centre": resp.centre_deflection,
                "m_peak": resp.m_peak,
                "sigma_peak": resp.sigma_peak,
                "strain_energy": resp.strain_energy,
                "external_work": resp.external_work,
                "energy_work_rel": abs(resp.strain_energy / resp.external_work - 1.0),
                "field_points_compared": int(mask.sum()),
                "field_rel_l2_error": rel_l2,
                "field_rel_l2_error_full": rel_l2_full,
                "interior_points": int(interior.sum()),
                "field_correlation_Mx": float(np.corrcoef(m_fe[:, 0], m_ref[:, 0])[0, 1]),
            }
        # FE-versus-series ratios on the finest mesh (same point set)
        fine = entry["meshes"][str(max(meshes))]
        entry["ratios"] = {
            "w_centre_fe_vs_series": fine["w_centre"] / entry["series"][str(modes[0])]["w_centre"] - 1.0,
            "m_peak_fe_vs_series_grid": fine["m_peak"] / entry["series"][str(modes[0])]["m_peak_interior"] - 1.0,
        }
        out["cases"][name] = entry
    # literature check on the uniform case
    uni = out["cases"]["uniform"]["meshes"][str(max(meshes))]
    out["literature_check"] = {
        "w_centre_fe": uni["w_centre"],
        "w_centre_literature": LITERATURE_UNIFORM_W * 1.0 * a ** 4 / D,
        "w_centre_rel_error": uni["w_centre"] / (LITERATURE_UNIFORM_W * a ** 4 / D) - 1.0,
        "M_peak_fe": uni["m_peak"],
        "M_peak_literature": LITERATURE_UNIFORM_M * 1.0 * a ** 2,
        "M_peak_rel_error": uni["m_peak"] / (LITERATURE_UNIFORM_M * a ** 2) - 1.0,
    }
    return out


# --------------------------------------------------------------------------
# 2. two-way coupled ice/panel sweep
# --------------------------------------------------------------------------
def gap_profile(foot: ContactFootprint, amplitude: float, span_y: float) -> torch.Tensor:
    """Sinusoidal ice-edge imperfection along the contact length."""
    y = foot.points[:, 1]
    y0, y1 = float(y.min()), float(y.max())
    return torch.as_tensor(amplitude * 0.5 * (1.0 - np.cos(
        2.0 * math.pi * (y.numpy() - y0) / max(y1 - y0, 1e-12))),
        dtype=torch.float64)


def coupling_sweep(panel: PanelSpec, footprints, amplitudes, residual_ratios,
                   law_kwargs, advance, model_cache=None) -> dict:
    nodes, elems = panel.mesh()
    key = (panel.nx, panel.ny)
    if model_cache is None:
        model_cache = {}
    if key not in model_cache:
        model_cache[key] = PlateModel(
            nodes, elems, young=panel.young, poisson=panel.poisson,
            thickness=panel.thickness, boundary=panel.boundary)
    model = model_cache[key]
    out = {"panel": {"span_x": panel.span_x, "span_y": panel.span_y,
                     "thickness": panel.thickness, "young": panel.young,
                     "poisson": panel.poisson, "boundary": panel.boundary,
                     "nx": panel.nx, "ny": panel.ny,
                     "n_nodes": int(nodes.shape[0]),
                     "n_elements": int(elems.shape[0])},
           "advance": advance, "law": dict(law_kwargs), "runs": {}}
    for fp_name, foot in footprints.items():
        C, _ = plate_contact_interpolation(nodes, elems, foot.points, foot.weights)
        for rr in residual_ratios:
            law = CrushInterfaceLaw(residual_ratio=rr, **law_kwargs)
            for amp in amplitudes:
                t0 = time.time()
                gap = gap_profile(foot, amp, panel.span_y)
                res = solve_contact_plate_coupling(
                    panel, foot, law, advance, gap=gap, model=model,
                    interpolation=C, point_weights=foot.weights)
                ident = {
                    "force_conservation_rel": abs(
                        float(res.load.forces_z.sum()) / res.load.total_force - 1.0),
                    "work_conjugacy_rel": abs(
                        float(torch.dot(res.load.weights * res.load.pressures,
                                        C @ res.response.w))
                        - float(torch.dot(res.load.forces_z, res.response.w))
                    ) / max(abs(float(torch.dot(res.load.forces_z, res.response.w))), 1e-30),
                    "reaction_equilibrium_rel": abs(
                        float(-res.response.reactions_z.sum()) / res.load.total_force - 1.0),
                    "energy_work_rel": abs(
                        res.response.strain_energy / res.response.external_work - 1.0),
                    "penetration_max": float(torch.max(res.penetration)),
                }
                out["runs"][f"{fp_name}|r{rr}|a{amp*1e3:.1f}mm"] = {
                    "footprint": fp_name,
                    "footprint_area": foot.area,
                    "residual_ratio": rr,
                    "gap_amplitude": amp,
                    "iterations": res.iterations,
                    "converged": bool(res.converged),
                    "last_residual": res.residual_history[-1]["residual"],
                    "total_force": res.total_force,
                    "mean_pressure": res.load.mean_pressure,
                    "peak_pressure": res.load.peak_pressure,
                    "peak_to_mean": res.peak_to_mean,
                    "coefficient_of_variation": res.coefficient_of_variation,
                    "sigma_peak": res.sigma_peak,
                    "m_peak": res.response.m_peak,
                    "sigma_peak_uniform": res.sigma_peak_uniform,
                    "amplification": res.amplification,
                    "w_max": float(torch.max(torch.abs(res.response.w))),
                    "one_way": res.one_way,
                    "identities": ident,
                    "wall_time_s": time.time() - t0,
                }
    return out


def mesh_refinement(panel_coarse: PanelSpec, foot, law, advance, gap) -> dict:
    out = {}
    cache = {}
    for nx, ny in ((panel_coarse.nx, panel_coarse.ny),
                   (2 * panel_coarse.nx, 2 * panel_coarse.ny)):
        panel = PanelSpec(span_x=panel_coarse.span_x, span_y=panel_coarse.span_y,
                          thickness=panel_coarse.thickness,
                          young=panel_coarse.young, poisson=panel_coarse.poisson,
                          boundary=panel_coarse.boundary, nx=nx, ny=ny)
        res = solve_contact_plate_coupling(panel, foot, law, advance, gap=gap,
                                           model=cache.get((nx, ny)))
        out[f"{nx}x{ny}"] = {
            "n_elements": int(panel.mesh()[1].shape[0]),
            "sigma_peak": res.sigma_peak,
            "sigma_peak_uniform": res.sigma_peak_uniform,
            "amplification": res.amplification,
            "m_peak": res.response.m_peak,
            "total_force": res.total_force,
            "peak_pressure": res.load.peak_pressure,
            "iterations": res.iterations,
            "converged": bool(res.converged),
        }
    keys = list(out)
    fine, coarse = out[keys[1]], out[keys[0]]
    # observed order from a 2-level refinement (h halved): p = log2(ratio)
    ratio = coarse["sigma_peak"] / fine["sigma_peak"]
    out["refinement"] = {
        "sigma_peak_ratio_coarse_over_fine": ratio,
        "observed_order": (math.log(ratio) / math.log(2.0)) if ratio > 0 else float("nan"),
        "gci_fine_percent": (1.25 * abs(ratio - 1.0) / (2.0 ** 1.0 - 1.0) * 100.0)
        if ratio > 0 else float("nan"),
    }
    return out



def prescribed_nonuniformity(panel: PanelSpec, foot: ContactFootprint,
                             model: PlateModel, total_force: float,
                             amplitudes) -> dict:
    """Structural sensitivity: same total force, pressure concentrated
    towards the clamped frame edge (``x = 0``).

    This isolates the *structural* answer to "does a high-pressure zone raise
    the peak panel stress?" from the interface law: the total force and the
    footprint are held fixed and only the pressure shape changes, so the
    amplification is a pure plate-response quantity.
    """
    x = foot.points[:, 0]
    y = foot.points[:, 1]
    x0, x1 = float(x.min()), float(x.max())
    weights = foot.weights
    out = {"total_force": total_force, "footprint_area": foot.area,
           "footprint": {"x": [x0, x1], "y": [float(y.min()), float(y.max())]},
           "runs": {}}
    base = None
    for amp in amplitudes:
        shape = torch.as_tensor(
            1.0 + amp * np.cos(math.pi * (x.numpy() - x0) / max(x1 - x0, 1e-12)),
            dtype=torch.float64)
        mean_shape = float(torch.sum(weights * shape) / torch.sum(weights))
        p = shape / mean_shape * (total_force / foot.area)
        load = patch_from_contact(foot.points, p, point_weights=weights,
                                  interpolation=None,
                                  nodes=model.nodes, elements=model.elements)
        resp = model.solve(dof_load=load.dof_load)
        if base is None:
            base = resp.sigma_peak
        out["runs"][f"amp{amp:.2f}"] = {
            "shape_amplitude": amp,
            "peak_to_mean": load.peak_to_mean,
            "coefficient_of_variation": load.coefficient_of_variation,
            "sigma_peak": resp.sigma_peak,
            "m_peak": resp.m_peak,
            "amplification": resp.sigma_peak / base,
            "total_force": load.total_force,
        }
    return out


# --------------------------------------------------------------------------
# 3. figures
# --------------------------------------------------------------------------
def figures(study: dict, out_dir: str) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    made = []

    # (a) the pressure field of a conforming versus an imperfect ice edge
    panel = PanelSpec()
    foot = ContactFootprint.grid(0.15, 0.60, 0.70, 1.70, 5, 21)
    law = CrushInterfaceLaw(**study["sweep"]["law"])
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), dpi=110)
    for ax, amp in zip(axes, (0.0, 0.006)):
        gap = gap_profile(foot, amp, panel.span_y)
        res = solve_contact_plate_coupling(panel, foot, law, 0.010, gap=gap)
        sc = ax.scatter(foot.points[:, 0], foot.points[:, 1],
                        c=np.asarray(res.load.pressures) / 1e6, s=26,
                        cmap="viridis", vmin=0.0)
        ax.set_title(f"gap amplitude {amp*1e3:.0f} mm\n"
                     f"peak/mean = {res.peak_to_mean:.2f}", fontsize=9)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        ax.set_xlim(0.0, panel.span_x); ax.set_ylim(0.0, panel.span_y)
        plt.colorbar(sc, ax=ax, label="p [MPa]")
    fig.tight_layout()
    path = os.path.join(out_dir, "pressure_fields.png")
    fig.savefig(path); plt.close(fig)
    made.append({"name": "pressure_fields.png",
                 "caption": "Contact pressure from the crush-interface solve on the "
                            "non-matching contact grid: conforming edge (left) versus a "
                            "6 mm sinusoidal ice-edge imperfection (right)."})

    # (b) non-uniformity sweep: peak-stress amplification
    fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=110)
    for fp in ("centre", "edge"):
        for rr in study["sweep"]["residual_ratios"]:
            xs, ys = [], []
            for key, run in study["sweep"]["runs"].items():
                if run["footprint"] == fp and run["residual_ratio"] == rr:
                    xs.append(run["coefficient_of_variation"])
                    ys.append(run["amplification"])
            order = np.argsort(xs)
            ax.plot(np.asarray(xs)[order], np.asarray(ys)[order], marker="o",
                    label=f"{fp} footprint, residual ratio {rr}")
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("pressure coefficient of variation  CV = sd/mean")
    ax.set_ylabel(r"$\sigma_{peak}$ / $\sigma_{peak}$(uniform, same force)")
    ax.set_title("Peak panel stress amplification of the contact patch", fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = os.path.join(out_dir, "amplification_sweep.png")
    fig.savefig(path); plt.close(fig)
    made.append({"name": "amplification_sweep.png",
                 "caption": "Peak-stress amplification of the non-uniform contact patch "
                            "over the statically equivalent uniform patch, against the "
                            "patch pressure coefficient of variation."})

    # (c) FE versus exact Navier series along the plate centre line
    from tensorfem.ice_panel_fsi import rect_mesh
    a, t = study["verification"]["plate"]["side"], study["verification"]["plate"]["thickness"]
    young, nu = study["verification"]["plate"]["young"], study["verification"]["plate"]["poisson"]
    nodes, elems = rect_mesh(a, a, 24, 24)
    fn = lambda x, y: pressure_ridge(x, y, 0.25 * a, 0.75 * a, 1.0, 1.5)  # noqa: E731
    load = load_from_pressure(nodes, elems, fn, 0.25 * a, 0.75 * a, 0.0, a, 20, 40)
    model = PlateModel(nodes, elems, young=young, poisson=nu, thickness=t,
                       boundary="simply_supported")
    resp = model.solve(dof_load=load.dof_load)
    line = np.linspace(0.05 * a, 0.95 * a, 61)
    pts = np.stack((line, np.full_like(line, 0.5 * a)), axis=1)
    ref = navier_series_reference(a, a, young=young, poisson=nu, thickness=t,
                                  pressure_fn=fn, modes=199, quadrature=48,
                                  points=pts)
    gp = resp.gauss_xy.numpy()
    band = np.abs(gp[:, 1] - 0.5 * a) < 0.02 * a
    fig, ax = plt.subplots(figsize=(6.2, 3.4), dpi=110)
    ax.plot(pts[:, 0] / a, ref["Mx"] / 1e3, "-", label="Navier series (199 modes)")
    ax.plot(gp[band, 0] / a, np.asarray(resp.gauss_moments)[band, 0] / 1e3, "o",
            ms=3, label="Q4 Mindlin FE, 24x24 (Gauss points)")
    ax.set_xlabel("x/a"); ax.set_ylabel(r"$M_x$ [kN]")
    ax.set_title("Parabolic ridge patch load: sagging moment along y = a/2", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, "ridge_moment_profile.png")
    fig.savefig(path); plt.close(fig)
    made.append({"name": "ridge_moment_profile.png",
                 "caption": "Sagging moment along the panel centre line for the parabolic "
                            "ridge patch: finite element versus the exact Navier series."})
    return made


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="single mesh / fewer sweep points (smoke test)")
    args = ap.parse_args()
    t_start = time.time()

    a, t, young, poisson = 1.0, 0.01, 210.0e9, 0.30
    meshes = (16,) if args.quick else (16, 24)
    ver = verification(a, t, young, poisson, meshes)

    panel = PanelSpec()
    footprints = {
        "centre": ContactFootprint.grid(0.15, 0.60, 0.70, 1.70, 5, 21),
        "edge": ContactFootprint.grid(0.00, 0.30, 0.70, 1.70, 3, 21),
    }
    law_kwargs = dict(stiffness=5.3e8, crush_strength=5.0e6, softening_ratio=4.0)
    amplitudes = (0.0,) if args.quick else (0.0, 0.002, 0.004, 0.006, 0.008)
    residual_ratios = (0.5,) if args.quick else (0.5, 0.2)
    sweep = coupling_sweep(panel, footprints, amplitudes, residual_ratios,
                           law_kwargs, advance=0.010)
    sweep["amplitudes"] = list(amplitudes)
    sweep["residual_ratios"] = list(residual_ratios)

    law = CrushInterfaceLaw(residual_ratio=0.5, **law_kwargs)
    foot = footprints["centre"]
    refine = mesh_refinement(panel, foot, law, 0.010,
                             gap_profile(foot, 0.004, panel.span_y))
    refine_edge = mesh_refinement(panel, footprints["edge"], law, 0.010,
                                  gap_profile(footprints["edge"], 0.004, panel.span_y))

    panel_model = PlateModel(*panel.mesh(), young=panel.young,
                             poisson=panel.poisson, thickness=panel.thickness,
                             boundary=panel.boundary)
    prescribed = prescribed_nonuniformity(
        panel, footprints["edge"], panel_model, 1.0e6,
        (0.0,) if args.quick else (0.0, 0.5, 1.0))

    study = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {"python": sys.version.split()[0],
                        "torch": torch.__version__,
                        "numpy": np.__version__},
        "verification": ver,
        "sweep": sweep,
        "mesh_refinement": {"centre_footprint": refine,
                            "edge_footprint": refine_edge},
        "prescribed_nonuniformity": prescribed,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    json_path = os.path.join(OUT_DIR, "study.json")
    with open(json_path, "w") as fh:
        json.dump(study, fh, indent=1, sort_keys=True)
    figs = figures(study, OUT_DIR)
    study["figures"] = figs
    study["wall_time_s"] = time.time() - t_start
    with open(json_path, "w") as fh:
        json.dump(study, fh, indent=1, sort_keys=True)
    print(f"wrote {json_path} in {study['wall_time_s']:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())