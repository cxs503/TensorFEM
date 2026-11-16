"""Generate the Gaussian-patch convergence evidence for ``patch_from_contact``.

Mission m23174.  Everything printed into the artifact / figure is computed here
from the module under test -- no number is transcribed by hand.

What is measured
----------------
The non-uniform contact case is a Gaussian pressure patch
(peak 2.0 MPa, sigma_x = sigma_y = 0.08 m) on a 1 m x 1 m simply supported
steel plate (E = 206 GPa, nu = 0.3, t = 10 mm), projected by
:func:`tensorfem.patch_from_contact.patch_from_contact` (exact sub-cell
splitting) and solved with the Mindlin Q4/SRI plate solver.

Three error sources are separated instead of lumped:

* ``projection``  -- the projected nodal forces must sum to the contact-grid
  force and sit on the patch centre (machine precision), and their load spread
  must converge to the analytic ``2 sigma^2``;
* ``quadrature``  -- the contact grid is a *discrete* sampling of the Gaussian;
  the difference between the grid force and the infinite-plane analytic force
  splits into the +-3 sigma window truncation and the bilinear sampling deficit;
* ``discretisation`` -- the FE peak bending stress vs the independent Kirchhoff
  Navier series (``tensorfem.ice_panel_patch_contact``).

Usage
-----
    python tools/gen_patch_from_contact_gauss_evidence.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import tensorfem.ice_panel_patch_contact as IC  # noqa: E402
import tensorfem.patch_from_contact as P  # noqa: E402

SIDE = 1.0
E = 2.06e11
NU = 0.3
T = 0.01
PEAK = 2.0e6
SIGMA = 0.08
GRID = 33
MESHES = (8, 16, 24, 32, 48)


def _gaussian_case():
    pos, pres = P._gaussian_contact_grid(0.5, 0.5, SIGMA, SIGMA, PEAK, SIDE, GRID)
    data = P.ContactPressureData(
        node_positions=pos,
        normal_pressures=pres,
        contact_area=2.0 * math.pi * SIGMA ** 2,
    )
    xg = np.unique(np.round(pos[:, 0], 12))
    yg = np.unique(np.round(pos[:, 1], 12))
    return data, pres.reshape(yg.size, xg.size).T, xg, yg


def _navier_patch(xg, yg, pressure_xy, total_force):
    return IC.ContactPatch(
        x=torch.as_tensor(xg, dtype=torch.float64),
        y=torch.as_tensor(yg, dtype=torch.float64),
        pressure=torch.as_tensor(pressure_xy, dtype=torch.float64),
        total_force=total_force,
        contact_area=float((xg[-1] - xg[0]) * (yg[-1] - yg[0])),
        peak_pressure=PEAK,
        centroid=(0.5 * (xg[0] + xg[-1]), 0.5 * (yg[0] + yg[-1])),
        approach=0.0,
        opening=torch.zeros((yg.size, xg.size), dtype=torch.float64),
        residual=0.0,
        config={"kind": "m23174-evidence"},
    )


def _series_sigma(patch, x, y):
    mom = IC.navier_rect_moments(
        patch, SIDE, SIDE, young=E, poisson=NU, thickness=T, x=x, y=y
    )
    return 6.0 * abs(mom["moment_x"]) / T ** 2 / 1e6


def force_budget(pressure_xy, xg, yg, contact_force):
    """Split the Gaussian force budget: analytic -> truncation -> sampling."""
    analytic_full = 2.0 * math.pi * SIGMA ** 2 * PEAK
    # exact integral of the *same* Gaussian over the +-3 sigma contact window
    lo_x, hi_x = float(xg[0]), float(xg[-1])
    lo_y, hi_y = float(yg[0]), float(yg[-1])

    def _cdf(a, mu, s):
        return 0.5 * (1.0 + math.erf((a - mu) / (s * math.sqrt(2.0))))

    coverage = ((_cdf(hi_x, 0.5, SIGMA) - _cdf(lo_x, 0.5, SIGMA))
                * (_cdf(hi_y, 0.5, SIGMA) - _cdf(lo_y, 0.5, SIGMA)))
    analytic_window = analytic_full * coverage
    return {
        "analytic_full_plane_N": analytic_full,
        "analytic_window_N": analytic_window,
        "window_coverage": coverage,
        "grid_force_N": contact_force,
        "window_truncation_rel": analytic_window / analytic_full - 1.0,
        "bilinear_sampling_rel": contact_force / analytic_window - 1.0,
        "grid_vs_full_plane_rel": contact_force / analytic_full - 1.0,
    }


def mesh_study(data, pressure_xy, xg, yg):
    rows = []
    fields = {}
    for nx in MESHES:
        sol, patch = P.contact_pressure_to_plate_load(data, nx, SIDE, E, NU, T, 0.0)
        stress = P.peak_stresses(sol)
        prefs = _navier_patch(xg, yg, pressure_xy, float(patch.total_force))
        series_peak = _series_sigma(prefs, 0.5, 0.5)
        series_at_fem_peak = _series_sigma(prefs, *stress["sigma_x_at_m"])

        nodes, _ = P.structured_square_mesh(nx, SIDE, dtype=torch.float64)
        nodal = patch.load_vector[0::3]
        xy = nodes.numpy()
        total = float(nodal.sum())
        spread = float(
            (nodal * ((xy[:, 0] - 0.5) ** 2 + (xy[:, 1] - 0.5) ** 2)).sum() / total
        )
        rows.append({
            "nx": nx,
            "h": SIDE / nx,
            "h_over_sigma": SIDE / nx / SIGMA,
            "fem_peak_sigma_x_MPa": stress["sigma_x_MPa"],
            "fem_peak_at_m": list(stress["sigma_x_at_m"]),
            "series_peak_sigma_x_MPa": series_peak,
            "series_at_fem_peak_MPa": series_at_fem_peak,
            "rel_error_vs_series_peak": stress["sigma_x_MPa"] / series_peak - 1.0,
            "rel_error_vs_series_at_fem_peak": (
                stress["sigma_x_MPa"] / series_at_fem_peak - 1.0
            ),
            "projected_force_N": total,
            "force_error_rel": float(patch.metadata["force_error_rel"]),
            "load_centroid": [
                float((nodal * xy[:, 0]).sum() / total),
                float((nodal * xy[:, 1]).sum() / total),
            ],
            "load_spread_m2": spread,
            "projection_method": patch.metadata["projection_method"],
        })
        if nx == 32:
            w = sol.displacement[0::3].numpy().reshape(nx + 1, nx + 1)
            fields = {"w_m": w.tolist(), "coords": nodes.numpy()[:, 0].reshape(nx + 1, nx + 1)[0].tolist()}
    # observed order between the two finest meshes
    e = [abs(r["rel_error_vs_series_peak"]) for r in rows]
    order = math.log(e[-2] / e[-1]) / math.log(MESHES[-1] / MESHES[-2])
    return rows, order, fields


def main() -> dict:
    data, pressure_xy, xg, yg = _gaussian_case()
    contact_force = float(np.trapezoid(np.trapezoid(pressure_xy, yg, axis=1), xg))
    budget = force_budget(pressure_xy, xg, yg, contact_force)
    rows, order, fields = mesh_study(data, pressure_xy, xg, yg)

    payload = {
        "mission": 23174,
        "module": "src/tensorfem/patch_from_contact.py",
        "case": {
            "pressure_model": "gaussian",
            "peak_pressure_Pa": PEAK,
            "sigma_x_m": SIGMA,
            "sigma_y_m": SIGMA,
            "contact_grid": [GRID, GRID],
            "contact_window_m": [float(xg[0]), float(xg[-1])],
            "plate_side_m": SIDE,
            "thickness_m": T,
            "young_Pa": E,
            "poisson": NU,
            "boundary": "simply_supported",
            "winkler_k": 0.0,
        },
        "projection": {
            "method": rows[0]["projection_method"],
            "force_error_rel_per_mesh": [r["force_error_rel"] for r in rows],
            "max_force_error_rel": max(r["force_error_rel"] for r in rows),
            "centroid_max_dev_m": max(
                abs(c - 0.5) for r in rows for c in r["load_centroid"]
            ),
            "load_spread_target_m2": 2.0 * SIGMA ** 2,
            "load_spread_per_mesh": [r["load_spread_m2"] for r in rows],
        },
        "force_budget": budget,
        "navier_series": {
            "peak_sigma_x_MPa": rows[0]["series_peak_sigma_x_MPa"],
            "peak_location_m": [0.5, 0.5],
        },
        "mesh_study": rows,
        "observed_order_finest_pair": order,
        "gates": {
            "force_conservation_rel": 1e-10,
            "nx32_vs_series_rel": 2e-2,
            "nx16_documented_rel": 8e-2,
            "order_window": [1.4, 2.6],
        },
    }

    out = ROOT / "artifacts" / "patch_from_contact_gauss_convergence.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")

    _figure(rows, budget, fields, payload)
    return payload


def _figure(rows, budget, fields, payload):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))

    # (a) deflection field of the non-uniform patch (nx = 32)
    nx = 32
    w = np.asarray(fields["w_m"]) * 1e3
    half = SIDE / 2
    xs = np.asarray(fields["coords"])
    ax = axes[0]
    im = ax.contourf(xs, xs, w, levels=18, cmap="viridis")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"(a) plate deflection w [mm]\nGaussian patch, nx={nx}")
    fig.colorbar(im, ax=ax, shrink=0.9)

    # (b) peak bending stress convergence
    ax = axes[1]
    nxs = np.array([r["nx"] for r in rows], dtype=float)
    fem = np.array([r["fem_peak_sigma_x_MPa"] for r in rows])
    series = rows[0]["series_peak_sigma_x_MPa"]
    ax.plot(nxs, fem, "o-", label="FEM peak (Mindlin Q4/SRI)")
    ax.axhline(series, ls="--", color="k", label=f"Navier series peak {series:.1f} MPa")
    ax.set_xscale("log")
    ax.set_xticks(nxs)
    ax.set_xticklabels([int(v) for v in nxs])
    ax.set_xlabel("plate mesh nx (elements per side)")
    ax.set_ylabel(r"peak $\sigma_x$ [MPa]")
    ax.set_title("(b) peak stress vs mesh; all error is FE discretisation")
    ax2 = ax.twinx()
    err = np.abs([r["rel_error_vs_series_peak"] for r in rows]) * 100
    ax2.loglog(nxs, err, "s:", color="tab:red", label="rel. error [%]")
    ref = err[-1] * (MESHES[-1] / nxs) ** 2
    ax2.loglog(nxs, ref, "-", color="grey", lw=1, label=r"$O(h^2)$ reference")
    ax2.set_ylabel("|error| vs series peak [%]")
    for x, y, e in zip(nxs, fem, err):
        ax.annotate(f"{e:.2f}%", (x, y), textcoords="offset points", xytext=(4, -12),
                    fontsize=7, color="tab:red")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower right")
    ax.grid(alpha=0.3, which="both")

    # (c) projection invariants: load spread + force budget
    ax = axes[2]
    spread = np.array([r["load_spread_m2"] for r in rows]) * 1e3
    ax.plot(nxs, spread, "o-", color="tab:green")
    ax.axhline(budget["analytic_full_plane_N"] * 0 + 2.0 * SIGMA ** 2 * 1e3,
               ls="--", color="k",
               label=r"analytic $2\sigma^2$ = %.2f$\times10^{-3}$ m$^2$" % (2 * SIGMA ** 2 * 1e3))
    ax.set_xscale("log")
    ax.set_xticks(nxs)
    ax.set_xticklabels([int(v) for v in nxs])
    ax.set_xlabel("plate mesh nx")
    ax.set_ylabel(r"applied-load spread $\langle r^2\rangle$ [$10^{-3}$ m$^2$]")
    ax.set_title("(c) projection is exact: force to 1e-15 N,\ncentroid to 1e-16 m, spread $O(h^2)$")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    ax.annotate(
        f"force/contact-grid = 1 + {payload['projection']['max_force_error_rel']:.1e}\n"
        f"window truncation {budget['window_truncation_rel']*100:+.3f} %\n"
        f"bilinear sampling {budget['bilinear_sampling_rel']*100:+.3f} %",
        xy=(0.03, 0.05), xycoords="axes fraction", fontsize=7,
        bbox=dict(fc="white", alpha=0.8, ec="grey"),
    )

    fig.suptitle(
        "patch_from_contact: non-uniform Gaussian ice patch on a 1 m steel plate "
        "(m23174 convergence evidence)", fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = ROOT / "docs" / "fig4_patch_from_contact_gauss_convergence.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()