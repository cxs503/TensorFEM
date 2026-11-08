"""Foundation plate bending validation run: JSON evidence + figures.

Produces ``benchmarks/foundation_plate_bending_reference.json`` and figures in
``docs/assets/foundation-plate-bending/``.  Every number in the campaign report
is emitted here programmatically (no hand-copied values).

Acceptance criteria (campaign P40-N03)

* (a) Winkler infinite plate vs Hetényi at 5 radii           -> error <= 2 %
* (b) Kerr (two-parameter) foundation vs closed form          -> error <= 2 %
* (c) buoyancy coupling ``k = rho g`` vs closed form          -> error <= 3 %
* (d) finite simply supported plate, FEM vs Navier series     -> error <= 2 %
* (e) h-refinement GCI of the centre deflection               -> GCI <= 2 %
* (f) pytest suite green (count reported separately)

Run: ``OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/foundation_plate_bending_validation.py``
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
import numpy as np
import torch

from tensorfem import __version__
from tensorfem.convergence_verification import ConvergenceLevel, evaluate_convergence_study
from tensorfem.foundation_plate_bending import (
    compare_infinite_plate_point_load,
    finite_plate_foundation_fem,
    infinite_plate_point_load_fem,
    infinite_plate_two_parameter_point_load,
    infinite_plate_winkler_point_load,
    kerr_equivalent_pasternak,
    navier_foundation_deflection,
    navier_mindlin_foundation_deflection,
    pasternak_foundation_roots,
    plate_rigidity,
    winkler_characteristic_length,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "benchmarks" / "foundation_plate_bending_reference.json"
ASSETS = ROOT / "docs" / "assets" / "foundation-plate-bending"
RADII = (0.5, 1.0, 1.5, 2.0, 3.0)

# Marine test bed: 50 mm steel plate on sea water
RHO_W, G_ACC, RHO_S = 1025.0, 9.81, 7850.0
YOUNG, POISSON, THICK = 2.1e11, 0.3, 0.05
K_BUOY = RHO_W * G_ACC
D_PLATE = plate_rigidity(YOUNG, POISSON, THICK)
L_CHAR = winkler_characteristic_length(D_PLATE, K_BUOY)
LOAD = 2.0e5
OWN_WEIGHT = RHO_S * G_ACC * THICK

FIGURES: list[Path] = []


def _save(fig: plt.Figure, name: str) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    path = ASSETS / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    FIGURES.append(path)


def hankel_oracle(r: float, load: float, d: float, k: float, g: float) -> float:
    """Independent Hankel quadrature ``P/(2 pi) int J0(a r) a /(D a^4 + G a^2 + k)``."""
    from scipy.integrate import quad
    from scipy.special import j0

    def integrand(alpha: float) -> float:
        return j0(alpha * r) * alpha / (d * alpha**4 + g * alpha**2 + k)

    a2, b2 = pasternak_foundation_roots(d, k, g)
    root_scale = max(abs(a2), abs(b2))
    cutoff = 80.0 * root_scale if r == 0.0 else max(80.0 * root_scale, 80.0 / r)
    step = math.pi / r if r > 0.0 else cutoff
    total, lo = 0.0, 0.0
    while lo < cutoff:
        hi = min(lo + step, cutoff)
        total += quad(integrand, lo, hi, limit=200)[0]
        lo = hi
    return load / (2.0 * math.pi) * total


def main() -> None:
    report: dict[str, object] = {
        "schema": "tensorfem.foundation-plate-bending/1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts/foundation_plate_bending_validation.py",
        "tensorfem_version": __version__,
        "host": {
            "platform": host.platform(),
            "machine": host.machine(),
            "python": host.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "case": {
            "description": "50 mm steel plate (E=210 GPa, nu=0.3) on a Winkler/Pasternak/Kerr "
                           "foundation and on buoyancy (rho=1025 kg/m3, g=9.81 m/s2)",
            "young": YOUNG, "poisson": POISSON, "thickness": THICK,
            "rigidity": D_PLATE,
            "winkler": K_BUOY, "rho_water": RHO_W, "gravity": G_ACC,
            "characteristic_length": L_CHAR,
            "point_load": LOAD,
            "own_weight_pressure": OWN_WEIGHT,
            "archimedes_draft": OWN_WEIGHT / K_BUOY,
        },
        "criteria": {},
        "aux": {},
    }

    # ------------------------------------------------------------------ (a)
    a = compare_infinite_plate_point_load(
        RADII, load=LOAD, young=YOUNG, poisson=POISSON, thickness=THICK,
        winkler=1.0e7, n=60, half_width_in_lengths=20.0,
    )
    ell_a = a["characteristic_length"]
    report["criteria"]["a_winkler_hetenyi"] = {
        "criterion": "FE vs Hetenyi kei solution at 5 radii, error <= 2 %",
        "winkler": 1.0e7,
        "characteristic_length": ell_a,
        "mesh": {"n": 60, "beta": 4.0, "domain_half_width": a["domain_half_width"],
                 "ndof": a["ndof"]},
        "rows": a["rows"],
        "max_rel_error": a["max_rel_error"],
        "equilibrium_rel_error": a["equilibrium_rel_error"],
        "passed": a["max_rel_error"] <= 0.02,
    }

    # mesh / truncation sensitivity (aux evidence that the agreement is converged)
    sensitivity = []
    for n, half_width in ((30, 20.0), (40, 20.0), (60, 20.0), (80, 20.0), (60, 10.0), (60, 30.0)):
        case = compare_infinite_plate_point_load(
            RADII, load=LOAD, young=YOUNG, poisson=POISSON, thickness=THICK,
            winkler=1.0e7, n=n, half_width_in_lengths=half_width,
        )
        sensitivity.append({
            "n": n, "half_width_in_lengths": half_width,
            "ndof": case["ndof"], "max_rel_error": case["max_rel_error"],
            "centre_fem": case["rows"][1]["fem"],
        })
    report["aux"]["mesh_and_truncation_sensitivity"] = sensitivity

    # ------------------------------------------------------------------ (b)
    k_upper, k_lower, g_shear = 4.0e7, 2.0e7, 6.0e7
    k_eq, g_eq = kerr_equivalent_pasternak(k_upper, k_lower, g_shear)
    b = compare_infinite_plate_point_load(
        RADII, load=LOAD, young=YOUNG, poisson=POISSON, thickness=THICK,
        winkler=k_eq, pasternak=g_eq, n=40,
    )
    roots = pasternak_foundation_roots(D_PLATE, k_eq, g_eq)
    quadrature = []
    for r_over in RADII:
        r = r_over * winkler_characteristic_length(D_PLATE, k_eq)
        closed = float(infinite_plate_two_parameter_point_load(
            torch.tensor([r]), LOAD, D_PLATE, k_eq, g_eq)[0])
        quadrature.append({
            "r_over_l": r_over, "r": r,
            "closed_form": closed, "hankel_quadrature": hankel_oracle(r, LOAD, D_PLATE, k_eq, g_eq),
        })
    for row in quadrature:
        row["rel_error"] = abs(row["closed_form"] - row["hankel_quadrature"]) / abs(row["closed_form"])
    report["criteria"]["b_kerr_two_parameter"] = {
        "criterion": "Kerr (reduced to Pasternak) FE vs closed form at 5 radii, error <= 2 %",
        "kerr": {"k_upper": k_upper, "k_lower": k_lower, "shear_modulus": g_shear},
        "equivalent_pasternak": {"k_eq": k_eq, "g_eq": g_eq},
        "foundation_roots": [{"re": r.real, "im": r.imag} for r in roots],
        "characteristic_length": winkler_characteristic_length(D_PLATE, k_eq),
        "mesh": {"n": 40, "beta": 4.0, "ndof": b["ndof"]},
        "rows": b["rows"],
        "max_rel_error": b["max_rel_error"],
        "equilibrium_rel_error": b["equilibrium_rel_error"],
        "closed_form_vs_hankel_quadrature": quadrature,
        "closed_form_max_rel_error": max(q["rel_error"] for q in quadrature),
        "passed": b["max_rel_error"] <= 0.02,
    }

    # ------------------------------------------------------------------ (c)
    buoyancy = infinite_plate_point_load_fem(
        LOAD, young=YOUNG, poisson=POISSON, thickness=THICK,
        winkler=K_BUOY, pressure=OWN_WEIGHT, n=60, half_width_in_lengths=20.0,
    )
    radii = torch.tensor(RADII, dtype=torch.float64) * L_CHAR
    points = torch.stack((radii, torch.zeros_like(radii)), dim=1)
    fem_buoyancy = buoyancy.deflection_at(points)
    ana_buoyancy = (
        infinite_plate_winkler_point_load(radii, LOAD, D_PLATE, K_BUOY) + OWN_WEIGHT / K_BUOY
    )
    rows_c = [
        {
            "r_over_l": r_over, "r": float(r),
            "analytic": float(an), "fem": float(f),
            "rel_error": float(abs(f - an) / an),
        }
        for r_over, r, an, f in zip(RADII, radii, ana_buoyancy, fem_buoyancy)
    ]
    still = infinite_plate_point_load_fem(
        0.0, young=YOUNG, poisson=POISSON, thickness=THICK,
        winkler=K_BUOY, pressure=OWN_WEIGHT, n=40,
    )
    draft_points = torch.tensor([[0.0, 0.0], [2.0 * L_CHAR, 0.0], [8.0 * L_CHAR, 8.0 * L_CHAR]])
    draft_fem = still.deflection_at(draft_points)
    report["criteria"]["c_buoyancy"] = {
        "criterion": "buoyancy coupling k = rho g vs Hetenyi superposition at 5 radii, error <= 3 %",
        "winkler": K_BUOY,
        "characteristic_length": L_CHAR,
        "own_weight_pressure": OWN_WEIGHT,
        "mesh": {"n": 60, "beta": 4.0, "ndof": buoyancy.ndof},
        "rows": rows_c,
        "max_rel_error": max(row["rel_error"] for row in rows_c),
        "equilibrium_rel_error": abs(buoyancy.foundation_reaction / buoyancy.applied_load - 1.0),
        "archimedes": {
            "draft_analytic": OWN_WEIGHT / K_BUOY,
            "draft_archimedes_rho_s_t_over_rho_w": RHO_S * THICK / RHO_W,
            "draft_fem": [float(v) for v in draft_fem],
            "points": [[float(p[0]), float(p[1])] for p in draft_points],
            "max_rel_error": max(
                abs(float(v) - OWN_WEIGHT / K_BUOY) / (OWN_WEIGHT / K_BUOY) for v in draft_fem
            ),
        },
        "passed": max(row["rel_error"] for row in rows_c) <= 0.03,
    }

    # ------------------------------------------------------------------ (d)
    side = 1.0
    q0 = 1.0e5
    kirch_winkler = 2.0e7
    centre = torch.tensor([[0.5 * side, 0.5 * side]])
    d_rows = []
    for thickness in (THICK, 0.02, 0.01):
        kirchhoff = float(navier_foundation_deflection(
            centre, side_x=side, side_y=side, young=YOUNG, poisson=POISSON,
            thickness=thickness, pressure=q0, winkler=kirch_winkler, n_modes=299)[0])
        mindlin = float(navier_mindlin_foundation_deflection(
            centre, side_x=side, side_y=side, young=YOUNG, poisson=POISSON,
            thickness=thickness, pressure=q0, winkler=kirch_winkler, n_modes=299)[0])
        fem = float(finite_plate_foundation_fem(
            side_x=side, side_y=side, nx=32, ny=32, young=YOUNG, poisson=POISSON,
            thickness=thickness, pressure=q0, winkler=kirch_winkler,
        ).deflection_at(centre)[0])
        d_rows.append({
            "thickness": thickness, "span_over_thickness": side / thickness,
            "navier_kirchhoff": kirchhoff, "navier_mindlin": mindlin, "fem": fem,
            "fem_vs_mindlin": abs(fem - mindlin) / mindlin,
            "fem_vs_kirchhoff": abs(fem - kirchhoff) / kirchhoff,
        })
    report["criteria"]["d_finite_plate_navier"] = {
        "criterion": "SSSS plate on Winkler foundation, FEM (32x32) vs Navier series, error <= 2 %",
        "side": side, "uniform_pressure": q0, "winkler": kirch_winkler,
        "rows": d_rows,
        "max_rel_error_vs_mindlin": max(row["fem_vs_mindlin"] for row in d_rows),
        "max_rel_error_vs_kirchhoff_thin": d_rows[-1]["fem_vs_kirchhoff"],
        "passed": max(row["fem_vs_mindlin"] for row in d_rows) <= 0.02,
    }

    # ------------------------------------------------------------------ (e)
    reference = float(navier_mindlin_foundation_deflection(
        centre, side_x=side, side_y=side, young=YOUNG, poisson=POISSON,
        thickness=THICK, pressure=q0, winkler=kirch_winkler, n_modes=399)[0])
    levels, gci_rows = [], []
    for n in (8, 16, 32, 64):
        value = float(finite_plate_foundation_fem(
            side_x=side, side_y=side, nx=n, ny=n, young=YOUNG, poisson=POISSON,
            thickness=THICK, pressure=q0, winkler=kirch_winkler,
        ).deflection_at(centre)[0])
        levels.append(ConvergenceLevel(1.0 / n, value, f"n={n}"))
        gci_rows.append({"n": n, "h": 1.0 / n, "centre_deflection": value,
                         "rel_error_vs_series": abs(value - reference) / reference})
    study = evaluate_convergence_study(
        levels, qoi="centre deflection", expected_order=2.0, reference_value=reference
    )
    report["criteria"]["e_gci"] = {
        "criterion": "h-refinement (n=8,16,32) GCI of the centre deflection, GCI <= 2 %",
        "levels": gci_rows,
        "navier_mindlin_reference": reference,
        "finest_order": study["finest_order"],
        "expected_order": study["expected_order"],
        "gci": study["gci"],
        "richardson_vs_series": abs(study["gci"]["richardson_value"] - reference) / reference,
        "status": study["status"],
        "passed": study["passed"] is True and study["gci"]["relative_gci"] <= 0.02,
    }

    # ------------------------------------------------------------------ aux: profiles
    profile_r = np.linspace(0.0, 6.0, 121) * L_CHAR
    profile_r_t = torch.tensor(profile_r, dtype=torch.float64)
    profiles = {
        "r_over_l": (profile_r / L_CHAR).tolist(),
        "hetenyi_winkler": infinite_plate_winkler_point_load(
            profile_r_t, LOAD, D_PLATE, K_BUOY).tolist(),
        "pasternak_g_rho_g_l2": infinite_plate_two_parameter_point_load(
            profile_r_t, LOAD, D_PLATE, K_BUOY, K_BUOY * L_CHAR**2).tolist(),
    }
    fem_profile = infinite_plate_point_load_fem(
        LOAD, young=YOUNG, poisson=POISSON, thickness=THICK,
        winkler=K_BUOY, n=60, half_width_in_lengths=20.0)
    probe = torch.tensor(np.linspace(0.0, 6.0, 61) * L_CHAR, dtype=torch.float64)
    profiles["fem_winkler_points"] = {
        "r_over_l": (probe.numpy() / L_CHAR).tolist(),
        "w": fem_profile.deflection_at(
            torch.stack((probe, torch.zeros_like(probe)), dim=1)).tolist(),
    }
    # Kerr parameter sweep at a fixed r: how the shear layer flattens the profile
    sweep = []
    for g_over in (0.0, 0.25, 1.0, 4.0, 16.0):
        g_val = g_over * K_BUOY * L_CHAR**2
        sweep.append({
            "g_over_k_l2": g_over,
            "shear_modulus": g_val,
            "w_at_0": float(infinite_plate_two_parameter_point_load(
                torch.zeros(1), LOAD, D_PLATE, K_BUOY, g_val)[0]),
            "w_at_3l": float(infinite_plate_two_parameter_point_load(
                torch.tensor([3.0 * L_CHAR]), LOAD, D_PLATE, K_BUOY, g_val)[0]),
            "uplift_ring": bool(float(infinite_plate_two_parameter_point_load(
                torch.linspace(0.0, 12.0, 2401) * L_CHAR, LOAD, D_PLATE, K_BUOY, g_val).min()) < 0.0),
        })
    report["aux"]["profiles"] = profiles
    report["aux"]["kerr_shear_sweep"] = sweep

    # ------------------------------------------------------------- figures
    _figure_profiles(profiles, sweep)
    _figure_errors(a, b, rows_c)
    _figure_field(fem_profile)
    _figure_convergence(gci_rows, reference)

    report["figures"] = [
        {"path": str(p.relative_to(ROOT)), "name": p.name} for p in FIGURES
    ]
    report["criterion_all_passed"] = all(
        report["criteria"][key]["passed"] for key in
        ("a_winkler_hetenyi", "b_kerr_two_parameter", "c_buoyancy",
         "d_finite_plate_navier", "e_gci")
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "output": str(OUTPUT.relative_to(ROOT)),
        "figures": [str(p.relative_to(ROOT)) for p in FIGURES],
        "a_max_rel_error": report["criteria"]["a_winkler_hetenyi"]["max_rel_error"],
        "b_max_rel_error": report["criteria"]["b_kerr_two_parameter"]["max_rel_error"],
        "c_max_rel_error": report["criteria"]["c_buoyancy"]["max_rel_error"],
        "d_max_rel_error_vs_mindlin": report["criteria"]["d_finite_plate_navier"]["max_rel_error_vs_mindlin"],
        "e_gci": report["criteria"]["e_gci"]["gci"]["relative_gci"],
        "all_passed": report["criterion_all_passed"],
    }, indent=2))


# ---------------------------------------------------------------- plotting
def _figure_profiles(profiles: dict, sweep: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
    ax = axes[0]
    r = np.array(profiles["r_over_l"])
    scale = 1.0e3
    ax.plot(r, np.array(profiles["hetenyi_winkler"]) * scale, lw=1.8,
            label="Hetényi (Winkler, closed form)")
    ax.plot(r, np.array(profiles["pasternak_g_rho_g_l2"]) * scale, lw=1.8, ls="--",
            label="two-parameter G = k l² (closed form)")
    ax.plot(np.array(profiles["fem_winkler_points"]["r_over_l"]),
            np.array(profiles["fem_winkler_points"]["w"]) * scale, "o", ms=3.5,
            label="FE, 60×60 graded quarter model")
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.set_xlabel("r / l   (l = (D/k)$^{1/4}$)")
    ax.set_ylabel("deflection w  [mm]  (P = 200 kN)")
    ax.set_title("Infinite plate on a buoyant (Winkler) foundation")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    g_over = [row["g_over_k_l2"] for row in sweep]
    ax.semilogx(np.array(g_over) + 1e-3, np.array([row["w_at_0"] for row in sweep]) * 1e3,
                "s-", label="w(0), centre")
    ax.semilogx(np.array(g_over) + 1e-3, np.array([row["w_at_3l"] for row in sweep]) * 1e3,
                "^-", label="w(3l), outer ring")
    for row in sweep:
        if not row["uplift_ring"]:
            ax.axvline(row["g_over_k_l2"] + 1e-3, color="0.85", ls=":", lw=0.8)
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.set_xlabel("foundation shear modulus  G / (k l$^2$)   (1e-3 marks G = 0)")
    ax.set_ylabel("deflection [mm]")
    ax.set_title("Kerr/Pasternak shear layer: uplift ring is suppressed")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "profiles.png")


def _figure_errors(a: dict, b: dict, rows_c: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    r = [row["r_over_l"] for row in a["rows"]]
    ax.loglog(r, [max(row["rel_error"], 1e-6) for row in a["rows"]], "o-",
              label=f"(a) Winkler vs Hetényi (max {100*a['max_rel_error']:.2f} %)")
    ax.loglog(r, [max(row["rel_error"], 1e-6) for row in b["rows"]], "s-",
              label=f"(b) Kerr vs closed form (max {100*b['max_rel_error']:.2f} %)")
    ax.loglog([row["r_over_l"] for row in rows_c],
              [max(row["rel_error"], 1e-6) for row in rows_c], "^-",
              label=f"(c) buoyancy (max {100*max(r['rel_error'] for r in rows_c):.3f} %)")
    ax.axhline(0.02, color="tab:red", ls="--", lw=1.0, label="2 % acceptance")
    ax.axhline(0.03, color="tab:purple", ls=":", lw=1.0, label="3 % acceptance (buoyancy)")
    ax.set_xlabel("r / l")
    ax.set_ylabel("relative error  |w$_{FE}$ - w$_{ref}$| / w$_{ref}$")
    ax.set_title("FE vs analytic references (60×60 / 40×40 graded quarter models)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "error_vs_radius.png")


def _figure_field(solution) -> None:
    n = solution.nx
    x = solution.nodes[: n + 1, 0].numpy()
    y = solution.nodes[:: n + 1, 1].numpy()
    w = solution.deflection[0::3].reshape(n + 1, n + 1).numpy()
    # plot the full plate by mirroring the quarter model
    xx = np.concatenate((-x[::-1], x[1:]))
    ww = np.concatenate((w[::-1, :][:, ::-1], w[:, 1:]), axis=1)
    yy = y
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
    ax = axes[0]
    levels = np.linspace(float(ww.min()), float(np.percentile(ww, 99.5)), 40)
    cf = ax.contourf(xx, yy, ww * 1e3, levels=levels, cmap="viridis", extend="both")
    ax.contour(xx, yy, ww * 1e3, levels=levels[::4], colors="w", linewidths=0.5, alpha=0.6)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("FE deflection field w [mm], buoyant plate, central 200 kN load")
    fig.colorbar(cf, ax=ax, label="w [mm]")

    ax = axes[1]
    ax.plot(x / L_CHAR, np.asarray(w)[:, 0] * 1e3, "o-", ms=2.5, label="FE (axis cut, y = 0)")
    probe = torch.tensor(np.linspace(0.0, 6.0, 121) * L_CHAR, dtype=torch.float64)
    analytic = infinite_plate_winkler_point_load(probe, LOAD, D_PLATE, K_BUOY) + OWN_WEIGHT / K_BUOY
    ax.plot(probe.numpy() / L_CHAR, analytic.numpy() * 1e3, "-", lw=1.5,
            label="Hetényi + Archimedes draft")
    ax.set_xlim(0.0, 6.0)
    ax.set_xlabel("r / l")
    ax.set_ylabel("w [mm]")
    ax.set_title("Axis cut: buoyant plate, own weight + 200 kN")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "field_and_axis.png")


def _figure_convergence(rows: list[dict], reference: float) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    h = np.array([row["h"] for row in rows])
    err = np.array([abs(row["centre_deflection"] - reference) / reference for row in rows])
    ax.loglog(h, np.maximum(err, 1e-10), "o-", label="FE centre deflection error")
    ax.loglog(h, err[0] * (h / h[0]) ** 2, "k--", lw=1.0, label="slope 2 (O(h²)) reference")
    ax.set_xlabel("mesh size h = 1/n")
    ax.set_ylabel("relative error vs Mindlin Navier series")
    ax.set_title("h-refinement of the SS foundation plate (n = 8, 16, 32, 64)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "convergence.png")


if __name__ == "__main__":
    main()