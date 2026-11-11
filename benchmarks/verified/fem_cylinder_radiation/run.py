#!/usr/bin/env python3
"""Independent verification surface for ``tensorfem.struct_acoustics.cylinder_radiation``.

Verification debt item #3 of the m14722 census: the module has a pytest gate
suite, but no standalone ``benchmarks/verified/`` anchor with a graded gate
manifest, a convergence ladder and figures.  This driver re-derives every
anchor from the literature (Junger & Feit 1992; standard multipole asymptotes)
and grades the production code against it.

Anchors (all closed-form, no hand-copied numbers):

A1  low-``ka`` piston law, ring mode n=0:  sigma -> (pi/2) k R
A2  high-``ka`` plateau:  sigma_0 -> 1,  sigma_n -> 1/2  (n >= 1)
A3  evanescent cutoff:  k_x > k collapses the conductance by orders of magnitude
A4  window closure:  finite-cylinder mode power -> infinite-cylinder modal value
A5  window convergence controls (sampling density, band width)
A6  ESM pulsating sphere:  Pi = 2 pi a^2 rho c V^2 (ka)^2 / (1 + (ka)^2)
A7  compact-body asymptote reproduces the pulsating sphere (dipole = 0)
A8  negative control:  rigid body (v_n = 0) radiates nothing
A9  quadratic velocity scaling  Pi ~ |V|^2
A10 axial-spectrum closed form vs dense quadrature
A11 Fourier mode identification isolates the excited (m, n) pair
A12 surface-grid geometry:  lateral area = 2 pi R L, caps carry zero weight

Run:  PYTHONPATH=src python benchmarks/verified/fem_cylinder_radiation/run.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

import os  # noqa: E402

torch.set_num_threads(min(8, os.cpu_count() or 8))

from tensorfem.struct_acoustics.cylinder_radiation import (  # noqa: E402
    axial_spectrum,
    compact_body_power,
    cylinder_surface_grid,
    equivalent_source_power,
    finite_cylinder_mode_power,
    fourier_cylinder_mode,
    infinite_cylinder_conductance,
    infinite_cylinder_efficiency,
)

RHO = 1.225
C = 343.0
RADIUS = 0.5
LENGTH = 2.0


def _lateral(n_axial=17, n_circ=32):
    points, normals, weights, meta = cylinder_surface_grid(
        radius=RADIUS, length=LENGTH, n_axial=n_axial, n_circ=n_circ
    )
    n_lateral = int(meta[0]) * int(meta[1])
    theta = torch.atan2(points[:n_lateral, 2], points[:n_lateral, 1])
    return points, normals, weights, n_lateral, theta


def _sphere(radius, n_theta=13, n_phi=24):
    theta = torch.linspace(0.0, math.pi, n_theta + 2, dtype=torch.float64)[1:-1]
    phi = torch.linspace(0.0, 2.0 * math.pi, n_phi + 1, dtype=torch.float64)[:-1]
    th, ph = torch.meshgrid(theta, phi, indexing="ij")
    th, ph = th.reshape(-1), ph.reshape(-1)
    normals = torch.stack(
        (torch.sin(th) * torch.cos(ph), torch.sin(th) * torch.sin(ph), torch.cos(th)),
        dim=1,
    )
    return radius * normals, normals


def main() -> dict:
    gates: list[dict] = []

    def gate(gid: str, quantity: str, reference: str, value: float,
             threshold: float, operator: str = "<=") -> None:
        passed = value <= threshold if operator == "<=" else value >= threshold
        gates.append({
            "id": gid, "quantity": quantity, "reference": reference,
            "value": value, "threshold": threshold, "operator": operator,
            "passed": bool(passed),
        })
        print(f"{gid}  {quantity:52s} {value:.6g}  {operator} {threshold:g}  "
              f"{'PASS' if passed else 'FAIL'}")

    # ------------------------------------------------------------------ A1
    a1_errs = []
    for ka in (1e-4, 1e-3, 1e-2):
        k = ka / RADIUS
        sigma = infinite_cylinder_efficiency(0, k, 0.0, radius=RADIUS, rho=RHO, c=C)
        a1_errs.append(abs(sigma / (0.5 * math.pi * ka) - 1.0))
    gate("G1_piston_law", "max |sigma/((pi/2)ka) - 1|, ka in {1e-4,1e-3,1e-2}",
         "sigma -> (pi/2) k R  (Junger & Feit 1992, low-ka piston law)",
         max(a1_errs), 5e-3)

    # ------------------------------------------------------------------ A2
    a2_errs = []
    sigma0 = infinite_cylinder_efficiency(0, 20.0 / RADIUS, 0.0,
                                           radius=RADIUS, rho=RHO, c=C)
    a2_errs.append(abs(sigma0 - 1.0))
    for n in (1, 2, 3):
        s = infinite_cylinder_efficiency(n, 20.0 / RADIUS, 0.0,
                                         radius=RADIUS, rho=RHO, c=C)
        a2_errs.append(abs(s - 0.5))
    gate("G2_high_ka_plateau", "max |sigma - plateau| at ka = 20 (n = 0..3)",
         "sigma_0 -> 1, sigma_n -> 1/2  (high-ka plateau, <cos^2 n theta> = 1/2)",
         max(a2_errs), 0.02)

    # ------------------------------------------------------------------ A3
    k = 1.0 / RADIUS
    base = infinite_cylinder_conductance(2, k, 0.0, radius=RADIUS, rho=RHO, c=C)
    evan = [abs(infinite_cylinder_conductance(2, k, f * k,
                                               radius=RADIUS, rho=RHO, c=C))
            for f in (1.05, 1.5, 3.0)]
    gate("G3_evanescent_cutoff", "max collapse ratio |G(k_x> k)| / |G(0)|",
         "evanescent decay for k_x > k (Hankel exponential decay)",
         max(evan) / base, 1e-3)

    # ------------------------------------------------------------------ A4
    n, m = 2, 1
    length = 400.0
    freq = 60.0
    k_axial = m * math.pi / length
    windowed = finite_cylinder_mode_power(
        m, n, radius=RADIUS, length=length, frequency=freq,
        rho=RHO, c=C, samples=8001,
    )
    analytic = infinite_cylinder_efficiency(n, 2.0 * math.pi * freq / C, k_axial,
                                            radius=RADIUS, rho=RHO, c=C)
    gate("G4_window_closure", "finite/infinite-cylinder sigma ratio - 1",
         "window closure: Pi = (1/2pi) int G_n(k_x)|V(k_x)|^2 dk_x",
         abs(windowed.sigma / analytic - 1.0), 0.02)

    # ------------------------------------------------------------------ A5
    coarse = finite_cylinder_mode_power(
        1, 2, radius=RADIUS, length=LENGTH, frequency=150.0,
        rho=RHO, c=C, samples=2001, half_width=8.0,
    )
    fine = finite_cylinder_mode_power(
        1, 2, radius=RADIUS, length=LENGTH, frequency=150.0,
        rho=RHO, c=C, samples=8001, half_width=16.0,
    )
    gate("G5_window_convergence", "coarse/fine sigma ratio - 1",
         "sampling 2001->8001, band 8->16 half-waves",
         abs(coarse.sigma / fine.sigma - 1.0), 5e-3)

    # ------------------------------------------------------------------ A6
    a6_errs = []
    sphere_pts, sphere_nrm = _sphere(0.5)
    vel = torch.ones(sphere_pts.shape[0], dtype=torch.float64)
    a6_ka, a6_ratio = [], []
    for ka in (0.5, 1.0, 2.0, 3.0):
        res = equivalent_source_power(
            sphere_pts, sphere_nrm, vel,
            frequency=ka * C / (2.0 * math.pi * 0.5), rho=RHO, c=C,
            regularization=1e-6,
        )
        exact = 2.0 * math.pi * 0.5 ** 2 * RHO * C * ka ** 2 / (1.0 + ka ** 2)
        a6_errs.append(abs(res.power / exact - 1.0))
        a6_ka.append(ka)
        a6_ratio.append(res.power / exact)
    gate("G6_esm_pulsating_sphere", "max |Pi_ESM/Pi_exact - 1|, ka in {0.5,1,2,3}",
         "Pi = 2 pi a^2 rho c V^2 (ka)^2 / (1 + (ka)^2)  (pulsating sphere, exact)",
         max(a6_errs), 0.01)

    # ------------------------------------------------------------------ G7
    pts, nrm = _sphere(0.5)
    wts = torch.full((pts.shape[0],), 4.0 * math.pi * 0.5 ** 2 / pts.shape[0],
                     dtype=torch.float64)
    v1 = torch.ones(pts.shape[0], dtype=torch.float64)
    ka7 = 0.1
    power7, q7, d7 = compact_body_power(
        pts, nrm, v1, wts, frequency=ka7 * C / (2.0 * math.pi * 0.5),
        rho=RHO, c=C,
    )
    exact7 = 2.0 * math.pi * 0.5 ** 2 * RHO * C * ka7 ** 2 / (1.0 + ka7 ** 2)
    gate("G7_compact_body_asymptote", "|Pi_mono+Pi_dip / Pi_exact - 1| at ka = 0.1",
         "monopole + dipole asymptote, dipole = 0 for breathing sphere",
         abs(power7 / exact7 - 1.0), 0.02)
    gate("G7b_dipole_null", "|dipole moment| for breathing sphere",
         "D = 0 by symmetry (v_n uniform)", abs(d7), 1e-12)

    # ------------------------------------------------------------------ G8
    lp, ln, lw, nl, _ = _lateral(n_axial=9, n_circ=16)
    zero = torch.zeros(lp.shape[0], dtype=torch.float64)
    rz = equivalent_source_power(lp, ln, zero, frequency=120.0, rho=RHO, c=C,
                                area_weights=lw)
    gate("G8_rigid_body_null", "|Pi| for v_n = 0 (negative control)",
         "rigid body radiates nothing", abs(rz.power), 1e-30)

    # ------------------------------------------------------------------ G9
    base = finite_cylinder_mode_power(1, 1, radius=RADIUS, length=LENGTH,
                                      frequency=100.0, rho=RHO, c=C,
                                      velocity_amplitude=1.0, samples=2001)
    dbl = finite_cylinder_mode_power(1, 1, radius=RADIUS, length=LENGTH,
                                     frequency=100.0, rho=RHO, c=C,
                                     velocity_amplitude=2.0, samples=2001)
    gate("G9_velocity_squared_scaling", "Pi(2V)/Pi(V) - 4",
         "Pi ~ |V|^2 (linearity of the exterior problem)",
         abs(dbl.power / base.power - 4.0), 1e-9)

    # ------------------------------------------------------------------ G10
    length10, m10 = 2.0, 2
    a10_errs = []
    grid = torch.linspace(0.0, length10, 400001, dtype=torch.float64)
    for kx in (0.3, 1.0, 4.0):
        f = torch.sin(m10 * math.pi * grid / length10) * torch.exp(-1j * kx * grid)
        numeric = complex(torch.trapezoid(f, grid))
        closed = complex(axial_spectrum(torch.tensor([kx], dtype=torch.float64),
                                        length10, m10)[0])
        a10_errs.append(abs(closed - numeric) / abs(numeric))
    gate("G10_axial_spectrum_closed_form", "max rel. error vs 4e5-point quadrature",
         "q (1 - (-1)^m e^{-i k_x L}) / (q^2 - k_x^2),  q = m pi / L",
         max(a10_errs), 1e-4)

    # ------------------------------------------------------------------ G11
    lp, ln, lw, nl, theta = _lateral()
    m11, n11 = 2, 3
    v11 = torch.sin(m11 * math.pi * lp[:nl, 0] / LENGTH) * torch.cos(n11 * theta)
    coeffs, dominant, share = fourier_cylinder_mode(
        lp[:nl, 0], theta, v11, length=LENGTH, axial_modes=4, circ_modes=6,
    )
    gate("G11_mode_identification", "energy share of excited (m, n) = (2, 3)",
         "Fourier decomposition isolates the injected mode", float(share), 0.95, ">=")
    gate("G11b_dominant_pair", "dominant pair == (2, 3)",
         "planted harmonic (2, 3)", 0.0 if dominant == (m11, n11) else 1.0, 0.0)

    # ------------------------------------------------------------------ G12
    lat_area = float(lw[:nl].sum())
    cap_area = float(lw[nl:].sum())
    gate("G12_grid_lateral_area", "|sum w_lateral - 2 pi R L| / (2 pi R L)",
         "lateral surface area of the cylinder",
         abs(lat_area - 2.0 * math.pi * RADIUS * LENGTH)
         / (2.0 * math.pi * RADIUS * LENGTH), 1e-12)
    gate("G12b_grid_caps_zero", "cap weight sum (rigid matching points)",
         "caps carry zero area weight", cap_area, 1e-14)

    # ------------------------------------------------------------------ figs
    _figures(a6_ka, a6_ratio)

    # ------------------------------------------------------------------ out
    passed = all(g["passed"] for g in gates)
    manifest = {
        "schema": "tensorfem.verified-gates.v1",
        "anchor": "fem_cylinder_radiation",
        "module": "tensorfem.struct_acoustics.cylinder_radiation",
        "criteria": {
            "G1": "<= 5e-3", "G2": "<= 0.02", "G3": "<= 1e-3",
            "G4": "<= 0.02", "G5": "<= 5e-3", "G6": "<= 0.01",
            "G7": "<= 0.02", "G7b": "<= 1e-12", "G8": "<= 1e-30",
            "G9": "<= 1e-9", "G10": "<= 1e-4", "G11": ">= 0.95",
            "G11b": "<= 0.0", "G12": "<= 1e-12", "G12b": "<= 1e-14",
        },
        "gates": gates,
        "passed": bool(passed),
        "limitations": [
            "Donnell roots are a branch-structure anchor only (lowest root is "
            "thickness-independent); they are not the shell frequency reference.",
            "ESM uses a shrunk interior source ring (shrink=0.6) with Tikhonov "
            "regularization 1e-6; the pulsating-sphere check is the calibration.",
            "Window closure integrates the infinite-cylinder conductance over a "
            "finite band; the band width and sampling are convergence-controlled "
            "(G5), not a closed-form identity.",
        ],
    }
    (HERE / "gates_verified.json").write_text(json.dumps(manifest, indent=2) + "\n")
    result = {
        "anchor": "fem_cylinder_radiation",
        "passed": bool(passed),
        "gates": {g["id"]: g["value"] for g in gates},
        "piston_law_errors": a1_errs,
        "esm_sphere_ratios": a6_ratio,
        "esm_sphere_ka": a6_ka,
        "window_closure_ratio": windowed.sigma / analytic,
        "window_coarse_over_fine": coarse.sigma / fine.sigma,
        "velocity_scaling_ratio": dbl.power / base.power,
        "mode_share": float(share),
        "dominant_pair": list(dominant),
    }
    (HERE / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nanchor passed = {passed}")
    return result


def _figures(a6_ka: list, a6_ratio: list) -> None:
    import numpy as np

    figs = HERE / "figs"
    figs.mkdir(exist_ok=True)

    # fig 1: radiation efficiency vs ka for n = 0, 1, 2 with asymptotes
    ka = np.logspace(-3, 1.3, 240)
    ks = ka / RADIUS
    sig = {n: [infinite_cylinder_efficiency(n, float(k), 0.0,
                                            radius=RADIUS, rho=RHO, c=C)
            for k in ks] for n in (0, 1, 2)}
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=140)
    for n, lab in ((0, "n = 0 (ring)"), (1, "n = 1"), (2, "n = 2")):
        ax.loglog(ka, sig[n], label=lab)
    ax.loglog(ka, 0.5 * np.pi * ka, "k:", label="piston law (pi/2)ka")
    ax.axhline(1.0, color="0.5", ls="--", lw=0.8)
    ax.axhline(0.5, color="0.5", ls=":", lw=0.8)
    ax.set_xlabel("ka = kR")
    ax.set_ylabel("radiation efficiency sigma")
    ax.set_title("Infinite-cylinder modal efficiency vs ka")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figs / "efficiency_vs_ka.png")
    plt.close(fig)

    # fig 2: ESM pulsating sphere vs exact
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=140)
    ax.plot(a6_ka, a6_ratio, "o-", label="ESM / exact")
    ax.axhline(1.0, color="0.5", ls="--", lw=0.8)
    ax.set_xlabel("ka")
    ax.set_ylabel("Pi_ESM / Pi_exact")
    ax.set_title("ESM pulsating-sphere power vs exact")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figs / "esm_pulsating_sphere.png")
    plt.close(fig)

    # fig 3: window closure convergence
    ratios = []
    for smp in (1001, 2001, 4001, 8001):
        r = finite_cylinder_mode_power(1, 2, radius=RADIUS, length=LENGTH,
                                        frequency=150.0, rho=RHO, c=C,
                                        samples=smp, half_width=16.0)
        ratios.append(r.sigma)
    ref = ratios[-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=140)
    ax.semilogx([1001, 2001, 4001, 8001], [x / ref for x in ratios], "o-")
    ax.axhline(1.0, color="0.5", ls="--", lw=0.8)
    ax.set_xlabel("spectral samples")
    ax.set_ylabel("sigma / sigma(8001)")
    ax.set_title("Window-closure sampling convergence")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(figs / "window_closure_convergence.png")
    plt.close(fig)

    # fig 4: evanescent decay
    k = 1.0 / RADIUS
    fac = np.linspace(0.0, 3.0, 121)
    ev = [abs(infinite_cylinder_conductance(2, k, float(f) * k,
                                             radius=RADIUS, rho=RHO, c=C))
          for f in fac]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=140)
    ax.semilogy(fac, ev, "-")
    ax.axvline(1.0, color="r", ls="--", lw=0.8, label="cutoff k_x = k")
    ax.set_xlabel("k_x / k")
    ax.set_ylabel("|G_2| (kg/(m s))")
    ax.set_title("Evanescent decay beyond the axial cutoff")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figs / "evanescent_decay.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
