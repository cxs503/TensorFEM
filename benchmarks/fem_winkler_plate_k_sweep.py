"""GAP-FEM-WINKLER — Winkler foundation parameter sweep against Hetényi.

The companion benchmark ``benchmarks/verified/fem_winkler_plate/run.py`` fixes a
single foundation modulus (buoyancy, ``k = rho g``) and sweeps the mesh.  This
benchmark exercises the *foundation coefficient* itself over three decades,
which is the part of the Winkler element that carries the ``k``-coupling
(``K_w = k * int N^T N dA``).  A wrong area/quadrature weight, a missing factor
in the element, or a unit slip does not show up at one ``k`` but does show up
as soon as ``k`` moves by 1000x, because ``k l^4 = D`` fixes the length scale.

Cases
-----
A. **Infinite plate, central point load** (quarter-symmetry Q4 Mindlin model,
   MITC shear treatment) against Hetényi's closed form
   ``w(r) = -P/(2 pi k l^2) kei(r/l)``, ``l = (D/k)^(1/4)``,
   ``w(0) = P/(8 sqrt(D k))``.  ``k`` in {1e3, 1e4, 1e5, 1e6} N/m^3 plus the
   physical buoyancy value ``rho_w g = 10055.25`` N/m^3 (4 decades).
B. **Finite simply supported beam on a Winkler foundation** (Euler-Bernoulli,
   consistent foundation matrix ``int N^T N dx``) under a central point load and
   a uniform line load, against the exact Fourier-sine solution of the same
   simply supported problem.  ``k_b`` in {1e4, 1e5, 1e6} N/m^2 (2 decades).
C. **Finite simply supported plate under uniform pressure** (Q4 Mindlin) against
   the Navier double-series solution with the foundation folded into the modal
   denominator.  ``k`` in {1e4, 1e5, 1e6} N/m^3 (2 decades).

Gates (hard, all must pass)
---------------------------
* centre deflection relative error ``<= 5 %`` in every case of A and C;
* beam sup-norm deflection error ``<= 5 %`` in every case of B;
* transverse profile relative error ``<= 5 %`` at r/l = 0.5, 1, 2, 3 (case A);
* foundation equilibrium residual ``<= 1 %`` (case A, free outer edge);
* at least three ``k`` values spanning ``>= 2`` decades in each family.

Usage
-----
    python benchmarks/fem_winkler_plate_k_sweep.py            # full sweep
    python benchmarks/fem_winkler_plate_k_sweep.py --quick    # coarse (tests)

Writes ``benchmarks/verified/fem_winkler_plate_k_sweep/result.json``,
``gates_verified.json`` and PNG figures.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from tensorfem.beam_foundation import solve_bilateral, uniform_beam
from tensorfem.foundation_plate_bending import (
    finite_plate_foundation_fem,
    infinite_plate_point_load_fem,
    navier_foundation_deflection,
    plate_rigidity,
    winkler_characteristic_length,
)
from tensorfem.hetenyi_reference import finite_beam_series, infinite_plate_point_load

HERE = Path(__file__).resolve().parent
OUT = HERE / "verified" / "fem_winkler_plate_k_sweep"

# --- plate parameters --------------------------------------------------------
# 50 mm steel plate: t/l stays below 0.04 for every k in the sweep, so the
# Mindlin-vs-Kirchhoff (transverse shear) gap is well under the 5 % gate and the
# measured error is dominated by the FE discretisation, not by model mismatch.
PLATE = dict(young=2.1e11, poisson=0.3, thickness=0.05)
POINT_LOAD = 2.0e5  # N
BUOYANCY_K = 1025.0 * 9.81  # N/m^3, rho_w * g
K_PLATE = (1.0e3, 1.0e4, BUOYANCY_K, 1.0e5, 1.0e6)
K_FINITE_PLATE = (1.0e4, 1.0e5, 1.0e6)

# --- beam parameters --------------------------------------------------------
BEAM_E = 3.0e10
BEAM_B = 1.0
BEAM_H = 0.15
BEAM_L = 20.0
BEAM_POINT_LOAD = 1.0e5  # N
BEAM_LINE_LOAD = 1.0e4  # N/m
K_BEAM = (1.0e4, 1.0e5, 1.0e6)  # N/m^2 (per unit length)

# --- finite plate parameters ------------------------------------------------
FINITE_PLATE_SIDE = 10.0
FINITE_PLATE_PRESSURE = 5.0e3  # N/m^2

TOL_PCT = 5.0
TOL_EQ_PCT = 1.0
PROFILE_STATIONS = (0.5, 1.0, 2.0, 3.0)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def rel_pct(got: float, ref: float) -> float:
    """Relative deviation in percent (reference may be negative)."""
    return abs(got - ref) / abs(ref) * 100.0


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _gate(gid, quantity, reference, value, threshold, operator, note=""):
    if operator == "<=":
        passed = value <= threshold
    elif operator == ">=":
        passed = value >= threshold
    else:  # pragma: no cover
        raise ValueError(operator)
    return {
        "id": gid,
        "quantity": quantity,
        "reference": reference,
        "value": float(value),
        "threshold": float(threshold),
        "operator": operator,
        "passed": bool(passed),
        "note": note,
    }


def decades(values) -> float:
    v = sorted(float(x) for x in values)
    return math.log10(v[-1] / v[0])


# ---------------------------------------------------------------------------
# A. infinite plate under a central point load, k swept over decades
# ---------------------------------------------------------------------------
def plate_case(k: float, n: int = 40, beta: float = 4.0,
               half_width_in_lengths: float = 10.0) -> dict:
    young, nu, thick = PLATE["young"], PLATE["poisson"], PLATE["thickness"]
    rigidity = plate_rigidity(young, nu, thick)
    ell = winkler_characteristic_length(rigidity, k)
    w0_ref = POINT_LOAD / (8.0 * math.sqrt(rigidity * k))

    t0 = time.perf_counter()
    sol = infinite_plate_point_load_fem(
        POINT_LOAD,
        young=young, poisson=nu, thickness=thick, winkler=k,
        half_width_in_lengths=half_width_in_lengths, n=n, beta=beta,
        dense_limit=3000,
    )
    elapsed = time.perf_counter() - t0

    # transverse DOFs are w, theta_x, theta_y per node -> w = u[0::3]
    w_all = sol.deflection[0::3]
    r_nodes = sol.axis_coordinates
    w_axis = w_all[: n + 1]
    ref_axis = infinite_plate_point_load(r_nodes, POINT_LOAD, rigidity, k)

    w0_fem = float(w_axis[0])
    centre_err = rel_pct(w0_fem, w0_ref)

    # profile check at fixed r/l stations (interpolated from the FE axis)
    prof_rows = []
    for xi in PROFILE_STATIONS:
        r_xi = xi * ell
        idx = int(torch.searchsorted(r_nodes.contiguous(), torch.tensor(r_xi)).item())
        idx = min(max(idx, 1), n)
        x0, x1 = float(r_nodes[idx - 1]), float(r_nodes[idx])
        w0_, w1_ = float(w_axis[idx - 1]), float(w_axis[idx])
        frac = 0.0 if x1 == x0 else (r_xi - x0) / (x1 - x0)
        w_fem_i = w0_ + frac * (w1_ - w0_)
        w_ref_i = float(infinite_plate_point_load(torch.tensor([r_xi]), POINT_LOAD, rigidity, k)[0])
        prof_rows.append({
            "r_over_l": xi,
            "w_fem_mm": w_fem_i * 1e3,
            "w_hetenyi_mm": w_ref_i * 1e3,
            # kei(r/l) crosses zero, so a pointwise *relative* error is
            # meaningless near the root: the gate uses the error normalised by
            # the centre deflection, and the plain relative value is reported
            # only where the signal is still O(w0).
            "error_over_w0_pct": abs(w_fem_i - w_ref_i) / w0_ref * 100.0,
            "error_rel_pct": rel_pct(w_fem_i, w_ref_i),
            "relative_meaningful": abs(w_ref_i) >= 0.2 * w0_ref,
        })

    # profile R^2 over the FE axis, r/l in [0.5, 3]
    mask = (r_nodes >= 0.5 * ell) & (r_nodes <= 3.0 * ell)
    wr = ref_axis[mask]
    wf = w_axis[mask]
    ss_res = float(((wf - wr) ** 2).sum())
    ss_tot = float(((wr - wr.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    eq_pct = abs(sol.foundation_reaction - sol.applied_load) / sol.applied_load * 100.0

    return {
        "k_N_m3": k,
        "characteristic_length_m": ell,
        "thickness_over_l": thick / ell,
        "w0_hetenyi_mm": w0_ref * 1e3,
        "w0_fem_mm": w0_fem * 1e3,
        "centre_error_pct": centre_err,
        "profile_error_max_pct": max(r["error_over_w0_pct"] for r in prof_rows),
        "profile_error_rel_max_pct": max(r["error_rel_pct"] for r in prof_rows),
        "profile_error_rel_max_pct_meaningful": max(
            [r["error_rel_pct"] for r in prof_rows if r["relative_meaningful"]] or [0.0]),
        "profile_r_squared": r2,
        "equilibrium_residual_pct": eq_pct,
        "n": n,
        "beta": beta,
        "ndof": int(sol.ndof),
        "n_elements": int(sol.elements.shape[0]),
        "elapsed_s": elapsed,
        "profile": prof_rows,
    }


# ---------------------------------------------------------------------------
# B. finite simply supported beam on a Winkler foundation
# ---------------------------------------------------------------------------
def beam_uniform_reference(x: torch.Tensor, L: float, EI: float, k_b: float,
                           q0: float, n_terms: int = 400) -> torch.Tensor:
    """Exact sine-series deflection of an SS beam under uniform load q0."""
    w = torch.zeros_like(x)
    for m in range(1, n_terms + 1, 2):
        mp = m * math.pi
        qm = 4.0 * q0 / mp
        w = w + (qm / (EI * (mp / L) ** 4 + k_b)) * torch.sin(mp * x / L)
    return w


def beam_case(k_b: float, kind: str, n_elem: int = 200) -> dict:
    iz = BEAM_B * BEAM_H ** 3 / 12.0
    ei = BEAM_E * iz
    lam = (k_b / (4.0 * ei)) ** 0.25
    L = BEAM_L
    nodal = None
    if kind == "point":
        nodal = torch.zeros(2 * (n_elem + 1), dtype=torch.float64)
        nodal[2 * (n_elem // 2)] = BEAM_POINT_LOAD
        q0 = 0.0
    elif kind == "uniform":
        q0 = BEAM_LINE_LOAD
    else:  # pragma: no cover
        raise ValueError(kind)

    model = uniform_beam(L, n_elem, BEAM_E, iz, k=k_b, q=q0, x0=0.0)
    res = solve_bilateral(model, nodal_loads=nodal,
                          fixed_dofs=[0, 2 * n_elem])
    w_fem = res["w_nodes"]
    x = model.x

    if kind == "point":
        w_ref = finite_beam_series(x, L, lam, k_b, ei,
                                   [(L / 2.0, BEAM_POINT_LOAD)], n_terms=200)
    else:
        w_ref = beam_uniform_reference(x, L, ei, k_b, q0)

    sup = float(w_ref.abs().max())
    err_max = float((w_fem - w_ref).abs().max()) / sup * 100.0
    mid = len(x) // 2
    err_mid = rel_pct(float(w_fem[mid]), float(w_ref[mid]))
    return {
        "k_b_N_m2": k_b,
        "load": kind,
        "lambda_inv_m": 1.0 / lam,
        "beam_length_m": L,
        "L_over_lambda_inv": L * lam,
        "w_max_fem_mm": float(w_fem.abs().max()) * 1e3,
        "w_max_ref_mm": sup * 1e3,
        "max_error_pct": err_max,
        "midspan_error_pct": err_mid,
        "n_elements": n_elem,
        "x_over_L": [float(v) for v in (x / L)],
        "w_fem_mm": [float(v) * 1e3 for v in w_fem],
        "w_ref_mm": [float(v) * 1e3 for v in w_ref],
    }


# ---------------------------------------------------------------------------
# C. finite simply supported plate under uniform pressure
# ---------------------------------------------------------------------------
def finite_plate_case(k: float, nx: int = 40, side: float = FINITE_PLATE_SIDE) -> dict:
    young, nu, thick = PLATE["young"], PLATE["poisson"], PLATE["thickness"]
    sol = finite_plate_foundation_fem(
        side_x=side, side_y=side, nx=nx, ny=nx,
        young=young, poisson=nu, thickness=thick, winkler=k,
        pressure=FINITE_PLATE_PRESSURE, dense_limit=2000,
    )
    centre = torch.tensor([[side / 2.0, side / 2.0]], dtype=torch.float64)
    w_fem = float(sol.deflection_at(centre)[0])
    w_ref = float(navier_foundation_deflection(
        centre, side_x=side, side_y=side, young=young, poisson=nu,
        thickness=thick, winkler=k, pressure=FINITE_PLATE_PRESSURE, n_modes=199,
    )[0])
    # NOTE: on a *supported* plate the edge reactions carry part of the load, so
    # |R_foundation - P| is NOT an equilibrium residual here.  Report the
    # foundation load share (0 < share <= 1) as a physical sanity number.
    share = sol.foundation_reaction / sol.applied_load
    return {
        "k_N_m3": k,
        "centre_w_fem_mm": w_fem * 1e3,
        "centre_w_navier_mm": w_ref * 1e3,
        "centre_error_pct": rel_pct(w_fem, w_ref),
        "foundation_load_share": share,
        "n": nx,
        "side_m": side,
        "ndof": int(sol.ndof),
    }


# ---------------------------------------------------------------------------
# D. regime diagnostic: thick plate in units of l (Mindlin vs Kirchhoff)
# ---------------------------------------------------------------------------
def regime_diagnostic(k: float = 1.0e6, n_levels=(20, 40, 60)) -> dict:
    """Where does the *model* (not the code) start to disagree with Hetényi?

    A 150 mm ice plate on a stiff foundation has ``t/l = 0.15``: transverse
    shear deformation is no longer negligible and the Mindlin FE answer must sit
    *above* the Kirchhoff kei solution by O((t/l)^2).  This is a property of the
    governing equations, not an implementation error, so it is reported as a
    diagnostic instead of being folded into the acceptance gates.
    """
    young, nu, thick = 3.0e9, 0.3, 0.15
    rigidity = plate_rigidity(young, nu, thick)
    ell = winkler_characteristic_length(rigidity, k)
    w0_ref = POINT_LOAD / (8.0 * math.sqrt(rigidity * k))
    rows = []
    for n in n_levels:
        sol = infinite_plate_point_load_fem(
            POINT_LOAD, young=young, poisson=nu, thickness=thick, winkler=k,
            half_width_in_lengths=10.0, n=n, beta=4.0, dense_limit=3000)
        w0 = float(sol.deflection[0])
        rows.append({"n": n, "w0_fem_mm": w0 * 1e3, "centre_error_pct": rel_pct(w0, w0_ref)})
    return {
        "k_N_m3": k,
        "thickness_over_l": thick / ell,
        "w0_hetenyi_mm": w0_ref * 1e3,
        "rows": rows,
        "note": "error grows with mesh refinement towards the converged Mindlin shear "
                "offset; expected O((t/l)^2) above the Kirchhoff solution",
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def make_figures(plate_rows, beam_rows, finite_rows):
    plt = pyplot()
    figs = {}

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for row in plate_rows:
        k = row["k_N_m3"]
        r = [p["r_over_l"] for p in row["profile"]]
        ax.plot([0.0] + r, [row["w0_fem_mm"] / row["w0_hetenyi_mm"]]
                + [p["w_fem_mm"] / row["w0_hetenyi_mm"] for p in row["profile"]],
                "-o", ms=4, lw=1.3, label=f"FEM k={k:.4g} N/m³")
        ax.plot([0.0] + r, [1.0] + [p["w_hetenyi_mm"] / row["w0_hetenyi_mm"]
                                   for p in row["profile"]],
                "k--", lw=0.8)
    ax.set_xlabel("r / l")
    ax.set_ylabel("w(r) / w(0)")
    ax.set_title("Winkler plate vs Hetényi kei: non-dimensional profile (dashed = analytic)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    figs["winkler_k_sweep_profile"] = fig

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ks = [r["k_N_m3"] for r in plate_rows]
    ax.loglog(ks, [r["centre_error_pct"] for r in plate_rows], "o-", lw=1.4,
              label="centroid deflection error")
    ax.loglog(ks, [r["profile_error_max_pct"] for r in plate_rows], "s--", lw=1.2,
              label="max profile error (r/l=0.5..3)")
    ax.loglog(ks, [max(r["equilibrium_residual_pct"], 1e-9) for r in plate_rows],
              "^:", lw=1.2, label="foundation equilibrium residual")
    ax.axhline(TOL_PCT, color="r", ls="--", lw=1.0, label="5 % acceptance")
    ax.set_xlabel("Winkler coefficient k [N/m³]")
    ax.set_ylabel("error [%]")
    ax.set_title("Plate anchor error vs foundation modulus (4 decades)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    figs["winkler_k_sweep_error"] = fig

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    for ax, kind in zip(axes, ("point", "uniform")):
        for row in [r for r in beam_rows if r["load"] == kind]:
            ax.plot(row["x_over_L"], row["w_fem_mm"], "-", lw=1.3,
                    label=f"FEM k={row['k_b_N_m2']:.3g}")
            ax.plot(row["x_over_L"][::10], row["w_ref_mm"][::10], "o", ms=3,
                    mfc="none", label=f"series k={row['k_b_N_m2']:.3g}")
        ax.set_xlabel("x / L")
        ax.set_ylabel("w [mm]")
        ax.set_title(f"Finite SS beam, {'central point load' if kind == 'point' else 'uniform load'}")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    figs["winkler_k_sweep_beam"] = fig
    return figs


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def run(quick: bool = False, out: Path | None = None) -> int:
    out_dir = Path(out) if out is not None else OUT
    n_plate = 20 if quick else 40
    n_finite = 20 if quick else 40
    n_beam = 100 if quick else 200

    print("=" * 74)
    print("GAP-FEM-WINKLER: Winkler foundation parameter sweep vs Hetényi")
    print("=" * 74)

    plate_rows = []
    for k in K_PLATE:
        row = plate_case(k, n=n_plate)
        plate_rows.append(row)
        print(f"  [plate ] k={k:>10.4g} N/m³  l={row['characteristic_length_m']:.3f} m  "
              f"w0 {row['w0_fem_mm']:.4f}/{row['w0_hetenyi_mm']:.4f} mm  "
              f"err={row['centre_error_pct']:.3f}%  "
              f"prof={row['profile_error_max_pct']:.3f}%  "
              f"eq={row['equilibrium_residual_pct']:.2e}%  t={row['elapsed_s']:.2f}s")

    beam_rows = []
    for kind in ("point", "uniform"):
        for k_b in K_BEAM:
            row = beam_case(k_b, kind, n_elem=n_beam)
            beam_rows.append(row)
            print(f"  [beam  ] {kind:<7s} k_b={k_b:>9.4g} N/m²  "
                  f"max err={row['max_error_pct']:.3f}%  "
                  f"mid err={row['midspan_error_pct']:.3f}%")

    regime = None if quick else regime_diagnostic()
    if regime is not None:
        print(f"  [diag  ] thick-plate regime t/l={regime['thickness_over_l']:.3f}: "
              + ", ".join(f"n={r['n']} err={r['centre_error_pct']:.2f}%"
                          for r in regime["rows"]))

    finite_rows = []
    for k in K_FINITE_PLATE:
        row = finite_plate_case(k, nx=n_finite)
        finite_rows.append(row)
        print(f"  [plate2] k={k:>10.4g} N/m³  uniform load  "
              f"w {row['centre_w_fem_mm']:.4f}/{row['centre_w_navier_mm']:.4f} mm  "
              f"err={row['centre_error_pct']:.3f}%")

    gates = []
    for row in plate_rows:
        tag = f"k={row['k_N_m3']:.4g}"
        gates.append(_gate(f"A_centre_{tag}", "Plate centre deflection error vs Hetényi",
                           "w(0)=P/(8√(Dk))", row["centre_error_pct"], TOL_PCT, "<=", tag))
        gates.append(_gate(f"A_profile_{tag}", "Plate profile max error vs kei(r/l)",
                           "kei solution, r/l=0.5,1,2,3", row["profile_error_max_pct"],
                           TOL_PCT, "<=", tag))
        gates.append(_gate(f"A_equilibrium_{tag}", "Foundation reaction residual",
                           "ΣR=ΣP", row["equilibrium_residual_pct"], TOL_EQ_PCT,
                           "<=", tag))
    for row in beam_rows:
        tag = f"{row['load']}_k={row['k_b_N_m2']:.4g}"
        gates.append(_gate(f"B_sup_{tag}", "Beam sup-norm deflection error vs Fourier series",
                           "simply supported series solution", row["max_error_pct"],
                           TOL_PCT, "<=", tag))
    for row in finite_rows:
        tag = f"k={row['k_N_m3']:.4g}"
        gates.append(_gate(f"C_centre_{tag}", "Finite plate centre deflection error vs Navier",
                           "Navier series", row["centre_error_pct"], TOL_PCT, "<=", tag))

    decades_plate = decades([r["k_N_m3"] for r in plate_rows])
    decades_beam = decades([r["k_b_N_m2"] for r in beam_rows])
    decades_plate2 = decades([r["k_N_m3"] for r in finite_rows])
    gates.append(_gate("D_sweep_span_plate", "Plate k sweep span", ">= 2 decades",
                       decades_plate, 2.0, ">=", f"{len(plate_rows)} values"))
    gates.append(_gate("D_sweep_span_beam", "Beam k sweep span", ">= 2 decades",
                       decades_beam, 2.0, ">=", f"{len(K_BEAM)} values"))
    gates.append(_gate("D_sweep_span_finite_plate", "Finite plate k sweep span",
                       ">= 2 decades", decades_plate2, 2.0, ">=", f"{len(finite_rows)} values"))

    all_pass = all(g["passed"] for g in gates)
    failed = [g["id"] for g in gates if not g["passed"]]

    result = {
        "benchmark": "fem_winkler_plate_k_sweep",
        "mission": "GAP-FEM-WINKLER",
        "anchor": "fem_winkler_plate_k_sweep",
        "generated_at": utc_now(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "machine": platform.machine(),
            "dtype": "torch.float64",
            "quick": quick,
        },
        "model": {
            "element": "Q4 Mindlin-Reissner + MITC shear treatment",
            "foundation_stiffness": "K_w = k * ∫ N^T N dA (2x2 Gauss)",
            "beam_element": "Euler-Bernoulli + consistent ∫ N^T N dx",
            "domain_scaling": "quarter model, half width = 10 l, graded axis (beta=4)",
        },
        "reference": {
            "external_anchor": "Hetényi, M. (1946) Beams on Elastic Foundation, "
                               "Univ. of Michigan Press — infinite plate: "
                               "w(r) = -P/(2πkl²)·kei(r/l); w(0)=P/(8√(Dk)); "
                               "finite simply supported beam: Fourier-sine (Hetényi's "
                               "method of end conditioning for SS ends).",
            "finite_plate_series": "Timoshenko & Woinowsky-Krieger (1959) Navier double "
                                   "series with foundation term k in the modal denominator",
        },
        "parameters": {
            "plate": PLATE,
            "point_load_N": POINT_LOAD,
            "k_plate_N_m3": list(K_PLATE),
            "buoyancy_k_N_m3": BUOYANCY_K,
            "beam": {"E": BEAM_E, "b": BEAM_B, "h": BEAM_H, "L": BEAM_L,
                     "point_load_N": BEAM_POINT_LOAD, "line_load_N_m": BEAM_LINE_LOAD,
                     "k_b_N_m2": list(K_BEAM)},
            "finite_plate": {"side_m": FINITE_PLATE_SIDE,
                             "pressure_N_m2": FINITE_PLATE_PRESSURE,
                             "k_N_m3": list(K_FINITE_PLATE)},
        },
        "sweep_span_decades": {
            "plate": decades_plate,
            "beam": decades_beam,
            "finite_plate": decades_plate2,
        },
        "plate_cases": plate_rows,
        "beam_cases": beam_rows,
        "finite_plate_cases": finite_rows,
        "regime_diagnostic": regime,
        "gates": gates,
        "passed": all_pass,
        "failed_gates": failed,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "result.json", result)

    gates_manifest = {
        "schema": "tensorfem.verified-gates.v1",
        "anchor": "fem_winkler_plate_k_sweep",
        "title": "Winkler foundation k-sweep (plate + beam) vs Hetényi analytical solutions",
        "mission": "GAP-FEM-WINKLER",
        "generated_at": result["generated_at"],
        "criteria": {"error": "<= 5 %", "equilibrium": "<= 1 %", "sweep": ">= 2 decades, >= 3 values"},
        "gates": gates,
        "passed": all_pass,
        "failed_gates": failed,
        "artifacts": {
            "result_json": "result.json",
            "profile_png": "winkler_k_sweep_profile.png",
            "error_png": "winkler_k_sweep_error.png",
            "beam_png": "winkler_k_sweep_beam.png",
        },
        "limitations": [
            "Quarter-symmetry plate model: central point load only (no eccentric load).",
            "Domain truncated at 10 characteristic lengths (free edge).",
            "Winkler (one-parameter) foundation only; shear-layer coupling not in this sweep.",
            "Beam family is Euler-Bernoulli (no shear deformation) — thin beam regime.",
            "Plate sweep uses a 50 mm steel plate (t/l <= 0.04); the thick-plate "
            "regime (t/l > 0.1) is documented in regime_diagnostic, where the "
            "Mindlin-vs-Kirchhoff model gap alone reaches ~5 %.",
        ],
        "external_anchor": result["reference"]["external_anchor"],
    }
    write_json(out_dir / "gates_verified.json", gates_manifest)

    figs = make_figures(plate_rows, beam_rows, finite_rows)
    plt = pyplot()
    for name, fig in figs.items():
        fig.savefig(out_dir / f"{name}.png", dpi=150)
        plt.close(fig)
        print(f"  saved {name}.png")

    print("-" * 74)
    for g in gates:
        status = "PASS" if g["passed"] else "FAIL"
        print(f"  [{status}] {g['id']:<34s} {g['value']:>12.4f} {g['operator']} {g['threshold']}")
    print("-" * 74)
    print("ALL GATES PASSED" if all_pass else f"FAILED GATES: {failed}")
    return 0 if all_pass else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="coarse meshes (test mode)")
    args = ap.parse_args(argv)
    return run(quick=args.quick)


if __name__ == "__main__":
    sys.exit(main())