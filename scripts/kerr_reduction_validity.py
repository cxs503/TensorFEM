"""Validity study of the Kerr -> Pasternak reduction (``kerr_equivalent_pasternak``).

Runs the exact two-field Kerr plate model (reference) against the production
single-field reduction and writes

* ``artifacts/kerr_reduction_validity.json`` -- every number used in the
  verification note ``docs/verification-kerr-foundation-reduction.md`` and in the
  unit tests ``tests/test_kerr_foundation_verification.py`` (mechanically
  injected, nothing hand-copied), and
* ``docs/assets/kerr-foundation-reduction-validity.png`` -- three panels:
  h-convergence of the reference model, the reduction error against the
  platform acceptance lines, and the centre-line deflection profile.

Reference model (Kerr 1964): springs ``k_u`` between plate and shear layer,
``k_l`` between shear layer and rigid base, shear parameter ``G_p`` of the
layer, exactly as defined by ``tensorfem.foundation_plate_bending``'s docstring::

    Pi = 1/2 int k_u (w-w_1)^2 + 1/2 int k_l w_1^2 + 1/2 int G_p |grad w_1|^2
         - int q w

The three stiffness blocks of that energy are *the production element*
``tensorfem.kerr_foundation.q4_kerr_foundation_stiffness`` evaluated with the
parameter triples ``(k_u, 0)``, ``(k_u+k_l, G_p)`` and ``(-k_u, 0)``.

Run:  PYTHONPATH=src python scripts/kerr_reduction_validity.py
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch

from tensorfem.foundation_plate_bending import kerr_equivalent_pasternak
from tensorfem.kerr_foundation import q4_kerr_foundation_stiffness
from tensorfem.plate import _shape, q4_mindlin_stiffness, structured_square_mesh

HERE = Path(__file__).resolve().parent.parent
OUT_JSON = HERE / "artifacts" / "kerr_reduction_validity.json"
OUT_PNG = HERE / "docs" / "assets" / "kerr-foundation-reduction-validity.png"

DTYPE = torch.float64
L = 1.0                 # plate side [m]
E = 2.1e11              # Young's modulus [Pa]
NU = 0.3                # Poisson's ratio
T = 0.01                # thickness [m]
Q0 = 1.0e4              # pressure amplitude [Pa]
D = E * T**3 / (12.0 * (1.0 - NU**2))
LAM = (math.pi / L) ** 2
K_U, K_L = 4.0e7, 2.0e7  # spring layers [N/m^3]
W_LOCAL = [0, 3, 6, 9]
G_P_SWEEP = (1.0e4, 3.0e4, 1.0e5, 3.0e5, 1.0e6, 3.0e6, 1.0e7, 3.0e7)


# ---------------------------------------------------------------------------
# exact two-field reference
# ---------------------------------------------------------------------------

def _load(nodes, elements):
    f = torch.zeros(3 * nodes.shape[0], dtype=DTYPE)
    gq = 1.0 / 3.0**0.5
    for conn in elements:
        xy = nodes[conn]
        fe = torch.zeros(12, dtype=DTYPE)
        for xi_f in (-gq, gq):
            for eta_f in (-gq, gq):
                N, dN_nat = _shape(nodes.new_tensor(xi_f), nodes.new_tensor(eta_f))
                J = dN_nat.T @ xy
                p = N @ xy
                q = Q0 * math.sin(math.pi * p[0].item() / L) * math.sin(
                    math.pi * p[1].item() / L
                )
                fe[0::3] += N * q * torch.linalg.det(J)
        dofs = torch.stack((3 * conn, 3 * conn + 1, 3 * conn + 2), dim=1).reshape(-1)
        f[dofs] += fe
    return f


def _support(nodes):
    tol = L * 1e-12
    xedge = (nodes[:, 0].abs() < tol) | ((nodes[:, 0] - L).abs() < tol)
    yedge = (nodes[:, 1].abs() < tol) | ((nodes[:, 1] - L).abs() < tol)
    ix = torch.nonzero(xedge, as_tuple=False).flatten()
    iy = torch.nonzero(yedge, as_tuple=False).flatten()
    ie = torch.nonzero(xedge | yedge, as_tuple=False).flatten()
    return torch.cat((3 * ie, 3 * iy + 1, 3 * ix + 2)).unique(), ie


def solve_two_field(n, k_u, k_l, g_p, *, centre_line=False):
    """Exact Kerr model; returns (w_centre, w1_centre, w_centre_line)."""
    nodes, elements = structured_square_mesh(n, L, dtype=DTYPE)
    nnode = nodes.shape[0]
    ndof = 4 * nnode
    K = torch.zeros((ndof, ndof), dtype=DTYPE)
    f = torch.zeros(ndof, dtype=DTYPE)
    f[: 3 * nnode] = _load(nodes, elements)
    for conn in elements:
        xy = nodes[conn]
        kuw = q4_kerr_foundation_stiffness(xy, k_u, 0.0)
        kw1 = q4_kerr_foundation_stiffness(xy, k_u + k_l, g_p)
        dw = torch.stack((3 * conn, 3 * conn + 1, 3 * conn + 2), dim=1).reshape(-1)
        K[dw[:, None], dw] += q4_mindlin_stiffness(xy, E, NU, T)
        K[dw[:, None], dw] += kuw
        d1 = 3 * nnode + conn
        K[d1[:, None], d1] += kw1[W_LOCAL][:, W_LOCAL]
        K[dw[W_LOCAL][:, None], d1] += -kuw[W_LOCAL][:, W_LOCAL]
        K[d1[:, None], dw[W_LOCAL]] += -kuw[W_LOCAL][:, W_LOCAL]
    fixed, iedge = _support(nodes)
    fixed = torch.cat((fixed, 3 * nnode + iedge)).unique()
    free = torch.arange(ndof)
    free = free[~torch.isin(free, fixed)]
    u = torch.zeros(ndof, dtype=DTYPE)
    u[free] = torch.linalg.solve(K[free[:, None], free], f[free])
    c = (n // 2) * (n + 1) + n // 2
    profile = None
    if centre_line:
        j = n // 2
        profile = [
            [float(nodes[j * (n + 1) + i, 0]), float(u[3 * (j * (n + 1) + i)])]
            for i in range(n + 1)
        ]
    return float(u[3 * c]), float(u[3 * nnode + c]), profile


def solve_reduced(n, k_eq, g_eq, *, centre_line=False):
    """Production single-field (Pasternak) model with the reduced parameters."""
    nodes, elements = structured_square_mesh(n, L, dtype=DTYPE)
    ndof = 3 * nodes.shape[0]
    K = torch.zeros((ndof, ndof), dtype=DTYPE)
    f = _load(nodes, elements)
    for conn in elements:
        xy = nodes[conn]
        ke = q4_mindlin_stiffness(xy, E, NU, T) + q4_kerr_foundation_stiffness(xy, k_eq, g_eq)
        dw = torch.stack((3 * conn, 3 * conn + 1, 3 * conn + 2), dim=1).reshape(-1)
        K[dw[:, None], dw] += ke
    fixed, _ = _support(nodes)
    free = torch.arange(ndof)
    free = free[~torch.isin(free, fixed)]
    u = torch.zeros(ndof, dtype=DTYPE)
    u[free] = torch.linalg.solve(K[free[:, None], free], f[free])
    c = (n // 2) * (n + 1) + n // 2
    profile = None
    if centre_line:
        j = n // 2
        profile = [
            [float(nodes[j * (n + 1) + i, 0]), float(u[3 * (j * (n + 1) + i)])]
            for i in range(n + 1)
        ]
    return float(u[3 * c]), profile


def exact_deflection(k_u, k_l, g_p):
    """Closed-form centre deflection of the exact two-field model."""
    p = k_u * (k_l + 2.0 * g_p * LAM) / (k_u + k_l + 2.0 * g_p * LAM)
    return Q0 / (4.0 * D * LAM**2 + p)


def reduced_deflection(k_u, k_l, g_p):
    """Closed-form centre deflection of the production reduction."""
    k_eq, g_eq = kerr_equivalent_pasternak(k_u, k_l, g_p)
    return Q0 / (4.0 * D * LAM**2 + k_eq + 2.0 * g_eq * LAM)


def bias(k_u, k_l, g_p):
    """Exact single-mode pressure bias p_eq / p_exact of the reduction."""
    return 1.0 + 2.0 * LAM * g_p / (k_u + k_l)


def main() -> None:
    t0 = time.time()
    payload = {
        "study": "kerr_reduction_validity",
        "anchor": "GAP-FEM-KERR-PLATE",
        "case": {
            "side_m": L, "young_Pa": E, "poisson": NU, "thickness_m": T,
            "q0_Pa": Q0, "rigidity_D_Nm": D, "lambda_1_m2": LAM,
            "k_upper_N_m3": K_U, "k_lower_N_m3": K_L,
            "load": "q0 sin(pi x/L) sin(pi y/L), simply supported edges",
        },
        "reference_model": (
            "exact two-field Kerr (k_u, k_l, G_p) assembled from the production "
            "foundation element with triples (k_u,0), (k_u+k_l,G_p), (-k_u,0)"
        ),
    }

    # ---- 1. h-convergence of the exact reference model ---------------------
    g_p = 1.0e6
    wex = exact_deflection(K_U, K_L, g_p)
    conv = []
    for n in (8, 16, 24):
        w_fem, w1_fem, _ = solve_two_field(n, K_U, K_L, g_p)
        conv.append({
            "n": n,
            "w_fem": w_fem,
            "w1_fem": w1_fem,
            "w1_over_w": w1_fem / w_fem,
            "rel_err_vs_closed_form": abs(w_fem / wex - 1.0),
        })
    order = math.log(conv[0]["rel_err_vs_closed_form"] / conv[1]["rel_err_vs_closed_form"]) / math.log(2.0)
    payload["reference_convergence"] = {
        "g_p_N_m": g_p,
        "w_closed_form": wex,
        "w1_over_w_closed_form": K_U / (K_U + K_L + 2.0 * g_p * LAM),
        "rows": conv,
        "observed_order_8_16": order,
    }

    # ---- 2. reduction bias / accuracy over the G_p sweep -------------------
    sweep = []
    for g_p in G_P_SWEEP:
        k_eq, g_eq = kerr_equivalent_pasternak(K_U, K_L, g_p)
        w_ex = exact_deflection(K_U, K_L, g_p)
        w_red = reduced_deflection(K_U, K_L, g_p)
        w_fe_two, _, _ = solve_two_field(8, K_U, K_L, g_p)
        w_fe_one, _ = solve_reduced(8, k_eq, g_eq)
        sweep.append({
            "g_p_N_m": g_p,
            "g_p_over_sum_springs": g_p / (K_U + K_L),
            "k_eq_N_m3": k_eq,
            "g_eq_N_m": g_eq,
            "bias_p_eq_over_p_exact": bias(K_U, K_L, g_p),
            "w_exact_closed_form": w_ex,
            "w_reduced_closed_form": w_red,
            "reduction_rel_err": abs(w_red / w_ex - 1.0),
            "w_exact_fe_n8": w_fe_two,
            "w_reduced_fe_n8": w_fe_one,
            "reduction_rel_err_fe": abs(w_fe_one / w_fe_two - 1.0),
        })
    payload["reduction_sweep"] = sweep

    # ---- 3. validity threshold (3% / 15% acceptance lines) ----------------
    def err_of(g_p):
        return abs(reduced_deflection(K_U, K_L, g_p) / exact_deflection(K_U, K_L, g_p) - 1.0)

    def bisect(target, lo, hi):
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if err_of(mid) < target:
                lo = mid
            else:
                hi = mid
        return lo

    g3 = bisect(0.03, 1.0e2, 1.0e8)
    g15 = bisect(0.15, 1.0e2, 1.0e8)
    payload["validity"] = {
        "g_p_at_3pct_error_N_m": g3,
        "g_p_at_15pct_error_N_m": g15,
        "g_p_over_sum_springs_at_3pct": g3 / (K_U + K_L),
        "bias_at_3pct": bias(K_U, K_L, g3),
        "note": (
            "single-mode load; multi-mode loading biases each mode by its own "
            "1 + 2 lambda_mn G_p/(k_u+k_l), so the fundamental mode sets the "
            "smallest bias"
        ),
    }

    # ---- 4. centre-line profiles at a valid and an invalid G_p ------------
    profiles = {}
    for g_p, tag in ((1.0e5, "g_p_1e5_valid"), (1.0e7, "g_p_1e7_invalid")):
        k_eq, g_eq = kerr_equivalent_pasternak(K_U, K_L, g_p)
        _, _, prof_two = solve_two_field(16, K_U, K_L, g_p, centre_line=True)
        _, prof_one = solve_reduced(16, k_eq, g_eq, centre_line=True)
        profiles[tag] = {
            "g_p_N_m": g_p,
            "x_m": [p[0] for p in prof_two],
            "w_exact_m": [p[1] for p in prof_two],
            "w_reduced_m": [p[1] for p in prof_one],
            "reduction_rel_err_centre": abs(prof_one[len(prof_one) // 2][1] / prof_two[len(prof_two) // 2][1] - 1.0),
        }
    payload["profiles"] = profiles
    payload["wall_s"] = time.time() - t0

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(json.dumps({
        "observed_order": order,
        "convergence": [(r["n"], f'{r["rel_err_vs_closed_form"]:.3e}') for r in conv],
        "g_p_3pct": g3,
        "bias_at_3pct": bias(K_U, K_L, g3),
        "reduction_err": [(f'{r["g_p_N_m"]:.0e}', f'{r["reduction_rel_err"]:.3%}')
                          for r in sweep],
    }, indent=2))

    # ---- 5. figures -------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3))

    ns = [r["n"] for r in conv]
    errs = [r["rel_err_vs_closed_form"] for r in conv]
    ax[0].loglog(ns, errs, "o-", label="exact Kerr FE vs closed form")
    ax[0].loglog(ns, [errs[0] * (ns[0] / x) ** 2 for x in ns], "k--", label="O(h$^2$)")
    ax[0].set_xlabel("elements per side")
    ax[0].set_ylabel("centre-deflection rel. error")
    ax[0].set_title(f"Reference model convergence (order {order:.2f})")
    ax[0].legend()
    ax[0].grid(alpha=0.3, which="both")

    gp = [r["g_p_N_m"] for r in sweep]
    e_single = [r["reduction_rel_err"] for r in sweep]
    e_fe = [r["reduction_rel_err_fe"] for r in sweep]
    ax[1].loglog(gp, e_single, "o-", label="closed form")
    ax[1].loglog(gp, e_fe, "s--", label="FE n=8 (both fields)")
    ax[1].axhline(0.03, color="tab:green", ls=":", label="3% acceptance")
    ax[1].axhline(0.15, color="tab:red", ls=":", label="15% rejection")
    ax[1].axvline(g3, color="tab:green", lw=0.8)
    ax[1].set_xlabel("$G_p$ (N/m)")
    ax[1].set_ylabel("reduction rel. error, centre deflection")
    ax[1].set_title("Kerr$\\to$Pasternak reduction accuracy")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, which="both")

    styles = {("g_p_1e5_valid", "w_exact_m"): ("-", 2.0), ("g_p_1e5_valid", "w_reduced_m"): ("o", 3),
              ("g_p_1e7_invalid", "w_exact_m"): ("-", 2.0), ("g_p_1e7_invalid", "w_reduced_m"): ("^", 3)}
    colors = {"g_p_1e5_valid": "tab:blue", "g_p_1e7_invalid": "tab:orange"}
    labels = {("g_p_1e5_valid", "w_exact_m"): "exact Kerr, $G_p$=1e5",
              ("g_p_1e5_valid", "w_reduced_m"): "reduced, $G_p$=1e5",
              ("g_p_1e7_invalid", "w_exact_m"): "exact Kerr, $G_p$=1e7",
              ("g_p_1e7_invalid", "w_reduced_m"): "reduced, $G_p$=1e7"}
    for key, (marker, ms) in styles.items():
        tag, field = key
        prof = profiles[tag]
        ax[2].plot(prof["x_m"], prof[field], marker + "-",
                   ms=ms, lw=1.6, color=colors[tag], label=labels[key])
    ax[2].set_xlabel("x (m), centre line y = L/2")
    ax[2].set_ylabel("w (m)")
    ax[2].set_title("Deflection profiles: exact vs reduction")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)

    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)
    print("wrote", OUT_JSON)
    print("wrote", OUT_PNG)


if __name__ == "__main__":
    main()