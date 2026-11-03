"""Verification-debt audit of the Winkler-foundation reference solutions.

Runs independent oracles (Green's-function quadrature, Hankel quadrature,
finite-difference beam solve) against the analytical references in
``tensorfem.hetenyi_reference`` / ``tensorfem.plate_mindlin`` and against the two
reference forms that the 2026-11-02 audit rejected:

* beam: ``w = (q0/k_b)[1 - 1/2(A(λ|x-a|)sgn(x-a) - A(λ|x-b|)sgn(x-b))]``
* plate: ``w = w0·exp(-β r)`` with ``w0 = P/(8√(Dk))``, ``β = (k/4D)^(1/4)``

Outputs ``artifacts/foundation_reference_audit.json`` plus two figures.

Usage:  PYTHONPATH=src python3 benchmarks/foundation_reference_audit.py
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tests"))
from test_foundation_reference_audit import (  # noqa: E402
    beam_simply_supported_fd,
    beam_uniform_greens_oracle,
    plate_hankel_oracle,
)

from tensorfem.hetenyi_reference import (  # noqa: E402
    hetenyi_A,
    infinite_beam_uniform_load,
    infinite_plate_point_load,
    kelvin_kei,
)

D = torch.float64
OUT = pathlib.Path("artifacts")
OUT.mkdir(exist_ok=True)


def old_beam_formula(x, q0, a, b, lam, k_b):
    """The pre-audit (incorrect) closed form, kept here only as a yardstick."""
    x = np.asarray(x, dtype=np.float64)
    xa, xb = np.abs(x - a), np.abs(x - b)
    sa, sb = np.sign(x - a), np.sign(x - b)
    return q0 / k_b * (1.0 - 0.5 * (hetenyi_A(torch.as_tensor(lam * xa)) * sa
                                   - hetenyi_A(torch.as_tensor(lam * xb)) * sb))


def main() -> dict:
    report: dict = {"benchmark": "foundation_reference_audit", "date": "2026-11-02"}

    # ---------------- Kelvin function ----------------
    kei_table = [(0.0, -0.7853981633974483), (1.0, -0.49499463651872),
                 (2.0, -0.20240006776470), (5.0, 0.01118758650987),
                 (10.0, -3.0752456909e-4)]
    kei_errors = []
    for xv, ref in kei_table:
        got = float(kelvin_kei(torch.tensor([xv], dtype=D)))
        kei_errors.append(abs(got - ref) / max(abs(ref), 1e-300))
    report["kelvin_kei_max_rel_error_vs_tabulated"] = max(kei_errors)
    lo = float(kelvin_kei(torch.tensor([13.999999], dtype=D)))
    hi = float(kelvin_kei(torch.tensor([14.000001], dtype=D)))
    report["kelvin_kei_branch_jump_rel"] = abs(hi - lo) / abs(lo)

    # ---------------- infinite beam, uniform load ----------------
    lam, k_b, q0, a, b = 0.35, 2.0e6, 1.0e4, 0.5, 1.5
    x = np.linspace(-8.0, 10.0, 181)
    w_new = infinite_beam_uniform_load(torch.as_tensor(x, dtype=D), q0, a, b, lam, k_b).numpy()
    w_oracle = beam_uniform_greens_oracle(x, q0, a, b, lam, k_b)
    w_old = old_beam_formula(x, q0, a, b, lam, k_b)
    rel_new = np.abs(w_new - w_oracle) / np.abs(w_oracle)
    rel_old = np.abs(w_old - w_oracle) / np.abs(w_oracle)
    report["beam_uniform_closed_vs_greens_quadrature_max_rel_error"] = float(rel_new.max())
    report["beam_uniform_pre_audit_formula_max_rel_error"] = float(rel_old.max())

    EI = k_b / (4.0 * lam ** 4)
    h = 1e-3
    xf = torch.as_tensor(np.linspace(-3.0, 5.0, 9), dtype=D)
    Wf = lambda t: infinite_beam_uniform_load(t, q0, a, b, lam, k_b)  # noqa: E731
    w4 = (Wf(xf - 2 * h) - 4 * Wf(xf - h) + 6 * Wf(xf) - 4 * Wf(xf + h) + Wf(xf + 2 * h)) / h ** 4
    q = torch.where((xf >= a) & (xf <= b), torch.tensor(q0, dtype=D), torch.tensor(0.0, dtype=D))
    report["beam_uniform_ode_residual_over_q0_max"] = float(
        ((EI * w4 + k_b * Wf(xf) - q) / q0).abs().max()
    )
    report["beam_uniform_far_field_abs_w"] = float(
        infinite_beam_uniform_load(torch.tensor([-40.0, 40.0], dtype=D), q0, a, b, lam, k_b).abs().max()
    )
    report["beam_uniform_pre_audit_far_field_abs_w"] = float(abs(old_beam_formula([-40.0], q0, a, b, lam, k_b)[0]))
    report["beam_uniform_q0_over_k_b"] = q0 / k_b

    # ---------------- infinite plate ----------------
    E, nu, t, k, P = 2.1e11, 0.3, 0.01, 1.0e6, 1.0
    Dp = E * t ** 3 / (12.0 * (1.0 - nu ** 2))
    l = (Dp / k) ** 0.25
    beta = (k / (4.0 * Dp)) ** 0.25
    w0 = P / (8.0 * math.sqrt(Dp * k))
    r_grid = np.array([0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 2.0, 3.0, 4.0]) * l
    w_closed = infinite_plate_point_load(torch.as_tensor(r_grid, dtype=D), P, Dp, k).numpy()
    w_hankel = plate_hankel_oracle(r_grid, P, Dp, k)
    w_fit = w0 * np.exp(-beta * r_grid)
    report["plate_centre_deflection_exact"] = w0
    report["plate_centre_deflection_formula_P_over_8sqrtDk"] = P / (8.0 * math.sqrt(Dp * k))
    report["plate_closed_vs_hankel_max_rel_error"] = float(
        (np.abs(w_closed - w_hankel) / np.abs(w_hankel)).max()
    )
    report["plate_pre_audit_exp_fit_vs_exact_max_rel_deviation"] = float(
        (np.abs(w_fit - w_closed) / np.abs(w_closed)).max()
    )
    report["plate_characteristic_length_m"] = l
    report["plate_sign_change_radius_over_l"] = float(
        np.interp(0.0, w_closed[::-1], r_grid[::-1]) / l
    )
    r_fine = np.linspace(0.0, 25.0 * l, 200001)
    w_fine = infinite_plate_point_load(torch.as_tensor(r_fine, dtype=D), P, Dp, k).numpy()
    report["plate_reaction_integral_over_P"] = float(
        np.trapezoid(k * w_fine * 2.0 * math.pi * r_fine, r_fine) / P
    )

    # ---------------- simply supported finite beam ----------------
    L, EIb, kbb, q0b = 10.0, 1.0e5, 1.0e5, 2.0e3
    lam_b = (kbb / (4.0 * EIb)) ** 0.25
    K = EIb * (math.pi / L) ** 4 + kbb
    x_fd, w_fd = beam_simply_supported_fd(lambda xx: q0b * np.sin(math.pi * xx / L), L, EIb, kbb, n=800)
    w_ex = q0b / K * np.sin(math.pi * x_fd / L)
    report["ss_beam_fd_vs_exact_mode_max_rel_error"] = float(
        np.abs(w_fd - w_ex).max() / np.abs(w_ex).max()
    )
    report["ss_beam_lam"] = lam_b

    # ---------------- figures ----------------
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        ax[0].semilogy(x, np.abs(w_oracle), "k-", lw=2, label="Green's quadrature (oracle)")
        ax[0].semilogy(x, np.abs(w_new), "r--", lw=1.4, label="corrected closed form")
        ax[0].semilogy(x, np.abs(w_old), "b:", lw=1.4, label="pre-audit formula (rejected)")
        ax[0].axvspan(a, b, color="0.85", label="loaded interval")
        ax[0].set_xlabel("x [m]")
        ax[0].set_ylabel("|w| [m]")
        ax[0].set_title("Infinite beam, uniform load q0 over [a,b]")
        ax[0].legend(fontsize=8)
        ax[0].grid(alpha=0.3)

        ax[1].plot(r_grid / l, w_closed / w0, "r-o", ms=4, label="exact Kelvin kei solution")
        ax[1].plot(r_grid / l, w_fit / w0, "b:", label="pre-audit exp(-βr) fit (rejected)")
        ax[1].axhline(0.0, color="k", lw=0.8)
        ax[1].set_xlabel("r / l")
        ax[1].set_ylabel("w / w(0)")
        ax[1].set_title("Infinite plate on Winkler foundation, point load")
        ax[1].legend(fontsize=8)
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "foundation_reference_audit.png", dpi=140)
        plt.close(fig)
        report["figures"] = ["artifacts/foundation_reference_audit.png"]
    except Exception as exc:  # pragma: no cover - plotting is optional
        report["figures_error"] = str(exc)

    (OUT / "foundation_reference_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    rep = main()
    print(json.dumps({k: v for k, v in rep.items() if k != "figures"}, indent=2))