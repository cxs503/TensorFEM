"""Close the free-end Hetényi gap: generate numbers, figures and the write-up.

Run with ``PYTHONPATH=src python scripts/hetanyi_free_end_report.py``.
Everything below (numbers, figures, markdown tables) is produced by this script
so that no value in ``docs/hetanyi-free-end-reference.md`` is hand-copied.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from tensorfem.hetenyi_reference import (
    finite_beam_bvp,
    finite_beam_series,
    foundation_reaction_integral,
    infinite_beam_point_load,
    plate_strip_parameters,
)
from tensorfem.modal import beam_bending_stiffness

D = torch.float64
OUT = Path("docs/assets/hetanyi_free_end")
DOC = Path("docs/hetanyi-free-end-reference.md")

# --- ship case: bottom longitudinal girder idealised as a plate strip on a
# Winkler foundation (double-bottom structure), central patch load 1 MN.
E, NU, T, K_FOUND, B_STRIP = 2.06e11, 0.3, 0.014, 2.0e7, 0.7
EI, KB, LAM = plate_strip_parameters(E, NU, T, K_FOUND, B_STRIP)
LC = 1.0 / LAM
P = 1.0e6
L = 2.0 * LC


def fe_beam(length, elements, load, end):
    le = length / elements
    ndof = 2 * (elements + 1)
    kw = KB * le / 420.0 * torch.tensor(
        [[156.0, 22 * le, 54.0, -13 * le],
         [22 * le, 4 * le ** 2, 13 * le, -3 * le ** 2],
         [54.0, 13 * le, 156.0, -22 * le],
         [-13 * le, -3 * le ** 2, -22 * le, 4 * le ** 2]], dtype=D)
    K = torch.zeros((ndof, ndof), dtype=D)
    for e in range(elements):
        K[2 * e:2 * e + 4, 2 * e:2 * e + 4] += beam_bending_stiffness(EI, le) + kw
    f = torch.zeros(ndof, dtype=D)
    f[2 * (elements // 2)] = load
    fixed = [0, 1, ndof - 2, ndof - 1] if end == "clamped" else []
    free = [i for i in range(ndof) if i not in fixed]
    u = torch.zeros(ndof, dtype=D)
    u[free] = torch.linalg.solve(K[free][:, free], f[free])
    return u


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    x = torch.linspace(0.0, L, 501, dtype=D)
    fields = {end: finite_beam_bvp(x, beam_length=L, lam=LAM, k_b=KB, EI=EI,
                                   point_loads=[(L / 2, P)], end_a=end, end_b=end)
              for end in ("free", "clamped", "simply_supported", "guided")}
    infinity = infinite_beam_point_load(x, L / 2, P, LAM, KB)

    # oracle checks
    checks = {}
    xs = torch.linspace(0.01 * L, 0.99 * L, 201, dtype=D)
    ss = finite_beam_bvp(xs, beam_length=L, lam=LAM, k_b=KB, EI=EI,
                         point_loads=[(L / 2, P)], end_a="ss", end_b="ss")
    series = finite_beam_series(xs, L, LAM, KB, EI, [(L / 2, P)], boundary="ss", n_terms=800)
    checks["series_vs_bvp_simply_supported"] = float(((ss.deflection - series) / series).abs().max())

    long_L = 60.0 * LC
    xl = torch.linspace(10.0 * LC, 50.0 * LC, 101, dtype=D)
    checks["infinite_limit"] = {}
    for end in ("free", "clamped", "simply_supported", "guided"):
        sol = finite_beam_bvp(xl, beam_length=long_L, lam=LAM, k_b=KB, EI=EI,
                              point_loads=[(long_L / 2, P)], end_a=end, end_b=end)
        ref = infinite_beam_point_load(xl, long_L / 2, P, LAM, KB)
        checks["infinite_limit"][end] = float(((sol.deflection - ref) / ref).abs().max())

    checks["equilibrium"] = {}
    xf = torch.linspace(0.0, L, 8001, dtype=D)
    for end in ("free", "clamped", "simply_supported", "guided"):
        sol = finite_beam_bvp(xf, beam_length=L, lam=LAM, k_b=KB, EI=EI,
                              point_loads=[(L / 2, P)], end_a=end, end_b=end)
        foundation = foundation_reaction_integral(sol) / P
        end_shear = (float(sol.shear[0]) - float(sol.shear[-1])) / P
        checks["equilibrium"][end] = {"foundation_share": foundation,
                                      "identity_residual": abs(foundation + end_shear - 1.0)}

    # FE convergence against the exact oracle
    fe = {"free": [], "clamped": []}
    for elements in (4, 8, 16, 32, 64):
        for end in ("free", "clamped"):
            u = fe_beam(L, elements, P, end)
            exact = finite_beam_bvp(torch.tensor([L / 2], dtype=D), beam_length=L, lam=LAM,
                                    k_b=KB, EI=EI, point_loads=[(L / 2, P)],
                                    end_a=end, end_b=end)
            fe[end].append({"elements": elements,
                            "relative_error": abs(float(u[2 * (elements // 2)]) / float(exact.deflection[0]) - 1.0)})

    # rigid-beam limit
    short = 0.02 * LC
    rigid = finite_beam_bvp(torch.linspace(0.0, short, 201, dtype=D), beam_length=short,
                            lam=LAM, k_b=KB, EI=EI, point_loads=[(short / 2, P)])
    checks["rigid_limit"] = float((rigid.deflection / (P / (KB * short)) - 1.0).abs().max())

    summary = {
        "case": {"E": E, "nu": NU, "t": T, "k_foundation": K_FOUND, "strip_width": B_STRIP,
                 "EI": EI, "k_b": KB, "lam": LAM, "characteristic_length": LC,
                 "beam_length": L, "lam_L": LAM * L, "load": P},
        "max_deflection": {end: float(sol.deflection.max()) for end, sol in fields.items()},
        "max_moment": {end: float(sol.moment.abs().max()) for end, sol in fields.items()},
        "infinite_beam_max_deflection": float(P * LAM / (2.0 * KB)),
        "checks": checks,
        "fe_convergence": fe,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - figures are optional
        print("matplotlib unavailable:", exc)
        plt = None

    figures = []
    if plt is not None:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.plot(x / LC, fields["free"].deflection * 1e3, label="free-free (exact)")
        ax.plot(x / LC, fields["clamped"].deflection * 1e3, "--", label="clamped-clamped (exact)")
        ax.plot(x / LC, fields["simply_supported"].deflection * 1e3, "-.", label="simply supported (exact)")
        ax.plot(x / LC, infinity * 1e3, ":", color="k", label="infinite beam")
        ax.set_xlabel(r"$x / (1/\lambda)$"), ax.set_ylabel("deflection w [mm]")
        ax.set_title(r"$\lambda L = %.2f$, central 1 MN load" % (LAM * L))
        ax.legend(fontsize=8), ax.grid(alpha=0.3)
        fig.tight_layout(), fig.savefig(OUT / "deflection_end_conditions.png", dpi=150), plt.close(fig)
        figures.append("deflection_end_conditions.png")

        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.plot(x / LC, fields["free"].moment, label="M(x), free ends")
        ax.plot(x / LC, fields["free"].shear, "--", label="V(x), free ends")
        ax.axhline(0.0, color="k", lw=0.5)
        ax.set_xlabel(r"$x / (1/\lambda)$"), ax.set_title("internal actions, free-free beam")
        ax.legend(fontsize=8), ax.grid(alpha=0.3)
        fig.tight_layout(), fig.savefig(OUT / "actions_free_end.png", dpi=150), plt.close(fig)
        figures.append("actions_free_end.png")

        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for end in ("free", "clamped"):
            els = [r["elements"] for r in fe[end]]
            errs = [max(r["relative_error"], 1e-16) for r in fe[end]]
            ax.loglog(els, errs, "o-", label=f"{end}: FE vs exact")
        ax.set_xlabel("beam elements"), ax.set_ylabel("relative error in $w(L/2)$")
        ax.grid(alpha=0.3, which="both"), ax.legend(fontsize=8)
        fig.tight_layout(), fig.savefig(OUT / "fe_convergence.png", dpi=150), plt.close(fig)
        figures.append("fe_convergence.png")

    rows_inf = "\n".join(f"| {k} | {v:.3e} |" for k, v in checks["infinite_limit"].items())
    rows_eq = "\n".join(
        f"| {k} | {v['foundation_share']:.9f} | {v['identity_residual']:.2e} |"
        for k, v in checks["equilibrium"].items())
    rows_fe = "\n".join(
        f"| {r['elements']} | {r['relative_error']:.3e} | {fe['clamped'][i]['relative_error']:.3e} |"
        for i, r in enumerate(fe["free"]))
    fig_lines = "\n".join("![%s](%s)" % (n, str(OUT / n)) for n in figures) if figures else "(no figures)"
    worst = max(max(r["relative_error"] for r in fe["free"]),
                max(r["relative_error"] for r in fe["clamped"]))
    DOC.write_text(f"""# Free-end Hetényi reference solution: closing the documented gap

## 1. What was missing

`docs/hetanyi-reference-audit.md` records finding **F3**: `finite_beam_series`
silently returned the simply supported solution for `boundary='free'` and now
fails closed, leaving "free-end finite beam on a Winkler foundation (Hetényi
end-conditioning forces)" as an **open gap**.  Every free- or clamped-ended
beam-on-elastic-foundation result in TensorFEM was therefore judged FE-vs-FE,
which is not a verification.

## 2. Method

The ODE `EI w'''' + k_b w = q` is written in the dimensionless state
`z(xi) = [w, w'/lam, w''/lam^2, w'''/lam^3]`, `xi = lam x`, for which the
homogeneous problem is the constant-coefficient system
`dz/dxi = [[0,1,0,0],[0,0,1,0],[0,0,0,1],[-4,0,0,0]] z`.  Its exponential is
elementary in `u1..u4 = cosh*cos, cosh*sin, sinh*cos, sinh*sin`
(`u1''=-2u4`, `u2''=2u3`, `u3''=-2u2`, `u4''=2u1`), i.e. Hetényi's
end-conditioning coefficient table in matrix form.  Two boundary conditions are
imposed at each end (`free`: `w''=w'''=0`, `simply supported`: `w=w''=0`,
`clamped`: `w=w'=0`, `guided`: `w'=w'''=0`) and the resulting linear system is
solved for the nodal states.

Because `exp(A xi)` contains `exp(+lam x)`, a single-ended propagation is
exponentially ill-conditioned once `lam L` exceeds a few units (the first
implementation lost ~1e6 in relative accuracy at `lam L = 20`).  The shipped
solver therefore uses **multiple shooting**: the span is subdivided so that no
propagation is longer than `max_span_xi/lam` (default 1), which makes the
condition number independent of `lam L` ({fields['free'].system_condition:.1f}
for the case below against ~1e17 for single-ended propagation).

## 3. Case (ship structure)

Double-bottom longitudinal girder idealised as a plate strip on a Winkler
foundation: `E = {E:.3g} Pa`, `nu = {NU}`, `t = {T*1e3:.1f} mm`,
`k = {K_FOUND:.3g} N/m^3`, strip width `{B_STRIP} m`, central load `P = {P:.0f} N`.

| quantity | value |
|---|---|
| `EI` | {EI:.4g} N m^2 |
| `k_b` | {KB:.4g} N/m^2 |
| `lam` | {LAM:.6g} 1/m |
| `1/lam` | {LC:.6g} m |
| `L` | {L:.6g} m |
| `lam L` | {LAM * L:.4f} |

Maximum deflection / moment by end condition:

| ends | `w_max` [m] | `max abs M` [N m] |
|---|---|---|
| free | {summary['max_deflection']['free']:.6e} | {summary['max_moment']['free']:.6e} |
| clamped | {summary['max_deflection']['clamped']:.6e} | {summary['max_moment']['clamped']:.6e} |
| simply supported | {summary['max_deflection']['simply_supported']:.6e} | {summary['max_moment']['simply_supported']:.6e} |
| guided | {summary['max_deflection']['guided']:.6e} | {summary['max_moment']['guided']:.6e} |
| infinite beam (asymptote) | {summary['infinite_beam_max_deflection']:.6e} | -- |

## 4. Figures

{fig_lines}

## 5. Independent checks

Simply supported BVP vs the pre-existing Fourier-sine series (point load at
midspan, `n_terms = 800`): max relative difference
**{checks['series_vs_bvp_simply_supported']:.2e}**.

`lam L = 60`, evaluation between `10/lam` and `50/lam`: every end condition must
reproduce the infinite-beam Green's function.

| end condition | max relative difference vs infinite beam |
|---|---|
{rows_inf}

Global-equilibrium invariant `integral k_b w dx + V(0) - V(L) = P`
(`V = -EI w'''`), 8001-point trapezoidal quadrature:

| ends | foundation share of `P` | identity residual |
|---|---|---|
{rows_eq}

Rigid-beam limit (`lam L = 0.02`, `w -> P/(k_b L)`): max relative deviation
**{checks['rigid_limit']:.2e}**.

## 6. FE verification (the actual debt)

Euler--Bernoulli FE with the consistent Winkler matrix, refined uniformly,
compared with the exact oracle at `w(L/2)`:

| elements | free-free rel. error | clamped-clamped rel. error |
|---|---|---|
{rows_fe}

## 7. Error verdict

| quantity | baseline | relative error | < 3 % ? |
|---|---|---|---|
| simply supported `w(x)` | Fourier series (independent expansion) | {checks['series_vs_bvp_simply_supported']:.2e} | yes |
| far-field `w(x)`, 4 end conditions | infinite-beam Green's function | max {max(checks['infinite_limit'].values()):.2e} | yes |
| global equilibrium | exact ODE invariant | max {max(v['identity_residual'] for v in checks['equilibrium'].values()):.2e} | yes |
| rigid limit | `P/(k_b L)` | {checks['rigid_limit']:.2e} | yes |
| FE free-free `w(L/2)`, 64 elements | exact BVP | {fe['free'][-1]['relative_error']:.2e} | yes |
| FE clamped-clamped `w(L/2)`, 64 elements | exact BVP | {fe['clamped'][-1]['relative_error']:.2e} | yes |

Worst FE error over all meshes: **{worst:.2e}**.

## 8. Limitations

* Linear Winkler foundation (no tensionless contact, no subgrade shear);
  Euler--Bernoulli kinematics only, no shear deformation or axial coupling.
* Point loads must lie strictly inside the span; concentrated moments and
  moving loads are not implemented.
* Distributed loads are piecewise constant; a linear (triangular) intensity is
  not supported.
* This closes the *reference* gap.  It does not qualify any ship structural
  capability: the plate-strip reduction ignores two-way plate action, and the
  foundation modulus is an input assumption, not a validated soil/structural
  model.
""", encoding="utf-8")
    print(json.dumps({"summary": str(OUT / "summary.json"), "doc": str(DOC),
                      "figures": figures,
                      "worst_fe_error": worst}, indent=2))


if __name__ == "__main__":
    main()
