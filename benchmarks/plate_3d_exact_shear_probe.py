"""Close the open item of ``docs/plate-3d-exact-anchor.md`` §5.4 / §7⑤.

The earlier evidence bundle sampled the through-thickness transverse shear
``tau_xz(z)`` at the plate **centre** ``(a/2, b/2)``.  For the double-sine load
``q = q0 sin(alpha x) sin(beta y)`` the exact field carries
``tau_xz ∝ cos(alpha x) sin(beta y)``, so ``tau_xz(a/2, b/2, z) ≡ 0`` for every
``z``: the field ``tau_xz_peak_over_q0`` of that bundle (1.5e-16) is a property
of the sampling point, not of the plate.  The physically meaningful transverse
shear section is the **edge midpoint** ``(0 or a, b/2)`` where ``|cos| = 1``.

This script re-samples there and reports, programmatically (no hand-typed
numbers):

1. ``tau_xz(z)`` on a 201-point through-thickness ladder, its peak, the ``z`` of
   the peak, and ``tau_xz_peak / q0``;
2. the closed-form plate-theory value ``V_x / h`` with ``V_x = q0 / alpha`` -- the
   *static* shear force of the Kirchhoff/FSDT solution, derived from equilibrium
   alone (``dV_x/dx = -q``), independent of the 3-D ODE state;
3. the equilibrium residual ``| int_{-h/2}^{h/2} tau_xz dz - V_x | / V_x`` --
   an integral identity the exact 3-D field must satisfy;
4. the dual section ``tau_yz(z)`` at ``(a/2, 0)`` for the same reasons;
5. the negative control at the centre, which must stay at round-off.

Outputs ``docs/assets/plate_3d_exact/shear_probe.json`` and
``docs/assets/plate_3d_exact/fig5_shear_section.png``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from tensorfem.plate_3d_exact import SSPlateExact3D

_DT = torch.float64
ASSETS = Path("docs/assets/plate_3d_exact")


def probe(ref: SSPlateExact3D, n_z: int = 201) -> dict:
    z = torch.linspace(-0.5 * ref.h, 0.5 * ref.h, n_z, dtype=_DT)
    a, b, h, q0 = ref.a, ref.b, ref.h, ref.q0

    # --- edge midpoint section (x = 0, y = b/2): cos(alpha*0)*sin(beta*b/2) = 1
    tz_edge = ref.stress(torch.zeros_like(z), torch.full_like(z, 0.5 * b), z)[4]
    # --- dual section (x = a/2, y = 0) for tau_yz
    ty_edge = ref.stress(torch.full_like(z, 0.5 * a), torch.zeros_like(z), z)[5]
    # --- negative control: the centre, where both sections vanish analytically
    tz_centre = ref.stress(torch.full_like(z, 0.5 * a), torch.full_like(z, 0.5 * b), z)[4]

    peak_idx = int(torch.argmax(tz_edge.abs()))
    peak = float(tz_edge.abs().max())
    z_peak = float(z[peak_idx])

    # plate theory (Kirchhoff/FSDT, and from equilibrium dV_x/dx = -q plus
    # Q_x = -D d(nabla^2 w)/dx with w = q0/(D (alpha^2+beta^2)^2) sin sin):
    #   V_x = q0 * alpha / (alpha^2 + beta^2) * cos(alpha x) sin(beta y)
    # at the edge midpoint cos = sin = 1.  (The naive q0/alpha is wrong for a
    # two-way plate; it is only right in the strip/beam limit beta -> 0.)
    denom = ref.alpha ** 2 + ref.beta ** 2
    v_x = q0 * ref.alpha / denom
    v_y = q0 * ref.beta / denom
    fsdt_tau = v_x / ref.h            # FSDT: constant through the thickness

    # integral identity int tau_xz dz = V_x (trapezoid on a uniform ladder is exact
    # to O(h_z^2) for a smooth integrand; also reported as a Richardson pair)
    def integrate(field: torch.Tensor, n: int) -> float:
        zz = torch.linspace(-0.5 * ref.h, 0.5 * ref.h, n, dtype=_DT)
        x = torch.zeros_like(zz) if field is tz_edge else torch.full_like(zz, 0.5 * a)
        y = torch.full_like(zz, 0.5 * b) if field is tz_edge else torch.zeros_like(zz)
        comp = 4 if field is tz_edge else 5
        vals = ref.stress(x, y, zz)[comp]
        return float(torch.trapezoid(vals, zz))

    int_n, int_2n = integrate(tz_edge, 201), integrate(tz_edge, 401)
    # magnitude identity only: the exact field's tau_xz carries the sign
    # convention tau_xz = -d psi_x/dz, i.e. the opposite of +V_x for this load
    # (the signed integral is reported alongside so the convention is auditable).
    shear_residual = abs(abs(int_2n) - abs(v_x)) / abs(v_x)
    richardson = int_2n + (int_2n - int_n) / 3.0  # second-order extrapolation

    return {
        "meta": {
            "script": "benchmarks/plate_3d_exact_shear_probe.py",
            "purpose": "close plate-3d-exact-anchor §5.4/§7⑤: resample tau_xz at the edge midpoint",
            "plate": {"a": a, "b": b, "h": h, "young": ref.young, "poisson": ref.poisson, "q0": q0},
            "aspect_a_over_h": a / ref.h,
            "sampling": {"section": "x=0, y=b/2", "n_z": n_z},
        },
        "edge_section": {
            "z": z.tolist(),
            "tau_xz": tz_edge.tolist(),
            "tau_xz_peak": peak,
            "tau_xz_peak_over_q0": peak / q0,
            "z_at_peak_over_h": z_peak / ref.h,
        },
        "dual_section": {
            "tau_yz_peak_over_q0": float(ty_edge.abs().max()) / q0,
            "tau_yz_peak_over_v_y_over_h": float(ty_edge.abs().max()) / (v_y / ref.h),
        },
        "plate_theory": {
            "v_x_closed_form_over_q0a": v_x / (q0 * a),
            "fsdt_tau_constant": fsdt_tau,
            "fsdt_tau_over_q0": fsdt_tau / q0,
            "peak_over_fsdt_constant": peak / fsdt_tau,
            "fsdt_value_at_midsurface_over_peak": fsdt_tau / peak,
        },
        "equilibrium": {
            "integral_tau_xz_dz": int_2n,
            "v_x": v_x,
            "signed_integral": int_2n,
            "sign_convention": "tau_xz = -dpsi_x/dz, so the signed integral is -V_x for this load",
            "residual_rel_magnitude": shear_residual,
            "richardson_integral": richardson,
            "richardson_residual_rel_magnitude": abs(abs(richardson) - abs(v_x)) / abs(v_x),
        },
        "negative_control_centre": {
            "tau_xz_max_abs": float(tz_centre.abs().max()),
            "tau_xz_max_abs_over_q0": float(tz_centre.abs().max()) / q0,
            "note": "identically zero for this load; the old bundle's 1.5e-16 field came from here",
        },
    }


def figure(data: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ASSETS.mkdir(parents=True, exist_ok=True)
    z = [v / data["meta"]["plate"]["h"] for v in data["edge_section"]["z"]]
    fig, ax = plt.subplots(1, 2, figsize=(10.0, 4.2))
    ax[0].plot(data["edge_section"]["tau_xz"], z, "-", label="3-D exact, edge midpoint")
    ax[0].axvline(data["plate_theory"]["fsdt_tau_constant"], color="tab:orange", ls="--",
                  label="plate theory $V_x/h$ (FSDT)")
    ax[0].axhline(data["edge_section"]["z_at_peak_over_h"], color="0.6", ls=":", lw=0.8)
    ax[0].set_xlabel(r"$\tau_{xz}$  [Pa]")
    ax[0].set_ylabel("$z/h$")
    ax[0].set_title("through-thickness transverse shear\n(at $x=0,\\ y=b/2$)")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    ax[1].bar(
        ["3-D peak", "plate theory\n$V_x/h$"],
        [data["edge_section"]["tau_xz_peak_over_q0"],
         data["plate_theory"]["fsdt_tau_over_q0"]],
        color=["tab:blue", "tab:orange"],
    )
    for i, v in enumerate([data["edge_section"]["tau_xz_peak_over_q0"],
                           data["plate_theory"]["fsdt_tau_over_q0"]]):
        ax[1].text(i, v, f"{v:.5f}", ha="center", va="bottom", fontsize=9)
    ax[1].set_ylabel(r"$\tau_{xz}^{\max}/q_0$")
    ax[1].set_title(f"peak shear normalised by $q_0$  ($a/h={data['meta']['aspect_a_over_h']:.0f}$)")
    ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(ASSETS / "fig5_shear_section.png", dpi=160)
    plt.close(fig)


def main() -> int:
    ref = SSPlateExact3D(a=1.0, b=1.0, h=0.1, young=2.0e11, poisson=0.3, q0=1.0e6)
    data = probe(ref)
    figure(data)
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "shear_probe.json").write_text(json.dumps(data, indent=2) + "\n")
    es, pt, eq, nc = data["edge_section"], data["plate_theory"], data["equilibrium"], data["negative_control_centre"]
    print(f"a/h = {data['meta']['aspect_a_over_h']:.0f}")
    print(f"tau_xz peak / q0        = {es['tau_xz_peak_over_q0']:.6f}  at z/h = {es['z_at_peak_over_h']:+.4f}")
    print(f"plate theory V_x/h / q0 = {pt['fsdt_tau_over_q0']:.6f}")
    print(f"peak / plate theory     = {pt['peak_over_fsdt_constant']:.6f}")
    print(f"int tau_xz dz vs V_x    = signed {eq['signed_integral']:.6e} vs V_x {eq['v_x']:.6e}, residual {eq['residual_rel_magnitude']:.3e} (Richardson {eq['richardson_residual_rel_magnitude']:.3e})")
    print(f"centre control |tau_xz|/q0 = {nc['tau_xz_max_abs_over_q0']:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())