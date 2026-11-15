"""Figure generator for the GAP-ICE-LIT-TRANSCRIPT honest-boundary report.

Produces two PNG figures:
1. ``lindqvist_structural_discrepancy.png`` — bar/component comparison of the
   original 4-component Lindqvist model vs the platform's calibrated
   3-component variant on the same synthetic case (shows the structural
   decomposition mismatch that is the core honest finding).
2. ``lindqvist_thickness_scaling.png`` — thickness-scaling comparison of the
   original 4-component model vs the platform's calibrated variant across
   h in [0.1, 2.0] m, showing that the two models scale differently (the
   original has R_c ~ h^2 and R_b ~ h^{5/4}; the platform variant has
   different exponents), which is why the platform model cannot stand in for
   the original verbatim.

All numbers are machine-computed from the structural model / platform
implementation; nothing is hand-typed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ANCHOR_DIR = Path(__file__).resolve().parent / "anchors"
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ANCHOR_DIR))
from lindqvist_riska_anchor import CASE, compute_total_resistance  # noqa: E402

_PLATFORM_BACKEND = Path("/srv/tensor/agent/work/m19459/backend")
if _PLATFORM_BACKEND.exists():
    sys.path.insert(0, str(_PLATFORM_BACKEND))
    from api.ice.formulas import (  # noqa: E402
        ShipGeometry as PlatShipGeometry,
        IceParams as PlatIceParams,
        calc_lindqvist,
    )


def _platform_on_case(h: float, v: float) -> dict:
    q = CASE["quantities"]
    geo = PlatShipGeometry(
        Lpp=q["L"]["value"], Lwl=q["L"]["value"], B=q["B"]["value"],
        T=q["T"]["value"], Cb=q["Cb"]["value"],
        displacement=q["Cb"]["value"] * q["L"]["value"] * q["B"]["value"] * q["T"]["value"],
        bow_angle=q["alpha_deg"]["value"], stem_angle=q["psi_deg"]["value"],
    )
    ice = PlatIceParams(
        ice_thickness=h, flexural_strength=q["sigma_f"]["value"],
        ice_density=q["rho_i"]["value"], water_density=q["rho_w"]["value"],
        ice_concentration=1.0, friction_coeff=q["mu"]["value"],
    )
    res = calc_lindqvist(geo, ice, v)
    return {
        "Rtot_N": (res.R_submersion + res.R_bending + res.R_clearing) * 1e3,
        "R_sub_N": res.R_submersion * 1e3,
        "R_bend_N": res.R_bending * 1e3,
        "R_clear_N": res.R_clearing * 1e3,
    }


def fig1_component_discrepancy() -> str:
    """Bar comparison of component decomposition on the synthetic case."""
    q = CASE["quantities"]
    h, v = q["h"]["value"], q["v"]["value"]
    four = compute_total_resistance(h, v)
    plat = _platform_on_case(h, v)

    orig = [four["Rc_N"], four["Rb_N"], four["Rs_N"], four["Rmom_N"]]
    orig_labels = ["R_c\n(crushing)", "R_b\n(bending)", "R_s\n(submersion)", "R_mom\n(momentum)"]
    plat_comp = [plat["R_sub_N"], plat["R_bend_N"], plat["R_clear_N"]]
    plat_labels = ["R_submersion", "R_bending", "R_clearing"]

    fig, ax = plt.subplots(figsize=(9, 5))
    x1 = np.arange(len(orig))
    x2 = np.arange(len(plat_comp)) + len(orig) + 1
    ax.bar(x1, [v / 1e6 for v in orig], color="#1f77b4", alpha=0.85, label="Original Lindqvist 4-component")
    ax.bar(x2, [v / 1e6 for v in plat_comp], color="#ff7f0e", alpha=0.85, label="Platform calibrated 3-component")
    ax.set_xticks(list(x1) + list(x2))
    ax.set_xticklabels(orig_labels + plat_labels, fontsize=8)
    ax.set_ylabel("Component force (MN)")
    ax.set_title(
        f"Structural decomposition mismatch (h={h} m, v={v} m/s)\n"
        f"Original Rtot={four['Rtot_N']/1e6:.2f} MN vs Platform Rtot={plat['Rtot_N']/1e6:.2f} MN"
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = ANCHOR_DIR / "lindqvist_structural_discrepancy.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


def fig2_thickness_scaling() -> str:
    """Thickness-scaling comparison across h in [0.1, 2.0] m."""
    q = CASE["quantities"]
    v = q["v"]["value"]
    hs = np.linspace(0.1, 2.0, 40)
    orig_tot = [compute_total_resistance(h, v)["Rtot_N"] for h in hs]
    plat_tot = [_platform_on_case(h, v)["Rtot_N"] for h in hs]

    # log-log slope for scaling exponent
    def slope(xs, ys):
        lx, ly = np.log(xs), np.log(ys)
        return np.polyfit(lx, ly, 1)[0]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(hs, orig_tot, "-", color="#1f77b4", label=f"Original 4-component (slope≈{slope(hs, orig_tot):.2f})")
    ax.loglog(hs, plat_tot, "--", color="#ff7f0e", label=f"Platform calibrated (slope≈{slope(hs, plat_tot):.2f})")
    ax.set_xlabel("Ice thickness h (m)")
    ax.set_ylabel("Total ice resistance (N)")
    ax.set_title("Thickness-scaling comparison: original vs platform calibrated variant")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = ANCHOR_DIR / "lindqvist_thickness_scaling.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return str(out)


if __name__ == "__main__":
    f1 = fig1_component_discrepancy()
    f2 = fig2_thickness_scaling()
    print(f"fig1: {f1}")
    print(f"fig2: {f2}")
