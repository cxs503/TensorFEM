"""Generate the GAP-FEM-WINKLER bending-moment verification figure + data.

Every number written to ``docs/assets/gap-fem-winkler-moments.json`` comes from
the library functions themselves (no hand transcription), and the PNG is drawn
only from that JSON, so the figure and the report table cannot drift apart.

Run with ``PYTHONPATH=src python scripts/gen_gap_fem_winkler_moments_figure.py``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from tensorfem.foundation_plate_bending import (  # noqa: E402
    compare_finite_plate_moments,
    finite_plate_foundation_fem,
    navier_foundation_deflection,
    navier_foundation_moments,
    nodal_bending_moments,
)

YOUNG, POISSON, THICK = 3.0e10, 0.3, 0.1
SIDE, PRESSURE = 10.0, 1.0e4
K_BUOY = 1025.0 * 9.81
ASSETS = Path(__file__).resolve().parents[1] / "docs" / "assets"


def centre_line(nx: int, n: int = 41):
    """FE ``Mx`` along ``y = b/2`` versus the Navier series."""
    sol = finite_plate_foundation_fem(
        side_x=SIDE, side_y=SIDE, nx=nx, ny=nx, pressure=PRESSURE,
        young=YOUNG, poisson=POISSON, thickness=THICK,
    )
    nodal = nodal_bending_moments(
        sol.nodes, sol.elements, sol.deflection,
        young=YOUNG, poisson=POISSON, thickness=THICK,
    )
    xs = torch.linspace(0.0, SIDE, n, dtype=torch.float64)
    pts = torch.stack((xs, torch.full_like(xs, 0.5 * SIDE)), dim=1)
    analytic = navier_foundation_moments(
        pts, side_x=SIDE, side_y=SIDE, young=YOUNG, poisson=POISSON,
        thickness=THICK, pressure=PRESSURE,
    )[:, 0]
    j = nx // 2
    k = min(nx - 1, int(round(0.5 * nx)))
    fem = torch.stack([nodal[j * (nx + 1) + i][0] for i in range(k + 1)])
    fem_x = torch.linspace(0.0, SIDE * k / nx, k + 1, dtype=torch.float64)
    return xs, analytic, fem_x, fem


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "case": {
            "young_Pa": YOUNG, "poisson": POISSON, "thickness_m": THICK,
            "side_x_m": SIDE, "side_y_m": SIDE, "pressure_Pa": PRESSURE,
            "buoyancy_k_N_per_m3": K_BUOY,
        },
        "convergence": {},
        "centre_line": {},
        "cases": {},
    }

    for nx in (8, 16, 32, 48):
        report = compare_finite_plate_moments(nx=nx, ny=nx)
        payload["convergence"][str(nx)] = {
            key: report[key]
            for key in (
                "max_mx_rel_error", "max_my_rel_error", "max_w_rel_error",
                "center_mx_rel_error", "mxy_rel_error", "center_mx",
                "ndof", "foundation_reaction", "applied_load",
            )
        }
        payload["convergence"][str(nx)]["rows"] = report["rows"]

    xs, analytic, fem_x, fem = centre_line(32)
    payload["centre_line"] = {
        "x": [float(v) for v in xs],
        "series_mx": [float(v) for v in analytic],
        "fe_x": [float(v) for v in fem_x],
        "fe_mx": [float(v) for v in fem],
    }

    cases = {
        "kirchhoff (k=0)": dict(),
        "winkler k=1e5": dict(winkler=1.0e5),
        "pasternak k=1e5 G=1e6": dict(winkler=1.0e5, pasternak=1.0e6),
        "marine buoyancy k=rho*g": dict(winkler=K_BUOY),
    }
    for label, kwargs in cases.items():
        report = compare_finite_plate_moments(nx=32, ny=32, **kwargs)
        payload["cases"][label] = {
            key: report[key]
            for key in (
                "max_mx_rel_error", "max_my_rel_error", "max_w_rel_error",
                "mxy_rel_error", "center_mx", "center_mx_rel_error",
            )
        }
        payload["cases"][label]["w_center_series"] = float(
            navier_foundation_deflection(
                torch.tensor([[SIDE / 2, SIDE / 2]], dtype=torch.float64),
                side_x=SIDE, side_y=SIDE, young=YOUNG, poisson=POISSON,
                thickness=THICK, pressure=PRESSURE, **kwargs,
            )[0]
        )

    data_path = ASSETS / "gap-fem-winkler-moments.json"
    data_path.write_text(json.dumps(payload, indent=2) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
    ax = axes[0]
    ax.plot(payload["centre_line"]["x"], payload["centre_line"]["series_mx"],
            "k-", lw=2, label="Navier series")
    ax.plot(payload["centre_line"]["fe_x"], payload["centre_line"]["fe_mx"],
            "o", ms=4, mfc="none", label="FE Q4 Mindlin, nx=32")
    ax.set_xlabel("x [m]  (y = 5 m)")
    ax.set_ylabel("$M_x$ [N m/m]")
    ax.set_title("Sagging moment along the centre line\n$q=10$ kPa, SS square $10\\times10$ m")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    sizes = sorted(int(k) for k in payload["convergence"])
    errors = [payload["convergence"][str(n)]["max_mx_rel_error"] for n in sizes]
    centre = [payload["convergence"][str(n)]["center_mx_rel_error"] for n in sizes]
    ax.loglog(sizes, [e * 100 for e in errors], "s-", label="max over sampled points")
    ax.loglog(sizes, [e * 100 for e in centre], "o-", label="centre point")
    ref = [errors[0] * 100 * (sizes[0] / n) ** 2 for n in sizes]
    ax.loglog(sizes, ref, "k--", lw=1, label="$O(h^2)$ reference")
    ax.axhline(5.0, color="r", lw=1, ls=":", label="acceptance 5 %")
    ax.set_xlabel("n (elements per side)")
    ax.set_ylabel("|$M_x$ - series| / series  [%]")
    ax.set_title("Moment convergence (second order)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    ax = axes[2]
    labels = list(payload["cases"])
    mx = [payload["cases"][k]["max_mx_rel_error"] * 100 for k in labels]
    mxy = [payload["cases"][k]["mxy_rel_error"] * 100 for k in labels]
    xx = range(len(labels))
    ax.bar([i - 0.2 for i in xx], mx, width=0.4, label="max $M_x$, $M_y$ error")
    ax.bar([i + 0.2 for i in xx], mxy, width=0.4, label="$M_{xy}$ error")
    ax.axhline(5.0, color="r", lw=1, ls=":", label="acceptance 5 %")
    ax.axhline(3.0, color="orange", lw=1, ls=":", label="precision bar 3 %")
    ax.set_xticks(list(xx))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("relative error [%]")
    ax.set_title("FE vs series, $n_x=n_y=32$")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig_path = ASSETS / "gap-fem-winkler-moments.png"
    fig.savefig(fig_path, dpi=140)
    print(f"wrote {data_path}\nwrote {fig_path}")
    print(json.dumps({
        n: {k: round(v, 6) for k, v in payload["convergence"][str(n)].items()
            if isinstance(v, float)}
        for n in sizes
    }, indent=2))
    for label in labels:
        print(label, {k: round(v, 6) for k, v in payload["cases"][label].items()})


if __name__ == "__main__":
    main()