"""Generate the thick-plate (Mindlin) bending-moment verification figure + data.

Every number written to ``docs/assets/case-fem-thick-plate-moments.json`` is
computed by the library functions in ``tensorfem.foundation_plate_bending``
(no hand transcription); the PNG is drawn only from that JSON, so figure and
report tables cannot drift apart.

Run with ``PYTHONPATH=src python scripts/gen_case_fem_thick_plate_moments.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from tensorfem.foundation_plate_bending import (  # noqa: E402
    compare_finite_plate_moments,
    navier_foundation_deflection,
    navier_foundation_moments,
    navier_mindlin_foundation_deflection,
    navier_mindlin_foundation_moments,
)

YOUNG, POISSON = 3.0e10, 0.3
SIDE, PRESSURE = 10.0, 1.0e4
TIMOSHENKO_CENTER_MOMENT = 0.0479          # nu = 0.3, Timoshenko table 8
ASSETS = Path(__file__).resolve().parents[1] / "docs" / "assets"
CENTRE = torch.tensor([[SIDE / 2, SIDE / 2]], dtype=torch.float64)
QUARTER = torch.tensor([[SIDE / 4, SIDE / 4]], dtype=torch.float64)


def centre_moments(thickness, **kwargs):
    args = dict(side_x=SIDE, side_y=SIDE, young=YOUNG, poisson=POISSON,
                thickness=thickness, pressure=PRESSURE, n_modes=99, **kwargs)
    thin = navier_foundation_moments(CENTRE, **args)
    thick = navier_mindlin_foundation_moments(CENTRE, **args)
    return float(thin[0, 0]), float(thick[0, 0])


def main() -> None:
    # 1. literature anchor: the thick series must reduce to 0.0479 q a^2.
    anchor_thin, anchor_thick = centre_moments(0.1)
    literature = TIMOSHENKO_CENTER_MOMENT * PRESSURE * SIDE**2

    # 2. identity audit: k = 0 -> Mindlin and Kirchhoff moments coincide.
    identity = {}
    for thickness in (0.1, 0.5, 1.0, 2.0, 3.0):
        args = dict(side_x=SIDE, side_y=SIDE, young=YOUNG, poisson=POISSON,
                    thickness=thickness, pressure=PRESSURE, n_modes=99)
        pts = torch.cat((CENTRE, QUARTER), dim=0)
        thick = navier_mindlin_foundation_moments(pts, **args)
        thin = navier_foundation_moments(pts, **args)
        identity[str(thickness)] = float(
            (thick - thin).abs().max() / thin.abs().max()
        )

    # 3. foundation-induced moment shift and its 1/t scaling.
    shifts = []
    for thickness in (0.5, 1.0, 2.0, 4.0):
        for bed in (1.0e5, 1.0e6):
            thin, thick = centre_moments(thickness, winkler=bed)
            shifts.append({
                "thickness": thickness,
                "slenderness": thickness / SIDE,
                "winkler": bed,
                "kirchhoff_center_mx": thin,
                "mindlin_center_mx": thick,
                "rel_shift": abs(thin - thick) / abs(thick),
            })

    # 4. FE vs exact thick-plate series, thick and thin.
    convergence = {}
    for thickness in (0.1, 1.0, 2.0):
        rows = {}
        for n in (16, 24, 32, 48):
            report = compare_finite_plate_moments(
                side_x=SIDE, side_y=SIDE, nx=n, ny=n,
                young=YOUNG, poisson=POISSON, thickness=thickness,
                pressure=PRESSURE, reference="mindlin",
            )
            rows[str(n)] = {
                key: float(report[key])
                for key in ("max_mx_rel_error", "max_my_rel_error",
                            "mxy_rel_error", "max_w_rel_error",
                            "center_mx", "center_mx_rel_error", "ndof")
            }
        convergence[str(thickness)] = rows

    # 5. reference-validity boundary: the same FE field, two truth models.
    boundary = {}
    for thickness in (0.1, 0.2, 1.0, 2.0):
        common = dict(side_x=SIDE, side_y=SIDE, nx=32, ny=32, thickness=thickness,
                      young=YOUNG, poisson=POISSON, pressure=PRESSURE)
        thin_ref = compare_finite_plate_moments(**common, reference="kirchhoff")
        thick_ref = compare_finite_plate_moments(**common, reference="mindlin")
        boundary[str(thickness)] = {
            "slenderness": thickness / SIDE,
            "kirchhoff_max_w_rel_error": float(thin_ref["max_w_rel_error"]),
            "mindlin_max_w_rel_error": float(thick_ref["max_w_rel_error"]),
            "kirchhoff_max_mx_rel_error": float(thin_ref["max_mx_rel_error"]),
            "mindlin_max_mx_rel_error": float(thick_ref["max_mx_rel_error"]),
        }

    # deflection series comparison at the centre (the shear correction itself)
    deflection = {}
    for thickness in (0.1, 0.5, 1.0, 2.0):
        args = dict(side_x=SIDE, side_y=SIDE, young=YOUNG, poisson=POISSON,
                    thickness=thickness, pressure=PRESSURE, n_modes=99)
        thin = float(navier_foundation_deflection(CENTRE, **args)[0])
        thick = float(navier_mindlin_foundation_deflection(CENTRE, **args)[0])
        deflection[str(thickness)] = {
            "kirchhoff_w_center": thin,
            "mindlin_w_center": thick,
            "shear_excess": (thick - thin) / thin,
        }

    payload = {
        "meta": {
            "case": "CASE-FEM-THICK-PLATE-MOMENTS",
            "anchor": "GAP-FEM-WINKLER",
            "geometry": {"side_x_m": SIDE, "side_y_m": SIDE,
                         "boundary": "hard simple support (w=0, tangential rotation=0)",
                         "pressure_Pa": PRESSURE},
            "material": {"young_Pa": YOUNG, "poisson": POISSON},
            "series": ("Mindlin-Reissner Navier solution: w=W sin(ax)sin(by), "
                       "theta_x=X cos(ax)sin(by), theta_y=Y sin(ax)cos(by)"),
            "moment_recovery": ("Mx=-D(aX+nu bY) sin sin, My=-D(bY+nu aX) sin sin, "
                                "Mxy=D(1-nu)/2 (bX+aY) cos cos"),
            "fe": "Q4 Mindlin plate with MITC/SRI transverse shear, 3 dof/node",
        },
        "literature_anchor": {
            "timoshenko_coefficient": TIMOSHENKO_CENTER_MOMENT,
            "reference_center_mx": literature,
            "kirchhoff_series_center_mx": anchor_thin,
            "mindlin_series_center_mx": anchor_thick,
            "kirchhoff_rel_error": abs(anchor_thin - literature) / literature,
            "mindlin_rel_error": abs(anchor_thick - literature) / literature,
        },
        "identity_kirchhoff_vs_mindlin_no_foundation": identity,
        "foundation_moment_shift": shifts,
        "fe_vs_mindlin_convergence": convergence,
        "reference_boundary": boundary,
        "deflection_shear_excess": deflection,
    }

    ASSETS.mkdir(parents=True, exist_ok=True)
    data_path = ASSETS / "case-fem-thick-plate-moments.json"
    data_path.write_text(json.dumps(payload, indent=2) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.3))

    ax = axes[0]
    for thickness, style in (("1.0", "s-"), ("2.0", "o-")):
        sizes = sorted(int(k) for k in convergence[thickness])
        mx = [convergence[thickness][str(n)]["max_mx_rel_error"] * 100 for n in sizes]
        mxy = [convergence[thickness][str(n)]["mxy_rel_error"] * 100 for n in sizes]
        ax.loglog(sizes, mx, style, label=f"max $M_x$, t/a={float(thickness)/SIDE:.1f}")
        ax.loglog(sizes, mxy, style, mfc="none",
                  label=f"$M_{{xy}}$, t/a={float(thickness)/SIDE:.1f}")
    first = convergence["2.0"]
    ref = [first["16"]["max_mx_rel_error"] * 100 * (16 / n) ** 2 for n in (16, 24, 32, 48)]
    ax.loglog([16, 24, 32, 48], ref, "k--", lw=1, label="$O(h^2)$ reference")
    ax.axhline(3.0, color="orange", lw=1, ls=":", label="precision bar 3 %")
    ax.set_xlabel("n (elements per side)")
    ax.set_ylabel("relative error [%]")
    ax.set_title("FE vs exact Mindlin series\n$q=10$ kPa, SS $10\\times10$ m, $n_{modes}=99$")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)

    ax = axes[1]
    labels = list(boundary)
    x = range(len(labels))
    thin = [boundary[k]["kirchhoff_max_w_rel_error"] * 100 for k in labels]
    thick = [boundary[k]["mindlin_max_w_rel_error"] * 100 for k in labels]
    ax.bar([i - 0.2 for i in x], thin, width=0.4,
           label="vs Kirchhoff series (thin-plate truth)")
    ax.bar([i + 0.2 for i in x], thick, width=0.4,
           label="vs Mindlin series (exact truth)")
    ax.set_yscale("log")
    ax.axhline(3.0, color="orange", lw=1, ls=":", label="precision bar 3 %")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"t/a={boundary[k]['slenderness']:.2f}" for k in labels], fontsize=8)
    ax.set_ylabel("FE deflection error [%], nx=32")
    ax.set_title("Reference-validity boundary\n(same FE field, two truth models)")
    ax.grid(alpha=0.3, axis="y", which="both")
    ax.legend(fontsize=7)

    ax = axes[2]
    for bed, marker in ((1.0e5, "o-"), (1.0e6, "s-")):
        rows = [r for r in shifts if r["winkler"] == bed]
        ax.loglog([r["thickness"] for r in rows],
                  [r["rel_shift"] * 100 for r in rows], marker,
                  label=f"Winkler k={bed:.0e} N/m$^3$")
    rows = [r for r in shifts if r["winkler"] == 1.0e6]
    law = [rows[0]["rel_shift"] * 100 * rows[0]["thickness"] / r["thickness"] for r in rows]
    ax.loglog([r["thickness"] for r in rows], law, "k--", lw=1, label="$\\propto k/(G t)$")
    ax.set_xlabel("plate thickness t [m]  (side 10 m)")
    ax.set_ylabel("|$M_x$ Kirchhoff - Mindlin| / Mindlin  [%]")
    ax.set_title("Shear deformation shifts moments only\nwhen a foundation is present")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)

    # markdown tables generated from the same payload (no hand transcription)
    md = ["<!-- generated by scripts/gen_case_fem_thick_plate_moments.py -->", ""]
    md.append("| FE vs exact Mindlin series | ndof | max|dMx|/Mx | max|dMy|/My | |dMxy|/Mxy | |dw|/w |")
    md.append("|---|---|---|---|---|---|")
    for thickness in ("0.1", "1.0", "2.0"):
        for n in ("16", "24", "32", "48"):
            row = convergence[thickness][n]
            md.append(
                f"| t/a={float(thickness)/SIDE:.2f}, n={n} | {int(row['ndof'])} | "
                f"{row['max_mx_rel_error']*100:.3f} % | {row['max_my_rel_error']*100:.3f} % | "
                f"{row['mxy_rel_error']*100:.3f} % | {row['max_w_rel_error']*100:.3f} % |"
            )
    md.append("")
    md.append("| t/a | FE w error vs Kirchhoff series | FE w error vs Mindlin series |")
    md.append("|---|---|---|")
    for key, row in boundary.items():
        md.append(f"| {row['slenderness']:.2f} | {row['kirchhoff_max_w_rel_error']*100:.3f} % "
                  f"| {row['mindlin_max_w_rel_error']*100:.3f} % |")
    md.append("")
    md.append("| t [m] | Winkler k [N/m^3] | centre Mx (Mindlin series) | |Mx_K - Mx_M|/Mx_M |")
    md.append("|---|---|---|---|")
    for row in shifts:
        md.append(f"| {row['thickness']:.1f} | {row['winkler']:.0e} | "
                  f"{row['mindlin_center_mx']:.1f} | {row['rel_shift']*100:.4f} % |")
    md.append("")
    (ASSETS / "case-fem-thick-plate-moments-tables.md").write_text("\n".join(md) + "\n")

    fig.tight_layout()
    fig_path = ASSETS / "case-fem-thick-plate-moments.png"
    fig.savefig(fig_path, dpi=140)
    print(f"wrote {data_path}\nwrote {fig_path}")
    print(json.dumps(payload["literature_anchor"], indent=2))
    print(json.dumps(payload["identity_kirchhoff_vs_mindlin_no_foundation"], indent=2))
    for key in ("1.0", "2.0"):
        for n, row in convergence[key].items():
            print(key, n, {k: round(v, 6) for k, v in row.items()})
    for key, row in boundary.items():
        print("boundary", key, {k: round(v, 6) for k, v in row.items()})
    for row in shifts:
        print("shift", round(row["thickness"], 3), row["winkler"],
              round(row["mindlin_center_mx"], 2), f"{row['rel_shift']*100:.4f}%")


if __name__ == "__main__":
    main()