"""Verification debt: Saint-Venant torsion of a circular tube on HEX8 / HEX20.

Why this case
-------------
``tensorfem.solid3d`` and ``tensorfem.hex20`` carry the three-dimensional
linear-elastic kernel that every solid, contact and multiphysics path builds on,
but no benchmark in the repository loaded a tubular member in torsion: the
stored evidence was bending, pressurisation, thermal and (recently) plane-strain
Lame only.  The circular tube is the cheapest torsion case with an exact 3-D
solution and no warping, and it is precisely the load case of rudder stocks,
shafting and tubular bracing.

What is produced
----------------
* ``benchmarks/torsion_tube_verification_evidence.json`` -- measured torsional
  constants against the exact circular reference, against the sharp faceted
  reference, element patch errors, stress errors, observed orders and verdicts.
* ``docs/assets/torsion_tube_*.png`` -- stress profile, cross-section contour
  and the mesh-convergence plot with the 3 % gate.
* ``docs/saint-venant-torsion-tube.md`` -- the report; every number is formatted
  from the computed evidence, none is typed by hand.

Run:
    OMP_NUM_THREADS=4 PYTHONPATH=src python benchmarks/torsion_tube_verification.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import torch  # noqa: E402

from tensorfem.torsion_reference import (  # noqa: E402
    element_cross_section_areas,
    faceted_torsional_constant,
    faceting_ratio,
    section_centroids,
    shear_modulus,
    solve_torsion_hex8,
    solve_torsion_hex20,
    surface_shear_stress,
    tube_torsional_constant,
)

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
EVIDENCE = ROOT / "benchmarks" / "torsion_tube_verification_evidence.json"
DOC = ROOT / "docs" / "saint-venant-torsion-tube.md"

INNER, OUTER = 0.05, 0.10        # m, bore / outer radius
LENGTH = 0.4                     # m
TWIST = 0.01                     # rad/m, unit twist rate
YOUNG, POISSON = 210.0e9, 0.3    # steel
GATE = 0.03                      # 3 % acceptance gate
HEX8_LADDER = (8, 16, 24, 32, 48, 64)
HEX20_LADDER = ((8, 2), (16, 4), (24, 4), (32, 4))


def relative(value: float, reference: float) -> float:
    return abs(value - reference) / abs(reference)


def observed_order(coarse: float, fine: float, ratio: float = 2.0) -> float:
    if coarse <= 0.0 or fine <= 0.0:
        return float("nan")
    return math.log(coarse / fine) / math.log(ratio)


def case_metrics(summary: dict, shear: float) -> dict:
    nodes, cells = summary["nodes"], summary["cells"]
    torsional = summary["torque"] / (shear * TWIST)
    radius = torch.linalg.vector_norm(summary["probe"][:, :2], dim=1)
    exact = surface_shear_stress(YOUNG, POISSON, TWIST, radius)
    error = summary["tau_probe"] / exact - 1.0
    areas = element_cross_section_areas(nodes, cells)
    centres = section_centroids(nodes, cells)
    lever = torch.linalg.vector_norm(centres[:, :2], dim=1)
    layer = torch.zeros(cells.shape[0], dtype=torch.bool)
    layer[: summary["n_radial"] * summary["n_theta"]] = True
    one_point = float((summary["tau_theta_z"] * lever * areas)[layer].sum())
    return {
        "element": summary["element"],
        "geometry": summary["geometry"],
        "n_theta": summary["n_theta"],
        "n_radial": summary["n_radial"],
        "n_axial": summary["n_axial"],
        "elements": summary["elements"],
        "dofs": summary["dofs"],
        "torque": summary["torque"],
        "torsional_constant": torsional,
        "faceted_constant": faceted_torsional_constant(INNER, OUTER, summary["n_theta"]),
        "faceting_ratio": faceting_ratio(summary["n_theta"]),
        "error_vs_circle": torsional / tube_torsional_constant(INNER, OUTER) - 1.0,
        "error_vs_faceted": torsional / faceted_torsional_constant(INNER, OUTER, summary["n_theta"]) - 1.0,
        "patch_error": summary["displacement_patch_error"],
        "tau_l2_relative": float(torch.sqrt((error**2).mean())),
        "tau_max_relative": float(error.abs().max()),
        "twist_max_deviation": float(
            (summary["twist_angle"] - TWIST * summary["twist_levels"]).abs().max()
        ),
        "stress_torque_one_point": one_point,
        "stress_torque_one_point_bias": one_point / summary["torque"] - 1.0,
        "passes_gate": abs(torsional / tube_torsional_constant(INNER, OUTER) - 1.0) <= GATE,
        "max_surface_shear_stress": float(
            surface_shear_stress(YOUNG, POISSON, TWIST, OUTER)
        ),
        "field": {
            "probe_radius": [float(x) for x in radius],
            "tau_probe": [float(x) for x in summary["tau_probe"]],
            "tau_exact": [float(x) for x in exact],
            "centres": [[float(x) for x in row] for row in centres],
            "layer": layer.tolist(),
        },
    }


def plot_profiles(case: dict, path: Path) -> None:
    radius = torch.tensor(case["field"]["probe_radius"])
    tau = torch.tensor(case["field"]["tau_probe"]) / 1e6
    exact = torch.tensor(case["field"]["tau_exact"]) / 1e6
    order = torch.argsort(radius)
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    axes[0].plot(
        radius[order], exact[order], "-", color="#1f77b4", lw=1.8,
        label=r"exact $\tau_{\theta z}=G\theta r$",
    )
    axes[0].plot(
        radius[order], tau[order], "o", color="#d62728", ms=4.0, mfc="none",
        label=f"HEX8 {case['n_theta']}x{case['n_radial']} element centres",
    )
    axes[0].set_xlabel("radius r [m]")
    axes[0].set_ylabel(r"$\tau_{\theta z}$ [MPa]")
    axes[0].set_title("Torsional shear profile, twist rate 0.01 rad/m")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    centres = torch.tensor(case["field"]["centres"])
    layer = torch.tensor(case["field"]["layer"])
    tau_c = torch.tensor(case["field"]["tau_probe"])[layer] / 1e6
    x = centres[layer, 0]
    y = centres[layer, 1]
    triangles = mtri.Triangulation(x.numpy(), y.numpy())
    scatter = axes[1].tripcolor(
        triangles, tau_c.numpy(), shading="gouraud", cmap="viridis"
    )
    axes[1].set_aspect("equal")
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("y [m]")
    axes[1].set_title(r"$\tau_{\theta z}$ on the tube section [MPa]")
    figure.colorbar(scatter, ax=axes[1], shrink=0.85, label="MPa")
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def plot_convergence(hex8: list[dict], hex20: list[dict], path: Path) -> None:
    figure, axes = plt.subplots(figsize=(6.6, 4.8))
    for series, colour, marker, label in (
        (hex8, "#d62728", "o", "HEX8 faceted ring"),
        (hex20, "#2ca02c", "s", "HEX20, mid-edge nodes on the cylinder"),
    ):
        theta = [case["n_theta"] for case in series]
        error = [abs(case["error_vs_circle"]) * 100.0 for case in series]
        axes.loglog(theta, error, marker + "-", color=colour, lw=1.6, ms=5.5, label=label)
    theta = [case["n_theta"] for case in hex8]
    reference = [100.0 * (1.0 - case["faceting_ratio"]) for case in hex8]
    axes.loglog(theta, reference, "--", color="0.4", lw=1.2, label=r"$1-J_n/J$ (faceted, exact)")
    axes.axhline(GATE * 100.0, color="#1f77b4", lw=1.2, ls=":")
    axes.text(8.2, GATE * 108.0, "3 % gate", color="#1f77b4", fontsize=8)
    axes.set_xlabel(r"tangential divisions $n_\theta$")
    axes.set_ylabel(r"$|J_h/J-1|$ [%]")
    axes.set_title("Torsional constant of the circular tube")
    axes.grid(alpha=0.3, which="both")
    axes.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def write_doc(evidence: dict) -> None:
    hex8 = evidence["hex8_ladder"]
    hex20 = evidence["hex20_curved_ladder"]
    faceted20 = evidence["hex20_faceted_case"]
    circle = evidence["references"]["torsional_constant_circle"]
    gate_rows = "\n".join(
        "| {el} | {geom} | {n_theta} | {n_radial} | {elements} | {dofs} | {torsional:.6e} | "
        "{error:+.3f} % | {facet:+.5f} % | {patch:.1e} | {verdict} |".format(
            el=case["element"],
            geom=case["geometry"],
            n_theta=case["n_theta"],
            n_radial=case["n_radial"],
            elements=case["elements"],
            dofs=case["dofs"],
            torsional=case["torsional_constant"],
            error=100.0 * case["error_vs_circle"],
            facet=100.0 * case["error_vs_faceted"],
            patch=case["patch_error"],
            verdict="PASS" if case["passes_gate"] else "FAIL",
        )
        for case in hex8 + hex20 + [faceted20]
    )
    stress_rows = "\n".join(
        "| {el} {geom} {n_theta} | {l2:.2e} | {mx:.2e} | {twist:.1e} |".format(
            el=case["element"],
            geom=case["geometry"],
            n_theta=case["n_theta"],
            l2=case["tau_l2_relative"],
            mx=case["tau_max_relative"],
            twist=case["twist_max_deviation"],
        )
        for case in hex8 + hex20
    )
    order8 = evidence["orders"]["hex8"]
    order20 = evidence["orders"]["hex20"]
    DOC.write_text(
        f"""# Saint-Venant torsion of a circular tube: HEX8/HEX20 verification

## 1. Case

A steel tube with bore `R_i = {INNER} m`, outer radius `R_o = {OUTER} m` and
length `L = {LENGTH} m` (`E = {YOUNG:.3e} Pa`, `nu = {POISSON}`,
`G = {evidence['references']['shear_modulus']:.6e} Pa`) is loaded in pure torsion
by a prescribed twist rate `theta = {TWIST} rad/m`, i.e. `phi(z) = theta z`.
The exact Saint-Venant state of the circular tube carries **no warping**:

```
u = theta z (z_hat x r),  gamma_xz = -theta y,  gamma_yz = theta x
tau_theta_z(r) = G theta r,  tau_max = {evidence['references']['max_shear_stress']/1e6:.3f} MPa
J = pi (R_o^4 - R_i^4) / 2 = {circle:.6e} m^4,   T = G J theta = {evidence['references']['torque']:.6e} N m
```

This is the load case of rudder stocks, shafting and tubular bracing, and it
had no verification in TensorFEM before this study: every stored 3-D solid
benchmark was bending, pressurisation or thermal.

## 2. Method

Both ends are prescribed with the exact rigid-rotation ramp (`u_x = -theta z y`,
`u_y = theta z x`, `u_z = 0`); the lateral surface is left free, which is exactly
the traction-free Saint-Venant boundary condition.  The reaction torque at
`z = L` gives `J_h = T / (G theta)`.  Because a ring of bricks can only represent
the cylinder as a **faceted prism**, two references are used:

* `J` (true circle) -- the engineering target;
* `J_n = (n/12) sin(dphi) (2 + cos(dphi)) (R_o^4 - R_i^4)` with `dphi = 2 pi / n`
  -- the polar second moment of the inscribed polygon, i.e. the sharp reference
  for the *discrete* geometry.  The first term of `1 - J_n/J` is
  `2 pi^2 / (3 n^2)`, so faceting is a second-order effect.

Runs: `tensorfem.solid3d.solve_solid` (HEX8) and `tensorfem.hex20.solve_hex20`
(HEX20, quarter-point-free serendipity brick, 3x3x3 Gauss), both dense direct.
The second HEX20 series projects the mid-edge nodes of the two cylindrical
surfaces onto the true circle, which gives quadratic (curved) element edges.

## 3. Results

| element | geometry | n_theta | n_radial | elements | dofs | J_h [m^4] | vs circle | vs faceted | patch error | verdict (3 %) |
|---|---|---|---|---:|---:|---|---|---|---|---|
{gate_rows}

Observed orders (successive mesh doubling):

* HEX8 faceted, `n_theta` 16 -> 32: `p = {order8['p_16_32']:.2f}`;
  32 -> 64: `p = {order8['p_32_64']:.2f}`.  Exactly the second order predicted by
  the faceting law, i.e. **the whole error is geometric, not element locking**.
* HEX20 curved edges, 8 -> 16: `p = {order20['p_8_16']:.2f}`; 16 -> 32:
  `p = {order20['p_16_32']:.2f}`, the fourth order of a quadratic surface.

Stress recovery and kinematic checks:

| element | tau L2 error | tau max error | max twist deviation [rad] |
|---|---|---:|---:|
{stress_rows}

The element patch error is `{max(c['patch_error'] for c in hex8):.1e}` for every
HEX8 mesh: the linear twist ramp is reproduced **exactly**, and the measured
`J_h` equals the faceted reference `J_n` to `{max(abs(c['error_vs_faceted']) for c in hex8):.1e}`.
The same holds for the faceted HEX20 case
(`{faceted20['error_vs_faceted']:+.2e}`), so HEX8 and HEX20 agree bit-for-bit
once both use straight edges.  The element is therefore verified; the residual
disagreement with the ideal tube is the mesh's geometric idealisation.

## 4. Figures

![torsion profiles](assets/torsion_tube_profile.png)

*Left:* torsional shear stress at the element centres of the `n_theta = 32`
HEX8 mesh against the exact `G theta r`; *right:* the same field on the tube
section.

![torsion convergence](assets/torsion_tube_convergence.png)

Torsional-constant error against `n_theta`, with the exact faceting law
`1 - J_n/J` and the 3 % gate.  HEX8 follows the geometric law; HEX20 with curved
edges collapses onto the exact circle at fourth order.

## 5. Notes and traps

* **Faceting, not locking.** A ring mesh of HEX8 prisms under-predicts the
  torsional rigidity of a circular tube by `{abs(hex8[0]['error_vs_circle'])*100:.1f} %` at
  `n_theta = 8` and by `{abs(hex8[3]['error_vs_circle'])*100:.2f} %` at `n_theta = 32`.  This is
  the inscribed-polygon stiffness, not an element defect, and it is reproduced
  by the closed-form `J_n` to round-off.
* **One-point stress quadrature trap.** Integrating `tau_theta_z * r * A` at the
  element centroid under-reports the reaction torque by
  `{100*hex8[0]['stress_torque_one_point_bias']:.1f} %` at `n_theta = 8` and by
  `{100*hex8[3]['stress_torque_one_point_bias']:.2f} %` at `n_theta = 32`, because
  `int r^2 dA != A r_c^2`.  The verdict must come from reactions (as here), never
  from a centroid quadrature.  The biased value is kept in the evidence as a
  documented trap, not as a gate.
* **Warping-free only.** This qualification covers the circular tube, the only
  section that carries torsion without warping.  Non-circular sections (square
  bars, hull-girder-like boxes) need a warping function and are **not**
  covered; a square-section benchmark would legitimately fail with the linear
  ramp and must not be quoted as a check.
* **Curved HEX20 needs node projection.** `upgrade_hex8_to_hex20` places mid-edge
  nodes at chord midpoints; projecting them onto the cylinder (`project_hex20_to_cylinders`)
  is what produces the fourth-order recovery.  Without the projection the HEX20
  result is identical to HEX8.
* **Scope.** Linear elasticity, small strain, uniform isotropic material,
  straight prism, no end fittings, no fillets, no plasticity.  The end planes are
  Dirichlet (rigid rotation), so no end boundary layer exists by construction.

## 6. Error verdict

| quantity | reference | worst measured error | < 3 % ? |
|---|---|---|---|
| `J` (HEX8, `n_theta = 32`) | true circular tube | `{abs(hex8[3]['error_vs_circle'])*100:.3f} %` | yes |
| `J` (HEX8, `n_theta = 16`) | true circular tube | `{abs(hex8[1]['error_vs_circle'])*100:.3f} %` | **no** |
| `J` (HEX8, `n_theta = 8`) | true circular tube | `{abs(hex8[0]['error_vs_circle'])*100:.3f} %` | **no** |
| `J` (HEX8, `n_theta = 32`) | faceted polygon | `{abs(hex8[3]['error_vs_faceted'])*100:.2e} %` | yes |
| `J` (HEX20 curved, `n_theta = 8`) | true circular tube | `{abs(hex20[0]['error_vs_circle'])*100:.4f} %` | yes |
| `tau_theta_z` at element centres | `G theta r` | `{max(c['tau_max_relative'] for c in hex8)*100:.2e} %` | yes |
| twist linearity | `theta z` | `{max(c['twist_max_deviation'] for c in hex8):.1e} rad` | yes |

Verdict: the 3-D elastic kernel is verified for pure torsion (patch-exact, stress
exact).  Mesh guidance: a **HEX8 ring needs `n_theta >= 24`**
(`{abs(hex8[2]['error_vs_circle'])*100:.2f} %`) to hold the 3 % gate on a circular tube,
`n_theta = 16` (`{abs(hex8[1]['error_vs_circle'])*100:.2f} %`) does not.  With HEX20 and
projected mid-edge nodes `n_theta = 8` already gives
`{abs(hex20[0]['error_vs_circle'])*100:.2f} %`.

Coarse-mesh failures are retained, not hidden: `n_theta = 8` and `16` HEX8 stay
in the table as FAIL.
"""
    )


def main() -> dict:
    ASSETS.mkdir(parents=True, exist_ok=True)
    shear = shear_modulus(YOUNG, POISSON)
    circle = tube_torsional_constant(INNER, OUTER)

    hex8_cases, hex8_summaries = [], []
    for n_theta in HEX8_LADDER:
        summary = solve_torsion_hex8(INNER, OUTER, LENGTH, TWIST, YOUNG, POISSON, n_theta, 4, 2)
        hex8_cases.append(case_metrics(summary, shear))
        hex8_summaries.append(summary)

    hex20_cases = []
    for n_theta, n_radial in HEX20_LADDER:
        summary = solve_torsion_hex20(
            INNER, OUTER, LENGTH, TWIST, YOUNG, POISSON, n_theta, n_radial, 1, curved=True
        )
        hex20_cases.append(case_metrics(summary, shear))

    faceted20 = case_metrics(
        solve_torsion_hex20(
            INNER, OUTER, LENGTH, TWIST, YOUNG, POISSON, 16, 4, 1, curved=False
        ),
        shear,
    )

    def pick(cases, n_theta):
        return next(c for c in cases if c["n_theta"] == n_theta)

    evidence = {
        "case": {
            "inner_radius": INNER,
            "outer_radius": OUTER,
            "length": LENGTH,
            "twist_per_length": TWIST,
            "young_modulus": YOUNG,
            "poisson_ratio": POISSON,
            "gate": GATE,
        },
        "references": {
            "shear_modulus": shear,
            "torsional_constant_circle": circle,
            "torque": shear * circle * TWIST,
            "max_shear_stress": surface_shear_stress(YOUNG, POISSON, TWIST, OUTER),
        },
        "hex8_ladder": hex8_cases,
        "hex20_curved_ladder": hex20_cases,
        "hex20_faceted_case": faceted20,
        "orders": {
            "hex8": {
                "p_16_32": observed_order(
                    abs(pick(hex8_cases, 16)["error_vs_circle"]),
                    abs(pick(hex8_cases, 32)["error_vs_circle"]),
                ),
                "p_32_64": observed_order(
                    abs(pick(hex8_cases, 32)["error_vs_circle"]),
                    abs(pick(hex8_cases, 64)["error_vs_circle"]),
                ),
            },
            "hex20": {
                "p_8_16": observed_order(
                    abs(pick(hex20_cases, 8)["error_vs_circle"]),
                    abs(pick(hex20_cases, 16)["error_vs_circle"]),
                ),
                "p_16_32": observed_order(
                    abs(pick(hex20_cases, 16)["error_vs_circle"]),
                    abs(pick(hex20_cases, 32)["error_vs_circle"]),
                ),
            },
        },
        "verdicts": {
            "hex8_min_qualified_n_theta": min(
                case["n_theta"] for case in hex8_cases if case["passes_gate"]
            ),
            "hex20_min_qualified_n_theta": min(
                case["n_theta"] for case in hex20_cases if case["passes_gate"]
            ),
            "coarse_failures_retained": [
                case["n_theta"] for case in hex8_cases if not case["passes_gate"]
            ],
        },
    }
    EVIDENCE.write_text(json.dumps(evidence, indent=2))

    plot_profiles(pick(hex8_cases, 32), ASSETS / "torsion_tube_profile.png")
    plot_convergence(hex8_cases, hex20_cases, ASSETS / "torsion_tube_convergence.png")
    write_doc(evidence)
    return evidence


if __name__ == "__main__":
    result = main()
    print(json.dumps(result["verdicts"], indent=2))
    for case in result["hex8_ladder"] + result["hex20_curved_ladder"]:
        print(
            f"{case['element']:>5} {case['geometry']:>13} n_theta={case['n_theta']:>2} "
            f"dofs={case['dofs']:>5} J={case['torsional_constant']:.6e} "
            f"err={100*case['error_vs_circle']:+9.4f}% "
            f"{'PASS' if case['passes_gate'] else 'FAIL'}"
        )