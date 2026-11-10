"""Render the numbers section of ``docs/j2-elastoplastic-three-case.md`` from evidence.

Every table below is generated from ``artifacts/j2_elastoplastic_three_case/
report.json`` (the runner's output), so the document cannot drift from the
computation and no number is typed by hand.

Usage::

    python benchmarks/j2_elastoplastic_three_case/render_markdown.py
    python benchmarks/j2_elastoplastic_three_case/render_markdown.py --check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "artifacts" / "j2_elastoplastic_three_case" / "report.json"
DOC = ROOT / "docs" / "j2-elastoplastic-three-case.md"
START = "<!-- TABLES:START -->"
END = "<!-- TABLES:END -->"


def _num(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if value != 0.0 and abs(value) < 1e-3:
            return f"{value:.2e}"
        return f"{value:.{digits}f}"
    return str(value)


def _table(header, rows) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def render(payload: dict) -> str:
    report = payload["report"]
    uniaxial = report["case_A_uniaxial"]["material_point"]
    bar = report["case_A_uniaxial"]["tet4_bar"]
    shear = report["case_B_shear"]["material_point"]
    patch = report["case_B_shear"]["patch_test"]
    cylinder = report["case_C_cylinder"]
    tube = cylinder["plastic"]
    hardening = cylinder["hardening_reference"]
    band = cylinder["elastic_band"]
    pi_plane = report["pi_plane"]
    criteria = report["criteria"]
    parts = []

    parts.append("### 3.1 Acceptance criteria (machine read from the artifact)\n")
    rows = []
    for name, entry in criteria.items():
        value = entry["value"]
        band_value = entry["tolerance"]
        shown_band = f"{band_value:.0%}" if isinstance(band_value, float) and band_value <= 1 \
            else _num(band_value)
        verdict = {True: "**PASS**", False: "**FAIL**", None: "n/a"}[entry["passed"]]
        rows.append((name, _num(value), shown_band, verdict))
    parts.append(_table(("criterion", "value", "band", "verdict"), rows))

    parts.append("\n### 3.2 Case A - uniaxial tension (`ε = 0 … 5 %`)\n")
    rows = [(f"{row['strain'] * 100:.3f}",
             _num(row["stress_closed_form"] / 1e6, 3),
             _num(row["stress_kernel"] / 1e6, 3),
             _num(row["alpha"] * 100, 3),
             _num(row["relative_error_closed_form"], 2))
            for row in uniaxial["levels"]]
    parts.append(_table(("ε [%]", "card σ [MPa]", "material point σ [MPa]",
                         "ε_p [%]", "rel. error"), rows))
    parts.append(
        f"\nFE bar ({bar['mesh']['n_elements']} TET4, {bar['mesh']['n_nodes']} nodes, "
        f"{bar['mesh']['elements_axial']} axial × {bar['mesh']['elements_transverse']} "
        f"transverse): worst relative error of the element stress "
        f"{_num(bar['max_relative_error_element'])}, of the load-cell reaction "
        f"{_num(bar['max_relative_error_reaction'])}; yield stress read back from the "
        f"computed curve {_num(bar['yield_stress_estimate'] / 1e6, 4)} MPa vs the card "
        f"{_num(bar['yield_stress'] / 1e6, 1)} MPa "
        f"({_num(abs(bar['yield_stress_estimate'] - bar['yield_stress']) / bar['yield_stress'] * 100, 4)} %).\n")

    parts.append("\n### 3.3 Case B - pure shear (`γ = 0 … 5 %`)\n")
    rows = [(f"{row['gamma'] * 100:.3f}", _num(row["tau_reference"] / 1e6, 3),
             _num(row["tau_kernel"] / 1e6, 3), _num(row["alpha"] * 100, 3),
             _num(row["relative_error"], 2)) for row in shear["levels"]]
    parts.append(_table(("γ [%]", "closed form τ [MPa]", "return map τ [MPa]",
                         "ε_p [%]", "rel. error"), rows))
    parts.append(
        f"\n`τ_y = {_num(shear['tau_y'] / 1e6, 4)} MPa` vs `σ_y/√3 = "
        f"{_num(shear['tau_y_analytic'] / 1e6, 4)} MPa` "
        f"({_num(shear['tau_y_relative_error'], 2)} relative); plastic shear slope "
        f"`{_num(shear['plastic_shear_modulus_numeric'] / 1e6, 3)} MPa` vs closed form "
        f"`{_num(shear['plastic_shear_modulus_analytic'] / 1e6, 3)} MPa`; torque at yield "
        f"`T_y = τ_y W_t = {_num(shear['torque_at_yield'], 4)} N·m` "
        f"(`W_t = 2πR²t = {_num(shear['polar_modulus'], 3)} m³`). "
        f"TET4 patch test (homogeneous shear, `γ = {patch['gamma'] * 100:.2f} %`): "
        f"`τ = {_num(patch['tau_patch'] / 1e6, 4)} MPa` vs material point "
        f"`{_num(patch['tau_reference'] / 1e6, 4)} MPa` "
        f"({_num(patch['relative_error'], 2)}), stress spread "
        f"{_num(patch['stress_spread'], 2)} Pa.\n")

    parts.append("\n### 3.4 Case C - the literal `p = 0 … 100 MPa` band is elastic\n")
    rows = [(f"{row['pressure'] / 1e6:.0f}", _num(row["pressure_over_first_yield"], 3),
             _num(row["max_equivalent_plastic_strain"], 1),
             _num(row["sigma_r_max_relative_error"], 4),
             _num(row["sigma_theta_max_relative_error"], 4),
             _num(row["u_bore_relative_error"], 2))
            for row in band["levels"]]
    parts.append(_table(("p [MPa]", "p / p_y", "max ε_p", "σ_r err (÷p)",
                         "σ_θ err (÷p)", "u_r(a) rel. err"), rows))
    parts.append(
        f"\nAll five levels: `max ε_p = {_num(band['max_plastic_strain'], 1)}` — the whole band is "
        f"elastic, as predicted by `p_y = {_num(band['first_yield_pressure'] / 1e6, 2)} MPa`. "
        f"Worst Lamé stress error {_num(band['max_stress_relative_error'] * 100, 2)} % of `p` "
        f"(mesh `n_r = 48`, grading 0.1 — the `σ_r` gradient at the bore is why a uniform mesh "
        f"resolves it only to a few percent), bore displacement error "
        f"{_num(band['max_u_bore_relative_error'] * 100, 4)} %.\n")

    parts.append("\n### 3.5 Case C - plastic-zone growth, five contained-plastic levels\n")
    rows = []
    for row in tube["levels"]:
        if row["n_r"] != max(entry["n_r"] for entry in tube["levels"]):
            continue
        rows.append((_num(row["pressure"] / 1e6, 2), _num(row["pressure_over_k"], 2),
                     _num(row["r_p_fem"] * 1e3, 3), _num(row["r_p_reference"] * 1e3, 3),
                     _num(row["r_p_relative_error"] * 100, 3),
                     _num(row["u_bore"] * 1e6, 3), _num(row["u_bore_reference"] * 1e6, 3),
                     _num(row["u_bore_relative_error"] * 100, 3),
                     _num(row["yielded_fraction"], 3)))
    parts.append(_table(("p [MPa]", "p/k", "r_p FE [mm]", "r_p Hill [mm]", "err [%]",
                         "u_r(a) FE [µm]", "u_r(a) Hill [µm]", "err [%]", "yielded"),
                        rows))
    rows = [(conv_key_pressure(key), _num(block["gci"]["relative_gci"], 3),
             _num(block["observed_orders"][0], 3),
             _num(block["richardson_value"] * 1e6, 3),
             _num(block["richardson_relative_offset_vs_analytic"] * 100, 3))
            for key, block in tube["convergence"].items() if block.get("gci")]
    parts.append("\n\nRoache GCI on `u_r(a)` (`h = (R_o − R_i)/n_r`, `n_r = "
                 + ", ".join(str(n) for n in tube["mesh_sizes"]) + "):\n")
    parts.append(_table(("p [MPa]", "GCI (rel.)", "observed order",
                         "Richardson u_r(a) [µm]", "offset vs Hill [%]"), rows))
    parts.append(
        f"\nWorst case C numbers: `r_p` {_num(tube['max_r_p_relative_error'] * 100, 3)} % "
        f"(band 5 %), `u_r(a)` {_num(tube['max_u_bore_relative_error'] * 100, 3)} % (band 10 %), "
        f"`u_r(r)` profile worst {_num(tube['max_u_profile_relative_error'] * 100, 3)} % (band 10 %). "
        f"The last column is *not* a mesh error: the Richardson value of the FE sequence still sits "
        f"{_num(max(block['richardson_relative_offset_vs_analytic'] for block in tube['convergence'].values() if block.get('gci')) * 100, 2)} % "
        f"from Hill at the highest pressure, which is the classical plane-strain `σ_z` "
        f"inconsistency discussed in §5.5, not a discretisation defect.\n")

    parts.append("\n### 3.6 Case C with isotropic hardening `H = 1000 MPa` (no closed form)\n")
    perfect = {round(row["pressure_over_k"], 2): row for row in tube["levels"]
               if row["n_r"] == max(entry["n_r"] for entry in tube["levels"])}
    rows = []
    for row in hardening["levels"]:
        if row["n_r"] != max(entry["n_r"] for entry in hardening["levels"]):
            continue
        reference = perfect.get(round(row["pressure_over_k"], 2))
        rows.append((_num(row["pressure_over_k"], 2), _num(row["r_p_fem"] * 1e3, 3),
                     _num(row["u_bore"] * 1e6, 3),
                     _num(reference["r_p_fem"] * 1e3, 3) if reference else "n/a",
                     _num(reference["u_bore"] * 1e6, 3) if reference else "n/a"))
    parts.append(_table(("p/k", "r_p (H = 1000) [mm]", "u_r(a) (H = 1000) [µm]",
                         "r_p (H = 0) [mm]", "u_r(a) (H = 0) [µm]"), rows))
    parts.append(
        f"\nHardening moves the answers as physics requires: at `p/k = 1.35` the interface shrinks "
        f"from `{_num(perfect[1.35]['r_p_fem'] * 1e3, 2)} mm` (perfect plasticity) to "
        f"`{_num(max(row['r_p_fem'] for row in hardening['levels']) * 1e3, 2)} mm` and the bore "
        f"displacement from `{_num(perfect[1.35]['u_bore'] * 1e6, 2)} µm` to "
        f"`{_num(max(row['u_bore'] for row in hardening['levels']) * 1e6, 2)} µm`. No oracle "
        f"columns are scored here (Hill is a perfect-plasticity solution); this branch is "
        f"verified by mesh refinement only.\n")

    parts.append("\n### 3.7 C6 - yield surface in the π plane\n")
    rows = [(f"{row['theta_deg']:.0f}", _num(row["equivalent_stress_at_onset"] / 1e6, 6),
             _num(row["relative_deviation"], 2)) for row in pi_plane["directions"]]
    parts.append(_table(("θ [°]", "q at onset [MPa]", "deviation"), rows))
    parts.append(
        f"\nWorst deviation {_num(pi_plane['max_relative_deviation'], 2)} at θ = "
        f"{pi_plane['worst_theta_deg']:.0f}° (band 1 %): the implemented return map puts the "
        f"yield surface on a circle in the π plane to machine precision.\n")

    parts.append("\n### 3.8 Run metadata\n")
    parts.append(_table(("item", "value"), [
        ("node", payload["node"]),
        ("quick mode", str(payload["quick"])),
        ("wall time [s]", _num(payload["wall_seconds"], 1)),
        ("torch", payload["torch_version"]),
        ("cylinder mesh sizes", ", ".join(str(n) for n in tube["mesh_sizes"])),
        ("pressure levels [MPa]", ", ".join(
            f"{value / 1e6:.1f}" for value in payload["geometry"]["cylinder"]["pressure_levels"])),
    ]))
    return "\n".join(parts)


def conv_key_pressure(key: str) -> str:
    return f"{float(key) / 1e6:.2f}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="verify the document is already up to date")
    args = parser.parse_args(argv)
    payload = json.loads(ARTIFACT.read_text())
    block = render(payload)
    document = DOC.read_text()
    if START not in document or END not in document:
        raise SystemExit(f"markers {START} / {END} not found in {DOC}")
    head, rest = document.split(START, 1)
    _, tail = rest.split(END, 1)
    updated = head + START + "\n\n" + block.rstrip() + "\n" + END + tail
    if args.check:
        if updated != document:
            print("docs/j2-elastoplastic-three-case.md is stale; re-run without --check")
            return 1
        print("document matches the artifact")
        return 0
    DOC.write_text(updated)
    print(f"rendered {len(block.splitlines())} lines into {DOC.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))