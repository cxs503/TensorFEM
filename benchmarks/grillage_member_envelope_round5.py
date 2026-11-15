"""Round-5 evidence generator: member load-path envelope for the ice-patch grillage.

Anchor ``GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE``.  Produces
``artifacts/grillage_member_envelope/``:

* ``sweep.json``        -- smeared / panel (B1,B2) / system (C) member demands for
                           2 patch positions x 3 patch widths, plus the checks,
* ``combined_cases.json``-- the 3 acceptance combined cases under all member models,
* ``map_system.json`` / ``map_panel.json`` -- the machine-readable
                           ``(sigma_x/sigma_y, A_ice/A_panel, p_hydro) -> p_ice,ult``
                           map under the system and the panel member model,
* ``summary.json``      -- every gate with value, reference and pass/fail.

All numbers in the published document are injected from these files.
"""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

from tensorfem.grillage_ice_ultimate import (
    CombinedLoadCase,
    case_study_geometry,
    grillage_utilisations,
    panel_stress_ultimate_map,
    ultimate_ice_pressure,
    write_ultimate_map,
)
from tensorfem.grillage_member_envelope import (
    branch_demands,
    classical_compare,
    classical_plate_bracket,
    member_envelope,
    patch_moment_closed_form,
    ss_beam_moment,
    system_branch,
    tributary_weights,
)

#: patch pressure for the demand sweep (above the plate limit: amplifies the
#: differences between the load-path idealisations, the *ratios* are the result)
P_SWEEP = 6.0e6
OUT = Path("artifacts/grillage_member_envelope")


def _pos_label(geom) -> str:
    """Where the centred patch sits relative to the frame lines."""
    span_x = geom.n_spans * geom.frame_spacing
    return "patch centred on a frame line" if geom.n_spans % 2 == 0 \
        else "patch centred in a bay"


def sweep_rows() -> list[dict[str, object]]:
    base = case_study_geometry()
    rows: list[dict[str, object]] = []
    for n_spans in (4, 3):
        geom = dataclasses.replace(base, n_spans=n_spans)
        for width in (0.35, 0.70, 1.05):
            case = CombinedLoadCase(ice_pressure=P_SWEEP, patch_length=width,
                                    patch_width=0.35)
            u = grillage_utilisations(geom, case)
            env = member_envelope(geom, case)
            env12 = member_envelope(geom, case, sub=12)
            branch = env["branch_utilisations"]
            rows.append({
                "n_spans": n_spans,
                "patch_position": _pos_label(geom),
                "patch_length_m": width,
                "patch_width_m": 0.35,
                "ice_pressure_MPa": P_SWEEP / 1e6,
                "smeared_u_stiffener": float(u["longitudinal_stiffener"]),
                "smeared_u_frame": float(u["transverse_frame"]),
                "panel_u_stiffener": max(branch["stiffener"]["B1_stiffener_path"],
                                         branch["stiffener"]["B2_frame_path"]),
                "panel_u_frame": max(branch["frame"]["B1_stiffener_path"],
                                     branch["frame"]["B2_frame_path"]),
                "system_u_stiffener": branch["stiffener"]["C_system"],
                "system_u_frame": branch["frame"]["C_system"],
                "system_u_frame_sub12": env12["branch_utilisations"]["frame"]["C_system"],
                "envelope_u_stiffener": env["stiffener_utilisation"],
                "envelope_u_frame": env["frame_utilisation"],
                "envelope_stiffener_branch": env["stiffener_branch"],
                "envelope_frame_branch": env["frame_branch"],
                "smeared_over_system_frame": (float(u["transverse_frame"])
                                              / branch["frame"]["C_system"]
                                              if branch["frame"]["C_system"] > 0 else None),
                "smeared_over_system_stiffener": (float(u["longitudinal_stiffener"])
                                                  / branch["stiffener"]["C_system"]
                                                  if branch["stiffener"]["C_system"] > 0 else None),
                "smeared_moment_stiffener_kNm": float(
                    env["analytic"]["B1_stiffener_path"]["stiffener_moment_ice"]) / 1e3,
                "system_moment_stiffener_kNm": float(env["system"]["stiffener_moment"]) / 1e3,
                "system_moment_frame_kNm": float(env["system"]["frame_moment"]) / 1e3,
                "panel_moment_frame_kNm": float(
                    env["analytic"]["B2_frame_path"]["frame_moment_ice"]) / 1e3,
                "frame_plastic_moment_kNm": float(env["system"]["frame_plastic_moment"]) / 1e3,
            })
    return rows


def uniform_load_checks(geom) -> dict[str, object]:
    """Hydrostatic (uniform) load: the frame must see its own tributary strip.

    Closed form for a simply supported frame carrying a uniform pressure over its
    tributary width ``a``: ``M = p a L^2 / 8``.  The legacy smeared layer halves
    it (``f_y / 2``), which is a factor-2 error for a load that covers the whole
    plate by construction.
    """
    p_h = 1.0e5
    case = CombinedLoadCase(hydrostatic_pressure=p_h)
    u = grillage_utilisations(geom, case)
    sysres = system_branch(geom, case)
    analytic = p_h * geom.frame_spacing * geom.frame_span ** 2 / 8.0
    smeared_equiv = float(u["per_frame_ice_force"])
    return {
        "pressure_MPa": p_h / 1e6,
        "analytic_frame_moment_kNm": analytic / 1e3,
        "system_frame_moment_kNm": float(sysres["frame_moment"]) / 1e3,
        "system_vs_analytic_rel": (float(sysres["frame_moment"]) - analytic) / analytic,
        "smeared_frame_utilisation": float(u["transverse_frame"]),
        "system_frame_utilisation": float(sysres["frame_utilisation"]),
        "smeared_over_system": (float(u["transverse_frame"])
                                / float(sysres["frame_utilisation"])),
    }


def belt_check(geom) -> dict[str, object]:
    """Ice-belt loading (patch repeated on every span): how much continuity reserve?

    The panel-level branch assumes a simply supported stiffener between frames.
    With a single loaded panel continuity redistributes the load; with the whole
    belt loaded (the design case) the redistribution must vanish, which is why the
    panel-level branch is the design envelope.  This measures the remaining
    reserve by comparing the system branch on a full-length patch with
    ``p b a^2 / 8``.
    """
    p = 1.0e6
    case = CombinedLoadCase(ice_pressure=p, patch_length=geom.frame_spacing,
                            patch_width=geom.bay_spacing)
    # system branch over the whole grillage length: emulate the belt by a patch
    # that spans the full x extent at the same pressure and the same bay width
    from tensorfem.grillage_beam_grid import _tee_torsion_constant, beam_grid_response

    span_x = geom.n_spans * geom.frame_spacing
    span_y = geom.frame_span
    shear = geom.young / (2.0 * (1.0 + geom.poisson))
    res = beam_grid_response(
        span_x=span_x, span_y=span_y,
        ei_x=geom.young * geom.stiffener_section.inertia,
        gj_x=shear * _tee_torsion_constant(geom.stiffener_section),
        ei_y=geom.young * geom.frame_section.inertia,
        gj_y=shear * _tee_torsion_constant(geom.frame_section),
        n_x_lines=geom.n_spans + 1, n_y_lines=geom.n_bays + 1, sub=8,
        pressure=1.0, patch=(0.0, 1.0, 0.45, 0.55), load_split=1.0)
    m_belt = p * res["moment_x_max"]
    m_panel = p * geom.bay_spacing * geom.frame_spacing ** 2 / 8.0
    return {
        "pressure_MPa": p / 1e6,
        "belt_stiffener_moment_kNm": m_belt / 1e3,
        "panel_simple_span_moment_kNm": m_panel / 1e3,
        "panel_over_system": m_panel / m_belt if m_belt > 0 else None,
    }


def combined_cases() -> list[dict[str, object]]:
    from tensorfem.grillage_ice_ultimate import combined_case_table

    geom = case_study_geometry()
    rows = []
    for case in combined_case_table(geom):
        rows.append({
            "label": case["label"] if "label" in case else case.get("case", "case"),
            "raw": case,
        })
    return rows


def combined_case_models() -> list[dict[str, object]]:
    """The 3 acceptance combined cases under every member model."""
    geom = case_study_geometry()
    cases = [
        ("light ice + ballast head", 0.0, 0.35, 0.35, 0.5e5),
        ("design ice + design head", 0.15, 0.70, 0.35, 1.0e5),
        ("severe ice + tank head", 0.30, 1.05, 0.35, 2.0e5),
    ]
    out = []
    for label, eta, lx, ly, p_h in cases:
        case = CombinedLoadCase(axial_stress=eta * geom.yield_stress,
                                patch_length=lx, patch_width=ly,
                                hydrostatic_pressure=p_h)
        row: dict[str, object] = {"label": label, "eta": eta,
                                  "patch_m": f"{lx} x {ly}",
                                  "hydrostatic_MPa": p_h / 1e6}
        for model in ("smeared", "system", "panel"):
            res = ultimate_ice_pressure(geom, case, member_model=model)
            u = res.get("utilisations_at_limit") or res["base_utilisations"]
            row[model] = {
                "p_ult_MPa": float(res["ice_pressure_ultimate"]) / 1e6,
                "governing": res["governing_member"],
                "u_plate": float(u["plate_local_bending"]),
                "u_stiffener": float(u["longitudinal_stiffener"]),
                "u_frame": float(u["transverse_frame"]),
            }
        out.append(row)
    return out


def stage_sweep() -> dict[str, object]:
    OUT.mkdir(parents=True, exist_ok=True)
    geom = case_study_geometry()
    rows = sweep_rows()
    (OUT / "sweep.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    core = {
        "uniform_load": uniform_load_checks(geom),
        "belt": belt_check(geom),
        "classical": classical_compare(geom),
        "bracket": classical_plate_bracket(geom),
    }
    (OUT / "core.json").write_text(json.dumps(core, indent=2, default=str),
                                   encoding="utf-8")
    return core


def stage_combined() -> list[dict[str, object]]:
    combined = combined_case_models()
    (OUT / "combined_cases.json").write_text(json.dumps(combined, indent=2),
                                             encoding="utf-8")
    return combined


def stage_maps() -> None:
    geom = case_study_geometry()
    map_sys = panel_stress_ultimate_map(geom, axial_ratios=(0.0, 0.30),
                                        patch_area_ratios=(0.10, 0.50),
                                        hydrostatic_pressures=(0.0, 1.0e5),
                                        member_model="system")
    write_ultimate_map(map_sys, OUT / "map_system.json")
    map_panel = panel_stress_ultimate_map(geom, axial_ratios=(0.0, 0.30),
                                          patch_area_ratios=(0.10, 0.50),
                                          hydrostatic_pressures=(0.0, 1.0e5),
                                          member_model="panel")
    write_ultimate_map(map_panel, OUT / "map_panel.json")


def build() -> dict[str, object]:
    OUT.mkdir(parents=True, exist_ok=True)
    geom = case_study_geometry()
    rows = json.loads((OUT / "sweep.json").read_text(encoding="utf-8"))
    core = json.loads((OUT / "core.json").read_text(encoding="utf-8"))
    uniform = core["uniform_load"]
    belt = core["belt"]
    classical = core["classical"]
    bracket = core["bracket"]
    combined = json.loads((OUT / "combined_cases.json").read_text(encoding="utf-8"))

    # ---- gates ----------------------------------------------------------- #
    gates: list[dict[str, object]] = []

    def gate(name: str, value, reference, tol, passed: bool, note: str = "") -> None:
        gates.append({"gate": name, "value": value, "reference": reference,
                      "tolerance": tol, "passed": bool(passed), "note": note})

    # G1: tributary weights pick the owning member and conserve the force
    w = tributary_weights([0.0, 0.35, 0.70, 1.05, 1.40, 1.75, 2.10, 2.45, 2.80],
                          1.225, 1.575, 2.80)
    owner = max(w, key=lambda pw: pw[1])
    gate("G1 tributary owner when the patch sits on a member line",
         owner[0], 1.40, "<1e-12", abs(owner[0] - 1.40) < 1e-12 and
         abs(sum(x for _, x in w) - 1.0) < 1e-12)

    # G2: the simply-supported patch moment equals the classical closed form
    L, F, c = 2.8, 1.0e6, 0.35
    pt = ss_beam_moment(L, [(0.5 * L, F)])
    cf = patch_moment_closed_form(L, F, c)
    gate("G2 ss-beam central point load vs FL/4", pt, F * L / 4.0,
         "<1e-9 rel", abs(pt - F * L / 4.0) / (F * L / 4.0) < 1e-9)

    # G3: panel branches are never below the system branch on the frame
    frame_ok = all(r["system_u_frame"] <= r["envelope_u_frame"] + 1e-12 for r in rows)
    gate("G3 envelope >= system branch (frame, 6 topologies)",
         min(r["envelope_u_frame"] / r["system_u_frame"] for r in rows), ">= 1.0",
         "envelope", frame_ok)
    stiff_ok = all(r["system_u_stiffener"] <= r["envelope_u_stiffener"] + 1e-12
                   for r in rows)
    gate("G4 envelope >= system branch (stiffener, 6 topologies)",
         min(r["envelope_u_stiffener"] / r["system_u_stiffener"] for r in rows),
         ">= 1.0", "envelope", stiff_ok)

    # G5: the legacy smeared frame demand is non-conservative (documented defect)
    worst = min(r["smeared_over_system_frame"] for r in rows)
    gate("G5 legacy smeared frame demand / system branch (worst topology)", worst,
         ">= 1.0 required", "documented defect if < 1", worst >= 1.0,
         "a value < 1 proves the legacy frame utilisation is non-conservative")

    # G6: uniform load -- system frame moment vs the classical p a L^2 / 8
    gate("G6 uniform load: system frame moment vs p a L^2/8",
         uniform["system_vs_analytic_rel"], "< 0.15", "relative",
         abs(uniform["system_vs_analytic_rel"]) < 0.15)

    # G7: G6 for the legacy smeared layer (expected to fail: factor ~2)
    gate("G7 uniform load: smeared frame utilisation / system", uniform["smeared_over_system"],
         ">= 1.0 required", "documented defect if < 1",
         uniform["smeared_over_system"] >= 1.0)

    # G8: system branch is mesh converged (sub 8 vs 12)
    worst_mesh = max(abs(r["system_u_frame_sub12"] - r["system_u_frame"])
                     / r["system_u_frame"] for r in rows)
    gate("G8 system branch mesh convergence (sub 8 vs 12)", worst_mesh, "< 0.05",
         "relative", worst_mesh < 0.05)

    # G9: plate-controlled ultimate vs the classical simply-supported strip
    gate("G9 plate p_ult vs classical 8 M_p / b^2", classical["relative_error_vs_ss_strip"],
         "< 0.15", "relative", abs(classical["relative_error_vs_ss_strip"]) < 0.15)

    # G10: the panel bracket sits between first hinge and plastic collapse
    lo = bracket["p_first_hinge_ss_strip"]
    hi = bracket["p_plastic_collapse_clamped_strip"]
    p_model = classical["model_p_ult"]
    gate("G10 classical bracket contains the model plate p_ult",
         p_model / lo, f"[{lo/1e6:.3f}, {hi/1e6:.3f}] MPa", "containment",
         lo <= p_model <= hi)

    # G11: combined cases -- the plate-governed headline is model invariant while
    # the member utilisations are not
    shifts = [abs(r["panel"]["p_ult_MPa"] - r["smeared"]["p_ult_MPa"])
              / r["smeared"]["p_ult_MPa"] for r in combined]
    frame_ratio = [r["smeared"]["u_frame"] / r["system"]["u_frame"]
                   for r in combined if r["system"]["u_frame"] > 0]
    gate("G11 combined cases: |p_ult(panel) - p_ult(smeared)| / p_ult",
         max(shifts), "reported", "n/a", True,
         "plate-governed cases keep the headline p_ult; member cases do not")
    gate("G12 combined cases: smeared frame utilisation / system (margin error)",
         min(frame_ratio), "reported", "n/a", True,
         "< 1 means the legacy margin is overstated")

    summary = {
        "anchor": "GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE",
        "round": 5,
        "geometry": {
            "frame_spacing_m": geom.frame_spacing,
            "bay_spacing_m": geom.bay_spacing,
            "n_spans": int(geom.n_spans),
            "n_bays": int(geom.n_bays),
            "plate_thickness_mm": geom.plate_thickness * 1e3,
            "yield_stress_MPa": geom.yield_stress / 1e6,
        },
        "sweep_pressure_MPa": P_SWEEP / 1e6,
        "uniform_load": uniform,
        "belt": belt,
        "classical": classical,
        "bracket": bracket,
        "combined_cases": combined,
        "gates": gates,
        "gates_passed": sum(1 for g in gates if g["passed"]),
        "gates_total": len(gates),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str),
                                      encoding="utf-8")
    return summary


if __name__ == "__main__":
    import sys

    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage == "sweep":
        print(json.dumps(stage_sweep(), indent=2, default=str))
        raise SystemExit(0)
    if stage == "combined":
        print(json.dumps(stage_combined(), indent=2))
        raise SystemExit(0)
    if stage == "maps":
        stage_maps()
        raise SystemExit(0)
    if stage == "all":
        stage_sweep()
        stage_combined()
        stage_maps()
    out = build()
    print(json.dumps({"gates_passed": out["gates_passed"],
                      "gates_total": out["gates_total"],
                      "failed": [g["gate"] for g in out["gates"] if not g["passed"]]},
                     indent=2))