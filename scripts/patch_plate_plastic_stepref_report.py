"""Figure + case-note generator for the plate-panel load-step refinement study.

Anchor ``GAP-PP-3-ICE-STRUCT-PLASTIC-SET``.  Everything in
``docs/CASE-FEM-PLATE-PATCH-PLASTIC-STEP-REFINEMENT.md`` and in
``artifacts/patch_plate_plastic_stepref.png`` is produced here from the queue
artifacts written by ``scripts/patch_plate_plastic_step_refinement.py`` -- no
number is transcribed by hand, and a missing case is printed as ``not_run``
instead of being dropped silently.

Usage::

    PYTHONPATH=src python scripts/patch_plate_plastic_stepref_report.py
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from patch_plate_plastic_step_refinement import (  # noqa: E402
    ART,
    MESH_SEQUENCE,
    PLAN,
    STEP_SEQUENCE,
    analyse_series,
    bracket_check,
    matched_mesh_effect,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "CASE-FEM-PLATE-PATCH-PLASTIC-STEP-REFINEMENT.md"
FIG = ART / "patch_plate_plastic_stepref.png"


def load_records() -> list[dict]:
    files = sorted(ART.glob("patch_plate_plastic_stepref_nx*_dp*.json"))
    return [json.loads(f.read_text()) for f in files]


def fmt(value, scale=1e6, digits=4, unit=" MPa") -> str:
    if value is None:
        return "not_run"
    return f"{value / scale:.{digits}f}{unit}"


def pct(value) -> str:
    return "n/a" if value is None else f"{100.0 * value:+.2f} %"


def main() -> int:
    records = load_records()
    if not records:
        print("no step-refinement artifacts yet")
        return 1
    by_case = {(r["nx"], round(r["step_p_y"], 6)): r for r in records}
    series = []
    for nx in MESH_SEQUENCE:
        rows = sorted([r for r in records if r["nx"] == nx],
                      key=lambda q: -q["step_p_y"])
        if rows:
            series.append(analyse_series(rows))
    mesh_effect = matched_mesh_effect(records)
    ref_record = records[0]
    bounds = (ref_record.get("references") or {}).get("bounds") or {}
    p_y = (ref_record.get("references") or {}).get("p_yield_analytic")

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.9))
    colours = {8: "#1f77b4", 12: "#2ca02c", 16: "#d62728"}

    # (a) limit load vs load step, one curve per mesh
    ax = axes[0]
    for nx in MESH_SEQUENCE:
        rows = sorted([r for r in records if r["nx"] == nx],
                      key=lambda q: -q["step_p_y"])
        xs = [r["step_p_y"] for r in rows if r["limit_tight"] is not None]
        ys = [r["limit_tight"] / 1e6 for r in rows if r["limit_tight"] is not None]
        if not xs:
            continue
        ax.plot(xs, ys, "o-", color=colours.get(nx, "k"), label=f"nx = {nx}")
    if bounds.get("strip_lower") and bounds.get("johansen_twist"):
        ax.axhspan(bounds["strip_lower"] / 1e6, bounds["johansen_twist"] / 1e6,
                   color="0.85", zorder=0,
                   label="rigid-plastic bracket (strip lower / Johansen twist)")
    for key, style, label in (
            ("yield_line", "--", "yield-line upper bound"),
            ("ilyushin_sine", ":", "Ilyushin double-sine")):
        if bounds.get(key):
            ax.axhline(bounds[key] / 1e6, linestyle=style, color="k", lw=1.0,
                       label=label)
    ax.set_xscale("log")
    ax.set_xlabel(r"load step $\Delta p/p_y$ [-]")
    ax.set_ylabel("limit pressure [MPa]")
    ax.set_title("(a) limit load vs load step")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best")

    # (b) load-deflection paths at the coarsest common step
    ax = axes[1]
    common = sorted({round(r["step_p_y"], 6) for r in records})
    pick = common[0] if common else None
    for nx in MESH_SEQUENCE:
        rec = by_case.get((nx, pick))
        if not rec:
            continue
        pp = [q["p_over_py"] for q in rec["path"]]
        ww = [q["w"] * 1e3 for q in rec["path"]]
        ax.plot(ww, pp, "-", color=colours.get(nx, "k"),
                label=f"nx = {nx} ({len(rec['path'])} pts"
                      f"{', truncated' if rec['ramp_truncated'] else ''})")
    ax.set_xlabel("centre deflection [mm]")
    ax.set_ylabel(r"$p/p_y$ [-]")
    ax.set_title(f"(b) load-deflection paths, all meshes at "
                 rf"$\Delta p/p_y$ = {pick:g}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    if p_y:
        ax.axhline(1.0, color="0.5", lw=0.8, ls=":")

    # (c) step-size convergence estimate per mesh
    ax = axes[2]
    labels, values, colours_bar = [], [], []
    for s in series:
        ft = s.get("finest_two") or {}
        change = ft.get("relative_change")
        labels.append(f"nx = {s['nx']}")
        values.append(float("nan") if change is None else 100.0 * change)
        colours_bar.append(colours.get(s["nx"], "k"))
    ax.bar(labels, values, color=colours_bar)
    ax.axhline(2.0, color="r", ls="--", lw=1.0, label="2 % step-independence band")
    ax.axhline(-2.0, color="r", ls="--", lw=1.0)
    ax.set_ylabel("change of limit load between the two\nfinest load steps [%]")
    ax.set_title("(c) step-size convergence")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.suptitle("GAP-PP-3 plate-panel plastic limit: separation of load-step and "
                 "mesh effects", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------ doc
    lines: list[str] = []
    add = lines.append
    add("# CASE-FEM-PLATE-PATCH-PLASTIC-STEP-REFINEMENT")
    add("")
    add("Anchor `GAP-PP-3-ICE-STRUCT-PLASTIC-SET`. Generated by "
        "`scripts/patch_plate_plastic_stepref_report.py` from the queue artifacts "
        "`artifacts/patch_plate_plastic_stepref_nx*_dp*.json`; no number below is "
        "transcribed by hand.")
    add("")
    add("## 1. The defect this case closes")
    add("")
    add("`docs/CASE-FEM-PLATE-PATCH-PLASTIC-PATH-ADMISSIBILITY.md` §5 records the "
        "limitation verbatim: the FE limit load moves by 39 % between `nx = 8` and "
        "`nx = 12`, **and the two ramps were run with different load-step density** "
        "(5 vs 21 load points), so the comparison mixes a mesh effect with a "
        "*stepping* effect of the criterion-based limit extractor. Nothing in the "
        "archived ladder can separate the two.")
    add("")
    add("The fix is a two-way study on a full 3 x 3 grid "
        f"(`nx` in {list(MESH_SEQUENCE)}, `delta p/p_y` in {list(STEP_SEQUENCE)}): "
        "each mesh gets its own step-refinement series, and every mesh has "
        "*matched-step* partners, so the mesh effect is read at constant ramp "
        "density.")
    add("")
    add("## 2. Case and parameters")
    add("")
    add("| quantity | value |")
    add("|---|---|")
    add("| plate | simply supported square, a = b = 1.000 m, t = 20 mm |")
    add("| patch | 500 x 500 mm centred (25 % of the area) |")
    add("| material | DH36 @ -40 degC, near-ideal hardening H = 1 MPa, 4 through-thickness layers |")
    add(f"| analytic first-yield pressure p_y | {fmt(p_y)} |")
    for key, label in (("strip_lower", "strip lower bound"),
                       ("johansen_twist", "Johansen twist upper bound"),
                       ("yield_line", "yield-line mechanism bound"),
                       ("ilyushin_sine", "Ilyushin double-sine bound")):
        if bounds.get(key):
            add(f"| {label} | {fmt(bounds[key])} |")
    add("| ramp top | 3.0 p_y, then 6 unloading points to measure the permanent set |")
    add("")
    add("## 3. Measurements")
    add("")
    add("| mesh | `dp/p_y` | committed points | limit (tight) | criteria spread | "
        "ramp end | peak pressure | first yield err. vs p_y | permanent set | bracket |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for nx in MESH_SEQUENCE:
        for dp in STEP_SEQUENCE:
            rec = by_case.get((nx, round(dp, 6)))
            if rec is None:
                add(f"| {nx} | {dp:g} | not_run | not_run | - | - | - | - | - | - |")
                continue
            bracket = bracket_check(rec["limit_tight"], rec)
            spread = rec["criteria_spread_tight"]
            fy = rec["first_yield"]
            add(f"| {nx} | {dp:g} | {rec['n_committed_points']}"
                f"/{rec['n_load_steps_requested']} | {fmt(rec['limit_tight'])} | "
                f"{'n/a' if spread is None else f'{spread:.3f}'} | "
                f"{'truncated' if rec['ramp_truncated'] else 'completed'} | "
                f"{fmt(rec['peak_factor'])} | "
                f"{'n/a' if (fy is None or not p_y) else f'{100.0 * (fy / p_y - 1.0):+.2f} %'} | "
                f"{fmt(rec['permanent_deflection'], 1e3, 2, ' mm') if rec['permanent_set_valid'] else 'withheld'} | "
                f"{bracket['verdict']} |")
    add("")
    add("## 4. Load-step convergence (fixed mesh)")
    add("")
    add("| mesh | usable steps | limit at each step | successive changes | verdict | "
        "order in `dp` | `dp -> 0` estimate |")
    add("|---|---|---|---|---|---|---|")
    for s in series:
        limits = ", ".join(fmt(v) if v is not None else "withheld" for v in s["limits"])
        changes = ", ".join(pct(c) if c is not None else "n/a"
                            for c in s.get("successive_changes", []))
        rich = s.get("step_richardson") or {}
        add(f"| {s['nx']} | {s['n_usable']}/{len(s['steps'])} | {limits} | {changes} | "
            f"**{s['verdict']}** | "
            f"{'n/a' if not rich else format(rich['order'], '.2f')} | "
            f"{'n/a' if not rich else fmt(rich.get('limit_dp_to_zero'))} |")
    add("")
    add("## 5. Mesh effect at matched ramp density")
    add("")
    add("| `dp/p_y` | meshes present | limits | pairwise relative change |")
    add("|---|---|---|---|")
    for entry in mesh_effect:
        limits = ", ".join(fmt(v) if v is not None else "withheld"
                           for v in entry["limits"])
        pairs = ", ".join(f"{p['pair'][0]}->{p['pair'][1]}: {pct(p['relative_change'])}"
                          for p in entry["pairs"])
        add(f"| {entry['step_p_y']:g} | {entry['meshes']} | {limits} | {pairs} |")
    add("")
    add(f"![step refinement](artifacts/{FIG.name})")
    add("")
    add("## 6. Acceptance verdicts")
    add("")
    add("| acceptance item | evidence | verdict |")
    add("|---|---|---|")
    step_verdicts = {s["nx"]: s["verdict"] for s in series}
    add("| limit load is independent of the load step (<= 2 % between the two finest steps) | "
        + "; ".join(f"nx = {nx}: {v}" for nx, v in sorted(step_verdicts.items()))
        + " | " + ("**met**" if step_verdicts and
                   all(v == "step_independent" for v in step_verdicts.values())
                   else "**not met**") + " |")
    three = [s for s in series if s["n_usable"] >= 2]
    add("| three-mesh convergence with matched ramp density | "
        + (", ".join(f"nx = {s['nx']} ({s['n_usable']} steps)" for s in series)
           if series else "not_run")
        + " | " + ("**partially met**" if len(three) >= 2 else "**not run**") + " |")
    add("")
    add("## 7. Honest reading and limits of validity")
    add("")
    add("* The extracted quantity is a **criterion-based limit load** "
        "(`intersection_of_tangents` / `double_deflection` / "
        "`stiffness_criterion`). The load step is the numerics knob that can "
        "move it; this study measures that sensitivity instead of assuming it away.")
    add("* A ramp that the admissibility gate truncated contributes only its "
        "committed (equilibrium) points; its peak pressure is a lower estimate of "
        "the limit load, never an extrapolation.")
    add("* Permanent set is reported only when the unloading leg converged; "
        "otherwise it is withheld.")
    add("* The plate model is a small-strain flat-facet Mindlin layered element "
        "with in-plane displacements free, so no membrane tension develops and "
        "the comparison against rigid-plastic *bending* bounds is admissible.")
    add("")
    add("## 8. Reproduction")
    add("")
    add("```bash")
    add("run_compute submit --repo TensorFEM --stage_dir <worktree> --shards 9 \\")
    add("  --cmd_tpl 'PYTHONPATH=src python scripts/patch_plate_plastic_step_refinement.py --plan-index {i}'")
    add("PYTHONPATH=src python scripts/patch_plate_plastic_step_refinement.py --analyse")
    add("PYTHONPATH=src python scripts/patch_plate_plastic_stepref_report.py")
    add("PYTHONPATH=src python -m pytest tests/test_plate_patch_plastic_step_refinement.py -q")
    add("```")
    DOC.write_text("\n".join(lines) + "\n")
    print(f"wrote {DOC.relative_to(ROOT)} and {FIG.relative_to(ROOT)}")
    print(f"cases present: {len(records)}/{len(PLAN)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())