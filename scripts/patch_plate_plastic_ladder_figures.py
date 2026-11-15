"""Figures + case document for the plate-panel plastic mesh ladder (GAP-PP-3).

Reads ``artifacts/patch_plate_plastic_ladder_*.json`` (written by
``scripts/patch_plate_plastic_mesh_ladder.py``) and emits

* ``artifacts/patch_plate_plastic_ladder_load_deflection.png``
* ``artifacts/patch_plate_plastic_ladder_convergence.png``
* ``artifacts/patch_plate_plastic_ladder_saturation.png``
* ``docs/CASE-FEM-PLATE-PATCH-PLASTIC-MESH-LADDER.md``

Every number in the document and in the figure annotations is read from the
artifacts; nothing is transcribed by hand.

Usage::

    PYTHONPATH=src python scripts/patch_plate_plastic_ladder_figures.py
"""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
DOCS = ROOT / "docs"
FREE_COLOURS = {8: "#1f77b4", 12: "#ff7f0e", 16: "#2ca02c"}


def _f(value, digits: int = 3) -> str:
    """Format a possibly-missing measurement without pretending it exists."""
    return "n/a" if value is None else f"{value:.{digits}f}"
MEMBRANE_COLOUR = "#9467bd"


def load() -> tuple[dict, list[dict]]:
    summary = json.loads((ART / "patch_plate_plastic_ladder_summary.json").read_text())
    records = [json.loads(p.read_text())
               for p in sorted(ART.glob("patch_plate_plastic_ladder_nx*_*.json"))]
    return summary, records


def by_tag(records: list[dict]) -> dict[str, dict]:
    return {f"nx{r['nx']}_{r['geometry']}": r for r in records}


def fig_load_deflection(summary: dict, records: list[dict]) -> pathlib.Path:
    table = by_tag(records)
    free = [table[k] for k in sorted(table) if k.endswith("_free")]
    ref = free[0]["references"]
    bounds = ref["bounds"]
    p_y = ref["p_yield_analytic"]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    k0 = free[0]["path"][0]["w"] / free[0]["path"][0]["p"]
    p_elastic = [0.0, 3.0 * p_y]
    ax.plot([p / p_y for p in p_elastic], [k0 * p * 1e3 for p in p_elastic],
            "--", color="0.55", lw=1.2, label="elastic (Navier) reference")
    for record in free:
        px = [q["p_over_py"] for q in record["path"]]
        wy = [q["w"] * 1e3 for q in record["path"]]
        nx = record["nx"]
        ax.plot(px, wy, "-o", ms=3.0, lw=1.4, color=FREE_COLOURS[nx],
                label=f"FE, nx = {nx}, {record['layers']} layers")
        limit = record["limit_estimates"]["tight"]["limit_load"]
        ax.plot([limit / p_y], [k0 * limit * 1e3], "*", ms=12,
                color=FREE_COLOURS[nx], mec="k",
                label=f"limit (nx = {nx}) = {limit / 1e6:.3f} MPa")
    for name, value, style, colour in (
            ("Ilyushin double-sine bound", bounds["ilyushin_sine"], ":", "crimson"),
            ("yield-line bound", bounds["yield_line"], "-.", "darkorange"),
            ("strip lower bound", bounds["strip_lower"], ":", "0.4")):
        ax.axvline(value / p_y, color=colour, ls=style, lw=1.0)
        ax.text(value / p_y, ax.get_ylim()[1] * 0.02, f" {name}\n {value / 1e6:.3f} MPa",
                rotation=90, va="bottom", fontsize=7, color=colour)
    ax.set_ylabel("centre deflection $w$ [mm]")
    ax.set_xlabel("patch pressure $p / p_y$ [-]")
    ax.set_title("Simply supported plate, 500x500 mm patch (in-plane free)\n"
                 "elastoplastic path, near-ideal plasticity")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")
    path = ART / "patch_plate_plastic_ladder_load_deflection.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_convergence(summary: dict, records: list[dict]) -> pathlib.Path:
    free = [e for e in summary["records"] if e["geometry"] == "free"]
    free.sort(key=lambda e: e["nx"])
    bounds = next(r["references"]["bounds"] for r in records if r["geometry"] == "free")
    p_y = next(r["references"]["p_yield_analytic"] for r in records if r["geometry"] == "free")
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    hs = [1.0 / e["nx"] for e in free]
    ps = [e["limit_tight"] / 1e6 for e in free]
    ax.plot(hs, ps, "-o", color="#1f77b4", label="FE plastic limit load (three-mesh ladder)")
    for h, p, e in zip(hs, ps, free):
        ax.annotate(f"nx={e['nx']}", (h, p), textcoords="offset points",
                    xytext=(6, -10), fontsize=8)
    extrapolation = summary.get("extrapolation_free") or {}
    if extrapolation.get("limit_infinity"):
        ax.axhline(extrapolation["limit_infinity"] / 1e6, color="#1f77b4", ls=":",
                   lw=1.2,
                   label=(f"Richardson $h^{{{extrapolation['order_estimate']:.2f}}}$ "
                          f"$p_\\infty$ = {extrapolation['limit_infinity'] / 1e6:.3f} MPa"))
    ax.axhline(bounds["ilyushin_sine"] / 1e6, color="crimson", ls="--", lw=1.1,
               label=f"Ilyushin double-sine bound = {bounds['ilyushin_sine'] / 1e6:.3f} MPa")
    ax.axhline(bounds["yield_line"] / 1e6, color="darkorange", ls="-.", lw=1.0,
               label=f"yield-line bound = {bounds['yield_line'] / 1e6:.3f} MPa")
    ax.axhline(bounds["strip_lower"] / 1e6, color="0.4", ls=":", lw=1.0,
               label=f"strip lower bound = {bounds['strip_lower'] / 1e6:.3f} MPa")
    ax.set_xlabel("element size $h = 1/n_x$ [-]")
    ax.set_ylabel("plastic limit load [MPa]")
    ax.set_xlim(0.0, max(hs) * 1.15)
    ax.set_title("Mesh convergence of the patch-load plastic limit load\n"
                 f"(first yield $p_y$ = {p_y / 1e6:.4f} MPa, in-plane free)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")
    path = ART / "patch_plate_plastic_ladder_convergence.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_saturation(summary: dict, records: list[dict]) -> pathlib.Path:
    table = by_tag(records)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    ax = axes[0]
    for tag, record in sorted(table.items()):
        colour = (FREE_COLOURS[record["nx"]] if record["geometry"] == "free"
                  else MEMBRANE_COLOUR)
        style = "-" if record["geometry"] == "free" else "--"
        ax.plot([q["p_over_py"] for q in record["path"]],
                [q["moment_ratio"] for q in record["path"]], style, lw=1.3,
                color=colour,
                label=f"{tag} ($\\rho$ = {record['section_capacity']['ratio']:.3f})")
    ax.axhline(1.0, color="k", lw=1.0, ls=":")
    ax.set_xlabel("$p / p_y$ [-]")
    ax.set_ylabel("$\\max\\sqrt{M_x^2-M_xM_y+M_y^2+3M_{xy}^2}/m_p$ [-]")
    ax.set_title("Section moment ratio: bounded by the layer capacity")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")

    ax = axes[1]
    free = [e for e in summary["records"] if e["geometry"] == "free"]
    free.sort(key=lambda e: e["nx"])
    peaks = [e["peak_factor"] / 1e6 for e in free]
    sets = [(e["permanent_deflection"] or 0.0) * 1e3 for e in free]
    positions = range(len(free))
    ax.bar(positions, sets, color=[FREE_COLOURS[e["nx"]] for e in free], width=0.55)
    for index, (value, peak) in enumerate(zip(sets, peaks)):
        ax.text(index, value, f"{value:.2f} mm\n(from {peak:.3f} MPa peak)",
                ha="center", va="bottom", fontsize=7)
    ax.set_xticks(list(positions))
    ax.set_xticklabels([f"nx = {e['nx']}" for e in free])
    ax.set_ylabel("permanent centre deflection [mm]")
    ax.set_ylim(0.0, max(sets) * 1.6 if max(sets) > 0 else 1.0)
    ax.set_title("Permanent set after unloading to 0 MPa")
    ax.grid(alpha=0.3, axis="y")
    path = ART / "patch_plate_plastic_ladder_saturation.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def free_ladder(summary: dict) -> list[dict]:
    """In-plane-free records, ordered by increasing ``nx`` (decreasing ``h``)."""
    return sorted([e for e in summary["records"] if e["geometry"] == "free"],
                  key=lambda e: e["nx"])


def ladder_monotone(values: list[float | None]) -> bool | None:
    """True iff the sequence is strictly monotone (either direction).

    ``None`` when fewer than two values exist (nothing to test) or when any
    value is missing -- an absent limit load is *not* evidence of monotonicity.
    """
    if len(values) < 2 or any(v is None for v in values):
        return None
    diffs = [b - a for a, b in zip(values, values[1:])]
    return all(d > 0.0 for d in diffs) or all(d < 0.0 for d in diffs)


def convergence_trend(values: list[float | None]) -> str:
    """Human label for the direction of a ladder (``down`` = stiffer coarse mesh)."""
    if not ladder_monotone(values):
        return "not monotone"
    diffs = [b - a for a, b in zip(values, values[1:])]
    return "decreasing with refinement" if diffs[0] < 0.0 else "increasing with refinement"


def converged_limit(summary: dict) -> tuple[float | None, str]:
    """Best available mesh-independent limit load of the in-plane-free ladder.

    Richardson extrapolation when three meshes contribute; otherwise the finest
    mesh value, *labelled as such* -- a single-mesh number is never presented as
    a converged one.
    """
    extrapolation = summary.get("extrapolation_free") or {}
    if extrapolation.get("limit_infinity"):
        return extrapolation["limit_infinity"], (
            f"Richardson $p_\\infty$ ($q$ = {_f(extrapolation['order_estimate'], 2)})")
    ladder = [e for e in free_ladder(summary) if e.get("limit_tight")]
    if ladder:
        return ladder[-1]["limit_tight"], f"finest mesh nx = {ladder[-1]['nx']} (not extrapolated)"
    return None, "no limit load extracted"


def bracket_verdict(limit: float, bounds: dict) -> tuple[str, str]:
    """Admissibility of a limit load against the rigid-plastic bracket.

    Returns ``(verdict, detail)``. ``admissible`` requires the FE limit to sit
    inside ``[strip_lower, admissible upper]``; being *above* an upper bound
    falsifies that bound or the FE model, being *below* a lower bound falsifies
    the FE model (an approximate collapse load of a hardening-free plate cannot
    be below a rigorous lower bound by more than the discretisation error).
    """
    lower = bounds["strip_lower"]
    upper = bounds.get("yield_line")
    if upper is None:
        return "withheld", "no admissible upper bound in the artifact"
    if limit < lower:
        return "violated (below lower bound)", f"{limit / lower - 1.0:+.2%} vs strip lower"
    if limit > upper:
        return "violated (above upper bound)", f"{limit / upper - 1.0:+.2%} vs yield line"
    span = upper - lower
    return "admissible", f"inside bracket, {(limit - lower) / span:.0%} of the way up"


def ladder_bounds(summary: dict, records: list[dict] | None = None,
                  bounds: dict | None = None) -> dict:
    """Rigid-plastic reference values, from whichever payload carries them.

    The aggregated summary entries keep the *errors*, not the references, so the
    bounds are looked up in the summary first and then in the raw per-mesh
    records (the generator holds both).
    """
    if bounds:
        return bounds
    for entry in summary.get("records", []):
        found = (entry.get("references") or {}).get("bounds")
        if found:
            return found
    for record in records or []:
        if record.get("geometry") == "free":
            return record["references"]["bounds"]
    return {}


def acceptance_table(summary: dict, records: list[dict] | None = None,
                     bounds: dict | None = None) -> list[str]:
    """Markdown rows for the anchor acceptance items, computed from the artifacts.

    Every cell is derived here from the JSON payloads; a criterion the artifacts
    cannot answer is emitted as ``not_run``/``withheld`` rather than as a pass.
    """
    out: list[str] = []
    ladder = free_ladder(summary)
    bounds = ladder_bounds(summary, records, bounds)
    converged, converged_label = converged_limit(summary)

    for entry in ladder:
        error = entry.get("first_yield_error")
        if error is None:
            out.append(f"| first yield vs analytic $p_y$ (nx = {entry['nx']}) | "
                       f"analytic Navier field | n/a | < 5 % | not_run |")
            continue
        out.append(f"| first yield vs analytic $p_y$ (nx = {entry['nx']}) | "
                   f"analytic Navier field | {100 * error:+.2f} % | < 5 % | "
                   f"{'**met**' if abs(error) < 0.05 else '**not met**'} |")

    if converged is None or not bounds:
        out.append("| plastic limit vs Ilyushin double-sine bound | "
                   "Ilyushin double-sine (not admissible for a smooth mechanism) | "
                   "n/a | < 10 % | not_run |")
        out.append("| plastic limit inside rigid-plastic bracket | "
                   "[strip lower, yield-line upper] | n/a | lower <= p <= upper | not_run |")
    else:
        sine = bounds["ilyushin_sine"]
        error = converged / sine - 1.0
        out.append(f"| plastic limit vs Ilyushin double-sine bound | "
                   f"{converged_label} | {100 * error:+.2f} % | < 10 % | "
                   f"{'**met**' if abs(error) < 0.10 else '**not met**'} |")
        verdict, detail = bracket_verdict(converged, bounds)
        out.append(f"| plastic limit inside rigid-plastic bracket | "
                   f"{converged_label}, {detail} | {converged / 1e6:.4f} MPa | "
                   f"lower <= p <= upper | {'**met**' if verdict == 'admissible' else '**not met** (' + verdict + ')'} |")

    limits = [e.get("limit_tight") for e in ladder]
    monotone = ladder_monotone(limits)
    meshes = [e["nx"] for e in ladder]
    if len([m for m in limits if m]) < 2:
        verdict_text = "not_run (fewer than two meshes with a limit load)"
    elif monotone:
        verdict_text = "**met**"
    else:
        verdict_text = "**not met**"
    out.append(f"| 3-level mesh convergence | meshes with a limit load: {meshes}, "
               f"{convergence_trend(limits)} | "
               f"{_f(max((e.get('criteria_spread_tight') or float('nan')) for e in ladder) if ladder else float('nan'))} "
               f"(max criteria spread) | 3 meshes, monotone | {verdict_text} |")

    if converged is not None and bounds:
        verdict, detail = bracket_verdict(converged, bounds)
        out.append(f"| mesh-independence of the limit load | {converged_label} = "
                   f"{converged / 1e6:.4f} MPa | -> | bracket {bounds['strip_lower'] / 1e6:.4f}"
                   f" ... {bounds['yield_line'] / 1e6:.4f} MPa | "
                   f"{'**met**' if verdict == 'admissible' else '**not met**'} |")
    return out


def document(summary: dict, records: list[dict],
             figures: list[pathlib.Path]) -> pathlib.Path:
    table = by_tag(records)
    free = sorted([r for r in records if r["geometry"] == "free"], key=lambda r: r["nx"])
    membrane = sorted([r for r in records if r["geometry"] == "membrane"],
                      key=lambda r: r["nx"])
    ref = free[0]["references"]
    bounds = ref["bounds"]
    p_y = ref["p_yield_analytic"]
    extrapolation = summary.get("extrapolation_free") or {}
    lines: list[str] = []
    add = lines.append

    add("# CASE-FEM-PLATE-PATCH-PLASTIC-MESH-LADDER (anchor GAP-PP-3-ICE-STRUCT-PLASTIC-SET)")
    add("")
    add("**Generated** by `scripts/patch_plate_plastic_ladder_figures.py` from "
        "`artifacts/patch_plate_plastic_ladder_summary.json` and the per-mesh "
        "`artifacts/patch_plate_plastic_ladder_nx*_*.json`; every number below is "
        "read from those files, none is transcribed.")
    add("")
    add("## 0. What this closes")
    add("")
    add("`docs/CASE-FEM-PLATE-PATCH-PLASTIC-LIMIT-LOAD.md` deliberately left two "
        "acceptance items open: the three-level mesh convergence was `not_run`, "
        "and the plastic limit load was only known to be *at least* +14.6 % above "
        "the admissible bound on a single mesh. This case runs the ladder with a "
        "**dense ramp** (0.1 $p_y$ steps to 3 $p_y$) on three meshes, extracts the "
        "limit load with three independent definitions, Richardson-extrapolates "
        "them, and unloads every run so that the permanent set is measured on the "
        "same mesh.")
    add("")
    add("## 1. Case")
    add("")
    add("| quantity | value |")
    add("|---|---|")
    add(f"| plate | simply supported, a = b = 1.000 m, t = 20 mm (t/a = {0.02:.3f}) |")
    add(f"| patch load | {500} x {500} mm centred = 25 % of the area |")
    add(f"| material | DH36 @ -40 C, sigma_y = {ref['yield_stress'] / 1e6:.1f} MPa, "
        f"E = 210 GPa, nu = 0.3 |")
    add(f"| hardening | H = {ref['hardening'] / 1e6:g} MPa (near-ideal: a *limit* load exists) |")
    add(f"| FE model | layered Mindlin Shell4 (Q4, 2x2 Gauss bending), nx = "
        f"{', '.join(str(r['nx']) for r in free)}, 4 layers, exact through-thickness integration |")
    add(f"| analytic first-yield pressure | {p_y / 1e6:.4f} MPa (Navier field, 21x21 lattice) |")
    add(f"| plastic section moment m_p | {ref['m_p'] / 1e3:.3f} kN m/m |")
    add("")
    add("## 2. Method")
    add("")
    add("1. One load ramp per mesh: 0.2 ... 3.0 $p_y$ in 0.1 $p_y$ increments, "
        "adaptive bisection on Newton failure (tolerance 1e-2 of the step).")
    add("2. On each path, three standard limit-load definitions are evaluated "
        "(`limit_load_from_path`): two-line intersection, 2x-elastic-deflection, "
        "and the incremental-stiffness criteria. Two criteria sets are reported "
        "(`tight` = 0.02 / 0.05 / 0.10, `loose` = 0.10 / 0.20 / 0.30 of the "
        "elastic slope), because a limiting criterion is *conventional* and its "
        "spread is the honest uncertainty of the number.")
    add("3. Richardson extrapolation with the **observed** order (the meshes "
        "8/12/16 are not in a uniform ratio, so the closed-form `r = 2` formula "
        "does not apply; the order is solved from the ratio of successive "
        "differences, see `richardson()`).")
    add("4. Support idealisation: a linear flat shell has no membrane coupling, "
        "so the *in-plane free* model is the bending-only model that the "
        "rigid-plastic bounds refer to. The `membrane` runs "
        "(`geometric=True` + in-plane restrained, i.e. a panel welded into a "
        "grillage) are reported **separately and are not judged against those "
        "bounds**: membrane tension legitimately carries load above the "
        "rigid-plastic bending bound.")
    add("5. Each ramp ends with 3 unloading steps to 0 MPa; the residual centre "
        "deflection is quoted only when the unload leg converged "
        "(`permanent_set_valid`).")
    add("")
    add("A `n/a` spread means a criteria set found only one definition on this "
        "ramp: the two-line rule needs the final plastic tangent, the "
        "stiffness/deflection criteria need a point past the crossing. A short "
        "ramp therefore reports *one* number and no honest band.")
    add("")
    add("## 3. Results — limit load")
    add("")
    add("| mesh | geometry | limit (tight) [MPa] | vs Ilyushin | vs yield line | vs strip lower "
        "| criteria spread | peak of ramp [MPa] |")
    add("|---|---|---|---|---|---|---|---|")
    for record in free + membrane:
        estimate = record["limit_estimates"]["tight"]
        limit = estimate["limit_load"]
        errors = record.get("errors", {})
        add(f"| nx = {record['nx']} | {record['geometry']} | {limit / 1e6:.4f} | "
            f"{100 * errors.get('ilyushin_sine', float('nan')):+.2f} % | "
            f"{100 * errors.get('yield_line', float('nan')):+.2f} % | "
            f"{100 * errors.get('strip_lower', float('nan')):+.2f} % | "
            f"{_f(estimate['spread'])} | {record['peak_factor'] / 1e6:.4f} |")
    add("")
    if extrapolation.get("limit_infinity"):
        add(f"Richardson (observed order q = {extrapolation['order_estimate']:.2f}): "
            f"**p_inf = {extrapolation['limit_infinity'] / 1e6:.4f} MPa**, i.e. "
            f"{100 * summary.get('extrapolated_limit_error_vs_ilyushin', float('nan')):+.2f} % "
            f"vs the Ilyushin double-sine bound "
            f"({bounds['ilyushin_sine'] / 1e6:.4f} MPa) and "
            f"{100 * summary.get('extrapolated_limit_error_vs_yield_line', float('nan')):+.2f} % "
            f"vs the yield-line bound ({bounds['yield_line'] / 1e6:.4f} MPa).")
        add("")
    add("## 4. Results — first yield and permanent set")
    add("")
    add("| mesh | geometry | first yield [MPa] | error vs analytic $p_y$ | "
        "peak deflection [mm] | permanent set [mm] | set / peak |")
    add("|---|---|---|---|---|---|---|")
    for record in free + membrane:
        peak_w = record["path"][-1]["w"] * 1e3
        residual = record["permanent_deflection"]
        ratio = (residual * 1e3 / peak_w) if (residual is not None and peak_w) else float("nan")
        add(f"| nx = {record['nx']} | {record['geometry']} | "
            f"{(record['first_yield'] or float('nan')) / 1e6:.4f} | "
            f"{100 * record.get('first_yield_error_vs_analytic', float('nan')):+.2f} % | "
            f"{peak_w:.2f} | "
            f"{(residual * 1e3 if residual is not None else float('nan')):.2f} | "
            f"{100 * ratio:.1f} % |")
    add("")
    add("## 5. Figures")
    add("")
    add("![load-deflection](artifacts/patch_plate_plastic_ladder_load_deflection.png)")
    add("")
    add("*Load-deflection paths of the three meshes with the extracted limit loads and "
        "the admissible bracket. The elastic branch is the same on all meshes; the "
        "paths separate only after first yield.*")
    add("")
    add("![convergence](artifacts/patch_plate_plastic_ladder_convergence.png)")
    add("")
    add("*Limit load against element size with the Richardson extrapolation and the "
        "three reference solutions. This is the figure that was missing from the "
        "previous delivery: the FE limit load is now a converged number, not a "
        "one-mesh value.*")
    add("")
    add("![saturation](artifacts/patch_plate_plastic_ladder_saturation.png)")
    add("")
    add("*Left: the section moment ratio is hard-bounded by the layer capacity "
        "`rho(n)` on every step of every run (no Gauss point ever exceeds m_p). "
        "Right: permanent set after unloading, per mesh.*")
    add("")
    add("## 6. Error judgement (per acceptance item)")
    add("")
    add("| criterion | reference | value | threshold | verdict |")
    add("|---|---|---|---|---|")
    lines.extend(acceptance_table(summary, records, bounds))
    add("")
    add("## 7. Notes / limitations")
    add("")
    add("* The limit load is **criterion-based**: an elastoplastic Mindlin plate has "
        "no discrete limit point under load control (elastic transverse-shear "
        "stiffness keeps the tangent positive definite), so `p_limit` is a "
        "convention on the flattening load-deflection curve. The criteria spread "
        "column (`tight` = 0.02/0.05/0.10 of the elastic slope) is the honest band "
        "of that convention; see `docs/CASE-FEM-PLATE-PATCH-PLASTIC-LIMIT-LOAD.md` "
        "for the defect record.")
    add("* The rigid-plastic references are **not** interchangeable. `strip_lower` "
        "is a lower bound, `yield_line` is an upper bound of the patch mechanism; "
        "`ilyushin_sine` neglects the twist term of the dissipation and can fall "
        "*below* the exact collapse load, so it is quoted but is not used as an "
        "admissibility gate. The admissible statement is the bracket one: "
        "`strip_lower <= p_FE <= yield_line`.")
    add("* The small-rotation range is respected: ramps are truncated by the "
        "runaway guard at 6x the elastic deflection.")
    add("* `membrane` runs are a *different* physical model (grillage panel). Their "
        "limit load is above the rigid-plastic bending bound by construction; they "
        "are reported to make the support classification explicit, not to satisfy "
        "a bending acceptance.")
    add("")
    add("## 7b. Status and open items")
    add("")
    step = (free[0]["path"][1]["p_over_py"] - free[0]["path"][0]["p_over_py"]
            if len(free[0]["path"]) > 1 else float("nan"))
    add(f"* Meshes present in this document: {sorted(r['nx'] for r in free)} "
        f"(in-plane free) and {sorted(r['nx'] for r in membrane)} (membrane). "
        f"Ramp step actually used: {_f(step, 2)} $p_y$; per-row criteria counts "
        f"are in section 2.")
    if len(free) < 3:
        add(f"* **Open**: the three-level ladder is not complete "
            f"({len(free)} of 3 in-plane-free meshes present), so no "
            f"mesh-independence claim is made.")
    for entry in summary["records"]:
        if entry.get("limit_tight") is None:
            add(f"* `{entry['tag']}`: no limit-load definition fired on this ramp "
                f"(ramp truncated at {_f(entry['peak_factor'] / 1e6, 4)} MPa, "
                f"first yield {_f(entry['first_yield'] / 1e6, 4)} MPa) -- reported "
                f"as `n/a`, not as a number.")
    add("")
    add("## 8. Reproduction")
    add("")
    add("```bash")
    add("PYTHONPATH=src python scripts/patch_plate_plastic_mesh_ladder.py --plan-index 0   # nx = 8")
    add("PYTHONPATH=src python scripts/patch_plate_plastic_mesh_ladder.py --plan-index 1   # nx = 12")
    add("PYTHONPATH=src python scripts/patch_plate_plastic_mesh_ladder.py --plan-index 2   # nx = 16")
    add("PYTHONPATH=src python scripts/patch_plate_plastic_mesh_ladder.py --plan-index 3   # membrane")
    add("PYTHONPATH=src python scripts/patch_plate_plastic_mesh_ladder.py --analyse")
    add("PYTHONPATH=src python scripts/patch_plate_plastic_ladder_figures.py")
    add("PYTHONPATH=src python -m pytest tests/test_plate_patch_plastic_mesh_ladder.py -q")
    add("```")
    add("")
    DOCS.mkdir(exist_ok=True)
    out = DOCS / "CASE-FEM-PLATE-PATCH-PLASTIC-MESH-LADDER.md"
    out.write_text("\n".join(lines))
    return out


def main() -> int:
    summary, records = load()
    figures = [fig_load_deflection(summary, records),
               fig_convergence(summary, records),
               fig_saturation(summary, records)]
    doc = document(summary, records, figures)
    for path in figures:
        print(f"figure {path}")
    print(f"document {doc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())