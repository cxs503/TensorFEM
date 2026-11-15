"""Render the W30 ``ice-coupled-6dof`` report from the artefacts (no hand-copied
numbers) and emit the tensor_reports payload.

Inputs (artefacts only):

    papers/artifacts/w30/ice-coupled-6dof/ice_load_vs_iso19906.json
    benchmarks/anchors/iacs-ur-i-ice-load.json
    papers/artifacts/w30/ice-coupled-6dof/figures/*.png

Outputs:

    docs/ice-coupled-6dof-w30-load-vs-iacs-iso19906.md
    reports/w30_report.json      (curl -d @... to /tensor/reports)

    python benchmarks/ice_coupled_6dof/make_w30_report.py
"""

from __future__ import annotations

import base64
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ART = ROOT / "papers" / "artifacts" / "w30" / "ice-coupled-6dof"
ANCHOR = ROOT / "benchmarks" / "anchors" / "iacs-ur-i-ice-load.json"
OUT_MD = ROOT / "docs" / "ice-coupled-6dof-w30-load-vs-iacs-iso19906.md"
OUT_JSON = ROOT / "reports" / "w30_report.json"

FIG_CAPTIONS = {
    "fig1_peak_force_distribution.png":
        "Per-impact peak ice load distributions per parameter group with Weibull/Gumbel fits "
        "and the IACS PC6 design force.",
    "fig2_pressure_area.png":
        "Pressure--area plane: ensemble peaks, the declared ISO 19906 law (closure arm) and the "
        "IACS PC6 design point with the code-implied trend.",
    "fig3_convergence.png":
        "Seed-axis convergence of the running mean peak force against the +/-5% band.",
    "fig4_timeseries.png":
        "Time-domain hull-ice contact force for one representative seed per group.",
    "fig5_deviations.png":
        "Per-item deviation from the IACS design values with the +/-15% gate.",
}


def fnum(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def main() -> int:
    ARTIFACT = ART / "ice_load_vs_iso19906.json"
    data = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    anchor = json.loads(ANCHOR.read_text(encoding="utf-8"))
    code = data["code_side"]["per_class"]["PC6"]
    iso = anchor["iso_19906"]["declared_values_used_downstream"]

    L = []
    A = L.append
    A("# ice-coupled-6dof (W30): random-ice 6-DOF load ensemble vs IACS UR I2 / ISO 19906")
    A("")
    A(f"- task: `{data['task_key']}` (mission {data['mission_id']}), repository TensorFEM, "
      f"commit `{data['git']['commit']}`")
    A(f"- artefact: `papers/artifacts/w30/ice-coupled-6dof/ice_load_vs_iso19906.json` "
      f"(generated {data['generated_utc']})")
    A(f"- anchor artefact: `benchmarks/anchors/iacs-ur-i-ice-load.json` "
      f"(sha256 `{data['anchors']['sha256']}`), {data['anchors']['anchor_count']} clause anchors")
    A("")

    # ---- 1 case ----
    hull = data["experiment"]["hull"]
    A("## 1. Case and parameters")
    A("")
    A(f"Rigid 6-DOF hull (surge, sway, heave, roll, pitch, yaw) with constant added mass, "
      f"hydrostatic restoring in heave/roll/pitch and Cummins radiation memory kernels: "
      f"L = {hull['length_m']} m, B = {hull['breadth_m']} m, T = {hull['draught_m']} m, "
      f"Cb = {hull['block_coefficient']}, displacement = {hull['displacement_kt']:.2f} kt. "
      f"These are the declared `ShipModel` particulars of the W23 chain, not a validated ship.")
    A("")
    A("| group | ice concentration | mean thickness [m] | median floe diameter [m] | seeds |")
    A("|---|---|---|---|---|")
    for g in data["groups"]:
        f = g["ice_field"]
        A(f"| {g['group']['name']} | {f['concentration']} | {f['mean_thickness_m']} | "
          f"{f['median_floe_diameter_m']} | {len(g['seeds'])} |")
    A("")
    A(f"dt = {data['experiment']['dt_s']} s, duration = {data['experiment']['duration_s']} s, "
      f"commanded speed = {data['experiment']['commanded_speed_mps']} m/s, seeds = "
      f"{data['experiment']['seeds'][0]}..{data['experiment']['seeds'][-1]} "
      f"({len(data['experiment']['seeds'])} per group = {len(data['groups']) * len(data['experiment']['seeds'])} runs).")
    A("")

    # ---- 2 method ----
    A("## 2. Method and tool chain")
    A("")
    A("1. **Anchor transcription** (`benchmarks/anchors/iacs_ur_i_ice_load_anchor.py`): the "
      "unaltered IACS PDFs in `benchmarks/anchors/sources/iacs/` are text-extracted; each anchor "
      "is the exact substring between two textual markers on one page, hashed "
      "(`sha256_file`, `sha256_quote`), and every constant used by the design-load routine is "
      "re-parsed out of those quotes and machine-compared with the Python literal.")
    A("2. **Ensemble** (`benchmarks/ice_coupled_6dof/run_w30_ice_load_vs_code.py`): random floe "
      "field (Poisson arrivals with rate n(B+D)V, lognormal diameters, Gaussian thicknesses, "
      "random azimuth and drift) advanced against the hull with velocity-Verlet; for every "
      "impact cluster the peak contact force, peak contact area and peak contact pressure are "
      "recorded. The contact law is the ISO 19906 pressure--area form with declared coefficients.")
    A("3. **Comparison**: IACS UR I2 bow design load from the transcribed clauses (per Polar "
      "Class), plus distribution fits (Weibull MLE, Gumbel) with KS/AIC and tail extrapolation.")
    A("")
    A("### Anchored clauses (each with sha256)")
    A("")
    A("| anchor | clause | what it carries | page | sha256(quote) |")
    A("|---|---|---|---|---|")
    for a in anchor["anchors"]:
        A(f"| `{a['anchor_id']}` | {a['clause']} | {a['carries']} | {a.get('pdf_page')} | "
          f"`{a['sha256_quote'][:16]}...` |")
    A("")
    A(f"Source documents: " + "; ".join(
        f"`{s['file']}` ({s.get('revision_as_printed', s.get('revision', 'n/a'))}, sha256 `{s['sha256_file']}`, "
        f"{s.get('pages', '?')} pp, {s['url']})" for s in data["anchors"]["documents"]) + ".")
    A("")
    A(f"ISO 19906 status: **{anchor['iso_19906']['status']}** — "
      f"{anchor['iso_19906']['entries'][0]['reason']}. Declared coefficients used downstream: "
      f"C_R = {iso['C_R_MPa']} MPa, A_0 = {iso['A_0_m2']} m2, ex = {iso['ex']} "
      f"(grade `{iso['grade']}`).")
    A("")

    # ---- 3 results ----
    A("## 3. Results")
    A("")
    A("### 3.1 Ensemble peak-load statistics (per impact)")
    A("")
    A("| group | impacts | mean F [MN] | sd F [MN] | p95 F [MN] | max F [MN] | mean p [MPa] | "
      "max p [MPa] | code area / mean area |")
    A("|---|---|---|---|---|---|---|---|---|")
    for g in data["groups"]:
        s = g["statistics"]
        A(f"| {g['group']['name']} | {s['impacts']} | {fnum(s['peak_force_MN']['mean'], 4)} | "
          f"{fnum(s['peak_force_MN']['sd'], 4)} | {fnum(s['peak_force_MN']['p95'], 4)} | "
          f"{fnum(s['peak_force_MN']['max'], 4)} | {fnum(s['peak_pressure_MPa']['mean'])} | "
          f"{fnum(s['peak_pressure_MPa']['max'])} | "
          f"{fnum(s['area_overlap_with_code_patch']['area_ratio_code_over_ensemble'], 1)} |")
    A("")
    A("### 3.2 Convergence on the seed axis (criterion 2)")
    A("")
    A("| group | full-mean F [MN] | last-10-seed mean F [MN] | relative difference | <=5%? |")
    A("|---|---|---|---|---|")
    for g in data["groups"]:
        c = g["statistics"]["convergence"]
        A(f"| {g['group']['name']} | {fnum(c['mean_force_full_MN'], 4)} | "
          f"{fnum(c['mean_force_tail_MN'], 4)} | {c['mean_force_rel_diff']:+.2%} | "
          f"{'yes' if c['passed'] else 'no'} |")
    A("")
    A("### 3.3 Distribution fits and tail assumption (criterion 4)")
    A("")
    A("| group | Weibull k | Weibull KS p | Gumbel scale | Gumbel KS p | preferred (AIC) | "
      "extrapolated q(1/N) [MN] | observed max [MN] |")
    A("|---|---|---|---|---|---|---|---|")
    for g in data["groups"]:
        f = g["distribution_fits"]
        n = g["statistics"]["impacts"]
        qn = g["tail"][f["preferred_by_aic"]].get(f"q_{n}")
        A(f"| {g['group']['name']} | {fnum(f['weibull']['params']['k'], 2)} | "
          f"{fnum(f['weibull']['ks_pvalue'], 3)} | {fnum(f['gumbel']['params']['scale'], 3)} | "
          f"{fnum(f['gumbel']['ks_pvalue'], 3)} | {f['preferred_by_aic']} | "
          f"{fnum(qn)} | {fnum(g['tail']['observed_max_MN'])} |")
    A("")
    A("Tail assumption: per-impact peaks treated as i.i.d.; the ensemble is clustered inside each "
      "floe field, so the fitted tail is optimistic. The extrapolated 1/N quantile is compared "
      "with the observed maximum in the table above — that column pair *is* the check.")
    A("")

    # ---- 3.4 comparison table ----
    A("### 3.4 Comparison with the code (criterion 3 and 5)")
    A("")
    A("| group | item | measured | reference | deviation | <15%? | cause class |")
    A("|---|---|---|---|---|---|---|")
    for g in data["groups"]:
        for i in g["comparison_items"]:
            dev = "n/a" if i.get("deviation_pct") is None else f"{i['deviation_pct']:+.1f}%"
            w15 = {True: "yes", False: "no", None: "not evaluated"}[i.get("within_15pct")]
            A(f"| {g['group']['name']} | {i['item']} | {fnum(i.get('measured'))} "
              f"{i['measured_unit']} | {fnum(i['reference'])} | {dev} | {w15} | "
              f"{i['difference_class']} |")
    A("")
    A(f"Code side (PC6, DUI = {code['displacement_kt']:.2f} kt): design force "
      f"{fnum(code['design_force_MN'])} MN, design pressure {fnum(code['design_pressure_MPa'])} "
      f"MPa, patch {fnum(code['patch_area_m2'])} m2 "
      f"(b = {fnum(code['patch_height_m'])} m, w = {fnum(code['patch_width_m'])} m). "
      f"IACS-implied pressure--area exponent: "
      f"{fnum(data['code_side']['code_pressure_area_exponent']['dlnp_dlnA'])}.")
    A("")

    # ---- criteria ----
    A("## 4. Criteria")
    A("")
    A("| criterion | statement | value | verdict |")
    A("|---|---|---|---|")
    for c in data["criteria"]:
        v = c["value"]
        if isinstance(v, dict):
            v = "; ".join(f"{k}={x}" for k, x in v.items())
        A(f"| {c['id']} | {c['statement']} | {v} | {'PASS' if c['passed'] else 'FAIL'} |")
    gate = data["criteria"][2]["evidence"]["deviation_gate_15pct"]
    A("")
    A(f"**15% deviation gate over anchored items: {gate}.** "
      f"Every anchored item is outside 15%; the causes are classified item by item as "
      f"`formulation_scope` (the IACS design load is a prescribed glancing impact of a large "
      f"floe, the ensemble is a random small-floe field), `model_difference` (same quantity, "
      f"different model/scale) or `closure_tautology` (the ISO law is inside the simulation, so "
      f"agreement only proves the implementation).")
    A("")

    # ---- 5 pitfalls ----
    A("## 5. Pitfalls, limitations, scope")
    A("")
    for note in data["honesty"]["not_claimed"]:
        A(f"- not claimed: {note}")
    A(f"- {data['honesty']['closure_arms']}")
    A(f"- {data['honesty']['iso_19906_withheld']}")
    A(f"- {data['honesty']['geometry_gate']}")
    for lim in anchor["limitations"]:
        A(f"- {lim}")
    A("- the area scale separation is explicit: the code patch is "
      f"{fnum(code['patch_area_m2'])} m2 while the G1-G3 peak contact areas are ~0.02-0.03 m2; "
      "only G4 overlaps the code patch area within a factor 2, and there the matched-band items "
      "are the only transport-free pressure comparisons.")
    A("")

    # ---- 6 error judgement ----
    A("## 6. Error judgement (vs baseline)")
    A("")
    A("| quantity | baseline | deviation | <3%? |")
    A("|---|---|---|---|")
    A("| ISO closure arm: measured ln p vs ln A slope | declared ex = "
      f"{iso['ex']} | see C5 item (closure, expected ~0) | "
      "yes by construction (closure, not validation) |")
    for i in data["groups"][0]["comparison_items"]:
        if i["item"] in ("C1-force-mean", "C3-pressure-peak-vs-Pavg"):
            A(f"| {i['item']} (G1) | IACS PC6 design value {fnum(i['reference'])} | "
              f"{i['deviation_pct']:+.1f}% | no |")
    A("")
    A("There is **no experimental benchmark** for the coupled ice-load level in this chain, so "
      "the 3% question can only be answered against the code value (above, not met) and against "
      "the closure arm (met by construction, but that is an implementation check). An honest "
      "statement is: the 3% criterion is **not met** for the physics comparison and the 15% gate "
      "is **not met** either; the reason is a formulation-scope gap, quantified below.")
    A("")

    # ---- figures ----
    A("## 7. Figures")
    A("")
    figs = []
    for name, cap in FIG_CAPTIONS.items():
        path = ART / "figures" / name
        if not path.is_file():
            continue
        A(f"![{name}](figures/{name})")
        A("")
        A(f"*{cap}*")
        A("")
        figs.append(dict(name=name, caption=cap,
                         png_base64=base64.b64encode(path.read_bytes()).decode("ascii")))
    A(f"(figure files: `{ART.relative_to(ROOT)}/figures/`)")

    markdown = "\n".join(L) + "\n"
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(markdown, encoding="utf-8")

    payload = {
        "title": ("ice-coupled-6dof W30: random-ice 6-DOF load ensemble vs IACS UR I2 Polar Class "
                  "design load (ISO 19906 withheld)"),
        "summary": (
            f"{len(data['groups'])} parameter groups x {len(data['experiment']['seeds'])} seeds, "
            f"{sum(g['statistics']['impacts'] for g in data['groups'])} impacts. "
            f"IACS UR I2 Rev.5 anchored at {data['anchors']['anchor_count']} clauses with "
            f"sha256; ISO 19906 WITHHELD (no licensed copy). Ensemble mean peak force "
            f"{data['groups'][0]['statistics']['peak_force_MN']['mean']:.3f} MN (G1) vs PC6 "
            f"design {code['design_force_MN']:.2f} MN: 15% gate FAIL, cause = formulation scope."),
        "scenario_key": "ice-coupled-6dof",
        "mission_id": data["mission_id"],
        "tags": ["ice", "6dof", "IACS-UR-I2", "ISO-19906", "polar-class",
                 "ice-load-ensemble", "weibull", "gumbel", "tensorfem"],
        "markdown": markdown,
        "figures": figs,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote {OUT_MD.relative_to(ROOT)} and {OUT_JSON.relative_to(ROOT)} "
          f"({len(figs)} figures, {len(markdown)} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())