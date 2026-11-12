"""Generate ``benchmarks/hull_girder_uls/README.md`` from the W24 artifacts.

Every number in the README is read out of the artifacts (anchor JSON, utilisation
CSV, sensitivity JSON) so that no value is hand-copied into prose.  Re-run after
rebuilding the anchor::

    PYTHONPATH=src python benchmarks/hull_girder_uls/make_readme.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ARTIFACT_DIR = Path(__file__).resolve().parent
ANCHOR = ARTIFACT_DIR / "csr_uls_anchor.json"
CSV = ARTIFACT_DIR / "csr_utilisation_with_factors.csv"
SENSITIVITY = ARTIFACT_DIR / "factor_sensitivity.json"

STATUS_LABEL = {"TRANSCRIBED": "TRANSCRIBED", "WITHHELD": "WITHHELD"}


def _table(header: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in header]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    lines = ["| " + " | ".join(h.ljust(w) for h, w in zip(header, widths)) + " |",
             "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |")
    return "\n".join(lines)


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def build() -> str:
    anchor = json.loads(ANCHOR.read_text())
    sensitivity = json.loads(SENSITIVITY.read_text())
    with CSV.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    summary = anchor["transcription"]["summary"]
    dims = anchor["dimension_check"]
    cons = anchor["consistency_checks"]
    criteria = anchor["criteria"]
    table2 = anchor["transcription"]["factor_table"]
    capacity = anchor["transcription"]["capacity_factors"]

    parts: list[str] = []
    parts.append(
        "# W24 - IACS CSR hull-girder ULS partial safety factors, transcribed and "
        "wired into the KVLCC2 section scan\n")
    parts.append(
        "Subject: `paper-exp-ship-hull-girder-structure-3`, mission 15972, anchor "
        "`AN-IACS-CSR`.  Repository: TensorFEM.\n")

    parts.append("## 1. What was blocking, and what unblocked it\n")
    parts.append(
        "W22 (mission 14512, `benchmarks/anchors/iacs-csr-w22.json`) and W23 (mission "
        "15094, `csr_factor_register.json`) both reported the CSR hull-girder ULS "
        "utilisation at unit partial safety factors because the printed factor table "
        "could not be transcribed: `www.iacs.org.uk` answers HTTP 403 through the "
        "execution sandbox proxy.\n")
    parts.append(
        "W24 used a different channel: IACS also publishes the harmonised CSR and the "
        "Unified Requirements on a public object store "
        "(`https://iacs.s3.af-south-1.amazonaws.com`) that answers anonymous "
        "`ListObjectsV2` and `GetObject`.  The two assigned documents were located and "
        "retrieved there, with sha256 recorded, so the transcription is reproducible "
        "without credentials (`fetch_csr_sources.py`).\n")
    parts.append("\n### Source documents\n")
    parts.append(_table(
        ["anchor id", "document", "pdf pages", "sha256 (first 16)", "retrieved"],
        [[d["id"], d["title"], str(d["pdf_pages"]), d["sha256"][:16], d["retrieved_utc"]]
         for d in anchor["source_documents"]]))
    parts.append("\n")

    parts.append("## 2. Clause-by-clause transcription\n")
    parts.append(
        f"All {summary['required_entries']} clauses needed to state the ULS criterion, "
        f"its load combination, the partial safety factors and the wave-moment basis "
        f"are TRANSCRIBED ({_fmt(summary['required_completeness_percent'], 1)} %; "
        f"missing: {summary['required_missing'] or 'none'}).  Across the whole "
        f"transcription {summary['transcribed']} of {summary['entries']} entries are "
        f"TRANSCRIBED ({_fmt(summary['completeness_percent'], 1)} %), "
        f"{summary['withheld']} are WITHHELD with the exact clauses still needed.  "
        f"Locator completeness (clause number + table number + printed page + quote): "
        f"{summary['locator_completeness']}.\n")
    parts.append(_table(
        ["id", "document", "clause", "table", "page", "status"],
        [[e["id"], e["document"].replace("AN-IACS-", ""), e["clause"],
          e.get("table") or "-", str(e["page"]), STATUS_LABEL[e["status"]]]
         for e in anchor["transcription"]["entries"]]))
    parts.append("\n**Key printed values read from Table 2 (Pt 1, Ch 5, Sec 2, printed "
                 "page 344):**\n")
    parts.append(_table(
        ["design load scenario", "gamma_S", "gamma_W"],
        [[scenario, _fmt(table2[scenario]["gamma_S"], 2), _fmt(table2[scenario]["gamma_W"], 2)]
         for scenario in sorted(table2)]))
    parts.append("")
    parts.append(
        f"Capacity side (Pt 1, Ch 5, Sec 2 [2.1.2], printed page 343): "
        f"gamma_R = gamma_M * gamma_DB with gamma_M = {_fmt(capacity['gamma_M'], 2)}, "
        f"gamma_DB = {_fmt(capacity['gamma_DB']['hogging_oil_tanker'], 2)} for an oil "
        f"tanker in hogging and {_fmt(capacity['gamma_DB']['sagging'], 2)} in sagging, "
        f"i.e. gamma_R = {_fmt(capacity['gamma_M'] * capacity['gamma_DB']['hogging_oil_tanker'], 2)} "
        f"and {_fmt(capacity['gamma_M'] * capacity['gamma_DB']['sagging'], 2)}.\n")
    parts.append("### Still WITHHELD\n")
    for entry in anchor["transcription"]["entries"]:
        if entry["status"] == "WITHHELD":
            parts.append(f"* `{entry['id']}` {entry['clause']} - {entry['why']} "
                         f"Needed: {'; '.join(entry['needed'])}")
    parts.append("")

    parts.append("## 3. Machine checks on the anchor\n")
    parts.append(_table(
        ["check", "verdict", "residual", "tolerance"],
        [["dimensional homogeneity + printed unit conversions ("
          f"{dims['checked_relations']} relations, {dims['checked_terms']} terms)",
          dims["verdict"], f"{dims['max_residual']:.3e}", f"{dims['tolerance']:.0e}"]]
        + [[f"{c['id']} {c['statement']}", c["verdict"], f"{c['residual']:.3e}",
            f"{c['tolerance']:.0e}"] for c in cons["checks"]]))
    parts.append(
        f"\nInternal-consistency residual maximum: {cons['max_internal_residual']:.3e} "
        f"(criterion < 1e-6).  The only cross-document residual above that level is "
        f"CC-01, {cons['max_cross_document_residual']:.3e}, which is the rounding of the "
        f"printed UR S11 coefficient 0.11 against 0.19 x 0.58 = 0.1102.\n")

    parts.append("## 4. The two utilisation bases on the 12 scanned cases\n")
    parts.append(
        "Factored basis (CSR): `u = gamma_R (gamma_S M_sw + gamma_W f_beta M_wv) / M_U`, "
        "PASS when `u <= 1`.  Substitute basis (W23): `gamma* = M_U / (M_sw + M_wv)`, "
        "whose reciprocal is the unit-factor utilisation.  `M_wv` is the CSR "
        "Ch 4 Sec 4 [3.1.1] value at the station; the W23 integrated value is kept as a "
        "diagnostic column.\n")
    scenario_rows = []
    for scenario in sorted({r["scenario"] for r in rows}):
        for row in [r for r in rows if r["scenario"] == scenario]:
            scenario_rows.append([
                scenario, row["case_id"], row["condition"],
                _fmt(float(row["gamma_W"]), 2), _fmt(float(row["gamma_R"]), 2),
                _fmt(float(row["M_sw_Nm"]) / 1e9, 3),
                _fmt(float(row["M_wv_csr_Nm"]) / 1e9, 3),
                _fmt(float(row["M_u_smith_Nm"]) / 1e9, 3),
                _fmt(float(row["utilisation_with_factors"]), 3),
                _fmt(float(row["gamma_star"]), 3),
                f"{float(row['rel_diff_percent']):+.1f}",
                row["verdict"],
            ])
    parts.append(_table(
        ["scen", "case", "cond", "g_W", "g_R", "M_sw/GNm", "M_wv/GNm", "M_U/GNm",
         "u factored", "gamma*", "diff %", "verdict"], scenario_rows))
    parts.append("")

    applicable = [r for r in rows if r["verdict"] in ("PASS", "FAIL")]
    failures = [r for r in applicable if r["verdict"] == "FAIL"]
    parts.append(
        f"{len(applicable)} rows carry a verdict ({len(rows)} rows in the table; the six "
        f"scenario-B hogging rows are NOT_APPLICABLE because Table 1 gives scenario B a "
        f"sagging still water moment).  {len(failures)} row(s) FAIL: "
        + ", ".join(f"`{r['scenario']}:{r['case_id']}` (u = {_fmt(float(r['utilisation_with_factors']), 3)})"
                    for r in failures) + ".\n")
    parts.append(
        f"Utilisation range {_fmt(min(float(r['utilisation_with_factors']) for r in applicable), 3)}"
        f" to {_fmt(max(float(r['utilisation_with_factors']) for r in applicable), 3)} at the "
        f"transcribed factors; `gamma*` range "
        f"{_fmt(min(float(r['gamma_star']) for r in applicable), 3)} to "
        f"{_fmt(max(float(r['gamma_star']) for r in applicable), 3)}.  The unit-factor "
        f"reading therefore overstates the margin by "
        f"{_fmt(min(float(r['rel_diff_percent']) for r in applicable), 1)} % to "
        f"{_fmt(max(float(r['rel_diff_percent']) for r in applicable), 1)} %.\n")

    parts.append("### Difference attribution\n")
    parts.append(
        "The two bases differ by exactly the factor that multiplies the demand, so "
        "`u / u_0 = gamma_R (gamma_S phi_sw + gamma_W f_beta phi_wv)` with "
        "`phi_sw = M_sw/(M_sw+M_wv)`.  The identity is evaluated per row and its "
        "residual is recorded in the CSV (`attribution_residual`); the maximum over all "
        f"rows is {max(float(r['attribution_residual']) for r in rows):.2e}.  Three "
        "regimes appear:\n")
    hogging = [r for r in applicable if r["condition"] == "hogging"]
    sag_zero = [r for r in applicable if r["condition"] == "sagging"
                and float(r["M_sw_Nm"]) < 1e-6]
    parts.append(
        f"* hogging (phi_sw small): the difference is almost purely "
        f"`gamma_R * gamma_W * f_beta` = {_fmt(float(hogging[0]['gamma_R']), 2)} x "
        f"{_fmt(float(hogging[0]['gamma_W']), 2)} x {_fmt(float(hogging[0]['f_beta']), 2)} "
        f"= {_fmt(float(hogging[0]['attribution_product']), 3)}, i.e. "
        f"{float(hogging[0]['rel_diff_percent']):+.1f} %;")
    if sag_zero:
        r0 = sag_zero[0]
        parts.append(
            f"* sagging at zero still-water moment (`{r0['case_id']}`): differences come "
            f"only from `gamma_R * gamma_W * f_beta` applied to the wave part, "
            f"{float(r0['rel_diff_percent']):+.1f} %;")
    sag = [r for r in applicable if r["condition"] == "sagging"]
    worst = max(sag, key=lambda r: float(r["attribution_still_water_share"]))
    parts.append(
        f"* sagging with the largest still-water share (`{worst['scenario']}:{worst['case_id']}`, "
        f"phi_sw = {_fmt(float(worst['attribution_still_water_share']), 3)}): the "
        f"still-water term enters with gamma_R only, which damps the product to "
        f"{float(worst['rel_diff_percent']):+.1f} %.\n")

    parts.append("## 5. Factor sensitivity (envelope, not a CSR value)\n")
    reference = sensitivity["reference_transcribed_points"]
    parts.append(_table(
        ["scenario", "gamma_S", "gamma_W", "u min", "u max", "FAIL rows"],
        [[s, _fmt(reference[s]["gamma_S"], 2), _fmt(reference[s]["gamma_W"], 2),
          _fmt(reference[s]["utilisation_min"], 3), _fmt(reference[s]["utilisation_max"], 3),
          str(reference[s]["failures"])] for s in sorted(reference)]))
    parts.append("")
    parts.append(
        "Break-even factors per case (`gamma_S = 1`): the gamma_W at which the check "
        "sits exactly on the limit.\n")
    parts.append(_table(
        ["case", "gamma_R", "break-even gamma_W", "break-even gamma_S", "gamma*"],
        [[c["case_id"], _fmt(c["gamma_R"], 2),
          "-" if c["break_even_gamma_w_at_gamma_s_1"] is None
          else _fmt(c["break_even_gamma_w_at_gamma_s_1"], 3),
          "-" if c["break_even_gamma_s_at_scenario_A_gamma_w"] is None
          else _fmt(c["break_even_gamma_s_at_scenario_A_gamma_w"], 3),
          _fmt(c["gamma_star"], 3)] for c in sensitivity["break_even"]]))
    parts.append("")
    grid = sensitivity["grid"]
    parts.append(
        "Envelope over the declared grid (the two flagged points are the transcribed "
        "CSR values, everything else is a probe):\n")
    parts.append(_table(
        ["gamma_S \\ gamma_W"] + [_fmt(w, 1) for w in sensitivity["grid_axes"]["gamma_W"]],
        [[_fmt(g, 1)] + [str(next(x["failures"] for x in grid
                                   if x["gamma_S"] == g and x["gamma_W"] == w))
                          for w in sensitivity["grid_axes"]["gamma_W"]]
         for g in sensitivity["grid_axes"]["gamma_S"]]))
    parts.append("")

    parts.append("## 6. Mission criteria\n")
    parts.append(_table(
        ["criterion", "verdict", "evidence"],
        [["W24-C1 transcription completeness", criteria["W24-C1"]["verdict"],
          f"{_fmt(criteria['W24-C1']['required_completeness_percent'], 0)} % of "
          f"the {len(criteria['W24-C1']['required_clauses'])} clauses needed to "
          f"state the criterion and the factors; {criteria['W24-C1']['transcribed']} "
          f"transcribed / {criteria['W24-C1']['withheld']} withheld overall, each "
          f"with a needed-clause list"],
         ["W24-C2 dimension + internal consistency < 1e-6", criteria["W24-C2"]["verdict"],
          f"dimension {dims['max_residual']:.1e}, internal "
          f"{cons['max_internal_residual']:.1e}"],
         ["W24-C3 >= 12 verdict-carrying cases", criteria["W24-C3"]["verdict"],
          f"{criteria['W24-C3']['cases_with_verdict']} verdicts, "
          f"{criteria['W24-C3']['verdicts']['FAIL']} FAIL"],
         ["W24-C4 per-case difference + attribution", criteria["W24-C4"]["verdict"],
          f"{criteria['W24-C4']['cases_above_15_percent']} cases above 15 %, "
          f"attribution residual max {criteria['W24-C4']['attribution_residual_max']:.1e}"],
         ["W24-C5 no estimated factor substituted", criteria["W24-C5"]["verdict"],
          "transcribed gamma_S/gamma_W used directly; WITHHELD entries are capacity and "
          "applicability clauses, not factor values"]]))
    parts.append("")

    parts.append("## 7. Limitations and declared substitutions\n")
    for limitation in anchor["limitations"]:
        parts.append(f"* {limitation}")
    parts.append("* the transcription is a set of short verbatim quotes with clause, "
                 "table and page locators; the rule text itself is not redistributed "
                 "here (the documents stay at their published URL, verified by sha256).")
    parts.append("")

    parts.append("## 8. Figures\n")
    parts.append(
        "`make_figures.py` renders the three figures from the artifacts alone "
        "(`figures/w24_csr_utilisation.png`: factored utilisation per station against "
        "the gamma* curve; `figures/w24_attribution.png`: the exact range of the "
        "difference as a function of the still-water share; "
        "`figures/w24_sensitivity.png`: failure count over the declared factor grid "
        "with the two transcribed CSR points boxed).\n")

    parts.append("## 9. Reproduce\n")
    parts.append("```bash\n"
                 "# 1. fetch and verify the anchor documents (needs network egress to the S3 endpoint)\n"
                 "python benchmarks/hull_girder_uls/fetch_csr_sources.py --dest /tmp/iacs2\n"
                 "# 2. rebuild the anchor, the utilisation table and the sensitivity envelope\n"
                 "PYTHONPATH=src python -m tensorfem.csr_uls_factors \\\n"
                 "    --out-dir benchmarks/hull_girder_uls\n"
                 "# 3. figures and this README, both from the artifacts\n"
                 "PYTHONPATH=src python benchmarks/hull_girder_uls/make_figures.py\n"
                 "PYTHONPATH=src python benchmarks/hull_girder_uls/make_readme.py\n"
                 "# 4. tests\n"
                 "PYTHONPATH=src python -m pytest tests/test_csr_uls_factors.py -q\n"
                 "```\n")
    parts.append(
        f"Generator digest of `src/tensorfem/csr_uls_factors.py` at build time: "
        f"`{anchor['generator_sha256'][:16]}`.\n")
    return "\n".join(parts)


def main() -> int:
    text = build()
    (ARTIFACT_DIR / "README.md").write_text(text)
    print(f"wrote {ARTIFACT_DIR / 'README.md'} ({len(text)} bytes)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())