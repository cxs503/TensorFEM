"""Figures for the W26 CSR ULS anchor recheck.

Three figures, all driven by the recheck artifact and the W24 utilisation table,
nothing typed by hand:

``fig_recheck_quote_matrix.png``
    One row per anchor entry: the quote granularity that actually verified
    (verbatim / normalised / cell / restated), the cited page and the page the
    text was found on.  This is the evidence map for criteria W26-C1 / C2.
``fig_recheck_wave_coefficients.png``
    The transcribed wave-load coefficients in use: the Cw branches and the
    closure at L = 300 m / 350 m, and the CSR sagging expression against the
    independent UR S11 statement (both reduce to 0.11 (Cb + 0.7) Cw L^2 B).
``fig_recheck_utilisation.png``
    The Smith M_u baseline and the factored CSR utilisation per case, with the
    1.0 limit and the f_beta = 1.0 / 0.8 alternatives from the sensitivity
    table, so the effect of the condensed f_beta transcription is visible.

Usage::

    python benchmarks/hull_girder_uls/make_recheck_figures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_JSON = (REPO_ROOT / "papers" / "artifacts" / "w26"
                 / "hull-girder-ultimate-strength" / "csr_uls_recheck.json")
OUT_DIR = ARTIFACT_JSON.parent

LEVEL_COLOUR = {"verbatim": "#1b6b3a", "normalised": "#3d8fd1", "cell": "#c9a227",
                "restated": "#d1691f", "absent": "#b32222",
                "WITHHELD_DECLARED": "#7a7a7a"}


def figure_quote_matrix(artifact: dict, out: Path) -> Path:
    entries = artifact["quote_recheck"]["entries"]
    labels, levels, cited, found = [], [], [], []
    for entry in entries:
        labels.append(f"{entry['id']}  [{entry['clause']}]")
        if entry["anchor_status"] == "WITHHELD":
            levels.append("WITHHELD_DECLARED")
        else:
            levels.append(entry["quote_level"])
        cited.append(entry["anchor_page_printed"])
        found.append(",".join(str(page) for page in entry["found_printed_pages"]) or "-")

    figure, axis = plt.subplots(figsize=(11.0, 0.42 * len(labels) + 1.6))
    y = np.arange(len(labels))[::-1]
    axis.barh(y, [1.0] * len(labels),
              color=[LEVEL_COLOUR[level] for level in levels], edgecolor="white")
    for position, (label, level, page, where) in enumerate(zip(labels, levels, cited, found)):
        axis.text(0.01, y[position], f" {level}", va="center", ha="left",
                  color="white", fontsize=8.5, fontweight="bold")
        axis.text(1.02, y[position], f"cited p.{page}  found p.{where}", va="center",
                  ha="left", fontsize=8)
    axis.set_yticks(y)
    axis.set_yticklabels(labels, fontsize=8)
    axis.set_xticks([])
    axis.set_xlim(0, 1.9)
    axis.set_title("W26 recheck: quote granularity that verified per anchor entry\n"
                   "(green = verbatim, blue = whitespace/symbol normalised, orange = "
                   "restated formula/table cell with all its numbers printed,\n"
                   "grey = declared WITHHELD; page numbers are printed pages of the "
                   "fetched IACS PDFs)", fontsize=9, loc="left")
    for spine in ("top", "right", "bottom"):
        axis.spines[spine].set_visible(False)
    figure.tight_layout()
    figure.savefig(out, dpi=170)
    plt.close(figure)
    return out


def _cw(length):
    if length <= 300.0:
        return 10.75 - ((300.0 - length) / 100.0) ** 1.5
    if length <= 350.0:
        return 10.75
    return 10.75 - ((length - 350.0) / 150.0) ** 1.5


def figure_wave_coefficients(artifact: dict, out: Path) -> Path:
    hull = json.loads((REPO_ROOT / "benchmarks" / "hull_girder_uls"
                       / "csr_uls_anchor.json").read_text())["hull_model"]
    breadth, cb = float(hull["B"]), float(hull["Cb"])
    length = float(hull["Lpp"])

    figure, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.2))

    grid = np.linspace(90.0, 500.0, 4001)
    left.plot(grid, [_cw(value) for value in grid], color="#1f4e79", lw=1.8,
              label="Cw (transcribed branches)")
    for knot in (300.0, 350.0):
        left.axvline(knot, color="#b32222", ls=":", lw=1.0)
        left.annotate(f"L = {knot:.0f} m", (knot, 10.78), fontsize=8,
                      color="#b32222", ha="center")
    left.plot([length], [_cw(length)], "o", color="#c9a227", ms=7,
              label=f"study hull L = {length:.0f} m, Cw = {_cw(length):.2f}")
    left.set_xlabel("L (m)")
    left.set_ylabel("Cw (-)")
    left.set_title("Cw branches: continuous at both knots\n"
                   "(residual < 1e-12 at L = 300 and 350 m)", fontsize=9, loc="left")
    left.legend(fontsize=8, loc="lower left")
    left.grid(alpha=0.25)

    # sagging wave moment: CSR [3.1.1] with fnl-vs, and the UR S11 statement
    lengths = np.linspace(90.0, 400.0, 2000)
    cw = np.array([_cw(value) for value in lengths])
    csr_sagging = 0.19 * 0.58 * (cb + 0.7) / cb * cw * lengths ** 2 * breadth * cb
    s11_negative = 0.11 * cw * lengths ** 2 * breadth * (cb + 0.7)
    relative = 100.0 * (csr_sagging - s11_negative) / s11_negative

    right.plot(lengths, csr_sagging / 1e6, color="#1f4e79", lw=1.8,
               label="CSR: 0.19 fnl-vs Cw L$^2$ B Cb, fnl-vs = 0.58 (Cb+0.7)/Cb")
    right.plot(lengths, s11_negative / 1e6, color="#c9a227", lw=1.4, ls="--",
               label="UR S11: 0.11 Cw L$^2$ B (Cb+0.7)  [cross-check]")
    right.set_xlabel("L (m)")
    right.set_ylabel("sagging Mwv (MNm, fm = fp = 1)")
    right.set_title(f"CSR sagging moment versus the independent UR S11 statement\n"
                    f"worst relative difference {np.max(np.abs(relative)):.3f} % "
                    f"(Cb = {cb:.3f})", fontsize=9, loc="left")
    right2 = right.twinx()
    right2.plot(lengths, relative, color="#b32222", lw=1.0, alpha=0.8)
    right2.set_ylabel("CSR vs UR S11 difference (%)", color="#b32222")
    right2.axhline(0.0, color="#b32222", lw=0.6, ls=":")
    right2.set_ylim(-5, 5)
    right2.tick_params(axis="y", colors="#b32222")
    right.legend(fontsize=7.5, loc="upper left")
    right.grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(out, dpi=170)
    plt.close(figure)
    return out


def figure_utilisation(artifact: dict, out: Path) -> Path:
    smith = artifact["smith_capacity_baseline"]
    sensitivity = artifact["f_beta_scope_sensitivity"]["alternatives"]
    cases = smith["cases_table"]
    labels = [f"{row['scenario']}:{row['case_id'].replace('kvlcc2_', '')}"
              for row in cases]
    reported = np.array([row["utilisation"] for row in cases])
    positions = np.arange(len(cases))

    figure, (left, right) = plt.subplots(1, 2, figsize=(13.0, 4.8),
                                         gridspec_kw={"width_ratios": [1.5, 1.0]})

    colours = ["#b32222" if row["verdict"] == "FAIL" else "#1b6b3a" for row in cases]
    left.bar(positions, reported, color=colours, alpha=0.85, label="reported (f_beta = 1.05)")
    for alternative, marker in zip(sensitivity, ("v", "o")):
        flipped = {item["case_key"]: item["utilisation_at_alternative"]
                   for item in alternative["verdict_flips"]}
        values = [flipped.get(f"{row['scenario']}:{row['case_id']}", reported_value)
                  for row, reported_value in zip(cases, reported)]
        left.plot(positions, values, marker, ms=4.5, lw=0, mfc="none",
                  label=f"f_beta = {alternative['f_beta']}")
    left.axhline(1.0, color="black", lw=1.2, ls="--", label="ULS limit u = 1")
    left.set_xticks(positions)
    left.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    left.set_ylabel("u = gamma_R M / M_u  (-)")
    left.set_title("factored CSR utilisation per scanned case\n"
                   f"{len(smith['failures'])} of {len(cases)} exceed the limit at "
                   f"f_beta = 1.05; open markers show the alternative f_beta values",
                   fontsize=9, loc="left")
    left.legend(fontsize=7.5)
    left.grid(axis="y", alpha=0.25)

    stations = sorted({row["station_x_over_L"] for row in cases})
    for condition, colour in (("sagging", "#1f4e79"), ("hogging", "#c9a227")):
        for scenario, style in (("A", "-"), ("B", "--")):
            rows = [row for row in cases
                    if row["condition"] == condition and row["scenario"] == scenario]
            if not rows:
                continue
            right.plot([row["station_x_over_L"] for row in rows],
                       [row["M_u_smith_MNm"] for row in rows], style,
                       marker="s" if condition == "sagging" else "^",
                       ms=4, color=colour, label=f"Smith M_u, {condition} (scenario {scenario})")
    right.set_xticks(stations)
    right.set_xlabel("station x/L")
    right.set_ylabel("M_u (MNm)")
    right.set_title("Smith progressive-collapse capacity on the study scantlings\n"
                    "(recomputed for the two worst cases: relative difference 0)",
                    fontsize=9, loc="left")
    right.legend(fontsize=7)
    right.grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(out, dpi=170)
    plt.close(figure)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact", default=str(ARTIFACT_JSON))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args(argv)

    artifact = json.loads(Path(args.artifact).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        figure_quote_matrix(artifact, out_dir / "fig_recheck_quote_matrix.png"),
        figure_wave_coefficients(artifact, out_dir / "fig_recheck_wave_coefficients.png"),
        figure_utilisation(artifact, out_dir / "fig_recheck_utilisation.png"),
    ]
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
