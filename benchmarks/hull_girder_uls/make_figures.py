"""Figures for the W24 CSR ULS factor study (generated from the artifacts only).

    PYTHONPATH=src python benchmarks/hull_girder_uls/make_figures.py

Writes three PNGs next to the artifacts and prints their paths, so the mission
report can embed them without any number being typed by hand.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

ARTIFACT_DIR = Path(__file__).resolve().parent
FIGURE_DIR = ARTIFACT_DIR / "figures"
CSV = ARTIFACT_DIR / "csr_utilisation_with_factors.csv"
SENSITIVITY = ARTIFACT_DIR / "factor_sensitivity.json"
ANCHOR = ARTIFACT_DIR / "csr_uls_anchor.json"


def _rows() -> list[dict]:
    with CSV.open(newline="") as handle:
        return list(csv.DictReader(handle))


def figure_utilisation(rows: list[dict]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)
    for ax, scenario in zip(axes, ("A", "B")):
        subset = [r for r in rows if r["scenario"] == scenario]
        for condition, colour, marker in (("sagging", "#c62828", "o"),
                                          ("hogging", "#1565c0", "s")):
            cases = [r for r in subset if r["condition"] == condition]
            xs = [float(r["station_x_over_L"]) for r in cases]
            us = [float(r["utilisation_with_factors"]) for r in cases]
            applicable = [r["verdict"] in ("PASS", "FAIL") for r in cases]
            ax.plot(xs, us, marker=marker, color=colour, label=f"{condition} (factored)")
            ax.scatter([x for x, ok in zip(xs, applicable) if ok],
                       [u for u, ok in zip(us, applicable) if ok],
                       s=70, facecolors="none", edgecolors=colour,
                       label=f"{condition} CSR load case")
        stars = [float(r["gamma_star"]) for r in subset]
        xs = [float(r["station_x_over_L"]) for r in subset]
        ax.plot(xs, stars, linestyle=":", color="#424242",
                label=r"$\gamma^* = M_U/(M_{sw}+M_{wv})$")
        ax.axhline(1.0, color="k", linewidth=0.8)
        factors = subset[0]
        ax.set_title(f"scenario {scenario}: " r"$\gamma_S$" f"={float(factors['gamma_S']):.1f}, "
                     r"$\gamma_W$" f"={float(factors['gamma_W']):.1f}, "
                     r"$\gamma_R$" f"={float(factors['gamma_R']):.2f} (hogging)")
        ax.set_xlabel("station x / L")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("utilisation  (PASS if <= 1)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("KVLCC2 6 sections x 2 wave conditions: CSR factored utilisation "
                 "against the gamma* substitute basis", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    path = FIGURE_DIR / "w24_csr_utilisation.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_attribution(rows: list[dict]) -> Path:
    """Exact linear decomposition of u/u_0 in the still-water share.

    ``u/u_0 = gamma_R (gamma_S phi_sw + gamma_W f_beta (1 - phi_sw))`` is linear in
    ``phi_sw``, so the two endpoints are the wave-only and the still-water-only
    limits of that same case; the actual value must lie between them.  Nothing is
    invented here: the interval comes from the transcribed factors, the marker from
    the artifact row.
    """
    applicable = [r for r in rows if r["verdict"] in ("PASS", "FAIL")]
    labels = [f"{r['scenario']}:{r['case_id'].replace('kvlcc2_', '')}" for r in applicable]
    total = [float(r["rel_diff_percent"]) for r in applicable]
    low, high, shares = [], [], []
    for r in applicable:
        gamma_r = float(r["gamma_R"])
        g_s = float(r["gamma_S"])
        wave_factor = float(r["attribution_wave_factor"])
        low.append((gamma_r * min(g_s, wave_factor) - 1.0) * 100.0)
        high.append((gamma_r * max(g_s, wave_factor) - 1.0) * 100.0)
        shares.append(float(r["attribution_still_water_share"]))
    fig, ax = plt.subplots(figsize=(11.5, 4.4))
    x = list(range(len(labels)))
    ax.vlines(x, low, high, color="#5c6bc0", linewidth=6,
              label=r"exact range of $\gamma_R(\gamma_S\phi_{sw}+\gamma_W f_\beta(1-\phi_{sw}))$")
    ax.scatter(x, total, color="k", zorder=3, s=18, label="reported difference (artifact row)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel(r"$(u/u_0 - 1)$  [ % ]")
    for index, share in enumerate(shares):
        ax.annotate(f"{share:.2f}", (index, total[index]), fontsize=6,
                    textcoords="offset points", xytext=(0, 6), ha="center")
    ax.set_title("Difference between the factored utilisation and the gamma* basis: exact "
                 "range from the transcribed factors (labels = still-water share)", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIGURE_DIR / "w24_attribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_sensitivity(sensitivity: dict) -> Path:
    axes_s = sensitivity["grid_axes"]["gamma_S"]
    axes_w = sensitivity["grid_axes"]["gamma_W"]
    grid = {(g["gamma_S"], g["gamma_W"]): g for g in sensitivity["grid"]}
    matrix = [[grid[(s, w)]["failures"] for w in axes_w] for s in axes_s]
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    image = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", origin="lower")
    ax.set_xticks(range(len(axes_w)))
    ax.set_xticklabels([f"{w:.1f}" for w in axes_w])
    ax.set_yticks(range(len(axes_s)))
    ax.set_yticklabels([f"{s:.1f}" for s in axes_s])
    ax.set_xlabel(r"$\gamma_W$")
    ax.set_ylabel(r"$\gamma_S$")
    for i, s in enumerate(axes_s):
        for j, w in enumerate(axes_w):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center", fontsize=9)
            if grid[(s, w)]["is_transcribed_csr_point"]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor="#0d47a1", linewidth=2.2))
    fig.colorbar(image, ax=ax, label="cases failing the CSR check (of "
                 f"{len(sensitivity['break_even'])})")
    ax.set_title("Sensitivity envelope: boxed cells are the transcribed CSR Table 2 "
                 "points (never quoted as anything else)", fontsize=9)
    fig.tight_layout()
    path = FIGURE_DIR / "w24_sensitivity.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> int:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    rows = _rows()
    sensitivity = json.loads(SENSITIVITY.read_text())
    paths = [figure_utilisation(rows), figure_attribution(rows),
             figure_sensitivity(sensitivity)]
    anchor = json.loads(ANCHOR.read_text())
    for path in paths:
        print(path)
    print("anchor criteria:", {k: v["verdict"] for k, v in anchor["criteria"].items()})
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())