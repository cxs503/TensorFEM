"""Figures for the beam reference-method selection study (docs/CASE-...md)."""
from __future__ import annotations

import json
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
RECORD = ROOT / "benchmarks" / "beam_reference_method_selection_record.json"
OUT = ROOT / "docs" / "figures"


def main() -> int:
    d = json.loads(RECORD.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0))

    states = d["states"]
    sl = [s["slenderness"] for s in states]

    # (a) EB model deficit vs slenderness, two probes, tolerance lines
    ax = axes[0, 0]
    for probe, style in (("w_tip", "o-"), ("w_mid", "s-")):
        err = [s["probes_nodal"]["EB-closed"][probe] * 100.0 for s in states]
        ax.loglog(sl, err, style, label=f"EB deficit on `{probe}`")
    for tol, colour in ((3.0, "tab:red"), (15.0, "tab:orange")):
        ax.axhline(tol, color=colour, ls="--", lw=1.0)
        ax.text(sl[-1] * 0.95, tol * 1.06, f"{tol:.0f} % line", color=colour,
                ha="right", va="bottom", fontsize=8)
    for probe, colour in (("w_tip", "tab:blue"), ("w_mid", "tab:green")):
        for tol in (0.03, 0.15):
            be = d["breakeven"][f"tol={tol}"][f"closed_form_{probe}"]
            ax.plot([be], [tol * 100.0], "*", ms=11, color=colour)
    ax.set_xlabel("slenderness L/h")
    ax.set_ylabel("Euler-Bernoulli model deficit vs Timoshenko [%]")
    ax.set_title("(a) reference-model deficit, nodal protocol")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    # (b) protocol inflation: same model, coarse field sampling
    ax = axes[0, 1]
    nodal = [s["probes_nodal"]["EB-closed"]["w_mid"] * 100.0 for s in states]
    field = [s["probes_field"]["EB-fem-n1-field"]["w_mid"] * 100.0 for s in states]
    ax.loglog(sl, nodal, "s-", label="nodal protocol (mesh-resolved)")
    ax.loglog(sl, field, "^-", label="field protocol (1-element interpolation)")
    ax.axhline(3.0, color="tab:red", ls="--", lw=1.0)
    ax.text(sl[2], 3.2, "3 % line", color="tab:red", fontsize=8)
    for s, n, f in zip(sl, nodal, field):
        if abs(s - 10.0) < 1e-9:
            ax.annotate(f"{f/n:.2f}x inflation", (s, f), textcoords="offset points",
                        xytext=(8, 6), fontsize=8)
    ax.set_xlabel("slenderness L/h")
    ax.set_ylabel("inferred EB deficit on `w_mid` [%]")
    ax.set_title("(b) protocol bias: interpolation masquerading as model error")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    # (c) selection map: which reference is admissible
    ax = axes[1, 0]
    labels, keys = [], [
        ("nodal_all_probes|tol=0.03", "3 % · all probes"),
        ("tip_only|tol=0.03", "3 % · tip only"),
        ("nodal_all_probes|tol=0.15", "15 % · all probes"),
        ("tip_only|tol=0.15", "15 % · tip only"),
    ]
    for row, (key, label) in enumerate(keys):
        rows = d["selections"][key]["rows"]
        xs = [r["slenderness"] for r in rows]
        ys = [row + (0.28 if r["selected"] == "EB-closed" else 0.0) for r in rows]
        cols = ["tab:blue" if r["selected"] == "EB-closed" else "tab:purple" for r in rows]
        ax.scatter(xs, ys, c=cols, marker="s", s=70)
        labels.append(label)
    ax.set_yticks([i + 0.14 for i in range(len(keys))])
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("slenderness L/h")
    ax.set_title("(c) selected reference (blue = EB closed form, purple = Timoshenko)")
    ax.grid(True, which="both", alpha=0.3)

    # (d) field-probe convergence at L/h = 10
    ax = axes[1, 1]
    conv = d["field_probe_convergence_at_slenderness_10"]
    n = [c["elements"] for c in conv]
    err = [c["w_mid_rel_error"] * 100.0 for c in conv]
    ax.loglog(n, err, "o-")
    ax.axhline(3.0, color="tab:red", ls="--", lw=1.0)
    ax.text(n[-1] * 0.9, 3.2, "3 % line", color="tab:red", ha="right", fontsize=8)
    ax.annotate("both elements nodal-exact from n=2:\nresidual = pure model deficit",
                (n[1], err[1]), textcoords="offset points", xytext=(12, 18), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlabel("elements per span (EB, L/h = 10, probe at mid-span)")
    ax.set_ylabel("`w_mid` relative error [%]")
    ax.set_title("(d) field-sampled probe convergence")
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("TensorFEM · beam benchmark reference-model selection "
                 "(anchor XVAL-…-METHOD-SELECTION practice)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = OUT / "beam_reference_method_selection.png"
    fig.savefig(path, dpi=160)
    print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())