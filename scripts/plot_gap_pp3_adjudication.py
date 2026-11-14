"""Aggregate the independent adjudication JSONs into tables and figures.

Reads ``outputs/gap_pp3/nx<N>.json`` produced by
``scripts/adjudicate_gap_pp3_plastic_set.py`` and writes

* ``outputs/gap_pp3/summary.json`` -- machine-readable verdict table,
* ``outputs/gap_pp3/summary.md``   -- the same table as markdown,
* ``outputs/gap_pp3/figures/*.png`` -- load-deflection paths, mesh
  convergence against the rigid-plastic bracket, unload/reload residual set,
  and the elastic-bias convergence plot.

Nothing here recomputes mechanics: every number is read from the run JSON, so
the report tables and the figures cannot drift from the solver output.
"""
from __future__ import annotations

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "outputs", "gap_pp3")
FIG = os.path.join(OUT, "figures")


def load_levels() -> list[dict]:
    files = sorted(glob.glob(os.path.join(OUT, "nx*.json")),
                   key=lambda p: int(os.path.basename(p)[2:-5]))
    levels = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            levels.append(json.load(fh))
    return levels


def main() -> int:
    os.makedirs(FIG, exist_ok=True)
    levels = load_levels()
    if not levels:
        raise SystemExit("no level JSON found; run scripts/adjudicate_gap_pp3_plastic_set.py")
    ref = levels[0]["analytic"]

    # ---------------- table ----------------
    rows = []
    for r in levels:
        a = r["analytic"]
        rows.append({
            "nx": r["nx"],
            "py_analytic": a["py"],
            "first_yield_fe": r["first_yield_fe"],
            "first_yield_err_raw_pct": 100.0 * (r["first_yield_fe"] / a["py"] - 1.0)
            if r["first_yield_fe"] else None,
            "first_yield_err_same_mesh_pct": 100.0 * (r["first_yield_fe"] / r["elastic"]["same_mesh_py"] - 1.0)
            if r["first_yield_fe"] else None,
            "elastic_w_bias_pct": r["elastic"]["w_bias_pct"],
            "elastic_sigma_bias_pct": r["elastic"]["sigma_bias_pct"],
            "strip_lb": a["strip_lb"],
            "sine_ub": a["sine_ub"],
            "yield_line_ub": a["yield_line_ub"],
            "collapse_knee": r["collapse_knee"],
            "collapse_knee_vs_sine_pct": 100.0 * (r["collapse_knee"] / a["sine_ub"] - 1.0)
            if r["collapse_knee"] else None,
            "collapse_last_converged": r["collapse_last_converged"],
            "ramp_completed": r["ramp_completed"],
            "peak_w_over_t": r["loading"][-1]["w_over_t"],
            "permanent": r["permanent"],
            "peak_w": r["peak_w"],
            "permanent_over_peak": (r["permanent"] / r["peak_w"]
                                    if r["permanent"] is not None and r["peak_w"] else None),
            "unload_converged": r["unload_converged"],
        })

    summary = {"case": {"a": 1.0, "b": 1.0, "t": 0.02, "E": 210e9, "nu": 0.3,
                        "sigma_y": 355e6, "patch": [0.5, 0.5],
                        "hardening": levels[0]["hardening"],
                        "layers": levels[0]["layers"]},
               "analytic_reference": ref, "levels": rows}
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    head = ("| mesh | p_y analytic [MPa] | p_y FE [MPa] | raw err | same-mesh err | "
            "w bias | sigma bias | collapse knee [MPa] | vs sine bound | "
            "w_peak/t | permanent [mm] | resid/peak | unload ok |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    lines = [head]
    for r in rows:
        lines.append(
            f"| {r['nx']}x{r['nx']} | {r['py_analytic']/1e6:.3f} | "
            f"{(r['first_yield_fe'] or float('nan'))/1e6:.3f} | "
            f"{r['first_yield_err_raw_pct']:+.2f}% | {r['first_yield_err_same_mesh_pct']:+.2f}% | "
            f"{r['elastic_w_bias_pct']:+.2f}% | {r['elastic_sigma_bias_pct']:+.2f}% | "
            f"{(r['collapse_knee'] or float('nan'))/1e6:.3f} | "
            f"{(r['collapse_knee_vs_sine_pct'] if r['collapse_knee'] else float('nan')):+.2f}% | "
            f"{r['peak_w_over_t']:.2f} | "
            f"{(r['permanent']*1e3 if r['permanent'] is not None else float('nan')):.2f} | "
            f"{(r['permanent_over_peak'] if r['permanent_over_peak'] is not None else float('nan')):.3f} | "
            f"{'yes' if r['unload_converged'] else 'NO'} |")
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    # ---------------- figure 1: load-deflection ----------------
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    wpa = ref["w_per_pa_navier"]
    wmax = max(r["peak_w"] for r in levels) * 1.05
    wline = [wmax * k / 50 for k in range(51)]
    ax.plot([w / 0.02 for w in wline], [w / wpa / 1e6 for w in wline],
            "k--", lw=1.0, label="Navier elastic (independent)")
    colors = plt.cm.viridis([0.05 + 0.85 * i / max(len(levels) - 1, 1)
                             for i in range(len(levels))])
    for c, r in zip(colors, levels):
        f = [s["factor"] / 1e6 for s in r["loading"]]
        w = [s["w_over_t"] for s in r["loading"]]
        ax.plot(w, f, "o-", color=c, ms=3.2, lw=1.2, label=f"FE {r['nx']}x{r['nx']}")
        if r["unloading"]:
            fu = [s["factor"] / 1e6 for s in r["unloading"]]
            wu = [s["w"] / 0.02 for s in r["unloading"]]
            ax.plot(wu, fu, ":", color=c, lw=1.0)
    for val, lab, ls in ((ref["strip_lb"], "strip lower bound", "-."),
                         (ref["sine_ub"], "Ilyushin sine upper bound", "-."),
                         (ref["yield_line_ub"], "yield-line upper bound", "-.")):
        ax.axhline(val / 1e6, color="0.35", ls=ls, lw=1.0)
        ax.text(0.02, val / 1e6, f"  {lab}: {val/1e6:.2f} MPa", fontsize=7,
                va="bottom", color="0.25")
    ax.axhline(ref["py"] / 1e6, color="crimson", ls=":", lw=1.0)
    ax.text(0.02, ref["py"] / 1e6, f"  analytic first yield: {ref['py']/1e6:.2f} MPa",
            fontsize=7, va="bottom", color="crimson")
    ax.set_xlabel("centre deflection $w/t$ [-]")
    ax.set_ylabel("patch pressure [MPa]")
    ax.set_title("Patch-loaded panel: elastic-plastic paths vs rigid-plastic bounds")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "load_deflection_paths.png"), dpi=160)
    plt.close(fig)

    # ---------------- figure 2: mesh convergence ----------------
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    h = [r["nx"] for r in rows]
    ax.axhspan(ref["strip_lb"] / 1e6, ref["sine_ub"] / 1e6, color="0.85",
               label="admissible bracket [strip lb, sine ub]")
    ax.axhline(ref["yield_line_ub"] / 1e6, color="0.4", ls="--", lw=1.0,
               label="yield-line upper bound")
    ax.plot(h, [(r["collapse_knee"] or float("nan")) / 1e6 for r in rows], "o-",
            color="C3", label="FE plastic-mechanism knee (tangent $k/k_0<0.05$)")
    ax.plot(h, [(r["collapse_last_converged"] or float("nan")) / 1e6 for r in rows],
            "s--", color="C0", label="FE last converged load (load control)")
    ax.set_xlabel("element count per edge $n_x=n_y$")
    ax.set_ylabel("patch pressure [MPa]")
    ax.set_title("Mesh dependence of the measured collapse load")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "collapse_mesh_convergence.png"), dpi=160)
    plt.close(fig)

    # ---------------- figure 3: unload / permanent set ----------------
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for c, r in zip(colors, levels):
        seq = r["loading"] + r["unloading"]
        ax.plot([s["w_over_t"] for s in seq], [s["factor"] / 1e6 for s in seq],
                "o-", ms=2.6, lw=1.0, color=c, label=f"{r['nx']}x{r['nx']}")
    ax.set_xlabel("centre deflection $w/t$ [-]")
    ax.set_ylabel("patch pressure [MPa]")
    ax.set_title("Loading + unloading: residual (permanent) set")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "unload_permanent_set.png"), dpi=160)
    plt.close(fig)

    # ---------------- figure 4: elastic bias convergence ----------------
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    inv = [1.0 / r["nx"] for r in rows]
    axes[0].plot(inv, [abs(r["elastic_w_bias_pct"]) for r in rows], "o-",
                 label="centre deflection")
    axes[0].plot(inv, [abs(r["elastic_sigma_bias_pct"]) for r in rows], "s-",
                 label="surface von Mises")
    axes[0].set_xlabel("$1/n_x$")
    axes[0].set_ylabel("|FE - Navier| / Navier [%]")
    axes[0].set_title("Elastic field bias vs analytic Navier")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)
    axes[1].plot(inv, [r["first_yield_err_raw_pct"] for r in rows], "o-",
                 label="raw vs analytic $p_y$")
    axes[1].plot(inv, [r["first_yield_err_same_mesh_pct"] for r in rows], "s--",
                 label="same-mesh normalised")
    axes[1].axhspan(-5, 5, color="0.88", label="acceptance band $\\pm$5%")
    axes[1].set_xlabel("$1/n_x$")
    axes[1].set_ylabel("first-yield error [%]")
    axes[1].set_title("Elastic-plastic transition load")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "elastic_bias_and_transition.png"), dpi=160)
    plt.close(fig)

    print(json.dumps({"levels": [r["nx"] for r in rows],
                      "figures": sorted(os.listdir(FIG))}, indent=2))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())