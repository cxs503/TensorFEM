"""Cross-module S-N curve consistency audit -- procurement of the evidence bundle.

Writes ``artifacts/fatigue_sn_curve_consistency.json`` and
``docs/assets/fatigue_sn_curves.png``; prints the markdown tables used by
``docs/fatigue-sn-curve-consistency-audit.md``.  No number in that page is typed
by hand.

Run: ``PYTHONPATH=src python benchmarks/fatigue_sn_curve_consistency.py``
"""
from __future__ import annotations

import json
import math
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfem.fatigue.sn_curve_consistency import DNV_D_AIR, audit, curve_rows, reference_life
from tensorfem.structural_fatigue_detail import DNVDualSlopeSN, sn_curve_verification_report

OUT = pathlib.Path("artifacts/fatigue_sn_curve_consistency.json")
FIG = pathlib.Path("docs/assets/fatigue_sn_curves.png")

PROBES = (200.0e6, 100.0e6, 50.0e6, 30.0e6)


def main() -> int:
    report = audit()
    rows = report["rows"]
    summary = report["summary"]

    # ---- the historical (broken) m=2 protocol, reproduced for the errata table ----
    curve = DNVDualSlopeSN()
    knee_pa = curve.stress_knee_pa
    legacy = {
        "ds3_pa": 100.0e6,
        "ds4_pa": 50.0e6,
        "n3": curve.cycles_to_failure(100.0e6),
        "n4": curve.cycles_to_failure(50.0e6),
        "knee_stress_pa": knee_pa,
    }
    legacy["ratio"] = legacy["n4"] / legacy["n3"]
    legacy["ratio_expected"] = 32.0
    legacy["error_percent"] = abs(legacy["ratio"] - 32.0) / 32.0 * 100.0
    legacy["straddles_knee"] = bool(legacy["ds3_pa"] > knee_pa > legacy["ds4_pa"])

    fixed = sn_curve_verification_report(curve)

    # ---- the cross-module drift that was repaired (ice_slamming_whipping) ----
    from tensorfem.ice_slamming_whipping import SNCurve

    live = SNCurve()
    old_c1, old_c2 = 1.52e12, 4.34e15   # v1 declared defaults
    drift = {
        "old_c1": old_c1,
        "old_c2": old_c2,
        "old_log_a1": math.log10(old_c1),
        "old_knee_stress_mpa": (old_c1 / 1.0e7) ** (1.0 / 3.0),
        "new_c1": live.c1,
        "new_c2": live.c2,
        "new_log_a1": math.log10(live.c1),
        "new_knee_stress_mpa": live.knee_stress / 1e6,
        "published_log_a1": DNV_D_AIR["log_a1"],
    }
    drift["old_life_100mpa"] = old_c1 / 100.0 ** 3
    drift["new_life_100mpa"] = live.cycles_to_failure(100.0e6)
    drift["reference_life_100mpa"] = reference_life(100.0e6)
    drift["old_bias_percent"] = (drift["old_life_100mpa"] / drift["reference_life_100mpa"] - 1.0) * 100.0
    drift["new_bias_percent"] = (drift["new_life_100mpa"] / drift["reference_life_100mpa"] - 1.0) * 100.0
    drift["knee_stress_shift_percent"] = (
        drift["old_knee_stress_mpa"] / drift["new_knee_stress_mpa"] - 1.0) * 100.0

    bundle = {
        "reference": report["reference"],
        "iiw_reference": report["iiw_reference"],
        "summary": summary,
        "rows": rows,
        "legacy_protocol": legacy,
        "branch_guarded_report": fixed,
        "ice_slamming_drift": drift,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(bundle, indent=2, ensure_ascii=False))

    # ------------------------------- figure -------------------------------
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.8))
    stress = np.logspace(math.log10(20e6), math.log10(250e6), 60)
    ax.loglog(stress / 1e6, [reference_life(s) for s in stress], "k-", lw=2.4,
              label="DNV-RP-C203 D air (table)")
    markers = ["o", "s", "^", "v", "D", "P", "X", "*"]
    for index, row in enumerate(curve_rows()):
        ax.loglog(stress / 1e6, [row.life(s) for s in stress], markers[index % len(markers)],
                  ms=3.0, alpha=0.65, label=row.name)
    ax.plot([knee_pa / 1e6], [1e7], "k*", ms=14, label="knee (52.64 MPa, 1e7)")
    ax.set_xlabel("stress range Δσ [MPa]")
    ax.set_ylabel("cycles to failure N")
    ax.set_title("every curve-D implementation of the library on one axis")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=5.6, loc="lower left")

    labels = [r["name"].split(".")[-1] + "\n" + r["name"].split(".")[0][:14] for r in rows]
    x = np.arange(len(rows))
    width = 0.38
    ax2.bar(x - width / 2, [abs(r["life_100mpa_delta_vs_dnv_d"]) * 100 for r in rows],
            width, label="|ΔN| vs curve D at 100 MPa [%]")
    ax2.bar(x + width / 2, [abs(r["life_30mpa_delta_vs_dnv_d"]) * 100 for r in rows],
            width, label="|ΔN| vs curve D at 30 MPa [%]")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax2.set_ylabel("relative life deviation [%]")
    ax2.set_yscale("symlog", linthresh=1e-6)
    ax2.set_title("cross-module drift (0 = identical to the published curve)")
    ax2.grid(True, axis="y", alpha=0.25)
    ax2.legend(fontsize=7)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=160)

    # ------------------------------ tables -------------------------------
    print("\n### per-implementation audit\n")
    print("| implementation | standard | log a1 | m1 | log a2 | m2 | knee [MPa] | knee continuity | N(100 MPa) | ΔN vs D @100 MPa |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| `{r['name']}` | {r['standard']} | {r['log_a1']:.6f} | {r['m1']:.1f} | "
              f"{r['log_a2']:.6f} | {r['m2']:.1f} | {r['knee_stress_mpa']:.3f} | "
              f"{r['knee_continuity_rel']:.2e} | {r['life_100mpa']:.6e} | "
              f"{r['life_100mpa_delta_vs_dnv_d'] * 100:+.3f} % |")
    print("\n### summary")
    for key, value in summary.items():
        print(f"* {key} = {value}")
    print("\n### legacy m=5 protocol (v1, broken)")
    for key, value in legacy.items():
        print(f"* {key} = {value}")
    print("\n### branch-guarded protocol (this work)")
    for key in ("branch_guard", "ratio_m3_actual", "ratio_m5_actual", "knee_error_percent",
                "ratio_m5_error_percent", "passed"):
        print(f"* {key} = {fixed[key]}")
    print("\n### ice_slamming_whipping drift repair")
    for key, value in drift.items():
        print(f"* {key} = {value}")
    print(f"\nwrote {OUT} and {FIG}")
    return 0 if summary["passed"] and fixed["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
