#!/usr/bin/env python3
"""m17488 审计图：XVAL 收敛阶梯复算（数据源=本单实跑 reports/audit_xval_msfe.json，零手抄）。"""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
D = json.loads((HERE / "reports/audit_xval_msfe.json").read_text(encoding="utf-8"))
rows = D["E2_full_run"]["arm_rows"]
a = D["E_arm_a"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

# 左：臂 A(TensorFEM) HEX8/HEX20 的 M1/M2 阶梯 —— 度量泛函决定结论
h8 = sorted([(v["n"], v["M1_harness"], v["M2_harness"]) for k, v in a.items() if v["order"] == 1])
h20 = sorted([(v["n"], v["M1_harness"], v["M2_harness"]) for k, v in a.items() if v["order"] == 2])
ax1.plot([x[0] for x in h8], [x[1] * 100 for x in h8], "o-", label="HEX8  M1 (centroid-radius)")
ax1.plot([x[0] for x in h8], [x[2] * 100 for x in h8], "o--", label="HEX8  M2 (layer integral)")
ax1.plot([x[0] for x in h20], [x[1] * 100 for x in h20], "s-", label="HEX20 M1")
ax1.plot([x[0] for x in h20], [x[2] * 100 for x in h20], "s--", label="HEX20 M2")
ax1.axhline(3, color="k", lw=0.8, ls=":")
ax1.axhline(5, color="r", lw=0.8, ls=":")
ax1.text(11.6, 3.3, "record claim <3%", fontsize=7.5, ha="right")
ax1.text(11.6, 5.4, "case line <5%", fontsize=7.5, ha="right", color="r")
ax1.annotate("HEX20 n=4: M1=2.88% ok / M2=11.99% fail", xy=(4, 11.99), xytext=(4.4, 17),
             fontsize=7.5, arrowprops=dict(arrowstyle="->", lw=0.7))
ax1.set_xlabel("cube-sphere subdivision n (per face)")
ax1.set_ylabel("worst probe relative error [%]")
ax1.set_title("Arm A (TensorFEM): the verdict depends on the probe functional")
ax1.legend(fontsize=7.5)
ax1.grid(alpha=0.3)
ax1.set_xlim(3, 12.5)

# 右：dofs vs error，含独立实现臂 B（TET4，仅线性档可跑）
for lab, sel, style in (("HEX8 M1", [(v["dofs"], v["M1_harness"]) for k, v in a.items() if v["order"] == 1], "o-"),
                        ("HEX20 M1", [(v["dofs"], v["M1_harness"]) for k, v in a.items() if v["order"] == 2], "s-"),
                        ("HEX20 M2", [(v["dofs"], v["M2_harness"]) for k, v in a.items() if v["order"] == 2], "s--")):
    p = sorted(sel)
    ax2.plot([x[0] for x in p], [x[1] * 100 for x in p], style, label=lab)
bp = [(r["dofs"], r["M1"]) for r in rows if r["arm"] == "arm B"]
ax2.plot([x[0] for x in bp], [x[1] * 100 for x in bp], "^-", color="tab:green",
         label="skfem TET4 M1 (independent arm, linear only)")
ax2.axhline(5, color="r", lw=0.8, ls=":")
ax2.set_xscale("log")
ax2.set_xlabel("degrees of freedom")
ax2.set_ylabel("worst probe relative error [%]")
ax2.set_title("TET10 independent arm CRASHES → selection robustness undecided")
ax2.legend(fontsize=7.5)
ax2.grid(alpha=0.3, which="both")

fig.suptitle("m17488 audit · anchor XVAL-FEM-METHOD-SELECTION-FIRST-EVIDENCE (m17412, TensorFEM) · "
             "all numbers re-run in this audit", fontsize=9)
fig.tight_layout()
out = HERE / "fig_xval_ladder.png"
fig.savefig(out, dpi=140)
print("wrote", out, out.stat().st_size, "B")