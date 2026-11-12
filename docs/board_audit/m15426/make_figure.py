#!/usr/bin/env python3
"""m15426 审计图：独立复算 vs 被审算例（数据全部来自 verify_output.json，零手抄）。

(a) 本审计自写 1D 轴对称 FE（n=256）的 σ_r/σ_θ 剖面 vs Lamé 闭式解；
(b) 收敛曲线：被审 Q4 阶梯（12x4→96x32）与本审计 1D 元阶梯（n=32→256）的
    u_r / σ_r 最大相对误差 log-log，附二阶参考斜率。
"""
from __future__ import annotations

import json
import math
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
V = json.loads((HERE / "verify_output.json").read_text(encoding="utf-8"))

# 参数与闭式解（与审计脚本同源，此处为作图重算）
p = {"a": 0.5, "b": 0.8, "pi": 1e8, "E": 210e9, "nu": 0.3}
a, b, pi_, E, nu = p["a"], p["b"], p["pi"], p["E"], p["nu"]
A = pi_ * a**2 / (b**2 - a**2)
B = a**2 * b**2 * pi_ / (b**2 - a**2)
r = np.linspace(a, b, 400)
sr = (A - B / r**2) / 1e6
st = (A + B / r**2) / 1e6
sz = np.full_like(r, 2 * nu * A / 1e6)

prof = V["C_independent_fe"]["runs"][-1]["profile"]
pr = np.array([x["r"] for x in prof])
psr = np.array([x["sigma_r"] for x in prof]) / 1e6
pst = np.array([x["sigma_theta"] for x in prof]) / 1e6

fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.6))

ax[0].plot(r, sr, "k--", lw=2.0, label=r"Lamé closed form $\sigma_r$")
ax[0].plot(r, st, "k-.", lw=2.0, label=r"Lamé closed form $\sigma_\theta$")
ax[0].plot(r, sz, ":", color="0.55", lw=1.6, label=r"Lamé $2\nu A$ ($\sigma_z$)")
ax[0].plot(pr, psr, "o", ms=3.0, mfc="none", mec="tab:blue", label=r"audit 1D FE $n$=256, $\sigma_r$")
ax[0].plot(pr, pst, "s", ms=3.0, mfc="none", mec="tab:red", label=r"audit 1D FE $n$=256, $\sigma_\theta$")
ax[0].set_xlabel("radius r [m]")
ax[0].set_ylabel("stress [MPa]")
ax[0].set_title("(a) audit-side independent 1D axisymmetric FE vs Lame closed form")
ax[0].grid(alpha=0.3)
ax[0].legend(fontsize=8, loc="center right")
ax[0].annotate(f"max|Δσr|/|σr| = {max(x['sigma_r_err'] for x in prof):.2e}\n"
               f"max|Δσθ|/|σθ| = {max(x['sigma_theta_err'] for x in prof):.2e}",
               xy=(0.03, 0.05), xycoords="axes fraction", fontsize=8,
               bbox=dict(fc="white", ec="0.7"))

# (b) 收敛
lad = V["B_referee"]["convergence_ladder"]
h_q4 = [0.3 / int(x["mesh"].split("x")[1]) for x in lad]
u_q4 = [x["u_r_max_err"] for x in lad]
s_q4 = [x["sigma_r_max_err"] for x in lad]
fe = V["C_independent_fe"]["runs"]
h_1d = [0.3 / x["n_elem"] for x in fe]
u_1d = [x["u_r_max_rel_err"] for x in fe]
s_1d = [x["sigma_r_max_rel_err"] for x in fe]

ax[1].loglog(h_q4, u_q4, "o-", label="audited Q4: $u_r$ max")
ax[1].loglog(h_q4, s_q4, "o--", label=r"audited Q4: $\sigma_r$ max")
ax[1].loglog(h_1d, u_1d, "s-", label="audit 1D FE: $u_r$ max")
ax[1].loglog(h_1d, s_1d, "s--", label=r"audit 1D FE: $\sigma_r$ max")
h0, h1 = 2e-3, 1e-2
ax[1].loglog([h0, h1], [0.4 * (h1 / h0) ** 2 * (h0 / h1) ** 0 * 4e-2, 0.4 * 4e-2 * 4],
             color="0.6", lw=1.0, ls=":")
ax[1].annotate("slope 2 reference", xy=(4e-3, 1.2e-2), fontsize=8, color="0.45", rotation=-24)
ax[1].set_xlabel("element size h [m]")
ax[1].set_ylabel("max relative error [-]")
ax[1].set_title("(b) convergence ladder: audited Q4 vs audit-side 1D FE (both actually run)")
ax[1].grid(alpha=0.3, which="both")
ax[1].legend(fontsize=8)
ax[1].invert_xaxis()

fig.suptitle("CASE-LAME-CYLINDER board audit (m15426): independent recompute and convergence", fontsize=11)
fig.tight_layout()
out = HERE / "fig_lame_audit.png"
fig.savefig(out, dpi=140)
print("wrote", out, out.stat().st_size, "bytes")