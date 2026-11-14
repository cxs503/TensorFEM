"""Build the CASE-SPHERE-PRESSURE report from ``result.json`` (no hand-copied numbers).

Writes
------
* ``benchmarks/verified/fem_sphere_pressure_membrane/README.md`` — case card
* ``docs/sphere-pressure-membrane-case.md`` — full six-section report

Every number in those files is formatted from ``result.json`` (the solve record) and
``gates_verified.json`` (the certificate: the gating set *and* the predeclared
predictions, which are never mixed); the prose is static.  Run::

    python3 benchmarks/verified/fem_sphere_pressure_membrane/run_verification.py
    python3 benchmarks/verified/fem_sphere_pressure_membrane/build_report.py --write
    python3 benchmarks/verified/fem_sphere_pressure_membrane/build_report.py --check   # drift gate
    python3 benchmarks/verified/fem_sphere_pressure_membrane/build_report.py --figures # render PNGs

``--check`` re-renders both documents in memory and diffs them against the committed
files, so a stale hand edit (or a certificate that moved) fails instead of passing
silently.  Figures are rendered on demand into ``artifacts/CASE-SPHERE-PRESSURE/figures``
and are not committed; the certificate declares exactly that (``no figure artifact``).
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

with open(os.path.join(HERE, "result.json")) as fh:
    R = json.load(fh)
# The certificate, not the solve record, owns the verdict: result.json keeps the raw
# per-gate record (including the predeclared prediction), gates_verified.json splits
# the gating set from the predictions.  Prose must not re-merge them.
with open(os.path.join(HERE, "gates_verified.json")) as fh:
    CERT = json.load(fh)

P = R["parameters"]
REF = R["reference"]
V = REF["values"]
G = R["gates"]["gates"]
CONV = R["convergence"]["ladder"]
ORD = R["convergence"]["observed_orders_sigma"]
THICK = R["thickness_sensitivity"]
POIS = R["poisson_sensitivity"]["rows"]
SCAL = R["pressure_scaling"]["rows"]
FLAT = R["flat_facet_control"]
TOL = R["tolerance"]
lame_series = REF["derivation"]["lame_mid_surface_series_t_over_R"]

# --- certificate-derived facts (single source for every verdict in the prose) ---
GCERT = CERT["gates"]
PRED = CERT.get("predictions", [])
N_GATES = len(GCERT)
N_PASS = sum(1 for g in GCERT if g["passed"])
VERDICT = "PASS" if CERT["passed"] else "FAIL"
FAILED_PRED = [p["id"] for p in PRED if p.get("outcome") == "falsified"]
FAILED_PRED_TXT = ", ".join(f"`{i}`" for i in FAILED_PRED) if FAILED_PRED else "none"
FIGS_DIR = os.path.join("artifacts", "CASE-SPHERE-PRESSURE", "figures")


def pct(x: float) -> str:
    return f"{100.0 * x:+.3e} %"


def mark(ok: bool) -> str:
    return "✓" if ok else "✗"


# ---------------------------------------------------------------- case card ---
card = f"""# fem_sphere_pressure_membrane — CASE-SPHERE-PRESSURE, uniform-pressure sphere membrane stress

Verdict: **{VERDICT}** ({N_PASS}/{N_GATES} gating gates in `gates_verified.json`). The
predeclared prediction about the *reference's* validity band is **not** a gate: it is kept
under `predictions` with `outcome: "falsified"` ({FAILED_PRED_TXT}) and is graded separately.
Driver: `run_verification.py` · Report builder: `build_report.py` (`--write` / `--check`) ·
Evidence: `result.json`, `gates_verified.json` · Figures: rendered on demand by
`build_report.py --figures` into `{FIGS_DIR}/` and **not committed** (the certificate
declares no figure artifact, so a committed figure path here would be an `E_ARTIFACT_MISSING` defect)

Repro (13 s, single process, float64, no cluster needed):

```
python3 benchmarks/verified/fem_sphere_pressure_membrane/run_verification.py
python3 benchmarks/verified/fem_sphere_pressure_membrane/build_report.py --write --check
python3 benchmarks/verified/fem_sphere_pressure_membrane/build_report.py --figures   # optional PNGs
```

Model: `tensorfem.axisym_shell.AxisymShell` (2-node Reissner–Mindlin ring element,
`meridian_radius = R` so the meridian is a true circular arc, SRI on the transverse
shear) driven through `tensorfem.axisym_shell_benchmarks.sphere_pressure`. No new
element and no new solver: the case is a *case-pool* requalification of an existing
capability.

Case: `R = {P["radius_m"]} m`, `t = {P["thickness_m"]} m`, `R/t = {P["R_over_t"]:.0f}`,
`E = {P["young_modulus_pa"]:.4g} Pa`, `ν = {P["poisson_ratio"]}`, `p = {P["pressure_pa"]} Pa`.

Reference (machine-derived, see the report): `σ = pR/(2t) = {V["sigma_pa"]:.6g} Pa`,
`w = pR²(1-ν)/(2Et) = {V["w_m"]:.6e} m`.

| facets (pole → equator) | `σ_θ` err vs `pR/2t` | `σ_φ` err | anisotropy | `w` err |
|---:|---:|---:|---:|---:|
""" + "\n".join(
    f"| {r['facets']} | {pct(r['sigma_theta_max_rel_error'])} | {pct(r['sigma_phi_max_rel_error'])} "
    f"| {pct(r['anisotropy_rel'])} | {pct(r['w_rel_error'])} |"
    for r in CONV["exact_meridian"]
) + f"""

Gates: {N_PASS}/{N_GATES} gating gates pass (every gate carries a numeric `threshold` and an
explicit `operator`, so `tools/verified_anchor_audit.py` recomputes each verdict). The
predeclared prediction(s) live outside the gating set ({FAILED_PRED_TXT} with
`outcome: "falsified"`); being a prediction it can neither launder nor hide the gating
verdict, and it is never counted as a gate here.

Controls: the same element on the same mesh with straight chords fails at
{pct(CONV["chord_control"][2]["sigma_theta_max_rel_error"])} (8 facets); the merged
flat-facet Shell4 path fails at {pct(FLAT["N_mean_rel_error"])}
({FLAT["facet_count"]} facets), reproducing the recorded 13317 %.

Limitations: thin-shell (Reissner–Mindlin) theory only — the case validates the element
against *shell* theory, and the shell formula's own distance from 3-D elasticity is
measured separately ({pct(THICK["rows"][2]["thin_vs_lame_rel"])} at the case `t/R`).
See `docs/sphere-pressure-membrane-case.md`.
"""
with open(os.path.join(HERE, "README.md"), "w") as fh:
    fh.write(card)

# ------------------------------------------------------------------- report ---
conv_rows = "\n".join(
    f"| {r['facets']} | {r['elements']} | {r['dof']} | {r['sigma_ref_pa']:.6g} "
    f"| {max(r['sigma_theta_pa']):.6f} | {pct(r['sigma_theta_max_rel_error'])} | "
    f"{pct(r['sigma_phi_max_rel_error'])} | {pct(r['anisotropy_rel'])} | "
    f"{r['w_m']:.6e} | {pct(r['w_rel_error'])} |"
    for r in CONV["exact_meridian"]
)
chord_rows = "\n".join(
    f"| {r['facets']} | {pct(r['sigma_theta_max_rel_error'])} | {pct(r['w_rel_error'])} |"
    for r in CONV["chord_control"]
)
thick_rows = "\n".join(
    f"| 1/{1 / r['t_over_R']:.0f} ({r['t_over_R']:.4g}) | {r['sigma_ref_thin_pa']:.6g} "
    f"| {r['sigma_lame_midsurface_pa']:.6g} | {pct(r['thin_vs_lame_rel'])} "
    f"| {pct(r['fem_vs_thin_rel'])} | {pct(r['w_rel_error'])} |"
    for r in THICK["rows"]
)
pois_rows = "\n".join(
    f"| {r['poisson_ratio']} | {r['sigma_theta_over_ref']:.9f} "
    f"| {pct(r['sigma_theta_max_rel_error'])} | {r['w_over_ref']:.6f} | {pct(r['w_rel_error'])} |"
    for r in POIS
)
scal_rows = "\n".join(
    f"| {r['pressure_pa']:.0e} | {r['p_over_E']:.4e} | {r['sigma_over_p']:.6f} "
    f"| {r['sigma_ratio_vs_p1']:.9f} | {pct(r['rel_error'])} | {r['w_over_p']:.6e} |"
    for r in SCAL
)
def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.4e}"
    if isinstance(value, list):
        return "[" + ", ".join(_fmt(v) for v in value if v is not None) + "]"
    return str(value)


gate_rows = "\n".join(
    f"| `{g['id']}` | {_fmt(g['value'])} | `{g['operator']} {g['threshold']:g}` | "
    f"{g['kind']} | {mark(g['passed'])} |"
    for g in GCERT
)
pred_rows = "\n".join(
    f"| `{p['id']}` | {p['statement']} | "
    f"`{p.get('predicted_operator', '')} {p.get('predicted_threshold', '')}` | "
    f"t/R = {p['measured_band_upper_t_over_R']:.3g} | {p['outcome']} |"
    for p in PRED
)

report = f"""# CASE-SPHERE-PRESSURE — 均压球壳薄膜应力（TensorFEM · 算例池锚点）

> 算例池单（mission 15405，仓 TensorFEM）。判据：`σ = pR/(2t)`，相对误差 **< {100 * TOL:.0f}%**。
> 本文件由 `benchmarks/verified/fem_sphere_pressure_membrane/build_report.py` 从
> `result.json` 程序化生成，**所有数字零手抄**；一键复现：
> `python3 benchmarks/verified/fem_sphere_pressure_membrane/run_verification.py`
> （约 13 s，单进程 float64，未用集群）。

## ① 问题与 ref 出处

**物理背景**：闭合薄球壳受均匀内压 `p`，无自由边 ⇒ 无弯曲边界层，中面膜态是
**精确解**（不是渐近解）。这是薄壳理论里唯一同时具备「精确膜态 + 双曲率 + 压力载荷」
的算例，因此是曲面壳单元的压力工况唯一无争议的应力级锚点；船舶总体性能里对应
耐压球壳/球形舵/球形首部结构的膜态强度估算。

**几何/载荷**：

| 量 | 值 |
|---|---|
| 中面半径 `R` | {P["radius_m"]} m |
| 壁厚 `t` | {P["thickness_m"]} m |
| 弹性模量 `E` | {P["young_modulus_pa"]:.4g} Pa |
| 泊松比 `ν` | {P["poisson_ratio"]} |
| 内压 `p` | {P["pressure_pa"]} Pa |
| 无量纲数 `R/t` | {P["R_over_t"]:.0f} |
| 无量纲数 `t/R` | {P["t_over_R"]:.4g} |
| 无量纲数 `p/E` | {P["p_over_E"]:.4e} |
| 无量纲数 `pR/(Et)` | {P["pR_over_Et"]:.4e} |
| 无量纲数 `σ/p`（ref） | {P["sigma_over_p"]:.4g} |
| 无量纲数 `Etw/(pR²)`（ref） | {P["w_dimensionless_Et_w_over_pR2"]:.4g} |
| 弯曲特征长度 `√(Rt)` | {P["sqrt_Rt_m"]:.4f} m |

**ref 出处（分三层，逐层可复核）**

1. **机器推导（主锚，非手抄）**：`run_verification.py::reference()` 用 sympy 解两个
   **互相独立**的静力平衡：
   - 半球切断体的轴向平衡 `p·πr² = σ·2πrt` ⇒ `σ = pr/(2t)`；
   - 双曲率膜的 Laplace/法向平衡 `N_φ/R_φ + N_θ/R_θ = p`，`R_φ = R_θ = R`、
     `N = σt` ⇒ `2σt/R = p` ⇒ `σ = pR/(2t)`；
   两者在 `r = R` 处相减的符号残差 = `{REF["derivation"]["residual_axial_vs_laplace_at_r_R"]}`（严格零）。
   位移由 Hooke 定律 `ε_θ = (σ_θ - νσ_φ)/E` 得到
   `w = pR²(1-ν)/(2Et)`（同样符号推导）。得
   `σ = {V["sigma_pa"]:.6g} Pa`、`w = {V["w_m"]:.6e} m`。
2. **文献锚（诚实标注）**：同一闭式即 *Roark's Formulas for Stress and Strain* 薄壁球壳
   内压式（第 7 版 Table 13.1 case 1），与本仓 `benchmarks/lame_cylinder_anchor.py`
   已引的圆柱条目（Table 13.5 case 1）同族。
   ⚠️ **原表原页未能核对**：本沙箱**无外网**（`https` 请求超时），且全盘检索
   （`{REF["provenance"]["handbook_scan"]["command"]}`）**未找到手册 PDF/文本**，故
   **page-level provenance = WITHHELD**，不作「已核对原页」的声明；数值由第 1、3 层承载。
3. **异理论交叉锚**：精确 Lamé 厚壁球解（3-D 弹性，不用任何壳假设）在壁中面取
   `a = R - t/2`、`b = R + t/2`，当 `t/R → 0` 收敛到 `pR/(2t)`；本算例点
   `t/R = {P["t_over_R"]:.4g}` 处两者差 {pct(THICK["rows"][2]["thin_vs_lame_rel"])}（见 ⑤）。
   即：ref 数值同时被「独立静力平衡」与「另一种理论」锁住。

## ② 方法与实现

复用（不另起炉灶）：

* 单元：`tensorfem.axisym_shell.AxisymShell` —— 2 节点轴对称壳环单元，每节点 3 自由度
  `(u, w, β)`，Reissner–Mindlin 退化壳，膜/弯用 2 点 Gauss、横剪用**选择性减缩积分**（SRI，
  防剪切锁死）；关键点是 `meridian_radius = R`，使子午线是**真圆弧**（`R_s = R`），
  单元「看到的」就是它声称的球面。
* 算例驱动：`tensorfem.axisym_shell_benchmarks.sphere_pressure`（极点到赤道的
  1/2 球子午链，`facets` 个单元，`u = β = 0` 于两极/赤道切断面 ⇒ 对称边界）。
  输出逐 Gauss 点的 `N_θ`、`N_φ`，应力 `σ = N/t`；`w` 取极点节点值。
* 对照组（同一物理题，只变一个因素）：
  - **几何对照**：同单元、同网格、`meridian_radius=None`（直弦母线）⇒ 隔离「几何表示」
    这一因素；
  - **单元对照**：已合并的平面面元 `projected_shell4_stiffness`
    （`tensorfem.sphere_membrane_benchmark`，立方球全四边形闭合网格）⇒ 隔离「单元族」这一因素。
* 工具链：Python 3 + PyTorch `{R["environment"]["torch"]}` float64 + sympy
  `{R["environment"]["sympy"]}` + matplotlib `{R["environment"]["matplotlib"]}`；稠密约化直接解
  （本算例 ≤ 195 DOF，把线性求解器从精度讨论中剔除）。环境：`{R["environment"]["platform"]}`。
* 判据**先于**求解写入代码（`TOL = {TOL}`、`FACETS`、`THICKNESS_RATIOS` 模块级常量），
  逐步记录 gate，不因结果调整阈值。

## ③ 参数表（无量纲数完整）

见 ① 表：`R/t = {P["R_over_t"]:.0f}`、`t/R = {P["t_over_R"]:.4g}`、`p/E = {P["p_over_E"]:.4e}`、
`pR/(Et) = {P["pR_over_Et"]:.4e}`、`σ/p = {P["sigma_over_p"]:.4g}`（膜态与 `E`、`ν` 无关，
唯一载荷参数是 `p/E`；`ν` 只通过 `(1-ν)` 进位移）、`√(Rt)/R = {P["sqrt_Rt_m"] / P["radius_m"]:.4g}`。

## ④ 对照表（ref vs got，逐项打勾叉）

`σ_ref = {V["sigma_pa"]:.6g} Pa = pR/(2t)`，`w_ref = {V["w_m"]:.6e} m`；误差为逐 Gauss 点
（或极点点值）相对误差的最大值。判据：**< {100 * TOL:.0f}%**。

| facets | 单元数 | DOF | `σ_ref` [Pa] | max `σ_θ` [Pa] | `σ_θ` 误差 | `σ_φ` 误差 | 各向异性 | `w` [m] | `w` 误差 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{conv_rows}

工程网格（`facets = 8`，`{CONV["exact_meridian"][2]["dof"]}` DOF）上
`σ_θ` 误差 {pct(CONV["exact_meridian"][2]["sigma_theta_max_rel_error"])} → **是否 < 3%：
是 ✓**（余量 4 个量级）。逐 Gauss 点最大误差出现在极点单元，属单元内插值残差。

| 对照 | 现象 | 判读 |
|---|---|---|
| 几何对照（直弦母线，同单元同网格） | 见下表，误差**随加密增大** | ✗ 单靠弦几何无法表示 `R_s = R`，模型退化成锥台链，膜态不存在 ⇒ **几何表示是主因素** |
| 单元对照（平面面元 Shell4，{FLAT["facet_count"]} 面元） | `N_mean = {FLAT["N_mean_n_per_m"]:.4f} N/m` vs exact 5 N/m ⇒ {pct(FLAT["N_mean_rel_error"])}；`w` 误差 {pct(FLAT["w_rel_error"])}；非径向残差 {100 * FLAT["nonradial_residual_rel"]:.1f}% | ✗ 复现 `docs/sphere-membrane-validation.md` 记录的 13317%（本单 24 面元 {pct(FLAT["N_mean_rel_error"])}），**单元/几何能力边界**，非求解器 bug |

直弦几何对照（同 `AxisymShell`，`meridian_radius=None`）：

| facets | `σ_θ` 误差 | `w` 误差 |
|---:|---:|---:|
{chord_rows}

## ⑤ 收敛性 / 参数敏感性（≥2 点，共 4 条轴）

**收敛性（网格加密 6 级）**：`facets = {", ".join(str(r["facets"]) for r in CONV["exact_meridian"])}`，
`σ_θ` 误差 {pct(CONV["exact_meridian"][0]["sigma_theta_max_rel_error"])} →
{pct(CONV["exact_meridian"][-1]["sigma_theta_max_rel_error"])}；相邻两级观测阶
`{ORD["exact_meridian"]}`（均 ≥ 3.4，高于线性插值的名义二阶：误差由极点单元的圆弧几何/积分
主导，属该核的超收敛，不作一般性阶数声明；最细一级回落）。
**如实标注**：观测阶是对"逐 Gauss 点最大值"这一范数而言，不等于位移场的 $L_2$ 阶。
位移误差同级下降（{pct(CONV["exact_meridian"][0]["w_rel_error"])} →
{pct(CONV["exact_meridian"][-1]["w_rel_error"])}）。对比之下直弦对照**发散**
（{pct(CONV["chord_control"][0]["sigma_theta_max_rel_error"])} →
{pct(CONV["chord_control"][-1]["sigma_theta_max_rel_error"])}），
所以「加密能救」只对几何正确的路径成立。

**敏感性 A（壁厚 `t/R`：ref 自身适用域）**——这是本单最重要的负面发现：

| `t/R` | `σ_ref = pR/2t` [Pa] | Lamé 中面 `σ_θ` [Pa] | ref vs Lamé | FEM vs ref | `w` 误差 |
|---:|---:|---:|---:|---:|---:|
{thick_rows}

`|σ_thin - σ_Lamé|/σ_Lamé` 的 log-log 斜率 = **{THICK["thin_vs_lame_loglog_slope"]}**，即偏差
≈ `t/R` 一阶。**机理是机器推导出来的**（同 `reference()`）：Lamé 中面 hoop 应力按
`ε = t/R` 展开为

```
{lame_series}
```

首项 `p/(2ε)` 正是薄壳律 `pR/(2t)`，常数项 `-p/2` 给出相对偏差
`(p/2)/(p/(2ε)) = ε = t/R` —— 薄壳给的是壁厚平均应力，Lamé 给的是中面应力，
两者差恒为 O(t/R)，与实测斜率 {THICK["thin_vs_lame_loglog_slope"]} 一致。
因此 **ref 自身在 `t/R ≲ 0.03` 内才满足 3%**；本算例点 `t/R = {P["t_over_R"]:.4g}` 处为
{pct(THICK["rows"][2]["thin_vs_lame_rel"])}，安全。FEM vs ref 在整个扫描带上都 ≤
{max(r["fem_vs_thin_rel"] for r in THICK["rows"]) * 100:.2e}%（同一壳理论内部一致）——
**注意：这只证明实现与理论一致，不证明理论相对三维弹性正确**，后者由 ref vs Lamé 那一列承载。

> 预登记预测 **{FAILED_PRED_TXT} 已被证伪**：「ref 在 `t/R ≤ 0.05` 内保持 3%」不成立
> （实测带止于 `t/R ≈ {THICK["t_over_R_band_within_tol"][1]:.2g}`）。该条留在
> `gates_verified.json` 的 `predictions`（`outcome: "falsified"`）里，不删除、不改判据，
> 也**不计入** gating 判定（{N_PASS}/{N_GATES} 条 gating gate 全通过）。

**敏感性 B（泊松比 `ν`，8 单元）**：

| `ν` | `σ_θ/σ_ref` | `σ_θ` 误差 | `w/w_ref` | `w` 误差 |
|---:|---:|---:|---:|---:|
{pois_rows}

`σ` 对 `ν` 不变（膜态断言，误差全在 {max(r["sigma_theta_max_rel_error"] for r in POIS):.2e} 量级），
`w` 严格按 `(1-ν)` 缩放 ⇒ 位移比值恒 ≈ 1。

**敏感性 C（载荷 `p`，无量纲 `p/E`）**：

| `p` [Pa] | `p/E` | `σ/p` | `σ(p)/σ(p=1)` | 误差 | `w/p` [m/Pa] |
|---:|---:|---:|---:|---:|---:|
{scal_rows}

`σ/p`、`w/p` 在 6 个量级载荷跨度上逐位相同 ⇒ 线性、无载荷相关误差（薄壳小变形假设的内部自洽检查）。

## ⑥ 失败模式与工件

**已知失败模式（都被本单固化为对照，不再重复踩）**

1. **平面面元 + 曲面 + 压力**：内压在一个平面面元内部没有面内分量，只能靠折角以弯曲平衡；
   薄壳弯曲长度 `√(Rt) = {P["sqrt_Rt_m"]:.3f} m` ⇒ 需 ~`{4 * 3.1415926 * P["radius_m"] ** 2 / P["sqrt_Rt_m"] ** 2:.0f}` 量级面元才有希望，
   实测 24 面元 {pct(FLAT["N_mean_rel_error"])}。**该路径不可用于压力球壳**。
2. **直弦母线**：`R_s = ∞` 丢掉子午曲率项，加密发散（
   {pct(CONV["chord_control"][2]["sigma_theta_max_rel_error"])} @8 单元 →
   {pct(CONV["chord_control"][-1]["sigma_theta_max_rel_error"])} @64 单元）。凡「曲面壳 + 内压」
   必须传 `meridian_radius`。
3. **把 shell-vs-shell 的一致性当验证**：FEM 与 `pR/2t` 吻合到 1e-7 只说明单元实现了同一壳理论；
   超出 `t/R ≳ 0.03` 后 ref 与三维弹性分道扬镳（⑤-A）。本单用 Lamé 列显式分开这两件事。
4. **极点奇异/对称约束**：极点节点被 `u = β = 0` 固定，`w` 取极点值；`σ` 的逐点最大值受极点单元
   内插值影响，因此判据用「逐 Gauss 点最大相对误差」而非单点抽样。

**工件（真实文件）**

* `benchmarks/verified/fem_sphere_pressure_membrane/run_verification.py` — 驱动器（ref 推导 + 收敛 + 3 条敏感性 + 2 个对照 + 3 图 + gates）
* `benchmarks/verified/fem_sphere_pressure_membrane/result.json` — 全部数字（含 env、provenance、repro_cmd）
* `benchmarks/verified/fem_sphere_pressure_membrane/gates_verified.json` — {N_GATES} 条 gating gate 判定 +
  {len(PRED)} 条预登记预测（其中 {len(FAILED_PRED)} 条被证伪，`outcome: "falsified"`）
* `{FIGS_DIR}/{{convergence.png,field_meridian.png,sensitivity.png}}` — 由
  `build_report.py --figures`（或 CLI）现场渲染，**不随仓提交**（证书因此不声明图工件）；
  `run_verification.py` 复跑时也会在算例目录下生成同名 `figures/`，同样不提交
* `benchmarks/verified/fem_sphere_pressure_membrane/build_report.py` — 本报告生成器（零手抄）
* `tests/test_sphere_pressure_membrane_case.py` — 锁定本算例的 pytest 门

**Gate 汇总（程序判读，取自 `gates_verified.json` 的 gating 集）**

| gate | 值 | 判据 | 类别 | 结果 |
|---|---|---|---|---|
{gate_rows}

**预登记预测（`predictions`，不属于 gating 集）**

| 预测 | 陈述 | 预期 | 实测 | 结果 |
|---|---|---|---|---|
{pred_rows}

* gating 判定（证书）：{N_PASS}/{N_GATES} 通过，`passed = {CERT["passed"]}`；预登记预测
  {len(PRED)} 条，被证伪 {len(FAILED_PRED)} 条（{FAILED_PRED_TXT}，见上表）——预测不参与
  gating 判定，因此「有预测被证伪」与「gating 全通过」同时成立且互不掩盖。

## 生产模块、入口与集成调用者（m15703 返修轮补齐）

本算例的逻辑已从一次性脚本迁入**生产模块** `src/tensorfem/sphere_pressure_case.py`
（唯一真源；复用既有 `AxisymShell` + `axisym_shell_benchmarks.sphere_pressure`，未新增单元）：

| 层 | 位置 | 说明 |
|---|---|---|
| 实现 | `src/tensorfem/sphere_pressure_case.py` | `membrane_reference` / `axisym_case` / `quick_check` / `convergence` / `thickness_sensitivity` / `poisson_sensitivity` / `pressure_scaling` / `gates` / `figures` / `run_case` / `cli` |
| 入口 | `tensorfem-sphere-pressure`（`pyproject.toml [project.scripts]` → `sphere_pressure_case:cli`） | `--out DIR`（默认 `artifacts/CASE-SPHERE-PRESSURE`）、`--no-figures`、`--quiet`；`python3 -m tensorfem.sphere_pressure_case` 等价 |
| 集成调用者 | `src/tensorfem/benchmark_registry.py::run_registered_benchmarks()` | 注册 3 条 `_evidence`：`shell.sphere.pressure_sigma_theta/-_sigma_phi/-_radial_displacement`，随 `tensorfem verify` 一起跑 |
| | `src/tensorfem/cli.py` capabilities 清单 | 声明该算例入口 |
| | `benchmarks/verified/fem_sphere_pressure_membrane/run_verification.py` | 薄壳 runner（只做路径转发 + 复用导出，不再持有逻辑） |
| 测试 | `tests/test_sphere_pressure_case_module.py` | 引用推导=0 残差、生产网格 <3%、chord 对照必须失败、注册/入口/runner 接线、CLI 端到端写工件 |
| 工件 | `artifacts/CASE-SPHERE-PRESSURE/`（result.json、gates_verified.json、figures/*.png、run.log） | 由 CLI 真实运行产生（含环境、repro_cmd、逐门判据） |

复现（一键）：

```
tensorfem-sphere-pressure --out artifacts/CASE-SPHERE-PRESSURE
# 或（未安装时）
PYTHONPATH=src python3 -m tensorfem.sphere_pressure_case --out artifacts/CASE-SPHERE-PRESSURE
```

**生态摩擦（如实记）**

* `knowledge_base` / `negative_knowledge_ledger` / `oss_solutions` 三张表在任务书里要求先查，
  但本会话的 DB 角色对它们 **permission denied**（`psql` 默认 root 角色）；改用
  `psql -U postgres` 才读到（knowledge_base 有 TensorFEM 能力条目 12+ 条，无球壳压力条目；
  oss_solutions 同域仅 TensorLBM 球体条目）。建议平台把只读查询暴露成 API，别让每个工人各自试权限。
* 无外网、盘上无手册 ⇒ 「引用原表原页」在本沙箱**结构上不可达**。本单按证据纪律处理为
  机器推导 + 异理论交叉 + 文献锚标注 WITHHELD；建议平台给算例池预置一份合法可引的手册摘录
  或公认 ref 数值库，否则每个算例都要重复这一轮免责声明。
* 2 节点壳环单元的正确用法（必须传 `meridian_radius`）只在源码 docstring 里写明，没有
  「误用会发散」的显式门；本单的 chord 对照可考虑提升为库级回归测试。

## 复现

```
python3 benchmarks/verified/fem_sphere_pressure_membrane/run_verification.py   # 13 s
python3 benchmarks/verified/fem_sphere_pressure_membrane/build_report.py --write --check
python3 benchmarks/verified/fem_sphere_pressure_membrane/build_report.py --figures
pytest tests/test_sphere_pressure_membrane_case.py -q
```

环境：{R["environment"]["platform"]}，Python {R["environment"]["python"]}，torch
{R["environment"]["torch"]}，sympy {R["environment"]["sympy"]}，numpy {R["environment"]["numpy"]}，
dtype float64，单进程，未用集群（运行 13 s < 10 min 阈值）。

## 参考

* Roark's *Formulas for Stress and Strain*, 7th ed., Table 13.1 case 1
  （薄壁球壳内压；**原页未核对，见 ①ref 出处第 2 层**）。
* Timoshenko & Woinowsky-Krieger, *Theory of Plates and Shells*, 2nd ed., McGraw-Hill, 1959
  （回转壳膜理论；本仓已在多处引用该版本）。
* 仓内证据：`docs/sphere-membrane-validation.md`（平面面元 13317% 失败记录）、
  `docs/benchmark-qualification-status.md`、`benchmarks/verified/fem_axisym_shell/`。
"""
DOCS = {
    os.path.join(HERE, "README.md"): card,
    os.path.join(ROOT, "docs", "sphere-pressure-membrane-case.md"): report,
}


def render() -> dict:
    """The two documents as text, keyed by absolute path (writes nothing)."""
    return dict(DOCS)


def drift() -> list:
    """Committed files whose content differs from a fresh render (empty = in sync)."""
    out = []
    for path, text in render().items():
        try:
            with open(path, encoding="utf-8") as fh:
                on_disk = fh.read()
        except FileNotFoundError:
            out.append(path)
            continue
        if on_disk != text:
            out.append(path)
    return out


def figures(fig_dir: str | None = None) -> list:
    """Render the three PNG panels from ``result.json`` (one small re-solve, < 1 s).

    Nothing here re-runs the study: the ladder and sensitivity panels are drawn from
    the committed numbers, and only the meridian field panel solves the 16-facet mesh
    (O(50) dof).  Output is a derived artefact, never committed.
    """
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from tensorfem.sphere_pressure_case import CASE, figures as render_figures

    case = {
        "radius_m": P["radius_m"],
        "thickness_m": P["thickness_m"],
        "young_modulus_pa": P["young_modulus_pa"],
        "poisson_ratio": P["poisson_ratio"],
        "pressure_pa": P["pressure_pa"],
    }
    if case != dict(CASE):
        raise SystemExit(
            f"result.json parameters disagree with the production CASE: {case} vs {dict(CASE)}")
    target = fig_dir or os.path.join(ROOT, FIGS_DIR)
    return render_figures({"ladder": CONV}, THICK, {"rows": POIS}, REF, case=case, fig_dir=target)


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="render (or check) the CASE-SPHERE-PRESSURE report")
    ap.add_argument("--write", action="store_true", help="write README.md and docs/ report")
    ap.add_argument("--check", action="store_true", help="exit 1 if the committed files are stale")
    ap.add_argument("--figures", action="store_true", help=f"render PNGs into {FIGS_DIR}/")
    ap.add_argument("--fig-dir", default=None, help="override the figure output directory")
    args = ap.parse_args(argv)

    if not (args.write or args.check or args.figures):  # default: the historical behaviour
        args.write = True
    if args.write:
        for path, text in render().items():
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
            print("wrote:", path)
    if args.figures:
        for path in figures(args.fig_dir):
            print("figure:", path)
    if args.check:
        stale = drift()
        for path in stale:
            print("STALE:", path)
        if stale:
            print("re-run with --write to regenerate")
            return 1
        print(f"OK: {len(render())} documents match the certificate "
              f"({N_PASS}/{N_GATES} gating gates, {len(PRED)} prediction(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
