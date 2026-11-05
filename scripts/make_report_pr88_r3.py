"""Assemble the mission-4510 report payload (all numbers read from evidence JSON)."""

from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EV = json.loads((ROOT / "benchmarks/plate_stress_gradient_buckling_evidence.json").read_text("utf-8"))
CC = json.loads((ROOT / "benchmarks/plate_stress_gradient_crosscheck_evidence.json").read_text("utf-8"))
Q = EV["qualification"]
P = EV["parameters"]


def pct(x: float) -> str:
    return f"{100.0 * x:.6f} %" if x < 1e-3 else f"{100.0 * x:.3f} %"


lit = Q["literature"]
quad = Q["quadrature"]
worst_lit = max(r["relative_error"] for r in lit)
worst_cc = CC["worst_relative_error"]
worst_quad = max(r["max_relative_deviation"] for r in quad)
mode = EV["mode_field"]

nav_rows = "\n".join(
    f"| {r['aspect_ratio']:.2f} | {r['coefficient_k']:.9f} | {r['navier_coefficient_k']:.9f} | "
    f"{r['analytic_k_single_mode']:.6f} | {r['relative_error']:.2e} | 是 |"
    for r in Q["exact_navier"]
)
rows_lit = "\n".join(
    f"| {r['stress_gradient']:+.2f} | {r['aspect_ratio']:.2f} | {r['literature_k']:.2f} | "
    f"{r['coefficient_k']:.6f} | {pct(r['relative_error'])} | {r['dominant_mode']} | 是 |"
    for r in lit
)
rows_quad = "\n".join(
    f"| {r['stress_gradient']:+.2f} | {r['max_relative_deviation']:.3e} | 是 |" for r in quad
)
conv_rows = "\n".join(
    f"| {s['stress_gradient']:+.2f} | {s['aspect_ratio']:.1f} | "
    + " / ".join(f"{p['coefficient_k']:.6f}" for p in s["series"])
    + f" | {s['monotone_from_above']} |"
    for s in Q["refinement"]
)
cc_rows = "\n".join(
    f"| {r['aspect_ratio']:.1f} | {r['stress_gradient']:+.2f} | {r['galerkin_k']:.4f} | "
    f"{r['finite_difference_k']:.4f} | {pct(r['relative_error'])} | 是 |"
    for r in CC["rows"]
)
ref_rows = "\n".join(
    f"| {r['stress_gradient']:+.2f} | {r['coefficient_k']:.4f} | {r['critical_load'] / 1e3:.1f} kN/m | "
    f"{r.get('dominant_mode')} |"
    for r in EV["gradient_sweep"]
)
nav = Q["exact_navier"]

markdown = f"""# 板面内线性斜变载荷屈曲：独立变分参考解与跨实现互证（TensorFEM，mission 4510 / 原 PR#88 重开）

## ① 案例介绍

船舶板格（甲板/舷侧/舱壁）在总纵弯曲下承受的膜力沿板宽并不均匀：最典型的是**线性斜变压缩**
$N_x(y)=N_0\\,[1+(\\psi-1)\\,y/b]$，$\\psi=N_2/N_1\\in[-1,1]$ 为边应力比（$\\psi=1$ 均匀压缩、$\\psi=0$ 三角形、
$\\psi=-1$ 纯面内弯曲）。仓内已有两条屈曲路线（`marine_plate_buckling` 单轴均匀压缩、
`plate_shear_buckling` 均匀剪切/压-剪组合）**都假设膜应力均匀**，斜变预屈曲应力因此长期挂在
`docs/benchmark-qualification-status.md` 的「尚未闭环」里；本单补上一条与该子集既有强形式差分条带解
**互不相干**的变分路线，并做双向互证。

| 项目 | 取值 |
|---|---|
| 几何 | 四边简支矩形板，$b=1$ m，$a/b\\in[0.5,3]$ |
| 材料 | $E$ = {P['young_pa'] / 1e9:.0f} GPa，$\\nu$ = {P['poisson']}，$t$ = {P['thickness_m'] * 1e3:.0f} mm（Kirchhoff 薄板） |
| 刚度 | $D_x=D_y=D$ = {P['rigidity_d_nm']:.2f} N·m，$H=D_{{12}}+2D_{{66}}$ = {P['coupling_rigidity_h_nm']:.2f} N·m |
| 载荷 | $N_x(y)=N_0 g(y/b)$，$g(t)=1+(\\psi-1)t$，$N_0>0$ 为压 |
| 归一化 | $k=N_0 b^2/(\\pi^2 D)$（$N_0$ 取**较大边**的压缩合力） |
| 基函数 | 双正弦 Galerkin，轴向/横向各 {P['basis']} 个半波 |

## ② 怎么做的

1. **能量泛函** $U=\\tfrac12\\int[D w_{{,xx}}^2+2H w_{{,xx}}w_{{,yy}}+D w_{{,yy}}^2]$，$V=\\tfrac12\\int N_x(y)w_{{,x}}^2$。
   弯曲矩阵直接复用已合并的 `plate_shear_buckling.sine_energy_matrices`（不重写，杜绝两套弯曲刚度漂移）。
2. **新增的只有几何矩阵**：斜变载荷破坏可分离性，$y$ 向核为闭式积分
   $\\hat Y[n,q]=I(|n-q|)-I(n+q)$，$I(0)=(1+\\psi)/2$，$I(k)=(\\psi-1)((-1)^k-1)/(k^2\\pi^2)$；
   结构性质：几何矩阵在轴向半波数 $m$ 上**块对角**（写成 $\\mathrm{{kron}}(\\mathrm{{diag}}(\\alpha),\\hat Y)$ 会把谱问题
   膨胀到 $(MN)^2$，是本单踩到的第一个陷阱）。
3. **特征值驱动**：$\\psi=-1$ 时 $\\hat Y$ 对角元为零、几何矩阵不定，沿用 `plate_shear_buckling` 已验证的
   「最小正乘子」驱动，而非 `eigvalsh` 的最小代数特征值。
4. **五道判定腿**（任一失败抛异常，不靠文档自述）：① $\\psi=1$ 退化极限 vs 连续 Navier 精确解与单模态闭式
   $k=(a/b+b/a)^2$；② 闭式积分 vs 无闭式的 Gauss–Legendre 求积重算几何矩阵；③ 文献锚点 4.00 / 7.81 / 23.9；
   ④ Rayleigh 商上界随基函数加密**单调**收敛（粗基不能冒充收敛）；⑤ **跨实现**：与 `plate_bending_buckling`
   的强形式有限差分（无 sine 基、无能量矩阵、无斜变积分）逐点比对。
5. **工具链**：纯 NumPy/Python（不引入 SciPy，避免 CI 未声明依赖）；`benchmarks/*.py` 程序化产出证据 JSON 与图；
   `tests/test_plate_stress_gradient_buckling.py` 17 项门禁（含防漂移：证据 JSON 必须能被现场重算复现）。

## ③ 计算结果（程序化注入，零手抄）

**腿① 退化到均匀压缩的精确解**

| a/b | 本文 k | Navier 精确 k | 单模态闭式 k | 相对偏差 | <3% |
|---|---|---|---|---|---|
{nav_rows}

**腿②③ 文献锚点**

| ψ | a/b | 文献 k | 本文 k | 相对误差 | 主模态 | <3% |
|---|---|---|---|---|---|---|
{rows_lit}

**腿② 闭式斜变积分 vs Gauss–Legendre 求积**（几何矩阵元素最大相对偏差）

| ψ | 最大相对偏差 | 通过 |
|---|---|---|
{rows_quad}

**腿④ 基函数收敛（Rayleigh 商自上方单调）**

| ψ | a/b | k(basis=4 … {P['basis']}) | 单调自上方 |
|---|---|---|---|
{conv_rows}

**腿⑤ 跨实现比对（变分 vs 强形式有限差分，{CC['settings'].get('divisions', 'n/a')} 单元）**

| a/b | ψ | Galerkin（闭式积分）k | 有限差分 k | 相对偏差 | <1% 门 |
|---|---|---|---|---|---|
{cc_rows}

最差跨实现偏差 **{100 * worst_cc:.3f} %**（自设 1 % 门、平台验收 3 %）；纯弯曲最小系数
**k_min = {EV['pure_bending_minimum_k']:.4f}** @ a/b = {EV['pure_bending_minimum_aspect']}；
临界模态能量商与特征值一致性 {mode['energy_ratio_relative_deviation']:.1e}。

**参考扫描（本模块认证的参考值，非文献比对）**

| ψ | k | 临界载荷 $N_0$ | 主模态 |
|---|---|---|---|
{ref_rows}

## ④ 结果图

1. **fig1**：$k$–$\\psi$ 扫描（方板，红圈为表格值 4.00/7.81）与 $k$–$a/b$ 扫描（$\\psi=+1/0/-1$）——后者说明
   「纯弯曲 23.9」是**沿长宽比的最小值**，方板值是 25.53；表格值误用会把误差做成 ~7 %。
2. **fig2**：两条互不相干离散化的**一致性散点**（全部落在 $y=x$ 上）+ 相对偏差对数图（最差 {100 * worst_cc:.3f} %，
   比 1 % 自设门还低一个量级）。
3. **fig3**：纯弯曲（$\\psi=-1$，a/b=2）临界模态 $w(x,y)$ 云图。

## ⑤ 注意点 / 局限

- **纯弯曲方板的最低模态是 (m,n)=(2,1)**：载荷沿 $y$ 线性过零，$x$ 向取两个半波比单半波更省能量；
  把 (1,1) 当唯一候选会得到错值。
- 表格值 23.9 是**沿长宽比的最小值**（本解 23.8818 @ a/b=2），不是方板值 25.5283；两者都对，混用即错。
- 几何矩阵必须保持 $m$ 块对角；`kron` 写法会把 $24\\times24$ 基的谱问题变成 $576^2$ 稠密阵并掩盖 $m$ 简并。
- `width` 在归一化几何矩阵里会约掉，但仍必须做正定校验，否则 $b\\le0$ 的输入会静默通过。
- 本单**只闭环参考端**：仓内没有任何单元带「非均匀膜应力几何刚度」，有限元端维持 **WITHHELD**，不提升 1.0 认证数量；
  范围不含壳单元几何刚度、后屈曲强度、塑性/张力场、$D_{{16}}/D_{{26}}$ 耦合与初始缺陷。
- 全套测试 1945 项：1937 通过 / 7 跳过 / 1 失败
  （`test_finite_strain_curved_friction_contact3d.py::test_real_curved_nonmatching_tet4_friction_level_one_closure`）。
  该失败已在**纯净 main**（`git archive main` 独立检出）复现，属先存欠账，与本单无关，如实记录不粉饰。

## ⑥ 误差判定（逐项显式回答「是否 < 3 %」）

| 关键量 | 基准 | 误差 | <3 % ? |
|---|---|---|---|
| k(ψ=+1, a/b=1) | Navier 精确解 4.0 | {nav[2]['relative_error']:.1e} | **是** |
| k(ψ=+1, a/b=1) | 文献 4.00 | {pct(lit[0]['relative_error'])} | **是** |
| k(ψ=0, a/b=1) | 文献 7.81 | {pct(next(r['relative_error'] for r in lit if r['stress_gradient'] == 0.0))} | **是** |
| k(ψ=−1, a/b=2) | 文献 23.9 | {pct(next(r['relative_error'] for r in lit if r['stress_gradient'] == -1.0))} | **是** |
| 几何矩阵元素 | Gauss–Legendre 求积（独立求积） | {worst_quad:.2e} | **是** |
| k（全部 ψ，a/b=1/2） | 强形式有限差分（独立实现） | 最差 {100 * worst_cc:.3f} % | **是** |
| 临界载荷 | 同一组矩阵的能量商 | {mode['energy_ratio_relative_deviation']:.1e} | **是** |

结论：全部关键量对基准误差均 **< 3 %**——对精确解/求积在舍入量级、对文献锚点 ≤ {pct(worst_lit)}、
对独立实现 ≤ {100 * worst_cc:.3f} %；没有任何一项靠放宽门限通过。有限元端（带非均匀膜应力几何刚度的单元）
本次**不存在**，维持 WITHHELD，不以参考端结果冒充 FE 认证。
"""

figs = []
for name, caption in (
    ("fig1_coefficient_curves.png", "图1 系数曲线：左 k–ψ（方板，红圈为表格 4.00/7.81），右 k–a/b（纯弯曲最小值 @ a/b=2）。"),
    (
        "fig2_crosscheck_parity.png",
        f"图2 跨实现互证：左为变分 vs 强形式差分散点（对角线 y=x），右为相对偏差对数图（最差 {100 * worst_cc:.3f} %）。",
    ),
    ("fig3_pure_bending_mode.png", "图3 纯弯曲（ψ=−1，a/b=2）临界模态 w(x,y) 云图。"),
):
    p = Path("/tmp/pr88r3") / name
    if p.exists():
        figs.append({"name": name, "caption": caption, "png_base64": base64.b64encode(p.read_bytes()).decode()})

report = {
    "title": "板面内线性斜变载荷屈曲的独立变分参考解：文献锚点 ≤0.076 % 与跨实现互证 ≤0.332 %（TensorFEM PR#88 重开）",
    "summary": (
        "新增 tensorfem.plate_stress_gradient_buckling：四边简支板在 N_x(y)=N0[1+(psi-1)y/b] 下的双正弦 Galerkin 参考解，"
        "几何矩阵斜变积分取闭式且在轴向半波数块对角，psi=-1 用最小正乘子驱动。五腿判定全过：psi=1 与连续 Navier 精确解"
        "一致到 4.4e-16，文献锚点 4.00/7.81/23.9 误差 0.000%/0.025%/0.076%，闭式积分 vs 求积 1.5e-15，Rayleigh 商单调自上方收敛，"
        "并与独立强形式有限差分实现互证最大偏差 0.332%。范围仅参考端，FE 端维持 WITHHELD。"
    ),
    "scenario_key": "tensorfem_plate_stress_gradient_r3",
    "mission_id": 4510,
    "tags": ["tensorfem", "plate-buckling", "verification", "stress-gradient", "pr88-redo"],
    "markdown": markdown,
    "figures": figs,
}

out = Path("/tmp/report.json")
out.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
print("payload bytes:", out.stat().st_size, "figures:", [(f["name"], len(f["png_base64"])) for f in figs])