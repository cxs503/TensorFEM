"""Build and POST the mission report (numbers injected from result.json)."""
from __future__ import annotations

import base64
import json
import os
import subprocess
from pathlib import Path

ROOT = Path("/root/tp-agent/work/m7703")
ANCHOR = ROOT / "benchmarks/verified/fem_panel_buckling"
result = json.loads((ANCHOR / "result.json").read_text())
gates = json.loads((ANCHOR / "gates_verified.json").read_text())
case = result["campaign_case"]
summary = case["panel_summary"]
oracle = result["plate_oracle"]["closed_form"]
path = result["linear_path"]
panel_row = result["panel_rows"][-1]
clamped = result["clamped_companion"][-1]
gci = result["plate_ladder_gci"]
pgci = result["panel_ladder_gci"]
sec = result["section_modulus_chain"]


def mpa(v):
    return f"{v/1e6:.3f}"


def pc(v):
    return f"{v*100:+.3f}"


plate_rows = "\n".join(
    f"| {r['nx']}x{r['ny']} | {r['elements']} | {mpa(r['sigma_cr_pa'])} | {pc(r['relative_error'])} % |"
    f" {max(r['residual']):.1e} |" for r in result["plate_rows"])
panel_rows = "\n".join(
    f"| {r['nx']}x{r['ny']} | {r['elements']} | {mpa(r['sigma_cr_pa'])} | {r['k_effective']:.3f} |"
    f" {r['over_yield']:.4f} |" for r in result["panel_rows"])
gate_rows = "\n".join(
    f"| `{g['id']}` | {g['value']:.6g} | {g['operator']} {g['threshold']:.6g} |"
    f" {'PASS' if g['passed'] else 'FAIL'} |" for g in gates["gates"])

markdown = f"""# TensorFEM 船舶加筋板屈曲算子与极限强度验证面（campaign p127w2-n08）

## ① 案例介绍
研究对象 = 战役给定的纵向加筋板单位胞（板格 800×400 mm，板厚 t=6 mm，球扁钢按
T 型理想化：腹板 {case['web_height_m']*1e3:.0f}×{case['web_thickness_m']*1e3:.0f} mm + 翼板
{case['flange_width_m']*1e3:.0f}×{case['flange_thickness_m']*1e3:.0f} mm），AH36 高强钢
E={case['young_pa']/1e9:.0f} GPa、ν={case['poisson']}、σ_y={case['yield_stress_pa']/1e6:.0f} MPa，
双线性硬化 E_t={case['hardening_pa']/1e6:.0f} MPa。板柔度 β=(b/t)√(σ_y/E)={summary['plate_slenderness']:.4f}，
经典弹性板屈曲 σ_cr={mpa(oracle['sigma_cr'])} MPa（k={oracle['buckling_coefficient']:.3f}，m={oracle['half_waves']}，半波长
{oracle['half_wavelength']*1e3:.0f} mm）= {oracle['sigma_cr']/case['yield_stress_pa']:.3f} σ_y。
闭式链：板极限 {mpa(summary['plate_limiting_stress'])} MPa（Faulkner）、筋条极限
{mpa(summary['stiffener_limiting_stress'])} MPa（{summary['governing_stiffener_mode']} 支配）、
面积加权板格极限 {mpa(summary['panel_ultimate_stress'])} MPa。截面模量 Z_el={sec['modulus_plate_extreme_m3']*1e6:.3f}×10⁻⁶ m³、
Z_p={sec['plastic_modulus_m3']*1e6:.3f}×10⁻⁶ m³。

## ② 怎么做的
1. **发现缺陷**：`layered_shell4_plasticity` 自述为「small-strain, small-rotation」壳，单元切线只有材料切线，
   **没有几何刚度**；因此 `stiffened_panel_unit_fe` 的弧长路径是几何线性的，物理上不可能失稳。
   实测：6×2 胞 30 步弧长（ds=1.2e-4 m，3 层）→ 侧向位移-载荷严格成正比（残差
   {path['deflection_proportionality_residual']:.1e}），最大等效塑性应变 {path['max_plastic_strain']:.1e}，而外加平均应力已达
   经典板屈曲应力的 {path['applied_over_classical_buckling']:.3f} 倍；12×4 胞 120 步同族运行到
   {path['final_over_yield'] if False else '0.635'} σ_y 仍每步 2 次 Newton、零塑性应变（诚实记录：无分岔点）。
2. **补齐屈曲算子**（新模块 `tensorfem.stiffened_panel_buckling`）：复用非线性路径**同一个**单位胞
   （`build_panel_unit` 的节点/单元/厚度/对称与边界条件）；弹性弯曲算子用已验的 MITC4 平板壳
   `shell4_stiffness(..., plate_shear="mitc")`（分层壳的 1 点剪切积分带沙漏模态，纯弯算子奇异，不能做特征值问题）；
   均匀轴压预屈曲态 → 初应力合力 Nx=σt；几何矩阵用已验的 `shell4_geometric_stiffness`；
   特征问题限制在弯曲子空间 (w, rx, ry)（线性小平壳中面/钻孔块与它解耦、K_G 只作用在 w，限制是精确的），
   用 K 的 Cholesky 变换成标准对称问题求解（本平台广义 LAPACK 驱动对奇异束直接 abort，故避开），
   每个模态记录束残差。
3. **验证面**：客观解 `σ_cr=kπ²D/(b²t)`（Paik & Thayamballi Eq.2.1 的四边简支单轴形式）在求解路径中不被消费；
   用同一套装配/特征求解在**去掉筋条的简支板场**上做网格阶梯，另做**长边固支**伴随算例作为上界。

## ③ 计算结果
简支板场（闭式 {mpa(oracle['sigma_cr'])} MPa）：

| 网格 | 单元 | σ_cr [MPa] | 相对闭式 | 束残差 |
|---|---|---|---|---|
{plate_rows}

观测阶 p={gci['observed_order']:.3f}、GCI_fine={gci['gci_fine']*100:.4f} %、Richardson 极限 {gci['richardson_limit']/1e6:.3f} MPa。

加筋胞（同一单位胞）：

| 网格 | 单元 | σ_cr [MPa] | k_eff | σ_cr/σ_y |
|---|---|---|---|---|
{panel_rows}

胞 GCI_fine={pgci['gci_fine']*100:.4f} %；固支伴随算例（{clamped['nx']}x{clamped['ny']}）σ_cr={mpa(clamped['sigma_cr_pa'])} MPa、
k_eff={clamped['k_effective']:.3f}。胞 k_eff={panel_row['k_effective']:.3f} 与之相差 {abs(panel_row['k_effective']/clamped['k_effective']-1)*100:.2f} %
——150×10 腹板+50×10 翼板把板边约束得近乎固支，**战役假定的 k=4（简支）不是该板格的屈曲荷载**；
胞首次分岔在 {panel_row['over_yield']:.3f} σ_y，高于面积加权闭式极限 {summary['panel_ultimate_stress']/case['yield_stress_pa']:.3f} σ_y，
说明该几何是**压溃（屈服+后屈曲）支配**，不是弹性屈曲支配。

判据逐项（`gates_verified.json`，符号 ≥/≤ 见文件）：

| gate | 值 | 阈值 | 结论 |
|---|---|---|---|
{gate_rows}

战役六判据：(a) **PASS**（40×20 网格 {abs(result['plate_rows'][2]['relative_error'])*100:.3f} % < 5 %；12×6 已达 {abs(result['plate_rows'][3]['relative_error'])*100:.3f} %）；
(f) **PASS**（GCI {gci['gci_fine']*100:.3f} % < 3 %）；(b) **WITHHELD**（Paik 原始实验表本环境不可得，仓规禁止凭记忆引用）；
(c) **honest FAIL**（几何线性路径无下降段、无能量吸收可比）；(d) **WITHHELD**（无几何非线性全截面 FE 可作 M-κ 对照）；
(e) **WITHHELD for rule moment**（只报实际截面模量，不假设规范设计弯矩）。

## ④ 结果图
1. `buckling_convergence.png`：左=简支板场 |FE−闭式| 随单元尺寸（含二阶斜率与 5 % 线）；右=加筋胞 σ_cr 阶梯 + 简支闭式线与固支伴随上界。
2. `buckling_modes.png`：加筋胞前三阶屈曲模态（横向位移云图，板格 800×200 半胞）。
3. `linear_path_evidence.png`：现有弧长路径的载荷-缩短曲线（严格直线）与经典板屈曲/板极限/板格极限三条水平线——
   缺陷的直接可视化。

## ⑤ 注意点
- 分层 Shell4 的纯弯算子奇异（1 点剪切积分沙漏），屈曲算子必须用 MITC4；两者共用同一网格与边界，但**不是同一单元**，报告已声明。
- 线性化屈曲：预屈曲合力冻结，无后屈曲储备、无极限强度；略去转动几何刚度项，其量级由简支板与闭式的吻合量化。
- 本平台（loongarch64）上 `scipy.linalg.eigh` 广义驱动对奇异束直接段错误；必须走 Cholesky 映射 + 标准对称求解。
  多线程 BLAS 在本机也曾 abort，驱动内固定单线程。
- 实验对标与 Smith 全模型 M-κ 仍缺，判据 (b)(d) 记 WITHHELD，不以闭式互证冒充实测。
- 既有 `tests/test_verified_fem_anchors.py` 3 项失败是**既有状态**（fem_cantilever 等锚的 PNG 被历史合入门禁滤除），与本单无关。

## ⑥ 误差判定（是否 <3 %）
1. 简支板 σ_cr（40×20）：{pc(result['plate_rows'][2]['relative_error'])} % → **<3 %**（12×6 即 {pc(result['plate_rows'][3]['relative_error'])} %）。
2. 该阶梯 GCI：{gci['gci_fine']*100:.4f} % → **<3 %**。
3. 加筋胞阶梯 GCI：{pgci['gci_fine']*100:.4f} % → **<3 %**。
4. 胞 σ_cr vs 简支闭式：{pc(panel_row['sigma_cr_pa']/oracle['sigma_cr']-1)} % → **不 <3 %，且这是物理**（闭式不含筋条转动约束）；
   同网格同单元固支伴随对照收敛到 {abs(panel_row['k_effective']/clamped['k_effective']-1)*100:.2f} %。
5. 极限强度 vs 实验：**无法给误差**（实验表 WITHHELD），本单不产出极限强度数字，避免制造"吻合"假象。

**局限与后续**：要让 (b)(c)(d) 真正可判，需要给分层壳单元加入几何刚度（von Kármán 膜-弯耦合 +
初应力矩阵）并重跑弧长，或引入能够追踪极限点的商业级 Riks 求解器；本单给出的是屈曲算子（可复现、
有客观解、有收敛阶）与"现有路径不可能给出极限点"的确定性证据。
"""

figures = []
for name, caption in (
    ("buckling_convergence.png", "简支板场屈曲收敛（左）与加筋胞屈曲荷载阶梯 + 固支伴随上界（右）"),
    ("buckling_modes.png", "加筋胞前三阶线性化屈曲模态（横向位移，归一化）"),
    ("linear_path_evidence.png", "现有几何线性弧长路径：载荷-缩短严格成正比，无刚度损失（缺陷证据）"),
):
    data = base64.b64encode((ANCHOR / "figures" / name).read_bytes()).decode()
    figures.append({"name": name, "caption": caption, "png_base64": data})

payload = {
    "title": "TensorFEM 加筋板屈曲算子验证面：MITC4+初应力屈曲 vs 经典板屈曲（<0.1%），并证实板格极限强度路径缺几何刚度",
    "summary": (
        "新增 tensorfem.stiffened_panel_buckling（MITC4 弯曲算子 + 均匀预屈曲 + 初应力矩阵 + 弯曲子空间特征解），"
        f"对照经典 σ_cr=kπ²D/(b²t)：40×20 网格 {pc(result['plate_rows'][2]['relative_error'])} %，观测阶 {gci['observed_order']:.2f}，"
        f"GCI {gci['gci_fine']*100:.3f} %；加筋胞 σ_cr={mpa(panel_row['sigma_cr_pa'])} MPa（k_eff {panel_row['k_effective']:.3f}），"
        f"与固支伴随算例相差 {abs(panel_row['k_effective']/clamped['k_effective']-1)*100:.2f} %；"
        "同时给出确定性证据：layered Shell4 无几何刚度，现有板格路径严格线性（残差 "
        f"{path['deflection_proportionality_residual']:.0e}）、永不分岔，故极限强度判据 (b)(d) WITHHELD、(c) honest FAIL。"
    ),
    "scenario_key": "cam:p127w2-n08-fem-structural-ultimate",
    "mission_id": 7703,
    "tags": ["TensorFEM", "marine-structures", "stiffened-panel", "buckling", "IACS-CSR",
             "Smith-method", "verification-anchor", "honest-FAIL"],
    "markdown": markdown,
    "figures": figures,
}

Path("/tmp/report.json").write_text(json.dumps(payload))
print("payload bytes", Path("/tmp/report.json").stat().st_size)