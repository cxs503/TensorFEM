# Lindqvist/Riska 冰阻力文献转录与量纲审计（GAP-ICE-LIT-TRANSCRIPT）

> 节点：`GAP-ICE-LIT-TRANSCRIPT` 仓：`TensorFEM`
> 分支：`agent/m17161` 脚本：`benchmarks/anchors/lindqvist_riska_anchor.py`
> 产物：`benchmarks/anchors/lindqvist-riska-ice-resistance.json`（全部数字源）
> 转录注册表：`benchmarks/anchors/sources/lindqvist_riska_ice_resistance.transcription.json`
> 测试：`tests/test_lindqvist_riska_anchor.py`（29 项全绿）
> 时间戳（UTC）：2026-11-13

---

## 1. 案例介绍（物理背景/几何/参数表）

### 1.1 物理背景

船舶在冰区航行时，船体与冰层的相互作用产生冰阻力。Lindqvist (1989) 在 POAC'89 会议上提出了著名的**三分解冰阻力模型**，将总冰阻力分解为四个分量：

1. **压溃阻力 R_c**：艏斜板压碎冰缘所需的力，∝ h²
2. **破断阻力 R_b**：冰盖按弹性地基板弯曲破断所需的力，∝ h^{5/4}
3. **浸没阻力 R_s**：碎冰被压入水下并沿船体滑移的阻力，含静载（∝ h）和动载（∝ v²）两部分
4. **动量阻力 R_mom**：加速碎冰到船速所需的力，∝ h·v²

该模型是国际冰阻力计算的标准方法之一，被 ITTC 冰委员会推荐。

### 1.2 合成算例参数表

| 参数 | 符号 | 值 | 单位 | 来源 |
|------|------|-----|------|------|
| 船宽 | B | 22.0 | m | 声明假设 |
| 吃水 | T | 7.5 | m | 声明假设 |
| 首部柱角 | ψ | 30.0 | deg | 声明假设 |
| 水线半角 | α | 20.0 | deg | 声明假设 |
| 冰厚 | h | 0.8 | m | 声明假设 |
| 冰弯曲强度 | σ_f | 500 | kPa | 声明假设 |
| 冰压溃强度 | σ_c | 2.5 | MPa | 声明假设 |
| 水密度 | ρ_w | 1025 | kg/m³ | 声明假设 |
| 冰密度 | ρ_i | 900 | kg/m³ | 声明假设 |
| 冰弹性模量 | E | 5 | GPa | 声明假设 |
| 冰泊松比 | ν | 0.30 | — | 声明假设 |
| 摩擦系数 | μ | 0.10 | — | 声明假设 |
| 航速 | v | 2.57 | m/s (5 kn) | 声明假设 |
| 重力加速度 | g | 9.81 | m/s² | 声明假设 |

---

## 2. 方法/流程/关键设置

### 2.1 转录注册表框架

创建了与 SNAME 转录注册表同构的 Lindqvist/Riska 转录注册表：
- `sources_sought`: 3 个文献源（Lindqvist 1989 POAC'89, Riska 1997, Lindqvist 1983 论文）
- `entries`: 6 个锚点条目（R_c, R_b, R_s, R_mom, R_total, Table 1 验证表）
- 所有条目当前状态为 `WITHHELD`（未获得原文授权副本）
- M1 门槛：需 ≥4 个逐字转录三元组（公式, quote, page/table）

### 2.2 量纲审计链

对 Lindqvist 模型的 8 个公式步骤进行了完整的量纲齐次性检查：
- S1: R_c = f_c·σ_c·h²·(tan ψ + μ·cos ψ/sin α) → [N] ✓
- S2: λ = (k/4D)^{1/4} → [m⁻¹] ✓
- S3: R_b = f_b·σ_f·B·h²·λ·(sin ψ + μ·cos ψ) → [N] ✓
- S4: A = f_s·μ·ρ_w·g·B·(T/tan α) → [N/m] ✓
- S5: δ = C_p·v²/(2g) → [m] ✓
- S6: R_s = A·(h+δ) → [N] ✓
- S7: R_mom = C_m·ρ_i·h·B·v² → [N] ✓
- S8: R_ice = R_c + R_b + R_s + R_mom → [N] ✓

### 2.3 机器检查（6 项）

| 检查项 | 方法 | 结果 |
|--------|------|------|
| CHK1 量纲齐次性 | (kg,m,s) 指数代数 | ✅ PASS (8/8) |
| CHK2 标度律 | 数值扰动 log-log | ✅ PASS (Rc:h², Rb:h^1.25, Rmom:h¹, δ:v²) |
| CHK3 零厚极限 | h→0 时各分量→0 | ✅ PASS |
| CHK4 速度分解 | R(v)=R₀+c·v² | ✅ PASS (ratio=4.000) |
| CHK5 单位不变性 | SI↔CGS 映射 | ✅ PASS (误差<1e-10) |
| CHK6 厚度扫描 | h∈[0.1,2.0] 单调性 | ✅ PASS (指数∈(1,2)) |

---

## 3. 计算结果

### 3.1 基准算例结果（h=0.8m, v=5kn）

| 分量 | 值 (kN) | 占比 (%) |
|------|---------|----------|
| R_c (压溃) | 程序化注入 | 程序化注入 |
| R_b (破断) | 程序化注入 | 程序化注入 |
| R_s (浸没) | 程序化注入 | 程序化注入 |
| R_mom (动量) | 程序化注入 | 程序化注入 |
| **R_total** | **程序化注入** | **100%** |

> 注：具体数值见 `benchmarks/anchors/lindqvist-riska-ice-resistance.json` 的 `dry_run_fixture.results` 节，程序化注入零手抄。

### 3.2 标度律验证

| 分量 | 预期指数 | 测量指数 | 误差 |
|------|----------|----------|------|
| R_c vs h | 2.000 | 2.000 | <1e-6 |
| R_b vs h | 1.250 | 1.250 | <1e-4 |
| R_mom vs h | 1.000 | 1.000 | <1e-6 |
| δ vs v | 2.000 | 2.000 | <1e-6 |
| R_mom vs v | 2.000 | 2.000 | <1e-6 |

---

## 4. 结果图

### 图1：冰阻力分量 vs 冰厚

![fig1](docs/assets/lindqvist-riska-transcription/fig1_resistance_vs_thickness.png)

*各分量随冰厚变化：R_c ∝ h²（红色，增长最快），R_b ∝ h^{1.25}（蓝色），R_s（绿色，含恒定动载偏移），R_mom ∝ h（紫色）。总阻力（黑色虚线）在 h∈[0.1,2.0] 范围内单调递增。*

### 图2：总阻力 vs 航速

![fig2](docs/assets/lindqvist-riska-transcription/fig2_resistance_vs_speed.png)

*不同冰厚下的总阻力-速度曲线。动载项使曲线呈二次型，低速时准静态基值主导，高速时动量项主导。*

### 图3：分量占比饼图

![fig3](docs/assets/lindqvist-riska-transcription/fig3_component_breakdown.png)

*h=0.8m, v=5kn 时的分量占比。压溃项通常占比最大，破断项次之。*

### 图4：对数坐标标度律

![fig4](docs/assets/lindqvist-riska-transcription/fig4_scaling_exponents.png)

*对数坐标下的阻力-冰厚关系，验证解析标度指数。虚线为理论斜率参考线。*

---

## 5. 注意点（踩过的坑/局限/适用范围）

### 5.1 转录门槛未达成（HONEST FAIL）

- **现状**：M1 门槛要求 ≥4 个逐字转录三元组，当前为 0（所有条目 WITHHELD）
- **原因**：未获得 Lindqvist 1989 POAC'89 原文授权副本
- **影响**：转录门为 WITHHELD，总体判定为 WITHHELD（诚实失败）
- **路径**：获取原文后逐字转录 ≥4 个（公式, quote, page/table）三元组即可通过

### 5.2 公式来源声明

- 本实现中的公式来自**二手文献综合**（ship-performance-platform 仓的 DERIVATION.md 等），**非逐字转录**
- 所有公式在转录注册表中明确标记为 `WITHHELD`，不升级为 REAL
- 量纲审计和标度律验证的是**公式结构**的正确性，而非原文系数的准确性

### 5.3 适用范围

- 模型适用于**平整冰**（level ice）低速航行工况
- 不含碎冰（brash ice）和冰脊（ridge）的完整模型（需 Riska 公式补充）
- 不含冲击放大效应（不能直接用作结构强度输入）

---

## 6. 误差判定

| 判据 | 基准 | 结果 | 是否<3% |
|------|------|------|---------|
| 量纲齐次性 | 精确匹配 | 8/8 通过 | N/A（精确） |
| R_c 标度指数 | 2.000 | 2.000 | ✅ <1e-6 |
| R_b 标度指数 | 1.250 | 1.250 | ✅ <1e-4 |
| R_mom 标度指数 | 1.000 | 1.000 | ✅ <1e-6 |
| 速度二次律 | ratio=4.0 | 4.000 | ✅ <0.25% |
| 单位不变性 | 误差=0 | <1e-10 | ✅ |
| 原文系数对标 | Lindqvist 表1 | WITHHELD | ❌ 无基准 |

**结论**：所有**可机器裁定**的判据均通过（量纲/标度/极限/不变性）。原文系数对标因未获得源文献而标记 WITHHELD（诚实失败，不伪造）。

---

## 7. 产物清单

| 文件 | 说明 |
|------|------|
| `benchmarks/anchors/sources/lindqvist_riska_ice_resistance.transcription.json` | 转录注册表（6 条目，3 文献源） |
| `benchmarks/anchors/lindqvist_riska_anchor.py` | 锚点对账脚本（量纲审计+机器检查） |
| `benchmarks/anchors/lindqvist-riska-ice-resistance.json` | 产物 JSON（全部数字源） |
| `benchmarks/gen_lindqvist_riska_figures.py` | 图生成脚本 |
| `docs/assets/lindqvist-riska-transcription/fig*.png` | 4 张结果图 |
| `tests/test_lindqvist_riska_anchor.py` | pytest 测试套件（29 项） |
