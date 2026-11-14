# GAP-PP-3-ICE-STRUCT-PLASTIC-SET 独立验证附页（verifier 席）

**锚点**: `GAP-PP-3-ICE-STRUCT-PLASTIC-SET`（板格低温钢弹塑性 + 永久挠度）
**模块**: `tensorfem.plate_patch_plastic_limit`（+ 新增 `ilyushin_sine_upper_bound` / `plastic_bound_bracket`）
**核验脚本**: `scripts/verify_gap_pp3_independent_bounds.py` → `artifacts/gap_pp3_bound_bracket.json`
**测试**: `tests/test_plate_patch_plastic_bounds.py`（15 项）
**性质**: 对已合并实现（#1008，m20895）的独立复核 —— 不改物理模型，只补验证欠账、纠正上一份报告的不实/失效数字。
**独立性限制**: 复核者与被复核代码同仓；FE 侧复用了同一单元（`layered_shell4_plasticity`），因此"单元实现错误"无法被本页排除。可以在本页被排除的是：解析基准算错、验收判据被放宽/归一化掉、以及报告表格数字与代码输出不一致。

---

## 1. 为什么这一页要存在

锚点验收判据原文要求四项：

| 判据 | 原文 |
|---|---|
| A1 | 弹性→塑性过渡点与**解析屈服载荷**误差 <5% |
| A2 | 塑性极限载荷与 **Ilyushin/上限分析解**误差 <10% |
| A3 | 3 档网格收敛 |
| A4 | pytest ≥ 10 绿 |

上一份报告（`docs/gap_pp3_ice_struct_plastic_set_verification.md`）把 A1 改写成"**同网格弹性场归一化后** <5%"，
并自陈"原始误差（vs 解析 Navier 解）为 -20% ~ +5%"。这是一次**判据弱化**：
拿 FE 自己的弹性解当基准，A1 就退化成"FE 在自己网格上自洽"，不能证明过渡点对得上解析屈服载荷。
同时 A2 只对了一条屈服线上界，而验收点名的 **Ilyushin** 解在仓里根本不存在（`grep -ri ilyushin src/` 为空）。

本页做两件事：
1. **补上 Ilyushin 运动学上界**（`ilyushin_sine_upper_bound`，双正弦机制，与屈服线金字塔机制不同族，故为独立基准）；
2. **按原文口径重测** A1（不做归一化，直接用解析屈服载荷）与 A3，并把 A2 的区间宽度摊开。

---

## 2. 方法

### 2.1 新增的独立上界（解析）

机制取板的第一个平滑模态 `w = w0 sin(πx/a) sin(πy/b)`（简支边界上为零，运动学容许），
由 Ilyushin 运动学定理：

```
p_u = W_int / W_ext
W_int/w0 = m_p π²(m²/a² + n²/b²)·4ab/(π²mn)
W_ext/w0 = (a/mπ)[cos(mπx1/a) − cos(mπx2/a)]·(b/nπ)[cos(nπy1/b) − cos(nπy2/b)]
```

Johansen（Neville）方形弯矩判据下耗散为 `m_p(|κ1|+|κ2|)`；双正弦模态两个主曲率**同号**，
故 `|κ1|+|κ2| = π²(m²/a²+n²/b²)φ` 无歧义，上界是闭式的、与网格无关。
（同号曲率下 von Mises 判据的耗散与之相差 <2%，见 §5 局限。）

自检（已写成测试）：满布载荷方板退化到 `2π²m_p/a² = 19.74 m_p/a²`；
`b ≫ a` 退化为梁的正弦机制值 `π²m_p/a² = 9.87 m_p/a²`（大于精确值 `8m_p/a²`，上界方向正确）。

### 2.2 复核口径

* 弹性分支：同网格中心挠度 vs Navier 双级数（41 项）。
* 过渡点：`surface_extrapolated_mises` 达到 σ_y 的载荷 vs `analytic_first_yield_pressure`，**不归一化**。
* 区间：`strip_lower_bound`（静力容许下界）、`yield_line_patch_upper_bound`（屈服线上界）、
  `ilyushin_sine_upper_bound`（新增上界），报告最紧上界与区间宽度 `U/L`。

---

## 3. 计算结果（程序化注入，零手抄）

几何/材料：a=b=1.0 m，t=0.02 m，E=210 GPa，ν=0.3，σ_y=355 MPa（DH36@20 °C），
硬化 H=2×10¹⁰ Pa，中心斑 0.5a×0.5b。

### 3.1 解析基准

| 量 | 值 |
|---|---|
| 塑性弯矩 m_p = σ_y t²/4 | 35 500 N·m/m |
| 弹性首次屈服 p_y（Navier + Mises） | **804 007 Pa** @ (0.5, 0.5) |
| 条带下界 p_lb | 757 333 Pa |
| 屈服线上界 p_u(yield-line) | 1 704 000 Pa |
| Ilyushin 双正弦上界 p_u(sine) | **1 401 484 Pa** |
| 两条独立上界互差 | **21.59 %** |
| 区间宽度 U/L（最紧上界 / 下界） | **1.851** |

### 3.2 A1 复核：过渡点误差（原始口径，无归一化）

| 网格 | 中心挠度误差 vs Navier | FE 首次屈服 | 解析 p_y | 原始误差 |
|---|---|---|---|---|
| 8×8 | +1.33 % | 809 347 Pa | 804 007 Pa | **+0.66 %** |
| 12×12 | +1.33 % | 805 575 Pa | 804 007 Pa | **+0.20 %** |
| 16×16 | +1.47 % | 803 493 Pa | 804 007 Pa | **−0.06 %** |

**结论**：A1 在**原始口径**下就满足 <5%（最大 0.66 %），无需任何归一化。
上一份报告"原始误差 -20% ~ +5%"的说法在当前 HEAD 上**不复现**——
那是 `exact_thickness`（中点积分 z² 缺项 `dz³/12` 修正）引入之前的状态。

### 3.3 A3 网格收敛

过渡点误差 0.66 % → 0.20 % → 0.06 %，单调收敛（`verdicts.mesh_convergence_monotone_in_first_yield = true`）。
弹性挠度误差 1.33 % → 1.33 % → 1.47 % **不随加密下降**，是 ~1.4 % 的模型残差（Q4 Mindlin，a/t=50），
不再是被上一份报告归因的"中点积分 3 % 偏差"。

### 3.4 永久挠度

8×8、H=2×10¹⁰、加载到 1.2 MPa 后卸载到 0：残余中心挠度 **0.683 mm**（= 3.4 % t）。
8×8 上加/卸载路径点见 `artifacts/gap_pp3_bound_bracket.json` 的 `meshes[0].path_points` / `unload_points`。

---

## 4. 注意点：本次复核**查出的问题**

| 编号 | 严重度 | 问题 | 证据 | 处置 |
|---|---|---|---|---|
| F1 | 高 | 上份报告 §6 用"同网格弹性场归一化"替代"A1 vs 解析屈服载荷"，属判据弱化 | 两文对读；本页 §3.2 证明原口径即可通过 | 已在本页按原口径重测并给出原始误差 |
| F2 | 高 | 锚点 A2 点名的 **Ilyushin 解缺失**，collapse 只对一条上界 | `grep -ri ilyushin src/` 为空（复核前）；上份报告 §6 只列 yield-line | 已新增 `ilyushin_sine_upper_bound` + `plastic_bound_bracket` |
| F3 | 高 | 上份报告"解析基准值"表两处数字与代码输出不符：p_u 写成 1.42 MPa（代码 1.704 MPa）、p_y 写成 ~0.48 MPa（代码 0.804 MPa） | 表 vs `artifacts/gap_pp3_bound_bracket.json` | 报告数字不可作证据；以 JSON 为准 |
| F4 | 中 | "collapse 与上界 <10 %" 的证明力弱：区间宽达 **1.85×**，两条上界互差 21.6 %，FE 落在区间任意位置都能说"通过" | §3.1 | 判为 **WITHHELD**（见 §6），需更紧的下界或溃缩判据重定义 |
| F5 | 中 | 上份报告把残余弹性偏差归因"中点积分 ~3 %"，当前实现该偏差已被 `exact_thickness` 消掉，残留 1.4 % 另有来源（Q4 剪切/斑载集中） | §3.3 | 归因待查，不重复其结论 |
| F6 | 低 | 上份报告 §3.3 的三档网格表整列是"—"占位，非实测 | 原文表格 | 已用 JSON 数值替代 |

---

## 5. 适用范围与局限

* 板单元为小应变 Mindlin 平壳，`w/t ≈ 0.5–1.5` 内可用；再大则膜强化未捕获，溃缩压力只是纯弯曲机制值。
* 新上界取 Johansen 方形弯矩判据；同号曲率下与 von Mises 耗散差 <2 %，故上界对 Mises 材料略偏松（方向安全）。
* `terms>1` 只是在同一族内取更小值；非正定模态（斑内变号）被跳过，其 `|w|` 形式不光滑，未用作机制。
* **未能排除**：FE 单元本身系统性偏差（复核者与被复核代码同源）；永久挠度无独立解析基准，
  仅有"为正/有界/卸载单调"三条弱不变量。

---

## 6. 误差判定（逐项回答"是否 <3%"）

| 关键量 | 基准 | 误差 | <3 %？ |
|---|---|---|---|
| m_p | σ_y t²/4 | 0（定义式） | 是 |
| 满布方板上界 | 2π²m_p/a² | <1e−12 | 是 |
| 条带上界退化 | π²m_p/a²（b/a=40） | <0.5 % | 是 |
| 弹性中心挠度 | Navier 41 项 | +1.33 / +1.33 / +1.47 % | 是（临界） |
| **过渡点 A1** | 解析 p_y | **+0.66 / +0.20 / −0.06 %** | **是** |
| 过渡点 A3 收敛 | 单调 | 是 | — |
| **极限载荷 A2** | Ilyushin 上界 | FE 溃缩点未在本页取得（硬化模型无极限点，`run_patch_plastic_path` 在 1.48 MPa 仍收敛） | **无基准 → 判 WITHHELD** |

**诚实声明**：A2 判为 **WITHHELD**（证据不足），不是 pass。理由是双重的：区间宽 1.85×，
且带等向硬化 H=2×10¹⁰ 的路径在 1.48 MPa（已超最紧上界 1.401 MPa）仍收敛——
"崩溃载荷"必须由近理想塑性 + 极限点判据（或位移控制/弧长）定义后重测，本页未做。

---

## 7. 复现

```bash
PYTHONPATH=src python scripts/verify_gap_pp3_independent_bounds.py
PYTHONPATH=src python -m pytest tests/test_plate_patch_plastic_bounds.py -q
```

`artifacts/gap_pp3_bound_bracket.json` 由脚本直接写出，报告表格与该 JSON 一一对应。