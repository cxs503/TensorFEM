# 冰压斑+侧压组合下加筋板架极限强度：非线性 FE 交叉校核（验证欠账闭合）

锚点 `GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE`（加筋板架在冰压斑 + 侧压组合下的极限强度）。
本单是**验证欠账闭合**：`docs/GRILLAGE-ICE-STRUCT-ULTIMATE.md` §5 与
`docs/CASE-GAP-PP-5-...-round6.md` 均如实声明——"本模块相对 FE 的后备承载力偏差 **未知**，
与 `ice_belt_collapse` 的非线性弹塑性 FE 交叉比对**本次未运行**（容量纪律只做轻量活），已登记为
下阶段验证项"。本单把这条欠账**跑完并落盘**。

## 1. 案例介绍（物理背景）

冰区（Polar Class）舷侧冰带是"板 + 纵骨 + 横框架"的正交加筋板架。锚点主交付
`grillage_ice_ultimate` 用**闭式/经验链**（Navier 正交板 + 梁柱 + 刚塑性倒塌）把"板格弯曲应力
-> 结构极限"的最后一跳闭合：板格局部弯曲承载力取塑性铰弯矩 `m_p = σ_y t²/4`（受共存膜压经
von Mises 因子 `√(1−η²)` 退化），极限冰压 = Navier 峰值弯矩达到该承载力的值。

**验收判据**（feature_ledger）明确允许"非线性求解 **或** 经验公式"二选一，但闭式分支**从未**被
独立**非线性有限元**倒塌交叉校核过——这正是本单偿还的验证欠账。

| 参数 | 值 |
|---|---|
| 板厚 t | 20 mm |
| 纵骨间距 b / 横框架间距 a | 0.35 m / 2.10 m |
| 代表板格 | 2.10 m × 0.35 m（6:1）|
| 材料 | E=206 GPa, ν=0.3, σ_y=355 MPa（EH36 理想弹塑性）|
| 板格柔度 β | 0.73（Faulkner 有效宽度比 1.00）|

## 2. 方法（怎么做的）

对三个已文档化组合工况（light/design/severe 冰压斑 + 静水头），各建一个**裸简支板**（代表受载
板格，`ice_belt_collapse.PlateModel` 分层弹塑性 von-Karman 板），施加**组合侧向**载荷（居中冰压斑
+ 满布静水头）作为弧长参考，用 Crisfield 球面弧长延拓追踪塑性倒塌路径。FE 峰值载荷因子
`lam_peak` 是板实际倒塌的载荷乘子。

**保守性判据**：闭式 `plate_local_bending` 是**首屈**（first-yield）判据（Navier 峰值弯矩达到塑性铰
弯矩），是真实屈服线倒塌的**下界**。因此 FE 倒塌必须 **≥** 闭式预测（`lam_peak ≥ 1/u_plate_eta0`）；
`lam_peak / (1/u_plate_eta0)` 即量化保守裕度。若 FE < 闭式（过预测）则标记为不安全设计公式。

**范围与诚实**：
- FE 模型**仅侧向**：共存膜压（η=σ_x/σ_y）不作为面内预载施加，`lam_peak` 与闭式**未退化**承载力
  `m_p=σ_y t²/4`（`u_plate_eta0`）对比。轴向退化 `√(1−η²)` 是塑性铰的 von Mises 交互，已由
  `T11b` 闭式自洽单独验证——FE 仅侧向交叉校核是最干净的"最后一跳"，不纠缠两支载荷。
- 小应变、中等转动（von Karman）板；无纵骨腹板侧倾、无残余应力、无应变率——与闭式模块同范围。
- FE 倒塌是**塑性铰**机构（板厚实，β=0.73），非屈曲路径；弧长驱动用于解极限点，非分叉。

## 3. 计算结果（数字表，程序化注入）

`outputs/grillage_ice_fe_crosscheck/fe_crosscheck.json`（`run_fe_crosscheck` 生成，无手抄）：

| 工况 | lam_peak (FE) | 1/u_plate_eta0 (闭式) | 保守裕度 | 弧长状态 |
|---|---:|---:|---:|---|
| light-ice + ballast head | 4.8222 | 3.2273 | **1.494** | still-ascending |
| design-ice + design head | 1.8138 | 1.2121 | **1.496** | bracketed |
| severe-ice + tank head | 0.9815 | 0.7536 | **1.302** | bracketed |
| uniform-head yield-line (T11a) | 2.8467 | 1.0000 | **2.847** | still-ascending |

**全部 4 档 `pass_conservative = True`**：FE 倒塌 ≥ 闭式首屈承载力，闭式是保守下界。

**T11a 均匀支屈服线锚**：均匀 1 MPa 水头（无斑）下 FE 倒塌压力 vs 刚塑性 `p_u = 8 M_p / b²`
（6:1 板格单向条带机构）。FE 倒塌 2.847× 高于屈服线——膜拉（membrane stiffening）抬升倒塌，
与经典板理论一致。

## 4. 结果图

![FE vs closed-form conservatism](grillage_ice_fe_crosscheck/fig1_fe_vs_closedform_conservatism.png)
*图 1：三档组合工况 + 均匀支锚下 FE 倒塌载荷因子 vs 闭式首屈乘子（保守裕度 1.30–1.50×）*

![Arc-length collapse path](grillage_ice_fe_crosscheck/fig2_arc_length_path.png)
*图 2：design-ice 工况弧长延拓倒塌路径（载荷因子 vs 中心挠度，极限点括弧）*

## 5. 注意点（踩过的坑/局限）

1. **light-ice 与 uniform 支 `arc=still-ascending`**：默认 80–100 步内弧长未完全到达极限点，
   `lam_peak` 是倒塌的**下界**（仍在爬升）。对保守性判据无碍（FE 已 ≥ 闭式），但真实保守裕度
   可能更高——已如实标注，不把"下界"当"精确峰值"。
2. **FE 仅侧向**：膜压退化 `√(1−η²)` 未在 FE 轨迹中作为面内预载，故 `lam_peak` 与**未退化**闭式
   对比。这是刻意的"最干净最后一跳"，轴向交互由 T11b 闭式自洽单独设门。
3. **无纵骨腹板侧倾/残余应力/应变率**：与闭式模块同范围，不宣称覆盖这些失效模式。
4. **保守性方向**：闭式首屈是下界，FE 倒塌 30–50% 更高——对**设计**偏保守（安全），但若把闭式
   `p_ult` 当"结构极限载荷"引用会低估真实倒塌（此处偏保守 1.3–1.5×）。与 round6 F3 的
   "member limit state vs collapse limit" 区分一致。

## 6. 误差判定（逐项，是否 <3%）

| 关键量 | 参照 | 值 | 判定 |
|---|---|---:|---|
| FE 倒塌 ≥ 闭式首屈（三档） | 保守性判据（下界） | 1.30–1.50× | **是**（保守方向，非误差）|
| 均匀支 FE 倒塌 vs 屈服线 p_u | 刚塑性闭式 | 2.847× | **是**（膜拉抬升，物理预期）|
| 闭式轴向退化 √(1−η²) | von Mises 交互（T11b 既有） | 自洽 | **是**（<3%，既有门）|
| 闭式 vs Paik-Thayamballi（T7 既有） | 经验式 | 4.9–6.3% | 是（15% 门）|

**显式回答**：本单**无"误差 <3%"的数值对标**——因为 FE 与闭式是**不同极限状态**（FE=真实倒塌，
闭式=首屈下界），不可互校为"误差"，只能校为"保守方向"。保守方向全部成立（FE ≥ 闭式），
且与经典板理论（膜拉抬升屈服线倒塌）一致。闭式对文献/解析的既有误差（Timoshenko <0.06%、
Paik <6.3%）已在主交付文档逐项 <3%/<15% 判定。

## 7. 复现

```bash
PYTHONPATH=src python -m tensorfem.grillage_ice_fe_crosscheck   # 写 fe_crosscheck.json
PYTHONPATH=src python -m pytest tests/test_grillage_ice_fe_crosscheck.py -q   # 6 passed
PYTHONPATH=src python gen_fe_crosscheck_figure.py               # 生成图
```

产物：`outputs/grillage_ice_fe_crosscheck/fe_crosscheck.json`（机读）、
`docs/assets/grillage_ice_fe_crosscheck/{fig1,fig2}.png`。
