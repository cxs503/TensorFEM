# 加筋板架极限强度：板/腹板接合面修正（轮次 7）

> 锚点 `GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE`。本文闭合独立复核 `docs/AUDIT-GAP-PP-5-GRILLAGE-ULTIMATE-independent-review.md` 的 **F5** 项：
> 构件截面把腹板根部放在**板中面**（`z = 0`），使腹板上半 `t_p/2` 与板条重叠并被重复计入。F5 当时把它判为「低、非阻断」，并估计影响「A ≈ 0.5 %、I/形心 < 1 %」，因「改动会挪动已被引用的历史数字」而未改。本轮的结论是：**该估计偏低一个量级**——对 `I` 的实际影响达 19.01 %，对 `Z_p` 达 7.47 %。

## 1. 案例介绍

**物理背景**：冰级船体加筋板架的构件极限（板格弯曲、纵骨梁柱、横框架塑性倒塌）都由带有效带板的 T 型截面驱动。截面的力学定义里有一个不属于「材料/载荷假设」、而属于**几何约定**的自由度：腹板根部从板的哪一位置起算。焊接 T 型材的腹板焊在板的下表面，因此几何上是 `z = -t_p/2`；把根部放在板中面则让腹板上 `t_p/2` 那段落在板条内部，同一块材料被数了两次（面积重叠）同时也是把构件做矮了 `t_p/2`。

**几何/材料（文档算例，非规范摘录）**

| 参数 | 值 |
| --- | --- |
| frame_spacing a | 2.1 m |
| bay_spacing b | 0.35 m |
| plate thickness t_p | 20 mm |
| longitudinal stiffener | T250×10 + 80×12 |
| attached plate width | 350 mm |
| yield stress | 355 MPa (EH36) |
| E, ν | 206 GPa, 0.3 |

被比较的两个约定：

| 约定 | 腹板根部 z | 截面总深 | 重叠面积 |
| --- | --- | --- | --- |
| `plate_mid`（历史默认） | `0` | `t_p/2 + h + t_f` | `t_w·t_p/2` |
| `plate_bottom`（本轮默认） | `-t_p/2` | `t_p + h + t_f` | `0` |

## 2. 方法

1. `MemberSection` 新增字段 `web_origin ∈ {plate_bottom, plate_mid}`，默认 `plate_bottom`；`web_start_z` 统一给出腹板根部坐标，`_parts()`（弹性 A/形心/I/Z_e）与 `plastic_modulus`（刚塑性等分轴与 Z_p 的 100 条带求和）都改为从 `web_start_z` 起算。`double_counted_area` 显式报告重叠面积（`plate_bottom` 下恒为 0）。`GrillageGeometry.with_plate()` 传递该约定，避免截面拼装时回落到旧行为。
2. `web_origin="plate_mid"` 保留为**仅用于复现历史数字**的选项，本轮给出的差分表就是靠它一次算两遍；默认路径已无重叠。
3. 独立校核链同步到同一几何约定（这是它与模块唯一共享的约定，方法本身仍独立）：`grillage_ice_verification._effective_section`（T12 的显式代数手算链）。它原先硬编码 `-h/2`，若不改，T12 会以 ~17 % 的假误差失败——第一版运行即如此，说明该门确实在比对「截面代数」，不是空门。
4. 两条**非模块**参照：
   - R1：本脚本 `slice_properties()`，用 100000 条 1e-6 级水平薄片从原始尺寸重建截面，数值积分求 A、形心、I，二分求等分轴后求和求 Z_p；不使用 `MemberSection` 的任何属性。
   - R2：`ice_belt_collapse.tee_section()`——同一生态里独立实现的 T 型截面例程，本来就写成「腹板自板底起算」，与 `plate_bottom` 约定应当逐位一致。
5. 跑既有验证链确认无回归：`grillage_ice_verification`（T1–T13，40 条门）与 `grillage_plastic_verification`（V1–V15，33 条门）。

## 3. 计算结果

### 3.1 截面属性：历史 vs 修正

| 板架 | 构件 | A_eff [cm²] 修正/历史 | I [cm⁴] 修正/历史 | Z_p [cm³] 修正/历史 | ΔI [%] | ΔZ_p [%] |
| --- | --- | --- | --- | --- | --- | --- |
| case-study | longitudinal stiffener | 104.600 / 103.600 | 9315.14 / 8544.79 | 619.31 / 584.95 | +9.02 | +5.87 |
| case-study | transverse frame | 295.800 / 294.800 | 91156.31 / 87393.96 | 2620.55 / 2548.85 | +4.31 | +2.81 |
| member-governed | longitudinal stiffener | 150.600 / 149.850 | 2031.92 / 1707.31 | 235.56 / 219.19 | +19.01 | +7.47 |
| member-governed | transverse frame | 291.800 / 290.550 | 91855.30 / 87156.10 | 2660.86 / 2571.28 | +5.39 | +3.48 |
| slender | longitudinal stiffener | 82.500 / 81.915 | 4103.90 / 3817.40 | 334.41 / 318.88 | +7.51 | +4.87 |
| slender | transverse frame | 202.400 / 201.750 | 57063.69 / 55370.74 | 1934.23 / 1894.01 | +3.06 | +2.12 |

### 3.2 对抗独立参照（本轮的精度证据）

| 板架 | 构件 | A vs R1 [%] | I vs R1 [%] | Z_p vs R1 [%] | 形心 vs R1 [%] | A vs R2 [%] | I vs R2 [%] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| case-study | longitudinal stiffener | +2.42e-03 | +3.24e-03 | +2.75e-03 | +6.61e-04 | +0.00e+00 | +0.00e+00 |
| case-study | transverse frame | -6.94e-03 | +2.76e-03 | +2.98e-03 | +9.77e-03 | +0.00e+00 | +2.22e-14 |
| member-governed | longitudinal stiffener | -1.25e-03 | -1.84e-03 | -1.86e-03 | -1.81e-03 | +0.00e+00 | -2.22e-14 |
| member-governed | transverse frame | +8.27e-03 | +5.71e-05 | -4.42e-04 | -8.34e-03 | +0.00e+00 | +2.22e-14 |
| slender | longitudinal stiffener | -4.39e-03 | +1.38e-03 | +1.66e-03 | +5.83e-03 | +0.00e+00 | +0.00e+00 |
| slender | transverse frame | +4.62e-03 | -2.60e-03 | -3.31e-03 | -7.86e-03 | +0.00e+00 | +2.22e-14 |

### 3.3 对交付量（极限冰压）的影响

| 板架 | 工况 | p_ult 历史 [MPa] | p_ult 修正 [MPa] | Δ [%] | 控制构件（修正） |
| --- | --- | --- | --- | --- | --- |
| case-study | light-ice + ballast head | 3.3546 | 3.3546 | +0.00 | plate_local_bending |
| case-study | design-ice + design head | 2.3298 | 2.3298 | +0.00 | plate_local_bending |
| case-study | severe-ice + tank head | 1.9688 | 1.9688 | +0.00 | plate_local_bending |
| member-governed | light ice + ballast head | 1.6253 | 1.9254 | +18.47 | longitudinal_stiffener |
| member-governed | design ice + design head | 0.9979 | 1.2219 | +22.44 | longitudinal_stiffener |
| member-governed | severe ice + tank head | 0.7477 | 0.9918 | +32.66 | longitudinal_stiffener |

### 3.4 既有验证链

- `grillage_ice_verification`：40 条门，失败 0 条（T1–T13 含 T11 独立 FD、T12 手算闭式）。
- `grillage_plastic_verification`：33 条门，失败 0 条（V1–V15）。
- `pytest tests/test_grillage_*.py tests/test_stiffener_tripping.py`：tests/test_grillage_ice_ultimate.py + tests/test_grillage_section_junction.py: 26 passed (93.7 s); tests/test_grillage_plastic_collapse.py: 10 passed; grillage_ice_verification 40/40 gates; grillage_plastic_verification 33/33 gates

## 4. 结果图

![junction](../assets/grillage_section_junction/fig1_junction.png)
*图 1：板/腹板接合面的两种约定（左：历史，腹板自板中面起算，红色斜网为被重复计入的重叠区；右：修正，腹板焊在板下表面）。虚线为组合形心。*

![properties](../assets/grillage_section_junction/fig2_property_change.png)
*图 2：三种板架 × 两种构件（6 个真实截面）的属性相对变化。*

![ultimate](../assets/grillage_section_junction/fig3_ultimate_shift.png)
*图 3：修正对交付量（极限冰压）的影响，柱顶标注相对变化。*

## 5. 注意点

1. **这不是「调参」，是修正几何**。修正后构件更深 `t_p/2`、且不再重复计面积，所以板格控制的工况 p_ult **不动**（板格极限 `σ_y t²/4` 不含构件截面），构件控制的工况 p_ult 上移——方向与物理一致（历史值偏保守）。
2. **历史数字的去向**：默认口径改了，round2/3/5 与独立复核文档里的**构件相关**数字属 `plate_mid` 口径。其中被引用最多的独立复核 T12 三工况 1.6253 / 0.9979 / 0.7477 MPa 现为 **1.9254 / 1.2219 / 0.9918 MPa**（+18.5 / +22.4 / +32.7 %，本表 §3.3），差幅已超过 15 % 精度线，**不得再按旧值引用**；该表已就地加注。round6 的对照量是板格/Timoshenko/Paik-Thayamballi 口径，不经过 `MemberSection`，因此不需要改（已加注说明）。`docs/grillage_ice_ultimate/*.json` 是 round-6 之前的快照（21 门、无 member-governed 段），本文不复用它，当前口径以 §3.4 的 40 门/33 门为准。
3. **Paik-Thayamballi 门（T7，15 % 门）** 走的是 `stiffened_panel_ultimate` 的独立板格模型，不经过 `MemberSection`，因此不受影响；它在 T 链里仍报 −6.3 %。
4. **板/腹板重叠的另一种口径**（把板条宽度扣掉腹板厚 `t_w`，即「有效带板扣除腹板附着宽度」）没有采用：现行链里 `plate_width` 是带板宽度，工程惯例取轴线间距,不再二次扣减；这一项的量级是 `t_w/b`（此处 2.9 %），远小于本次修正的 I 变化，且会改变带板定义本身，留给后续单独评审。
5. **未做**（不得当作通过）：焊接 residual stress 与初始挠度对接合面细节的影响；腹板局部屈曲/倾覆（在 `stiffener_tripping` 单独建模，本轮仅确认其测试仍绿）；规范（CSR/IACS）条文对照；模型试验数据对比（沙箱无外部算例）。
7. **回归锁随之调整一处**：`tests/test_grillage_ultimate_independent_audit.py` 原先要求构件控制算例的板格利用率 < 0.9；修正后纵骨更强、p_ult 上移，板格在构件极限处的利用率升到 0.911（light 工况），故该上界放宽到 0.95——判据本意仍是「板格不控制」（1.0 才是板格控制），0.9 只是当时的余量经验值。这是一处**判据收紧面变松**的改动，如实记录，仍需后续用更厚的板重设算例以恢复 0.9 级余量。
6. **对既有 21 门链的影响面**：`web_origin` 默认改动只经由 `MemberSection` 属性进入，`GrillageGeometry.validate()` 与所有载荷/分配逻辑未改；差分表给出的属性变化是唯一的影响路径。

## 6. 误差判定

| 关键量 | 参照 | 误差 | < 3 % ? |
| --- | --- | --- | --- |
| 修正后 A（6 截面） | R1 薄片数值积分 | ≤9.77e-03 % | 是 |
| 修正后 I（6 截面） | R1 薄片数值积分 | ≤9.77e-03 % | 是 |
| 修正后 Z_p（6 截面） | R1 薄片数值积分 | ≤9.77e-03 % | 是 |
| 修正后 A/I/形心（去板的裸 T 材） | R2 `ice_belt_collapse.tee_section` | ≤2.22e-14 % | 是 |
| 构件控制 p_ult（3 工况） | 修正前/后自比较（同一求解器） | ≤32.66 % 差异 | 差异本身即结论 |
| 板格控制 p_ult（3 工况） | 修正前/后自比较 | ≤0.00e+00 % | 是（不受影响，符合预期） |
| 手算闭式链 vs 模块（T12，3 工况） | 显式代数（同约定、独立方法） | ≤4.0e-5 %（§3.4 全绿） | 是 |

**总体判定**：截面属性对两条非同源参照的误差 ≤ 9.77e-03 %（R1）/ ≤ 2.22e-14 %（R2），全部 < 3 %；构件属性本身的修正量为 ΔI 最大 19.01 %、ΔZ_p 最大 7.47 %，对应极限冰压最大变化 32.66 %。**没有 >15 % 的未解释偏差**；F5 原先的「<1 %」估计被本轮否定（见 §7）。

## 7. 结论与复现

* **F5 估计被否定**：属性影响最大 19.01 %（I），不是 < 1 %；修正是必要的，而不是「低、非阻断」的记账项。
* **默认口径已改为物理约定**（腹板焊在板下表面），`plate_mid` 仅用于复现历史数字。
* **交付量影响有界且方向保守**：板格控制工况不变，构件控制工况 +32.66 % 以内。

```
# 分段执行（每段都在控制面短测试预算内）
for st in sections cases gates-ice gates-plastic report; do
  PYTHONPATH=src python benchmarks/grillage_section_junction.py $st
done
# 或一次性：PYTHONPATH=src python benchmarks/grillage_section_junction.py all
PYTHONPATH=src python -m pytest tests/test_grillage_section_junction.py -q
PYTHONPATH=src python -m pytest tests/test_grillage_ice_ultimate.py tests/test_grillage_plastic_collapse.py -q
```

---
*Generated by `benchmarks/grillage_section_junction.py` — 文中数字全部由脚本注入。*
*Anchor: GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE, Round 7（F5 closure）*
