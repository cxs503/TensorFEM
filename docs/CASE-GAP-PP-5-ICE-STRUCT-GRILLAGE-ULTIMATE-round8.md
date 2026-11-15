# 加筋板架极限强度：恢复 0.9 级板格余量（轮次 8）

> 锚点 `GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE`。本文闭合 `docs/CASE-GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE-round7.md` §5.7 的**遗留项**：
> round 7 修正板/腹板接合面（腹板自板下表面起算）后，构件控制算例的板格在构件极限处的利用率升到 0.911（light 工况），
> 使 `tests/test_grillage_ultimate_independent_audit.py` 的回归锁从 0.9 放宽到 0.95——判据收紧面变松。
> round 7 明确记录「仍需后续用更厚的板重设算例以恢复 0.9 级余量」。本轮的结论：**把构件控制板架的板厚从 25 mm 重设为 26 mm（+1 mm），
> 三档组合工况的板格利用率全部回到 < 0.9（0.838 / 0.759 / 0.719），回归锁收紧回 0.9，且纵骨仍为控制构件。**

## 1. 案例介绍

**物理背景**：冰级船体加筋板架的构件极限（纵骨梁柱）由带有效带板的 T 型截面驱动。round 7 修正了
板/腹板接合面的几何约定（腹板焊在板下表面，不再重复计面积），使纵骨更强、构件控制 p_ult 上移，
板格在构件极限处的利用率从 0.77 级升到 0.911（light 工况）。这使「板格不控制」的回归锁从 0.9 放宽到 0.95——
判据本意是「板格不控制」（1.0 才是板格控制），0.9 只是当时的余量经验值。本轮的增量是**重设构件控制板架的板厚**，
用更厚的板把板格余量恢复到 0.9 级，同时保持纵骨为控制构件。

**几何/材料（文档算例，非规范摘录）**

| 参数 | round 7（旧） | round 8（新） |
| --- | --- | --- |
| frame_spacing | 2.4 m | 2.4 m |
| bay_spacing | 0.55 m | 0.55 m |
| n_bays / n_spans | 4 / 4 | 4 / 4 |
| **plate_thickness** | **25 mm** | **26 mm（+1 mm）** |
| longitudinal stiffener | T160×6 + 50×7 | T160×6 + 50×7 |
| transverse frame | T550×10 + 120×14 | T550×10 + 120×14 |
| yield stress | 355 MPa (EH36) | 355 MPa (EH36) |
| E, ν | 206 GPa, 0.3 | 206 GPa, 0.3 |

## 2. 方法

1. `member_governed_geometry()`（`src/tensorfem/grillage_ice_verification.py`）的 `plate_thickness` 从 `0.025` 改为 `0.026`。
   这是**唯一**改动——纵骨/横框架截面、板架布局、材料全部不动。板格局部弯曲容量 `m_p = σ_y·t²/4` 随 t² 增大，
   在相同的构件极限 p_ult 下板格利用率下降；同时更厚的板增大纵骨有效截面（更多附着板面积/惯性），
   使纵骨极限应力略升、p_ult 略降（见 §3）。
2. `tests/test_grillage_ultimate_independent_audit.py::test_member_governed_branch_is_exercised` 的回归锁
   从 `0.95` 收紧回 `0.9`（判据本意「板格不控制」不变，0.9 是恢复的余量经验值）。
3. 独立校核链（`grillage_ice_verification` T1–T13、`grillage_plastic_verification` V1–V15）与
   `hand_stiffener_limit` 闭式代数链同步到同一几何（它们与模块共享的唯一约定是板/腹板接合面，方法本身仍独立）。
4. 跑既有验证链确认无回归：`grillage_ice_verification`（40 门）与 `grillage_plastic_verification`（33 门）。

## 3. 计算结果

### 3.1 三档组合工况：round 7（t=25 mm）vs round 8（t=26 mm）

| 工况 | p_ult r7 [MPa] | p_ult r8 [MPa] | Δp_ult [%] | U_plate r7 | U_plate r8 | U_stiff r8 | U_frame r8 | 控制构件 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| light ice + ballast head | 1.9254 | 1.9149 | −0.55 | 0.911 | **0.838** | 1.000 | 0.066 | longitudinal_stiffener |
| design ice + design head | 1.2219 | 1.2042 | −1.45 | 0.832 | **0.759** | 1.000 | 0.093 | longitudinal_stiffener |
| severe ice + tank head | 0.9918 | 0.9627 | −2.94 | 0.797 | **0.719** | 1.000 | 0.119 | longitudinal_stiffener |

**关键结论**：三档 U_plate 全部 < 0.9（0.838 / 0.759 / 0.719），回归锁可收紧回 0.9；纵骨仍为控制构件（U_stiff=1.000）。
p_ult 变化 −0.55…−2.94 %（更厚的板使纵骨有效截面略增、极限应力略升，p_ult 略降——方向与物理一致）。

### 3.2 板厚扫描（light 工况，U_plate 与 p_ult 随 t）

| t [mm] | U_plate | p_ult [MPa] | 控制构件 |
| --- | --- | --- | --- |
| 25 | 0.911 | 1.9254 | longitudinal_stiffener |
| **26** | **0.838** | **1.9149** | longitudinal_stiffener |
| 27 | 0.774 | 1.9058 | longitudinal_stiffener |
| 28 | 0.716 | 1.8982 | longitudinal_stiffener |
| 29 | 0.666 | 1.8920 | longitudinal_stiffener |
| 30 | 0.621 | 1.8872 | longitudinal_stiffener |

### 3.3 既有验证链

- `grillage_ice_verification`：40 门，失败 0 条（T1–T13 含 T11 独立 FD、T12 手算闭式）。
- `grillage_plastic_verification`：33 门，失败 0 条（V1–V15）。
- `pytest tests/test_grillage_ultimate_independent_audit.py`：10 passed（含收紧后的 0.9 回归锁）。

## 4. 结果图

![margin](../assets/grillage_section_junction/fig_round8_margin.png)
*图 1：板格利用率（构件极限处）随板厚变化。红色虚线为 round 8 恢复的 0.9 回归锁，橙色点线为 round 7 放宽的 0.95；
绿色竖线为 t=26 mm（round 8 重设）。*

![ultimate](../assets/grillage_section_junction/fig_round8_ultimate.png)
*图 2：构件控制极限冰压随板厚变化（更厚的板使 p_ult 略降，方向与物理一致）。*

## 5. 注意点

1. **这是「重设算例」，不是「调参」**：round 7 的接合面修正是几何约定（物理上必须），本轮的板厚重设是
   恢复回归锁余量的**算例级**改动。板厚 +1 mm 是工程上合理的冰区板厚增量（25→26 mm），不改变任何物理模型。
2. **round 7 的 §3.3 表（1.9254 / 1.2219 / 0.9918 MPa）是 t=25 mm 口径**：本文的 §3.1 给出 t=26 mm 的新值
   （1.9149 / 1.2042 / 0.9627 MPa）。引用构件控制 p_ult 时须以本文（当前默认几何）为准；round 7 表保留为
   接合面修正的历史记录。
3. **回归锁收紧回 0.9**：`test_member_governed_branch_is_exercised` 的 `utilisation_plate_at_limit < 0.9`。
   判据本意「板格不控制」（1.0 才是板格控制）不变；0.9 是恢复的余量经验值。若未来再改几何使板格利用率
   重新逼近 1.0，该锁会立即失败。
4. **未做**（不得当作通过）：焊接 residual stress 与初始挠度对接合面细节的影响；腹板局部屈曲/倾覆
   （在 `stiffener_tripping` 单独建模）；规范（CSR/IACS）条文对照；模型试验数据对比（沙箱无外部算例）。
5. **对既有 21 门链的影响面**：`plate_thickness` 改动只经由 `member_governed_geometry` 进入 T12/T13 与
   `member_governed_case_table`；T1–T11 用 case-study 几何，不受影响。差分表给出的属性变化是唯一影响路径。

## 6. 误差判定

| 关键量 | 参照 | 误差 | < 3 % ? |
| --- | --- | --- | --- |
| 构件控制 p_ult（3 工况，t=26 mm） | `hand_stiffener_limit` 闭式代数（同几何、独立方法） | ≤4.8e-08（§3.1 全绿） | 是 |
| 板格利用率（3 工况） | 自比较（同一求解器，t=25 vs 26） | 0.911→0.838（−8.0 %） | 差异本身即结论（恢复余量） |
| 构件控制 p_ult（3 工况） | 自比较（t=25 vs 26） | −0.55…−2.94 % | 是（< 3 %，方向与物理一致） |
| 手算闭式链 vs 模块（T12，3 工况） | 显式代数（同约定、独立方法） | ≤4.0e-5 %（§3.3 全绿） | 是 |

**总体判定**：构件控制 p_ult 对闭式代数的误差 ≤ 4.8e-08（全部 < 3 %）；板厚重设使板格利用率从 0.911 恢复到
0.838（< 0.9），回归锁收紧回 0.9，纵骨仍为控制构件。**没有 >15 % 的未解释偏差**；round 7 的「仍需后续用更厚的板
重设算例以恢复 0.9 级余量」遗留项已闭合。

## 7. 结论与复现

* **round 7 遗留项闭合**：板厚 25→26 mm 使三档板格利用率全部 < 0.9（0.838 / 0.759 / 0.719），回归锁收紧回 0.9。
* **纵骨仍为控制构件**：U_stiff=1.000，U_plate=0.719–0.838，U_frame=0.066–0.119。
* **p_ult 变化有界且方向保守**：−0.55…−2.94 %（更厚的板使 p_ult 略降）。

```
# 分段执行（每段都在控制面短测试预算内）
PYTHONPATH=src python -m pytest tests/test_grillage_ultimate_independent_audit.py -q
PYTHONPATH=src python -m tensorfem.grillage_ice_verification
PYTHONPATH=src python -m tensorfem.grillage_plastic_verification
```

---
*Generated by `benchmarks/grillage_section_junction.py`（round 8 段）— 文中数字全部由脚本注入。*
*Anchor: GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE, Round 8（0.9 级余量恢复）*
