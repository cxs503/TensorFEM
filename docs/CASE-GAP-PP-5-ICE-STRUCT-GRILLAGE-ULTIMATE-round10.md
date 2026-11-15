# GAP-PP-5 ICE-STRUCT 板架极限 · Round 10：纵骨屈曲失稳（tripping + 腹板局部屈曲）集成

> 锚点 `GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE`，Round 10。
> 模块 `src/tensorfem/grillage_ice_instability.py`（新增 200 行），
> 门禁 `tests/test_grillage_ice_instability.py`（T14–T17，8 项全绿），
> 产物 `docs/grillage_ice_ultimate/instability_cases.json`（机读）。

## 0. 为什么还要做这一轮

Round 1–9 已把锚点的三条验收口径做实（板架模型、三档组合工况、
板格应力→极限载荷映射表 80 行机读），独立复核处置 approve with notes，
数值保真度到 0.02 %。但复核的 **not_run** 项里有一条是本锚点物理上的
关键：纵骨 web local buckling / tripping 未集成到板架极限评估中。
`grillage_ice_ultimate` 模块自身文档声明"stiffener web local buckling /
tripping are not modelled in this module"，而 `stiffener_tripping` 模块
已独立验证（T14a–c，25 项全绿）但未接入板架链。

本轮闭合此缺口：**将 tripping + 腹板局部屈曲作为第四极限状态集成**，
不修改已验证的三支函数（保留 40 gates 的出版数字），新增独立函数
`grillage_utilisations_with_instability` 和 `ultimate_ice_pressure_with_instability`。

## 1. 方法

### 1.1 第四极限状态：纵骨屈曲失稳

纵骨在面内压缩 + 侧向冰压弯矩下的总应力：

```
sigma_total = sigma_axial + sigma_bending
```

其中 `sigma_bending` 已含 secant 放大 `beta_m`（来自 `grillage_utilisations`）。

极限应力取 `stiffener_instability_check` 的下界：

- **tripping**：`k_theta = 0`（板格完全屈服/无约束），Ritz 闭式 `sigma_cr`
- **web local buckling**：`k = 4`（两边简支），`sigma_w = 4 pi^2 D_w / (t_w h_w^2)`

取小值作为 `sigma_instability`，利用率 `u = sigma_total / sigma_instability`。

### 1.2 集成方式

新增模块 `grillage_ice_instability.py`，**不修改** `grillage_ice_ultimate.py`
的任何已有函数：

| 函数 | 作用 |
|---|---|
| `stiffener_instability_utilisation` | 计算纵骨失稳利用率 |
| `grillage_utilisations_with_instability` | 调用已有 `grillage_utilisations` + 追加失稳键 |
| `ultimate_ice_pressure_with_instability` | 四支二分（含 `stiffener_instability`） |
| `tripping_governed_geometry` | T350×7+70×10 纵骨、0.50 m 间距、2.5 m 框架间距 |

### 1.3 验证门禁

| 门 | 内容 | 判据 |
|---|---|---|
| T14 | 案例几何失稳利用率有限正值、tripping 应力 < σ_y | 通过 |
| T15 | tripping-governed 几何：失稳利用率 > 梁柱利用率 | 通过 |
| T16 | 失稳利用率对冰压单调递增 | 通过 |
| T17 | tripping-governed 几何：含失稳 p_ult < 不含 | 通过 |

## 2. 结果

### 2.1 案例几何（T250×10+80×12）三档工况

| 工况 | σ_x [MPa] | U_plate | U_beam-col | U_instability | tripping σ [MPa] | σ_L [MPa] |
|---|---|---|---|---|---|---|
| light-ice + ballast head | 53 | 0.313 | 0.202 | 0.301 | 222.6 | 331.3 |
| design-ice + design head | 106 | 0.864 | 0.462 | 0.688 | 222.6 | 331.3 |
| severe-ice + tank head | 160 | 1.486 | 0.741 | 1.103 | 222.6 | 331.3 |

**关键发现**：即使案例几何的"stocky"纵骨 T250×10+80×12，tripping 应力
（223 MPa）也低于梁柱极限应力（331 MPa）。在 severe-ice 工况下，
失稳利用率已达 1.10——纵骨在板格到达极限之前就已 tripping。
这意味着 **tripping 检查对典型冰带纵骨是相关的，不应省略**。

### 2.2 Tripping-governed 几何（T350×7+70×10）三档工况

| 工况 | σ_x [MPa] | p_ult (无失稳) [MPa] | p_ult (含失稳) [MPa] | 降幅 | 控制构件(含) |
|---|---|---|---|---|---|
| light-ice + ballast head | 50 | 1.288 | 1.288 | 0 % | plate_local_bending |
| design-ice + design head | 100 | 0.864 | 0.000 | 100 % | stiffener_instability |
| severe-ice + tank head | 150 | 0.702 | 0.000 | 100 % | stiffener_instability |

在 σ_x ≥ 100 MPa 时，纵骨在零冰压下已超过 tripping 极限（u > 1），
p_ult = 0。这表明 **深薄腹板纵骨在高轴压下完全由 tripping 控制**，
梁柱检查严重高估承载力。

## 3. 误差判定

| 关键量 | 参照 | 误差 | < 3 % ? |
|---|---|---|---|
| tripping 应力（k_theta=0） | Ritz 闭式 vs stiffener_tripping 模块 | 0（同一代码） | 是 |
| web buckling 应力（k=4） | 闭式 `4 pi^2 D / (t h^2)` | 0（同一代码） | 是 |
| 总应力 sigma_total | grillage_utilisations 输出 | 0（同一代码） | 是 |
| 失稳利用率 vs 手算 | sigma_total / sigma_instability | 0（代数恒等） | 是 |

本模块不引入新的数值近似：所有物理量来自已验证的
`grillage_ice_ultimate` 和 `stiffener_tripping` 模块，
集成层只做代数组合。

## 4. 注意点 / 局限

- **tripping 应力取下界**（k_theta = 0）：对有板格约束的实际结构偏保守；
  上界（k_theta = 8D/b_s）给出的 tripping 应力更高，但单模态 Ritz 对
  有约束端是上界（非保守），故下界是安全侧。
- **弹性分叉**：tripping 和腹板局部屈曲都是弹性分叉应力，
  不含塑性交互或后屈曲储备。
- **不修改已有 40 gates**：`grillage_ice_ultimate` 的三支函数完全不动，
  新增函数是独立的第四支。
- **案例几何的 tripping 应力低于预期**：T250×10+80×12 的 tripping 应力
  仅 223 MPa（vs σ_y = 355 MPa），这是因为开截面 Tee 的 St Venant
  常数 J 很小（~1.7e-7 m^4），而极惯性半径 I_p/A + d^2 较大。
  这不是代码错误，是物理事实。

## 5. 复现

```bash
PYTHONPATH=src python -m pytest tests/test_grillage_ice_instability.py -q
PYTHONPATH=src python benchmarks/grillage_instability_benchmark.py
```

---
*Anchor: GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE, Round 10（tripping + web buckling 集成）*
*machine-readable: `docs/grillage_ice_ultimate/instability_cases.json`*
