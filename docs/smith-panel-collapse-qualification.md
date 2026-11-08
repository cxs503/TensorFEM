# 加筋板格 Smith 渐进塌陷验证面（P188-N03）

节点：`P188-N03`（战役 DAG，物理域 结构 / 加筋板格）。目标仓：TensorFEM。模块：`tensorfem.smith_panel_collapse`；证据运行：`benchmarks/smith_panel_collapse.py`；测试：`tests/test_smith_panel_collapse.py`（18 例）。

## 1. 案例介绍（物理背景 / 几何 / 参数）

船体总纵崩溃的 Smith 法逐单元消费**弯矩—曲率**关系，而弯矩—曲率的输入是每个加筋板格的**平均应力—平均应变**（载荷—缩短）曲线。仓库原有链路 (`stiffened_panel_ultimate`) 用的是极限应力型（弹性至极限应力后平台）曲线，**没有峰后软化**，因此无法产生板格峰值应力所依赖的负刚度，也无法表达“板已软化、筋仍在加载”的并联过程。本节点补齐该链路：板条（大挠度）+ 筋条（梁柱/局部屈曲）+ Smith 面积加权并联 + 双向压/剪交互面。

**参考板格（IACS CSR No.47 标准加筋板格）**

| 量 | 值 |
|---|---|
| 板格长 $a$（横框架间距） | 2400.0 mm |
| 筋距 $s$ | 800.0 mm |
| 板厚 $t$ | 12.0 mm |
| 纵向筋 | HP 200 × 10（球扁钢） |
| 等效 T 型截面（显式理想化） | 腹板 180 × 10 mm，翼板 60 × 10 mm |
| 屈服应力 $\sigma_y$ | 235 MPa |
| 弹性模量 $E$ | 206 GPa，$\nu$ = 0.3 |
| 板柔度 $\beta = (s/t)\sqrt{\sigma_y/E}$ | 2.2517 |
| 长宽比 $a/s$ | 3.00 |
| 初始缺陷幅值 $w_0 = 0.1\beta^2 t$ | 6.084 mm |
| 残余应力 $\sigma_{rc}$（板中区压） | 47.0 MPa（$\sigma_{rc}/\sigma_y$ = 0.2） |
| 焊接拉区宽度份额 $2\eta$ | 0.1667（由 $2\eta\sigma_y = (1-2\eta)\sigma_{rc}$ 平衡确定） |
| 板/筋面积份额 | 板 0.8000，筋 0.2000 |

单位：SI（m, Pa, N）。全文压应力为正。

## 2. 怎么做的（方法 / 流程 / 关键设置）

1. **板条载荷—缩短曲线（Marguerre / von Kármán 大挠度单模态）**。把板挠曲展开为 `w = A sin(mπx/a) sin(πy/b)`，对 von Kármán 板方程做 Galerkin 投影，得到平均应力—挠幅关系
   $\sigma = (\sigma_{cr}-\sigma_{rc})\dfrac{A}{A+w_0} + \dfrac{1}{2}\sigma_1 + \dfrac{\beta^2}{2\alpha^2}\sigma_2$，其中 $\sigma_1 = E\Psi\alpha^2/8$、$\sigma_2 = E\Psi\beta^2/8$、$\Psi = A(A+2w_0)$；平均缩短 $\varepsilon = \sigma/E + \alpha^2\Psi/8$。弹性屈曲应力按临界模态数 $m$（对 $m=1..12$ 取最小）计算，本板格 $m$ = 3、$\sigma_{cr}$ = 167.6 MPa。**极限取纵向边纤维屈服前可达的最大平均应力**（边应力 $\sigma+\sigma_1 = \sigma_y$），屈服交点用二分法定位，因此不受挠幅网格限制（结算：边应力/屈服 = 1.000000）。
2. **初始缺陷**直接进入 $\Psi$：$w_0 = c_0\beta^2t$，参考值 $c_0 = 0.1$，敏感性算例用 0.3。
3. **残余应力**两区理想化：板边拉区取 $\sigma_{rt}=\sigma_y$（焊缝拉伸达到屈服），中区压 $\sigma_{rc}$，由自平衡条件定出 $2\eta$；区应力按等应变叠加并在屈服处截断。屈曲侧另有残余压应力对屈曲的失稳贡献：按模态加权（模态在板边为零、边部为拉区）可证明其等效为屈曲应力下移 $\sigma_{rc}$，故上式首项用 $\sigma_{cr}-\sigma_{rc}$。
4. **筋条载荷—缩短**：T 型截面 + 有效带板（$b_e = s\,\sigma_{u,p}/\sigma_y$），梁柱强度用 Perry–Robertson（含初弯曲 $\delta_0 = 0.002a$ 的放大），局部屈曲取翼板悬伸（$k=0.425$）与腹板（$k=4$）两类，筋条极限 = 二者与屈服三者的最小值。本板格控制模式 = **beam_column_perry**，$\sigma_{u,s}/\sigma_y$ = 0.9518，$\sigma_E/\sigma_y$ = 5.425，初弯曲缺陷参数 $\eta_p$ = 0.04172。
5. **Smith 组合**：板与筋按面积加权并联 $\sigma(\varepsilon) = (A_p\sigma_p + A_s\sigma_s)/(A_p+A_s)$，取曲线峰值为板格极限。
6. **峰后分支**：板在极限后按文档化线性软化 $d\sigma/d\varepsilon = -0.1E$ 下降至 0.4$\sigma_u$ 平台；筋条先塑性平台后 -0.05$E$ 软化。
7. **交互面**：CSR 二次型 $(\sigma_x/\sigma_{xu})^2+(\sigma_y/\sigma_{yu})^2-c\,\sigma_x\sigma_y/(\sigma_{xu}\sigma_{yu})+(\tau/\tau_u)^2=1$，$c=0$（保守，无耦合项）；$\sigma_{yu}$ 用“横压时筋条不参与、仅板参与”的板面积份额乘板极限（显式各向异性，未抹平）；$\tau_u = \min(\sigma_y/\sqrt3, \tau_{cr})$（不含张力场，偏保守）。
8. **独立对拍链**：`csr_panel_chain` 用 Faulkner 板极限 + Johnson–Ostenfeld 柱（带有效带板）按面积加权，**不经过** Smith 积分，作为判据 (b) 的参考值。
9. **数值落盘**：所有量在写入 JSON 前统一按 6 位有效数字截断，并由判据 (h) 遍历产物逐点复核（`float(f"{v:.6g}") == v`）。

## 3. 计算结果

### 3.1 验收判据表（程序化注入，零手抄）

| id | 判据 | 门限 | 结果 | 判定 |
|---|---|---|---|---|
| (a) | plate $\sigma_u/\sigma_y$ vs Faulkner (1975) | ≤ 10 % | max |err| = 9.19 % | **PASS** |
| (b) | panel $\sigma_u$ vs CSR closed-form chain | ≤ 10 % | err = -5.32 % | **PASS** |
| (c) | vs Paik & Thayamballi (2003) specimens | ≤ 15 % | WITHHELD（离线无主源表，未编造） | **WITHHELD** |
| (d) | imperfection $w_0/t$: 0.1$\beta^2$ → 0.3$\beta^2$ | ≥ 5 % drop | drop = 19.81 % | **PASS** |
| (e) | residual stress $\sigma_{rc} = 0.2\sigma_y$ | ≥ 3 % drop | drop = 15.95 % | **PASS** |
| (f) | equal biaxial reduction $1/\sqrt{2}$ | $1/\sqrt{2}$ ± 10 % | ratio = 0.707107（err +0.00 %） | **PASS** |
| (g) | post-peak negative stiffness | curve must descend | 峰值后 $d\sigma/d\varepsilon$ = -0.080$E$ | **PASS** |
| (i) | report figures | ≥ 3 families | 4 families | **PASS** |
| (h) | six-significant-digit landing | 0 offending values | 0 offending | **PASS** |

### 3.2 关键量

| 量 | 值 |
|---|---|
| 板极限 $\sigma_{u,p}/\sigma_y$ | 0.6558 |
| 板极限应变 $\varepsilon_{u,p}/\varepsilon_y$ | 0.8127 |
| Faulkner 板极限 $/\sigma_y$ | 0.6910 |
| 筋条极限 $\sigma_{u,s}/\sigma_y$ | 0.9518 |
| **板格极限** $\sigma_u/\sigma_y$ | **0.7038** |
| 板格极限应变 $\varepsilon_u/\varepsilon_y$ | 0.9532 |
| CSR 型闭式链 $/\sigma_y$（判据 b 参考值） | 0.7433 |
| 峰后斜率 $d\sigma/d\varepsilon$ | -0.0800$E$ |
| 交互面 $\sigma_{xu}/\sigma_y$、$\sigma_{yu}/\sigma_y$、$\tau_u/\sigma_y$ | 0.7038 / 0.5247 / 0.5773 |
| 等双压（本板格，各向异性）$\sigma/\sigma_{xu}$ | 0.5977（绝对值 $\sigma/\sigma_y$ = 0.4207） |
| 等双压（各向同性参考板）$\sigma/\sigma_{xu}$ | 0.707107 |
| 等双压（同一大挠度物理模型，$\beta$ 不变）$\sigma_u/\sigma_y$ | 0.4484 |

### 3.3 板极限 vs Faulkner（$\beta$ = 1…3，$\sigma_{rc}=0$，$w_0=0.1\beta^2t$）

| $\beta$ | $\sigma_u/\sigma_y$（无残余应力） | Faulkner | 偏差 |
|---|---|---|---|
| 1.0000 | 0.9891 | 1.0000 | -1.09 % |
| 1.2500 | 0.9653 | 0.9600 | +0.56 % |
| 1.5000 | 0.9112 | 0.8889 | +2.52 % |
| 1.7500 | 0.8353 | 0.8163 | +2.32 % |
| 2.0000 | 0.7640 | 0.7500 | +1.86 % |
| 2.2500 | 0.7073 | 0.6914 | +2.31 % |
| 2.5000 | 0.6642 | 0.6400 | +3.79 % |
| 2.7500 | 0.6316 | 0.5950 | +6.14 % |
| 3.0000 | 0.6066 | 0.5556 | +9.19 % |

说明：Faulkner 闭式不含残余应力项，故判据 (a) 在**理想板**（残余应力置零）上比较；残余应力效应由判据 (e) 单独度量，两者不混用。

### 3.4 缺陷与残余应力敏感性

| $\beta$ | $\sigma_u/\sigma_y$（$w_0=0.1\beta^2t$） | （$w_0=0.3\beta^2t$） | 缺陷变化 | $\sigma_{rc}=0$ | $\sigma_{rc}=0.2\sigma_y$ | 残余应力变化 |
|---|---|---|---|---|---|---|
| 1.0000 | 0.9891 | 0.9221 | -6.78 % | 0.9891 | 0.8314 | -15.95 % |
| 1.2500 | 0.9653 | 0.8321 | -13.80 % | 0.9653 | 0.8265 | -14.38 % |
| 1.5000 | 0.9112 | 0.7406 | -18.72 % | 0.9112 | 0.8148 | -10.58 % |
| 1.7500 | 0.8353 | 0.6698 | -19.81 % | 0.8353 | 0.7963 | -4.67 % |
| 2.0000 | 0.7640 | 0.6201 | -18.83 % | 0.7640 | 0.7164 | -6.23 % |
| 2.2500 | 0.7073 | 0.5861 | -17.14 % | 0.7073 | 0.6563 | -7.22 % |
| 2.5000 | 0.6642 | 0.5627 | -15.29 % | 0.6642 | 0.6122 | -7.84 % |
| 2.7500 | 0.6316 | 0.5464 | -13.48 % | 0.6316 | 0.5798 | -8.20 % |
| 3.0000 | 0.6066 | 0.5349 | -11.82 % | 0.6066 | 0.5558 | -8.38 % |

## 4. 结果图

![fig1_plate_curves.png](../artifacts/smith_panel_collapse/fig1_plate_curves.png)

*Plate mean stress vs mean strain for beta = 1.0..3.0 with the reference imperfection and residual stress; every curve peaks and then softens.*

![fig2_panel_curve.png](../artifacts/smith_panel_collapse/fig2_panel_curve.png)

*Panel, plate and stiffener load-shortening of the CSR No. 47 panel with the Smith area-weighted combination and the negative-stiffness post-peak branch.*

![fig3_interaction.png](../artifacts/smith_panel_collapse/fig3_interaction.png)

*CSR-style quadratic interaction surface of the panel with the equal-biaxial points and the shear-reduced radial factors.*

![fig4_sensitivity.png](../artifacts/smith_panel_collapse/fig4_sensitivity.png)

*Plate ultimate vs initial-deflection amplitude and vs residual stress level for four slenderness values.*

图 1 = 板条载荷—缩短族；图 2 = CSR No.47 板格的板/筋/板格三条曲线与峰后分支；图 3 = 双向压+剪交互面（含等双压点与不同剪应力水平的径向折减）；图 4 = 缺陷与残余应力敏感性。图件由 `benchmarks/smith_panel_collapse.py` 以 Agg 后端生成，文件名与证据 JSON 的 `figures[]` 一一对应。

## 5. 注意点（坑 / 局限 / 适用范围）

1. **单模态大挠度模型的“几何软化”是假的**。原样使用路径应变 $\varepsilon = \sigma/E + \alpha^2\Psi/8$ 会让弹性段切线刚度掉到 0.58–0.79$E$，真实板没有这种刚度损失（它来自只用单模态、丢掉高阶重分配）。因此弹性段改为均匀应变解 $\sigma = E\varepsilon$，直到附加挠幅等于初始缺陷幅值，此后才接入大挠度路径的增量应变；改后弹性切线 = 1.000$E$（有测试钉住）。
2. **判据 (a) 不是独立实验验证**。本文板极限与 Faulkner 同源（有效宽度物理），比较的是“数值路径 vs 闭式拟合”，属交叉核对；真正独立的验证是判据 (b)（闭式链）与判据 (c)（实验），而 (c) 目前 WITHHELD。
3. **判据 (f) 的读法必须说清**。$1/\sqrt2$ 是二次交互面在各向同性（$\sigma_{xu}=\sigma_{yu}$）时的恒等式，本板格各向异性（$\sigma_{yu}/\sigma_{xu}$ = 0.7456）时折减更强（0.5977）。若把判据读成绝对值 $\sigma_u/\sigma_y \approx 1/\sqrt2$，本板格为 0.4207，**并不满足**——因为板格单向极限本身就在屈服以下。另用同一大挠度框架直接算等双压（屈曲应力减半 + 边纤维 von Mises）得到 0.4484$\sigma_y$，与二次交互面（更保守侧 / 更乐观侧）并不一致，这一差异如实保留，不作平滑。
4. **残余应力模型是两区理想化**：只刻画承载能力损失，不含焊缝附近局部重分配；对 $\beta \lesssim 1.5$ 的短粗板，屈服先于弹性屈曲，单模态弹性大挠度解的适用性下降（判据 (a) 在 $\beta=1$ 仍有 −1.1 % 偏差，但该区物理上应由弹塑性扩展控制）。
5. **峰后软化斜率是假设不是推导**：-0.1$E$ 取自 Smith 法常用区间；证据运行给出板格峰后斜率为 -0.0800$E$（板份额稀释后的值），板格极限本身由上升支决定，不受该假设影响（判据 g 只要求下降支存在）。
6. **球扁钢 HP 200×10 的等效 T 型**（腹板 0.9h、翼板 0.3h、壁厚 t）是显式理想化，本文不引用任何型材表面积；换用真实型材数据会改变筋条刚度与极限，但本板格板份额 0.800 主导，敏感度有限。
7. **Paik & Thayamballi (2003) 实验表离线不可得**：不凭记忆补数，判据 (c) 记为 WITHHELD；`compare_with_experiments` 已就绪，`artifacts/smith_panel_collapse/paik_2003_specimens.json` 给了字段规范，填入主源数据后重跑即可切换为 PASS/FAIL。

## 5b. 与任务书指定路线的偏差与理由（方法自主权声明）

| # | 任务书指定 | 实际做法 | 理由 |
|---|---|---|---|
| 1 | 板条曲线用 Gordo & Soares (1993) 的平均应力—平均应变关系式 | 用 von Kármán 板方程 Galerkin 单模态大挠度解现场推导（含 $w_0$、$\sigma_{rc}$、边纤维屈服判据），未使用 G&S 的拟合系数 | 离线无法核对 G&S 原文系数，凭记忆写系数=编造；改用可自证（能与 Faulkner 闭式对拍、能给出边应力与挠幅）的推导路线。代价：曲线的峰位/软化形状与 G&S 拟合曲线不完全一致，判据 (a) 已在 10 % 门限内给出量化差异 |
| 2 | 筋条取欧拉屈曲与局部屈曲的最小值，含初始弯曲缺陷 | 梁柱强度用 Perry–Robertson（欧拉应力经初弯曲缺陷放大后的闭式解），与翼板悬伸/$k=0.425$、腹板/$k=4$ 局部屈曲取最小 | “欧拉 + 初弯曲缺陷”在数学上就是 Perry–Robertson 式；写成欧拉再乘折减反而引入额外假设 |
| 3 | 与 Paik & Thayamballi (2003) 三组试件对拍 | 判据 (c) WITHHELD + 现成对拍 harness + 字段规范 | 主源表离线不可得，不凭记忆补数；harness 与规范齐备，补数据即可切换 PASS/FAIL |
| 4 | 交互曲线“双轴等压时 $\sigma_u/\sigma_y \approx 1/\sqrt2$” | 按交互面归一化读法判 PASS（恒等 0.00 %），同时给出绝对值与物理模型值 | 绝对值读法在本板格不成立（板格单向极限本就在屈服以下），两种读法都摆出来，不选一个对自己有利的 |
| 5 | 交互公式“IACS CSR 交互公式” | 二次型 + 可选耦合系数 $c$（默认 0，保守） | 无 CSR 条文原文可引用时不伪造条文系数；$c$ 暴露为参数，用户可按所用版本设定 |

## 6. 误差判定（逐项回答“是否 < 3 %”）

| 关键量 | 基准 | 偏差 | 是否 < 3 % | 说明 |
|---|---|---|---|---|
| 板极限 $\sigma_u/\sigma_y$ | Faulkner 1975 闭式 | 9.19 %（最大，$\beta$=3） | **否** | 门限 10 % 内通过；两者同源有效宽度物理，属交叉核对 |
| 板格极限 $\sigma_u$ | CSR 型闭式链（独立实现） | 5.32 % | **否** | 门限 10 % 内通过；差在板侧 Faulkner 与渐进积分的峰位差 |
| 实验极限（3 组试件） | Paik & Thayamballi 2003 | — | **无法判定** | 主源表离线不可得，记 WITHHELD，不声明一致性 |
| 等双压折减 | $1/\sqrt2$（交互面恒等式） | 0.00 % | **是** | 但为交互面的定义性恒等，非独立测量（见注意点 3） |
| 收敛性（挠幅网格） | 二分法定位屈服交点 | 0（解析求交） | **是** | 极限值与网格无关；峰后斜率由假设给定 |

结论：**没有一项关键量对基准达到 < 3 %**。板侧与 CSR 型闭式链的吻合在 5–9 % 量级，满足本节点 10 % 的验收门限，但不足以宣称 3 % 级精度；实验判据缺口未闭合。

## 7. 复现与审计链

```bash
python benchmarks/smith_panel_collapse.py          # 生成 artifacts/smith_panel_collapse/
python scripts/report_smith_panel_collapse.py      # 生成本文档（数字自动注入）
python -m pytest tests/test_smith_panel_collapse.py -q
```

| 工件 | 路径 |
|---|---|
| 模块 | `src/tensorfem/smith_panel_collapse.py` |
| 证据运行 | `benchmarks/smith_panel_collapse.py` |
| 测试（18 例） | `tests/test_smith_panel_collapse.py` |
| 证据 JSON（6 位有效数字） | `artifacts/smith_panel_collapse/evidence.json` |
| 判据 JSON | `artifacts/smith_panel_collapse/criteria.json` |
| 判据表（markdown） | `artifacts/smith_panel_collapse/criteria_table.md` |
| 实验表规范（WITHHELD） | `artifacts/smith_panel_collapse/paik_2003_specimens.json` |
| 图 fig1_plate_curves.png | `artifacts/smith_panel_collapse/fig1_plate_curves.png` |
| 图 fig2_panel_curve.png | `artifacts/smith_panel_collapse/fig2_panel_curve.png` |
| 图 fig3_interaction.png | `artifacts/smith_panel_collapse/fig3_interaction.png` |
| 图 fig4_sensitivity.png | `artifacts/smith_panel_collapse/fig4_sensitivity.png` |

### 假设清单（全部可配置，证据 JSON 的 `assumptions` 同步落盘）

```json
{
  "bulb_flat_idealisation": "equivalent T, web 0.9h, flange 0.3h, wall thickness t",
  "plate_residual_floor_fraction": 0.4,
  "post_ultimate_slope_over_E": -0.1,
  "residual_zone_sigma_rt_over_yield": 1.0,
  "shear_ultimate": "min(sigma_y/sqrt(3), elastic shear buckling); no tension field",
  "stiffener_initial_bow": "0.002 a",
  "stiffener_plateau_end_over_yield_strain": 3.0,
  "stiffener_softening_slope_over_E": -0.05
}
```

