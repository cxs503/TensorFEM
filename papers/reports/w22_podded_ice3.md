# W22 / m14511 吊舱推进冰区操纵：MSC.137(76) 判据与 KVLCC2 单桨基准的对齐研究

**仓库** TensorFEM ｜ **分支** agent/m14511 ｜ **commit** `5fb5781f02f1959deb04cb1190a71da1e550977a` ｜ **工件生成** 2026-11-11T07:47:17+00:00

**工件链（数字全部由工件注入，本文件由 `benchmarks/w22_podded_ice_report.py` 生成）**

| 工件 | 路径 | sha256（工件内登记） |
|---|---|---|
| 锚件 | `benchmarks/anchors/msc137-simman-w22.json` | `e3f413f12796da8e2be496ece543c918a2f6d99e72ee04d29c8928199eee0946` |
| 结果工件 | `docs/w22_podded_ice/turning_zigzag.json` | `5abdf74130f4ee8a430a2986fd9f40324edf6028ac58633ae88beede580a2fcf` |
| 计算脚本 | `benchmarks/podded_ice_maneuvering_w22.py` | `8eab47a6e23dc05bcb3d38d3a39814644d1e5bb6dfbe3a29383459d408377f6f` |
| 出图脚本 | `benchmarks/w22_podded_ice_figures.py` | – |
| 测试 | `tests/test_podded_ice_maneuvering_w22.py` | 23 passed |

---

## ① 案例介绍

### 1.1 问题

前身任务 `paper-exp-maneuvering-podded-ice-1` 零报告、零工件（本检出内检索无该任务的操纵性工件）。本单显式承接其缺口，回答两个问题：

1. IMO MSC.137(76) 的进距（≤4.5L）／战术直径（≤5L）／Z 形首向超调判据，以及 SIMMAN 2008 KVLCC2 基准值，能否**逐字核销**（即：能否在本环境内把锚件抄录到可机裁的完整度）？
2. 吊舱推进（方位推力矢量转向、无舵）的 35° 回转与 10/10、20/20 Z 形，与**同壳体单桨+舵**构型相比，进距／战术直径差多少，判据逐条 PASS/FAIL 如何？

### 1.2 几何先行校验（性能数字之前）

| 量 | 值 | 判据 | 结论 |
|---|---|---|---|
| L_pp | 320.0 m | 锚件 AN-CASE-01 | – |
| B | 58.0 m | 锚件 AN-CASE-01 | – |
| T | 20.8 m | 锚件 AN-CASE-01 | – |
| Cb | 0.8098 | 0.35–0.90 | PASS |
| L/B | 5.52 | 3.0–12.0 | PASS |
| B/T | 2.79 | 1.5–6.5 | PASS |
| Δ = ρ·Cb·L·B·T | 320437 t（记录值 320000 t） | 闭合 ≤3% | PASS（0.137%） |

锚件交叉核对：Fr = V/√(gL) = 0.1422 vs AN-SIMMAN-02 0.142（差 0.175%，PASS）；L/V = 40.2 s ≥ 10 s → 10/10 超调判据取「≥10 s」分支。

### 1.3 模型与参数（声明，非抄录）

三自由度 MMG：`(m+m_x)u̇-(m+m_y)vr=X`、`(m+m_y)v̇+(m+m_x)ur=Y`、`(I_z+J_z)ṙ=N`，RK4，操舵速率限幅（舵 5.0°/s，吊舱 3.0°/s）。

| 量 | 值 | 来源口径 |
|---|---|---|
| Y'_v / Y'_r | -0.450 / 0.050 | 声明（大油轮量级，[-0.55,-0.35]/[0.03,0.08]） |
| Y'_vv / Y'_vr | -1.400 / -0.400 | 声明 |
| N'_v / N'_r | -0.100 / -0.050 | 声明（N'_v<0 ⇒ 航向不稳定，KVLCC2 级实况） |
| N'_vv / N'_vr | 0.400 / -0.200 | 声明 |
| m_x/m, m_y | 0.050, 1.00·ρπ(T/2)²L | 声明典型范围 |
| J_zz/I_zz | 0.20 | 声明 |
| Cr（附加阻力系数） | 0.0020 | 声明：使设计点 Ct≈0.0036 |
| A_R/(L·T) | 0.0160，f_α=2.724，ε_r=0.80，C_N,max=1.20 | 声明（Fujii 升力线 + 失速封顶） |
| 吊舱 | 2 台，x=−0.45L，y=±0.35B，A_pod=0.008LT，C_N,max=1.20 | 声明 |
| 冰况 | ΔR_ice = 0.00·R(U₀)（参数扫描） | 声明情景，非冰物理 |

**为什么是声明而不是抄录**：离线环境无 SIMMAN 2008 KVLCC2 导数列与操纵基准表（见锚件 `blocked`），按「禁凭记忆写数」纪律，模型系数以「带典型范围的声明值 + 不确定性带扫描」处理，绝对精度判 WITHHELD。

---

## ② 怎么做的

1. **先落锚件**：`benchmarks/anchors/msc137-simman-w22.json`，每个字段带 locator + 逐字引文 + 文件 sha256 + 方向（≤）+ 单位 + 尺度口径；缺源字段显式置 MISSING（AN-SIMMAN-03/04/05）。
2. **判据注入**：仿真脚本不硬编码任何判据阈值，运行时从锚件读（`imo_limits()` → 7 个字段）。
3. **两构型同壳体**：单桨+舵（螺旋桨滑流舵力、Fujii 升力线、a_H/x_H 干扰）vs 双吊舱（推力矢量 + 挂架侧力，无舵、无 a_H）。除操舵硬件外全部力模型相同 ⇒ 差异可归因。
4. **机动科目**：35° 与 20° 定舵角回转（1800 s）；10/10 与 20/20 Z 形（事件驱动、步内二分定位换舵与艏向极值，超调角按「本次换舵触发艏向」度量）。
5. **验证**：(i) Δt 1.0/0.5/0.25 s 减半收敛；(ii) 时域稳态再代入运动方程残差 + 自该态出发的 Newton 修正量；(iii) 自航点推力-阻力平衡（1e-6）。
6. **不确定性带**：7 个船体导数组（Y'_v/N'_r/N'_v 各软硬两档）+ Cr 两档 + 冰阻 4 档。

工具链：numpy（无外部数据）、matplotlib（图）、pytest（23 条）。

---

## ③ 计算结果（程序注入）

### 3.1 回转（35°／20°）

| 构型 | 舵/方位角 | 进距/L | 横距(90°)/L | 战术直径/L | 稳态? | 末速 u/u₀ | 末漂角/° |
|---|---|---|---|---|---|---|---|
| 单桨+舵 | 35° | 3.585 | 1.869 | 7.972 | 是 | 0.587 | 9.0 |
| 单桨+舵 | 20° | 5.155 | 3.418 | 14.884 | 是 | 0.774 | 5.9 |
| 双吊舱 | 35° | 2.434 | 0.916 | 3.233 | 是 | 0.112 | 23.5 |
| 双吊舱 | 20° | 3.532 | 1.831 | 7.293 | 是 | 0.166 | 20.5 |

### 3.2 Z 形（10/10、20/20）

| 构型 | 科目 | 首超调/° | 二超调/° | 状态 |
|---|---|---|---|---|
| 单桨+舵 | 10/10 | 2.152 | 2.671 | ok |
| 单桨+舵 | 20/20 | 5.134 | 6.135 | ok |
| 双吊舱 | 10/10 | 3.284 | 4.457 | ok |
| 双吊舱 | 20/20 | 8.911 | 12.869 | ok |

### 3.3 IMO MSC.137(76) 逐条裁决（两构型）

| 构型 | 判据量 | 量值 | 限值 | 结论 | 证据指针 |
|---|---|---|---|---|---|
| 单桨+舵 | `applicability.L_ge_min_length` | 320.000 m | 100.000 | **PASS** | `anchor AN-MSC137-07` |
| 单桨+舵 | `turning.advance_over_L` | 3.585 L | 4.500 | **PASS** | `turning_35deg.advance_over_L` |
| 单桨+舵 | `turning.tactical_diameter_over_L` | 7.972 L | 5.000 | **FAIL** | `turning_35deg.tactical_diameter_over_L` |
| 单桨+舵 | `zigzag.10_10.first_overshoot_deg` | 2.152 deg | 20.000 | **PASS** | `anchor AN-MSC137-03/04 (L/V branch)` |
| 单桨+舵 | `zigzag.10_10.second_overshoot_deg` | 2.671 deg | 40.000 | **PASS** | `anchor AN-MSC137-06` |
| 单桨+舵 | `zigzag.20_20.first_overshoot_deg` | 5.134 deg | 25.000 | **PASS** | `zigzag_20_20.first_overshoot_deg` |
| 单桨+舵 | `zigzag.20_20.second_overshoot_deg` | 6.135 deg | 40.000 | **PASS** | `zigzag_20_20.second_overshoot_deg` |
| 双吊舱 | `applicability.L_ge_min_length` | 320.000 m | 100.000 | **PASS** | `anchor AN-MSC137-07` |
| 双吊舱 | `turning.advance_over_L` | 2.434 L | 4.500 | **PASS** | `turning_35deg.advance_over_L` |
| 双吊舱 | `turning.tactical_diameter_over_L` | 3.233 L | 5.000 | **PASS** | `turning_35deg.tactical_diameter_over_L` |
| 双吊舱 | `zigzag.10_10.first_overshoot_deg` | 3.284 deg | 20.000 | **PASS** | `anchor AN-MSC137-03/04 (L/V branch)` |
| 双吊舱 | `zigzag.10_10.second_overshoot_deg` | 4.457 deg | 40.000 | **PASS** | `anchor AN-MSC137-06` |
| 双吊舱 | `zigzag.20_20.first_overshoot_deg` | 8.911 deg | 25.000 | **PASS** | `zigzag_20_20.first_overshoot_deg` |
| 双吊舱 | `zigzag.20_20.second_overshoot_deg` | 12.869 deg | 40.000 | **PASS** | `zigzag_20_20.second_overshoot_deg` |

### 3.4 吊舱 vs 单桨（同口径、同壳体）

| 量 | 单桨+舵 | 双吊舱 | 绝对差 | 相对差 |
|---|---|---|---|---|
| 进距/L | 3.585 | 2.434 | -1.151 | -32.12% |
| 战术直径/L | 7.972 | 3.233 | -4.739 | -59.45% |
| 横距(90°)/L | 1.869 | 0.916 | -0.953 | -50.99% |
| 10/10 首超调/° | 2.152 | 3.284 | 1.132 | – |
| 20/20 首超调/° | 5.134 | 8.911 | 3.776 | – |

**归因**：吊舱转向力矩来自推力矢量（T·sinφ，作用于 x=−0.45L），其大小不随漂角衰减；舵力则按有效攻角 α_R = δ − atan(v_R/u_R) 随漂角迅速衰减（本算例稳态时 α_R 已降到数度）。⇒ 吊舱在定方位角工况下给出**更强且不自限**的转向力矩，回转更紧（战术直径 -59.4%），Z 形首超调更大（10/10 1.13°）。

### 3.5 时间步收敛与稳态一致性

| 量 | Δt 序列/s | 减半相对变化 | 判据 | 结论 |
|---|---|---|---|---|
| 单桨 进距/L | 3.5846 / 3.5850 / 3.5859 | +0.0111% / +0.0249% | <2% | **PASS**（两构型四次减半最大 0.0778%） |
| 吊舱 进距/L | 2.4340 / 2.4336 / 2.4355 | +0.0164% / +0.0778% | <2% | 见上 |
| 单桨+舵 稳态残差 (u̇/u₀, ṙ) | – | -6.49e-09 1/s, 6.61e-10 1/s² | 机器零 | Newton 修正 6.0e-06 / 2.9e-06 |
| 双吊舱 稳态残差 (u̇/u₀, ṙ) | – | 2.47e-12 1/s, -1.84e-12 1/s² | 机器零 | Newton 修正 4.9e-09 / 3.1e-09 |

### 3.6 不确定性带（船体导数组 + 冰阻情景）

| 扫描轴 | 变体/值 | 单桨 进距/L | 吊舱 进距/L | 单桨 D_T/L | 吊舱 D_T/L |
|---|---|---|---|---|---|
| hull_derivative_variant | nominal | 3.585 | 2.434 | 7.972 | 3.233 |
| hull_derivative_variant | soft_lift | 3.499 | 2.400 | 7.308 | 2.964 |
| hull_derivative_variant | stiff_lift | 3.679 | 2.470 | 8.666 | 3.513 |
| hull_derivative_variant | weak_yaw_damping | 3.266 | 2.276 | 6.141 | 2.696 |
| hull_derivative_variant | strong_yaw_damping | 4.304 | 2.799 | 11.734 | 4.718 |
| hull_derivative_variant | more_unstable | 3.218 | 2.279 | 5.959 | 2.745 |
| hull_derivative_variant | less_unstable | 4.078 | 2.634 | 10.712 | 3.944 |
| residuary_coeff | 0.0 | 4.225 | 2.760 | 9.885 | 4.121 |
| residuary_coeff | 0.0035 | 3.304 | 2.254 | 7.139 | 2.809 |

吊舱冰阻情景（ΔR_ice/R(U₀) 参数扫描，声明情景）：

| ΔR_ice/R(U₀) | 进距/L | 战术直径/L | u/u₀ | 10/10 首超调/° |
|---|---|---|---|---|
| 0.0 | 2.434 | 3.233 | 0.112 | 3.284 |
| 0.5 | 2.382 | 2.962 | 0.080 | 3.343 |
| 1.0 | 2.326 | 2.686 | 0.030 | 3.410 |
| 2.0 | 2.206 | 2.132 | 0.000 | 3.572 |

---

## ④ 结果图

![回转轨迹与 Z 形](figs/w22_tracks_zigzag.png)

左：35° 回转轨迹（CG 航迹，与 IMO 进距 4.5L／战术直径 5L 限值对照）；右：10/10 Z 形艏向（实线）与操舵角（点线），超调角按触发艏向度量。

![敏感性](figs/w22_sensitivity.png)

左：进距 vs 船体导数组（IMO 4.5L 线）；中：战术直径 vs 导数组（IMO 5L 线）；右：吊舱构型在声明冰阻情景下的进距／战术直径／稳态速度。

---

## ⑤ 注意点（坑与局限）

1. Hull forces are a DECLARED SNAME-prime derivative set (linear + quadratic), not a transcribed PMM table: no digit-exact KVLCC2 derivative set exists offline (see the anchor blocked block). Absolute accuracy is WITHHELD.
2. Constant-azimuth pod model: the thrust-vector yaw moment does not decay with drift the way a rudder's does (the rudder's effective incidence falls as the drift builds). Consequently the podded 35 deg circle settles into a tight spiral (u -> 0.11 u0, kinematic 2u/r diameter 0.24 L) instead of a wide steady circle. The IMO indicators are measured inside the first half turn (advance at 90 deg, tactical diameter at 180 deg) and are therefore still defined, but the podded steady speed/drift must not be read as a podded-tanker speed loss.
3. No drift-induced longitudinal resistance (X_vv = X_vr = 0), no heel/roll coupling, no free surface, no shaft-power limit (the propeller speed is held at its self-propulsion value).
4. Ice is a declared scenario parameter (an extra steady resistance as a fraction of R(U0)); no ice-hull or ice-pod contact physics, no added mass from ice, no channel/broken-channel asymmetry.
5. IMO overshoot limits are a grade-B transcription (in-ecosystem transcription of Res. MSC.137(76) annex 2.3); one label conflict between two in-ecosystem sources is recorded and left unresolved (anchor conflicts block).
6. 吊舱构型在本模型中进入紧螺旋（末速 0.112 u₀，运动学直径 0.238 L），与文献上吊舱油轮 40–60% 降速不符：本单据此**只把前半个回转内的进距／战术直径作为 IMO 指标**（IMO 定义即如此），稳态速度/漂角不外推。
7. 单桨构型战术直径 7.97L 超出锚件记录的公开量级带 [2.5, 4.5]L（AN-SIMMAN-06，C 级括号值）：该括号值只用于**量级合理性核对**，本单不因落入/落出而改判，也不据此宣称精度。

---

## ⑥ 误差判定（逐项回答是否 <3%）

| 项目 | 量值 | 误差/差异 | 是否 <3% | 依据 |
|---|---|---|---|---|
| 几何闭合 Δ=ρCbLBT | 320437 t | 0.137% | 是 | 工件 `geometry_gate.closure_pct` |
| Fr 与锚件 AN-SIMMAN-02 | 0.1422 | 0.175% | 是 | `anchor_cross_checks` |
| 自航点推力-阻力平衡 | – | <1e-6（相对） | 是 | 测试 `test_self_propulsion_balance` |
| Δt 减半 → 进距 | 3.585 L | max Δ进距/advance over the two halvings = 0.0778% | 是（阈值 2%） | `verification.time_step_convergence` |
| 时域稳态 vs Newton | – | 修正 6.0e-06 | 是（机器零） | `steady_vs_marching` |
| **vs SIMMAN 2008 KVLCC2 单桨基准（进距/D_T/超调）** | 无基准值 | **不可计算** | **无法回答（WITHHELD）** | 锚件 AN-SIMMAN-03/04/05 MISSING |
| vs 公开量级带（AN-SIMMAN-06，C 级） | 单桨 D_T 7.97L | 带外（>4.5L） | 否 | 量级核对，非验证 |
| 吊舱 vs 单桨（同口径同壳体） | 进距 -32.1%，D_T -59.4% | —（>15%，属构型差异，已归因⑤） | 不适用 | `comparison_podded_vs_conventional` |

**结论**：<3% 只对「数值自洽类」（几何闭合、Fr 对齐、自航平衡、时间步收敛、稳态残差）成立；对**实验/参考基准**的 <3%（或 <15%）**不成立且未宣称**——离线无 SIMMAN KVLCC2 数值基准可抄录。

---

## ⑦ 判据逐条裁决表

| # | 判据 | 实测 | 阈值 | 裁决 | 证据指针 |
|---|---|---|---|---|---|
| 1 | anchor completeness 100% (direction + unit + scale caliber recorded) | 9/12 = 75% | 100% | **FAIL** | `benchmarks/anchors/msc137-simman-w22.json :: completeness` |
| 2 | advance / tactical diameter / Z overshoot tabulated with PASS/FAIL | 3 quantity families x 2 configurations x 4 manoeuvres (35deg turn, 20deg turn, 10/10 Z, 20/20 Z), every row carries value, limit, verdict and pointer | all three quantity families present with verdicts | **PASS** | `turning_zigzag.json :: runs.*.imo_verdicts` |
| 3 | same-caliber difference vs KVLCC2 single-screw SIMMAN baseline <15% | not computable | <15% | **WITHHELD** | `anchor AN-SIMMAN-03/04/05 (MISSING) + anchor.blocked` |
| 4 | time-step convergence: dt halved -> advance change <2% | max |d advance|/advance over the two halvings = 0.0778% | <2% | **PASS** | `verification.time_step_convergence.*.advance_over_L.halving` |

差距说明：

- 判据 1：SIMMAN 2008 KVLCC2 numerical baselines (advance / tactical diameter / Z-test overshoot / PMM derivatives) are not transcribed anywhere offline -> fields AN-SIMMAN-03/04/05 are MISSING; see the anchor 'blocked' block.
- 判据 3：No digit-exact SIMMAN KVLCC2 single-screw baseline (advance, D_T, overshoot) is available offline, so no same-caliber anchor difference can be formed. Only the internal podded-vs-single-screw delta is reported (comparison block); it is a model difference, not an anchor validation.

---

## ⑧ 锚件完整率与 blocked 说明

- 必填字段 12，已填 9，完整率 75% → **FAIL**。
- 缺失：AN-SIMMAN-03, AN-SIMMAN-04, AN-SIMMAN-05（SIMMAN 2008 KVLCC2 进距／战术直径／Z 形超调／PMM 导数列）。
- 缺什么：Digit-exact SIMMAN 2008 KVLCC2 advance / tactical-diameter / 10-10 overshoot (and PMM derivative) table values, i.e. the 'SIMMAN 2008 proceedings' numeric pages, model scale or full scale.
- 为什么缺：Offline environment: no network egress (external HTTPS is blocked by policy) and no in-repo/on-disk transcription of those tables exists.
- 解锁路径：Any of: (i) SIMMAN 2008 proceedings PDF/table scan ingested as a source transcription (benchmarks/anchors/sources/*.transcription.json style); (ii) an authoritative digitised KVLCC2 PMM/zigzag dataset (e.g. ITTC 7.5-02-06-03 sample or Yasukawa & Yoshimura 2015 table) supplied as an anchor document; (iii) a KVLCC2 EFD-to-full-scale zigzag table from the towing tank report.
- 记录在案的抄录冲突：ship-performance-platform charger lists "zigzag_10_second_overshoot_deg": 25.0 while TensorLBM's transcription puts 25 deg as the 20/20 FIRST-overshoot limit and 40 deg as the 20/20 SECOND. The two are incompatible as labels.（处置：Use the TensorLBM set (10/10 first: 10 deg for L/V<10 s, 20 deg for L/V>=10 s; 20/20 first 25, second 40) because it is the only transcription that states the L/V branch required by the mission brief ('Z 形首向超调判据按 L/V 条件另列'). The ship-performance-platform value is recorded, not used.）

---

## 复现

```bash
python benchmarks/podded_ice_maneuvering_w22.py     # -> docs/w22_podded_ice/turning_zigzag.json
python benchmarks/w22_podded_ice_figures.py        # -> docs/w22_podded_ice/figs/*.png
python benchmarks/w22_podded_ice_report.py         # -> papers/reports/w22_podded_ice3.md
python -m pytest tests/test_podded_ice_maneuvering_w22.py -q
```

环境：Python 3.11.6，numpy 2.4.6，Linux-6.6.52-1.7.lns23.loongarch64-loongarch64-with-glibc2.38。

