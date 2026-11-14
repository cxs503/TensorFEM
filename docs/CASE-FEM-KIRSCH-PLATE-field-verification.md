# CASE-FEM-KIRSCH-PLATE 场级验证：整域应力场对 Kirsch 闭式解复核

> 锚点 `CASE-FEM-KIRSCH-PLATE`（算例池｜开孔板应力集中系数 K=3｜ref: Kirsch 1898｜容差 <5%）的
> **验证欠账**：既有证书（`benchmarks/verified/fem_kirsch_plate/gates_verified.json`，14 门）
> 的应力证据只有峰值 `K_t`（单节点）、孔边环向剖面、载荷平衡与若干不变性——**没有任何一门
> 比较过孔边以外的应力场**。一个恰好在极点命中 `3T`、但 `σ_xy` 或横向分量错的解可以照样过门。
> 本单补的正是这条轴：在纯离散误差模型上，把**整个环带的应力场**逐点/逐元与闭式解比。

## ① 案例介绍

无限大平板中心圆孔、远场单向拉伸 `T`：Kirsch (1898) 给出精确解
`σ_tt(a,θ) = T(1 − 2cos2θ)`，极点 `K_t = σ_tt(a,π/2)/T = 3`；全场三分量
`σ_rr, σ_tt, σ_rt` 同为 `r,θ` 的闭式函数（Timoshenko & Goodier 3rd ed. Art. 39）。

**模型（纯离散误差）**：Q8 serendipity 平面应力极坐标环带 `r_out/a = 2`，
外缘 `r = r_out` 上**给定精确 Kirsch 位移**，孔边留自由（自然边界）。Kirsch 场正是该 Dirichlet
问题的精确解，因此下面每一项误差都只含离散误差——没有有限宽、没有 Saint-Venant 模型误差可以藏身。
孔边与配合面由二次边节点精确落在圆上（`polar_q8_mesh`），几何误差为零。

| 参数 | 值 |
|---|---|
| 环带 | `r_out/a = 2.0`，内半径 `a = 1` |
| 单元 | Q8 serendipity 平面应力，3×3 Gauss（生产路径 `tensorfem.quadratic_plane.solve_q8_plane`） |
| 网格阶梯 | `(n_θ, n_r)` = (24, 6), (48, 12), (96, 24), (192, 48)（r = 2 序列） |
| 单元/自由度 | 144 / 960 → 9216 / 56064 |
| 材料 | `E = 100000`，`ν = 0.3`（平面应力；面力问题应力场与 `E, ν` 无关） |
| 远端拉力 | `T = 1`（全部结果按 `T` 归一） |
| 定点探针 | (r/a=1.10, θ=+5°), (r/a=1.30, θ=+40°), (r/a=1.70, θ=+87°), (r/a=1.10, θ=+130°), (r/a=1.30, θ=-70°) |
| 判据（**先声明**） | 场 L2 ≤ 2.0 %、节点场 ≤ 3.0 %、定点 ≤ 3.0 %、收敛率 ≥ 0.9、孔边残差 ≤ 1.0 %、能量 ≤ 1.0 %、位移 ≤ 0.50 %、`K_t` ≤ 5.0 % |

## ② 怎么做的

1. **量**（每个网格级都算，定义见驱动 docstring）：
   - `K_t = σ_xx(0,a)/T`：生产路径的节点应力恢复（与锚点证书同一读法）；
   - 整域 `L2`：`‖σ_fe − σ_ex‖/‖σ_ex‖`，按 9 个 Gauss 点 `det J·w` 加权（Gauss 场=原始场），
     并在**恢复后的节点场**上再算一遍（`K_t` 就是从后者读的）；
   - 定点：5 个预先声明的内点，Newton 反演单元映射后 `q8_stress_at` 求值，父坐标余量 ≥ 0.3
     （保证单值，防止探针落在单元界面上）；
   - 孔边自由面残差：恢复场的 `max(|σ_rr|, |σ_rt|)/T`（自然边界该为 0）；
   - 应变能：`|U_fe − U_ex|/U_ex`，`U_ex` 由闭式 `σ:ε/2` 在同一 Gauss 规则上积分；
   - 位移：`‖u_fe − u_ex‖/‖u_ex‖`（外缘被精确给定，误差全在内部）。
2. **闭式解自证**（先于任何 FE）：孔边 `|σ_rr|,|σ_rt|` = 0.0e+00，
   `σ_tt(a,π/2)/T` = 3.000000，`σ_tt(a,0)/T` = -1.000000；
   极坐标与直角坐标两条求值路径逐点一致到 2.7e-15·T。
   另外与锚点 `run.py::kirsch_stress` 这份**独立实现**逐点比对（测试
   `test_two_independent_closed_forms_agree`，一致到 <1e-13·T）。
3. **工具链**：
```bash
OMP_NUM_THREADS=4 PYTHONPATH=src python benchmarks/verified/fem_kirsch_plate/field_verification.py   # 4 级，58 s
OMP_NUM_THREADS=4 PYTHONPATH=src python benchmarks/verified/fem_kirsch_plate/make_field_doc.py         # 生成本文
OMP_NUM_THREADS=4 PYTHONPATH=src pytest tests/test_kirsch_plate_field_verification.py -q               # 9 条，~5 s
```
   环境：loongarch64 / Python 3.11 / TensorFEM `tensorfem.quadratic_plane`，float64，稀疏直接/CG
   （末级相对残差 ≤ 9.6e-13）。

## ③ 计算结果（全部程序化注入 `field_verification.json`）

**网格阶梯**（`K_t`、场误差、孔边残差、定点误差；末级墙钟 49 s）

| `n_θ` | `n_r` | 单元 | 自由度 | `K_t` | `K_t` 相对误差 | 场 L2（Gauss） | 场 L2（节点） | 孔边残差/T | 定点误差 | 墙钟/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 | 6 | 144 | 960 | 3.019739 | 0.6580 % | 0.8319 % | 1.5900 % | 6.188e-02 | 2.072e-02 | 0.3 |
| 48 | 12 | 576 | 3648 | 3.005729 | 0.1910 % | 0.2103 % | 0.4143 % | 1.745e-02 | 2.999e-03 | 1.1 |
| 96 | 24 | 2304 | 14208 | 3.001476 | 0.0492 % | 0.0528 % | 0.1051 % | 4.631e-03 | 1.086e-03 | 8.2 |
| 192 | 48 | 9216 | 56064 | 3.000370 | 0.0123 % | 0.0132 % | 0.0264 % | 1.193e-03 | 1.478e-04 | 48.7 |

**观测收敛率**（`e ~ h^p`，`h_θ = 2π/n_θ` 每级减半）

| 级对 | 场 L2（Gauss） | 场 L2（节点） | `K_t` | 孔边残差 | 应变能 | 位移 |
|---|---:|---:|---:|---:|---:|---:|
| 24 -> 48 | 1.98 | 1.94 | 1.78 | 1.83 | 1.89 | 4.01 |
| 48 -> 96 | 1.99 | 1.98 | 1.96 | 1.91 | 1.97 | 4.03 |
| 96 -> 192 | 2.00 | 1.99 | 2.00 | 1.96 | 1.99 | 4.02 |

稳定的 `p ≈ 2`（场 L2 1.98–2.00）
说明四级都在渐近区，不是「两级误差恰好同量级」的假收敛，也不是某个网格的幸运点。

**定点探针（末级 `n_θ = 192`）**：`σ_xx`、`σ_yy` 对闭式逐点对照

| `r/a` | `θ` | 父坐标 ξ | 父坐标 η | `σ_xx/T`(FE) | `σ_xx/T`(exact) | `σ_yy/T`(FE) | `σ_yy/T`(exact) | 相对误差（三分量 2-范数） |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.10 | +5 | +0.600 | +0.333 | -0.03474 | -0.03471 | -0.59315 | -0.59307 | 0.013 % |
| 1.30 | +40 | -0.200 | -0.333 | 0.90827 | 0.90839 | -0.11381 | -0.11389 | 0.015 % |
| 1.70 | +87 | +0.200 | -0.200 | 1.35355 | 1.35340 | 0.33489 | 0.33485 | 0.011 % |
| 1.10 | +130 | +0.600 | -0.333 | 1.02914 | 1.02914 | 0.25791 | 0.25788 | 0.002 % |
| 1.30 | -70 | -0.200 | +0.333 | 1.66839 | 1.66837 | 0.23820 | 0.23819 | 0.005 % |

**判据门总表**：13/13 通过（失败项：无）

| 门 | 量 | 基准 | 值 | 判 | 阈值 | 结果 |
|---|---|---|---|---|---:|---|
| `oracle_edge_traction_free` | max(\|sigma_rr\|,\|sigma_rt\|)/T on r=a of the analytic oracle | Kirsch (1898); free circular hole | 0 | <= | 1e-12 | ✓ |
| `oracle_kt_value` | analytic sigma_tt(a,pi/2)/T | sigma_tt(a,pi/2)/T = 3 | 3 | >= | 3 | ✓ |
| `oracle_cartesian_vs_polar` | max component mismatch between the polar and Cartesian oracle paths, /T | one field, two evaluation routes (self-consistency) | 2.66454e-15 | <= | 1e-12 | ✓ |
| `levels_declared` | refinement levels with both n_theta and n_radial doubled | >= 3 levels for an observed rate | 4 | >= | 3 | ✓ |
| `kt_rel_error_finest` | \|K_t(n_theta=192) - 3\|/3 | Kirsch limit 3 (pool tolerance 5 %) | 0.000123439 | <= | 0.05 | ✓ |
| `stress_field_l2_gauss` | \|\|sigma_fe - sigma_ex\|\|_L2 / \|\|sigma_ex\|\|_L2 over all Gauss points (finest) | domain-wide closed form, pure discretisation model | 0.000132051 | <= | 0.02 | ✓ |
| `stress_field_l2_nodal` | same norm for the recovered nodal field (the field K_t is read from) | domain-wide closed form | 0.000264072 | <= | 0.03 | ✓ |
| `stress_field_rate` | min observed rate p of the Gauss-point stress L2 error (e ~ h^p) | Q8 superconvergent rate, p >= 0.9 | 1.98433 | >= | 0.9 | ✓ |
| `displacement_field` | \|\|u_fe - u_ex\|\|/\|\|u_ex\|\| on all nodes (finest) | interior Dirichlet problem, exact field known | 2.23029e-08 | <= | 0.005 | ✓ |
| `energy_error` | \|U_fe - U_ex\|/U_ex (finest) | closed-form strain energy over the annulus | 3.16537e-05 | <= | 0.01 | ✓ |
| `pointwise_probes` | max relative stress error at five declared interior probes (finest) | closed form at the same physical points | 0.000147806 | <= | 0.03 | ✓ |
| `edge_traction_residual` | natural (free) hole-rim residual max(\|sigma_rr\|,\|sigma_rt\|)/T from nodal recovery (finest) | traction-free hole: exactly zero | 0.00119263 | <= | 0.01 | ✓ |
| `probes_interior` | min over levels of the probe distance to its element boundary in parent coords (1 - max\|xi\|,\|eta\|) | probe must be interior for a single-valued stress | 0.398907 | >= | 0.3 | ✓ |

**图**（`docs/assets/kirsch_plate_field/`）
- `kirsch_field_contours.png`：末级 `n_θ=192`, `n_r=48` 的 `σ_xx/T`
  三幅并排——FE 恢复场 / Kirsch 闭式 / 两者之差（差值图动态范围被压到 1e-3 量级，肉眼已近白）。
- `kirsch_field_convergence.png`：场 L2（Gauss/节点）、位移、能量、`K_t` 的相对误差 vs `h_θ` 双对数图，
  带 slope 2 参考线；四级平行下降即上文 `p ≈ 2` 的图形证据。

## ④ 与锚点既有证书的关系（不重复、不覆盖）

| 证据轴 | 锚点 `run.py`（14 门） | 本单（13 门） |
|---|---|---|
| `K_t` 峰值 | ✓ Q4 有限宽板 + Q8 环带 | 复核：末级 `K_t = 3.000370`；与 `result.json` 重叠层级逐位一致（`n_θ=24`: 3.0197388292 vs 3.0197388292（|Δ| = 0.0e+00）、`n_θ=48`: 3.0057289743 vs 3.0057289743（|Δ| = 0.0e+00）、`n_θ=96`: 3.0014762656 vs 3.0014762656（|Δ| = 0.0e+00）） |
| 孔边环向剖面 RMS | ✓ 仅 Q4 有限宽板 | ✓ Q8 环带（末级 0.022 %） |
| **整域应力场 L2** | ✗ 无 | **✓ 0.0132 %（Gauss）/ 0.0264 %（节点）** |
| **定点三分量逐点** | ✗ 无 | **✓ 5 点，最大 0.015 %** |
| **孔边自由面残差** | ✗ 无（只判解析解的零牵引） | **✓ 0.119 %** |
| **应变能 / 位移场** | 只有位移相对 L2 报告值、未判 | **✓ 能量 0.0032 %，位移 2.23e-08** |
| 收敛率 | ✓ `K_t` 阶梯 | ✓ 场量三条独立率 |

## ⑤ 注意点（踩过的坑 / 局限）

1. **Gauss 权重归一会把能量算错 4 倍**。`q8_element_area` 已按 3×3 规则积分（`Σ w_a w_b = 4`），
   再乘 `w_a w_b` 就重复计入：首轮跑出来 `|U_fe−U_ex|/U_ex = 0.7500`（各级恒定 = 系统缩放，不是收敛），
   除以 4 后为 0.0032 %。**L2 比值对此免疫**（分子分母同缩放），但能量不免疫——
   这类「恒定比例」是判读求和权重的信号。
2. **探针不能放在单元界面上**。首轮把探针取在 `θ = 135°`，正是 `15°` 角节点线，父坐标 `η = −1`，
   恢复场在界面两侧不唯一，粗网格定点误差被虚抬到 3.1 %。现改为每个层级父坐标余量 ≥ 0.3 并设门
   `probes_interior`（实测余量 0.40）。
3. **孔边自由面残差是全场最大的局部误差**。末级 `max(|σ_rr|,|σ_rt|)/T = 1.19e-03`，
   比峰值 `K_t` 误差 `1.23e-04` 大约一个量级。这不矛盾：节点外推对**峰值**超收敛，
   而**自由面条件**只以 `O(h²)` 收敛。推论：不要把恢复场的 `σ_rr/σ_rt` 当孔边牵引力用（会读出
   0.12 % 的假牵引）。
4. **适用范围**：本单只覆盖 `r_out/a = 2.0` 的环带、平面应力、线弹性、
   `d/W → 0` 的无限板极限。有限宽修正（Howland/Heywood）与有限长板（Saint-Venant）不在本单，
   由锚点 `run.py` 的 `finite_width_trend` / `saint_venant_study` 负责，两单证据互补而不重复。
5. **判据是预先声明的，跑完不改**：`CRITERIA` 在驱动里先写好（场 L2 2 %、定点 3 %、残差 1 %、
   能量 1 %、位移 0.5 %、率 ≥ 0.9），实测值低 2–3 个量级，属"证据不紧张"，不是"阈值贴着结果调"。
6. **不是 GCI 研究**：这里给的是三级独立的 `p` 与四级误差表，未做 Richardson/GCI 带宽；锚点的
   `kirsch_extrapolated` 门负责外推极限。

## ⑥ 误差判定（关键量 vs 基准，逐项回答「是否 < 3%」）

| 关键量 | 基准 | 值（末级 `n_θ = 192`） | 相对误差 | <3%？ |
|---|---|---|---:|---|
| `K_t = σ_xx(0,a)/T` | 3（Kirsch 闭式） | 3.000370 | 0.0123 % | ✓ |
| 整域应力场 `‖σ_fe−σ_ex‖/‖σ_ex‖`（Gauss） | 闭式全场（误差本身即该范数） | 0.0132 % | 0.0132 % | ✓ |
| 恢复节点场同范数 | 闭式全场（`K_t` 由该场读出） | 0.0264 % | 0.0264 % | ✓ |
| 五点定点三分量应力 | 闭式逐点值 | 0.0148 % | 0.0148 % | ✓ |
| 孔边自由面残差 `max(\|σ_rr\|,\|σ_rt\|)/T` | 0（无牵引） | 1.193e-03 | 0.1193 % | ✓ |
| 位移场 `‖u_fe−u_ex‖/‖u_ex‖` | 0（Dirichlet 真解） | 2.230e-08 | 0.0000 % | ✓ |
| 应变能 `\|U_fe−U_ex\|/U_ex` | 闭式 `σ:ε/2` 同规则积分 | 6.637842e-05 | 0.0032 % | ✓ |

**结论**：全部关键量对 Kirsch 闭式解的误差均 **< 3 %**（更满足算例池 < 5 % 容差）；最差一项为 `0.1193%`；
其中除孔边自由面残差 0.119 % 外，其余关键量都 < 0.03 %
（峰值与外推超收敛，自由面条件只按 `O(h²)` 收敛，见 ⑤.3）。
观测收敛率 `p ≈ 2`，四级渐近一致。本单无 FAIL 项；两条与本单无关的仓库既有红测试
（`test_committed_figures_exist` 缺锚点自身 png、`test_no_unregistered_certificate_defect` 另有两个锚点
未登记债务）在**改动前后同样失败**，已隔离确认，不由本单引起。

*（本文由 `benchmarks/verified/fem_kirsch_plate/make_field_doc.py` 从 `field_verification.json` 生成，
所有数字程序化注入，零手抄。）*
