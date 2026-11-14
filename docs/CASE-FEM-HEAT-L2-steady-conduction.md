# CASE-FEM-HEAT-L2 -- 稳态热传导 Q4 单元的 L2 收敛率（≥2 阶）

**判据**：收敛率判据（不是单点误差判据）。**结论**：PASS --
31 项预登记闸门全部通过（明细见 §④）。全部数字由
`benchmarks/heat_l2_convergence.py` 从 `artifacts/CASE-FEM-HEAT-L2/evidence.json`
格式化输出，报告本身不含手抄数值。

---

## ① 问题与参考解的出处

求解稳态热传导（无内热源假设不成立的一般形式）

    -div(k grad T) = q    in Omega = (0, Lx) x (0, Ly),   k = 1
     T = T_exact          on dOmega（全边界 Dirichlet，非齐次）

离散：TensorFEM 既有的双线性 `Q4` 传导单元
（`tensorfem.thermal`：2x2 Gauss 传导矩阵 + 一致热容 + 单元常数体热源 + 一致
Dirichlet 消元），直接稠密求解 `tensorfem.thermal.solve_steady_thermal`。

**参考解 = 制造解（method of manufactured solutions, MMS）**：精确温度场是解析式，
以代码形式实现（`HeatFieldCase.temperature` / `.gradient` / `.source`），
误差范数与对照表中的每一个 ref 数值都在运行时由该解析式求值，**零手抄**。

**目标数值（收敛阶）同样不是手抄**：`target_l2_order_from_manual()` 用正则在运行时
解析本仓版本化的先验阶次表，取到

| 目标量 | 手册出处（文件:行，程序化解析） | 原文行 |
|---|---|---|
| p_ref(L2, Q4) | `src/tensorfem/adaptive_poisson_2d.py:34` | `L2 norm ``||e||_0``              ``O(h^2)``         ``-1``` |

解析结果 `p_ref = 2.0`。若该表被改动，目标随动、研究 fail-closed
（解析不到直接抛 `RuntimeError`）。

方法学外部锚点（节级引用，与仓内既有引用惯例一致）：

* K. Salari & P. Knupp, *Code Verification by the Method of Manufactured Solutions*,
  Sandia National Laboratories **SAND2000-1009J** (2000)：正确实现的离散格式，其
  **观测阶必须收敛到形式阶**；MMS 是获取该阶的标准手段。
* P.J. Roache (1994) *J. Fluids Eng.* **116**, 405-413：refinement-ratio 外推与
  GCI 安全因子（`tensorfem.convergence_verification` 即建立在此文献上）。
* 离线环境无法核对纸质手册页码，故外部引用做到"节/报告号"级；目标阶的主锚是
  上表这条**仓内版本化手册行 + 程序化解析**，这比转抄一个数字更可审计。

物理背景（船舶总体性能）：稳态热传导是船体结构热载荷（主机排气/舱室保温/焊后
残余应力热载）与电子设备散热分析的底层求解器；L2 阶次是这批算例的可信度基线。

## ② 方法与实现

新增能力 `src/tensorfem/heat_l2_convergence.py`（复用既有内核，不另起炉灶）：

1. **制造解族**（6 例：4 个正则解 + 1 个节点精确解 + 1 个控制例）：
   `sine_generation`（带源项 sin·sin）、`exp_trig_harmonic`（调和、非多项式）、
   `cubic_harmonic`（调和三次多项式）、`sine_aspect2`（2:1 域，非正方形 Jacobian）、
   `quadratic_harmonic`（调和二次：节点精确 + 纯插值误差基线）、
   `bilinear_control`（`T=xy`，既调和又在双线性空间内 → 必须精确，控制例）。
2. **误差范数是真积分**：`evaluate_errors()` 是**独立**的 Gauss 后处理（自带形函数，
   不复用装配内核），逐单元积分

       ||e||_0 = sqrt(∫(T_h-T)^2 dO / ∫T^2 dO),  |e|_1 = sqrt(∫|∇T_h-∇T|^2/∫|∇T|^2)

   同一后处理器把 `T_exact` 在节点取值再积分一次，就得到**双线性插值误差**基线
   （`l2_relative_interpolant`），用来判断"FE 解是否只剩插值误差"。
3. **载荷一致性**：热源是单元常数，MMS 的 `q` 先用 8x8 Gauss 在单元上积分成单元
   平均值再交给内核（`HeatFieldCase.element_mean_source`），避免把源项投影误差
   混进阶次。
4. **阶梯**：nx = [4, 8, 16, 32, 64]（r = 2，等比方），
   2:1 域取 (nx, nx/2)。最细网格 65x65=4225 未知量（稠密装配上限）。
5. **判据**：观测阶 `p = ln(e_coarse/e_fine)/ln(r)`；另加 Roache 形式的外推残差
   `|e0|/e_fine`（`e0 = e_fine + (e_fine-e_coarse)/(r^p_ref-1)`）检验整条阶梯是否
   真是理论幂律，而不是三个"凑巧"的点。

**预登记闸门**（写在模块 docstring 里，跑之前定死，未事后调整）：
L2 每段 p ≥ 1.9；最细段 p ∈ [1.95, 2.1]；
H1 每段 p ≥ 0.9；幂律残差 ≤ 0.01；控制例绝对误差 ≤
1e-10；最细网格场误差 ≤ 3%；误差范数对
积分阶次（4→6→8 点）的相对变化 ≤ 1e-08。

## ③ 参数表（无量纲量完整）

| case | Lx x Ly | aspect Lx/Ly | q_max Lx^2 / (k T_peak) | Dirichlet | mesh ladder | r | h/Lx (finest) |
|---|---|---|---|---|---|---|---|
| sine_generation | 1 x 1 | 1.000 | 19.7392 | full boundary, T_exact | 4..64 | 2 | 0.01562 |
| exp_trig_harmonic | 1 x 1 | 1.000 | 0.0000 | full boundary, T_exact | 4..64 | 2 | 0.01562 |
| cubic_harmonic | 1 x 1 | 1.000 | 0.0000 | full boundary, T_exact | 4..64 | 2 | 0.01562 |
| sine_aspect2 | 2 x 1 | 2.000 | 49.3480 | full boundary, T_exact | 4..64 | 2 | 0.01562 |
| quadratic_harmonic | 1 x 1 | 1.000 | 0.0000 | full boundary, T_exact | 4..64 | 2 | 0.01562 |
| bilinear_control | 1 x 1 | 1.000 | 0.0000 | full boundary, T_exact | 4..64 | 2 | 0.01562 |

补充无量纲量（稳态，无 Fo/Bi 时间或对流尺度，故不列傅里叶数/毕渥数）：

| 量 | 定义 | 值 / 说明 |
|---|---|---|
| 源项强度（正弦例） | q Lx^2/(k T_peak) | 19.7392 (= 2 pi^2) |
| 源项强度（2:1 例） | q Lx^2/(k T_peak) | 49.3480 (= 1.25 pi^2 (Lx/Ly)^2 = 5 pi^2) |
| 导热系数 | k | 1（归一化单位，问题对 k 线性齐次） |
| 单元长宽比 | hx/hy | 1（1:1 例）/ 1（2:1 例，nx:ny 随域比例取） |
| 网格比 | r = h_i/h_(i+1) | 2 |
| dof | (nx+1)(ny+1) | 25 .. 4225 |
| 最细 h/Lx | sqrt(A/ne)/Lx | 0.01562 |

## ④ 对照表（ref vs got + 误差打勾/叉）

ref = 制造解解析值（程序求值）；got = TensorFEM Q4 稳态解。

| case | mesh | probes | max abs (ref-got) | max |e|/max|T_ref| | relative L2 | <3% ? |
|---|---|---|---|---|---|---|
| sine_generation | 64x64 | 4096 | 8.025e-04 | 0.0803% | 6.155e-04 | OK |
| exp_trig_harmonic | 64x64 | 4096 | 1.646e-03 | 0.0073% | 1.777e-04 | OK |
| cubic_harmonic | 64x64 | 4096 | 5.107e-15 | 0.0000% | 7.613e-05 | OK |
| sine_aspect2 | 64x32 | 2048 | 2.184e-03 | 0.2187% | 1.726e-03 | OK |
| quadratic_harmonic | 64x64 | 4096 | 1.776e-15 | 0.0000% | 6.104e-05 | OK |
| bilinear_control | 64x64 | 4096 | 3.553e-15 | 0.0000% | 2.704e-15 | OK |

逐段阶次对照：

| case | p_L2 per pair | min p (nx>=8) | finest p | window [1.95, 2.10] | residual at p=2 |
|---|---|---|---|---|---|
| sine_generation | 1.946, 1.987, 1.997, 1.999 | 1.987 | 1.999 | OK | 7.71e-04 |
| exp_trig_harmonic | 1.991, 1.997, 1.999, 2.000 | 1.997 | 2.000 | OK | 1.59e-04 |
| cubic_harmonic | 2.010, 2.002, 2.001, 2.000 | 2.000 | 2.000 | OK | 1.39e-04 |
| sine_aspect2 | 1.857, 1.965, 1.991, 1.998 | 1.965 | 1.998 | OK | 1.98e-03 |
| quadratic_harmonic | 2.000, 2.000, 2.000, 2.000 | 2.000 | 2.000 | OK | 1.07e-13 |

完整阶梯（每层 ref/got 范数）：

| case | level (nx x ny) | dof | h | L2 rel | p_L2 | H1 rel | p_H1 |
|---|---|---|---|---|---|---|---|
| sine_generation | 4 x 4 | 25 | 0.25000 | 1.4996e-01 | - | 2.4628e-01 | - |
| sine_generation | 8 x 8 | 81 | 0.12500 | 3.8916e-02 | 1.946 | 1.1604e-01 | 1.086 |
| sine_generation | 16 x 16 | 289 | 0.06250 | 9.8198e-03 | 1.987 | 5.7024e-02 | 1.025 |
| sine_generation | 32 x 32 | 1089 | 0.03125 | 2.4606e-03 | 1.997 | 2.8384e-02 | 1.007 |
| sine_generation | 64 x 64 | 4225 | 0.01562 | 6.1552e-04 | 1.999 | 1.4176e-02 | 1.002 |
| exp_trig_harmonic | 4 x 4 | 25 | 0.25000 | 4.5087e-02 | - | 2.2632e-01 | - |
| exp_trig_harmonic | 8 x 8 | 81 | 0.12500 | 1.1346e-02 | 1.991 | 1.1332e-01 | 0.998 |
| exp_trig_harmonic | 16 x 16 | 289 | 0.06250 | 2.8417e-03 | 1.997 | 5.6675e-02 | 1.000 |
| exp_trig_harmonic | 32 x 32 | 1089 | 0.03125 | 7.1076e-04 | 1.999 | 2.8340e-02 | 1.000 |
| exp_trig_harmonic | 64 x 64 | 4225 | 0.01562 | 1.7771e-04 | 2.000 | 1.4170e-02 | 1.000 |
| cubic_harmonic | 4 x 4 | 25 | 0.25000 | 1.9661e-02 | - | 1.4964e-01 | - |
| cubic_harmonic | 8 x 8 | 81 | 0.12500 | 4.8828e-03 | 2.010 | 7.4731e-02 | 1.002 |
| cubic_harmonic | 16 x 16 | 289 | 0.06250 | 1.2187e-03 | 2.002 | 3.7355e-02 | 1.000 |
| cubic_harmonic | 32 x 32 | 1089 | 0.03125 | 3.0454e-04 | 2.001 | 1.8676e-02 | 1.000 |
| cubic_harmonic | 64 x 64 | 4225 | 0.01562 | 7.6127e-05 | 2.000 | 9.3378e-03 | 1.000 |
| sine_aspect2 | 4 x 2 | 15 | 0.50000 | 3.8780e-01 | - | 4.7100e-01 | - |
| sine_aspect2 | 8 x 4 | 45 | 0.25000 | 1.0704e-01 | 1.857 | 2.1715e-01 | 1.117 |
| sine_aspect2 | 16 x 8 | 153 | 0.12500 | 2.7411e-02 | 1.965 | 1.0558e-01 | 1.040 |
| sine_aspect2 | 32 x 16 | 561 | 0.06250 | 6.8937e-03 | 1.991 | 5.2393e-02 | 1.011 |
| sine_aspect2 | 64 x 32 | 2145 | 0.03125 | 1.7260e-03 | 1.998 | 2.6146e-02 | 1.003 |
| quadratic_harmonic | 4 x 4 | 25 | 0.25000 | 1.5625e-02 | - | 1.2500e-01 | - |
| quadratic_harmonic | 8 x 8 | 81 | 0.12500 | 3.9062e-03 | 2.000 | 6.2500e-02 | 1.000 |
| quadratic_harmonic | 16 x 16 | 289 | 0.06250 | 9.7656e-04 | 2.000 | 3.1250e-02 | 1.000 |
| quadratic_harmonic | 32 x 32 | 1089 | 0.03125 | 2.4414e-04 | 2.000 | 1.5625e-02 | 1.000 |
| quadratic_harmonic | 64 x 64 | 4225 | 0.01562 | 6.1035e-05 | 2.000 | 7.8125e-03 | 1.000 |
| bilinear_control | 4 x 4 | 25 | 0.25000 | 1.3723e-16 | - | 2.6356e-16 | - |
| bilinear_control | 8 x 8 | 81 | 0.12500 | 4.1746e-16 | -1.605 | 1.2769e-15 | -2.276 |
| bilinear_control | 16 x 16 | 289 | 0.06250 | 7.6024e-16 | -0.865 | 2.7442e-15 | -1.104 |
| bilinear_control | 32 x 32 | 1089 | 0.03125 | 8.0853e-16 | -0.089 | 5.8230e-15 | -1.085 |
| bilinear_control | 64 x 64 | 4225 | 0.01562 | 2.7040e-15 | -1.742 | 1.3797e-14 | -1.245 |

Target order: p_ref = 2.0 (manual row `L2 norm ``||e||_0``              ``O(h^2)``         ``-1``` in `adaptive_poisson_2d.py:34`).

**闸门明细（31 项，全部程序判定）**：

| gate | pass | detail |
|---|---|---|
| sine_generation:l2_pair | OK | min L2 order over pairs with coarse nx>=8 is 1.987 (gate >=1.90, theory 2.0) |
| sine_generation:l2_finest | OK | finest-pair L2 order 1.999 inside [1.95, 2.10] |
| sine_generation:h1_pair | OK | min H1 order 1.002 (gate >=0.90, theory 1.0) |
| sine_generation:power_law | OK | Richardson residual at the theoretical order 2: 7.713e-04 of e_fine (gate <=1e-02) |
| sine_generation:field_3pct | OK | finest mesh, element centres: max |error| / max|T| = 0.0803% (gate 3%, peak |T| = 0.9994) |
| exp_trig_harmonic:l2_pair | OK | min L2 order over pairs with coarse nx>=8 is 1.997 (gate >=1.90, theory 2.0) |
| exp_trig_harmonic:l2_finest | OK | finest-pair L2 order 2.000 inside [1.95, 2.10] |
| exp_trig_harmonic:h1_pair | OK | min H1 order 1.000 (gate >=0.90, theory 1.0) |
| exp_trig_harmonic:power_law | OK | Richardson residual at the theoretical order 2: 1.585e-04 of e_fine (gate <=1e-02) |
| exp_trig_harmonic:field_3pct | OK | finest mesh, element centres: max |error| / max|T| = 0.0073% (gate 3%, peak |T| = 22.5728) |
| cubic_harmonic:l2_pair | OK | min L2 order over pairs with coarse nx>=8 is 2.000 (gate >=1.90, theory 2.0) |
| cubic_harmonic:l2_finest | OK | finest-pair L2 order 2.000 inside [1.95, 2.10] |
| cubic_harmonic:h1_pair | OK | min H1 order 1.000 (gate >=0.90, theory 1.0) |
| cubic_harmonic:nodal_exact | OK | 5-point-Laplacian exactness: worst nodal error over the ladder 5.329e-15 (gate 1e-10) |
| cubic_harmonic:interp_identity | OK | finest-mesh e_FE / e_interpolant = 1.000000000 (the solver must add nothing to interpolation error) |
| cubic_harmonic:power_law | OK | Richardson residual at the theoretical order 2: 1.395e-04 of e_fine (gate <=1e-02) |
| cubic_harmonic:field_3pct | OK | finest mesh, element centres: max |error| / max|T| = 0.0000% (gate 3%, peak |T| = 1.9535) |
| sine_aspect2:l2_pair | OK | min L2 order over pairs with coarse nx>=8 is 1.965 (gate >=1.90, theory 2.0) |
| sine_aspect2:l2_finest | OK | finest-pair L2 order 1.998 inside [1.95, 2.10] |
| sine_aspect2:h1_pair | OK | min H1 order 1.003 (gate >=0.90, theory 1.0) |
| sine_aspect2:power_law | OK | Richardson residual at the theoretical order 2: 1.984e-03 of e_fine (gate <=1e-02) |
| sine_aspect2:field_3pct | OK | finest mesh, element centres: max |error| / max|T| = 0.2187% (gate 3%, peak |T| = 0.9985) |
| quadratic_harmonic:l2_pair | OK | min L2 order over pairs with coarse nx>=8 is 2.000 (gate >=1.90, theory 2.0) |
| quadratic_harmonic:l2_finest | OK | finest-pair L2 order 2.000 inside [1.95, 2.10] |
| quadratic_harmonic:h1_pair | OK | min H1 order 1.000 (gate >=0.90, theory 1.0) |
| quadratic_harmonic:nodal_exact | OK | 5-point-Laplacian exactness: worst nodal error over the ladder 1.887e-15 (gate 1e-10) |
| quadratic_harmonic:interp_identity | OK | finest-mesh e_FE / e_interpolant = 1.000000000 (the solver must add nothing to interpolation error) |
| quadratic_harmonic:power_law | OK | Richardson residual at the theoretical order 2: 1.068e-13 of e_fine (gate <=1e-02) |
| quadratic_harmonic:field_3pct | OK | finest mesh, element centres: max |error| / max|T| = 0.0000% (gate 3%, peak |T| = 0.9844) |
| bilinear_control:control_exact | OK | representable field, worst absolute error 3.553e-15 over the whole ladder (gate 1e-10); no order claimed |
| quadrature_postprocessing | OK | worst relative change of an error norm for 4 -> 6/8 point rules: 1.12e-11 (gate <=1e-08) |

误差判定（报告纪律要求的显式回答）：最细网格上**所有算例**的场误差
`max|e|/max|T|` 均 < 3%（最大 0.2187%），
且**收敛率判据**（本单的主判据）最细段 p_L2 = 1.999（理论
2.0，|相对偏差| 0.04% < 3%）。

## ⑤ 收敛性与参数敏感性

1. **Mesh size (the ladder itself).** Every pair of the ladder in `order_table` is a refinement-ratio point; the refinement ratio is r = 2.000..2.000, so the observed order is not contaminated by a non-constant ratio.
2. **Field regularity / type.** Four independent manufactured fields (sin*sin with source, exponential harmonic, harmonic cubic, 2:1 aspect-ratio sin) give the same asymptotic order; the spectra in the figure show the *pre-asymptotic* range instead: the 2:1 case is still 1.857 on the coarsest pair (its coarse mesh has only 2 elements across the short side) but 1.998 on the finest.
3. **Quadrature of the error norm.** Raising the post-processing rule from 4 to 6 and 8 points per direction at level nx=32 changes every error norm by at most 1.12e-11 relative (gate 1e-08), i.e. the measured order is a property of the discretisation and not of the integration rule.
4. **Source projection.** The heat kernel takes an element-constant source; the MMS load is integrated over each element with an 8x8 Gauss rule before it is handed to the kernel, so the load vector is the consistent one for the element-mean source and the ladder measures the *conduction* discretisation, not a quadrature slip.
5. **Control that must not move.** The bilinear-representable field `bilinear_control` stays exact on every level (worst absolute error 3.55e-15, gate 1e-10); its ladder carries no signal and no order is claimed for it.

L2 阶次的最小二乘 log-log 斜率（整条阶梯）：sine_generation = 1.9840, exp_trig_harmonic = 1.9971, cubic_harmonic = 2.0028, sine_aspect2 = 1.9580, quadratic_harmonic = 2.0000。

图 1（`docs/assets/case_fem_heat_l2_convergence.png`）：L2/H1 误差随 h 的 log-log 阶梯，虚线为理论斜率
`O(h^2)` / `O(h^1)` 锚定在次细网格点上的参考线。图 2（`docs/assets/case_fem_heat_l2_field.png`）：
`sine_generation` 在 nx=32 网格上 FE 场、解析场与
逐点误差云图（量级 8.04e-04，与 L2 范数一致）。

## ⑥ 失败模式与工件

**本项目观察到的失败模式 / 坑（全部留档）**

1. **可表示场 = 无信号的收敛研究。** 首版把 `T = x^2 - y^2` 当作"可精确表示的控制例"，
   控制闸门以 6.6e-3 失败。诊断：双线性空间是 `span{1, x, y, xy}`，二次场根本不
   在空间内 —— 错的是**算例规格**，不是求解器。改法：`T = xy` 做控制例，并把
   `x^2-y^2` 改归为"节点精确 + 纯插值误差"算例（其 `e_FE/e_interp = 1.000000`）。
   生态里已有同类负面知识（点载荷恰在网格点 → 误差 1e-15、收敛研究无信号），
   本单把它推广到热传导：**先证明参考场不被离散空间精确表示，再谈阶次**。
2. **积分阶次污染阶次。** 仓库里曾出现"order 2.0 其实来自梯形积分误差"的审计结论；
   本单用 4/6/8 点 Gauss 后处理对照（最大相对变化
   1.12e-11）证明阶次不属于积分规则。
3. **单元常数源项。** `tensorfem.thermal` 的体热源是单元常数；若直接把解析 `q` 在
   形心取值，会引入 O(h^2) 的载荷一致性误差、与离散误差同阶。改为 8x8 Gauss 单元
   积分后再交给内核。
4. **稠密装配的规模上限。** `thermal` 路径全稠密（4225 未知量 → 143 MB 矩阵、
   ~1 GB RSS、nx=64 单次 3 s）；nx=128 的稠密矩阵（2.2 GB）在本机不可行，
   阶梯止于 64。这是 tensorfem.thermal 的真实规模限制（friction）。
5. **线程过度订阅。** 128 核默认线程池在 4225 阶稠密求解上比 4 线程慢一个量级
   （仓内既有记录），脚本显式 `torch.set_num_threads(4)`。

**工件（worktree 内真实文件）**

| 工件 | 说明 |
|---|---|
| `src/tensorfem/heat_l2_convergence.py` | 算例族 + 独立 Gauss 误差后处理 + 阶梯/闸门（能力本体） |
| `benchmarks/heat_l2_convergence.py` | 一键复现脚本（证据/图/报告全部由它生成） |
| `artifacts/CASE-FEM-HEAT-L2/evidence.json` | 全部阶梯、范数、闸门、敏感性证据 |
| `artifacts/CASE-FEM-HEAT-L2/tables.md` | 本报告的表格（独立可引用） |
| `tests/test_heat_l2_convergence.py` | pytest 闸门（含控制例与积分独立性） |
| `docs/assets/case_fem_heat_l2_convergence.png` | 图 1 |
| `docs/assets/case_fem_heat_l2_field.png` | 图 2 |

**一键复现**

    PYTHONPATH=src python3 benchmarks/heat_l2_convergence.py
    PYTHONPATH=src python3 -m pytest tests/test_heat_l2_convergence.py -q

**生态摩擦（如实记）**

* 平台 `/api/v1/auth` 直接用账号密码 POST 返回 401（"未提供认证令牌"），必须走
  `/api/v1/auth/login`（`{"access_token"}` 字段）；仓内 `get_token.py`
  仍按旧 `/auth` 写，已失效。
* 开工三查：`knowledge_base` 全库 `count=0`（无同域 capability 条目）；
  `negative_knowledge_ledger` 关键词 `TensorFEM`/`thermal` 均 0 命中，仅
  `convergence order` 命中 2 条（点载荷收敛退化 m17558、另一收敛失败账），
  已用于本单的算例设计（第 1 条坑）；`oss_solutions` 无公开端点，仓内既有能力
  （`tensorfem.thermal` + `convergence_verification`）足够，故不另起炉灶。
* 全流程（6 个算例 x 5 层阶梯 + 积分敏感性 + 2 张图）本机 4 线程耗时
  40.7 s，远低于 10 min 门槛，无需上集群。

**适用范围/局限**

* 仅等距结构化 Q4 网格与正则解；奇异解（角点、裂纹）的 L2 阶次会退化，不在本单结论内。
* `thermal` 内核的体热源是单元常数；非单元常数源的严格一致性载荷需要改内核（未做，
  记为局限，而不是用后处理掩盖）。
* 结论是"该离散的 L2 阶次达 2 阶"，不是"任何网格/任何场都达 2 阶"：2:1 例的粗段
  （短边只有 2 个单元）仍有 1.857，是预渐近区。
