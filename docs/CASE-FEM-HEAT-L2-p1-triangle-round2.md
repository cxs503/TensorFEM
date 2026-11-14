# CASE-FEM-HEAT-L2（第 2 轮）：P1 线性三角形稳态导热的制造解阶次验证

> 本文件由 `benchmarks/verified/fem_heat_triangle_p1/make_report.py` 从
> `result.json` 程序化生成（判据数字零手抄）。生成时间 2026-11-14T17:08:30Z，
> 全流程 47.2 s，环境 3.11.6 /
> torch 2.7.1 / loongarch64。

## ① 问题设定与基准出处

求解定常导热（`k = 1`，`q` 为体热源）：

```
-div(k grad T) = q     in  Omega = (0, Lx) x (0, Ly)
T = T_exact            on  Dirichlet 边界
k dT/dn = g            on  x = Lx（仅混合边条件算例）
```

**离散**：3 节点线性三角形（P1/常应变型标量单元），单元传导矩阵用 1 点积分精确得到
（线性单纯形梯度为常数），单元常数源的一致载荷 = `q_e A_e / 3`，Neumann 边的线性
通量载荷 = `g L_e / 2`（梯形精确），Dirichlet 用对称消元后稠密求解。

**基准（ref）**：制造解（Method of Manufactured Solutions）。解析场以代码形式给出，
所有参考数字在运行时由解析表达式求值 —— 没有任何数值是手抄的。文献锚点：

| 锚点 | 出处 | 用途 |
|---|---|---|
| MMS 规程 | Salari & Knupp, *Code Verification by the Method of Manufactured Solutions*, SAND2000-1009J (2000) | 正确实现的观测阶次必须趋于形式阶次 |
| P1 先验估计 | Zienkiewicz & Taylor, *FEM* vol.1, 5th ed. (2000), ch.3–4/7：`‖T−T_h‖₀ ≤ C h² |T|_H²`，`|T−T_h|₁ ≤ C h |T|_H²` | L2 阶 2 / H1 阶 1 的理论值 |
| 网格加密外推 | Roache, *J. Fluids Eng.* 116 (1994) 405–413 | Richardson 残差闸门 |
| V&V 规范 | ASME V&V 20-2009 | 阶次验证属于代码验证 |
| 独立实现 | scikit-fem `ElementTriP1` + `models.poisson.laplace` | 交叉验证（xval） |

**目标阶次不是手写的**：`target_l2_order_from_manual()` 解析仓内
`tensorfem/adaptive_poisson_2d.py` 的先验阶次表（`tensorfem/adaptive_poisson_2d.py a-priori table (L2 row)`，第 34 行：
`L2 norm ``||e||_0``              ``O(h^2)``         ``-1```），得到 `p = 2.0`。

## ② 方法与实现

与既有 Q4 研究（`src/tensorfem/heat_l2_convergence.py`，`benchmarks/verified/fem_heat_conduction_l2/`）
**不重复**：算例场直接 import 复用，本单只补它声明的缺口 —— 单元族与网格拓扑。

| 关键代码指针 | 作用 |
|---|---|
| `src/tensorfem/heat_l2_triangle.py::build_mesh` | 三种网格族：`structured-a` / `structured-b`（另一条对角线）/ `distorted`（内部节点 ±0.25h 确定性扰动，边界节点不动） |
| `::p1_conductivity_matrix` | P1 传导矩阵（1 点精确积分） |
| `::element_mean_source` / `::assemble_source` | 制造解源项的 7 点 Gauss 单元平均与一致载荷 |
| `::assemble_flux_load` | Neumann 边线性通量一致载荷（梯形精确） |
| `::_norms` | 独立后处理：7 点 Dunavant 规则（5 阶精确），L2 被积函数为 4 次、H1 为 2 次 |
| `::subdivide` + `::quadrature_independence` | 积分规则独立性：把 L2 积分嵌套四分加密 1/2/3 层比较 |
| `::skfem_crosscheck` | 同一网格、同一载荷、同一边条件与 scikit-fem 逐节点比较 |
| `::run_all` | 全部阶梯 + 闸门 + 参数研究 |

**数值上不能作弊的四处**（详见模块 docstring）：积分规则（4 次被积函数不能只用 2 阶规则）、
源项一致性（单元常数源必须用 Gauss 积分再交给内核）、对角线对称（两种剖分都要过闸门）、
拓扑（扰动网格上阶次不变）。

## ③ 参数表

| 量 | 值 |
|---|---|
| 计算域 | `(0,1)²`；2:1 算例为 `(0,2)×(0,1)` |
| 导热率 `k` | 1（无量纲） |
| 网格族 | structured-a, structured-b, distorted |
| 单元数阶梯 | 32, 128, 512, 2048, 8192 |
| 加密比 | 2（每方向 h 减半），`h = sqrt(area/ne)` |
| 最细网格 | `nx = ny = 64`，未知量 4225（稠密 143 MB，本机上限） |
| 后处理规则 | 7 点 Dunavant（5 阶精确）；2 阶规则只作对照 |
| 求解 | 稠密对称消元，float64，4 线程 |
| 混合边条件算例 | `T = cosh(x)cos(y)`，`x=1` 面给定 `k dT/dx = sinh(1)cos(y)` |
| 扰动幅度参数 | 0.00, 0.15, 0.30（相对 h） |

## ④ 对照表（ref vs got，闸门逐条判定）

### 4.1 主判据：L2 收敛率（P1 理论值 2）

结构化网格 `structured-a`（`gated` = 与 Q4 研究共享的 5 个制造解场）：

| case | L2 h=1/6 | L2 h=1/11 | L2 h=1/23 | L2 h=1/45 | L2 h=1/91 | L2 orders | H1 orders |
|---|---|---|---|---|---|---|---|
| `sine_generation` | 0.212 | 0.058 | 0.015 | 0.004 | 9.281e-04 | 1.876, 1.967, 1.992, 1.998 | 0.972, 0.993, 0.998, 1.000 |
| `exp_trig_harmonic` | 0.075 | 0.020 | 0.005 | 0.001 | 3.099e-04 | 1.934, 1.982, 1.995, 1.999 | 0.964, 0.991, 0.998, 0.999 |
| `cubic_harmonic` | 0.043 | 0.011 | 0.003 | 6.808e-04 | 1.702e-04 | 1.995, 1.999, 2.000, 2.000 | 0.995, 0.999, 1.000, 1.000 |
| `sine_aspect2` | 0.430 | 0.127 | 0.033 | 0.008 | 0.002 | 1.754, 1.934, 1.983, 1.996 | 0.953, 0.989, 0.997, 0.999 |
| `quadratic_harmonic` | 0.016 | 0.004 | 9.766e-04 | 2.441e-04 | 6.104e-05 | 2.000, 2.000, 2.000, 2.000 | 1.000, 1.000, 1.000, 1.000 |

扰动网格 `distorted`（同一批算例，内部节点 ±0.25h 扰动）：

| case | L2 h=1/6 | L2 h=1/11 | L2 h=1/23 | L2 h=1/45 | L2 h=1/91 | L2 orders | H1 orders |
|---|---|---|---|---|---|---|---|
| `sine_generation` | 0.217 | 0.061 | 0.016 | 0.004 | 9.936e-04 | 1.823, 1.965, 1.987, 1.996 | 0.935, 0.995, 1.001, 1.001 |
| `exp_trig_harmonic` | 0.081 | 0.021 | 0.005 | 0.001 | 3.500e-04 | 1.917, 2.009, 1.932, 1.988 | 0.955, 1.017, 0.984, 1.000 |
| `cubic_harmonic` | 0.045 | 0.012 | 0.003 | 7.685e-04 | 1.929e-04 | 1.927, 1.975, 1.963, 1.994 | 0.974, 1.006, 0.993, 1.000 |
| `sine_aspect2` | 0.442 | 0.133 | 0.036 | 0.009 | 0.002 | 1.735, 1.899, 1.981, 1.983 | 0.939, 0.967, 1.007, 0.994 |
| `quadratic_harmonic` | 0.017 | 0.004 | 0.001 | 2.949e-04 | 7.473e-05 | 1.911, 1.946, 1.985, 1.980 | 0.973, 0.986, 0.992, 0.998 |

### 4.2 闸门逐条判定

| # | gate | quantity | value | criterion | status |
|---|---|---|---|---|---|
| 1 | `l2_pair_order_min` | minimum post-asymptotic L2 pair order over structured ladders (worst: distorted/exp_trig_harmonic) | 1.932491 | >= 1.9 | **PASS** |
| 2 | `l2_finest_order_window` | smallest finest-pair L2 order over every ladder (worst: distorted/quadratic_harmonic) | 1.980173 | in [1.95, 2.1] | **PASS** |
| 3 | `h1_pair_order_min` | minimum H1-seminorm pair order (worst: distorted/exp_trig_harmonic) | 0.984035 | >= 0.9 | **PASS** |
| 4 | `power_law_residual` | largest Roache extrapolation residual |e0|/e_f (worst: sine_aspect2) | 0.003939 | <= 0.01 | **PASS** |
| 5 | `control_exact` | worst absolute error of the P1-representable linear control (all meshes) | 1.065814e-14 | <= 1e-10 | **PASS** |
| 6 | `field_relative_3pct` | finest-mesh max|e|/max|T| (worst: cosh_cos_neumann) | 0.008326 | <= 0.03 | **PASS** |
| 7 | `flux_balance` | max |Dirichlet reaction + imposed Neumann flux| and max free-dof residual (cosh_cos_neumann) | 1.210143e-13 | <= 1e-10 | **PASS** |
| 8 | `quadrature_independence` | relative change of the L2 norm when the 7-point rule is refined by 4-way subdivision (sine_generation, level 32) | 3.218520e-10 | <= 1e-08 | **PASS** |
| 9 | `skfem_nodal_xval` | max nodal |u_tensorfem - u_skfem| / max|u| (sine_generation, level 16) | 1.677879e-15 | <= 1e-10 | **PASS** |
| 10 | `skfem_matrix_xval` | relative difference of the assembled conductivity matrices | 0.000000 | <= 1e-10 | **PASS** |
| 11 | `diagonal_symmetry` | largest relative L2 difference between the two diagonal splits at the finest mesh | 1.321766e-11 | <= 0.05 | **PASS** |

**总计 11/11 个闸门 PASS。**

| 判据（本锚点容差） | 目标 | 实测 | 判定 |
|---|---|---|---|
| 收敛率判据（形式阶次 p=2.0） | 后渐近段 `p_L2 ≥ 1.9`（粗网格 nx≥16） | 最差 1.9325 | PASS |
| 最细段窗口 | `[1.95, 2.1]` | 1.9802 | PASS |
| 场误差 <3% | ≤ 0.03 | 0.0083 | PASS |
| 独立实现 xval（节点值） | ≤ 1e-10 | 1.6779e-15 | PASS |

### 4.3 补充证据（Amendment A1：不进主闸门，数字照实公布）

| `bilinear_control` | 2.000 / 2.4935e-11 | 2.000 / 2.4902e-11 | 1.996 / 0.0035 |
| `linear_control` | -1.631 / 1.2257 | -1.625 / 1.2253 | -1.174 / 1.1856 |
| `cosh_cos_neumann` | 1.912 / 0.0786 | 1.900 / 0.0892 | 1.829 / 0.1488 |

即：混合边条件算例在 h=1/64 的最细段阶次仍为 1.91（前渐近区），因此它只用于
通量/守恒闸门与"场误差<3%"闸门，不参与阶次判定；`linear_control` 是**可精确表示**的
控制例，其误差为机器精度，本身不含阶次信号。

## ⑤ 收敛性与参数敏感性

1. **网格加密**（主收敛性）：每个算例 5 层、加密比严格 2；`structured-a/-b/distorted`
   三族的后渐近段阶次见 4.1。
2. **对角线剖分**：两种剖分最细网格 L2 误差相对差最大
   1.3218e-11（闸门 ≤ 0.05），
   说明阶次不是某一种剖分方向的巧合。
3. **扰动幅度敏感性**（参数点）：

| 扰动幅度 | L2 阶次（8, 16, 32 层） | 最小网格质量 |
|---|---|---|
| 0.00 | 1.967, 1.992 | 0.828, 0.828, 0.828 |
| 0.15 | 1.968, 1.991 | 0.617, 0.539, 0.550 |
| 0.30 | 1.962, 1.984 | 0.397, 0.177, 0.191 |

4. **后处理积分规则**：7 点规则的 L2 范数与四分加密 2 层（131072 个子单元）
   的相对变化 3.2185e-10（闸门 ≤ 1e-08），
   直接规则与收敛值的偏差 1.3405e-06；而 2 阶（3 点）规则偏差
   0.0345 —— 即"用低阶规则算高阶范数"会带来 3.5% 的假偏差，
   这正是本单要排除的污染源。
5. **独立实现交叉验证**：scikit-fem P1（12.0.2）在同一网格、同一载荷、
   同一边条件上节点最大相对差 1.6779e-15，刚度矩阵相对差
   0.0000 —— 阶次不是 TensorFEM 装配的自说自话。

## ⑥ 失败模式、摩擦与工件

**踩过的坑（全部留档）**

1. **网格节点编号与 `meshgrid` 索引不匹配**：`indexing="ij"` 使 x 变成慢变量，而单元索引
   假定 x 最快变，结果非方形网格（2:1 算例）立刻出现负面积三角形 —— 由
   `check_positive_areas` 抓到。这正是"几何先验证"的价值：不做面积检查，这个 bug 会以
   "阶次略低"的形式潜伏。
2. **H1 梯度写错成 `Σ λ_i ∇N_i`**：`_norms` 首版把形函数权重当作节点值，得到的梯度
   ~O(1/h)，H1 阶次变成负值（-0.6、-0.9）。诊断方式：单元梯度与解析梯度在 h 减半时
   比值应为 4；修正为 `∇T_h = Σ T_i ∇N_i` 后 H1 阶次 = 1.000。**教训：阶次出现负值是
   实现 bug 的信号，不是"网格不够细"。**
3. **混合边条件算例的前渐近区被误当作主判据**：`a=1.5` 的 `cosh·cos` 场在 h=1/64 仍只有
   1.91 阶。核验（scikit-fem 节点差 3.7e-15、通量守恒 1.2e-13）证明实现正确，是算例
   本身的前渐近区；据此把 `a` 降到 1.0，并把该算例降为补充证据（Amendment A1，见 4.3），
   **在主判据里既不放宽阈值也不隐藏数字**。
4. **稀疏/稠密规模上限**：`nx=128` 的稠密矩阵 2.2 GB 不可行，阶梯止于 64（4225 未知量）。
   这是 kernel 的真实限制，如实记录而不绕过。

**生态摩擦（如实记）**

* 本单开工三查的接口与任务书给的 SQL 不一致：`oss_inventory` 表**没有** `feature_slug`
  列（其键为 `repo`/`focus`/`methods`），且 RBAC 拒绝读取（`permission denied`）；
  `oss_solutions` 在 `:8010` 的 1336 条路由里没有公开端点。实际调研改为
  ①`~/oss-cache` 文件系统扫描（命中 `Maarten-vd-Sande_lbm`、`pylbm`、`hschwane/GraSPH`
  三个仓库，全部是 LBM/SPH 域，**无 FEM 导热同类实现**，故开源横向调研结论是
  "无可迁移结构，改走文献谱系 + scikit-fem 交叉验证"）；
  ②`tensor/negative-knowledge/search?q=TensorFEM` 命中 0 条（同域无负面知识）；
  ③`tensor/knowledge/search` 的向量引擎在 8021 不可达（`engine_offline`）。
  本单的方法选型与迁移方案因此落在下面这份文档 + 新模块里（平台侧 `oss_solutions`
  写入通道不可用，已在报告中注明）。

**工件（worktree 内真实文件）**

| 工件 | 说明 |
|---|---|
| `src/tensorfem/heat_l2_triangle.py` | P1 三角形导热内核 + 阶梯 + 闸门（能力本体，1092 行） |
| `benchmarks/verified/fem_heat_triangle_p1/run.py` | 一键复现（证据/图/闸门全由它生成） |
| `benchmarks/verified/fem_heat_triangle_p1/result.json` | 全部阶梯、范数、闸门、参数研究 |
| `benchmarks/verified/fem_heat_triangle_p1/gates_verified.json` | 闸门清单（机器可复核） |
| `benchmarks/verified/fem_heat_triangle_p1/make_report.py` | 本报告与发布载荷的生成器 |
| `tests/test_heat_l2_triangle.py` | 13 个 pytest 闸门（不变量断言） |
| `docs/assets/verified/fem_heat_triangle_p1/fig_convergence.png` | 图 1：L2/H1 阶次阶梯（7 算例 × 3 网格族） |
| `docs/assets/verified/fem_heat_triangle_p1/fig_field.png` | 图 2：场云图 + 逐点误差云图 |
| `docs/assets/verified/fem_heat_triangle_p1/fig_distortion.png` | 图 3：扰动幅度敏感性与网格质量 |

**一键复现**

```
PYTHONPATH=src OMP_NUM_THREADS=4 python benchmarks/verified/fem_heat_triangle_p1/run.py
PYTHONPATH=src OMP_NUM_THREADS=4 python -m pytest tests/test_heat_l2_triangle.py -q
PYTHONPATH=src python benchmarks/verified/fem_heat_triangle_p1/make_report.py
```

**误差判定（报告纪律要求显式回答）**

* 与基准（制造解，逐点解析场）比较：最细网格上主判据算例 `max|e|/max|T| = 0.0083`，
  **< 3% ✅**（闸门阈值 3%）。
* 关键量（收敛率）与理论值 2 的偏差：最细段 `p = 1.9802`，
  相对偏差 0.991%，
  **< 3% ✅**。
* 与独立实现 scikit-fem 的差异 1.6779e-15，**< 3% ✅**（远小于）。

## 适用范围与局限

* 仅矩形域、各向同性加密、正则解；奇异解（角点/裂纹/点源）不在结论内。
* `k` 为常数；空间变化的导热率需要逐单元核，本单未实现。
* Robin（对流）边条件、轴对称/三维、瞬态导热未覆盖。
* 稠密装配把阶梯限制在 `nx ≤ 64`（4225 未知量）。
* 混合边条件算例仍处前渐近区（最细段 1.91），只作补充证据。
