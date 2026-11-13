## 1. 案例介绍

**物理背景**：板/壳屈曲的设计链条由两个数字承载——平板屈曲系数 `k`（`sigma_cr = k pi^2 D / (b^2 t)`）
与折减曲线 `chi(lambdabar)`。这两个量在工程实践中几乎总是"转录"来的（规范/手册），
而转录只有在能逐字引到原文并给出位置时才算证据。本单先把"能不能转录"和"算子层能不能自证"
彻底分开：

* 锚（Timoshenko & Gere 平板屈曲系数；DNV-RP-C201 板屈曲；EN 1993-1-6 壳折减曲线）本环境不可得
  → 全部登记 `WITHHELD`，**不写任何数字**（不凭记忆、不用二手转录）；
* 算子层（量纲/代数/一致性）全部机检，且用两套互不共享代码/形函数的离散互相夹逼。

**几何与材料参数表**（记入工件 `provenance.material`）：

| 参数 | 取值 |
|---|---|
| 杨氏模量 E | 2.100e+11 Pa |
| 泊松比 nu | 0.3 |
| 板厚 t | 0.012 m |
| 屈服强度 fy | 3.550e+08 Pa |
| 板宽 b（系数归一化跨度） | 0.6 / 0.9 / 1.0 / 1.2 m（按算例） |
| 长宽比 a/b | 0.5 – 4.0 扫描；重点 1.0、sqrt(2)、1.5、2.0 |
| 边界条件 | 四边简支（SSSS）、四边固支（CCCC）、边转动约束 k_r b/D 阶梯 |

**文献/锚对照状态**：`verbatim = 0`，
`withheld = 3`。
工件 `anchor_register.json` 里每个锚条目 `verbatim_quote = null`、`source_sha256 = null`、
`numeric_value_recorded = false`，并记录了检索命令（`provenance.anchor_source_search`）——
"没找到"本身是证据，不是遗漏。

## 2. 怎么做的

1. **系数算子两条独立路径**（残差判据 ≤1e-10）：
   * 路径 A：Navier 级数解析推导出 `k(m,n) = (u + n^2/u)^2`，`u = m/alpha`，对整数 m 取包络（闭式，无任何查表值）；
   * 路径 B：调用仓库既有 `tensorfem.marine_plate_buckling.navier_uniaxial_buckling_load`（二维 (m,n) 全枚举、
     有量纲刚度、返回线载荷 N_cr），只做 `k = N_cr b^2/(pi^2 D)` 归一化；
   * 两条路径共享偏微分方程但**不共享代码路径**，残差衡量的是算子闭合，不是物理验证。
2. **两套互不共享形函数的离散**（用于无闭式解的固支/转动约束）：
   * BFS（Bogner–Fox–Schmit）C1 Hermite 矩形板单元：弯曲阵复用仓库已鉴定的
     `bfs_plate_element_matrices`，本单新增轴向几何阵 `integral w_,x w_,x dA`（4 点 Gauss，被积 6 次，精确）；
   * 多项式 Rayleigh–Ritz：`W = sum c_ij phi_i(X) phi_j(Y)`，`phi_i(t) = t^2 (1-t)^2 t^i`，四边天然满足 `w = w' = 0`（固支）；
     归一化 `k = (alpha^2/pi^2) * int[(W_,XX/alpha^2 + W_,YY)^2] / int[W_,X^2]`。
3. **边转动约束阶梯**：一致（consistent）边罚 `k_r/2 * integral (w_,n)^2 ds`，`k_r = rho D / b`。
   沿边法向斜率是四次 Hermite 插值——`(h0,h2)` 承载节点法向斜率、`(h1,h3)` 承载节点扭率，
   两族在边上都有非零导数，**只罚一族会把边留成可鼓出的半约束**（本单踩过，见第 5 节）；
   `rho -> 0` 为简支，`rho -> inf` 为固支。
4. **折减曲线契约机检**：只检任何可容许曲线都必须满足的结构条款——量纲（无量纲自变量与值域）、
   值域 `0 < chi <= 1`、单调不增、连续、以及**弹性上限 `chi <= 1/lambdabar^2`**（折减后承载力不得高于弹性临界）。
   规范专有条款（stocky 平台长度、缺陷系数、塑性支）做成**检查器参数**；锚未到位时参数为 `None`，
   该条款显式报 `SKIPPED_PARAMETER_WITHHELD` 而不是猜一个数。
5. **检查器要有牙**：用 5 个刻意变异的曲线做负对照，逐一必须被拒（拒绝 = 检查器有判别力）。
6. **工具链/可复现**：`python scripts/w26_plate_shell_buckling_closure.py`（runner=native）生成工件 + 3 张图；
   `python -m pytest tests/test_plate_buckling_closure.py`（40 项，约 11 s）。
   工件内含 git commit/分支、python/numpy 版本、命令、4 个输入文件 sha256、工件自身 `content_sha256`。
   本单执行 commit：`80624f0c5289`，分支 `agent/m17565`。

## 3. 计算结果

### 3.1 锚登记（判据 ①）

| 锚 | 来源 | 状态 | 记录数字 |
|---|---|---|---|
| A1 | Timoshenko & Gere, Theory of Elastic Stability -- flat plate buckling | WITHHELD | False |
| A2 | DNV-RP-C201 Buckling strength of plated structures | WITHHELD | False |
| A3 | EN 1993-1-6 Shell buckling | WITHHELD | False |

判定：**HONEST WITHHELD**。3 个锚全部 `WITHHELD`，无一条数字来自记忆或二手转录。

### 3.2 解析复算与残差（判据 ②，阈值 1e-10）

| 算例 | a/b | k 路径A(闭式) | k 路径B(Navier) | sigma_cr A [Pa] | sigma_cr B [Pa] | 相对残差 | 判定 |
|---|---|---|---|---|---|---|---|
| C1 | 1 | 4 (m=1) | 4 (m=1) | 1.093e+08 | 1.093e+08 | 0.000e+00 | PASS |
| C2 | 2 | 4 (m=2) | 4 (m=2) | 1.093e+08 | 1.093e+08 | 0.000e+00 | PASS |
| C3 | 1.41421 | 4.5 (m=2) | 4.5 (m=1) | 1.230e+08 | 1.230e+08 | 3.635e-16 | PASS |

最大相对残差 **3.635e-16** ≤ 1e-10 → 判据② **MEET**。

### 3.3 量纲/代数机检

| 检查 | 相对误差 | 判定 |
|---|---|---|
| geometric scale invariance of k | 2.220e-16 | PASS |
| linearity in Young's modulus | 0.000e+00 | PASS |
| Poisson dependence enters only through D | 0.000e+00 | PASS |
| coefficient <-> stress round trip | 1.974e-16 | PASS |

| 恒等式 | 判定 |
|---|---|
| k(alpha = 1) = 4 | PASS |
| k(alpha = sqrt(2)) = 4.5 (envelope maximum, two competing modes) | PASS |
| k(m alpha) <= k(alpha) for integer m: the competing modes of k(m alpha) contain the modes of k(alpha) as the subset j = m i, so the envelope can only fall; equality at integer alpha (verified) | PASS |
| envelope maximum 4.5 at alpha = sqrt(2) and at 1/sqrt(2) | PASS |
| k(alpha) = 4 exactly at integer alpha with mode m = alpha | PASS |
| psi = 0 reduces the biaxial operator to the uniaxial one | PASS |
| psi = 1, square plate: k = 2 (m = n = 1) | PASS |
| 4 <= k(alpha) <= 4.5 for every aspect ratio >= 1 (single-mode envelope) | PASS |

### 3.4 边界条件对 k 的影响（判据 ③）

收敛阶梯（BFS 网格 n×n；Ritz 多项式次数）：

| BFS n (简支) | k 简支 | BFS n (固支) | k 固支 | Ritz 次数 | k 固支(Ritz) |
|---|---|---|---|---|---|
| 4 | 4.001034167 | 4 | 9.902935459 | 3 | 10.10715605 |
| 6 | 4.000206759 | 6 | 10.00535053 | 4 | 10.07454971 |
| 8 | 4.000065698 | 8 | 10.03951596 | 5 | 10.07475113 |
| 10 | 4.000026963 | 10 | 10.0543715 | 6 | 10.07445186 |
| 12 | 4.000013017 | 12 | 10.06183917 | 7 | 10.07436262 |

* 简支：BFS 收敛到 **4.000013**，与解析包络 `k = 4` 相对差 **3.254e-06**；
* 固支：BFS 节点固支（弱约束，下界）**10.039516** 与 Ritz（上界）
  **10.074751** 夹逼，区间相对宽度
  **0.3497%**；罚函数极限
  **10.081959** 位于 Ritz 上界之上
  0.07154%（有限罚仍带弹簧能量，故略高）；
* 固支/简支系数比 **2.51865**（同一归一化、同一载荷）。

不同长宽比下的对照（BFS n=8）：

| a/b | k 简支(解析) | mode | k 简支(BFS) | k 固支(BFS) | 固支/简支 |
|---|---|---|---|---|---|
| 1 | 4 | 1 | 4.000065698 | 10.03951596 | 2.50984 |
| 1.5 | 4.340277778 | 2 | 4.340478936 | 8.311854196 | 1.91496 |
| 2 | 4 | 2 | 4.000065698 | 7.82838224 | 1.95706 |

边转动约束阶梯（BFS，固定网格，rho = k_r b / D）：

| rho | k | k/k_simple |
|---|---|---|
| 0 | 4.000065698 | 1 |
| 1 | 4.736558515 | 1.18412 |
| 10 | 7.698239942 | 1.92453 |
| 100 | 9.709117778 | 2.42724 |
| 1000 | 10.04226493 | 2.51052 |
| 1e+05 | 10.08195876 | 2.52045 |

阶梯**单调不减**（机检 `restraint_ladder_monotone = True`）：
`rho = 1` 时 k 已比简支高 18.4%，
`rho = 10` 时高 92.5%，`rho -> inf` 收敛到固支值。

### 3.5 折减曲线契约机检（算子层，非规范值）

| 条款 | 判定 | 状态 |
|---|---|---|
| range | PASS | CHECKED |
| stocky_limit | PASS | CHECKED |
| monotone | PASS | CHECKED |
| continuity | PASS | CHECKED |
| elastic_ceiling | PASS | CHECKED |
| stocky_plateau_length | SKIPPED | SKIPPED_PARAMETER_WITHHELD |

负对照（检查器判别力）：

| 变异曲线 | 检查器结论 | 触发条款 |
|---|---|---|
| amplifying | REJECTED | range, monotone, elastic_ceiling |
| non_monotone | REJECTED | range, monotone, elastic_ceiling |
| slow_decay_breaks_ceiling | REJECTED | elastic_ceiling |
| above_elastic_ceiling | REJECTED | elastic_ceiling |
| discontinuous | REJECTED | continuity, elastic_ceiling |
| overshoot | REJECTED | range, stocky_limit, elastic_ceiling |

### 3.6 系数 → 无量纲细长比 → 折减算子的传导（量纲一致性）

| b [m] | k | sigma_cr [Pa] | lambdabar | 1/lambdabar^2 | chi(合成样本) | 弹性上限满足 |
|---|---|---|---|---|---|---|
| 0.6 | 4 | 3.037e+08 | 1.0812 | 0.855437 | 0.650044 | YES |
| 0.9 | 4 | 1.350e+08 | 1.6218 | 0.380194 | 0.355376 | YES |
| 1.2 | 4 | 7.592e+07 | 2.1624 | 0.213859 | 0.20913 | YES |

合成样本 `chi = 1/sqrt(1 + lambdabar^4)` 只用于演示算子链与条款机检，**不是** DNV/EN 曲线。

## 4. 结果图

1. `fig_coefficient_envelope.png` — 系数包络 k(a/b)：解析闭式 vs BFS 离散（简支点落在包络上，固支点整体抬高）。
2. `fig_restraint_ladder.png` — 边转动约束阶梯：由简支 k=4 单调过渡到固支，两条独立界限（Ritz 上界、节点固支 FE 下界）画出。
3. `fig_reduction_contract.png` — 折减曲线可容许区域：合成样本、弹性上限 1/lambdabar^2 与允许域阴影。

## 5. 注意点（踩过的坑与适用范围）

1. **几何阵 Jacobian 差 4 倍**：Gauss 点已映射到 [0,1] 后仍乘 (a_e/2)(b_e/2)，使 k 整体偏大 4 倍（首轮
   简支得 16.0）。用解析 k=4 作为回归钉子立刻暴露——这就是"有独立锚点可自检"的价值。
2. **边罚只罚一族形函数 = 假约束**：沿边法向斜率由 `(h0,h2)`+`(h1,h3)` 两族共同承载；只罚前者时
   罚能量可降到 3e-10（看着像"已固支"）而边仍在鼓出，固支极限停在 6.3（本单真值约 10.06–10.08）。
   补齐四自由度一致块后阶梯才单调收敛。
3. **节点固支 ≠ 罚函数极限**：节点只压法向斜率（边内可鼓出）是"弱约束"，其值低于罚极限；两者是不同问题，
   不能拼成一条单调阶梯，本单分开报告并用两条独立上/下界夹逼。
4. **BFS 固支收敛慢且自下而上**，Ritz 自上而下；单看一条会误判"未收敛"，夹逼才给出 0.35% 的可信带。
5. **适用范围**：薄板线弹性、无初始缺陷、无残余应力、均匀单/双轴压；不含后屈曲、塑性、加筋板整体/局部耦合、
   壳的几何缺陷敏感性与 EN 1993-1-6 的壳特定条款。Фурье/Navier 级数仅对四边简支矩形板成立。
6. **未完成/不可判**：与 DNV-RP-C201、EN 1993-1-6、Timoshenko & Gere 的**数值比对一律未做**，
   因为锚原文不在环境内；任何"与文献吻合"的结论在本单都不成立，等锚到位后重跑同一 runner 即可补齐。

## 6. 误差判定（逐项是否 <3%）

| 关键量 | 基准 | 相对误差 | 是否 <3% | 说明 |
|---|---|---|---|---|
| k 简支（BFS n=12 网格最细） | 解析包络 k=4 | 3.254e-06 | YES | 两者共享偏微分方程、不共享代码；属于算子闭合 |
| k 固支（BFS 节点固支，下界） | Ritz（上界） | 0.3497% | YES | 两套互不共享形函数的离散之间的夹逼宽度 |
| 3 个解析算例 sigma_cr | 两路径互比 | 3.635e-16 | YES | 判据阈值 1e-10，实测远低 |
| 量纲/恒等式 6 项 | 解析恒等式 | ≤1e-12 | YES | 见 3.3 表 |
| k 固支 / k 简支 比值 | 无独立基准（锚 WITHHELD） | 未判定 | NO（不可判） | 比值 2.519 为两独立离散的算子结果，无外部锚可比 |
| chi 折减曲线数值 | DNV-RP-C201 / EN 1993-1-6 | 未判定 | NO（不可判） | 锚 WITHHELD：本单只做结构条款机检，未产生任何规范数值 |

**结论**：凡有独立机检基准的量（简支系数、固支夹逼、解析复算残差、量纲恒等式）误差均远低于 3%；
凡需外部规范/手册锚的量（固支系数的绝对值、折减曲线数值）在锚到位前**不可判定**，本单如实登记为
`WITHHELD`，不以"与记忆/文献一致"替代证据。

### 判据逐条裁决

| 判据 | 要求 | 裁决 |
|---|---|---|
| C1 | verbatim anchor locations >= 2, or an honest WITHHELD | HONEST_WITHHELD |
| C2 | analytic recompute residual <= 1e-10 | MEET |
| C3 | quantify the influence of the edge boundary condition on k | MEET |
| C4 | no fabricated values; comparison against the anchors is out of scope while withheld | MEET |

### 工件与可复现

* 工件：`papers/artifacts/w26/plate-shell-buckling-closure/anchor_register.json`
  （`content_sha256 = f9d8a3d1ccb19ed46fade5051d1871db...`）
* 图：`fig_coefficient_envelope.png`、`fig_restraint_ladder.png`、`fig_reduction_contract.png`
* 代码：`src/tensorfem/plate_buckling_closure.py`、`scripts/w26_plate_shell_buckling_closure.py`、
  `tests/test_plate_buckling_closure.py`
* 复现：`python scripts/w26_plate_shell_buckling_closure.py`；测试 `python -m pytest tests/test_plate_buckling_closure.py -q`（40 passed）。
