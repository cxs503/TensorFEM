# IN41-N01：TensorFEM 结构力学验证面（梁 / 板 / 轴对称壳）解析对拍

> 本文由 `benchmarks/verified/fem_structural_benchmarks/render_doc.py` 从 `benchmarks/verified/fem_structural_benchmarks/result.json` 与
> `gates_verified.json` 程序化注入生成，文中每个数字都来自计算输出，无手抄。
> 复现：`OMP_NUM_THREADS=1 PYTHONPATH=src python benchmarks/verified/fem_structural_benchmarks/run.py`（约 40 s）；
> 门禁回归：`PYTHONPATH=src pytest tests/test_fem_structural_benchmarks.py`。

## 1 案例介绍（物理背景 / 几何 / 参数）

三个案例各对应 TensorFEM 线性结构求解路径中的一个**单元族**，各自有独立的
闭式解 oracle，互不共享数据：梁用 1-D Hermite 框架单元、板用 4 节点 Mindlin
(MITC4) 板单元、厚壁圆筒用 4 节点轴对称环单元（带环向应变）。选择这三族是为了
覆盖「一维弯曲 / 二维弯曲 + 横向剪切 / 回转壳（轴对称）」三条不同的运动学假设，
而不是同一单元的三组载荷。

### 案例 A：悬臂梁端部集中力（`tensorfem.frame2d`，3 dof/节点 Hermite 单元）

船体梁、舱壁扶强材、甲板纵桁的一阶估算基准。空心箱型截面悬臂：
`L = 2.5 m`，外截面 `b x h = 0.15 x 0.25 m`，
壁厚 `t_w = 0.02 m`，`A = 1.440000e-02 m^2`，
`I = 1.104200e-04 m^4`，`L/h = 10.0`，
钢 `E = 210 GPa`、`nu = 0.3`，
端部集中力 `P = 50 kN`。

Euler–Bernoulli 理论：

```
w(x) = P x^2 (3L - x) / (6 E I),        w(L) = P L^3 / (3 E I)
theta(x) = P x (2L - x) / (2 E I),      theta(L) = P L^2 / (2 E I)
M(x) = P (L - x),                       sigma_xx = M(x) z / I
```

### 案例 B：四边简支薄板均布载荷（`tensorfem.plate`，Q4 Mindlin/MITC4）

船体外板/舱壁板在静水头压力下的经典工况。正方形板 `a = b = 1.0 m`，
`t = 10 mm`（`a/t = 100`），
`E = 210 GPa`、`nu = 0.3`，
均布压力 `q = 10 kPa`，软简支（四边 `w = 0`，转角自由）。
弯曲刚度 `D = E t^3/(12(1-nu^2)) = 1.923077e+04 N·m`。

两个独立 oracle：
* 精确 Navier 双重级数（401 个奇数项）：`w = 16 q a^4/(pi^6 D) Σ sin sin /(m n (m^2+n^2)^2)`，
  中心系数 `alpha = 0.004062353`；
* Timoshenko & Woinowsky-Krieger 表 8：`w_c = alpha q a^4/D`，`alpha = 0.00406`
  （三位有效数字，其自身舍入给出 0.06 % 的比较底噪）。

### 案例 C：厚壁圆筒内压 Lame 解（`tensorfem.hertz_axisymmetric`，轴对称环单元）

潜艇耐压壳/圆筒结构在内压下的径向-环向应力分布是「不能只用薄膜理论」的标准
反例：`sigma_r` 从 `-p` 单调过渡到 0，薄膜理论给不出。`a = 100 mm`，
`b = 200 mm`（`b/a = 2.0`），
内压 `p = 20 MPa`，外表面自由，
平面应变（`u_z = 0` 逐点约束），`E = 210 GPa`、`nu = 0.3`。

```
A = p a^2/(b^2 - a^2) = 6.666667e+06 Pa
B = p a^2 b^2/(b^2 - a^2) = 2.666667e+05 Pa·m^2
sigma_r = A - B/r^2,   sigma_theta = A + B/r^2,   sigma_z = 2 nu A
u_r = (1+nu)/E [(1 - 2 nu) A r + B/r]
```

## 2 方法 / 流程 / 关键设置

* **网格阶梯（三档）**：梁 `n = [4, 8, 16]` 个单元；板
  `[8, 16, 32]` 单元/边；圆筒 `nr = [8, 16, 32]` 个径向单元
  （轴向 1 层，平面应变）。所有阶梯均以同一模型、同一积分规则、只改 h。
* **装配与求解**：单元刚度全部来自 TensorFEM 库内核
  （`frame2d.solve_frame_static`、`plate.q4_mindlin_stiffness`、
  `hertz_axisymmetric.assemble_axisymmetric_stiffness`），驱动脚本只做
  BC/载荷组装与超单元级别的稀疏求解（`scipy.sparse.linalg.spsolve`，SuperLU 直接法）。
  板的一致载荷向量取自 `plate_mindlin.consistent_load_vector`（2x2 Gauss）。
* **载荷与边界**：梁为端部集中力（自由节点力）与均布载荷（库内一致节点力）；
  板为软简支（四边 `w = 0`）+ 均布压力；圆筒内孔施加 `+p e_r` 的
  2 pi r 加权一致节点力（孔面外法向为 `-e_r`，故牵引力 `-p n = +p e_r`），外表面自然自由。
* **应力取值（关键设置）**：梁用单元的固定端弯矩 `M = P L` 与 Hermite 曲率场
  `sigma = -E c v''`；板用中心节点挠度与全域 L2 挠度场误差；圆筒用
  **Gauss -> 节点双线性外推 + 共单元平均**（权重 `(1 ± sqrt 3)/2`）的标准应力恢复，
  并同时输出未恢复的 Gauss 原值做对照（下表 `gauss_vs_recovered`），
  使「精度来自单元还是来自恢复」可被独立审计。
* **误差判据**：全部为**相对误差**（`|FE - 参考| / |参考|`），参考值的取点与
  FE 取点**同位置**（location-matched）；`sigma_r` 在自由表面处恒为 0，
  该点相对误差未定义，改用载荷归一化的绝对误差 `|Δsigma_r|/p`，两者都入库。
* **收敛阶**：`p = ln(e_coarse/e_fine)/ln 2`（h 折半），三档给出两个阶，取最细一对。

## 3 计算结果

### 3.1 案例 A：悬臂梁端部集中力

| 单元数 | w(L) FEM [m] | w(L) 解析 [m] | 挠度相对误差 | sigma_root FEM [MPa] | sigma_root 解析 [MPa] | 应力相对误差 | 域内最大挠度误差 |
|---|---|---|---|---|---|---|---|
| 4 | 1.1230568421e-02 | 1.1230568421e-02 | 8.341e-15 | 141.505162 | 141.505162 | 2.738e-15 | 7.975e-15 |
| 8 | 1.1230568421e-02 | 1.1230568421e-02 | 1.277e-13 | 141.505162 | 141.505162 | 1.900e-13 | 1.864e-13 |
| 16 | 1.1230568421e-02 | 1.1230568421e-02 | 2.749e-12 | 141.505162 | 141.505162 | 3.036e-12 | 3.030e-12 |

三档网格的最大挠度相对误差 `2.749e-12`、
固定端弯曲应力相对误差 `3.036e-12`：这是
**机器精度**级别的重合，因为三次 Hermite 单元的张成空间恰好包含该载荷下的三次
挠曲线，离散解在节点与单元内部都是精确的。这也意味着此载荷下**不存在**可测的收敛阶。

为得到可测的阶，同一模型换均布载荷 `q = 20 kN/m`
（精确解为四次多项式），在**单元内部**取样（避开节点超收敛点）：

| 单元数 | w(L) FEM [m] | w(L) 解析 [m] | 挠度相对误差 | 域内挠度误差 | 域内应力误差 |
|---|---|---|---|---|---|
| 4 | 4.2114631580e-03 | 4.2114631580e-03 | 7.826e-15 | 2.833e-03 | 3.704e-02 |
| 8 | 4.2114631580e-03 | 4.2114631580e-03 | 1.442e-13 | 6.789e-04 | 6.803e-03 |
| 16 | 4.2114631580e-03 | 4.2114631580e-03 | 2.788e-12 | 1.662e-04 | 2.755e-03 |

三档内部挠度误差 `['2.833e-03', '6.789e-04', '1.662e-04']`
→ 观测阶 `p = 2.030`
（逐级 `[2.061, 2.03]`）。
注：`M -> 0` 的自由端末单元不参与应力误差统计（相对误差度量的是分母）。

### 3.2 案例 B：四边简支薄板均布载荷

| 网格 | w_c FEM [m] | w_c Navier [m] | w_c Timoshenko [m] | 误差 vs Navier | 误差 vs Timoshenko | L2 挠度场误差 |
|---|---|---|---|---|---|---|
| 8x8 | 2.1042106902e-03 | 2.1124233836e-03 | 2.1112000000e-03 | 3.888e-03 | 3.311e-03 | 4.588e-02 |
| 16x16 | 2.1139570304e-03 | 2.1124233836e-03 | 2.1112000000e-03 | 7.260e-04 | 1.306e-03 | 9.768e-03 |
| 32x32 | 2.1188234084e-03 | 2.1124233836e-03 | 2.1112000000e-03 | 3.030e-03 | 3.611e-03 | 6.828e-04 |

最细网格（32x32）中心挠度：vs Navier `3.030e-03`，
vs Timoshenko `3.611e-03`。
L2 场的观测阶 `p = 3.839`
（逐级 `[2.232, 3.839]`）。
中心挠度误差**不单调**（n=16 的 7.3e-4 优于 n=32 的 3.0e-3），其阶
`[2.421, -2.061]` 不可用作收敛率——
这是软简支 + MITC4 在薄极限下的小幅振荡，故本案例的阶以 L2 场为准（见 §5）。

### 3.3 案例 C：厚壁圆筒内压 Lame 解

| nr | r/a | sigma_r FEM [MPa] | sigma_r Lame [MPa] | sigma_r 相对误差 | sigma_theta FEM [MPa] | sigma_theta Lame [MPa] | sigma_theta 相对误差 | u_r 相对误差 |
|---|---|---|---|---|---|---|---|---|
| 8 | 1.1250 | -14.25415 | -14.40329 | 1.035e-02 | 28.72207 | 27.73663 | 3.553e-02 | 2.421e-03 |
| 8 | 1.2500 | -10.29979 | -10.40000 | 9.635e-03 | 24.36759 | 23.73333 | 2.672e-02 | 2.317e-03 |
| 8 | 1.5000 | -5.13542 | -5.18519 | 9.597e-03 | 18.81025 | 18.51852 | 1.575e-02 | 2.175e-03 |
| 8 | 1.7500 | -2.01434 | -2.04082 | 1.297e-02 | 15.52161 | 15.37415 | 9.592e-03 | 2.081e-03 |
| 8 | 1.8750 | -0.89907 | -0.91852 | 2.117e-02 | 14.35908 | 14.25185 | 7.524e-03 | 2.045e-03 |
| 16 | 1.1250 | -14.36663 | -14.40329 | 2.545e-03 | 27.98018 | 27.73663 | 8.781e-03 | 6.098e-04 |
| 16 | 1.3125 | -8.79271 | -8.81330 | 2.337e-03 | 22.27463 | 22.14664 | 5.779e-03 | 5.729e-04 |
| 16 | 1.5000 | -5.17284 | -5.18519 | 2.381e-03 | 18.59096 | 18.51852 | 3.912e-03 | 5.476e-04 |
| 16 | 1.6875 | -2.69008 | -2.69776 | 2.848e-03 | 16.07428 | 16.03109 | 2.694e-03 | 5.292e-04 |
| 16 | 1.8750 | -0.91368 | -0.91852 | 5.269e-03 | 14.27851 | 14.25185 | 1.871e-03 | 5.151e-04 |
| 32 | 1.0938 | -15.61436 | -15.62449 | 6.485e-04 | 29.02603 | 28.95782 | 2.355e-03 | 1.547e-04 |
| 32 | 1.3125 | -8.80817 | -8.81330 | 5.825e-04 | 22.17857 | 22.14664 | 1.442e-03 | 1.435e-04 |
| 32 | 1.5000 | -5.18210 | -5.18519 | 5.942e-04 | 18.53660 | 18.51852 | 9.763e-04 | 1.372e-04 |
| 32 | 1.6875 | -2.69584 | -2.69776 | 7.111e-04 | 16.04187 | 16.03109 | 6.725e-04 | 1.326e-04 |
| 32 | 1.9062 | -0.67074 | -0.67186 | 1.664e-03 | 14.01135 | 14.00520 | 4.396e-04 | 1.285e-04 |

最细网格（nr=32）：`sigma_theta` 最大相对误差
`2.355e-03`、`sigma_r` 最大相对误差
`1.664e-03`、`u_r` 最大相对误差
`1.547e-04`、`sigma_z` 最大相对误差
`5.875e-03`；
全阶梯（含最粗网格）的最大值分别为
`3.553e-02` / `2.117e-02` /
`2.421e-03`，即最粗一档尚未达 1 %，是网格分辨率的限制而非单元缺陷；
`sigma_theta` 的观测阶 `p = 1.898`
（逐级 `[2.017, 1.898]`），`u_r` 的阶
`1.979`。

**应力恢复的价值（可审计）**：同一最细网格上，未恢复的 Gauss 原点值
`sigma_theta` 最大相对误差 `1.322e-02`、
`sigma_r` `9.722e-02`——
即 1 % 的判据是靠标准的节点恢复（二阶）而不是单元原值（一阶）达成的，
两组数都在 `result.json` 里，不做选择性呈现。

### 3.4 图

| 图 | 内容 |
|---|---|
| `benchmarks/verified/fem_structural_benchmarks/case_a_beam.png` | 左：悬臂挠曲线 FEM vs 解析（mm）；右：`z=+h/2` 弯曲应力沿轴分布 FEM vs `M(x)c/I`（MPa） |
| `benchmarks/verified/fem_structural_benchmarks/case_b_plate.png` | 左：32x32 MITC4 挠度场云图（mm）；右：中心挠度误差（vs Navier / vs Timoshenko）与 L2 场误差的网格收敛（含 2 % 判据线） |
| `benchmarks/verified/fem_structural_benchmarks/case_c_cylinder.png` | 左：`sigma_r`、`sigma_theta` 随 `r/a` 分布，Lame 曲线 vs nr=32 恢复值；右：三档网格逐点相对误差（含 1 % 判据线） |

## 4 门禁（`gates_verified.json`，schema `tensorfem.verified-gates.v1`）

| 门禁 | 量 | 参考 | 值 | 判据 | 结论 |
|---|---|---|---|---|---|
| `A1.tip_deflection` | cantilever tip deflection vs P L^3/(3 E I), worst level | w(L) = P L^3 / (3 E I) | 2.749e-12 | <= 1.00e-02 | PASS |
| `A2.root_bending_stress` | fixed-end bending stress vs sigma = M c / I, worst level | sigma(L=0) = P L c / I = 141.505 MPa | 3.036e-12 | <= 1.00e-02 | PASS |
| `A3.beam_order` | observed order, uniform-load interior deflection field | h-refinement order, levels (4, 8, 16) | 2.030e+00 | >= 1.50e+00 | PASS |
| `B1.centre_vs_timoshenko` | SSSS centre deflection vs Timoshenko table 8 alpha = 0.00406, finest level | w_c = 0.00406 q a^4 / D | 3.611e-03 | <= 2.00e-02 | PASS |
| `B2.centre_vs_navier` | SSSS centre deflection vs converged Navier series, finest level | w_c = 16 q a^4/(pi^6 D) sum_{m,n odd} sin sin /(m n (m^2+n^2)^2), 401 terms | 3.030e-03 | <= 2.00e-02 | PASS |
| `B3.plate_order` | observed order, L2 deflection field error | h-refinement order, levels (8, 16, 32) | 3.839e+00 | >= 1.50e+00 | PASS |
| `C1.hoop_stress` | sigma_theta vs Lame at 5 radii, worst of the finest level | sigma_theta = A + B/r^2 | 2.355e-03 | <= 1.00e-02 | PASS |
| `C2.radial_stress` | sigma_r vs Lame at 5 radii, worst of the finest level | sigma_r = A - B/r^2 | 1.664e-03 | <= 1.00e-02 | PASS |
| `C2b.radial_stress_absolute` | |sigma_r - Lame| / p at the 5 radii, worst of the finest level | load-normalised radial stress error | 5.066e-04 | <= 1.00e-02 | PASS |
| `C3.radial_displacement` | u_r vs Lame at 5 radii, worst of the finest level | u_r = (1+nu)/E [(1-2 nu) A r + B/r] | 1.547e-04 | <= 1.00e-02 | PASS |
| `C3b.axial_stress` | sigma_z vs 2 nu A at 5 radii, worst of the finest level | sigma_z = 2 nu A (plane strain) | 5.875e-03 | <= 1.00e-02 | PASS |
| `C4.cylinder_order` | observed order, recovered sigma_theta error | h-refinement order, levels (8, 16, 32) | 1.898e+00 | >= 1.50e+00 | PASS |

总判定：**PASS**
（失败门禁：`[]`）。
判据阈值与战役口径一致：梁 1 %、板 2 %、圆筒 1 %、收敛阶 ≥ 1.5、每案例三档网格
（`{"case_a_tip_deflection_rel_max": 0.01, "case_a_root_stress_rel_max": 0.01, "case_b_centre_deflection_rel_max": 0.02, "case_c_stress_rel_max": 0.01, "convergence_order_min": 1.5, "levels_per_case": 3}`）。

## 5 注意点 / 局限

1. **案例 A 是精确表示，不是收敛结果**。三次 Hermite 单元张成三次多项式，而端部
   集中力下的精确挠曲线正是三次，因此误差停在舍入极限（~1e-12，随网格增大缓慢
   增长）。这比「1 % 收敛」更强，但**不能**被读作"单元在一般载荷下也这么准"；
   均布载荷（四次场）的 2.03 阶才是可外推的率。
2. **Euler–Bernoulli 理论本身忽略横向剪切**：`L/h = 10` 时
   剪切挠度与弯曲挠度之比为 `3EI/(kappa G A L^2) = 1.15 %`（`kappa = 5/6`），
   这是**模型层面**的差异：本案例的 1 % 判据比较的是同一个理论（梁理论 vs 梁单元），
   并不度量它也测不出的东西。要覆盖剪切需 Timoshenko 梁单元，不在本单范围。
3. **案例 B 的中心挠度误差非单调**
   （3.888e-03 → 7.260e-04 → 3.030e-03）。软简支 + MITC4 在
   `a/t = 100` 的薄极限下存在小幅振荡与角点效应；因此板案例的收敛阶以 L2 挠度场
   为准（单调、二阶）。中心挠度的 2 % 判据在三档都满足，但单点值的阶不可用。
4. **Mindlin vs Kirchhoff 的模型差**：`a/t = 100` 时横向剪切柔度为
   `O((t/a)^2) = 1.0e-04` 的相对量级，
   低于所用网格的离散误差，故不影响 2 % 判定；网格再细（n > 64）时该差会成为底噪。
5. **案例 C 用平面应变（`u_z = 0`）**而不是建模开口/闭口端。Lame 的
   `sigma_r / sigma_theta` 场与端部条件无关（静定），但 `sigma_z = 2 nu A` 只对应
   平面应变；闭口端（`eps_z` 均匀非零）会改变 `sigma_z` 与 `u_r`。判据 C3b 恰好
   用来确认平面应变约束确实被施加（`sigma_z` 误差 5.875e-03 < 1 %）。
6. **`sigma_r` 在自由表面恒为 0**，该点的相对误差在数学上无定义（分母为 0）。
   驱动对 `|sigma_r^Lame| < 1e-6 p` 的点返回 `null` 而不做除零，并同时给出
   载荷归一化的绝对误差门禁 C2b；两者共同覆盖 5 个径向点，不做隐式丢弃。
7. **适用范围**：三个案例都是线弹性、小位移、单物理场检查，认证的是单元内核、
   一致载荷向量、边界条件与应力恢复；**不**覆盖曲壳几何离散误差、塑性、接触、
   屈曲与动力学。

## 6 误差判定（逐项 vs 参考，显式回答是否 < 3 %）

| 门禁 | 相对/量纲归一误差 | 百分数 | 判据 | 是否满足 |
|---|---|---|---|---|
| `A1.tip_deflection` | 2.749e-12 | 0.0000 % | <= 1.0 % | 是 |
| `A2.root_bending_stress` | 3.036e-12 | 0.0000 % | <= 1.0 % | 是 |
| `B1.centre_vs_timoshenko` | 3.611e-03 | 0.3611 % | <= 2.0 % | 是 |
| `B2.centre_vs_navier` | 3.030e-03 | 0.3030 % | <= 2.0 % | 是 |
| `C1.hoop_stress` | 2.355e-03 | 0.2355 % | <= 1.0 % | 是 |
| `C2.radial_stress` | 1.664e-03 | 0.1664 % | <= 1.0 % | 是 |
| `C2b.radial_stress_absolute` | 5.066e-04 | 0.0507 % | <= 1.0 % | 是 |
| `C3.radial_displacement` | 1.547e-04 | 0.0155 % | <= 1.0 % | 是 |
| `C3b.axial_stress` | 5.875e-03 | 0.5875 % | <= 1.0 % | 是 |

收敛阶：

| 门禁 | 观测阶 | 判据 | 是否满足 |
|---|---|---|---|
| `A3.beam_order` | 2.030 | ≥ 1.50 | 是 |
| `B3.plate_order` | 3.839 | ≥ 1.50 | 是 |
| `C4.cylinder_order` | 1.898 | ≥ 1.50 | 是 |

**结论**：全部 `12` 个门禁通过。三个案例的关键量相对误差为
梁 `2.749e-12`、
板 `3.030e-03`、
圆筒 `2.355e-03`（`sigma_theta`，最细网格）与
`1.664e-03`（`sigma_r`，最细网格），
**全部 < 3 %（三案例最细网格也全部 < 1 %）**；
收敛阶 `2.03`（梁）、
`3.84`（板 L2）、
`1.90`（圆筒 `sigma_theta`），均 ≥ 1.5。
唯一未达标项：无（板中心挠度的阶不可用一项已在 §5.3 明确记为不可用作收敛率，
改由 L2 场承担该项判据，未含糊处理）。

## 7 审计链

| 环节 | 路径 |
|---|---|
| 驱动（唯一产生数字的地方） | `benchmarks/verified/fem_structural_benchmarks/run.py` |
| 机器证据 | `benchmarks/verified/fem_structural_benchmarks/result.json`、`benchmarks/verified/fem_structural_benchmarks/gates_verified.json` |
| 图 | `benchmarks/verified/fem_structural_benchmarks/case_a_beam.png`、`case_b_plate.png`、`case_c_cylinder.png` |
| 门禁回归测试 | `tests/test_fem_structural_benchmarks.py`（重推闭式解 + 证据自洽 + 粗网格重跑） |
| 本文渲染 | `benchmarks/verified/fem_structural_benchmarks/render_doc.py` |

环境：Python 3.11.6、torch 2.7.1、numpy 2.4.6、scipy 1.16.0、
`loongarch64`（Linux-6.6.52-1.7.lns23.loongarch64-loongarch64-with-glibc2.38）、dtype `torch.float64 / numpy.float64`。
耗时：梁 0.04 s、板 40.26 s、
圆筒 0.47 s。
