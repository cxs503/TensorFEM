# 平面应变 Lamé 厚壁圆筒验证（Q4 平面单元）

`tensorfem.continuum` 自首次提交起就带有 `plane="strain"` 本构分支，但仓库中全部
既有算例（patch test、Cook 膜、悬臂梁）都走默认的 `plane="stress"`，即该分支从未
被任何测试或基准覆盖。本文用静定可解的 Lamé 厚壁圆筒补上这条验证欠账：面内应力
场对平面应力/平面应变是**同一**的，而径向位移因面外柔度不同而不同——这正好可以
反证 `plane` 开关真的接线，而不只是一个被静默忽略的参数。

## 1. 案例介绍

厚壁圆筒（压力容器/耐压壳/环肋圆筒的经典模型），单位面外厚度，平面应变状态
（$arepsilon_z = 0$）。内壁受静水压，外壁自由。

| 参数 | 符号 | 取值 |
|---|---|---|
| 内半径 | $a$ | 0.5 m |
| 外半径 | $b$ | 0.8 m |
| 壁厚比 | $b/a$ | 1.60 |
| 内压 | $p_i$ | 100 MPa |
| 外压 | $p_o$ | 0 MPa |
| 弹性模量 | $E$ | 210 GPa |
| 泊松比 | $\nu$ | 0.3 |
| 面外厚度 | $t$ | 1 m（平面应变） |

**解析解（Lamé）**。环向静定，$A,B$ 由两条面力边界条件唯一确定，与 $E,\nu$ 及
平面假设无关：

$$A = \frac{p_i a^2 - p_o b^2}{b^2 - a^2}, \qquad B = \frac{a^2 b^2 (p_i - p_o)}{b^2 - a^2}$$

$$\sigma_r = A - \frac{B}{r^2}, \qquad \sigma_\theta = A + \frac{B}{r^2}$$

平面应变（$\varepsilon_z=0$，$\sigma_z = \nu(\sigma_r+\sigma_\theta) = 2\nu A$）：

$$u_r = \frac{1+\nu}{E}\left[(1-2\nu) A r + \frac{B}{r}\right]$$

平面应力（$\sigma_z = 0$）：

$$u_r = \frac{1}{E}\left[(1-\nu) A r + (1+\nu)\frac{B}{r}\right]$$

## 2. 方法

* **单元/离散**：`tensorfem.continuum` 的 4 节点双线性等参元（2×2 Gauss，中心点 B 矩阵
  回写应力），`plane="strain"`，$E,\nu,t$ 逐单元展开，稠密直接求解。
* **网格**：极坐标结构网格的 1/4 环扇形（45°…90° 扇区），节点严格落在内/外圆上，
  径向均匀细分为 `n_radial` 层，环向 `n_theta` 等分；单元连通按逆时针排列保证 Jacobian 正定。
* **边界条件**：两条径向切面施加真实对称条件（$\theta=0$ 面 $u_y=0$，$\theta=\pi/2$ 面
  $u_x=0$），刚体模态被完全消除，不引入任何来自解析解的位移约束。
* **载荷**：内圆边每一段圆弧的合力 $2 p_i a \sin(\Delta\theta/2)$ 沿角平分线施加，
  两个端节点各取一半（直边精确、曲边只留 $O(\Delta\theta^2)$ 的重分配误差）；载荷总
  合力由反力平衡独立校验。
* **参考值**：全部来自上节闭式解；载荷、应力、位移、平衡四个量互不共用推导路径。
* **验收门槛**：位移、$\sigma_r$、$\sigma_\theta$ 的 $L_2$ 与最差采样点相对误差均 < 3 %。

工具链：`benchmarks/lame_cylinder_plane_strain.py`（本文件全部数字由其程序化注入：
`benchmarks/lame_cylinder_plane_strain_evidence.json`），测试见
`tests/test_lame_cylinder_plane_strain.py`。

## 3. 计算结果

网格阶梯（$\theta=0$ 射线上的位移采样 + 单元中心应力），`u_r` 列为最大相对误差：

| 网格 $n_\theta\times n_r$ | 单元 | 自由度 | $u_r$ max | $u_r$ $L_2$ | $\sigma_r$ $L_2$ | $\sigma_r$ max | $\sigma_\theta$ $L_2$ | $\sigma_\theta$ max |
|---|---|---|---|---|---|---|---|---|
| 12x4 | 48 | 130 | 0.5113 % | 0.4696 % | 1.4759 % | 6.4981 % | 0.3837 % | 0.4395 % |
| 24x8 | 192 | 450 | 0.1291 % | 0.1180 % | 0.3685 % | 3.2482 % | 0.0962 % | 0.1111 % |
| 48x16 | 768 | 1666 | 0.0324 % | 0.0295 % | 0.0921 % | 1.6227 % | 0.0241 % | 0.0279 % |

内壁理论应力为 $\sigma_r(a) = -p_i$、$\sigma_\theta(a) = 2A + p_i$；最细网格上这两条
Lamé 曲线与单元中心采样值的偏差见上表最后一列与下图。

观测收敛阶（两条最细网格，网格比 2）：

| 量 | 观测阶 |
|---|---|
| $u_r$ 最大误差 | 2.00 |
| $\sigma_r$ $L_2$ | 2.00 |
| $\sigma_\theta$ $L_2$ | 2.00 |

对称性检查（离散解应当严格轴对称）：$\max|\tau_{r\theta}|/\max|\sigma_r| = $
4.812e-14，$\max|u_\theta|/\max|u_r| = $
7.354e-14。

反力平衡：四分之一扇区上支撑反力合力范数 70.710678 MN，
与解析合力 $2 p_i a\sin(\pi/4)$ 的相对偏差 2.661e-08。

**平面应力/平面应变判别**：同一网格（48x16）下，平面应变与平面
应力两条路径各自匹配自己的闭式解（位移最大误差 0.0324 % 与
0.0266 %），而两者互相差 4.6838 %，
与闭式解的柔度差 $(1-\nu^2)$ 量级一致——说明 `plane` 参数真实生效。

径向梯度（`radial_grading>1` 时环向分层向内壁聚集）对精度的影响可用同一自由度预算下的
对比量化：24x8（grading=2.0）的
$\sigma_r$ $L_2$ 误差 0.2774 %，与均匀网格
24x8 的 0.3685 % 对比。

![应力与位移剖面](assets/lame_cylinder_profiles.png)

![网格收敛](assets/lame_cylinder_convergence.png)

![环向应力云图](assets/lame_cylinder_field.png)

## 4. 注意点与适用边界

* **应力比较位置**：Q4 只回写单元中心应力，比较在单元中心（`centre_radius`）进行；
  内壁峰值应力位于边界外推处，不参与逐点验收（与仓库既有 `pointwise` 审计口径一致）。
* **载荷重分配**：曲边合力精确、端节点均分带来 $O(\Delta\theta^2)$ 的重分配误差，
  在网格阶梯上可见其随 $\Delta\theta$ 缩小而消失（$\sigma_\theta$ 列收敛阶接近 2）。
* **对称条件是精确的**：$u_\theta=0$ 是 Lamé 解的真实性质，不是来自数值解的拟合；
  因此 1/4 扇区模型与整环模型等价，$u_\theta$ 与 $\tau_{r\theta}$ 的数量级即离散误差。
* **适用范围**：小应变、各向同性、单一材料、无体力（无旋转、无温度梯度）、无轴向
  端部约束（平面应变 $\varepsilon_z=0$ 对应无限长筒/端部约束筒）。不覆盖厚壁圆筒的
  极限承载（塑性/安定性）、开孔接管应力集中或热应力。
* **未做**：三维厚壁筒、多层套合/过盈、非线性材料、网格自适应误差估计。

## 5. 验收判定（对基准逐项）

| 锚点 | 结果 | 判据 |
|---|---|---|
| `radial_displacement` | PASS | u_r vs Lame (plane strain) on the theta=0 ray, finest mesh; max and L2 relative error < 3 % |
| `radial_stress` | PASS | sigma_r at element centres vs Lame; L2 and worst-point relative error < 3 % |
| `hoop_stress` | PASS | sigma_theta at element centres vs Lame; L2 and worst-point relative error < 3 % |
| `axisymmetry` | PASS | Symmetry of the discrete solution: max|tau_rtheta|/max|sigma_r| and max|u_theta|/max|u_r| < 1e-8 |
| `load_equilibrium` | PASS | Sum of support reactions equals 2 p_i a sin(pi/4) along the sector bisector; relative error below the 1e-6 round-off floor of the dense solve |
| `constitutive_paths` | PASS | plane stress and plane strain each match their own Lame reference (<3 %), and differ from one another as the closed forms predict |
| `mesh_convergence` | PASS | Observed order of u_r and of both stress L2 errors between the two finest meshes >= 1.8 |

**是否 < 3 %**：最细 48x16 网格上，$u_r$ 最大 0.0324 %
（$L_2$ 0.0295 %）、$\sigma_r$ 最差 1.6227 %
（$L_2$ 0.0921 %）、$\sigma_\theta$ 最差 0.0279 %
（$L_2$ 0.0241 %）——**均 < 3 %，通过**。粗网格（12x4）
的 $\sigma_r$ 最差点 6.4981 % 超过门槛，如实保留，未删除。
