# TET10 一致质量阵：清偿 Keast 积分欠账（m6515）

## 1. 选题理由（缺口账）

[`docs/tet10-integration-accuracy.md`](tet10-integration-accuracy.md) 已把「Keast 五点规则只到 3 次」
量化成一条**显式欠账**：该规则不能用于一致质量阵（Frobenius 19.7 %、对角最大 70.8 %–81.9 %），
「日后实现 `tet10_mass` 时必须换规则」。生产代码里只有 `tet10_stiffness`，没有质量路径，
因此三维二次四面体的动力学/模态分析在本仓**根本不可用**。本单把这条欠账同时作为
**验证欠账**（把欠账数字独立复算）与**功能缺口**（补生产质量阵）一起清偿。

## 2. 方法

* 新增 `tensorfem.quadratic_solid.tet10_mass_quadrature(order=4)`：参考四面体上的**坍缩
  (Duffy) Gauss 规则**，由 Jacobi 矩阵 + `torch.linalg.eigh` 现场生成 Gauss–Legendre 节点
  （不依赖 numpy），权已含坍缩 Jacobian `u²v` 与参考体积，`Σw = 1/6`。
* 新增 `tet10_consistent_mass(x, density, *, rule=None)`：`ρ ∫ N_i N_j detJ dV`，
  返回 `(ne,30,30)`，`rule=` 与 `tet10_stiffness` 同约定可覆盖。
* 新增 `tet10_volume(x, *, rule=None)` 与 `tet10_lumped_mass`（HRZ 对角集中）。
* 接线：`solid_mass.assemble_solid_mass / mass_properties / _exact_integrals` 与
  `solid_mass_audit.axial_rod_case / _solid_stiffness` 接受 `element_type="tet10"`。

**关键判据（先于动手）**：直边单元上 `N_i` 的体积分有闭式 `∫N_i dV = -V/20`（顶点）、
`+V/5`（棱中点），因此**行和**就是解析参考；再以独立 NumPy `24³` Duffy 参考
（`tet10_integration_audit.element_mass`）作全矩阵对照；最后以固定–自由杆轴向频率
`ω_n = (2n-1)π/(2L)·sqrt(M_c/ρ)` 作物理锚。

## 3. 结果

### 3.1 规则自验证与独立参考对照

| 检查 | 值 | 判据 |
|---|---|---|
| 坍缩规则单项式精确度 order=4 | 总度 ≤5 误差 ≤1e-14；order=6 总度 ≤6 误差 3.9e-16 | 舍入级 |
| `Σ w` vs 参考体积 1/6 | 1.7e-16 | 舍入级 |
| 直边单元质量阵 vs 24³ Duffy 参考（Frobenius） | **2.19e-15** | <1e-12 |
| 行和（顶点 / 棱中点，ρ=1） | −8.333e-3 / +3.333e-2（解析 −V/20= −8.333e-3，+V/5=3.333e-2） | 舍入级 |
| 总质量 vs ρV | 8.3e-17 | 舍入级 |
| 畸变+曲边（distorted, κ=0.12）vs 24³ | 2.27e-15 | <1e-12 |
| 曲边（regular, κ=0.12）vs 24³ | 2.19e-15 | <1e-12 |
| 单元素质量阵 | 对称 + 正定（λ_min>0），无零模态 | 通过 |

![Keast 对角误差与杆频率阶梯](assets/tet10-mass/keast-vs-higher-order-and-rod-ladder.png)

### 3.2 欠账复现（fail-closed）

用生产 `tet10_stiffness` 的**出厂 Keast 五点规则**积分同一质量阵：

| 量 | 数值 |
|---|---|
| Frobenius 相对误差 | **19.677 %** |
| 顶点对角偏差 | **−70.83 %**（与文档 70.8 % 一致） |
| 棱中点对角偏差 | −19.79 % |
| 总质量 | 仍精确等于 ρV（**质量守恒不能暴露该错误**） |

故 `tests/test_tet10_mass.py::test_shipped_keast_rule_still_fails_the_mass_matrix` 以
>15 % Frobenius 与 <−70 % 顶点对角为下界锁死：任何人把生产默认退回 Keast，该测试先红。

### 3.3 物理锚：固定–自由杆轴向频率（E=210 GPa, ν=0.3, ρ=7850, L=10 m）

解析 `ω₁ = 942.6317 rad/s`、`ω₂ = 2827.8951 rad/s`（`M_c` 为单轴应变模量）。

| 单元/长度 | TET4 模态1 | TET10 模态1 | HEX8 模态1 | TET10 模态2 |
|---|---|---|---|---|
| 4 | 0.641 % | **0.00158 %** | 0.644 % | 0.121 % |
| 8 | 0.159 % | 9.1e-5 % | 0.161 % | 7.25e-3 % |
| 16 | 3.96e-2 % | 5.0e-6 % | 4.02e-2 % | 3.77e-4 % |
| 32 | 9.85e-3 % | 2.4e-9 % | 1.00e-2 % | 2.0e-5 % |

TET10 在同一 4 单元网格上比 TET4 好两个数量级；二阶模态到 16 单元已进 1 %。

## 4. 注意点 / 局限

* **阶数记账**：坍缩权 `u²v` 吃掉两阶，`order` 点仅对总度 ≤`2·order−3` 精确（order=4 → 5 阶）。
  TET10 质量被积函数在坍缩变量上每方向 ≤2 阶，故 order=4 已精确；这一「低阶也够」是**利用了
  TET10 形状函数的结构**，不是通用结论（测试里两者分别锁死）。
* 曲边几何上 order=4 与 24³ 参考也到 2e-15，但曲率很大时仍建议 `rule=` 提阶；本报告只测到 κ=0.12。
* **不能行和集中**：二次四面体的顶点行和是负的（−ρV/20），行和集中会产生负质量。生产提供的是
  HRZ 对角缩放（正定 + 总质量精确），转动惯量不守恒（未测即未声明）。
* 杆频率算例是**单轴应变杆**（侧向约束），不含剪切/弯曲，故它验证的是质量阵与轴向波速，
  不是 TET10 的弯曲精度（后者已在 `tet10-integration-accuracy.md` 记录为离散主导的 2.44 %）。

## 5. 误差判定（是否 <3 %）

| 关键量 | 基准 | 误差 | <3 % |
|---|---|---|---|
| 质量阵全矩阵 | 24³ Duffy 独立参考 | 2.2e-15 | 是 |
| 行和（顶点/棱中点） | 闭式 −ρV/20、+ρV/5 | ≤4e-15 相对 | 是 |
| 总质量 | ρV | 8.3e-17 | 是 |
| 杆 ω₁（16 单元 TET10） | (2n−1)π/(2L)√(M_c/ρ) | 5.0e-6 % | 是 |
| 杆 ω₁（4 单元 TET10） | 同上 | 0.0016 % | 是 |
| 杆 ω₁（4 单元 TET4，对照） | 同上 | 0.64 % | 是（对照项） |
| 杆 ω₂（4 单元 TET10） | 同上 | 0.121 % | 是 |

复现：`PYTHONPATH=src pytest tests/test_tet10_mass.py -q`（8 passed）；
数值以 `docs/assets/tet10-mass/evidence.json` 落盘（图与表由该 JSON 程序化注入）。
