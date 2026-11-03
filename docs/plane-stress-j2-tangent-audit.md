# 平面应力 J2 切向算子的解析化：嵌套有限差分跨屈服面污染与修正（验证欠账）

## 1. 案例介绍

**物理背景**：船体板格在局部屈曲/极限强度/循环载荷分析中，材料点采用平面应力
J2 径向返回（`sigma_zz = 0` 局部凝聚）。Newton 迭代的收敛速度完全取决于切向算子
（consistent tangent）是否与所积分的返回映射一致。`tensorfem.layered_shell4_plasticity`
早期实现里，凝聚后的切向算子由**两层有限差分**得到：内层用 `sqrt(eps)` 差分求
`dsigma_zz/deps_zz`（局部凝聚 Newton），外层用 `eps**(1/3)` 中心差分对整条返回映射
做三列差分。这种做法在弹性远场精确，但在**接近屈服面时中心差分跨过屈服面的折点**，
把弹性支与塑性支的斜率平均，导致切线严重失真——而这正是 Newton 最需要正确切线的
区域（塑性增量步的首个迭代点通常就落在屈服面上下几百 kPa 内）。

| 参数 | 符号 | 取值 |
|---|---|---|
| 弹性模量 | E | 200 000 MPa |
| 泊松比 | ν | 0.30 |
| 初始屈服 | σ_y | 250 MPa |
| 线性等向硬化 | H | 1200 MPa |
| 层厚/壳厚 | — | 单层厚度 0.1 m（材料点级，与单元无关） |
| 材料点 | — | `LayeredShell4State.virgin` 未加载历史 |
| 加载 | — | 单轴 σ_xx 从 100 MPa 扫到 400 MPa（跨越屈服面） |

## 2. 怎么做的

### 2.1 精确算法切向算子

3-D 径向返回的算法切向算子有闭式（Simo & Hughes 的经典结果，本实现按本仓
`update_j2` 的记号重新推导）。设试应力偏量 `s_tr`、`seq = sqrt(1.5 s_tr:s_tr)`、
塑性乘子 `Δγ`（`update_j2` 的第三个返回值）、`G, K` 为剪切/体积模量：

    a  = 1 - 3G Δγ / seq
    da = -3G [ 1/((3G+H) seq) - Δγ/seq² ]
    c  = 2G (1.5/seq) da

    C = K (1⊗1) + 2G a I_dev + c s_tr ⊗ s_tr                （弹性支：a = 1, c = 0）

Voigt 顺序 `[xx, yy, zz, xy, yz, zx]`，剪应变列为工程剪应变 `γ = 2ε`。新函数
`plasticity.j2_consistent_tangent` 直接吃 `update_j2` 返回的 `Δγ`，因此**切线与
返回映射永远同支**（弹性→`Δγ = 0`→弹性算子；塑性→塑性算子），不存在跨支平均。

平面应力凝聚不再是差分，而是 Schur 补：`C_cond = C_aa − C_az C_za / C_zz`
（`plasticity.condense_plane_stress`，取 `a = [xx, yy, xy]`, `z = zz`）。这与局部凝聚
方程 `sigma_zz = 0` 的隐函数求导完全等价，且局部 Newton 的导数也换成同一个
`C_zz`，于是局部凝聚与外层切线**共用**一个解析算子，`update_j2` 的调用次数不增反降
（旧实现每迭代 3 次返回映射，新实现 1 次）。

### 2.2 独立基准（不来自本模块输出）

有限差分只在**单侧、且不跨支**时才是该分支的真导数，故基准取**前向差分**
（3 个减半步长 + Richardson 外推），方向固定 `+e_j`，对 `j = x, y, xy` 各一列。

### 2.3 工具链

```bash
PYTHONPATH=src python scripts/audit_plane_stress_tangent.py     # 表 + 图 + JSON
PYTHONPATH=src python -m pytest -q tests/test_layered_shell4_plasticity.py \
    tests/test_finite_rotation_layered_shell4.py tests/test_shell4_arc_line_search.py \
    tests/test_implicit_plastic_tangent.py tests/test_hybrid_plastic_tangent.py \
    tests/test_substep_tape_tangent.py tests/test_nonlinear.py tests/test_solid_plasticity.py
```

## 3. 计算结果

`scripts/audit_plane_stress_tangent.py` 程序化输出（`artifacts/plane_stress_tangent/verification.json`
逐项记录），无手抄：

| σ_xx [MPa] | 支 | legacy 相对误差 | 解析 相对误差 | 基准 C_xxxx | legacy C_xxxx | 解析 C_xxxx |
|---:|---|---:|---:|---:|---:|---:|
| 100.0 | elastic | 3.257e-13 | 3.275e-13 | 219780.219780 | 219780.219780 | 219780.219780 |
| 200.0 | elastic | 6.292e-13 | 6.273e-13 | 219780.219780 | 219780.219780 | 219780.219780 |
| 240.0 | elastic | 1.095e-12 | 1.102e-12 | 219780.219780 | 219780.219780 | 219780.219780 |
| 245.0 | elastic | 2.243e-12 | 2.249e-12 | 219780.219780 | 219780.219780 | 219780.219780 |
| 249.0 | elastic | **2.977e-02** | 1.336e-12 | 219780.219780 | 210112.885590 | 219780.219780 |
| 249.9 | elastic | **2.206e-01** | 1.904e-02¹ | 213668.777839¹ | 143909.238633 | 219780.219780 |
| 250.5 | plastic | 2.739e-05 | 2.533e-05 | 42683.930821 | 42686.969262 | 42686.443314 |
| 251.0 | plastic | 2.246e-05 | 2.100e-05 | 35562.398812 | 35564.455003 | 35564.142683 |
| 260.0 | plastic | 5.285e-06 | 5.209e-06 | 9835.742005 | 9835.855764 | 9835.850466 |
| 310.0 | plastic | 8.423e-07 | 8.400e-07 | 3171.962408 | 3171.966508 | 3171.966468 |
| 400.0 | plastic | 2.600e-07 | 2.598e-07 | 2233.343335 | 2233.344026 | 2233.344024 |

¹ 249.9 MPa 距屈服面仅 0.1 MPa，基准的前向探针（h = 8e-7 应变 ≈ 0.18 MPa 应力）
**越过屈服面**，Richardson 外推因此被污染（基准给出 213668 而非弹性 219780）。
该点的正确答案由闭式弹性平面应力刚度 `E/(1−ν²) = 219780.219780` 给出，解析算子
与之相差 1e-12；legacy 的 143 909 两者都不符。此行的 1.9e-2 是**基准伪差**，
不是解析算子的误差（塑性行 250.5–400 MPa 的残差 2.5e-5…2.6e-7 同理来自探针跨支）。

图：`artifacts/plane_stress_tangent/tangent_error.png`
（左：切向相对误差 vs 加载应力，log 纵轴，标出屈服面与 1e-6 门限；
右：凝聚刚度 `C_xxxx` 三分量对比，可看到 legacy 在 249–250 MPa 处塌陷到 143 GPa
而基准/解析保持弹性 219.78 GPa）。

## 4. 注意点（坑 / 局限 / 适用范围）

1. **不是精度调参，是算子替换**：`sqrt(eps)` 那层差分（旧 §局部凝聚）本身还限制了
   `sigma_zz` 残差只能到 ~1e-8 相对量级；换成解析导数后局部 Newton 达到
   `1e-15·E` 残差量级且二次收敛。
2. **中心差分在非光滑点上是错方法**：返回映射在屈服面处 C⁰ 但不 C¹，任何跨支中心
   差分算出的都是两支斜率的平均，与"实际积分了哪一支"无关。解析算子按 `Δγ` 分派，
   与被积分的支严格一致。
3. **恰好落在屈服面上（σ_xx = 250 MPa 整）切线不唯一**：本例表格刻意跳过该点；此时
   左/右导数分别是塑性/弹性算子，解析实现按返回映射实际走的支给出。
4. **适用范围**：小应变、线性等向硬化 J2、平面应力凝聚。不覆盖随动硬化、有限应变
   乘法分解塑性（那些模块另有实现，本次未改）。
5. 旧实现的"远离屈服面用解析弹性矩阵"的短路分支已删除——解析 Schur 凝聚在弹性支
   与它逐位一致（相对差 < 1e-12），保留两条代码路径只会让两者漂移。

## 5. 误差判定

| 关键量 | 基准 | 误差 | 是否 < 3% |
|---|---|---:|---|
| 弹性支凝聚刚度 C_xxxx（249 / 249.9 MPa） | 闭式 `E/(1−ν²)` = 219780.219780 | 1.3e-12 / 1e-12 | **是** |
| 塑性支凝聚刚度（250.5–400 MPa 全行） | 单侧 Richardson 差分 | ≤ 2.5e-5 | **是** |
| 切向算子全矩阵（塑性支 Frobenius） | 同上 | ≤ 2.5e-5 | **是** |
| legacy 实现（对照） | 同上 | 最差 2.206e-01（249.9 MPa） | **否**（修正前不满足 3%） |

结论：修正后切向算子在弹性支与闭式解一致到 1e-12、塑性支与单侧差分一致到 2.5e-5，
**全部 < 3%**；修正前在 249.0/249.9 MPa 分别为 2.98%/22.06%，**不满足** 3% 门限。

## 6. 未完成 / 边界（honest）

- `tests/test_shell4_arc_line_search.py` 与 `tests/test_v1_evidence_aggregator.py::test_nonlinear_evidence_is_independent_hashed_and_semantic`
  的两处 `xfail`（"machine-dependent"）在本次修正后**仍然 xfail**：切线精度不是它们的
  唯一根因（回溯线搜索的步长/收敛判据仍需独立排查）。本单不宣称已修复弧长路径。
- `tests/test_finite_rotation_layered_shell4.py::test_shallow_warped_reference_is_stress_free_objective_and_deep_warp_rejected`
  在本机**已 XPASS**（该 xfail 标记已过期，`strict=False` 时静默）；留给下一单清理，
  本单未改任何测试标记。