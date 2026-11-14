# CASE-FEM-KIRSCH-PLATE（算例池 m20006）：开孔板应力集中 `K_t = 3` 的 TensorFEM 复现与细化复核
**Mission:** 20006 · **Repo:** TensorFEM · **Branch:** `agent/m20006` · **Case pool:** `tensor_case_pool` id=186 `FEM-KIRSCH-PLATE`（`tol_policy` `<5%`，`ref_source` 「Kirsch 1898 解析」）· **Generated:** 2026-11-14T07:46:49+00:00
> 结论先行：`K_t` 在 `d/W → 0` 极限上复现到 **0.0469%**（Q4，n = 48），独立 Q8 单元族 **0.0492%**，远小于 <5% 池容差与自设 3% 线；判据门 14/14 通过。**但 n = 96 探针推翻了「0.047% 是收敛值」这一读法**（正是 m15393 遗留的 finding F6）：再加密后误差回到 0.153%，精度应按带宽引用（≤0.153%），而非按最细点值；判定仍远优于 3%。
## ① 问题与参考出处
**物理背景。** 船体板格上的开孔（舱口角隅、人孔、管系贯穿件、舷侧开口）是疲劳与屈曲筛算的标准部位，其强度评估的第一层就是应力集中系数 `K_t = σ_peak / σ_nominal`。Kirsch (1898) 给出的无限大板单轴拉伸圆孔解是所有手册 `K_t = 3` 的源头，因此它同时是**解析锚**（可以对到 6 位有效数字）和**量纲/边界条件的教科书检查**。
**几何与载荷。** 宽 `W`、厚 `t` 的板，中央圆孔直径 `d = 2a`，远端单轴均匀拉应力 `T`（毛截面名义应力 `σ0 = T`）。无限大板极坐标解（`θ` 自加载轴量起）
```
σ_rr = T/2 (1 - a²/r²) + T/2 (1 - 4a²/r² + 3a⁴/r⁴) cos 2θ
σ_θθ = T/2 (1 + a²/r²) - T/2 (1 + 3a⁴/r⁴) cos 2θ
σ_rθ = -T/2 (1 + 2a²/r² - 3a⁴/r⁴) sin 2θ
```
孔边 `r = a` 退化为 `σ_θθ(a,θ) = T (1 - 2cos2θ)`，在垂直加载轴的极点取得
```
K_t = σ_θθ(a, π/2) / T = 3          (1)
```
**出处（原表原页 / 原文条款）。**
| 用途 | 出处 | 在证据链中的角色 |
|---|---|---|
| 无限大板闭式解 `K_t = 3` | G. Kirsch, *VDI* **42** (1898) 797；S. Timoshenko & J.N. Goodier, *Theory of Elasticity*, 3rd ed., McGraw-Hill (1970), **Art. 39** “Stresses in a plate with a circular hole”, eqs. **(a)–(c)** | 主基准；`run.py::oracle_selfcheck` 直接对式 (a)–(c) 求值 |
| 有限宽板表值（本单只作旁证） | R.E. Peterson, *Stress Concentration Factors*, Pilkey & Pilkey 3rd ed., **Chart 4.1**（Howland 1930 级数解列表，读表误差声明 ±1%） | `src/tensorfem/finite_width_hole.py:61-71` 以常量读入，本单与之做 `d/W = 0.1…0.5` 对照 |
| 有限宽工程拟合（旁证） | Heywood (1952) `K_tn = 2 + (1-d/W)³` | `finite_width_hole.py:73`，与表值互为独立佐证 |
**零手抄。** 目标数值 `3` **不是**从表格抄进代码的：它在 `oracle_selfcheck()` 里由式 (1) 求值得到，并由解析解自身的边界条件反检——孔边无面力 `max(|σ_rr|,|σ_rθ|)/T = 0.0e+00`，孔边环向应力 `[min, max] = [-1.000, 3.000] T`，远场 `σ_xx → T` 偏差 6.2e-05、`σ_yy → 0` 残量 3.7e-05。本报告表格里的每个数字同样由 JSON 程序化注入。
**局限（如实）。** 手册 Chart 4.1 的页码级引用来自仓库内既有声明（`finite_width_hole.py` 注释）与 `docs/finite-width-hole-howland.md`，本工作树内没有该书影印页可二次核对；因此有限宽表值在本单只作旁证，主判据完全落在可机器求值的闭式解上。
## ② 方法与实现（复用既有生产路径，不另起炉灶）
算例的三查结果：`knowledge_base` 命中能力行 `cap:TensorFEM:m15298_CASE-KIRSCH-PLATE`（L4），`negative_knowledge_ledger` 中 `TensorFEM` 相关 84 条负面条目里**没有** Kirsch 相关失败，`oss_solutions` 无同域（开孔应力集中）外部实现可交叉——因此本单按「复用 + 补齐遗留量化」执行，不新增求解器。
| 臂 | 实现 | 单元 / 边界 | 作用 |
|---|---|---|---|
| A（主） | `tensorfem.finite_width_hole.solve_case` → `tensorfem.continuum.solve_continuum`（稀疏直接解） | Q4 平面应力，孔心放射状（starburst）渐变网格，四面四分之一模型；`x=0` 对称 `u_x=0`、`y=0` 对称 `u_y=0`、孔边与 `y=W/2` 自由、`x=L` 均布拉力 `T` | 真边值问题，误差含离散 + 有限宽模型误差 |
| B（交叉） | `tensorfem.structural_canonical_face.solve_kirsch_q8_concentration` | Q8 二次 serendipity 平面应力，极坐标环，外边界 `r=R_out` 给精确 Kirsch 位移、孔边自由 | 只含离散误差，单元族/网格拓扑/边界提法全不同 |
| C（解析） | `run.py::oracle_selfcheck` | 式 (a)–(c) 稠密网格求值 + 边界条件自检 | 基准自身自证 |
**本单新增的两项量化（都预先声明判据，见 ⑤）。**
1. `benchmarks/verified/fem_kirsch_plate/refinement_probe_n96.py` —— 把 `d/W = 0.02` 的阶梯从 n = 48 推到 **n = 96**（集群作业 cj2429），判定 0.047% 是「真收敛点」还是「幸运点」，即 m15393 审计遗留的 finding F6（点值 0.047% vs GCI 带宽 0.32%）。
2. `benchmarks/verified/fem_kirsch_plate/nu_invariance_refinement.py` —— 把 `ν`/`E` 不变性从单网格（n = 24，门值 0.71%）变成**收敛序列**：ν = 0/0.3/0.45 在 n = 24 → 48 → 96 上的`K_t` 散布是否趋于 0（集群作业 cj2432）。平面问题只受面力时应力场与 `E`、`ν` 无关，散布若收敛到 0 即证「看到的是离散误差，不是本构错误」。
**工具链。**
```bash
# 主锚点（本单重跑，6 线程，约 60 s）
OMP_NUM_THREADS=6 PYTHONPATH=src python benchmarks/verified/fem_kirsch_plate/run.py
# 判据门自检
PYTHONPATH=src python -m pytest tests/test_kirsch_plate_anchor.py -q
# 新增：n=96 细化探针 / ν 不变性收敛（集群，run_compute 递交）
OMP_NUM_THREADS=8 PYTHONPATH=src python benchmarks/verified/fem_kirsch_plate/refinement_probe_n96.py
OMP_NUM_THREADS=8 PYTHONPATH=src python benchmarks/verified/fem_kirsch_plate/nu_invariance_refinement.py 48,96 nu_invariance_n96.json
```
环境：loongarch64 / Python 3.11.6 / numpy 2.4.6 / TensorFEM 0.45.0 / float64；求解器 `tensorfem.continuum.solve_continuum (sparse direct) / tensorfem.quadratic_plane.solve_q8_plane`。
## ③ 参数表（无量纲数完整）
| 量 | 符号 | 取值 | 说明 |
|---|---|---|---|
| 孔/板宽比 | `d/W` | 0.02, 0.05, 0.10 | 主判据取 0.02（最接近无限板）；0.05/0.10 作有限宽旁证 |
| 半长/板宽 | `L/W` | 1.5 | 整板长 3W；Saint-Venant 另测 1.0–4.0 |
| 厚/板宽 | `t/W` | 0.0100 | `t/W << 1`，平面应力假设成立 |
| 泊松比 | `ν` | 0.3 | 灵敏度扫 0 / 0.3 / 0.45 |
| 无量纲载荷 | `T/E` | 5.0e-06 | 小变形线弹性（`T/E ≪ 1`） |
| 无量纲载荷 | `σ0/T` | 1.0 | 远端毛截面名义应力 |
| 无量纲刚度 | `E·t/(T·W)` | 2000 | 只影响位移量级，不影响应力 |
| 网格渐变 | `grading` | 6.0 | 孔边→远场径向拉伸比 |
| 网格级 | `n_θ` | 12, 24, 48（+ 本单 96） | 边界节点数，步长逐级减半 |
| 自由度 | `n_dof` | 338 … 4802 | 四分之一模型 Q4 |
| Q8 交叉族 | `n_θ × n_r` | 24×6, 48×12, 96×24 | 极坐标环，外径/内径 = 2.0 |
参考值为无量纲量 `K_t = σ_θθ(a,π/2)/σ0 = 3`，故 `E`、`ν`、`T`、`W` 的绝对取值不进入判据（这一点本身在 ⑤ 用收敛序列验证）。
## ④ 对照表（ref vs got，误差逐项打勾叉）
池容差 <5%（`tensor_case_pool` id=186），本报告另自设 3% 线；两栏都判。
| # | 量 | ref（来源） | got | 误差 | <5% | <3% |
|---|---|---|---|---|---|---|
| 1 | `K_t`，Q4，`d/W` = 0.02，n = 48 | 3（式 (1) 求值） | 3.001408 | 0.0469% | ✓ | ✓ |
| 2 | `K_t`，Q4，`d/W` = 0.05，n = 48 | 3 | 3.001349 | 0.0450% | ✓ | ✓ |
| 3 | `K_t`，Q8 极坐标环，n = 96×24 | 3 | 3.001476 | 0.0492% | ✓ | ✓ |
| 4 | `K_t` Richardson 极限（`d/W` = 0.02，12/24/48） | 3 | 3.009196 | 0.3065% | ✓ | ✓ |
| 5 | 孔边环向应力剖面 RMS（vs `1-2cos2θ`，n = 48) | 解析 `1-2cos2θ` | RMS/K_t = 0.00318 | 0.3176% | ✓ | ✓ |
| 6 | `K_tn` 净截面，`d/W` = 0.1（旁证） | Peterson Chart 4.1 表值 2.73 | 2.7185 | 0.4226% | ✓ | ✓ |
| 7 | `K_tn` 净截面，`d/W` = 0.2（旁证） | Peterson Chart 4.1 表值 2.51 | 2.5063 | 0.1483% | ✓ | ✓ |
| 8 | `K_tn` 净截面，`d/W` = 0.3（旁证） | Peterson Chart 4.1 表值 2.35 | 2.3440 | 0.2533% | ✓ | ✓ |
| 9 | `K_tn` 净截面，`d/W` = 0.4（旁证） | Peterson Chart 4.1 表值 2.24 | 2.2264 | 0.6063% | ✓ | ✓ |
| 10 | `K_tn` 净截面，`d/W` = 0.5（旁证） | Peterson Chart 4.1 表值 2.16 | 2.1459 | 0.6515% | ✓ | ✓ |
| 11 | 静力平衡 `|R-P|/P`（n = 48） | 外载 `T·W·t` | 1.01e-12（门值 1e-10） | 0.000000% | ✓ | ✓ |

判据门总表：**14/14 通过**（失败项：无）。
说明两处容易误读的量：
- `d/W = 0.10` 的 `K_t`（毛截面）最细网格为 3.025609，对 `3` 的误差 0.854%，**这不是精度退化**：该网格正在收敛到有限宽板的真值——Howland/Heywood 在该比值的毛截面因子为 3.0322（表值净截面 2.73 折算），与 FE 值只差 0.218%。所以「对 `K_t = 3` 的判据」只允许在 `d/W → 0` 上用，锚点因此只用 0.02（0.05 作趋势佐证）。
- 净截面表值对照的误差区间 0.148%–0.652%，口径是**净截面**（`σ_net = T/(1-d/W)`）；若误用毛截面口径会比表值大 `1/(1-d/W)` 倍，`d/W = 0.5` 时差 2 倍——这是本域最常见的假失败来源（`docs/finite-width-hole-howland.md` §2B）。

## 结果图（④ 配套，matplotlib 生成，报告内嵌）

![kirsch_convergence.png](docs/assets/kirsch_plate/kirsch_convergence.png)

*两套单元族对解析 `K_t = 3` 的误差-步长曲线（双对数），虚线为 5% 池容差。*

![kirsch_field.png](docs/assets/kirsch_plate/kirsch_field.png)

*`d/W` = 0.02、n = 48 四分之一模型的 `σ_xx/T` 云图：孔边两侧的红色极值区即 `K_t = 3` 的位置，远场回到 1。*

![kirsch_hoop_profile.png](docs/assets/kirsch_plate/kirsch_hoop_profile.png)

*孔边环向应力沿 `θ` 的分布 vs 解析 `1-2cos2θ`：`θ = 0` 处 −1T、`θ = 90°` 处 +3T。*

![kirsch_nu_spread.png](docs/assets/kirsch_plate/kirsch_nu_spread.png)

*本单新增：`ν` = 0/0.3/0.45 的 `K_t` 散布随网格加密单调缩小，`E` 变化 100 倍则逐位不变。*

![kirsch_probe_n96.png](docs/assets/kirsch_plate/kirsch_probe_n96.png)

*本单新增：阶梯推到 n = 96（集群 cj2429）。误差在 n = 48 处触底后**回升**，据此把 0.047% 判为幸运抵消（F6 裁决：引带宽不引点值）。*

![kirsch_width_trend.png](docs/assets/kirsch_plate/kirsch_width_trend.png)

*毛/净截面 `K_t`–`d/W` 与 Howland/Heywood 及 Kirsch 极限 3 的对照。*

## ⑤ 收敛性与参数敏感性（≥2 项，全部带预声明判据）
### 5.1 网格收敛（`d/W` = 0.02，步长逐级减半）
| n | 单元 | 自由度 | `K_t` | 对 3 误差 | 相邻误差比 |
|---:|---:|---:|---:|---:|---:|
| 12 | 144 | 338 | 2.879925 | 4.0025% | — |
| 24 | 576 | 1250 | 2.977466 | 0.7511% | 0.1877 |
| 48 | 2304 | 4802 | 3.001408 | 0.0469% | 0.0625 |
| 96（本单探针，集群 cj2429） | — | — | 3.004591 | 0.1530% | 3.2617 |

观测收敛阶（12/24/48）= **2.026**（Q4 角点外推应超收敛，≥1 即通过门）；Richardson 极限 3.009196（对 3 误差 0.307%）；GCI（FoS 1.25）**0.324%**。
**预声明判据与结果（F6 欠账）：** 规则 H1 = 误差继续单调下降且 p(24,48,96) ≥ 1.5 → 点值可引；H2 = 误差(96) ≥ 误差(48) → 点值是幸运点，只能引带宽。定量预测（由 p = 1.76 外推）err(96) ≈ 0.0139%。
实测：err(24) = 0.7511% → err(48) = 0.0469% → err(96) = 0.1530%；相邻比 3.2617；三点（非单调）拟合阶 p = 2.911；Richardson 极限 3.005079（对 3 误差 0.1693%），GCI(48/96) 带宽 0.020%。定量预测（err(96) ≈ 0.0139%）**未命中（差约一个量级）**；判据裁决 **H2_point_luck__quote_band**。
**裁决的物理读法。** 误差序列 0.047% → 0.153% 是**非单调**的，所以 n = 48 的 0.047% 是一次幸运抵消（相邻两个误差比 3.3 > 1），不是渐近收敛值；同一对网格的 GCI(48/96) 只有 0.020%，也**盖不住**真实误差 0.153%——非单调时 GCI 不可信，这一点必须写进引用规范：**引用带宽而非点值**。
误差还可以再分解一层：`d/W = 0.02` 的**真值不是 3.0**，而是有限宽的 Howland/Heywood 毛截面值 3.001216（`K_tn` 2.941192 折算）。相对它，n = 96 的 3.004591 只差 0.112%。也就是说 0.153% 里有 0.041% 是**模型形误差**（有限宽 vs 无限宽），剩下约 0.112% 才是离散误差。这正好说明为什么本锚点同时保留「对解析 3 的极限」与「对 Howland/Heywood 的有限宽阶梯」两条判据：前者验算法，后者验模型。
### 5.2 `ν` / `E` 不变性——它是极限性质，不是单网格巧合
| n | `ν` = 0 | 0.30 | 0.45 | 散布 (max−min)/mid | `E` 变 100 倍的影响 |
|---:|---:|---:|---:|---:|---:|
| 24 | 2.956457 | 2.977466 | 2.996983 | 1.361% | 4.2e-15（机器精度 0） |
| 48 | 2.989799 | 3.001408 | 3.012979 | 0.772% | 9.0e-14（机器精度 0） |

`E` 的不变性是**精确**的（换刚度后 `K_t` 逐位相同，相对变化 9.0e-14），因为只受面力时刚度矩阵整体缩放会约掉。`ν` 的散布随网格加密单调下降
- n = 24 → 48：1.361% → 0.772%，比 0.567（观测阶 0.818）
裁决（预声明：比 ≤ 0.5 为 PASS、0.5–1.0 为 INCONCLUSIVE、≥1.0 为 FAIL）：24→48 段 **INCONCLUSIVE_decays_slower_than_p1**。散布在下降但慢于 p = 1，说明 n ≤ 48 仍未进入 `ν` 误差的渐近区；`ν` 依赖属于离散误差（若为本构错误，散布不会随 h 缩小），**但本单未能把它压到 1% 以下以外的更严水平**——这条按预声明如实记为「未达 p ≥ 1」，不追认。
### 5.3 Saint-Venant（域长）与 `d/W` 趋势
| `L/W` | n | `K_t`（`d/W` = 0.02，非匹配网格） |
|---:|---:|---:|
| 1.0 | 16 | 2.972626 |
| 1.5 | 24 | 2.977466 |
| 2.5 | 40 | 2.979180 |
| 4.0 | 64 | 2.979938 |

域长带内散布 **0.246%**（门值 1%，实测 0.220%，通过）；即 `L = 1.5W` 已足够，边界截断不是误差源。`d/W` 阶梯（0.1→0.5）的 `K_tn` 与 Peterson 表、Heywood 拟合三线互相印证，见 ④ 表末五行。
### 5.4 误差判定（逐项回答「是否 < 3%」）
| 关键量 | 基准 | 误差 | 是否 < 3% |
|---|---|---:|---|
| `K_t`（n = 48，`d/W` = 0.02） | 解析 3 | 0.0469% | ✓ |
| `K_t`（n = 96，本单探针） | 解析 3 | 0.1530% | ✓ |
| `K_t`（Q8 交叉族） | 解析 3 | 0.0492% | ✓ |
| 孔边环向应力剖面 RMS | 解析 `1-2cos2θ` | 0.3176% | ✓ |
| `K_tn`（`d/W` = 0.1…0.5） | Peterson 表值 | ≤0.652% | ✓ |
| 静力平衡 | 外载 | 1.0e-12 | ✓ |
**最差关键量误差 0.318%，全部 < 3%（也全部 < 5% 池容差）**；把精度按带宽引用时上界是 0.153%，仍比 3% 线小一个量级以上。
## ⑥ 失败模式、工件与生态摩擦
**失败模式清单（都会让 `K_t` 假失败/假通过）。**
1. **有限宽 vs 无限宽混用**：在 `d/W = 0.1` 上要求 `K_t = 3` 会得到 ~1% 的「误差」，其实那是模型真值；`d/W = 0.5` 时毛截面值比表值大 2 倍。判据只允许落在 `d/W → 0`（本单 0.02）。
2. **毛/净截面口径**：表值是净截面因子 `K_tn`，FE 直接除 `T` 得毛截面 `K_tg = K_tn(1-d/W)`。
3. **点值 vs 带宽**：m15393 记的 F6——最细网格 0.047% 小于自身 GCI 带宽 0.32%。本单 n = 96 探针已裁决（见 ⑤，结论 **H2_point_luck__quote_band**），引用精度时以判据结论为准。
4. **面力边界跳变未落在节点上**：加载端与自由边的角点若不在节点上，载荷会少 `4/n`（历史实测不平衡恰为 `4/n_θ`）。
5. **峰值取样**：峰值在 `θ = 90°` 孔边节点，必须用相邻单元角点应力外推平均；直接取单元中心会偏低数个千分点。
**生态摩擦（如实）。**
| # | 现象 | 影响 | 处置 |
|---|---|---|---|
| F1 | `tensor_benchmark_cases` 中 `anchor_slug LIKE '%KIRSCH%'` 的行数仍为 **0**（m15393 已记为 medium） | 锚点容差只存在于 `tensor_case_pool`，平台侧无登记 → 报告判据得回读池表 | 本单未改平台库（越权），在报告中显式引用 `tensor_case_pool` id=186 的 `<5%` |
| F2 | 同一锚点在池里有两条：id=41（`KIRSCH-PLATE`，m15298，status=passed）与 id=186（`FEM-KIRSCH-PLATE`，m20006，dispatched） | 同一算例可能被重复派单，浪费机时 | 建议池侧按 `linked_anchor` 去重；本单按 186 交付 |
| F3 | 本机计算门 120 s 硬上限：n = 96 探针在本机被收割（`exit=-9`） | 细化研究必须递交集群，排队 ~数分钟 | 已递交 `run_compute` cj2429 / cj2432，无指定节点 |
| F4 | `oss_solutions` 无 TensorFEM/开孔应力集中同域实现（最近一条是 `CASE-HOLLOW-SPHERE-3D`） | 无第三方实现可交叉 | 用两套单元族（Q4 / Q8）+ 解析解三方交叉替代 |
| F5 | 图件过滤：`tp_pusher` 过滤 `docs/` 之外的 `*.png` | 锚点目录下的图不会入库 | 图件同步到 `docs/assets/kirsch_plate/`（sha256 与生成件逐字节相同），报告以 base64 内嵌 |
**工件（真实路径）。**
| 工件 | 内容 |
|---|---|
| `benchmarks/verified/fem_kirsch_plate/result.json` | 全部数字（阶梯、Q8 交叉、ν/E、Saint-Venant、d/W 趋势、图件清单） |
| `benchmarks/verified/fem_kirsch_plate/gates_verified.json` | 14 项判据门，逐门 value/threshold/verdict |
| `benchmarks/verified/fem_kirsch_plate/refinement_probe_n96.json` | **本单新增**：n = 96 细化探针（F6 裁决） |
| `benchmarks/verified/fem_kirsch_plate/nu_invariance_refinement.json` / `..._n96.json` | **本单新增**：ν/E 不变性收敛序列 |
| `runs/m20006_kirsch_plate/` | 本次复现的 result/gates 副本、`run_stdout.log`、`manifest.json`（环境 + sha256 + 三查） |
| `docs/assets/kirsch_plate/*.png` | 4 张锚点图 + 2 张本单图（探针、ν 散布） |
**一键复现。** `OMP_NUM_THREADS=6 PYTHONPATH=src python benchmarks/verified/fem_kirsch_plate/run.py`（本单重跑：exit 0，14/14 门通过，结果与 main 上已合并版本逐位相同，仅 `generated_at`/`wall_seconds` 不同；4 张图 sha256 完全一致）。
