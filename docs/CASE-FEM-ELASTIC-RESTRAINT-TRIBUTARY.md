# CASE-FEM-ELASTIC-RESTRAINT-TRIBUTARY — 板格弹性嵌固转动刚度的网格不变量与边缘弯矩口径

**锚点**：`GAP-PP-4-ICE-STRUCT-ELASTIC-RESTRAINT`（论文牵引 JOUR-ICE-STRUCT §4.2 边界口径）
**模块**：`src/tensorfem/plate_elastic_restraint.py`
**证据**：`docs/data/elastic_restraint_tributary_evidence.json`、`docs/figures/elastic_restraint_tributary.png`
**探针**：`scratch/probe_mesh_dep.py`、`scratch/probe_strip.py`、`scratch/probe_recovery.py`、`scratch/probe_invariant.py`
**测试**：`tests/test_plate_restraint_tributary.py`（新增不变量）、`tests/test_plate_restraint_panel.py`、`tests/test_plate_elastic_restraint.py`

---

## 1. 案例介绍

冰带板格（0.8 m × 0.7 m × 22 mm，E=206 GPa，ν=0.3）的边界既非简支（SSSS）也非固支（CCCC），
而由纵骨/横框架提供**有限转动刚度**。求解器用嵌入系数 β∈[0,1] 参数化该约束：

* β=0 → 转动自由（SSSS）；β=1 → 转动锁死（CCCC）；中间值为部分嵌固；
* 分布约束系数 `c_r = ζ·D/a`，`ζ = β/(1-β)`，`D = E t³/(12(1-ν²))`，`a` 为边界法向跨度；
* 边条件为 SS2 口径：`w=0`、切向转动固定、法向转动由转动弹簧承担。

本文（§4.2）需要「SSSS 结论只是上限」之外给出**β 区间的定量带**。因此 β 必须是一个
**与网格无关的物理参数**，且边缘弯矩必须是可信的物理量。

## 2. 怎么做的

用 Q4 Mindlin-Reissner 板（弯曲 2×2 Gauss + 剪切 1 点 SRI）对同一 β 做加密扫描，
并把求解器输出对三套**独立基准**复核：

1. **Kirchhoff 极限**：β=0 ↔ `w_c D/(q a⁴)=0.00406`，β=1 ↔ `0.00126`（Timoshenko &
   Woinowsky-Krieger, *Theory of Plates and Shells*, 2nd ed., Table 8）。
2. **圆柱弯曲薄条解析解**（新增的中间 β 基准）：长板 b≫a，x=0,a 边为转动弹簧、y 端简支，
   单位宽梁有效解
   `M_end = (q a²/12)/(1+2/ζ)`，`w_mid = (q a⁴/D)·(5/384 − 1/(96(1+2/ζ)))`。
   它同时检验「β→弹簧刚度」映射与求解精度；仅有两个极限值无法发现映射错误。
3. **弯矩反演自洽性**：由 `w_mid = 5qa⁴/384D − M_end a²/(8D)` 反解边缘弯矩，
   与恢复路径给出的 `M_n` 对比。

## 3. 计算结果

### 3.1 缺陷 A：β 不是网格无关的物理参数（已修）

原实现把分布系数 `c_r`（[N·m/m/rad]）**直接当作节点弹簧**（[N·m/rad]），
漏掉节点**从属边长** `h_i`。于是单位长度约束刚度为 `c_r/h`，随加密按 1/h 增长：
β 变成一个随网格漂移的旋钮，加密即静默趋向固支。

| β | nx=8 | nx=12 | nx=16 | nx=24 | 判定 |
|---|---|---|---|---|---|
| 0.2（修前） | 3.0163e-3 | 2.7631e-3 | 2.5703e-3 | 2.3017e-3 | 单调漂移，趋 CCCC |
| 0.5（修前） | 2.0946e-3 | 1.8942e-3 | 1.7706e-3 | 1.6277e-3 | 单调漂移 |
| 0.2（修后） | 3.8537e-3 | 3.8676e-3 | 3.8724e-3 | 3.8758e-3 | 收敛（跨度 0.6 %） |
| 0.5（修后） | 3.4133e-3 | 3.4311e-3 | 3.4372e-3 | 3.4415e-3 | 收敛（跨度 0.8 %） |

修复：`edge_tributary_lengths()` 计算每条边各节点的从属长度（`Σh_i = L`，端节点取半），
节点弹簧改为 `c_r·h_i`；`assemble_rotational_spring_stiffness` 与
`solve_restrained_panel` 两处装配路径统一。**总弹簧刚度 = `c_r × 边长`，与网格无关**
（由 `test_total_spring_stiffness_is_edge_length_times_coefficient` 断言）。

### 3.2 修复后的 β 响应带（nx=16 方板，无弹性地基）

| β | 0.0 | 0.2 | 0.4 | 0.5 | 0.6 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|
| `w_c D/(q a⁴)` | 4.0593e-3 | 3.8724e-3 | 3.6111e-3 | 3.4372e-3 | 3.2201e-3 | 2.5703e-3 | 2.0509e-3 | 1.2642e-3 |
| 嵌固比例 (SSSS→CCCC) | 0.00 | 0.07 | 0.16 | 0.22 | 0.30 | 0.53 | 0.72 | 1.00 |

注意：β 是**刚度比参数**（β↔ζ=β/(1-β)），与响应并非线性映射；β=0.5 只对应约 22 % 的
SSSS→CCCC 响应区间。论文写作时必须报「β–响应」曲线，不能把 β 读作线性嵌固度。

### 3.3 圆柱弯曲薄条复核（a=1 m，b=6 m，16×96）

| β | ζ | M_end 解析 | w_mid 解析 | w_mid FEM | w 误差 |
|---|---|---|---|---|---|
| 0.2 | 0.25 | 92.592 | 6.168982e-3 | 6.128112e-3 | −0.66 % |
| 0.5 | 1.00 | 277.777 | 4.965280e-3 | 4.932029e-3 | −0.67 % |
| 0.8 | 4.00 | 555.555 | 3.159728e-3 | 3.133551e-3 | −0.83 % |

→ **中间 β 的位移精度 <1 %**，说明「β→弹簧刚度」映射与整体求解均正确。

### 3.4 缺陷 B：受限边的弯矩恢复只一阶（已提供模型自洽口径）

同一薄条 β=0.5（解析 M_end = 277.78 N·m/m）：

| 网格 | w 误差 | 外推恢复 M_n | 误差 | 由位移反演 | `c_r·θ_n` | 误差 |
|---|---|---|---|---|---|---|
| 8×32 | −2.80 % | 0.003 | −100.0 % | 299.17 | — | — |
| 16×64 | −0.67 % | 130.21 | −53.1 % | 282.89 | 276.70 | **−0.39 %** |
| 32×128 | −0.14 % | 201.83 | −27.3 % | 278.83 | 277.51 | **−0.09 %** |

位移二阶收敛、由位移反演的边缘弯矩收敛到解析值，而 `moments_at_nodes`（Gauss→节点外推
B^TDB）在**受限边**只有一阶精度，16 单元时低估 ~53 %。原因：边缘弯矩的边界层恰好在边界，
外推把层内梯度抹平。
新增 `edge_spring_moment()`：按模型定义直接取 `M_n = c_r·θ_n`（二阶、nx=16 已 <0.5 %），
供受限边/固支边的峰值应力报表使用；`moments_at_nodes` 的适用性在文档字符串中标注。

## 4. 结果图

`docs/figures/elastic_restraint_tributary.png`：左图 β 加密曲线（修前虚线漂移 / 修后实线收敛，
附 SSSS/CCCC 理论线）；右图受限边弯矩两种口径的误差对比（外推 vs 弹簧自洽）。

## 5. 注意点（坑与适用范围）

* **漏乘从属边长是本类「弹簧边条件」的常见错误**：只要把分布系数直接加到每个边界节点，
  β 就随网格漂移，且用两个极限值（β=0、β=1）做的测试**完全测不出来**——两个极限对 h 不敏感。
  必须断言「网格无关性」与「总弹簧刚度 = c_r × 边长」这类不变量。
* 原有 `test_mesh_convergence_beta_half` 只断言 `1e-4 < w_norm < 1e-2`，属于"执行代码"而非
  "断言不变量"，缺陷因此未被发现（本文件新增的 `test_beta_response_is_mesh_independent` 补上）。
* β→1 用 `ζ=1/ε`（ε=1e-6）逼近固支；薄条与方板在 nx≥8 时与硬固支差 <1 %，
  但**极细网格（h/a<1e-4）时该正则化会失效**，此时应改用 `boundary="clamped"`。
* 中间 β 无「文献表格」可直接对标，本文用圆柱弯曲解析解替代；该基准要求 b/a≥4、
  取样在跨中（端部边界层影响 <1 %）。
* 受限边峰值应力请用 `edge_spring_moment()`；跨中/远离边界处 `moments_at_nodes` 二阶收敛，可继续使用。
* 弹性地基（Winkler）未参与本轮复核；β 结论与地基无关（约束项与地基项可加）。

## 6. 误差判定（是否 <3 %）

| 关键量 | 基准 | 误差 | <3 %？ |
|---|---|---|---|
| β=0 `w_c D/(q a⁴)`（nx=24） | Kirchhoff SSSS 0.00406 | +0.05 % | 是 |
| β=1 `w_c D/(q a⁴)`（nx=24） | Kirchhoff CCCC 0.00126 | +0.49 % | 是 |
| β=0 峰值应力（面板工况 nx=160） | Navier N=401（panel_patch_load_results.json） | −0.065 % | 是 |
| 中间 β 位移（薄条，β=0.2/0.5/0.8） | 圆柱弯曲解析解 | −0.66 / −0.67 / −0.83 % | 是 |
| 中间 β 边缘弯矩（弹簧口径，nx=16） | 圆柱弯曲解析解 | −0.39 % | 是 |
| 中间 β 边缘弯矩（外推口径，nx=16） | 同上 | −53.1 % | **否（已定级为缺陷，改用弹簧口径）** |
| β=0.2/0.5 网格无关性（nx=8↔24 跨度） | — | 0.6 % / 0.8 % | 是（修前 24 %/ 22 %，否） |

honest-FAIL：外推口径的边缘弯矩在工程网格上不可接受（−53 %），本文如实记为缺陷 B，
未通过修改基准或放宽判据掩盖；修复路径是换用模型自洽的弹簧弯矩，而非调整容差。

## 7. 复现

```bash
cd <worktree> && PYTHONPATH=src python scratch/probe_mesh_dep.py      # 表 3.1
PYTHONPATH=src python scratch/probe_strip.py                          # 表 3.3
PYTHONPATH=src python scratch/probe_recovery.py                       # 表 3.4
PYTHONPATH=src python gen_tributary_figure.py                         # 图
python -m pytest tests/test_plate_elastic_restraint.py tests/test_plate_restraint_panel.py tests/test_plate_restraint_tributary.py -q
```