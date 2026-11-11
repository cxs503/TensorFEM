# 交差核验：m15339 XVAL-FEM-COOK（Cook 膜三方横向对比）

- 审计单：m15387（board-audit）　被审单：m15339（TensorFEM，status=succeeded，PR **#697 merged**）
- 被审产物 commit：`d5713e28`（分支 `agent/m15339`）；main 侧落点：merge `f5bb473f`（Merge #697）
- 审计者工作树：`/root/tp-agent/work/m15387`（`git merge --ff-only origin/main` 后含 PR#697 全部代码件）
- 结论：**L4（基准验证）确认，非 L5**；未达标项与虚标项逐条见 §5/§6。

## 1. 案例介绍（物理背景/几何/参数）

Cook 膜是平面应力单元的标准基准：锥形悬臂薄板，受弯剪复合载荷，检验 Q4/Q8/P2 类单元在
弯曲主导问题中的剪切锁定与收敛性。

| 项 | 值 | 出处 |
|---|---|---|
| 角点 | (0,0)/(48,44)/(48,60)/(0,44) | `benchmarks/cook_membrane_xval.py`, `benchmarks/cook_xval_evidence.json:spec.corners` |
| E, ν, t | 1.0, 1/3, 1.0 | 同上 `spec.young/poisson/thickness` |
| 载荷 | 右边界均布竖向 trac，合力 R=1.0（t=1/16） | 同上 `spec.resultant/traction_y=0.0625` |
| 量测点 | (48,52) 竖向位移 uy_mid | 同上 `spec.midpoint` |
| 公开参考值 | **23.96**（Cook, *Concepts and Applications of FEM*） | 同上 `spec.published_reference_uy_mid` |
| 网格 | 结构化 n×n（n=8,16），双线性映射 | `_skfem_mesh` / Q4 映射 |

## 2. 怎么做的（方法/流程/工具链）

三方同一问题、同一载荷、同一后处理约定：

| 当事方 | 内核 | 类型 |
|---|---|---|
| `tensorfem_q4` | 本仓生产路径 `tensorfem.continuum.solve_continuum`（Q4 双线性，2×2 Gauss 全积分） | 生产 |
| `skfem_quad2` | scikit-fem 12.0.2 `ElementQuad2` | 第三方开源 |
| `skfem_tri2` | scikit-fem 12.0.2 `ElementTriP2` | 第三方开源 |

流程：`benchmarks/cook_membrane_xval.py`（生产求解器调用方）→ `artifacts/cook_xval/metrics.json`
→ 门面 `src/tensorfem/cook_xval.py`（`run_cook_xval`/`load_cook_xval_results`，由
`src/tensorfem/__init__.py:421-427` 导出）→ 文档 `docs/cook-xval-report.md`。

审计独立复跑（本单实做）：
```
PYTHONPATH=src python3 benchmarks/cook_membrane_xval.py --levels 8,16 \
  --parties tensorfem_q4,skfem_quad2,skfem_tri2 --out /tmp/cook_verify
```
结果与 `artifacts/cook_xval/metrics.json` **逐项数值一致**（除 wall_s）：uy_mid/uy_max/reaction_y/
energy/vm_max_nodal 全部 |Δ|<1e-9（见 `docs/m15387/audit-pack/repro_verify.log`）。
`pytest tests/test_cook_xval.py` → 15 passed。

## 3. 计算结果（数值表）

| 当事方 | n | ndof | uy_mid | 误差 vs 23.96 | <3% | ΣR_y |
|---|---:|---:|---:|---:|---|---:|
| tensorfem_q4 | 8 | 162 | 22.079183 | 7.85% | ✗ | −1.000000 |
| tensorfem_q4 | 16 | 578 | 23.430411 | **2.21%** | ✓ | −1.000000 |
| skfem_quad2 | 8 | 578 | 23.925405 | 0.14% | ✓ | −0.423109* |
| skfem_quad2 | 16 | 2178 | 23.949409 | 0.04% | ✓ | −0.390050* |
| skfem_tri2 | 8 | 578 | 23.860706 | 0.41% | ✓ | −0.569671* |
| skfem_tri2 | 16 | 2178 | 23.927125 | 0.14% | ✓ | −0.467722* |

\* 该列是**部分和**（只取顶点 y 自由度），非真实反力——见 §5 发现 F1。

审计侧独立参考值复算（scikit-fem Q2 加密，`docs/m15387/probes/probe_independent_ref.py`）：

| n | 16 | 24 | 32 | 48 | 64 |
|---|---:|---:|---:|---:|---:|
| uy_mid | 23.949409 | 23.957323 | 23.960775 | 23.963748 | 23.965040 |

即真实收敛值 ≈ **23.965**，公开参考 23.96 本身偏差 0.02%；把参考换成 23.965 后
TensorFEM Q4 n=16 误差 2.23%（原报 2.21%），**结论对参考值选取不敏感**。

## 4. 结果图

- `docs/m15387/audit-pack/convergence.png`：uy_mid vs 网格（三方 + 23.96 参考线 + 独立加密参考）
- `docs/m15387/audit-pack/error.png`：|误差|% vs 网格（对数纵轴，3% 门槛线）

## 5. 注意点（发现的问题，按严重度）

**F1（中，证据性缺陷｜影响第三方行可信度）** 报告 §4 note 把 skfem 的 `reaction_y≈−0.42`
解释为「traction 按节点集中力施加、非一致边载」。该因果**不成立**：静力学要求
ΣR_y = −(施加荷载合力) = −1.0，与荷载如何分布无关。实测（`probes/probe_reaction_full.py`）：
四组算例约束自由度反力**总和恒为 −1.0000**，ΣR_x ≈ 5e-14；报告的 −0.42/−0.39/−0.57/−0.47
只是 `residual[nodal_dofs[1, left_nodes]]` 的**部分和**——漏掉了二次单元在夹支边上的**中边
（facet）y 自由度**。正确取全（nodal∪facet∪interior 的 y 分量）后 ΣR_y = −1.0000。
⇒ 结论：失衡是**度量代码 bug**，不是加载方式限制；报告结论行「Equilibrium ✓ … scikit-fem:
known limitation」属**错误归因**。对位移结论无影响（位移与反力度量相互独立）。

**F2（中，登记缺口）** `tensor_benchmark_cases` 中 `anchor_slug='XVAL-FEM-COOK'` **0 行**
（实测 `SELECT ... WHERE anchor_slug='XVAL-FEM-COOK'` → 0 rows）。故「<3%」容差是
**报告自声明**，不是登记判据；本单只能按「公开文献 23.96 + 独立复算」判，机器无法做登记态核验。
建议：把 anchor 登记入表（metric=`uy_mid(48,52) vs 23.96`，tolerance `<3%`，n≥16）。

**F3（中，成果落地缺口）** `run_artifacts` 只存在于工作树：`artifacts/**` 与 `*.png` 被
平台推送门 `tp_pusher.py:351 _EXCLUDE/._EXCLUDE2` 按「记录区不上 GitHub」策略滤除 →
main（`f5bb473f`）里**没有** `artifacts/cook_xval/metrics.json`。代码/测试/文档/证据 JSON 都进了
main，唯独数值工件没进；L2/L4 证据若指向 `artifacts/...` 相对路径，门复核必判「工件缺失」。
本单已把审计快照固化到 `docs/m15387/audit-pack/metrics_snapshot.json`（`docs/` 下 png 放行）。

**F4（低，报告口径）** `artifacts/cook_xval2/metrics.json` 是同一次 n=8/16 的**旧中间轮**
（uy_mid 与终版一致，但 `vm_max_nodal`=0.213412 vs 终版 0.352518），两份工件并存易被误引；
建议删或标注 superseded。

**F5（低，测试强度）** `tests/test_cook_xval.py` 15 个用例中，数值类断言均为「读工件 → 断言
误差阈值」，没有一条**从零重算**校验（不驱动求解器）。因此测试绿 ≠ 数值可复现；真正的可复现
由本单的独立复跑背书。

**F6（低，n=8 不达标但已如实申报）** Q4 n=8 误差 7.85% > 3%，报告打了 ✗ 并说明 Q4 需 n≥32，
未粉饰；n=16 的 2.21% 是本单唯一达标点，属「该级别可达」而非「全域收敛」。

## 6. 误差判定（显式回答是否 <3%）

| 关键量 | got | 基准 | 误差 | <3%？ |
|---|---:|---:|---:|---|
| TensorFEM Q4 n=16 uy_mid | 23.430411 | 23.96（Cook 文献） | 2.211% | **是**（临界，余量 0.79pp） |
| TensorFEM Q4 n=16 uy_mid | 23.430411 | 23.9650（审计独立加密） | 2.231% | 是 |
| TensorFEM Q4 n=8 uy_mid | 22.079183 | 23.96 | 7.854% | 否（单内已如实申报） |
| TensorFEM ΣR_y（n=8,16） | −1.000000 | −1.0（静力平衡） | 0（<1e-12） | 是 |
| 三方跨工具 spread（n=16） | 23.430411 / 23.949409 / 23.927125 | — | 相对 TensorFEM 2.2%，两第三方互差 0.09% | 跨工具一致性成立（我方为慢收敛方） |
| 第三方 ΣR_y（报告值） | −0.390…−0.570 | −1.0 | 39%–57% | **否 → 度量 bug（F1），非物理** |

## 判级依据

- L1 数学/单元验证：`tests/test_cook_xval.py`（15 passed）
- L2 独立功能：`src/tensorfem/cook_xval.py`（真代码，subprocess 驱动 + 结果解析/误差属性）+ 入口
  `run_cook_xval` + 运行工件（快照见 `audit-pack/`）
- L3 系统集成：`benchmarks/cook_membrane_xval.py:153-215` 驱动生产求解器
  `tensorfem.continuum.solve_continuum`；包级导出 `src/tensorfem/__init__.py:421-427`
- L4 基准验证：ref vs got + 逐项误差 + 可复现命令（本单独立复跑数值一致）
- **L5 不给**：无新物理/新方法，属标准算例的横向核验；且基准未登记（F2）、反力度量有 bug（F1）。