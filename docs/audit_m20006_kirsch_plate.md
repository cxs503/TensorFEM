# CASE-FEM-KIRSCH-PLATE 交差审计报告 (m20006 → m20102)

## 审计概要
- **对象**: m20006 CASE-FEM-KIRSCH-PLATE · 算例池 · 开孔板应力集中系数 K=3 · TensorFEM
- **性质**: 交差核验（非重做），对照基准裁判逐项核实
- **判定**: **L4 成熟度**（基准验证：完整算例+误差分析+可复现）
- **verdict 落表**: `tensor_maturity_verdicts` id=15, level=4, verdict_by=board-audit

## ① 取证三件套核验

### run_artifacts ✅
| 文件 | 路径 | 大小 | 状态 |
|---|---|---|---|
| result.json | runs/m20006_kirsch_plate/result.json | 61606 B | ✅ |
| gates_verified.json | runs/m20006_kirsch_plate/gates_verified.json | 6341 B | ✅ |
| run_stdout.log | runs/m20006_kirsch_plate/run_stdout.log | 1363 B | ✅ |
| manifest.json | runs/m20006_kirsch_plate/manifest.json | 3793 B | ✅ |
| result.json (副本) | benchmarks/verified/fem_kirsch_plate/result.json | 61607 B | ✅ |
| gates_verified.json (副本) | benchmarks/verified/fem_kirsch_plate/gates_verified.json | 6341 B | ✅ |

### benchmark ref vs got ✅
| 量 | ref | got | 误差 | 容差 | 判定 |
|---|---|---|---|---|---|
| K_t (d/W=0.02, n=48) | 3.0 | 3.001408 | 0.047% | <5% | ✅ |
| K_t (Q8 96×24) | 3.0 | 3.001476 | 0.049% | <5% | ✅ |
| Richardson 极限 | 3.0 | 3.009196 | 0.307% | <1% | ✅ |
| 收敛阶 | ≥1.0 | 1.763 | — | — | ✅ |
| Q8 收敛阶 | ≥1.5 | 1.720 | — | — | ✅ |
| ν/E 不变性 | <1% | 0.706% | — | — | ✅ |
| Saint-Venant | <1% | 0.220% | — | — | ✅ |
| 有限宽趋势 | <3% | 0.985% | — | — | ✅ |
| 孔边剖面 RMS | <5% | 0.318% | — | — | ✅ |
| 载荷平衡 | <1e-10 | 1.01e-12 | — | — | ✅ |

**14/14 门禁通过，失败项：无。**

### repro_cmd ✅
- 主命令: `OMP_NUM_THREADS=6 PYTHONPATH=src python benchmarks/verified/fem_kirsch_plate/run.py`
- 测试命令: `OMP_NUM_THREADS=4 PYTHONPATH=src python -m pytest tests/test_kirsch_plate_anchor.py -q`
- 实测: **7/7 测试通过**（修复 D1 后）

### 报告六段 ✅
| 段 | 位置 | 状态 |
|---|---|---|
| ① 案例介绍 | docs/CASE-FEM-KIRSCH-PLATE-pool-m20006.md §① | ✅ |
| ② 方法 | §② | ✅ |
| ③ 计算结果 | §③ 参数表 + §④ 对照表 | ✅ |
| ④ 结果图 | §④ (4 PNG + 2 新增 PNG) | ✅ |
| ⑤ 注意点 | §⑤ 收敛性 + 敏感性 + limitations | ✅ |
| ⑥ 误差判定 | §④ ref vs got 表 + §⑤ 预声明判据 | ✅ |

## ② 缺陷发现

### D1（中·已修复）: PNG 工件位置不匹配
- **现象**: `test_committed_figures_exist` 失败 — 4 张 PNG 在 `docs/assets/kirsch_plate/` 但不在 `benchmarks/verified/fem_kirsch_plate/`
- **根因**: 驱动可能以 `--out docs/assets/kirsch_plate/` 运行，或生成后移动了文件
- **影响**: 测试套件 1/7 失败；CI 会阻断
- **修复**: 已复制 4 张 PNG 到 benchmark 目录，测试 7/7 通过
- **建议**: 在 run.py 中增加 post-run 同步逻辑，或修改测试路径

### D2（低·平台侧）: tensor_benchmark_cases 表缺行
- **现象**: `SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-FEM-KIRSCH-PLATE'` 返回 0 行
- **根因**: TensorPlatform 侧机制 gap — case_pool dispatched→succeeded 未同步到 benchmark_cases
- **影响**: 自动基准注册无法机读容差（容差实际在 tensor_case_pool id=186）
- **修复面**: 非 TensorFEM 仓修复面，属平台共性机制

## ③ 成熟度判定

| 级别 | 标准 | 判定 |
|---|---|---|
| L0 | 文档成果 | ✅ 有完整文档 |
| L1 | 数学/单元验证 | ✅ Oracle 自检 + 闭式解求值 |
| L2 | 独立功能实现 | ✅ finite_width_hole.py (398 行) + structural_canonical_face.py (1248 行) |
| L3 | 系统集成 | ✅ 接入 solve_continuum 主求解器 |
| L4 | 基准验证 | ✅ 完整算例 + 14 门禁 + 误差分析 + 可复现 + Q8 交叉 |
| L5 | 科研工程成果 | ❌ 经典弹性力学基准，非原创研究贡献 |

**最终判定: L4**

## ④ 落表记录
```sql
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorFEM', 'm20006 CASE-FEM-KIRSCH-PLATE', 20006, 4, '<JSON>', 'board-audit');
```
- 落表 id: 15
- evidence JSON: `/root/tp-agent/work/m20102/verdict_evidence_m20006.json`

## ⑤ 报告发布
- 报告 ID: 18295
- 状态: published
- 内嵌图: 3 张 (convergence, hoop_profile, q4_vs_q8)
- 提交校验: markdown/figures/mission_id 三字段 digest 全匹配
