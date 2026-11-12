# 交差审计 m15368 — CASE-MACNEAL-BEAM（细长梁剪切锁定，TensorFEM）

审计单 m15487（board-audit，2026-11-12）。对象：m15368（succeeded，PR #712 merged，
commit `221f25f0`，worktree `/root/tp-agent/work/m15368`）。本单=核验，非重做。

## 裁定

**level = 4（L4 基准验证：完整算例 + 误差分析 + 可复现）**，verdict 已落表
`tensor_maturity_verdicts` id=334（repo=TensorFEM, scope='m15368 CASE-MACNEAL-BEAM',
mission_id=15487, verdict_by=board-audit, gate_state=pending 待机器复核）。

## 验收三件套核验（全部独立复跑，非抄报告）

| 项 | 判据 | 实测 | 结论 |
|---|---|---|---|
| run_artifacts | 文件真实存在且入库 | results.json(214 行)/report.json(239KB,3 图内嵌)/docs/assets/macneal_beam/{convergence,error_decay,deflection_field}.png 均在 commit 221f25f0 | ✅ |
| ref vs got | 容差 <5%（mission goal + case_pool id=44 tol_policy） | sri n=16: 256.823437500006，vs Timoshenko −0.379%、vs EB +2.729%；mitc 同（−0.379%/+2.729%）；full n=16: 229.11475876802837，−11.127%（预期锁定 FAIL，假设本身） | ✅ sri/mitc 在容差内（且 <3%）；full 超差=被检验的锁定假设，报告已如实声明 |
| repro_cmd | 可执行 | `PYTHONPATH=src python benchmarks/macneal_beam_shear_locking/run.py`（4.17s，数值与 results.json 逐位一致）+ `validate.py` 18/18 PASS | ✅ |

## 基准裁判（唯一裁判）

- 参考值=闭式（零手抄）：Euler-Bernoulli `PL³/48EI` = 250.0；Timoshenko
  `PL³/48EI + PL/4kAG`（k=5/6）= 257.8。validate.py 独立重算核对：
  stored=250.000000/rederived=250.000000，stored=257.800000/rederived=257.800000。
- **审计发现 F1（已修复）**：`tensor_benchmark_cases` 中 anchor_slug='CASE-MACNEAL-BEAM'
  原无行（任务榜指定的 SELECT 返回 0 行）——锚点只登记在 tensor_case_pool（id=44）。
  本审计独立复跑确认 got 在容差内后，补登记 id=17（status=passed, passed_at=now），
  使锚点从此可机器复核。
- 报告六段（①问题+ref 出处 ②方法 ③参数表 ④ref vs got 对照 ⑤收敛/敏感性 ⑥失败模式+工件）
  齐全（tensor_reports id=14533，published）；报告数字与 results.json 逐项一致（零手抄）。

## 判级依据（级别=功能实质）

- L2 真代码：`src/tensorfem/plate.py:119 q4_mindlin_stiffness` 新增
  `shear_treatment="full"`（:146 校验、:210-229 全积分剪切块）——输入→计算→输出真代码。
- L3 集成：`q4_mindlin_stiffness` 被 src/tensorfem/ 下 18 个模块调用
  （plate_mindlin/shell4/shell4_mitc/ice_panel_fsi/buoyancy_coupling/…），
  新选项向后兼容（默认仍 "sri"），tests/test_plate*.py 37 passed（本审计复跑 32.45s）。
- L4 基准验证：完整算例（4 层网格 × 3 剪切处理）+ 误差分析（收敛阶/衰减因子/锁定特征）
  + 可复现（repro_cmd 本审计复跑逐位一致，validate.py 18/18 PASS）。
L5 不成立：无新物理/新方法的科研增量，属验证性基准算例。

## 审计发现

| # | 严重度 | 内容 | 处置 |
 sql |---|---|---|
| F1 | 中（已修复） | tensor_benchmark_cases 缺 CASE-MACNEAL-BEAM 行（任务榜指定 SELECT 返回 0 行），锚点仅存于 tensor_case_pool id=44 | 本审计补登记 id=17（status=passed），evidence 已写入 verdict |
| F2 | 低（声明） | MacNeal 1985 原文不在环境内，判据=论文自用的闭式标尺（EB 250.0 / Timoshenko 257.8），非论文表格数 | 报告摩擦节已如实声明，不粉饰 |
| F3 | 信息 | full 规则 −11.13% 超 5% 容差 | 预期锁定行为（被检验假设），非缺陷；报告与 validate.py 均如实标 FAIL |

## 复现（本审计实际执行）

```bash
cd /root/tp-agent/work/m15368   # @221f25f0 (PR #712 merged)
PYTHONPATH=src python benchmarks/macneal_beam_shear_locking/run.py      # 4.17s, 逐位一致
PYTHONPATH=src python benchmarks/macneal_beam_shear_locking/validate.py # 18/18 PASS
PYTHONPATH=src python -m pytest tests/test_plate.py tests/test_plate_mindlin.py tests/test_element_rank_audit.py -q  # 37 passed
```

## 落表

- `tensor_maturity_verdicts` id=334：repo=TensorFEM, scope='m15368 CASE-MACNEAL-BEAM',
  mission_id=15487, level=4, verdict_by=board-audit, gate_state=pending（待机器复核）。
- `tensor_benchmark_cases` id=17：CASE-MACNEAL-BEAM 补登记（见 F1）。
