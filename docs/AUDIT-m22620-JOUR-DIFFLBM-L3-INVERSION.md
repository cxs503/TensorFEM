# AUDIT m22620 — JOUR-DIFFLBM-L3-INVERSION 交差核验

**对象**：m22620（JOUR-DIFFLBM-L3-INVERSION·【论文牵引|JOUR-DIFFLBM 派生锚】·TensorFEM，已 succeeded）
**审计**：board-audit（交差核验单，非重做）
**工作树**：/root/tp-agent/work/m22620（agent/m22620 分支）
**commit**：a316abfd "TensorFEM: adjoint body-force field inversion from sparse nodal observations"

---

## ① 取证（三件套逐项核验）

### run_artifacts — ✅ 真实存在
`artifacts/body_force_inversion/` 全部 git-tracked（commit a316abfd）：
- `summary.json`（mesh/field_rel_l2=4.30e-06/iterations=13/acceptance 全 PASS）
- `case_acceptance.json`（convergence[] 13 行逐迭代）
- `observability.json`（10%/25%/100% 观测密度 vs 条件数）
- `noise_sweep.json`（噪声 vs 阻尼扫描）
- `controls.json`（out-of-span 真值 + 全节点未知量负控）
- `forward_verification.json`（Q4 patch 4.17e-16 + manufactured 收敛阶 [1.998, 2.0]）
- `figures/`（fig_convergence.png / fig_fields.png / fig_noise.png）

### benchmark ref vs got — ⚠️ 无注册判据行
`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='JOUR-DIFFLBM-L3-INVERSION'` → **(0 rows)**。
tensor_benchmark_cases 共 146 行、12 个 TensorFEM 锚点，**无一个是本锚**。
报告自声明判据（≤10% 节点 / ≤5% 场误差 / ≤50 迭代）且 got 值均在判据内
（9.96% / 4.30e-06 / 13），但**无注册容差** → L4 结构性不可能。

### repro_cmd — ✅ 可执行
`PYTHONPATH=src python scripts/run_body_force_inversion.py` 复算成功，
重产出 summary.json（field_rel_l2=4.30e-06、13 迭代、wall_clock 90.29s）。

### 报告六段 — ✅ 齐全
tensor_reports id=20343（mission_id=22620）：①案例介绍 ②怎么做的 ③计算结果
④结果图 ⑤注意点 ⑥误差判定，六段齐全，误差判定显式回答 <3%。

---

## ② 判级（L0-L5）

| 判据 | 证据 | 判定 |
|---|---|---|
| L0 文档成果 | 有真代码 | 否 |
| L1 数学/单元验证 | 有完整功能实现 | 否 |
| **L2 独立功能实现** | **真代码：输入→计算→输出**（Q4 刚度 + CGLS + 模态基 + 可观测性 SVD），16 测试全绿 | **是** |
| L3 系统集成 | 模块**不在 main**、**不在 __init__.py**、**不在 cli.py**、`import tensorfem.body_force_inversion` → ModuleNotFoundError | 否 |
| L4 基准验证 | **无注册判据行**（tensor_benchmark_cases 0 行） | 否 |
| L5 科研工程成果 | 单模块，非完整科研工程 | 否 |

**级别 = L2**（功能实质）。锚点验收判据（≤10% 节点 / ≤5% 场误差 / ≤50 迭代 / 收敛曲线入库）
全部达成，但功能实质是 L2 级独立模块，未接入主包。

---

## ③ 落表

```sql
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorFEM', 'm22620 JOUR-DIFFLBM-L3-INVERSION', 22620, 2, '<JSON>', 'board-audit');
-- id=905, level=2, verdict_by=board-audit, created_at=2026-11-16 01:19:13
```

**maturity_gate 裁定**：confirmed（消账）——L2 级独立功能实现，验收三件套真实存在，
报告六段齐全，repro_cmd 可执行。非 L3/L4 因模块未接入主包 + 无注册基准判据行。

---

## ④ 检查清单

- [x] psql tensor_benchmark_cases anchor 查询 → 0 行（无注册判据）
- [x] git ls-tree main -- src/tensorfem/body_force_inversion.py → 空（不在 main）
- [x] grep src/tensorfem/__init__.py for body_force → 无
- [x] import tensorfem.body_force_inversion → ModuleNotFoundError
- [x] PYTHONPATH=src pytest tests/test_body_force_inversion.py -q → 16 passed in 42.01s
- [x] PYTHONPATH=src python scripts/run_body_force_inversion.py → 复算成功
- [x] psql tensor_reports WHERE mission_id=22620 → id=20343，六段齐全
- [x] psql feature_ledger WHERE slug=JOUR-DIFFLBM-L3-INVERSION → status=in_progress

## ⑤ 未跑检查

- 未在 main 分支上跑测试（模块不在 main，无法在 main 上验证）
- 未做跨进程确定性验证（报告自述未覆盖）
