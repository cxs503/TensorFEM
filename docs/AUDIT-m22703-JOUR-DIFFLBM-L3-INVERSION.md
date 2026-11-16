# AUDIT m22703 — JOUR-DIFFLBM-L3-INVERSION 交差核验

**对象**：m22703（JOUR-DIFFLBM-L3-INVERSION·【论文牵引|JOUR-DIFFLBM 派生锚】·TensorFEM，已 succeeded）
**审计**：board-audit（交差核验单，非重做）
**工作树**：/srv/tensor/agent/work/m22729（agent/m22703 分支）
**commit**：161b1ec3 "body-force inversion CLI: fix output-path resolution for the default runs directory"
**前置**：m22620（a316abfd）为初始实现；m22703 在其上增加 L3 集成 + 基准注册 + 修复轮次

---

## ① 取证（三件套逐项核验）

### run_artifacts — ✅ 真实存在
`artifacts/body_force_inversion/` 全部 git-tracked（commit 161b1ec3）：
- `summary.json`（mesh/field_rel_l2=4.30e-06/iterations=13/acceptance 全 PASS）
- `case_acceptance.json`（convergence[] 13 行逐迭代）
- `observability.json`（10%/25%/100% 观测密度 vs 条件数）
- `noise_sweep.json`（噪声 vs 阻尼扫描）
- `controls.json`（out-of-span 真值 + 全节点未知量负控）
- `forward_verification.json`（Q4 patch 4.17e-16 + manufactured 收敛阶 [1.998, 2.0]）
- `benchmark_rows.json`（6 行 ref-vs-got 判据行）
- `pytest_entry.log` / `pytest_numerics.log`（测试日志）
- `figures/`（fig_convergence.png / fig_fields.png / fig_noise.png）

### benchmark ref vs got — ✅ 已注册判据行（m22620 审计时缺失，m22703 已补齐）
`SELECT case_name,metric,tolerance,status FROM tensor_benchmark_cases WHERE anchor_slug='JOUR-DIFFLBM-L3-INVERSION'` → **6 rows**：

| id | case_name | status | 关键 got 值 | 容差 |
|---|---|---|---|---|
| 158 | sparse-observation body-force recovery | passed | field_rel_l2=4.30e-06, 13 iter, 9.96% | <5%, ≤50 iter, ≤10% |
| 159 | Q4 constant-strain patch | passed | max_rel_err=4.17e-16 | <1e-12 |
| 160 | Q4 manufactured solution | passed | rates=[1.998, 2.000], finest=6.33e-4 | ≥1.9, <1e-3 |
| 161 | 1% sensor noise | **failed** | field_rel_l2=1.127 | <5% (预期失败) |
| 162 | negative control: full-nodal | passed | null_space_dim=164, field=0.816 | null_space>0, field>5% |
| 163 | negative control: out-of-span | passed | out_of_span=true, field=0.468 | out_of_span=true |

**5/6 passed，1/6 failed（预期）**。id=161 的失败是诚实记录：条件数 ~2.9e3 放大 1% 传感器噪声远超 5% 带宽。

### repro_cmd — ✅ 可执行
`PYTHONPATH=src python3 benchmarks/body_force_inversion_benchmark.py` 复算成功，
产出 benchmark_rows.json（field_rel_l2=4.297e-06、13 迭代、wall_clock 24.33s），
与已注册判据行一致。

### 报告六段 — ✅ 齐全
`docs/CASE-FEM-BODY-FORCE-INVERSION.md`：①案例介绍 ②怎么做的 ③计算结果
④结果图 ⑤注意点 ⑥误差判定，六段齐全，误差判定显式回答 <3%。

---

## ② 判级（L0-L5）

| 判据 | m22620 审计时 | m22703 现状 | 判定 |
|---|---|---|---|
| L0 文档成果 | 有真代码 | 有真代码 | 否 |
| L1 数学/单元验证 | 有完整功能实现 | 有完整功能实现 | 否 |
| L2 独立功能实现 | 是 | 是 | 否（超越） |
| **L3 系统集成** | **否**（不在 __init__.py / cli.py） | **是**（__init__.py 导出 + CLI 命令） | **是** |
| **L4 基准验证** | **否**（无注册判据行） | **是**（6 行注册，5 通过，1 预期失败） | **是** |
| L5 科研工程成果 | 单模块 | 单模块 + 基准套件 | 否 |

**级别 = L4**（基准验证级）。相比 m22620 的 L2，m22703 补齐了：
1. `__init__.py` 公共 API 导出（`from tensorfem import run_body_force_inversion`）
2. `cli.py` 命令行入口（`tensorfem body-force-inversion`）
3. `tensor_benchmark_cases` 注册（6 行判据，含诚实 FAIL）
4. 额外测试文件 `test_body_force_inversion_entry.py`（10 项）

---

## ③ 落表

```sql
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorFEM', 'm22703 JOUR-DIFFLBM-L3-INVERSION', 22703, 4, '<JSON>', 'board-audit');
-- id=907, level=4, verdict_by=board-audit
```

**maturity_gate 裁定**：confirmed（消账）——L4 级基准验证，验收三件套真实存在，
报告六段齐全，repro_cmd 可执行，6 项基准判据已注册且 5/6 通过（1 项预期失败如实记录）。

---

## ④ 检查清单

- [x] git checkout agent/m22703 → 工作树切换成功
- [x] artifacts/body_force_inversion/ 全部文件存在（11 文件 + 3 图）
- [x] psql tensor_benchmark_cases anchor 查询 → 6 行（5 passed, 1 failed）
- [x] `from tensorfem import run_body_force_inversion, InversionOptions` → Import OK
- [x] `tensorfem body-force-inversion` CLI 命令存在
- [x] PYTHONPATH=src pytest tests/test_body_force_inversion.py → 16 passed
- [x] PYTHONPATH=src pytest tests/test_body_force_inversion_entry.py → 10 passed
- [x] PYTHONPATH=src python3 benchmarks/body_force_inversion_benchmark.py → 复算成功
- [x] verdict_evidence_m22703.json 已创建
- [x] INSERT INTO tensor_maturity_verdicts → id=907, level=4

## ⑤ 未跑检查

- 未在 main 分支上跑完整测试（模块在 agent/m22703 分支，main 合并状态待确认）
- 未做跨进程确定性验证（报告自述未覆盖）

## ⑥ 与 m22620 审计的差异

| 项目 | m22620 审计 | m22703 审计 |
|---|---|---|
| 级别 | L2 | L4 |
| __init__.py 导出 | 无 | 有 |
| CLI 入口 | 无 | 有 |
| 注册判据行 | 0 行 | 6 行 |
| 测试数 | 16 | 26 (16+10) |
| verdict id | 未插入 | 907 |
